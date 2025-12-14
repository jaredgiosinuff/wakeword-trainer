import { useState, useRef, useCallback, useEffect } from 'react'
import JSZip from 'jszip'

// Types
interface AudioSample {
  id: string
  blob: Blob
  url: string
  duration: number
  timestamp: Date
}

interface CommunityWakeword {
  id: string
  name: string
  displayName: string
  description: string
  sampleCount: number
  ratingCount: number
  averageRating: number
  weightedRating: number
  createdAt: string
}

type Tab = 'record' | 'browse'
type SortBy = 'weightedRating' | 'sampleCount' | 'name' | 'createdAt'
type RecordingPhase = 'ready' | 'countdown' | 'listen' | 'speak' | 'done'

// API base URL - configure for your deployment
const API_BASE = import.meta.env.VITE_API_URL || ''

// Get or create visitor ID for rating tracking
function getVisitorId(): string {
  let id = localStorage.getItem('wakeword_visitor_id')
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem('wakeword_visitor_id', id)
  }
  return id
}

// Tooltip component
function Tooltip({ children, text }: { children: React.ReactNode; text: string }) {
  const [show, setShow] = useState(false)
  return (
    <span className="relative inline-block">
      <span
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="cursor-help"
      >
        {children}
      </span>
      {show && (
        <span className="absolute z-50 w-64 p-2 text-xs text-white bg-gray-900 rounded-lg shadow-lg -top-2 left-full ml-2">
          {text}
          <span className="absolute top-3 -left-1 w-2 h-2 bg-gray-900 transform rotate-45" />
        </span>
      )}
    </span>
  )
}

// Info icon for tooltips
function InfoIcon() {
  return (
    <svg className="w-4 h-4 inline-block ml-1 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
    </svg>
  )
}

// Progress bar component
function ProgressBar({ current, target, label }: { current: number; target: number; label: string }) {
  const percentage = Math.min((current / target) * 100, 100)
  const isComplete = current >= target

  return (
    <div className="mb-2">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-purple-300">{label}</span>
        <span className={isComplete ? 'text-green-400' : 'text-purple-300'}>
          {current}/{target} {isComplete && '✓'}
        </span>
      </div>
      <div className="h-2 bg-white/10 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-300 ${isComplete ? 'bg-green-500' : 'bg-purple-500'}`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  )
}

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState<Tab>('record')
  const [showQuickStart, setShowQuickStart] = useState(true)

  // Recording state
  const [wakeWord, setWakeWord] = useState('')
  const [samples, setSamples] = useState<AudioSample[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>('ready')
  const [error, setError] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [agreedToShare, setAgreedToShare] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState('')

  // Browse state
  const [communityWakewords, setCommunityWakewords] = useState<CommunityWakeword[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<SortBy>('weightedRating')
  const [isLoading, setIsLoading] = useState(false)
  const [userRatings, setUserRatings] = useState<Record<string, number>>({})

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const TARGET_DURATION = 3.0
  const SPEAK_START = 0.5  // Start speaking at 0.5s
  const SPEAK_END = 2.0    // Stop speaking by 2.0s

  // Fetch community wakewords
  const fetchWakewords = useCallback(async () => {
    if (!API_BASE) return

    setIsLoading(true)
    try {
      const params = new URLSearchParams({
        sortBy,
        sortOrder: sortBy === 'name' ? 'asc' : 'desc',
        ...(searchQuery && { search: searchQuery }),
      })
      const response = await fetch(`${API_BASE}/wakewords?${params}`)
      const data = await response.json()
      setCommunityWakewords(data.wakewords || [])
    } catch (err) {
      console.error('Failed to fetch wakewords:', err)
    } finally {
      setIsLoading(false)
    }
  }, [searchQuery, sortBy])

  useEffect(() => {
    if (activeTab === 'browse') {
      fetchWakewords()
    }
  }, [activeTab, fetchWakewords])

  // Rate a wakeword
  const rateWakeword = async (wakewordId: string, rating: number) => {
    if (!API_BASE) return

    try {
      await fetch(`${API_BASE}/wakewords/${wakewordId}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, visitorId: getVisitorId() }),
      })
      setUserRatings(prev => ({ ...prev, [wakewordId]: rating }))
      fetchWakewords()
    } catch (err) {
      console.error('Failed to rate:', err)
    }
  }

  // Recording functions
  const startRecording = useCallback(async () => {
    try {
      setError(null)
      chunksRef.current = []
      setRecordingPhase('countdown')

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })
      streamRef.current = stream

      audioContextRef.current = new AudioContext({ sampleRate: 16000 })

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/mp4'
      })

      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: mediaRecorder.mimeType })
        const url = URL.createObjectURL(blob)

        const audio = new Audio(url)
        await new Promise(resolve => {
          audio.onloadedmetadata = resolve
        })

        const sample: AudioSample = {
          id: crypto.randomUUID(),
          blob,
          url,
          duration: audio.duration,
          timestamp: new Date()
        }

        setSamples(prev => [...prev, sample])
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop())
        }
        setRecordingPhase('done')
        setTimeout(() => setRecordingPhase('ready'), 1000)
      }

      mediaRecorder.start(100)
      setIsRecording(true)
      setRecordingTime(0)

      const startTime = Date.now()
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000
        setRecordingTime(elapsed)

        // Update recording phase
        if (elapsed < SPEAK_START) {
          setRecordingPhase('listen')
        } else if (elapsed < SPEAK_END) {
          setRecordingPhase('speak')
        } else {
          setRecordingPhase('listen')
        }

        if (elapsed >= TARGET_DURATION) {
          stopRecording()
        }
      }, 50)

    } catch (err) {
      setError('Microphone access denied. Please allow microphone access and try again.')
      setRecordingPhase('ready')
      console.error(err)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
  }, [isRecording])

  const deleteSample = useCallback((id: string) => {
    setSamples(prev => {
      const sample = prev.find(s => s.id === id)
      if (sample) URL.revokeObjectURL(sample.url)
      return prev.filter(s => s.id !== id)
    })
  }, [])

  const playSample = useCallback((sample: AudioSample) => {
    if (audioRef.current) audioRef.current.pause()
    const audio = new Audio(sample.url)
    audioRef.current = audio
    setPlayingId(sample.id)
    audio.onended = () => setPlayingId(null)
    audio.play()
  }, [])

  const stopPlayback = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }
    setPlayingId(null)
  }, [])

  // Upload samples to community
  const uploadToCommunity = async () => {
    if (!API_BASE || samples.length === 0 || !agreedToShare) return

    setIsUploading(true)
    setUploadProgress('Creating wakeword entry...')

    try {
      const createRes = await fetch(`${API_BASE}/wakewords`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: wakeWord, agreedToShare: true }),
      })
      const { wakeword } = await createRes.json()

      for (let i = 0; i < samples.length; i++) {
        setUploadProgress(`Uploading sample ${i + 1}/${samples.length}...`)

        const sample = samples[i]
        const urlRes = await fetch(`${API_BASE}/wakewords/${wakeword.id}/samples`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: `${wakeWord}_${i + 1}.webm`,
            contentType: 'audio/webm',
            duration: sample.duration,
            agreedToShare: true,
          }),
        })
        const { uploadUrl } = await urlRes.json()

        await fetch(uploadUrl, {
          method: 'PUT',
          headers: { 'Content-Type': 'audio/webm' },
          body: sample.blob,
        })
      }

      setUploadProgress('Upload complete!')
      setTimeout(() => {
        setUploadProgress('')
        setIsUploading(false)
        samples.forEach(s => URL.revokeObjectURL(s.url))
        setSamples([])
        setWakeWord('')
        setAgreedToShare(false)
      }, 2000)

    } catch (err) {
      console.error('Upload failed:', err)
      setError('Upload failed. Please try again.')
      setIsUploading(false)
      setUploadProgress('')
    }
  }

  // Export locally
  const exportSamples = useCallback(async () => {
    if (samples.length === 0) return

    const zip = new JSZip()
    const folder = zip.folder(wakeWord.replace(/\s+/g, '_').toLowerCase() || 'wakeword')!

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i]
      const filename = `${wakeWord.replace(/\s+/g, '_').toLowerCase() || 'sample'}_${String(i + 1).padStart(3, '0')}.webm`
      folder.file(filename, sample.blob)
    }

    const metadata = {
      wakeWord,
      sampleCount: samples.length,
      exportDate: new Date().toISOString(),
      format: 'webm (convert to 16kHz mono WAV for training)',
      targetDuration: TARGET_DURATION,
      instructions: 'Each sample contains ONE utterance of the wake word. Convert to WAV using the included script before training.',
    }
    folder.file('metadata.json', JSON.stringify(metadata, null, 2))

    const convertScript = `#!/bin/bash
# Convert WebM samples to 16kHz mono WAV for OpenWakeWord training
mkdir -p wav_output
for f in *.webm; do
  if [ -f "$f" ]; then
    name="\${f%.webm}"
    ffmpeg -i "$f" -ar 16000 -ac 1 "wav_output/\${name}.wav" -y
  fi
done
echo "Done! WAV files are in wav_output/"
echo "Total files converted: $(ls -1 wav_output/*.wav 2>/dev/null | wc -l)"
`
    folder.file('convert_to_wav.sh', convertScript)

    const content = await zip.generateAsync({ type: 'blob' })
    const url = URL.createObjectURL(content)
    const a = document.createElement('a')
    a.href = url
    a.download = `${wakeWord.replace(/\s+/g, '_').toLowerCase() || 'wakeword'}_samples.zip`
    a.click()
    URL.revokeObjectURL(url)
  }, [samples, wakeWord])

  useEffect(() => {
    return () => {
      samples.forEach(s => URL.revokeObjectURL(s.url))
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  // Get phase-specific UI
  const getRecordingUI = () => {
    switch (recordingPhase) {
      case 'countdown':
        return {
          color: 'bg-yellow-500',
          icon: '⏳',
          text: 'Get ready...',
          subtext: 'Microphone activating'
        }
      case 'listen':
        return {
          color: 'bg-blue-500',
          icon: '👂',
          text: 'Silence',
          subtext: recordingTime < SPEAK_START ? 'Wait for the cue...' : 'Hold silence...'
        }
      case 'speak':
        return {
          color: 'bg-green-500 animate-pulse',
          icon: '🗣️',
          text: `Say "${wakeWord}" NOW!`,
          subtext: 'Speak clearly, one time only'
        }
      case 'done':
        return {
          color: 'bg-purple-500',
          icon: '✅',
          text: 'Sample saved!',
          subtext: 'Ready for another'
        }
      default:
        return {
          color: wakeWord.trim() ? 'bg-purple-600 hover:bg-purple-500' : 'bg-gray-600',
          icon: '🎙️',
          text: wakeWord.trim() ? 'Click to Record' : 'Enter wake word first',
          subtext: wakeWord.trim() ? 'One sample at a time' : ''
        }
    }
  }

  const ui = getRecordingUI()

  // Star rating component
  const StarRating = ({ wakewordId, currentRating, userRating, ratingCount }: {
    wakewordId: string
    currentRating: number
    userRating?: number
    ratingCount: number
  }) => (
    <div className="flex items-center gap-2">
      <div className="flex">
        {[1, 2, 3, 4, 5].map(star => (
          <button
            key={star}
            onClick={() => rateWakeword(wakewordId, star)}
            className={`text-xl transition ${
              star <= (userRating || 0)
                ? 'text-yellow-400'
                : star <= currentRating
                  ? 'text-yellow-400/50'
                  : 'text-gray-600 hover:text-yellow-400/70'
            }`}
          >
            ★
          </button>
        ))}
      </div>
      <span className="text-purple-300 text-sm">
        {currentRating.toFixed(1)} ({ratingCount})
      </span>
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/30 border-b border-white/10">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-white">Wake Word Trainer</h1>
          <p className="text-purple-300 mt-1">
            Record samples for training custom OpenWakeWord models
          </p>
        </div>
      </header>

      {/* Quick Start Guide */}
      {showQuickStart && activeTab === 'record' && (
        <div className="max-w-4xl mx-auto px-4 pt-6">
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-2xl p-6 border border-blue-500/30">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <span className="text-2xl">📖</span> Quick Start Guide
              </h2>
              <button
                onClick={() => setShowQuickStart(false)}
                className="text-purple-400 hover:text-white text-sm"
              >
                Hide ✕
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                  <span className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center text-sm">✓</span>
                  How Each Recording Works
                </h3>
                <ol className="space-y-2 text-purple-200 text-sm">
                  <li className="flex gap-2">
                    <span className="text-blue-400 font-mono">0.0-0.5s</span>
                    <span>👂 Silence - get ready</span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-green-400 font-mono">0.5-2.0s</span>
                    <span>🗣️ <strong>Say your wake word ONCE</strong></span>
                  </li>
                  <li className="flex gap-2">
                    <span className="text-blue-400 font-mono">2.0-3.0s</span>
                    <span>👂 Silence - let it finish</span>
                  </li>
                </ol>

                <div className="mt-4 p-3 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
                  <p className="text-yellow-200 text-sm">
                    <strong>Important:</strong> Say the wake word <strong>exactly ONE time</strong> per recording.
                    The model learns from the silence before and after.
                  </p>
                </div>
              </div>

              <div>
                <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                  <span className="w-6 h-6 bg-purple-500 rounded-full flex items-center justify-center text-sm">🎯</span>
                  Tips for Better Training
                </h3>
                <ul className="space-y-2 text-purple-200 text-sm">
                  <li className="flex gap-2">
                    <span>🔊</span>
                    <span>Vary your <strong>volume</strong> - quiet, normal, loud</span>
                  </li>
                  <li className="flex gap-2">
                    <span>📏</span>
                    <span>Vary your <strong>distance</strong> - close and far from mic</span>
                  </li>
                  <li className="flex gap-2">
                    <span>⏱️</span>
                    <span>Vary your <strong>speed</strong> - slow, normal, fast</span>
                  </li>
                  <li className="flex gap-2">
                    <span>🎭</span>
                    <span>Vary your <strong>tone</strong> - tired, excited, questioning</span>
                  </li>
                  <li className="flex gap-2">
                    <span>👥</span>
                    <span>Get <strong>multiple people</strong> if possible</span>
                  </li>
                </ul>

                <div className="mt-4 p-3 bg-purple-500/20 border border-purple-500/30 rounded-lg">
                  <p className="text-purple-200 text-sm">
                    <strong>Goal:</strong> 50-100+ diverse samples create robust models that work in real conditions.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="max-w-4xl mx-auto px-4 pt-6">
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('record')}
            className={`px-6 py-3 rounded-t-lg font-medium transition ${
              activeTab === 'record'
                ? 'bg-white/10 text-white'
                : 'text-purple-300 hover:text-white'
            }`}
          >
            🎙️ Record
          </button>
          <button
            onClick={() => setActiveTab('browse')}
            className={`px-6 py-3 rounded-t-lg font-medium transition ${
              activeTab === 'browse'
                ? 'bg-white/10 text-white'
                : 'text-purple-300 hover:text-white'
            }`}
          >
            🔍 Browse Community
          </button>
          {!showQuickStart && activeTab === 'record' && (
            <button
              onClick={() => setShowQuickStart(true)}
              className="ml-auto px-4 py-2 text-purple-400 hover:text-white text-sm"
            >
              📖 Show Guide
            </button>
          )}
        </div>
      </div>

      <main className="max-w-4xl mx-auto px-4 pb-8">
        {activeTab === 'record' ? (
          <>
            {/* Wake Word Input */}
            <section className="bg-white/10 backdrop-blur rounded-2xl rounded-tl-none p-6 mb-6">
              <label className="block text-white font-medium mb-2">
                Wake Word / Phrase
                <Tooltip text="Choose a word or short phrase that will activate your voice assistant. 2-4 syllables work best. Examples: 'Hey Jarvis', 'Computer', 'OK Nexus'">
                  <InfoIcon />
                </Tooltip>
              </label>
              <input
                type="text"
                value={wakeWord}
                onChange={(e) => setWakeWord(e.target.value)}
                placeholder='e.g., "Hey Nexus" or "Computer"'
                className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500 text-lg"
              />
              <div className="flex gap-4 mt-2 text-sm text-purple-300">
                <span>💡 Good: 2-4 syllables</span>
                <span>💡 Unique sounds</span>
                <span>💡 Easy to say</span>
              </div>
            </section>

            {/* Recording Section */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
              <div className="flex flex-col items-center">
                {/* Main record button */}
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!wakeWord.trim() || recordingPhase === 'countdown'}
                  className={`relative w-36 h-36 rounded-full flex flex-col items-center justify-center text-white transition-all shadow-lg ${
                    isRecording ? ui.color : (wakeWord.trim() ? 'bg-purple-600 hover:bg-purple-500 hover:scale-105' : 'bg-gray-600 cursor-not-allowed opacity-50')
                  }`}
                >
                  <span className="text-5xl">{isRecording ? ui.icon : '🎙️'}</span>
                </button>

                {/* Status text */}
                <div className="mt-4 text-center min-h-[80px]">
                  {isRecording ? (
                    <>
                      <p className={`text-xl font-bold ${recordingPhase === 'speak' ? 'text-green-400' : 'text-white'}`}>
                        {ui.text}
                      </p>
                      <p className="text-purple-300 mt-1">{ui.subtext}</p>
                      <div className="flex items-center justify-center gap-2 mt-2">
                        <span className="text-2xl font-mono text-white">{recordingTime.toFixed(1)}s</span>
                        <span className="text-purple-400">/ {TARGET_DURATION}s</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="text-white text-lg">{ui.text}</p>
                      {ui.subtext && <p className="text-purple-300 text-sm mt-1">{ui.subtext}</p>}
                    </>
                  )}
                </div>

                {/* Timeline visualization */}
                {isRecording && (
                  <div className="w-full max-w-md mt-4">
                    <div className="relative h-8 bg-white/10 rounded-full overflow-hidden">
                      {/* Sections */}
                      <div className="absolute inset-0 flex">
                        <div className="w-[16.7%] bg-blue-500/30 border-r border-white/20" title="Silence" />
                        <div className="w-[50%] bg-green-500/30 border-r border-white/20" title="Speak" />
                        <div className="w-[33.3%] bg-blue-500/30" title="Silence" />
                      </div>
                      {/* Progress indicator */}
                      <div
                        className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-50"
                        style={{ left: `${(recordingTime / TARGET_DURATION) * 100}%` }}
                      />
                      {/* Labels */}
                      <div className="absolute inset-0 flex items-center text-xs text-white/70 pointer-events-none">
                        <span className="w-[16.7%] text-center">👂</span>
                        <span className="w-[50%] text-center font-bold text-white">🗣️ SPEAK</span>
                        <span className="w-[33.3%] text-center">👂</span>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-purple-400 mt-1">
                      <span>0s</span>
                      <span>0.5s</span>
                      <span>2.0s</span>
                      <span>3.0s</span>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200">
                  {error}
                </div>
              )}
            </section>

            {/* Progress Tracking */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                📊 Training Progress
                <Tooltip text="More samples = better model. Aim for at least 50 samples with good variety. 100+ samples create highly accurate models.">
                  <InfoIcon />
                </Tooltip>
              </h2>

              <ProgressBar current={samples.length} target={20} label="Minimum (basic training)" />
              <ProgressBar current={samples.length} target={50} label="Recommended (good accuracy)" />
              <ProgressBar current={samples.length} target={100} label="Ideal (excellent accuracy)" />

              {samples.length > 0 && samples.length < 20 && (
                <p className="text-yellow-300 text-sm mt-3">
                  ⚠️ Keep recording! You need at least 20 samples for basic training.
                </p>
              )}
              {samples.length >= 20 && samples.length < 50 && (
                <p className="text-blue-300 text-sm mt-3">
                  👍 Good start! More samples will improve accuracy.
                </p>
              )}
              {samples.length >= 50 && (
                <p className="text-green-300 text-sm mt-3">
                  ✨ Great! You have enough for a solid model. More variety always helps!
                </p>
              )}
            </section>

            {/* Samples List */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">
                  Recorded Samples ({samples.length})
                </h2>
                <div className="flex gap-2">
                  {samples.length > 0 && (
                    <>
                      <button
                        onClick={() => {
                          samples.forEach(s => URL.revokeObjectURL(s.url))
                          setSamples([])
                        }}
                        className="bg-red-600/50 hover:bg-red-600 text-white px-3 py-2 rounded-lg text-sm transition"
                      >
                        🗑️ Clear All
                      </button>
                      <button
                        onClick={exportSamples}
                        className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
                      >
                        💾 Export ZIP
                      </button>
                    </>
                  )}
                </div>
              </div>

              {samples.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-purple-300 text-lg">No samples yet</p>
                  <p className="text-purple-400 text-sm mt-2">
                    Enter your wake word above and click the record button to start
                  </p>
                </div>
              ) : (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {samples.map((sample, index) => (
                    <div key={sample.id} className="flex items-center gap-4 bg-white/5 rounded-lg p-3 hover:bg-white/10 transition">
                      <span className="text-purple-300 font-mono w-10 text-sm">#{index + 1}</span>
                      <button
                        onClick={() => playingId === sample.id ? stopPlayback() : playSample(sample)}
                        className="w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-500 text-white flex items-center justify-center transition"
                      >
                        {playingId === sample.id ? '⏹' : '▶'}
                      </button>
                      <div className="flex-1">
                        <div className="text-white text-sm">{sample.duration.toFixed(1)}s</div>
                        <div className="text-purple-400 text-xs">{sample.timestamp.toLocaleTimeString()}</div>
                      </div>
                      <button
                        onClick={() => deleteSample(sample.id)}
                        className="text-red-400 hover:text-red-300 p-2 opacity-50 hover:opacity-100 transition"
                        title="Delete sample"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Share Agreement & Upload */}
            {samples.length >= 5 && API_BASE && (
              <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
                <h2 className="text-xl font-semibold text-white mb-4">📤 Share with Community</h2>

                <div className="bg-purple-900/50 rounded-lg p-4 mb-4">
                  <label className="flex items-start gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={agreedToShare}
                      onChange={(e) => setAgreedToShare(e.target.checked)}
                      className="mt-1 w-5 h-5 rounded"
                    />
                    <div className="text-purple-200 text-sm">
                      <strong className="text-white">I agree to share these recordings publicly.</strong>
                      <p className="mt-1">
                        By uploading, you release these audio samples under the{' '}
                        <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noopener noreferrer" className="text-purple-400 underline">
                          CC0 Public Domain
                        </a>{' '}
                        license. Anyone can use them freely for wake word training.
                      </p>
                    </div>
                  </label>
                </div>

                <button
                  onClick={uploadToCommunity}
                  disabled={!agreedToShare || isUploading}
                  className={`w-full py-3 rounded-lg font-semibold transition ${
                    agreedToShare && !isUploading
                      ? 'bg-green-600 hover:bg-green-500 text-white'
                      : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isUploading ? uploadProgress : '🌐 Upload to Community Library'}
                </button>
              </section>
            )}

            {/* Next Steps */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6">
              <h2 className="text-xl font-semibold text-white mb-4">🚀 After Recording</h2>
              <div className="space-y-4 text-purple-200">
                <div className="flex gap-4 p-4 bg-white/5 rounded-lg">
                  <span className="text-3xl">1️⃣</span>
                  <div>
                    <h3 className="text-white font-medium">Export Your Samples</h3>
                    <p className="text-sm">Click "Export ZIP" to download your recordings with metadata and conversion scripts.</p>
                  </div>
                </div>
                <div className="flex gap-4 p-4 bg-white/5 rounded-lg">
                  <span className="text-3xl">2️⃣</span>
                  <div>
                    <h3 className="text-white font-medium">Convert to WAV</h3>
                    <p className="text-sm">Run <code className="bg-black/30 px-2 py-1 rounded">./convert_to_wav.sh</code> (requires FFmpeg) to get 16kHz mono WAV files.</p>
                  </div>
                </div>
                <div className="flex gap-4 p-4 bg-white/5 rounded-lg">
                  <span className="text-3xl">3️⃣</span>
                  <div>
                    <h3 className="text-white font-medium">Train Your Model</h3>
                    <p className="text-sm">
                      Use the{' '}
                      <a href="https://github.com/dscripka/openWakeWord" target="_blank" rel="noopener noreferrer" className="text-purple-400 hover:text-purple-300 underline">
                        OpenWakeWord training notebook
                      </a>{' '}
                      to create your custom ONNX model.
                    </p>
                  </div>
                </div>
                <div className="flex gap-4 p-4 bg-white/5 rounded-lg">
                  <span className="text-3xl">4️⃣</span>
                  <div>
                    <h3 className="text-white font-medium">Use in Your Project</h3>
                    <p className="text-sm">
                      Works with Home Assistant, Knowledge Nexus, or any OpenWakeWord-compatible application.
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </>
        ) : (
          /* Browse Tab */
          <section className="bg-white/10 backdrop-blur rounded-2xl rounded-tl-none p-6">
            {/* Search & Sort */}
            <div className="flex flex-col sm:flex-row gap-4 mb-6">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search wake words..."
                className="flex-1 bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as SortBy)}
                className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
              >
                <option value="weightedRating">Top Rated</option>
                <option value="sampleCount">Most Samples</option>
                <option value="createdAt">Newest</option>
                <option value="name">Alphabetical</option>
              </select>
            </div>

            {/* Wakeword List */}
            {isLoading ? (
              <div className="text-center py-12 text-purple-300">Loading...</div>
            ) : communityWakewords.length === 0 ? (
              <div className="text-center py-12">
                <p className="text-purple-300 mb-4">
                  {API_BASE ? 'No wake words yet. Be the first to contribute!' : 'Community library coming soon!'}
                </p>
                <button
                  onClick={() => setActiveTab('record')}
                  className="bg-purple-600 hover:bg-purple-500 text-white px-6 py-2 rounded-lg transition"
                >
                  🎙️ Record a Wake Word
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {communityWakewords.map(ww => (
                  <div key={ww.id} className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <h3 className="text-white font-semibold text-lg">{ww.displayName}</h3>
                        {ww.description && (
                          <p className="text-purple-300 text-sm mt-1">{ww.description}</p>
                        )}
                        <div className="flex items-center gap-4 mt-2 text-sm text-purple-400">
                          <span>📁 {ww.sampleCount} samples</span>
                        </div>
                      </div>
                      <div className="text-right">
                        <StarRating
                          wakewordId={ww.id}
                          currentRating={ww.averageRating}
                          userRating={userRatings[ww.id]}
                          ratingCount={ww.ratingCount}
                        />
                        <a
                          href={`${API_BASE}/wakewords/${ww.id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-purple-400 hover:text-purple-300 text-sm mt-2 inline-block"
                        >
                          Download samples →
                        </a>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Rating explanation */}
            <div className="mt-8 p-4 bg-purple-900/30 rounded-lg text-purple-300 text-sm">
              <strong className="text-white">How ratings work:</strong> We use a Bayesian weighted algorithm so a single 5-star doesn't outrank 25 reviews at 4.5 stars. More ratings = more accurate scores.
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center text-purple-400 text-sm">
          <p>
            Built for{' '}
            <a href="https://github.com/dscripka/openWakeWord" target="_blank" rel="noopener noreferrer" className="underline hover:text-purple-300">
              OpenWakeWord
            </a>
            {' '}• All shared samples are{' '}
            <a href="https://creativecommons.org/publicdomain/zero/1.0/" target="_blank" rel="noopener noreferrer" className="underline hover:text-purple-300">
              CC0 Public Domain
            </a>
            {' '}•{' '}
            <a href="https://jaredcluff.com" className="underline hover:text-purple-300">
              jaredcluff.com
            </a>
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
