import { useState, useRef, useCallback, useEffect } from 'react'

// Types
interface AudioSample {
  id: string
  blob: Blob
  url: string
  duration: number
  timestamp: Date
  uploaded: boolean
}

interface Session {
  id: string
  wake_word: string
  recording_count: number
  status: string
}

interface TrainingJob {
  id: string
  status: string
  progress: number
  progress_message: string
  model_path?: string
}

type Tab = 'record' | 'train'
type RecordingPhase = 'ready' | 'countdown' | 'listen' | 'speak' | 'done'

// API base URL - use relative path in production, localhost in development
const API_BASE = import.meta.env.VITE_API_URL || (
  window.location.hostname === 'localhost'
    ? 'http://localhost:8400'
    : '/wakeword-trainer'
)

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

  // Session state
  const [session, setSession] = useState<Session | null>(null)
  const [wakeWord, setWakeWord] = useState('')

  // Recording state
  const [samples, setSamples] = useState<AudioSample[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [recordingPhase, setRecordingPhase] = useState<RecordingPhase>('ready')
  const [error, setError] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  // Training state
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null)
  const [isStartingTraining, setIsStartingTraining] = useState(false)
  const [useSynthetic, setUseSynthetic] = useState(true)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const pollIntervalRef = useRef<number | null>(null)

  const TARGET_DURATION = 3.0
  const SPEAK_START = 0.5
  const SPEAK_END = 2.0
  const MIN_RECORDINGS = 20
  const RECOMMENDED_RECORDINGS = 50
  const OPTIMAL_RECORDINGS = 100

  // Create session when wake word is set
  const createSession = async (word: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ wake_word: word }),
      })

      if (!response.ok) throw new Error('Failed to create session')

      const data = await response.json()
      setSession({ id: data.session_id, wake_word: word, recording_count: 0, status: 'recording' })
      setSamples([])
      setTrainingJob(null)
      return data.session_id
    } catch (err) {
      console.error('Session creation failed:', err)
      setError('Failed to connect to server. Running in local-only mode.')
      return null
    }
  }

  // Upload recording to server
  const uploadRecording = async (sessionId: string, blob: Blob) => {
    try {
      const formData = new FormData()
      formData.append('file', blob, 'recording.webm')

      const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/recordings`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) throw new Error('Upload failed')

      const data = await response.json()
      return data
    } catch (err) {
      console.error('Upload failed:', err)
      return null
    }
  }

  // Start training
  const startTraining = async () => {
    if (!session) return

    setIsStartingTraining(true)
    try {
      const response = await fetch(`${API_BASE}/api/sessions/${session.id}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          use_synthetic: useSynthetic,
          synthetic_voices: 10,
          augmentation_factor: 5,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Failed to start training')
      }

      const data = await response.json()
      setTrainingJob(data.job)
      setActiveTab('train')

      // Start polling for status
      startPollingTrainingStatus(data.job.id)
    } catch (err: any) {
      setError(err.message || 'Failed to start training')
    } finally {
      setIsStartingTraining(false)
    }
  }

  // Poll training status
  const startPollingTrainingStatus = (jobId: string) => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }

    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/api/training/${jobId}`)
        if (!response.ok) return

        const data = await response.json()
        setTrainingJob(data.job)

        // Stop polling when complete or failed
        if (data.job.status === 'completed' || data.job.status === 'failed') {
          if (pollIntervalRef.current) {
            clearInterval(pollIntervalRef.current)
            pollIntervalRef.current = null
          }
        }
      } catch (err) {
        console.error('Status poll failed:', err)
      }
    }, 2000)
  }

  // Download trained model
  const downloadModel = async () => {
    if (!trainingJob) return

    try {
      const response = await fetch(`${API_BASE}/api/training/${trainingJob.id}/download`)
      if (!response.ok) throw new Error('Download failed')

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${session?.wake_word || 'wakeword'}.onnx`
      a.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError('Failed to download model')
    }
  }

  // Recording functions
  const startRecording = useCallback(async () => {
    try {
      setError(null)
      chunksRef.current = []
      setRecordingPhase('countdown')

      // Create session if needed
      let currentSession = session
      if (!currentSession && wakeWord.trim()) {
        const sessionId = await createSession(wakeWord.trim())
        if (sessionId) {
          currentSession = { id: sessionId, wake_word: wakeWord.trim(), recording_count: 0, status: 'recording' }
        }
      }

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
          timestamp: new Date(),
          uploaded: false,
        }

        // Upload to server if we have a session
        if (currentSession) {
          setIsUploading(true)
          const result = await uploadRecording(currentSession.id, blob)
          if (result) {
            sample.uploaded = true
            sample.id = result.recording_id
            setSession(prev => prev ? { ...prev, recording_count: result.total_recordings } : null)
          }
          setIsUploading(false)
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
  }, [session, wakeWord])

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

  const deleteSample = useCallback(async (id: string) => {
    if (session) {
      try {
        await fetch(`${API_BASE}/api/sessions/${session.id}/recordings/${id}`, {
          method: 'DELETE',
        })
      } catch (err) {
        console.error('Delete failed:', err)
      }
    }

    setSamples(prev => {
      const sample = prev.find(s => s.id === id)
      if (sample) URL.revokeObjectURL(sample.url)
      return prev.filter(s => s.id !== id)
    })

    setSession(prev => prev ? { ...prev, recording_count: Math.max(0, prev.recording_count - 1) } : null)
  }, [session])

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

  // Reset session
  const resetSession = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current)
    }
    samples.forEach(s => URL.revokeObjectURL(s.url))
    setSamples([])
    setSession(null)
    setTrainingJob(null)
    setWakeWord('')
    setActiveTab('record')
  }

  useEffect(() => {
    return () => {
      samples.forEach(s => URL.revokeObjectURL(s.url))
      if (timerRef.current) clearInterval(timerRef.current)
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
    }
  }, [])

  // Get phase-specific UI
  const getRecordingUI = () => {
    switch (recordingPhase) {
      case 'countdown':
        return { color: 'bg-yellow-500', icon: '⏳', text: 'Get ready...', subtext: 'Microphone activating' }
      case 'listen':
        return { color: 'bg-blue-500', icon: '👂', text: 'Silence', subtext: recordingTime < SPEAK_START ? 'Wait for the cue...' : 'Hold silence...' }
      case 'speak':
        return { color: 'bg-green-500 animate-pulse', icon: '🗣️', text: `Say "${wakeWord}" NOW!`, subtext: 'Speak clearly, one time only' }
      case 'done':
        return { color: 'bg-purple-500', icon: '✅', text: 'Sample saved!', subtext: isUploading ? 'Uploading...' : 'Ready for another' }
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
  const recordingCount = session?.recording_count || samples.length
  const canTrain = recordingCount >= MIN_RECORDINGS

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/30 border-b border-white/10">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-white">Wake Word Trainer</h1>
          <p className="text-purple-300 mt-1">
            Create custom voice activation phrases for your projects - completely free
          </p>
          {session && (
            <div className="mt-2 flex items-center gap-4 flex-wrap">
              <span className="text-green-400 text-sm">
                ✓ Session active for "{session.wake_word}"
              </span>
              <span className="text-purple-400 text-sm">
                ({recordingCount} recordings)
              </span>
              <span className="text-yellow-400 text-sm animate-pulse">
                ⏱️ Auto-expires in 24 hours
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Comprehensive Instructions */}
      {showQuickStart && activeTab === 'record' && (
        <div className="max-w-4xl mx-auto px-4 pt-6 space-y-4">
          {/* What is this tool */}
          <div className="bg-gradient-to-r from-purple-900/50 to-indigo-900/50 rounded-2xl p-6 border border-purple-500/30">
            <div className="flex justify-between items-start mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                <span className="text-2xl">🎯</span> What is This Tool?
              </h2>
              <button onClick={() => setShowQuickStart(false)} className="text-purple-400 hover:text-white text-sm">
                Hide Instructions ✕
              </button>
            </div>
            <p className="text-purple-200 mb-4">
              This tool lets you create <strong className="text-white">custom wake words</strong> (like "Hey Siri" or "Alexa") for your own projects.
              Record your voice saying your chosen phrase, and we'll train an AI model that can detect when you say it.
            </p>
            <div className="grid md:grid-cols-3 gap-4 text-sm">
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-2xl mb-2">🏠</div>
                <div className="text-white font-medium">Home Assistant</div>
                <div className="text-purple-300">Custom wake words for your smart home</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-2xl mb-2">🤖</div>
                <div className="text-white font-medium">Voice Assistants</div>
                <div className="text-purple-300">Build your own voice-activated apps</div>
              </div>
              <div className="bg-white/5 rounded-lg p-3">
                <div className="text-2xl mb-2">🔒</div>
                <div className="text-white font-medium">Privacy-First</div>
                <div className="text-purple-300">Runs locally, no cloud required</div>
              </div>
            </div>
          </div>

          {/* Step by Step Process */}
          <div className="bg-gradient-to-r from-blue-900/50 to-cyan-900/50 rounded-2xl p-6 border border-blue-500/30">
            <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <span className="text-2xl">📋</span> How It Works (4 Simple Steps)
            </h2>
            <div className="grid md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-2">1</div>
                <div className="text-white font-medium">Choose Your Phrase</div>
                <div className="text-blue-200 text-sm mt-1">Pick something 2-4 syllables like "Hey Nexus" or "Computer"</div>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-2">2</div>
                <div className="text-white font-medium">Record 20+ Samples</div>
                <div className="text-green-200 text-sm mt-1">Click the mic button and say your phrase (one time per recording)</div>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-purple-500 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-2">3</div>
                <div className="text-white font-medium">Train the Model</div>
                <div className="text-purple-200 text-sm mt-1">Click "Train Model" and wait a few minutes</div>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-yellow-500 rounded-full flex items-center justify-center text-white text-xl font-bold mx-auto mb-2">4</div>
                <div className="text-white font-medium">Download & Use</div>
                <div className="text-yellow-200 text-sm mt-1">Get your .onnx file and use it in your projects</div>
              </div>
            </div>
          </div>

          {/* Recording Instructions */}
          <div className="bg-gradient-to-r from-green-900/50 to-emerald-900/50 rounded-2xl p-6 border border-green-500/30">
            <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <span className="text-2xl">🎙️</span> Recording Instructions
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-white font-semibold mb-3">Each 3-Second Recording:</h3>
                <div className="bg-black/30 rounded-lg p-4 mb-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-20 h-6 bg-blue-500/50 rounded" />
                    <span className="text-blue-300 text-sm">0.0s - 0.5s: Stay silent</span>
                  </div>
                  <div className="flex items-center gap-3 mb-2">
                    <div className="w-48 h-6 bg-green-500/50 rounded" />
                    <span className="text-green-300 text-sm">0.5s - 2.0s: <strong>SAY YOUR PHRASE</strong></span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-32 h-6 bg-blue-500/50 rounded" />
                    <span className="text-blue-300 text-sm">2.0s - 3.0s: Stay silent</span>
                  </div>
                </div>
                <div className="p-3 bg-red-500/20 border border-red-500/30 rounded-lg">
                  <p className="text-red-200 text-sm">
                    <strong>⚠️ Critical:</strong> Say your wake word <strong>exactly ONE time</strong> per recording.
                    Not twice, not three times - just once! The silence before and after is important for training.
                  </p>
                </div>
              </div>
              <div>
                <h3 className="text-white font-semibold mb-3">Tips for Best Results:</h3>
                <ul className="space-y-2 text-green-200 text-sm">
                  <li className="flex gap-2 items-start">
                    <span className="text-xl">🔊</span>
                    <span><strong>Vary your volume</strong> - Some loud, some soft, some normal. This helps the model work in different situations.</span>
                  </li>
                  <li className="flex gap-2 items-start">
                    <span className="text-xl">📏</span>
                    <span><strong>Vary your distance</strong> - Some recordings close to the mic, some further away.</span>
                  </li>
                  <li className="flex gap-2 items-start">
                    <span className="text-xl">⏱️</span>
                    <span><strong>Vary your speed</strong> - Some slow and deliberate, some faster and casual.</span>
                  </li>
                  <li className="flex gap-2 items-start">
                    <span className="text-xl">🎭</span>
                    <span><strong>Vary your emotion</strong> - Tired, excited, bored, questioning - however you might actually say it.</span>
                  </li>
                  <li className="flex gap-2 items-start">
                    <span className="text-xl">👥</span>
                    <span><strong>Multiple speakers</strong> - If others will use it, have them record samples too!</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Sample Counts */}
          <div className="bg-gradient-to-r from-yellow-900/50 to-orange-900/50 rounded-2xl p-6 border border-yellow-500/30">
            <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-4">
              <span className="text-2xl">📊</span> How Many Recordings Do I Need?
            </h2>
            <div className="grid md:grid-cols-3 gap-4">
              <div className="bg-white/5 rounded-lg p-4 border-2 border-yellow-500/30">
                <div className="text-3xl font-bold text-yellow-400 mb-1">20</div>
                <div className="text-white font-medium">Minimum</div>
                <div className="text-yellow-200 text-sm">Required to train. Model will work but may have some false positives.</div>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border-2 border-green-500/50">
                <div className="text-3xl font-bold text-green-400 mb-1">50</div>
                <div className="text-white font-medium">Recommended</div>
                <div className="text-green-200 text-sm">Good balance of quality and effort. Works well for most use cases.</div>
              </div>
              <div className="bg-white/5 rounded-lg p-4 border-2 border-purple-500/50">
                <div className="text-3xl font-bold text-purple-400 mb-1">100+</div>
                <div className="text-white font-medium">Ideal</div>
                <div className="text-purple-200 text-sm">Best accuracy and lowest false positive rate. Great for production use.</div>
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
            className={`px-6 py-3 rounded-t-lg font-medium transition ${activeTab === 'record' ? 'bg-white/10 text-white' : 'text-purple-300 hover:text-white'}`}
          >
            🎙️ Record ({recordingCount})
          </button>
          <button
            onClick={() => setActiveTab('train')}
            className={`px-6 py-3 rounded-t-lg font-medium transition ${activeTab === 'train' ? 'bg-white/10 text-white' : 'text-purple-300 hover:text-white'} ${!canTrain && !trainingJob ? 'opacity-50' : ''}`}
          >
            🧠 Train Model
          </button>
          {session && (
            <button onClick={resetSession} className="ml-auto px-4 py-2 text-red-400 hover:text-red-300 text-sm">
              🔄 Start Over
            </button>
          )}
          {!showQuickStart && activeTab === 'record' && (
            <button onClick={() => setShowQuickStart(true)} className="px-4 py-2 text-purple-400 hover:text-white text-sm">
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
                <Tooltip text="Choose a word or short phrase that will activate your voice assistant. 2-4 syllables work best.">
                  <InfoIcon />
                </Tooltip>
              </label>
              <input
                type="text"
                value={wakeWord}
                onChange={(e) => setWakeWord(e.target.value)}
                placeholder='e.g., "Hey Nexus" or "Computer"'
                disabled={!!session}
                className={`w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500 text-lg ${session ? 'opacity-50 cursor-not-allowed' : ''}`}
              />
              {session && (
                <p className="text-purple-400 text-sm mt-2">Wake word locked for this session. Click "Start Over" to change.</p>
              )}
            </section>

            {/* Recording Section */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
              <div className="flex flex-col items-center">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={!wakeWord.trim() || recordingPhase === 'countdown' || isUploading}
                  className={`relative w-36 h-36 rounded-full flex flex-col items-center justify-center text-white transition-all shadow-lg ${
                    isRecording ? ui.color : (wakeWord.trim() && !isUploading ? 'bg-purple-600 hover:bg-purple-500 hover:scale-105' : 'bg-gray-600 cursor-not-allowed opacity-50')
                  }`}
                >
                  <span className="text-5xl">{isRecording ? ui.icon : (isUploading ? '⏳' : '🎙️')}</span>
                </button>

                <div className="mt-4 text-center min-h-[80px]">
                  {isRecording ? (
                    <>
                      <p className={`text-xl font-bold ${recordingPhase === 'speak' ? 'text-green-400' : 'text-white'}`}>{ui.text}</p>
                      <p className="text-purple-300 mt-1">{ui.subtext}</p>
                      <div className="flex items-center justify-center gap-2 mt-2">
                        <span className="text-2xl font-mono text-white">{recordingTime.toFixed(1)}s</span>
                        <span className="text-purple-400">/ {TARGET_DURATION}s</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <p className="text-white text-lg">{isUploading ? 'Uploading...' : ui.text}</p>
                      {ui.subtext && <p className="text-purple-300 text-sm mt-1">{ui.subtext}</p>}
                    </>
                  )}
                </div>

                {/* Timeline visualization */}
                {isRecording && (
                  <div className="w-full max-w-md mt-4">
                    <div className="relative h-8 bg-white/10 rounded-full overflow-hidden">
                      <div className="absolute inset-0 flex">
                        <div className="w-[16.7%] bg-blue-500/30 border-r border-white/20" />
                        <div className="w-[50%] bg-green-500/30 border-r border-white/20" />
                        <div className="w-[33.3%] bg-blue-500/30" />
                      </div>
                      <div className="absolute top-0 bottom-0 w-1 bg-white shadow-lg" style={{ left: `${(recordingTime / TARGET_DURATION) * 100}%` }} />
                      <div className="absolute inset-0 flex items-center text-xs text-white/70 pointer-events-none">
                        <span className="w-[16.7%] text-center">👂</span>
                        <span className="w-[50%] text-center font-bold text-white">🗣️ SPEAK</span>
                        <span className="w-[33.3%] text-center">👂</span>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-purple-400 mt-1">
                      <span>0s</span><span>0.5s</span><span>2.0s</span><span>3.0s</span>
                    </div>
                  </div>
                )}
              </div>

              {error && (
                <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200">{error}</div>
              )}
            </section>

            {/* Progress Tracking */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-6">
              <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                📊 Training Progress
                <Tooltip text="More samples = better model. Aim for at least 50 samples with good variety.">
                  <InfoIcon />
                </Tooltip>
              </h2>

              <ProgressBar current={recordingCount} target={MIN_RECORDINGS} label="Minimum (can train)" />
              <ProgressBar current={recordingCount} target={RECOMMENDED_RECORDINGS} label="Recommended (good accuracy)" />
              <ProgressBar current={recordingCount} target={OPTIMAL_RECORDINGS} label="Ideal (excellent accuracy)" />

              {recordingCount < MIN_RECORDINGS && (
                <p className="text-yellow-300 text-sm mt-3">⚠️ Keep recording! You need at least {MIN_RECORDINGS} samples to train.</p>
              )}
              {recordingCount >= MIN_RECORDINGS && recordingCount < RECOMMENDED_RECORDINGS && (
                <p className="text-blue-300 text-sm mt-3">👍 Good start! More samples will improve accuracy. Ready to train!</p>
              )}
              {recordingCount >= RECOMMENDED_RECORDINGS && (
                <p className="text-green-300 text-sm mt-3">✨ Excellent! You have enough for a high-quality model.</p>
              )}

              {canTrain && (
                <button
                  onClick={() => setActiveTab('train')}
                  className="w-full mt-4 py-3 bg-green-600 hover:bg-green-500 text-white rounded-lg font-semibold transition"
                >
                  🧠 Ready to Train → Go to Training
                </button>
              )}
            </section>

            {/* Samples List */}
            <section className="bg-white/10 backdrop-blur rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold text-white">Recorded Samples ({recordingCount})</h2>
                {samples.length > 0 && (
                  <button
                    onClick={() => {
                      samples.forEach(s => URL.revokeObjectURL(s.url))
                      setSamples([])
                    }}
                    className="bg-red-600/50 hover:bg-red-600 text-white px-3 py-2 rounded-lg text-sm transition"
                  >
                    🗑️ Clear Local
                  </button>
                )}
              </div>

              {samples.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-purple-300 text-lg">No samples yet</p>
                  <p className="text-purple-400 text-sm mt-2">Enter your wake word above and click the record button to start</p>
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
                      <span className={`text-xs px-2 py-1 rounded ${sample.uploaded ? 'bg-green-500/20 text-green-300' : 'bg-yellow-500/20 text-yellow-300'}`}>
                        {sample.uploaded ? '✓ Uploaded' : 'Local only'}
                      </span>
                      <button
                        onClick={() => deleteSample(sample.id)}
                        className="text-red-400 hover:text-red-300 p-2 opacity-50 hover:opacity-100 transition"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </section>
          </>
        ) : (
          /* Train Tab */
          <section className="bg-white/10 backdrop-blur rounded-2xl rounded-tl-none p-6">
            <h2 className="text-2xl font-bold text-white mb-6">🧠 Train Your Wake Word Model</h2>

            {!trainingJob ? (
              <>
                {/* Training options */}
                <div className="bg-white/5 rounded-lg p-6 mb-6">
                  <h3 className="text-white font-semibold mb-4">Training Options</h3>

                  <label className="flex items-start gap-3 cursor-pointer mb-4">
                    <input
                      type="checkbox"
                      checked={useSynthetic}
                      onChange={(e) => setUseSynthetic(e.target.checked)}
                      className="mt-1 w-5 h-5 rounded"
                    />
                    <div>
                      <span className="text-white font-medium">Generate synthetic samples</span>
                      <p className="text-purple-300 text-sm mt-1">
                        Use AI text-to-speech to generate additional training samples with different voices. Recommended for better accuracy.
                      </p>
                    </div>
                  </label>

                  <div className="p-4 bg-blue-500/20 border border-blue-500/30 rounded-lg">
                    <p className="text-blue-200 text-sm">
                      <strong>What happens during training:</strong>
                      <br />1. Your {recordingCount} recordings are converted to 16kHz WAV
                      <br />2. {useSynthetic ? 'Synthetic voice samples are generated' : 'No synthetic samples (your recordings only)'}
                      <br />3. Audio is augmented with speed/volume variations
                      <br />4. Neural network is trained on the data
                      <br />5. ONNX model is exported for use
                    </p>
                  </div>
                </div>

                {/* Start training */}
                <button
                  onClick={startTraining}
                  disabled={!canTrain || isStartingTraining}
                  className={`w-full py-4 rounded-lg font-bold text-lg transition ${
                    canTrain && !isStartingTraining
                      ? 'bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white'
                      : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  {isStartingTraining ? '⏳ Starting...' : canTrain ? '🚀 Start Training' : `Need ${MIN_RECORDINGS - recordingCount} more recordings`}
                </button>

                {!canTrain && (
                  <p className="text-center text-yellow-300 mt-4">
                    You need at least {MIN_RECORDINGS} recordings to train. Go back to Record tab and add more samples.
                  </p>
                )}
              </>
            ) : (
              /* Training progress */
              <div className="space-y-6">
                <div className="text-center">
                  <div className={`text-6xl mb-4 ${trainingJob.status === 'completed' ? '' : 'animate-pulse'}`}>
                    {trainingJob.status === 'completed' ? '✅' : trainingJob.status === 'failed' ? '❌' : '🔄'}
                  </div>
                  <h3 className="text-2xl font-bold text-white capitalize">
                    {trainingJob.status.replace(/_/g, ' ')}
                  </h3>
                  <p className="text-purple-300 mt-2">{trainingJob.progress_message}</p>
                </div>

                {/* Progress bar */}
                <div className="bg-white/10 rounded-full h-4 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${
                      trainingJob.status === 'failed' ? 'bg-red-500' : 'bg-gradient-to-r from-purple-500 to-green-500'
                    }`}
                    style={{ width: `${trainingJob.progress}%` }}
                  />
                </div>
                <p className="text-center text-purple-400">{trainingJob.progress}% complete</p>

                {/* Download button */}
                {trainingJob.status === 'completed' && (
                  <div className="text-center">
                    <button
                      onClick={downloadModel}
                      className="px-8 py-4 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white rounded-lg font-bold text-lg transition transform hover:scale-105"
                    >
                      📥 Download {session?.wake_word}.onnx
                    </button>
                    <p className="text-purple-300 mt-4 text-sm">
                      Use this ONNX model with OpenWakeWord, Home Assistant, or Knowledge Nexus.
                    </p>
                  </div>
                )}

                {trainingJob.status === 'failed' && (
                  <div className="text-center">
                    <p className="text-red-300 mb-4">Training failed. You can try again with more samples or different options.</p>
                    <button
                      onClick={() => setTrainingJob(null)}
                      className="px-6 py-3 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition"
                    >
                      🔄 Try Again
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Info about sessions */}
            <div className="mt-8 p-4 bg-yellow-500/20 border border-yellow-500/30 rounded-lg">
              <p className="text-yellow-200 text-sm">
                <strong>⏱️ Session Expiry:</strong> All recordings are automatically deleted after 24 hours. Make sure to complete your training and download your model before then!
              </p>
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
            {' '}•{' '}
            <a href="https://knowledgenexus.ai" className="underline hover:text-purple-300">
              Knowledge Nexus
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
