import { useState, useRef, useCallback, useEffect } from 'react'
import JSZip from 'jszip'

interface AudioSample {
  id: string
  blob: Blob
  url: string
  duration: number
  timestamp: Date
}

function App() {
  const [wakeWord, setWakeWord] = useState('')
  const [samples, setSamples] = useState<AudioSample[]>([])
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [playingId, setPlayingId] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  // Target: 3 seconds for OpenWakeWord compatibility
  const TARGET_DURATION = 3.0

  const startRecording = useCallback(async () => {
    try {
      setError(null)
      chunksRef.current = []

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })

      // Create AudioContext for processing
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

        // Get duration
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

        // Clean up stream
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start(100)
      setIsRecording(true)
      setRecordingTime(0)

      // Start timer
      const startTime = Date.now()
      timerRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000
        setRecordingTime(elapsed)

        // Auto-stop at target duration
        if (elapsed >= TARGET_DURATION) {
          stopRecording()
        }
      }, 100)

    } catch (err) {
      setError('Microphone access denied. Please allow microphone access to record samples.')
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
      if (sample) {
        URL.revokeObjectURL(sample.url)
      }
      return prev.filter(s => s.id !== id)
    })
  }, [])

  const playSample = useCallback((sample: AudioSample) => {
    if (audioRef.current) {
      audioRef.current.pause()
    }

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

  const exportSamples = useCallback(async () => {
    if (samples.length === 0) return

    const zip = new JSZip()
    const folder = zip.folder(wakeWord.replace(/\s+/g, '_').toLowerCase() || 'wakeword')!

    for (let i = 0; i < samples.length; i++) {
      const sample = samples[i]
      const filename = `${wakeWord.replace(/\s+/g, '_').toLowerCase() || 'sample'}_${String(i + 1).padStart(3, '0')}.webm`
      folder.file(filename, sample.blob)
    }

    // Add metadata file
    const metadata = {
      wakeWord,
      sampleCount: samples.length,
      exportDate: new Date().toISOString(),
      format: 'webm (convert to 16kHz mono WAV for training)',
      targetDuration: TARGET_DURATION,
      instructions: [
        '1. Convert audio files to 16kHz mono WAV format',
        '2. Use ffmpeg: ffmpeg -i input.webm -ar 16000 -ac 1 output.wav',
        '3. Upload to OpenWakeWord training notebook',
        '4. See: https://github.com/dscripka/openWakeWord'
      ]
    }
    folder.file('metadata.json', JSON.stringify(metadata, null, 2))

    // Add conversion script
    const convertScript = `#!/bin/bash
# Convert all webm files to 16kHz mono WAV for OpenWakeWord training

mkdir -p wav_output

for f in *.webm; do
  if [ -f "$f" ]; then
    name="\${f%.webm}"
    ffmpeg -i "$f" -ar 16000 -ac 1 "wav_output/\${name}.wav" -y
    echo "Converted: $f"
  fi
done

echo "Done! WAV files are in wav_output/"
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      samples.forEach(s => URL.revokeObjectURL(s.url))
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black/30 border-b border-white/10">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-white">
            🎤 Wake Word Trainer
          </h1>
          <p className="text-purple-300 mt-1">
            Record samples for OpenWakeWord custom wake word training
          </p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        {/* Wake Word Input */}
        <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-8">
          <label className="block text-white font-medium mb-2">
            Wake Word / Phrase
          </label>
          <input
            type="text"
            value={wakeWord}
            onChange={(e) => setWakeWord(e.target.value)}
            placeholder='e.g., "Hey Jarvis" or "Computer"'
            className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <p className="text-sm text-purple-300 mt-2">
            Tip: Choose 3-4 syllables with diverse phonemes for best detection accuracy.
          </p>
        </section>

        {/* Recording Section */}
        <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-8">
          <div className="flex flex-col items-center">
            {/* Recording Button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={!wakeWord.trim()}
              className={`relative w-32 h-32 rounded-full flex items-center justify-center text-white text-5xl transition-all ${
                isRecording
                  ? 'bg-red-600 recording-pulse'
                  : wakeWord.trim()
                    ? 'bg-purple-600 hover:bg-purple-500 hover:scale-105'
                    : 'bg-gray-600 cursor-not-allowed opacity-50'
              }`}
            >
              {isRecording ? '⏹' : '🎙️'}
            </button>

            {/* Timer / Instructions */}
            <div className="mt-4 text-center">
              {isRecording ? (
                <div className="text-white">
                  <span className="text-3xl font-mono">
                    {recordingTime.toFixed(1)}s
                  </span>
                  <span className="text-purple-300 ml-2">/ {TARGET_DURATION}s</span>
                  <p className="text-purple-300 mt-2">
                    Say "{wakeWord}" now!
                  </p>
                </div>
              ) : (
                <p className="text-purple-300">
                  {wakeWord.trim()
                    ? 'Click to start recording (3 seconds)'
                    : 'Enter your wake word above to begin'}
                </p>
              )}
            </div>

            {/* Progress Bar */}
            {isRecording && (
              <div className="w-full max-w-xs mt-4 h-2 bg-white/20 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 transition-all duration-100"
                  style={{ width: `${(recordingTime / TARGET_DURATION) * 100}%` }}
                />
              </div>
            )}
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-200">
              {error}
            </div>
          )}
        </section>

        {/* Samples List */}
        <section className="bg-white/10 backdrop-blur rounded-2xl p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-white">
              Recorded Samples ({samples.length})
            </h2>
            {samples.length > 0 && (
              <button
                onClick={exportSamples}
                className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition"
              >
                📦 Export ZIP
              </button>
            )}
          </div>

          {samples.length === 0 ? (
            <p className="text-purple-300 text-center py-8">
              No samples recorded yet. Record at least 10-20 samples for basic training.
            </p>
          ) : (
            <div className="space-y-2">
              {samples.map((sample, index) => (
                <div
                  key={sample.id}
                  className="flex items-center gap-4 bg-white/5 rounded-lg p-3"
                >
                  <span className="text-purple-300 font-mono w-8">
                    #{index + 1}
                  </span>
                  <button
                    onClick={() => playingId === sample.id ? stopPlayback() : playSample(sample)}
                    className="w-10 h-10 rounded-full bg-purple-600 hover:bg-purple-500 text-white flex items-center justify-center transition"
                  >
                    {playingId === sample.id ? '⏹' : '▶'}
                  </button>
                  <div className="flex-1">
                    <div className="text-white text-sm">
                      {sample.duration.toFixed(1)}s
                    </div>
                    <div className="text-purple-400 text-xs">
                      {sample.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                  <button
                    onClick={() => deleteSample(sample.id)}
                    className="text-red-400 hover:text-red-300 transition p-2"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          )}

          {samples.length > 0 && samples.length < 10 && (
            <div className="mt-4 p-4 bg-yellow-500/20 border border-yellow-500/50 rounded-lg text-yellow-200">
              ⚠️ Record at least 10-20 samples for basic training. More samples = better accuracy.
            </div>
          )}
        </section>

        {/* Instructions */}
        <section className="bg-white/10 backdrop-blur rounded-2xl p-6">
          <h2 className="text-xl font-semibold text-white mb-4">
            📚 Training Instructions
          </h2>
          <div className="space-y-4 text-purple-200">
            <div className="flex gap-3">
              <span className="text-2xl">1️⃣</span>
              <div>
                <h3 className="text-white font-medium">Record Samples</h3>
                <p className="text-sm">Record 50-100+ samples of yourself saying the wake word. Vary your tone, speed, and distance from the mic.</p>
              </div>
            </div>
            <div className="flex gap-3">
              <span className="text-2xl">2️⃣</span>
              <div>
                <h3 className="text-white font-medium">Export & Convert</h3>
                <p className="text-sm">Export the ZIP file and run the included script to convert to 16kHz mono WAV format.</p>
              </div>
            </div>
            <div className="flex gap-3">
              <span className="text-2xl">3️⃣</span>
              <div>
                <h3 className="text-white font-medium">Train with OpenWakeWord</h3>
                <p className="text-sm">
                  Use the{' '}
                  <a
                    href="https://github.com/dscripka/openWakeWord/blob/main/notebooks/training_models.ipynb"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:text-purple-300 underline"
                  >
                    OpenWakeWord training notebook
                  </a>
                  {' '}or the{' '}
                  <a
                    href="https://www.home-assistant.io/voice_control/create_wake_word/"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:text-purple-300 underline"
                  >
                    Home Assistant guide
                  </a>.
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <span className="text-2xl">4️⃣</span>
              <div>
                <h3 className="text-white font-medium">Deploy Your Model</h3>
                <p className="text-sm">Use your trained .onnx model with OpenWakeWord, Home Assistant, or any compatible voice assistant.</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/10 mt-12">
        <div className="max-w-4xl mx-auto px-4 py-6 text-center text-purple-400 text-sm">
          <p>
            Built for{' '}
            <a href="https://github.com/dscripka/openWakeWord" target="_blank" rel="noopener noreferrer" className="underline hover:text-purple-300">
              OpenWakeWord
            </a>
            {' '}training •{' '}
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
