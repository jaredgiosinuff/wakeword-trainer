#!/usr/bin/env python3
"""Generate synthetic wake word samples with diverse voices, noise, and silence padding."""

import asyncio
import subprocess
import random
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav

# Edge-TTS voices - verified working voices only
EDGE_VOICES = [
    # US English - Female (verified working)
    "en-US-JennyNeural", "en-US-AriaNeural", "en-US-MichelleNeural",
    # US English - Male (verified working)
    "en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-EricNeural",
    "en-US-RogerNeural", "en-US-SteffanNeural",
    # UK English (verified working)
    "en-GB-SoniaNeural", "en-GB-LibbyNeural", "en-GB-MaisieNeural",
    "en-GB-RyanNeural", "en-GB-ThomasNeural",
    # Australian English (verified working)
    "en-AU-NatashaNeural", "en-AU-WilliamNeural",
    # Indian English (verified working)
    "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    # Irish English (verified working)
    "en-IE-EmilyNeural", "en-IE-ConnorNeural",
    # Canadian English (verified working)
    "en-CA-ClaraNeural", "en-CA-LiamNeural",
    # Other accents (verified working)
    "en-NZ-MollyNeural", "en-NZ-MitchellNeural",
    "en-ZA-LukeNeural",
    "en-KE-AsiliaNeural", "en-KE-ChilembaNeural",
    "en-NG-AbeoNeural", "en-NG-EzinneNeural",
    "en-PH-RosaNeural", "en-PH-JamesNeural",
    "en-SG-LunaNeural", "en-SG-WayneNeural",
]

WAKE_WORD = "hey nexus"
TARGET_SAMPLE_RATE = 16000
SILENCE_BEFORE_SEC = 0.5
SILENCE_AFTER_SEC = 1.0
OUTPUT_DIR = Path("/Users/jared.cluff/gitrepos/wakeword-trainer/synthetic_samples")


def add_noise(audio: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
    """Add background noise to audio."""
    noise = np.random.randn(len(audio)) * noise_level * np.max(np.abs(audio))
    return (audio + noise).astype(np.int16)


def add_silence_padding(audio: np.ndarray, sr: int, before_sec: float, after_sec: float) -> np.ndarray:
    """Add silence before and after audio."""
    silence_before = np.zeros(int(sr * before_sec), dtype=np.int16)
    silence_after = np.zeros(int(sr * after_sec), dtype=np.int16)
    return np.concatenate([silence_before, audio, silence_after])


async def generate_sample(voice: str, output_path: Path, temp_dir: Path) -> bool:
    """Generate a single sample with edge-tts."""
    import edge_tts

    mp3_path = temp_dir / f"{output_path.stem}.mp3"
    wav_temp = temp_dir / f"{output_path.stem}_temp.wav"

    try:
        # Generate with edge-tts
        communicate = edge_tts.Communicate(WAKE_WORD, voice)
        await communicate.save(str(mp3_path))

        if not mp3_path.exists():
            return False

        # Convert to 16kHz mono WAV
        cmd = [
            "ffmpeg", "-y", "-i", str(mp3_path),
            "-ar", str(TARGET_SAMPLE_RATE), "-ac", "1",
            str(wav_temp)
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=10)

        if not wav_temp.exists():
            return False

        # Load, add noise and padding
        sr, audio = wav.read(str(wav_temp))

        # Add noise (random level between 1-5%)
        noise_level = random.uniform(0.01, 0.05)
        audio = add_noise(audio, noise_level)

        # Add silence padding
        audio = add_silence_padding(audio, sr, SILENCE_BEFORE_SEC, SILENCE_AFTER_SEC)

        # Save final file
        wav.write(str(output_path), sr, audio)

        # Cleanup temp files
        mp3_path.unlink(missing_ok=True)
        wav_temp.unlink(missing_ok=True)

        return True

    except Exception as e:
        print(f"  Error with {voice}: {e}")
        return False


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = OUTPUT_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)

    num_samples = 48  # Generate 48 more to reach 100 total
    start_index = 52  # Start from where we left off
    successful = 0

    print(f"Generating {num_samples} more synthetic samples (starting at {start_index})...")
    print(f"Wake word: '{WAKE_WORD}'")
    print(f"Silence: {SILENCE_BEFORE_SEC}s before, {SILENCE_AFTER_SEC}s after")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Distribute samples across voices
    voices_to_use = []
    while len(voices_to_use) < num_samples:
        voices_to_use.extend(EDGE_VOICES)
    random.shuffle(voices_to_use)
    voices_to_use = voices_to_use[:num_samples]

    for i, voice in enumerate(voices_to_use):
        sample_num = start_index + i
        output_path = OUTPUT_DIR / f"sample_{sample_num:03d}_{voice.replace('-', '_')}.wav"

        print(f"[{i+1}/{num_samples}] #{sample_num} {voice}...", end=" ", flush=True)

        success = await generate_sample(voice, output_path, temp_dir)

        if success:
            successful += 1
            print("OK")
        else:
            print("FAILED")

    # Cleanup temp dir
    try:
        temp_dir.rmdir()
    except:
        pass

    print()
    print(f"Generated {successful}/{num_samples} samples successfully")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
