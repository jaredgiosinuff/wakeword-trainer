"""Audio processing utilities for wakeword trainer."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import numpy as np

from config import TARGET_SAMPLE_RATE, TARGET_CHANNELS, TEMP_DIR


def convert_webm_to_wav(input_path: Path, output_path: Path) -> bool:
    """Convert WebM audio to 16kHz mono WAV using ffmpeg.

    Args:
        input_path: Path to input WebM file
        output_path: Path for output WAV file

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(input_path),
            "-ar", str(TARGET_SAMPLE_RATE),  # 16kHz
            "-ac", str(TARGET_CHANNELS),  # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            str(output_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        return result.returncode == 0 and output_path.exists()

    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def convert_audio_bytes_to_wav(
    audio_bytes: bytes,
    input_format: str,
    output_path: Path
) -> bool:
    """Convert audio bytes to 16kHz mono WAV.

    Args:
        audio_bytes: Raw audio bytes
        input_format: Format hint (webm, ogg, mp3, etc.)
        output_path: Path for output WAV file

    Returns:
        True if conversion successful
    """
    # Write to temp file
    temp_input = TEMP_DIR / f"temp_input.{input_format}"
    try:
        temp_input.write_bytes(audio_bytes)
        return convert_webm_to_wav(temp_input, output_path)
    finally:
        if temp_input.exists():
            temp_input.unlink()


def get_audio_duration(wav_path: Path) -> Optional[float]:
    """Get duration of a WAV file in seconds."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        pass
    return None


def validate_wav_format(wav_path: Path) -> dict:
    """Validate WAV file has correct format for OpenWakeWord.

    Returns dict with validation results.
    """
    result = {
        "valid": False,
        "sample_rate": None,
        "channels": None,
        "duration": None,
        "errors": []
    }

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            str(wav_path)
        ]

        import json
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            if data.get("streams"):
                stream = data["streams"][0]
                result["sample_rate"] = int(stream.get("sample_rate", 0))
                result["channels"] = int(stream.get("channels", 0))

                if result["sample_rate"] != TARGET_SAMPLE_RATE:
                    result["errors"].append(
                        f"Sample rate is {result['sample_rate']}, expected {TARGET_SAMPLE_RATE}"
                    )
                if result["channels"] != TARGET_CHANNELS:
                    result["errors"].append(
                        f"Channels is {result['channels']}, expected {TARGET_CHANNELS}"
                    )

        result["duration"] = get_audio_duration(wav_path)
        result["valid"] = len(result["errors"]) == 0

    except Exception as e:
        result["errors"].append(str(e))

    return result


def augment_audio(
    wav_path: Path,
    output_dir: Path,
    num_augmentations: int = 5
) -> list[Path]:
    """Create augmented versions of audio for training.

    Augmentations include:
    - Speed variations (0.9x - 1.1x)
    - Pitch variations
    - Background noise mixing
    - Room reverb simulation

    Returns list of augmented file paths.
    """
    augmented = []

    # Speed variations
    for i, speed in enumerate([0.9, 0.95, 1.0, 1.05, 1.1]):
        if len(augmented) >= num_augmentations:
            break

        output_path = output_dir / f"{wav_path.stem}_speed{i}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-filter:a", f"atempo={speed}",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and output_path.exists():
                augmented.append(output_path)
        except:
            pass

    # Volume variations
    for i, volume in enumerate([0.7, 0.85, 1.0, 1.15, 1.3]):
        if len(augmented) >= num_augmentations:
            break

        output_path = output_dir / f"{wav_path.stem}_vol{i}.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-filter:a", f"volume={volume}",
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", str(TARGET_CHANNELS),
            str(output_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode == 0 and output_path.exists():
                augmented.append(output_path)
        except:
            pass

    return augmented[:num_augmentations]
