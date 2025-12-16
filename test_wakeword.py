#!/usr/bin/env python3
"""Test wake word model detection accuracy.

Usage:
    python test_wakeword.py <model_path> [--positive-dir DIR] [--threshold 0.5]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wav

def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and resample to target rate."""
    sr, audio = wav.read(str(path))

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import scipy.signal
        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_samples)

    # Convert to int16
    if audio.dtype != np.int16:
        if audio.dtype in (np.float32, np.float64):
            audio = (audio * 32767).astype(np.int16)
        else:
            audio = audio.astype(np.int16)

    return audio


def test_detection(model_path: str, audio_files: list[Path], threshold: float = 0.5) -> dict:
    """Test wake word detection on audio files."""
    from openwakeword.model import Model

    # Load model
    print(f"Loading model: {model_path}")
    model = Model(wakeword_models=[model_path], inference_framework='onnx')

    results = {
        'total': len(audio_files),
        'detected': 0,
        'not_detected': 0,
        'scores': [],
        'files': []
    }

    chunk_size = 1280  # 80ms at 16kHz

    for audio_path in audio_files:
        print(f"Testing: {audio_path.name}...", end=" ")

        try:
            audio = load_audio(audio_path)
            model.reset()

            max_score = 0.0
            # Process in chunks like streaming
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                predictions = model.predict(chunk)
                for name, score in predictions.items():
                    max_score = max(max_score, score)

            detected = max_score >= threshold
            results['scores'].append(max_score)
            results['files'].append({
                'name': audio_path.name,
                'score': max_score,
                'detected': detected
            })

            if detected:
                results['detected'] += 1
                print(f"DETECTED (score={max_score:.4f})")
            else:
                results['not_detected'] += 1
                print(f"not detected (score={max_score:.4f})")

        except Exception as e:
            print(f"ERROR: {e}")
            results['not_detected'] += 1

    return results


def generate_negative_samples(output_dir: Path, count: int = 20) -> list[Path]:
    """Generate negative test samples (random noise, silence, other speech)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    sr = 16000
    duration = 2  # seconds
    samples = sr * duration

    # Generate noise samples
    for i in range(count // 2):
        noise = np.random.randn(samples) * 5000
        path = output_dir / f"noise_{i:03d}.wav"
        wav.write(str(path), sr, noise.astype(np.int16))
        files.append(path)

    # Generate silence samples
    for i in range(count // 4):
        silence = np.zeros(samples, dtype=np.int16)
        path = output_dir / f"silence_{i:03d}.wav"
        wav.write(str(path), sr, silence)
        files.append(path)

    # Generate tone samples
    for i in range(count // 4):
        t = np.linspace(0, duration, samples)
        freq = np.random.uniform(200, 1000)
        tone = (np.sin(2 * np.pi * freq * t) * 10000).astype(np.int16)
        path = output_dir / f"tone_{i:03d}.wav"
        wav.write(str(path), sr, tone)
        files.append(path)

    return files


def main():
    parser = argparse.ArgumentParser(description="Test wake word detection")
    parser.add_argument("model_path", help="Path to ONNX model")
    parser.add_argument("--positive-dir", "-p", help="Directory with positive samples")
    parser.add_argument("--negative-dir", "-n", help="Directory with negative samples")
    parser.add_argument("--threshold", "-t", type=float, default=0.5)
    parser.add_argument("--generate-negatives", "-g", action="store_true",
                        help="Generate synthetic negative samples")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    # Collect positive samples
    positive_files = []
    if args.positive_dir:
        pos_dir = Path(args.positive_dir)
        positive_files = list(pos_dir.glob("*.wav"))
        print(f"Found {len(positive_files)} positive samples")

    # Collect/generate negative samples
    negative_files = []
    if args.negative_dir:
        neg_dir = Path(args.negative_dir)
        negative_files = list(neg_dir.glob("*.wav"))
    elif args.generate_negatives:
        print("Generating negative samples...")
        neg_dir = Path("/tmp/wakeword_neg_samples")
        negative_files = generate_negative_samples(neg_dir, 20)

    print(f"Found {len(negative_files)} negative samples")
    print(f"Threshold: {args.threshold}")
    print()

    # Test positive samples (should detect)
    if positive_files:
        print("=" * 60)
        print("POSITIVE SAMPLES (should detect)")
        print("=" * 60)
        pos_results = test_detection(str(model_path), positive_files, args.threshold)

        detection_rate = pos_results['detected'] / pos_results['total'] * 100
        avg_score = np.mean(pos_results['scores'])

        print()
        print(f"Detection Rate: {pos_results['detected']}/{pos_results['total']} ({detection_rate:.1f}%)")
        print(f"Average Score: {avg_score:.4f}")
        print()

    # Test negative samples (should NOT detect)
    if negative_files:
        print("=" * 60)
        print("NEGATIVE SAMPLES (should NOT detect)")
        print("=" * 60)
        neg_results = test_detection(str(model_path), negative_files, args.threshold)

        false_positive_rate = neg_results['detected'] / neg_results['total'] * 100
        avg_score = np.mean(neg_results['scores'])

        print()
        print(f"False Positive Rate: {neg_results['detected']}/{neg_results['total']} ({false_positive_rate:.1f}%)")
        print(f"Average Score: {avg_score:.4f}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if positive_files:
        print(f"  True Positive Rate:  {detection_rate:.1f}%")
    if negative_files:
        print(f"  False Positive Rate: {false_positive_rate:.1f}%")

    # Success criteria
    if positive_files and negative_files:
        success = detection_rate >= 80 and false_positive_rate <= 10
        print()
        print(f"  TEST {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
