"""OpenWakeWord training pipeline with streaming embeddings.

This module trains wake word models that work correctly with OpenWakeWord's
streaming inference. The key insight is that training must use the same
embedding extraction method as inference (streaming chunks, not batch).
"""

import asyncio
import copy
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional
import json
import os

import edge_tts
import numpy as np
import onnx
import scipy.io.wavfile as wav
import scipy.signal
import torch
from torch import nn, optim

from config import (
    MODELS_DIR, TEMP_DIR, RECORDINGS_DIR,
    TARGET_SAMPLE_RATE, SYNTHETIC_SAMPLES_PER_VOICE
)
from models import TrainingJob, TrainingStatus
from audio_processor import augment_audio, validate_wav_format


# OpenWakeWord model constants
SAMPLE_RATE = 16000
CLIP_DURATION_SAMPLES = 32000  # 2 seconds
INPUT_SHAPE = (16, 96)  # OpenWakeWord model input shape
CHUNK_SIZE = 1280  # 80ms at 16kHz - same as streaming inference


class WakeWordDNN(nn.Module):
    """Simple DNN model matching OpenWakeWord architecture."""

    def __init__(self, input_shape=INPUT_SHAPE, layer_dim=32, n_blocks=1):
        super().__init__()
        self.input_shape = input_shape

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_shape[0] * input_shape[1], layer_dim)
        self.relu1 = nn.ReLU()
        self.layernorm1 = nn.LayerNorm(layer_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_dim, layer_dim),
                nn.ReLU(),
                nn.LayerNorm(layer_dim)
            ) for _ in range(n_blocks)
        ])

        self.output = nn.Linear(layer_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.layernorm1(self.layer1(self.flatten(x))))
        for block in self.blocks:
            x = block(x)
        return self.sigmoid(self.output(x))


class TrainingPipeline:
    """Handles the full OpenWakeWord training pipeline with streaming embeddings."""

    def __init__(self, job: TrainingJob, recordings_dir: Path):
        self.job = job
        self.recordings_dir = recordings_dir
        self.work_dir = TEMP_DIR / f"training_{job.id}"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.positive_dir = self.work_dir / "positive"
        self.augmented_dir = self.work_dir / "augmented"

        for d in [self.positive_dir, self.augmented_dir]:
            d.mkdir(exist_ok=True)

        # Feature extractor (initialized lazily)
        self._feature_extractor = None

    @property
    def feature_extractor(self):
        """Lazily initialize the OpenWakeWord feature extractor."""
        if self._feature_extractor is None:
            from openwakeword.utils import AudioFeatures
            self._feature_extractor = AudioFeatures(inference_framework='onnx')
        return self._feature_extractor

    async def run(
        self,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Optional[Path]:
        """Run the full training pipeline.

        Args:
            progress_callback: Optional callback for progress updates (percent, message)

        Returns:
            Path to trained ONNX model, or None if failed
        """
        try:
            # Step 1: Collect and validate recordings
            await self._update_progress(progress_callback, 5, "Collecting recordings...")
            wav_files = await self._collect_recordings()

            if len(wav_files) < 5:
                raise ValueError(f"Need at least 5 recordings, got {len(wav_files)}")

            print(f"[Training] Collected {len(wav_files)} recordings")

            # Step 2: Generate synthetic samples based on percentage
            if self.job.use_synthetic and self.job.synthetic_percentage > 0:
                await self._update_progress(
                    progress_callback, 10,
                    "Generating synthetic voice samples..."
                )
                # Calculate how many synthetic samples to generate based on percentage
                # If we have N user samples and want X% synthetic, we need:
                # synthetic = (N * X) / (100 - X)
                user_sample_count = len(wav_files)
                percentage = min(self.job.synthetic_percentage, 80)  # Cap at 80%
                if percentage > 0:
                    synthetic_target = int((user_sample_count * percentage) / (100 - percentage))
                    synthetic_target = max(synthetic_target, 5)  # At least 5 synthetic samples
                else:
                    synthetic_target = 0

                synthetic_files = await self._generate_synthetic_samples(target_count=synthetic_target)
                wav_files.extend(synthetic_files)

                actual_percentage = (len(synthetic_files) / (len(wav_files))) * 100 if wav_files else 0
                print(f"[Training] Added {len(synthetic_files)} synthetic samples "
                      f"({actual_percentage:.1f}% of total, target was {percentage}%)")

            # Step 3: Augment audio using existing audio_processor
            await self._update_progress(progress_callback, 20, "Augmenting audio samples...")
            augmented_files = await self._augment_samples(wav_files)
            all_positive_files = wav_files + augmented_files
            print(f"[Training] Total positive samples: {len(all_positive_files)}")

            # Step 4: Load audio and compute streaming embeddings
            await self._update_progress(
                progress_callback, 35,
                "Computing audio embeddings (streaming mode)..."
            )
            positive_audio = self._load_audio_files(all_positive_files)
            positive_embeddings = self._compute_streaming_embeddings(positive_audio)
            print(f"[Training] Positive embeddings shape: {positive_embeddings.shape}")

            # Step 5: Generate negative embeddings
            await self._update_progress(progress_callback, 50, "Generating negative samples...")
            negative_embeddings = self._generate_negative_embeddings(
                n_samples=len(positive_embeddings) * 10
            )
            print(f"[Training] Negative embeddings shape: {negative_embeddings.shape}")

            # Step 6: Train the model
            await self._update_progress(progress_callback, 60, "Training neural network...")
            model = await self._train_model(positive_embeddings, negative_embeddings)

            # Step 7: Export to ONNX
            await self._update_progress(progress_callback, 90, "Exporting ONNX model...")
            onnx_path = self._export_to_onnx(model)

            await self._update_progress(progress_callback, 100, "Training complete!")
            return onnx_path

        except Exception as e:
            import traceback
            traceback.print_exc()
            await self._update_progress(
                progress_callback, -1,
                f"Training failed: {str(e)}"
            )
            raise

        finally:
            # Cleanup work directory
            await self._cleanup()

    async def _update_progress(
        self,
        callback: Optional[Callable],
        percent: int,
        message: str
    ):
        """Update progress through callback."""
        print(f"[Training {percent}%] {message}")
        if callback:
            callback(percent, message)
        await asyncio.sleep(0)  # Yield to event loop

    async def _collect_recordings(self) -> list[Path]:
        """Collect and convert user recordings to WAV format."""
        wav_files = []

        session_dir = self.recordings_dir / self.job.session_id
        if not session_dir.exists():
            print(f"[Training] Session directory not found: {session_dir}")
            return wav_files

        for recording_file in session_dir.iterdir():
            if recording_file.suffix in ['.webm', '.ogg', '.mp3']:
                # Convert to WAV
                wav_path = self.positive_dir / f"{recording_file.stem}.wav"
                from audio_processor import convert_webm_to_wav
                if convert_webm_to_wav(recording_file, wav_path):
                    validation = validate_wav_format(wav_path)
                    if validation["valid"]:
                        wav_files.append(wav_path)

            elif recording_file.suffix == '.wav':
                # Copy WAV directly
                wav_path = self.positive_dir / recording_file.name
                shutil.copy(recording_file, wav_path)
                validation = validate_wav_format(wav_path)
                if validation["valid"]:
                    wav_files.append(wav_path)

        return wav_files

    async def _generate_synthetic_samples(self, target_count: int = 50) -> list[Path]:
        """Generate synthetic TTS samples using edge-tts (Microsoft neural voices).

        Args:
            target_count: Target number of synthetic samples to generate

        Returns:
            List of paths to generated WAV files
        """
        synthetic_files = []
        wake_word = self.job.wake_word

        if target_count <= 0:
            return synthetic_files

        synthetic_dir = self.work_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)

        # Try edge-tts first (high quality Microsoft neural voices)
        edge_files = await self._generate_edge_tts_samples(
            wake_word, target_count, synthetic_dir
        )
        synthetic_files.extend(edge_files)

        # If edge-tts didn't generate enough, fall back to Piper
        remaining = target_count - len(synthetic_files)
        if remaining > 0:
            print(f"[Training] Edge-TTS generated {len(synthetic_files)}, trying Piper for {remaining} more")
            piper_files = await self._generate_piper_samples(
                wake_word, remaining, synthetic_dir
            )
            synthetic_files.extend(piper_files)

        print(f"[Training] Total synthetic samples generated: {len(synthetic_files)}")
        return synthetic_files

    async def _generate_edge_tts_samples(
        self,
        wake_word: str,
        target_count: int,
        output_dir: Path
    ) -> list[Path]:
        """Generate samples using Microsoft Edge TTS (free neural voices).

        Args:
            wake_word: The wake word text to synthesize
            target_count: Number of samples to generate
            output_dir: Directory to save generated files

        Returns:
            List of paths to generated WAV files
        """
        synthetic_files = []

        # High-quality Microsoft neural voices - diverse accents and genders
        edge_voices = self._get_edge_tts_voices()
        num_voices = min(self.job.synthetic_voices, len(edge_voices))
        selected_voices = edge_voices[:num_voices]

        # Calculate samples per voice
        samples_per_voice = max(1, target_count // num_voices)
        extra_samples = target_count % num_voices

        print(f"[Training] Generating ~{target_count} samples using {num_voices} Edge-TTS voices")

        for voice_idx, voice in enumerate(selected_voices):
            voice_sample_count = samples_per_voice + (1 if voice_idx < extra_samples else 0)

            for sample_idx in range(voice_sample_count):
                if len(synthetic_files) >= target_count:
                    break

                output_mp3 = output_dir / f"edge_{voice_idx}_{sample_idx}.mp3"
                output_wav = output_dir / f"edge_{voice_idx}_{sample_idx}_16k.wav"

                try:
                    # Generate audio with edge-tts
                    communicate = edge_tts.Communicate(wake_word, voice)
                    await communicate.save(str(output_mp3))

                    if output_mp3.exists():
                        # Convert to 16kHz mono WAV
                        resample_cmd = [
                            "ffmpeg", "-y",
                            "-i", str(output_mp3),
                            "-ar", str(TARGET_SAMPLE_RATE),
                            "-ac", "1",
                            str(output_wav)
                        ]
                        result = subprocess.run(
                            resample_cmd, capture_output=True, timeout=10
                        )

                        if output_wav.exists():
                            synthetic_files.append(output_wav)
                            output_mp3.unlink()  # Remove temp mp3

                except Exception as e:
                    print(f"[Training] Edge-TTS error for voice {voice}: {e}")
                    continue

            if len(synthetic_files) >= target_count:
                break

        print(f"[Training] Edge-TTS generated {len(synthetic_files)} samples")
        return synthetic_files

    def _get_edge_tts_voices(self) -> list[str]:
        """Get list of high-quality Microsoft Edge TTS voices.

        Returns a curated list of English neural voices with diverse
        accents (US, UK, AU, IN, etc.) and genders for training diversity.
        """
        return [
            # US English - Female (various styles)
            "en-US-JennyNeural",
            "en-US-AriaNeural",
            "en-US-SaraNeural",
            "en-US-MichelleNeural",
            "en-US-AmberNeural",
            "en-US-AshleyNeural",
            "en-US-CoraNeural",
            "en-US-ElizabethNeural",
            "en-US-JennyMultilingualNeural",
            "en-US-MonicaNeural",
            "en-US-NancyNeural",
            # US English - Male (various styles)
            "en-US-GuyNeural",
            "en-US-DavisNeural",
            "en-US-TonyNeural",
            "en-US-JasonNeural",
            "en-US-BrandonNeural",
            "en-US-ChristopherNeural",
            "en-US-EricNeural",
            "en-US-JacobNeural",
            "en-US-RogerNeural",
            "en-US-SteffanNeural",
            # UK English
            "en-GB-SoniaNeural",
            "en-GB-LibbyNeural",
            "en-GB-MaisieNeural",
            "en-GB-RyanNeural",
            "en-GB-ThomasNeural",
            "en-GB-AlfieNeural",
            "en-GB-BellaNeural",
            "en-GB-ElliotNeural",
            "en-GB-EthanNeural",
            "en-GB-HollieNeural",
            "en-GB-NoahNeural",
            "en-GB-OliverNeural",
            "en-GB-OliviaNeural",
            # Australian English
            "en-AU-NatashaNeural",
            "en-AU-WilliamNeural",
            "en-AU-AnnetteNeural",
            "en-AU-CarlyNeural",
            "en-AU-DarrenNeural",
            "en-AU-DuncanNeural",
            "en-AU-ElsieNeural",
            "en-AU-FreyaNeural",
            "en-AU-JoanneNeural",
            "en-AU-KenNeural",
            "en-AU-KimNeural",
            "en-AU-NeilNeural",
            "en-AU-TimNeural",
            # Indian English
            "en-IN-NeerjaNeural",
            "en-IN-PrabhatNeural",
            "en-IN-AaravNeural",
            "en-IN-AnanyaNeural",
            "en-IN-KavyaNeural",
            "en-IN-KunalNeural",
            "en-IN-RehaanNeural",
            # Irish English
            "en-IE-EmilyNeural",
            "en-IE-ConnorNeural",
            # Canadian English
            "en-CA-ClaraNeural",
            "en-CA-LiamNeural",
            # New Zealand English
            "en-NZ-MollyNeural",
            "en-NZ-MitchellNeural",
            # South African English
            "en-ZA-LeahNeural",
            "en-ZA-LukeNeural",
            # Kenyan English
            "en-KE-AsiliaNeural",
            "en-KE-ChilembaNeural",
            # Nigerian English
            "en-NG-AbeoNeural",
            "en-NG-EzinneNeural",
            # Philippines English
            "en-PH-RosaNeural",
            "en-PH-JamesNeural",
            # Singapore English
            "en-SG-LunaNeural",
            "en-SG-WayneNeural",
            # Hong Kong English
            "en-HK-SamNeural",
            "en-HK-YanNeural",
        ]

    async def _generate_piper_samples(
        self,
        wake_word: str,
        target_count: int,
        output_dir: Path
    ) -> list[Path]:
        """Generate samples using Piper TTS (fallback).

        Args:
            wake_word: The wake word text to synthesize
            target_count: Number of samples to generate
            output_dir: Directory to save generated files

        Returns:
            List of paths to generated WAV files
        """
        synthetic_files = []

        if target_count <= 0:
            return synthetic_files

        try:
            # Check if piper is available
            piper_check = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                timeout=5
            )

            if piper_check.returncode != 0:
                print("[Training] Piper TTS not available")
                return synthetic_files

            # Get available voices
            all_voices = self._get_available_piper_voices()
            num_voices = min(self.job.synthetic_voices, len(all_voices))
            voice_models = all_voices[:num_voices]

            samples_per_voice = max(1, target_count // num_voices)
            extra_samples = target_count % num_voices

            print(f"[Training] Generating ~{target_count} samples using {num_voices} Piper voices")

            for voice_idx, voice_model in enumerate(voice_models):
                voice_sample_count = samples_per_voice + (1 if voice_idx < extra_samples else 0)

                for sample_idx in range(voice_sample_count):
                    if len(synthetic_files) >= target_count:
                        break

                    output_path = output_dir / f"piper_{voice_idx}_{sample_idx}.wav"
                    final_path = output_dir / f"piper_{voice_idx}_{sample_idx}_16k.wav"

                    cmd = [
                        "piper",
                        "--model", voice_model,
                        "--output_file", str(output_path),
                    ]

                    try:
                        proc = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE
                        )
                        proc.communicate(input=wake_word.encode(), timeout=30)

                        if output_path.exists():
                            resample_cmd = [
                                "ffmpeg", "-y",
                                "-i", str(output_path),
                                "-ar", str(TARGET_SAMPLE_RATE),
                                "-ac", "1",
                                str(final_path)
                            ]
                            subprocess.run(resample_cmd, capture_output=True, timeout=10)

                            if final_path.exists():
                                synthetic_files.append(final_path)
                                output_path.unlink()

                    except Exception as e:
                        print(f"[Training] Piper error for {voice_model}: {e}")
                        continue

                if len(synthetic_files) >= target_count:
                    break

            print(f"[Training] Piper generated {len(synthetic_files)} samples")

        except Exception as e:
            print(f"[Training] Piper not available: {e}")

        return synthetic_files

    def _get_available_piper_voices(self) -> list[str]:
        """Get list of available Piper voice models.

        Returns a comprehensive list of English voices for training diversity.
        Voices include different accents (US, UK, AU) and genders.
        """
        # Comprehensive list of English Piper voices for maximum diversity
        # These are auto-downloaded by Piper when requested
        default_voices = [
            # US English - Female
            "en_US-amy-medium",
            "en_US-amy-low",
            "en_US-kathleen-low",
            "en_US-lessac-medium",
            "en_US-lessac-low",
            "en_US-lessac-high",
            "en_US-libritts-high",
            "en_US-libritts_r-medium",
            "en_US-ljspeech-medium",
            "en_US-ljspeech-high",
            # US English - Male
            "en_US-joe-medium",
            "en_US-ryan-medium",
            "en_US-ryan-low",
            "en_US-ryan-high",
            "en_US-arctic-medium",
            # UK English
            "en_GB-alan-medium",
            "en_GB-alan-low",
            "en_GB-alba-medium",
            "en_GB-aru-medium",
            "en_GB-cori-medium",
            "en_GB-cori-high",
            "en_GB-jenny_dioco-medium",
            "en_GB-northern_english_male-medium",
            "en_GB-semaine-medium",
            "en_GB-southern_english_female-low",
            "en_GB-vctk-medium",
            # Australian English
            "en_AU-lessac-medium",
            # Other accents
            "en-us-blizzard_lessac-medium",
            "en-us-blizzard_lessac-high",
        ]

        available = []
        piper_models_dir = Path.home() / ".local" / "share" / "piper" / "models"

        if piper_models_dir.exists():
            for model_dir in piper_models_dir.iterdir():
                if model_dir.is_dir() and model_dir.name.startswith("en"):
                    available.append(model_dir.name)

        # Combine available with defaults, prioritizing available
        all_voices = list(set(available + default_voices))
        return all_voices

    async def _augment_samples(self, wav_files: list[Path]) -> list[Path]:
        """Augment audio samples for training diversity."""
        augmented = []

        for wav_file in wav_files:
            aug_files = augment_audio(
                wav_file,
                self.augmented_dir,
                num_augmentations=self.job.augmentation_factor
            )
            augmented.extend(aug_files)

        return augmented

    def _load_audio_files(self, wav_files: list[Path]) -> list[np.ndarray]:
        """Load audio files and normalize to 16kHz int16."""
        audio_clips = []

        for wav_file in wav_files:
            try:
                sr, audio = wav.read(str(wav_file))

                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = audio.mean(axis=1)

                # Resample if needed
                if sr != SAMPLE_RATE:
                    num_samples = int(len(audio) * SAMPLE_RATE / sr)
                    audio = scipy.signal.resample(audio, num_samples)

                # Normalize to int16
                if audio.dtype != np.int16:
                    if audio.dtype == np.float32 or audio.dtype == np.float64:
                        audio = (audio * 32767).astype(np.int16)
                    else:
                        audio = audio.astype(np.int16)

                audio_clips.append(audio)

            except Exception as e:
                print(f"[Training] Failed to load {wav_file}: {e}")

        return audio_clips

    def _pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or trim audio to target length."""
        if len(audio) < target_length:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > target_length:
            audio = audio[:target_length]
        return audio

    def _compute_streaming_embeddings(self, audio_clips: list[np.ndarray]) -> np.ndarray:
        """Compute embeddings using streaming method (matches inference).

        This is the CRITICAL function - it processes audio exactly like
        OpenWakeWord's streaming inference does, ensuring the model
        works correctly when deployed.
        """
        all_embeddings = []

        for clip in audio_clips:
            clip = self._pad_or_trim(clip, CLIP_DURATION_SAMPLES).astype(np.int16)

            # Reset feature extractor for each clip
            self.feature_extractor.reset()

            # Process clip in chunks just like streaming does
            for i in range(0, len(clip), CHUNK_SIZE):
                chunk = clip[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.feature_extractor(chunk)

            # Get the last 16 frames of embeddings (what the model sees during inference)
            if self.feature_extractor.feature_buffer.shape[0] >= 16:
                embedding = self.feature_extractor.feature_buffer[-16:].copy()
                all_embeddings.append(embedding)

        if all_embeddings:
            return np.array(all_embeddings)
        return np.array([])

    def _generate_negative_embeddings(self, n_samples: int = 5000) -> np.ndarray:
        """Generate negative embeddings from various sources.

        Creates embeddings from:
        - Silence (various levels)
        - Random noise
        - Tone sweeps
        - Random speech-like patterns
        """
        all_embeddings = []

        # Generate different types of negative samples
        samples_per_type = n_samples // 4

        # Type 1: Silence with slight noise
        for _ in range(samples_per_type):
            noise_level = np.random.uniform(0, 500)
            audio = np.random.uniform(-noise_level, noise_level, CLIP_DURATION_SAMPLES).astype(np.int16)
            self.feature_extractor.reset()
            for i in range(0, len(audio), CHUNK_SIZE):
                chunk = audio[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.feature_extractor(chunk)
            if self.feature_extractor.feature_buffer.shape[0] >= 16:
                all_embeddings.append(self.feature_extractor.feature_buffer[-16:].copy())

        # Type 2: Random noise at various levels
        for _ in range(samples_per_type):
            noise_level = np.random.uniform(1000, 15000)
            audio = np.random.uniform(-noise_level, noise_level, CLIP_DURATION_SAMPLES).astype(np.int16)
            self.feature_extractor.reset()
            for i in range(0, len(audio), CHUNK_SIZE):
                chunk = audio[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.feature_extractor(chunk)
            if self.feature_extractor.feature_buffer.shape[0] >= 16:
                all_embeddings.append(self.feature_extractor.feature_buffer[-16:].copy())

        # Type 3: Tone sweeps (simulate non-speech audio)
        for _ in range(samples_per_type):
            t = np.linspace(0, 2, CLIP_DURATION_SAMPLES)
            freq_start = np.random.uniform(100, 500)
            freq_end = np.random.uniform(500, 2000)
            freq = np.linspace(freq_start, freq_end, CLIP_DURATION_SAMPLES)
            amplitude = np.random.uniform(5000, 20000)
            audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.int16)
            self.feature_extractor.reset()
            for i in range(0, len(audio), CHUNK_SIZE):
                chunk = audio[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.feature_extractor(chunk)
            if self.feature_extractor.feature_buffer.shape[0] >= 16:
                all_embeddings.append(self.feature_extractor.feature_buffer[-16:].copy())

        # Type 4: Filtered noise (simulate speech-like audio without actual words)
        for _ in range(samples_per_type):
            audio = np.random.randn(CLIP_DURATION_SAMPLES) * 10000
            # Apply bandpass filter (300-3400 Hz, speech frequency range)
            sos = scipy.signal.butter(4, [300, 3400], btype='bandpass', fs=SAMPLE_RATE, output='sos')
            audio = scipy.signal.sosfilt(sos, audio).astype(np.int16)
            self.feature_extractor.reset()
            for i in range(0, len(audio), CHUNK_SIZE):
                chunk = audio[i:i+CHUNK_SIZE]
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                self.feature_extractor(chunk)
            if self.feature_extractor.feature_buffer.shape[0] >= 16:
                all_embeddings.append(self.feature_extractor.feature_buffer[-16:].copy())

        return np.array(all_embeddings)

    async def _train_model(
        self,
        positive_embeddings: np.ndarray,
        negative_embeddings: np.ndarray,
        steps: int = 5000
    ) -> WakeWordDNN:
        """Train the wake word model using PyTorch."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Training] Training on device: {device}")

        model = WakeWordDNN(input_shape=INPUT_SHAPE, layer_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss(reduction='none')

        # Convert to tensors
        X_pos = torch.from_numpy(positive_embeddings).float()
        X_neg = torch.from_numpy(negative_embeddings).float()

        # Split for validation
        n_val = min(len(positive_embeddings) // 5, 100)
        val_positive = X_pos[-n_val:]
        train_positive = X_pos[:-n_val] if n_val > 0 else X_pos

        n_val_neg = min(len(negative_embeddings) // 5, 500)
        val_negative = X_neg[-n_val_neg:]
        train_negative = X_neg[:-n_val_neg] if n_val_neg > 0 else X_neg

        best_model = None
        best_f1 = 0.0

        for step in range(steps):
            model.train()

            # Sample batch
            batch_size = 64
            pos_count = min(batch_size // 2, len(train_positive))
            neg_count = batch_size // 2

            pos_indices = np.random.choice(len(train_positive), pos_count, replace=True)
            neg_indices = np.random.choice(len(train_negative), neg_count, replace=False)

            X_batch = torch.cat([train_positive[pos_indices], train_negative[neg_indices]]).to(device)
            y_batch = torch.cat([
                torch.ones(pos_count),
                torch.zeros(neg_count)
            ]).to(device).unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Apply class weighting (weight negatives higher to reduce false positives)
            weight = torch.where(y_batch == 0, torch.tensor(2.0).to(device), torch.tensor(1.0).to(device))
            loss = (loss * weight).mean()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Validation every 500 steps
            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    val_X = torch.cat([val_positive, val_negative]).to(device)
                    val_y = np.concatenate([np.ones(len(val_positive)), np.zeros(len(val_negative))])

                    val_outputs = model(val_X).cpu().numpy().squeeze()
                    val_preds = (val_outputs >= 0.5).astype(int)

                    # Calculate metrics
                    tp = np.sum((val_preds == 1) & (val_y == 1))
                    fp = np.sum((val_preds == 1) & (val_y == 0))
                    fn = np.sum((val_preds == 0) & (val_y == 1))

                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    print(f"[Training] Step {step}: Loss={loss.item():.4f}, "
                          f"Recall={recall:.3f}, Precision={precision:.3f}, F1={f1:.3f}")

                    # Save best model
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = copy.deepcopy(model.state_dict())

                # Yield to event loop periodically
                await asyncio.sleep(0)

        # Load best model
        if best_model is not None:
            model.load_state_dict(best_model)
            print(f"[Training] Loaded best model with F1={best_f1:.3f}")

        return model

    def _export_to_onnx(self, model: WakeWordDNN) -> Path:
        """Export trained model to ONNX format."""
        model.eval()
        model.cpu()

        # Sanitize wake word for output name
        output_name = self.job.wake_word.replace(' ', '_').replace('-', '_')

        # Output path
        output_path = MODELS_DIR / f"{output_name}_{self.job.id[:8]}.onnx"

        # Create dummy input
        dummy_input = torch.randn(1, *INPUT_SHAPE)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=['input'],
            output_names=[output_name],
            dynamic_axes={'input': {0: 'batch_size'}},
            opset_version=11,
        )

        # Fix IR version for ARM64 compatibility
        onnx_model = onnx.load(str(output_path))
        onnx_model.ir_version = 6  # Required for ARM64 onnxruntime
        onnx.save(onnx_model, str(output_path))

        print(f"[Training] Model exported to {output_path}")
        return output_path

    async def _cleanup(self):
        """Clean up temporary training files."""
        try:
            if self.work_dir.exists():
                shutil.rmtree(self.work_dir)
        except Exception as e:
            print(f"[Training] Cleanup error: {e}")
