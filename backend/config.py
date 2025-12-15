"""Configuration for wakeword trainer backend."""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
RECORDINGS_DIR = DATA_DIR / "recordings"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"
COMMUNITY_DIR = DATA_DIR / "community"
COMMUNITY_DB_PATH = COMMUNITY_DIR / "community.json"

# Ensure directories exist
for dir_path in [RECORDINGS_DIR, MODELS_DIR, TEMP_DIR, COMMUNITY_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Recording settings
# Set MAX_RECORDING_AGE_HOURS to 0 or -1 to disable deletion (keep forever)
# Examples: 1, 24, 72, 168 (1 week), 0 (never delete)
MAX_RECORDING_AGE_HOURS = int(os.getenv("MAX_RECORDING_AGE_HOURS", "0"))  # Default: never delete
CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))

# Audio settings
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
RECORDING_DURATION_SECONDS = 3

# Training settings
MIN_RECORDINGS_FOR_TRAINING = 20
RECOMMENDED_RECORDINGS = 50
OPTIMAL_RECORDINGS = 100

# Synthetic data settings
SYNTHETIC_SAMPLES_PER_VOICE = 50
NUM_SYNTHETIC_VOICES = 10

# Community settings
COMMUNITY_VOTE_THRESHOLD = 10  # Min votes before removal eligible
COMMUNITY_REMOVAL_RATIO = 0.5  # Remove if thumbs_down > thumbs_up (more than 50% negative)
