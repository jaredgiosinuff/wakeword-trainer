"""Data models for wakeword trainer."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class SessionStatus(str, Enum):
    RECORDING = "recording"
    READY_TO_TRAIN = "ready_to_train"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingStatus(str, Enum):
    QUEUED = "queued"
    GENERATING_SYNTHETIC = "generating_synthetic"
    AUGMENTING = "augmenting"
    TRAINING = "training"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"


class Recording(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: float = 3.0
    converted: bool = False
    wav_path: Optional[str] = None


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wake_word: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.RECORDING
    recording_count: int = 0
    training_job_id: Optional[str] = None


class TrainingJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    wake_word: str
    status: TrainingStatus = TrainingStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0  # 0-100
    progress_message: str = "Queued"
    model_path: Optional[str] = None
    error_message: Optional[str] = None

    # Training parameters
    use_synthetic: bool = True
    synthetic_voices: int = 10
    augmentation_factor: int = 5


# Request/Response models
class CreateSessionRequest(BaseModel):
    wake_word: str = Field(..., min_length=2, max_length=50)


class CreateSessionResponse(BaseModel):
    session_id: str
    wake_word: str
    message: str


class SessionInfoResponse(BaseModel):
    session: Session
    recordings: list[Recording]
    can_train: bool
    recommended_more: int


class StartTrainingRequest(BaseModel):
    use_synthetic: bool = True
    synthetic_voices: int = 10
    augmentation_factor: int = 5


class TrainingStatusResponse(BaseModel):
    job: TrainingJob
    download_ready: bool
    queue_position: Optional[int] = None  # 0 = currently running, 1+ = position in queue


# Community models
class CommunityWakeWord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wake_word: str
    description: Optional[str] = None
    contributor: Optional[str] = None  # Optional username/alias
    created_at: datetime = Field(default_factory=datetime.utcnow)
    model_filename: str
    download_count: int = 0
    thumbs_up: int = 0
    thumbs_down: int = 0

    # Training metadata (helps users evaluate quality)
    recording_count: int = 0
    used_synthetic: bool = True
    synthetic_voices: int = 0
    augmentation_factor: int = 0


class ShareToCommunnityRequest(BaseModel):
    job_id: str
    description: Optional[str] = None
    contributor: Optional[str] = None


class VoteRequest(BaseModel):
    vote: str = Field(..., pattern="^(up|down)$")  # "up" or "down"
    voter_id: str  # Browser fingerprint or session ID to prevent double voting


class CommunityListResponse(BaseModel):
    wake_words: list[CommunityWakeWord]
    total: int
