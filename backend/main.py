"""Wake Word Trainer Backend API.

Provides endpoints for:
- Session management (create, get, delete)
- Recording upload and conversion
- Training job management with queue (one at a time)
- Community wake word sharing with voting
- Model download
- Automatic 24-hour cleanup for sessions
"""

import asyncio
import shutil
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import (
    RECORDINGS_DIR, MODELS_DIR, DATA_DIR, COMMUNITY_DIR, COMMUNITY_DB_PATH,
    MAX_RECORDING_AGE_HOURS, CLEANUP_INTERVAL_MINUTES,
    MIN_RECORDINGS_FOR_TRAINING, RECOMMENDED_RECORDINGS, OPTIMAL_RECORDINGS,
    COMMUNITY_VOTE_THRESHOLD, COMMUNITY_REMOVAL_RATIO
)
from models import (
    Session, SessionStatus, Recording,
    TrainingJob, TrainingStatus,
    CreateSessionRequest, CreateSessionResponse,
    SessionInfoResponse, StartTrainingRequest, TrainingStatusResponse,
    CommunityWakeWord, ShareToCommunnityRequest, VoteRequest, CommunityListResponse
)
from audio_processor import convert_webm_to_wav, validate_wav_format
from training import TrainingPipeline


# In-memory storage (use Redis/DB in production)
sessions: dict[str, Session] = {}
recordings: dict[str, list[Recording]] = {}  # session_id -> recordings
training_jobs: dict[str, TrainingJob] = {}  # job_id -> job

# Training queue - only one training at a time
training_queue: deque[str] = deque()  # Queue of job_ids
current_training_job: Optional[str] = None
training_lock = asyncio.Lock()

# Community storage
community_models: dict[str, CommunityWakeWord] = {}
community_votes: dict[str, dict[str, str]] = {}  # model_id -> {voter_id: "up"/"down"}

# Scheduler for cleanup
scheduler = AsyncIOScheduler()


# ============================================================================
# Persistence for Community Data
# ============================================================================

def load_community_data():
    """Load community data from JSON file."""
    global community_models, community_votes
    if COMMUNITY_DB_PATH.exists():
        try:
            data = json.loads(COMMUNITY_DB_PATH.read_text())
            for model_data in data.get("models", []):
                model = CommunityWakeWord(**model_data)
                community_models[model.id] = model
            community_votes = data.get("votes", {})
            print(f"Loaded {len(community_models)} community models")
        except Exception as e:
            print(f"Error loading community data: {e}")


def save_community_data():
    """Save community data to JSON file."""
    try:
        data = {
            "models": [model.model_dump(mode='json') for model in community_models.values()],
            "votes": community_votes
        }
        COMMUNITY_DB_PATH.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        print(f"Error saving community data: {e}")


# ============================================================================
# Cleanup and Queue Processing
# ============================================================================

async def cleanup_old_recordings():
    """Remove recordings older than 24 hours."""
    print(f"[{datetime.utcnow()}] Running cleanup job...")
    cutoff = datetime.utcnow() - timedelta(hours=MAX_RECORDING_AGE_HOURS)
    sessions_to_remove = []

    for session_id, session in sessions.items():
        if session.created_at < cutoff:
            sessions_to_remove.append(session_id)

    for session_id in sessions_to_remove:
        # Remove recordings from disk
        session_dir = RECORDINGS_DIR / session_id
        if session_dir.exists():
            shutil.rmtree(session_dir)
            print(f"  Removed session directory: {session_dir}")

        # Remove from memory
        sessions.pop(session_id, None)
        recordings.pop(session_id, None)

        # Remove associated training jobs (but not models - those can stay if shared)
        jobs_to_remove = [
            job_id for job_id, job in training_jobs.items()
            if job.session_id == session_id
        ]
        for job_id in jobs_to_remove:
            job = training_jobs.pop(job_id, None)
            # Only remove model if not shared to community
            if job and job.model_path:
                model_path = Path(job.model_path)
                # Check if this model is in community
                is_community = any(
                    cm.model_filename == model_path.name
                    for cm in community_models.values()
                )
                if not is_community and model_path.exists():
                    model_path.unlink()
                    print(f"  Removed model: {model_path}")

    print(f"  Cleaned up {len(sessions_to_remove)} expired sessions")


async def process_training_queue():
    """Process the next job in the training queue."""
    global current_training_job

    async with training_lock:
        if current_training_job is not None:
            # Already processing a job
            return

        if not training_queue:
            # No jobs in queue
            return

        # Get next job
        job_id = training_queue.popleft()
        job = training_jobs.get(job_id)

        if not job or job.status != TrainingStatus.QUEUED:
            # Job was cancelled or doesn't exist, try next
            asyncio.create_task(process_training_queue())
            return

        current_training_job = job_id

    # Run training outside lock
    await run_training(job_id)

    # Mark as done and process next
    async with training_lock:
        current_training_job = None

    # Process next job
    asyncio.create_task(process_training_queue())


async def run_training(job_id: str):
    """Run the training pipeline for a job."""
    job = training_jobs.get(job_id)
    if not job:
        return

    job.started_at = datetime.utcnow()
    job.status = TrainingStatus.GENERATING_SYNTHETIC

    def progress_callback(percent: int, message: str):
        job.progress = max(0, percent)
        job.progress_message = message
        if percent < 0:
            job.status = TrainingStatus.FAILED
        elif percent >= 100:
            job.status = TrainingStatus.COMPLETED

    try:
        pipeline = TrainingPipeline(
            job=job,
            recordings_dir=RECORDINGS_DIR
        )

        model_path = await pipeline.run(progress_callback)

        if model_path and model_path.exists():
            job.model_path = str(model_path)
            job.status = TrainingStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.progress = 100
            job.progress_message = "Training complete! Model ready for download."

            # Update session
            if job.session_id in sessions:
                sessions[job.session_id].status = SessionStatus.COMPLETED
        else:
            job.status = TrainingStatus.FAILED
            job.error_message = "Training completed but no model produced"

    except Exception as e:
        job.status = TrainingStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()

        if job.session_id in sessions:
            sessions[job.session_id].status = SessionStatus.FAILED


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Load community data
    load_community_data()

    # Start cleanup scheduler
    scheduler.add_job(
        cleanup_old_recordings,
        'interval',
        minutes=CLEANUP_INTERVAL_MINUTES,
        id='cleanup_job'
    )
    scheduler.start()
    print(f"Cleanup scheduler started (every {CLEANUP_INTERVAL_MINUTES} minutes)")

    yield

    # Shutdown
    scheduler.shutdown()
    save_community_data()
    print("Cleanup scheduler stopped")


app = FastAPI(
    title="Wake Word Trainer API",
    description="Backend for recording, converting, and training custom wake words",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Session Endpoints
# ============================================================================

@app.post("/api/sessions", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new recording session for a wake word."""
    session = Session(wake_word=request.wake_word.strip().lower())
    sessions[session.id] = session
    recordings[session.id] = []

    # Create session directory
    session_dir = RECORDINGS_DIR / session.id
    session_dir.mkdir(parents=True, exist_ok=True)

    return CreateSessionResponse(
        session_id=session.id,
        wake_word=session.wake_word,
        message=f"Session created. Record at least {MIN_RECORDINGS_FOR_TRAINING} samples "
                f"({RECOMMENDED_RECORDINGS} recommended, {OPTIMAL_RECORDINGS} optimal)."
    )


@app.get("/api/sessions/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str):
    """Get session info including recordings and training readiness."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session_recordings = recordings.get(session_id, [])

    recording_count = len(session_recordings)
    can_train = recording_count >= MIN_RECORDINGS_FOR_TRAINING
    recommended_more = max(0, RECOMMENDED_RECORDINGS - recording_count)

    session.recording_count = recording_count

    return SessionInfoResponse(
        session=session,
        recordings=session_recordings,
        can_train=can_train,
        recommended_more=recommended_more
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its recordings."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove recordings from disk
    session_dir = RECORDINGS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)

    # Remove from memory
    sessions.pop(session_id, None)
    recordings.pop(session_id, None)

    return {"message": "Session deleted"}


# ============================================================================
# Recording Endpoints
# ============================================================================

@app.post("/api/sessions/{session_id}/recordings")
async def upload_recording(
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload a recording for a session.

    Accepts WebM, OGG, MP3, or WAV files.
    Automatically converts to 16kHz mono WAV.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session_dir = RECORDINGS_DIR / session_id

    # Validate file type
    content_type = file.content_type or ''
    if not any(t in content_type for t in ['audio', 'webm', 'ogg', 'mpeg', 'wav']):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Allowed: audio files"
        )

    # Save uploaded file
    recording_id = str(uuid.uuid4())
    ext = Path(file.filename or 'recording.webm').suffix or '.webm'
    original_path = session_dir / f"{recording_id}{ext}"

    content = await file.read()
    original_path.write_bytes(content)

    # Convert to WAV
    wav_path = session_dir / f"{recording_id}.wav"
    conversion_success = convert_webm_to_wav(original_path, wav_path)

    if not conversion_success:
        original_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Failed to convert audio to WAV format"
        )

    # Validate WAV format
    validation = validate_wav_format(wav_path)
    if not validation["valid"]:
        wav_path.unlink()
        original_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"Audio validation failed: {validation['errors']}"
        )

    # Remove original, keep WAV
    if original_path.exists() and original_path != wav_path:
        original_path.unlink()

    # Create recording entry
    recording = Recording(
        id=recording_id,
        session_id=session_id,
        filename=f"{recording_id}.wav",
        converted=True,
        wav_path=str(wav_path),
        duration_seconds=validation.get("duration", 3.0)
    )

    recordings.setdefault(session_id, []).append(recording)
    session.updated_at = datetime.utcnow()
    session.recording_count = len(recordings[session_id])

    return {
        "recording_id": recording.id,
        "message": "Recording uploaded and converted",
        "duration": recording.duration_seconds,
        "total_recordings": session.recording_count,
        "can_train": session.recording_count >= MIN_RECORDINGS_FOR_TRAINING
    }


@app.delete("/api/sessions/{session_id}/recordings/{recording_id}")
async def delete_recording(session_id: str, recording_id: str):
    """Delete a specific recording."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_recordings = recordings.get(session_id, [])
    recording = next((r for r in session_recordings if r.id == recording_id), None)

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Remove file
    if recording.wav_path:
        wav_path = Path(recording.wav_path)
        if wav_path.exists():
            wav_path.unlink()

    # Remove from list
    recordings[session_id] = [r for r in session_recordings if r.id != recording_id]
    sessions[session_id].recording_count = len(recordings[session_id])

    return {"message": "Recording deleted"}


# ============================================================================
# Training Endpoints (with Queue)
# ============================================================================

@app.post("/api/sessions/{session_id}/train", response_model=TrainingStatusResponse)
async def start_training(
    session_id: str,
    request: StartTrainingRequest
):
    """Start training a wake word model.

    Requires at least MIN_RECORDINGS_FOR_TRAINING recordings.
    Jobs are queued and processed one at a time.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    session_recordings = recordings.get(session_id, [])

    if len(session_recordings) < MIN_RECORDINGS_FOR_TRAINING:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_RECORDINGS_FOR_TRAINING} recordings, "
                   f"got {len(session_recordings)}"
        )

    # Check if already training or queued
    if session.training_job_id:
        existing_job = training_jobs.get(session.training_job_id)
        if existing_job and existing_job.status in [
            TrainingStatus.QUEUED,
            TrainingStatus.GENERATING_SYNTHETIC,
            TrainingStatus.AUGMENTING,
            TrainingStatus.TRAINING,
            TrainingStatus.EXPORTING
        ]:
            # Return existing job status with queue position
            queue_position = get_queue_position(session.training_job_id)
            return TrainingStatusResponse(
                job=existing_job,
                download_ready=False,
                queue_position=queue_position
            )

    # Create training job
    job = TrainingJob(
        session_id=session_id,
        wake_word=session.wake_word,
        use_synthetic=request.use_synthetic,
        synthetic_voices=request.synthetic_voices,
        augmentation_factor=request.augmentation_factor
    )
    training_jobs[job.id] = job
    session.training_job_id = job.id
    session.status = SessionStatus.TRAINING

    # Add to queue
    training_queue.append(job.id)
    queue_position = len(training_queue)

    # Trigger queue processing
    asyncio.create_task(process_training_queue())

    return TrainingStatusResponse(
        job=job,
        download_ready=False,
        queue_position=queue_position
    )


def get_queue_position(job_id: str) -> Optional[int]:
    """Get the position of a job in the queue (1-indexed)."""
    if current_training_job == job_id:
        return 0  # Currently running
    try:
        return list(training_queue).index(job_id) + 1
    except ValueError:
        return None


@app.get("/api/training/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]
    download_ready = (
        job.status == TrainingStatus.COMPLETED and
        job.model_path and
        Path(job.model_path).exists()
    )

    queue_position = get_queue_position(job_id)

    return TrainingStatusResponse(
        job=job,
        download_ready=download_ready,
        queue_position=queue_position
    )


@app.get("/api/training/{job_id}/download")
async def download_model(job_id: str):
    """Download the trained ONNX model."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")

    job = training_jobs[job_id]

    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Training not complete. Status: {job.status}"
        )

    if not job.model_path or not Path(job.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        job.model_path,
        media_type="application/octet-stream",
        filename=f"{job.wake_word}.onnx"
    )


@app.get("/api/queue")
async def get_queue_status():
    """Get the current training queue status."""
    queue_jobs = []
    for job_id in training_queue:
        job = training_jobs.get(job_id)
        if job:
            queue_jobs.append({
                "job_id": job_id,
                "wake_word": job.wake_word,
                "status": job.status
            })

    return {
        "current_job": current_training_job,
        "queue_length": len(training_queue),
        "queued_jobs": queue_jobs
    }


# ============================================================================
# Community Endpoints
# ============================================================================

@app.get("/api/community", response_model=CommunityListResponse)
async def list_community_models(
    sort_by: str = "downloads",  # downloads, rating, recent
    limit: int = 50,
    offset: int = 0
):
    """List community shared wake word models."""
    models = list(community_models.values())

    # Sort
    if sort_by == "downloads":
        models.sort(key=lambda m: m.download_count, reverse=True)
    elif sort_by == "rating":
        models.sort(key=lambda m: m.thumbs_up - m.thumbs_down, reverse=True)
    elif sort_by == "recent":
        models.sort(key=lambda m: m.created_at, reverse=True)

    total = len(models)
    models = models[offset:offset + limit]

    return CommunityListResponse(
        wake_words=models,
        total=total
    )


@app.post("/api/community")
async def share_to_community(request: ShareToCommunnityRequest):
    """Share a trained model to the community."""
    job = training_jobs.get(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    if job.status != TrainingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Training must be completed first")

    if not job.model_path or not Path(job.model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    # Check if this wake word already exists in community
    existing = next(
        (m for m in community_models.values() if m.wake_word == job.wake_word),
        None
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Wake word '{job.wake_word}' already exists in community. "
                   "Vote on the existing one or use a different wake word."
        )

    # Copy model to community directory
    source_path = Path(job.model_path)
    model_filename = f"{job.wake_word}_{str(uuid.uuid4())[:8]}.onnx"
    dest_path = COMMUNITY_DIR / model_filename
    shutil.copy(source_path, dest_path)

    # Get recording count from session
    session = sessions.get(job.session_id)
    recording_count = session.recording_count if session else 0

    # Create community entry
    community_model = CommunityWakeWord(
        wake_word=job.wake_word,
        description=request.description,
        contributor=request.contributor,
        model_filename=model_filename,
        recording_count=recording_count,
        used_synthetic=job.use_synthetic,
        synthetic_voices=job.synthetic_voices,
        augmentation_factor=job.augmentation_factor
    )

    community_models[community_model.id] = community_model
    community_votes[community_model.id] = {}
    save_community_data()

    return {
        "message": "Model shared to community",
        "community_id": community_model.id,
        "wake_word": community_model.wake_word
    }


@app.get("/api/community/{model_id}")
async def get_community_model(model_id: str):
    """Get details of a community model."""
    if model_id not in community_models:
        raise HTTPException(status_code=404, detail="Community model not found")

    return community_models[model_id]


@app.get("/api/community/{model_id}/download")
async def download_community_model(model_id: str):
    """Download a community wake word model."""
    if model_id not in community_models:
        raise HTTPException(status_code=404, detail="Community model not found")

    model = community_models[model_id]
    model_path = COMMUNITY_DIR / model.model_filename

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    # Increment download count
    model.download_count += 1
    save_community_data()

    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=f"{model.wake_word}.onnx"
    )


@app.post("/api/community/{model_id}/vote")
async def vote_on_model(model_id: str, request: VoteRequest):
    """Vote on a community model (thumbs up or down)."""
    if model_id not in community_models:
        raise HTTPException(status_code=404, detail="Community model not found")

    model = community_models[model_id]
    model_votes = community_votes.setdefault(model_id, {})

    # Check if user already voted
    previous_vote = model_votes.get(request.voter_id)

    if previous_vote == request.vote:
        raise HTTPException(status_code=400, detail="You already voted this way")

    # Update vote counts
    if previous_vote:
        # Change vote
        if previous_vote == "up":
            model.thumbs_up -= 1
        else:
            model.thumbs_down -= 1

    # Add new vote
    if request.vote == "up":
        model.thumbs_up += 1
    else:
        model.thumbs_down += 1

    model_votes[request.voter_id] = request.vote

    # Check for auto-removal
    total_votes = model.thumbs_up + model.thumbs_down
    should_remove = (
        total_votes >= COMMUNITY_VOTE_THRESHOLD and
        model.thumbs_down > model.thumbs_up
    )

    if should_remove:
        # Remove model
        model_path = COMMUNITY_DIR / model.model_filename
        if model_path.exists():
            model_path.unlink()
        community_models.pop(model_id, None)
        community_votes.pop(model_id, None)
        save_community_data()
        return {
            "message": "Model removed due to community votes",
            "removed": True,
            "thumbs_up": model.thumbs_up,
            "thumbs_down": model.thumbs_down
        }

    save_community_data()

    return {
        "message": "Vote recorded",
        "removed": False,
        "thumbs_up": model.thumbs_up,
        "thumbs_down": model.thumbs_down,
        "your_vote": request.vote
    }


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_sessions": len(sessions),
        "queue_length": len(training_queue),
        "current_training": current_training_job is not None,
        "community_models": len(community_models)
    }


@app.get("/api/info")
async def get_info():
    """Get API configuration info."""
    return {
        "min_recordings": MIN_RECORDINGS_FOR_TRAINING,
        "recommended_recordings": RECOMMENDED_RECORDINGS,
        "optimal_recordings": OPTIMAL_RECORDINGS,
        "max_recording_age_hours": MAX_RECORDING_AGE_HOURS,
        "supported_formats": ["webm", "ogg", "mp3", "wav"],
        "output_format": "onnx",
        "community_vote_threshold": COMMUNITY_VOTE_THRESHOLD
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8400, reload=True)
