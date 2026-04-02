"""
QuranClip API — FastAPI backend for Taraweeh audio extraction.

Endpoints match what the React frontend expects:
  POST /api/upload         — single-file upload (or YouTube URL placeholder)
  POST /api/upload-chunk   — chunked upload for large files
  GET  /api/status/{id}    — poll job progress
  GET  /api/download/{id}  — download finished MP3
"""

import os
import uuid
import shutil
import asyncio
import traceback
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline import process_audio

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="QuranClip API")

WORK_DIR = Path(os.environ.get("QURANCLIP_WORK_DIR", "/tmp/quranclip"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job store  { job_id: { status, stage, progress, error, summary, dir } }
jobs: dict[str, dict] = {}

# In-memory chunk store  { upload_id: { dir, total, received } }
chunk_store: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def new_job(audio_path: Path, remove_fatiha: bool) -> str:
    job_id = uuid.uuid4().hex[:12]
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Move uploaded file into job dir
    dest = job_dir / "input.audio"
    shutil.move(str(audio_path), str(dest))

    jobs[job_id] = {
        "status": "processing",
        "stage": "converting",
        "progress": 0.0,
        "error": None,
        "summary": None,
        "dir": str(job_dir),
    }

    # Run pipeline in background
    asyncio.get_event_loop().create_task(
        run_pipeline(job_id, dest, remove_fatiha)
    )
    return job_id


async def run_pipeline(job_id: str, audio_path: Path, remove_fatiha: bool):
    job = jobs[job_id]

    def update(stage: str, progress: float):
        job["stage"] = stage
        job["progress"] = progress

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            process_audio,
            audio_path,
            Path(job["dir"]),
            remove_fatiha,
            update,
        )
        job["status"] = "completed"
        job["stage"] = "done"
        job["progress"] = 1.0
        job["summary"] = result
    except Exception as e:
        traceback.print_exc()
        job["status"] = "failed"
        job["error"] = str(e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(
    audio: Optional[UploadFile] = File(None),
    youtube_url: Optional[str] = Form(None),
    remove_fatiha: str = Form("true"),
):
    if youtube_url:
        raise HTTPException(501, detail="YouTube URL support coming soon.")

    if not audio:
        raise HTTPException(400, detail="Provide an audio file.")

    tmp_path = WORK_DIR / f"upload_{uuid.uuid4().hex[:8]}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(audio.file, f)

    job_id = new_job(tmp_path, remove_fatiha.lower() == "true")
    return {"jobId": job_id}


@app.post("/api/upload-chunk")
async def upload_chunk(
    chunk: UploadFile = File(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    upload_id: Optional[str] = Form(None),
    remove_fatiha: str = Form("true"),
):
    # First chunk — create upload session
    if upload_id is None or upload_id not in chunk_store:
        upload_id = uuid.uuid4().hex[:10]
        upload_dir = WORK_DIR / f"chunks_{upload_id}"
        upload_dir.mkdir(parents=True, exist_ok=True)
        chunk_store[upload_id] = {
            "dir": str(upload_dir),
            "total": total_chunks,
            "received": set(),
        }

    session = chunk_store[upload_id]
    chunk_path = Path(session["dir"]) / f"chunk_{chunk_index:05d}"
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(chunk.file, f)
    session["received"].add(chunk_index)

    # All chunks received — assemble and start job
    if len(session["received"]) >= session["total"]:
        assembled = WORK_DIR / f"assembled_{upload_id}"
        with open(assembled, "wb") as out:
            for i in range(session["total"]):
                part = Path(session["dir"]) / f"chunk_{i:05d}"
                with open(part, "rb") as p:
                    shutil.copyfileobj(p, out)

        # Clean up chunks
        shutil.rmtree(session["dir"], ignore_errors=True)
        del chunk_store[upload_id]

        job_id = new_job(assembled, remove_fatiha.lower() == "true")
        return {"uploadId": upload_id, "jobId": job_id}

    return {"uploadId": upload_id}


@app.get("/api/status/healthcheck")
async def healthcheck():
    """Railway healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, detail="Job not found")
    job = jobs[job_id]
    return {
        "jobId": job_id,
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "summary": job["summary"],
    }


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(400, detail="Job not completed yet")

    output_path = Path(job["dir"]) / "output.mp3"
    if not output_path.exists():
        raise HTTPException(404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="audio/mpeg",
        filename="quran-recitation.mp3",
    )


# ---------------------------------------------------------------------------
# Serve frontend static files in production
# ---------------------------------------------------------------------------

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
