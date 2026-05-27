"""
FastAPI server for CLAM MIL training.
Port 8002 — server_id = 4

Endpoints:
  GET  /health
  POST /training/start   — start a training run
  GET  /training/{run_id} — get job status
  GET  /training/{run_id}/cancel
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

from .api_client import register_server
from .config import API_KEY, LARAVEL_SERVER_ID, PORT, RUNPOD_POD_HOSTNAME
from .main import run_training_job

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── In-memory job store ──────────────────────────────────────────────────────
# { run_id: {"status": str, "started_at": str, "thread": Thread} }
_JOBS: dict[int, dict[str, Any]] = {}

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RunPod CLAM Training Server",
    version="1.0.0",
    description="CLAM MIL training head — port 8002",
)


# ─── Auth ─────────────────────────────────────────────────────────────────────

def _verify_api_key(authorization: str = Header(...)) -> None:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    if API_KEY and token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class SamplePayload(BaseModel):
    sample_id: int
    label: int
    gdrive_features_path: str


class TrainingStartRequest(BaseModel):
    run_id: int
    feature_model: str = "TITAN"
    samples: list[SamplePayload]
    training_params: dict[str, Any] = {}
    gdrive_output_dir: str = ""


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("CLAM Training Server starting on port %d  server_id=%d", PORT, LARAVEL_SERVER_ID)

    if RUNPOD_POD_HOSTNAME:
        server_url = f"https://{RUNPOD_POD_HOSTNAME}-{PORT}.proxy.runpod.net"
    else:
        server_url = f"http://localhost:{PORT}"

    # Register asynchronously — don't block startup
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, register_server, server_url)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "clam_training",
        "server_id": LARAVEL_SERVER_ID,
        "port": PORT,
        "active_jobs": len([j for j in _JOBS.values() if j["status"] == "running"]),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/training/start", dependencies=[Depends(_verify_api_key)])
def start_training(req: TrainingStartRequest):
    run_id = req.run_id

    if run_id in _JOBS and _JOBS[run_id]["status"] == "running":
        raise HTTPException(status_code=409, detail=f"run_id={run_id} is already running")

    job_payload = {
        "run_id": run_id,
        "feature_model": req.feature_model,
        "samples": [s.model_dump() for s in req.samples],
        "training_params": req.training_params,
        "gdrive_output_dir": req.gdrive_output_dir or f"training/CLAM/run_{run_id}",
    }

    _JOBS[run_id] = {
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "run_id": run_id,
    }

    def _run():
        try:
            run_training_job(job_payload)
            _JOBS[run_id]["status"] = "completed"
        except Exception as exc:
            logger.exception("Unhandled error in training job run_id=%d: %s", run_id, exc)
            _JOBS[run_id]["status"] = "failed"
            _JOBS[run_id]["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True, name=f"clam-train-{run_id}")
    t.start()
    _JOBS[run_id]["thread"] = t

    logger.info("Dispatched training job run_id=%d", run_id)
    return {"status": "started", "run_id": run_id}


@app.get("/training/{run_id}", dependencies=[Depends(_verify_api_key)])
def get_job_status(run_id: int):
    if run_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"No job with run_id={run_id}")
    job = _JOBS[run_id]
    return {
        "run_id": run_id,
        "status": job["status"],
        "started_at": job.get("started_at"),
        "error": job.get("error"),
    }


@app.get("/training/{run_id}/cancel", dependencies=[Depends(_verify_api_key)])
def cancel_job(run_id: int):
    """Best-effort cancellation (marks as cancelled; thread will finish current epoch)."""
    if run_id not in _JOBS:
        raise HTTPException(status_code=404, detail=f"No job with run_id={run_id}")
    _JOBS[run_id]["status"] = "cancelled"
    return {"run_id": run_id, "status": "cancelled"}
