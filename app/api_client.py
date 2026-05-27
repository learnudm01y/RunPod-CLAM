"""
HTTP client — callbacks to Laravel backend from CLAM training server.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from .config import LARAVEL_BASE_URL, LARAVEL_SERVER_ID, API_KEY

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Server-ID": str(LARAVEL_SERVER_ID),
    }


def register_server(server_url: str) -> bool:
    """Tell Laravel that this CLAM server is online."""
    payload = {
        "server_id": LARAVEL_SERVER_ID,
        "server_url": server_url,
        "status": "online",
        "service": "clam_training",
        "port": 8002,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{LARAVEL_BASE_URL}/api/v1/servers/register",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            logger.info("Registered with Laravel: %s", resp.json())
            return True
    except Exception as exc:
        logger.warning("Could not register with Laravel: %s", exc)
        return False


def report_training_progress(
    run_id: int,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, Any],
) -> None:
    """Send progress update to Laravel."""
    payload = {
        "run_id": run_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "metrics": metrics,
        "status": "processing",
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            client.post(
                f"{LARAVEL_BASE_URL}/api/v1/training/progress",
                json=payload,
                headers=_headers(),
            )
    except Exception as exc:
        logger.warning("Progress callback failed: %s", exc)


def report_training_finished(
    run_id: int,
    metrics: dict[str, Any],
    model_path: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Notify Laravel that training completed (or failed)."""
    status = "failed" if error else "completed"
    payload = {
        "run_id": run_id,
        "status": status,
        "metrics": metrics or {},
        "model_path": model_path,
        "error": error,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{LARAVEL_BASE_URL}/api/v1/training/report",
                json=payload,
                headers=_headers(),
            )
            resp.raise_for_status()
            logger.info("Training report accepted by Laravel: run_id=%s status=%s", run_id, status)
    except Exception as exc:
        logger.error("Failed to send training report: %s", exc)
