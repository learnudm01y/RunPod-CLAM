"""
Main training pipeline — called by the FastAPI server to run a CLAM training job.

Workflow:
  1. Download feature HDF5 files from Google Drive (via rclone)
  2. Build BagDataset from sample list + labels
  3. Run ClamTrainer
  4. Upload best checkpoint to Google Drive
  5. Return metrics
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .api_client import report_training_progress, report_training_finished
from .config import TrainingConfig, FEATURES_DIR, MODELS_DIR
from .io_utils import sync_features_from_gdrive, save_model_to_gdrive
from .trainer import ClamTrainer

logger = logging.getLogger(__name__)


def run_training_job(job: dict[str, Any]) -> None:
    """
    Entry point for a background training job.

    Expected job schema:
    {
        "run_id": int,
        "feature_model": str,            # e.g. "TITAN" or "Virchow2"
        "samples": [
            {
                "sample_id": int,
                "label": int,
                "gdrive_features_path": str   # relative GDrive path to .h5 file
            },
            ...
        ],
        "training_params": { ... },       # fields of TrainingConfig
        "gdrive_output_dir": str,         # GDrive path to store checkpoint
    }
    """
    run_id: int = job["run_id"]
    feature_model: str = job.get("feature_model", "unknown")
    samples_payload: list[dict] = job.get("samples", [])
    training_params: dict = job.get("training_params", {})
    gdrive_output_dir: str = job.get("gdrive_output_dir", f"training/CLAM/run_{run_id}")

    logger.info(
        "Starting training run_id=%d  feature_model=%s  n_samples=%d",
        run_id, feature_model, len(samples_payload),
    )

    # ── 1. Download features ──────────────────────────────────────────────────
    local_features_dir = os.path.join(FEATURES_DIR, feature_model, f"run_{run_id}")
    os.makedirs(local_features_dir, exist_ok=True)

    sample_data: list[dict] = []
    n_failed = 0

    for s in samples_payload:
        gdrive_path = s.get("gdrive_features_path", "")
        local_path = os.path.join(local_features_dir, os.path.basename(gdrive_path))

        ok = True
        if gdrive_path and not os.path.exists(local_path):
            ok = sync_features_from_gdrive(gdrive_path, local_features_dir)

        if ok and os.path.exists(local_path):
            sample_data.append(
                {
                    "sample_id": s["sample_id"],
                    "label": int(s["label"]),
                    "features_local_path": local_path,
                }
            )
        else:
            logger.warning("Could not get features for sample_id=%s", s.get("sample_id"))
            n_failed += 1

    if not sample_data:
        error_msg = f"No usable feature files found (all {n_failed} samples failed to download)"
        logger.error(error_msg)
        report_training_finished(run_id, {}, error=error_msg)
        return

    logger.info("Loaded %d feature bags (%d failed)", len(sample_data), n_failed)

    # ── 2. Build config ───────────────────────────────────────────────────────
    # Auto-detect in_dim from first available feature file
    import torch
    first_h5 = sample_data[0]["features_local_path"]
    try:
        import h5py
        with h5py.File(first_h5, "r") as f:
            key = "features" if "features" in f else "feats"
            in_dim = f[key].shape[-1]
    except Exception:
        in_dim = training_params.get("in_dim", 1024)

    cfg_dict = {**training_params, "in_dim": in_dim}
    cfg = TrainingConfig.from_dict(cfg_dict)
    logger.info("TrainingConfig: %s", cfg.__dict__)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    def _progress(run_id_, epoch_, total_, metrics_):
        report_training_progress(run_id_, epoch_, total_, metrics_)

    trainer = ClamTrainer(cfg, sample_data, run_id, progress_callback=_progress)
    try:
        ckpt_path, final_metrics = trainer.train()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        report_training_finished(run_id, {}, error=str(exc))
        return

    # ── 4. Upload checkpoint to GDrive ────────────────────────────────────────
    model_gdrive_path = None
    if os.path.exists(ckpt_path):
        ok = save_model_to_gdrive(ckpt_path, gdrive_output_dir)
        if ok:
            model_gdrive_path = f"{gdrive_output_dir}/{os.path.basename(ckpt_path)}"
            logger.info("Checkpoint uploaded to GDrive: %s", model_gdrive_path)

    # ── 5. Report success ─────────────────────────────────────────────────────
    report_training_finished(run_id, final_metrics, model_path=model_gdrive_path)
    logger.info("Training run_id=%d finished. Best AUC=%.4f", run_id, final_metrics.get("best_val_auc", 0))
