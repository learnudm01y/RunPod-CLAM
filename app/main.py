"""
Main training pipeline — called by the FastAPI server to run a CLAM training job.

Workflow:
  1. Download feature HDF5 files from Google Drive (via rclone)
  2. Resolve the sample list + labels (fine, and coarse when hierarchical)
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
        # Preferred: explicit three-way split from Laravel
        "samples_train": [
            {"sample_id": int, "label": int, "parent_label": int, "gdrive_features_path": str}, ...
        ],
        "samples_val":   [ ... ],
        "samples_test":  [ ... ],
        # Legacy fallback: flat list (server splits 80/20 if samples_train is empty)
        "samples": [ ... ],
        "training_params": { ... },       # fields of TrainingConfig
        "label_map": [str, ...],          # fine class names, index-aligned
        "parent_label_map": [str, ...],   # coarse class names (hierarchical runs)
        "gdrive_output_dir": str,         # GDrive path to store checkpoint
    }

    `label` is always ONE fine class index (the exact disease entity).
    `parent_label` is its coarse family index, or -1 / absent for flat runs.
    """
    run_id: int = job["run_id"]
    feature_model: str = job.get("feature_model", "unknown")
    training_params: dict = job.get("training_params", {})
    gdrive_output_dir: str = job.get("gdrive_output_dir", f"training/CLAM/run_{run_id}")
    label_map: list = job.get("label_map", []) or []
    parent_label_map: list = job.get("parent_label_map", []) or []

    # ── Resolve split source ──────────────────────────────────────────────────
    # Prefer explicit three-way split; fall back to legacy flat list
    samples_train_raw: list[dict] = job.get("samples_train", [])
    samples_val_raw:   list[dict] = job.get("samples_val",   [])
    samples_test_raw:  list[dict] = job.get("samples_test",  [])
    samples_flat_raw:  list[dict] = job.get("samples",       [])

    use_explicit_split = bool(samples_train_raw)  # True when Laravel sent explicit split

    if use_explicit_split:
        all_samples_raw = samples_train_raw + samples_val_raw + samples_test_raw
        logger.info(
            "Starting training run_id=%d  feature_model=%s  explicit_split: "
            "train=%d val=%d test=%d",
            run_id, feature_model,
            len(samples_train_raw), len(samples_val_raw), len(samples_test_raw),
        )
    else:
        all_samples_raw = samples_flat_raw
        logger.warning(
            "Starting training run_id=%d  feature_model=%s  NO explicit split provided — "
            "will fall back to random 80/20 train/val split  n_samples=%d",
            run_id, feature_model, len(all_samples_raw),
        )

    # ── 1. Download features ──────────────────────────────────────────────────
    local_features_dir = os.path.join(FEATURES_DIR, feature_model, f"run_{run_id}")
    os.makedirs(local_features_dir, exist_ok=True)

    def _resolve_samples(raw_list: list[dict]) -> tuple[list[dict], int]:
        """Download features and return (resolved_sample_data, n_failed)."""
        resolved = []
        failed = 0
        for s in raw_list:
            gdrive_path = s.get("gdrive_features_path", "")
            local_path = os.path.join(local_features_dir, os.path.basename(gdrive_path))
            ok = True
            if gdrive_path and not os.path.exists(local_path):
                ok = sync_features_from_gdrive(gdrive_path, local_features_dir)
            if ok and os.path.exists(local_path):
                resolved.append({
                    "sample_id":           s["sample_id"],
                    "label":               int(s["label"]),
                    "parent_label":        int(s.get("parent_label", -1)),
                    "training_phase":      int(s.get("training_phase", 1)),
                    "features_local_path": local_path,
                })
            else:
                logger.warning("Could not get features for sample_id=%s", s.get("sample_id"))
                failed += 1
        return resolved, failed

    if use_explicit_split:
        train_data, n_failed_train = _resolve_samples(samples_train_raw)
        val_data,   n_failed_val   = _resolve_samples(samples_val_raw)
        test_data,  n_failed_test  = _resolve_samples(samples_test_raw)
        n_failed = n_failed_train + n_failed_val + n_failed_test
        sample_data = train_data + val_data + test_data
        logger.info(
            "Features resolved — train=%d val=%d test=%d  failed=%d",
            len(train_data), len(val_data), len(test_data), n_failed,
        )
    else:
        sample_data, n_failed = _resolve_samples(all_samples_raw)
        train_data, val_data, test_data = None, None, None  # ClamTrainer will random-split
        logger.info("Loaded %d feature bags (%d failed)", len(sample_data), n_failed)

    if not sample_data:
        error_msg = f"No usable feature files found (all {n_failed} samples failed to download)"
        logger.error(error_msg)
        report_training_finished(run_id, {}, error=error_msg)
        return

    # ── 2. Build config ───────────────────────────────────────────────────────
    # Auto-detect in_dim from first available feature file
    first_h5 = sample_data[0]["features_local_path"]
    try:
        import h5py
        with h5py.File(first_h5, "r") as f:
            key = "features" if "features" in f else "feats"
            in_dim = f[key].shape[-1]
    except Exception:
        in_dim = training_params.get("in_dim", 1024)

    cfg_dict = {**training_params, "in_dim": in_dim}

    # Derive n_classes from the labels actually present when Laravel did not
    # pin it — never train a head with fewer outputs than there are classes.
    max_label = max(int(s["label"]) for s in sample_data)
    if int(cfg_dict.get("n_classes", 0)) <= max_label:
        logger.warning(
            "n_classes=%s is too small for max label %d — raising to %d",
            cfg_dict.get("n_classes"), max_label, max_label + 1,
        )
        cfg_dict["n_classes"] = max_label + 1

    cfg = TrainingConfig.from_dict(cfg_dict)

    # ── Validate the hierarchy before it can silently misbehave ───────────────
    if cfg.n_parent_classes > 1:
        c2p = list(cfg.child_to_parent or [])
        if len(c2p) != cfg.n_classes:
            logger.error(
                "child_to_parent has %d entries but n_classes=%d — disabling hierarchical "
                "supervision for this run.", len(c2p), cfg.n_classes,
            )
            cfg.n_parent_classes = 0
            cfg.child_to_parent = []
            cfg.hier_weight = 0.0
        elif any(p >= cfg.n_parent_classes for p in c2p):
            logger.error(
                "child_to_parent references a parent index >= n_parent_classes=%d — "
                "disabling hierarchical supervision for this run.", cfg.n_parent_classes,
            )
            cfg.n_parent_classes = 0
            cfg.child_to_parent = []
            cfg.hier_weight = 0.0

    logger.info("TrainingConfig: %s", cfg.__dict__)

    # ── 3. Train ──────────────────────────────────────────────────────────────
    def _progress(run_id_, epoch_, total_, metrics_):
        report_training_progress(run_id_, epoch_, total_, metrics_)

    trainer = ClamTrainer(
        cfg,
        sample_data,
        run_id,
        progress_callback=_progress,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        label_map=label_map,
        parent_label_map=parent_label_map,
    )
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
    logger.info(
        "Training run_id=%d finished. best %s=%.4f (epoch %s) — balanced_acc=%.4f",
        run_id,
        final_metrics.get("monitor", "macro_auc"),
        final_metrics.get("best_monitor_value", 0.0),
        final_metrics.get("best_epoch", 0),
        final_metrics.get("best_val_balanced_acc", 0.0),
    )
