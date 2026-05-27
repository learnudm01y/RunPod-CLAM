"""
I/O utilities — load pre-extracted features from HDF5 files.

Feature files are expected at:
  /workspace/features/{model_name}/{slide_id}.h5

Each HDF5 contains a dataset 'features' of shape [N, D].
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch import Tensor

from .config import FEATURES_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


def load_features_from_h5(h5_path: str | Path) -> Optional[Tensor]:
    """
    Load patch feature matrix from an HDF5 file.

    Returns a float32 Tensor of shape [N, D], or None if load fails.
    """
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "features" in f:
                arr = f["features"][:]
            elif "feats" in f:
                arr = f["feats"][:]
            else:
                keys = list(f.keys())
                raise KeyError(f"No 'features'/'feats' key found; available: {keys}")
        return torch.from_numpy(arr.astype(np.float32))
    except Exception as exc:
        logger.error("Failed to load features from %s: %s", h5_path, exc)
        return None


def sync_features_from_gdrive(gdrive_path: str, local_dir: str) -> bool:
    """
    Pull a single feature file from Google Drive using rclone.

    Args:
        gdrive_path: path inside GDrive, e.g. 'samples/features/TITAN/40x/sample_123.h5'
        local_dir:   local directory to place the file in

    Returns:
        True if file now exists locally.
    """
    os.makedirs(local_dir, exist_ok=True)
    filename = os.path.basename(gdrive_path)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        return True

    cmd = [
        "rclone", "copy",
        f"gdrive:{gdrive_path}",
        local_dir,
        "--progress",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            logger.error("rclone error: %s", result.stderr)
            return False
        return os.path.exists(local_path)
    except subprocess.TimeoutExpired:
        logger.error("rclone timed out syncing %s", gdrive_path)
        return False
    except FileNotFoundError:
        logger.error("rclone not found — install it on the pod")
        return False


def save_model_to_gdrive(local_model_path: str, gdrive_output_dir: str) -> bool:
    """Upload a local checkpoint file to Google Drive via rclone."""
    cmd = [
        "rclone", "copy",
        local_model_path,
        f"gdrive:{gdrive_output_dir}",
        "--progress",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error("rclone upload error: %s", result.stderr)
            return False
        return True
    except Exception as exc:
        logger.error("Failed to upload model: %s", exc)
        return False
