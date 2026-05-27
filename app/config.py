"""
Configuration for RunPod-CLAM server.
Port 8002 — CLAM MIL training head.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ─── Server identity ───────────────────────────────────────────────────────────

LARAVEL_BASE_URL: str = os.environ.get("LARAVEL_BASE_URL", "https://ai.histopathology.cloud")
LARAVEL_SERVER_ID: int = int(os.environ.get("LARAVEL_SERVER_ID", "4"))
API_KEY: str = os.environ.get("API_KEY", "")
PORT: int = int(os.environ.get("PORT", "8002"))
RUNPOD_POD_HOSTNAME: str = os.environ.get("RUNPOD_POD_HOSTNAME", "")

# ─── Filesystem paths ──────────────────────────────────────────────────────────

WORKSPACE: str = "/workspace"
FEATURES_DIR: str = os.path.join(WORKSPACE, "features")
MODELS_DIR: str = os.path.join(WORKSPACE, "RunPod-CLAM", "models")
LOGS_DIR: str = os.path.join(WORKSPACE, "logs")


@dataclass
class TrainingConfig:
    """Hyperparameters for a single CLAM training run."""

    # Model architecture
    model_type: str = "clam_sb"         # clam_sb | clam_mb
    in_dim: int = 1024                  # Feature embedding dimension
    n_classes: int = 2                  # Number of output classes
    hidden_dim: int = 256               # Attention hidden layer size
    dropout: float = 0.25

    # Training
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    bag_size: int = -1                  # -1 = use all patches per slide

    # Loss
    bag_weight: float = 0.7             # Weight for bag-level CE loss
    use_instance_loss: bool = True      # CLAM clustering loss

    # Misc
    seed: int = 42
    device: str = "cuda"                # cuda | cpu

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        valid = {k: v for k, v in d.items() if hasattr(cls, k)}
        return cls(**valid)
