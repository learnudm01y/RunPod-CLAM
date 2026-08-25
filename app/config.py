"""
Configuration for RunPod-CLAM server.
Port 8002 — CLAM MIL training head.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
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
    n_classes: int = 2                  # Number of FINE (leaf / exact disease) classes
    hidden_dim: int = 256               # Attention hidden layer size
    dropout: float = 0.25

    # Training
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    bag_size: int = -1                  # -1 = use all patches per slide
    grad_clip: float = 1.0              # 0 = disabled

    # Loss
    bag_weight: float = 0.7             # Weight for bag-level CE loss
    use_instance_loss: bool = True      # CLAM clustering loss
    label_smoothing: float = 0.0        # >0 helps when leaf classes are visually close

    # ── Hierarchical supervision (coarse → fine) ──────────────────────────────
    # n_parent_classes = 0 means a flat run and every hierarchy feature is off.
    n_parent_classes: int = 0
    hier_weight: float = 0.30           # Weight of the auxiliary coarse-level loss
    # Dense list of length n_classes: fine class index → coarse class index.
    # -1 marks a fine class with no known parent (excluded from the coarse loss).
    child_to_parent: list = field(default_factory=list)
    # At eval time, rescale the fine probabilities by the coarse head's belief.
    hierarchy_consistent_inference: bool = True

    # ── Class imbalance ───────────────────────────────────────────────────────
    use_class_weights: bool = True      # Inverse-frequency weighting of the bag loss
    class_weights: list = field(default_factory=list)   # computed from the train split

    # ── Model selection ───────────────────────────────────────────────────────
    # macro_auc works for any number of classes; the old code only ever produced
    # an AUC for binary runs and reported 0.0 otherwise, which made every
    # multi-class run select its LAST epoch instead of its best one.
    monitor: str = "macro_auc"          # macro_auc | balanced_acc | acc
    early_stopping_patience: int = 0    # 0 = train the full schedule

    # Misc
    seed: int = 42
    device: str = "cuda"                # cuda | cpu

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        # NOTE: dataclass fields declared with default_factory are not class
        # attributes, so the previous hasattr() filter silently dropped every
        # list-valued field (child_to_parent, class_weights).
        valid_names = {f.name for f in fields(cls)}
        valid = {k: v for k, v in d.items() if k in valid_names}
        return cls(**valid)

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def is_hierarchical(self) -> bool:
        return self.n_parent_classes > 1 and len(self.child_to_parent) == self.n_classes

    def parent_of(self, fine_idx: int) -> int:
        if not self.is_hierarchical:
            return -1
        if 0 <= fine_idx < len(self.child_to_parent):
            return int(self.child_to_parent[fine_idx])
        return -1
