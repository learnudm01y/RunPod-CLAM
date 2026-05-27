"""
CLAM — Clustering-constrained Attention Multiple Instance Learning
Reference: Lu et al., Nature Biomedical Engineering 2021
GitHub:    https://github.com/mahmoodlab/CLAM

Implements:
  - CLAM_SB  (single-branch attention MIL)
  - CLAM_MB  (multi-branch attention MIL)
  - Training loop with bag-level CE + optional instance-level clustering loss
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import Tensor
from torch.optim import Adam

from .config import TrainingConfig, MODELS_DIR
from .io_utils import load_features_from_h5, sync_features_from_gdrive, save_model_to_gdrive

logger = logging.getLogger(__name__)


# ─── Attention Network ────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Gated attention mechanism."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.attn_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attn_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.attn_W = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        # features: [N, D]
        V = self.attn_V(features)           # [N, H]
        U = self.attn_U(features)           # [N, H]
        A_raw = self.attn_W(V * U)          # [N, 1]
        A = torch.softmax(A_raw, dim=0)     # [N, 1]  (sum-to-1 over patches)
        M = torch.mm(A.T, features)         # [1, D]  (attention-weighted slide embedding)
        return M, A


# ─── CLAM Single-Branch ───────────────────────────────────────────────────────

class CLAM_SB(nn.Module):
    """
    Single-branch CLAM.
    One attention branch → one pooled slide representation → classifier.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.attention = Attention(cfg.in_dim, cfg.hidden_dim, cfg.dropout)
        self.classifier = nn.Linear(cfg.in_dim, cfg.n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(cfg.in_dim, 2) for _ in range(cfg.n_classes)]
        )
        self.n_classes = cfg.n_classes
        self.use_instance_loss = cfg.use_instance_loss

    def forward(
        self,
        features: Tensor,
        label: Optional[Tensor] = None,
        instance_eval: bool = False,
    ) -> dict[str, Any]:
        M, A = self.attention(features)         # M: [1, D], A: [N, 1]
        logits = self.classifier(M)             # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        result = {"logits": logits, "Y_hat": Y_hat, "Y_prob": Y_prob, "A": A}

        if instance_eval and label is not None and self.use_instance_loss:
            inst_loss = self._instance_loss(features, A, label)
            result["instance_loss"] = inst_loss

        return result

    def _instance_loss(
        self, features: Tensor, A: Tensor, label: Tensor
    ) -> Tensor:
        """
        CLAM instance-level clustering loss.
        Top-k and bottom-k patches are pseudo-labelled for the target class.
        """
        k = min(8, max(1, features.size(0) // 8))
        cls_idx = label.item()
        classifier = self.instance_classifiers[cls_idx]

        # Positive instances: top-k attention patches → pseudo label 1
        # Negative instances: bottom-k attention patches → pseudo label 0
        _, top_idx = torch.topk(A.squeeze(), k)
        _, bot_idx = torch.topk(-A.squeeze(), k)

        pos_feats = features[top_idx]
        neg_feats = features[bot_idx]

        pos_logits = classifier(pos_feats)
        neg_logits = classifier(neg_feats)

        pos_labels = torch.ones(k, dtype=torch.long, device=features.device)
        neg_labels = torch.zeros(k, dtype=torch.long, device=features.device)

        loss = (
            F.cross_entropy(pos_logits, pos_labels)
            + F.cross_entropy(neg_logits, neg_labels)
        ) / 2.0
        return loss


# ─── CLAM Multi-Branch ────────────────────────────────────────────────────────

class CLAM_MB(nn.Module):
    """
    Multi-branch CLAM.
    One dedicated attention branch per class → per-class slide embedding → classifier.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.attention_branches = nn.ModuleList([
            Attention(cfg.in_dim, cfg.hidden_dim, cfg.dropout)
            for _ in range(cfg.n_classes)
        ])
        self.classifiers = nn.ModuleList([
            nn.Linear(cfg.in_dim, 1)
            for _ in range(cfg.n_classes)
        ])
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(cfg.in_dim, 2) for _ in range(cfg.n_classes)]
        )
        self.n_classes = cfg.n_classes
        self.use_instance_loss = cfg.use_instance_loss

    def forward(
        self,
        features: Tensor,
        label: Optional[Tensor] = None,
        instance_eval: bool = False,
    ) -> dict[str, Any]:
        logits = []
        A_all = []
        for i in range(self.n_classes):
            M_i, A_i = self.attention_branches[i](features)
            logit_i = self.classifiers[i](M_i)   # [1, 1]
            logits.append(logit_i)
            A_all.append(A_i)

        logits = torch.cat(logits, dim=1)          # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)

        result = {"logits": logits, "Y_hat": Y_hat, "Y_prob": Y_prob, "A": A_all}

        if instance_eval and label is not None and self.use_instance_loss:
            inst_loss = self._instance_loss(features, A_all, label)
            result["instance_loss"] = inst_loss

        return result

    def _instance_loss(
        self, features: Tensor, A_all: list[Tensor], label: Tensor
    ) -> Tensor:
        k = min(8, max(1, features.size(0) // 8))
        cls_idx = label.item()
        classifier = self.instance_classifiers[cls_idx]
        A = A_all[cls_idx]

        _, top_idx = torch.topk(A.squeeze(), k)
        _, bot_idx = torch.topk(-A.squeeze(), k)

        pos_logits = classifier(features[top_idx])
        neg_logits = classifier(features[bot_idx])
        pos_labels = torch.ones(k, dtype=torch.long, device=features.device)
        neg_labels = torch.zeros(k, dtype=torch.long, device=features.device)

        return (
            F.cross_entropy(pos_logits, pos_labels)
            + F.cross_entropy(neg_logits, neg_labels)
        ) / 2.0


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: TrainingConfig) -> nn.Module:
    if cfg.model_type == "clam_mb":
        return CLAM_MB(cfg)
    return CLAM_SB(cfg)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BagDataset(torch.utils.data.Dataset):
    """
    Dataset of pre-extracted feature bags.

    Args:
        sample_data: list of dicts:
            {
                "sample_id": int,
                "label": int,
                "features_local_path": str,   # local HDF5 path
            }
        bag_size: max patches per bag (-1 = no limit)
        device: 'cuda' | 'cpu'
    """

    def __init__(self, sample_data: list[dict], bag_size: int = -1, device: str = "cpu"):
        self.samples = sample_data
        self.bag_size = bag_size
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, int]:
        item = self.samples[idx]
        feats = load_features_from_h5(item["features_local_path"])
        if feats is None:
            # Return empty bag — will be skipped in training loop
            feats = torch.zeros(1, 1)
        if self.bag_size > 0 and feats.size(0) > self.bag_size:
            perm = torch.randperm(feats.size(0))[: self.bag_size]
            feats = feats[perm]
        label = torch.tensor(item["label"], dtype=torch.long)
        return feats, label, item["sample_id"]


# ─── Training Loop ────────────────────────────────────────────────────────────

class ClamTrainer:
    """Orchestrates the complete CLAM training run."""

    def __init__(
        self,
        cfg: TrainingConfig,
        sample_data: list[dict],
        run_id: int,
        progress_callback=None,
    ):
        self.cfg = cfg
        self.run_id = run_id
        self.progress_callback = progress_callback  # callable(run_id, epoch, total, metrics)

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        logger.info("CLAM trainer: device=%s  run_id=%s", self.device, run_id)

        # 80/20 split
        n = len(sample_data)
        train_n = max(1, int(n * 0.8))
        random.shuffle(sample_data)
        self.train_data = sample_data[:train_n]
        self.val_data = sample_data[train_n:]

        self.model = build_model(cfg).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        best_val_auc = 0.0
        best_epoch = 0
        history = []

        os.makedirs(MODELS_DIR, exist_ok=True)
        ckpt_path = os.path.join(MODELS_DIR, f"run_{self.run_id}_best.pt")

        for epoch in range(1, self.cfg.epochs + 1):
            train_metrics = self._run_epoch(self.train_data, train=True)
            val_metrics = self._run_epoch(self.val_data, train=False)

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 4),
                "train_acc": round(train_metrics["acc"], 4),
                "val_loss": round(val_metrics["loss"], 4),
                "val_acc": round(val_metrics["acc"], 4),
                "val_auc": round(val_metrics.get("auc", 0.0), 4),
            }
            history.append(epoch_metrics)
            logger.info("Epoch %d/%d — %s", epoch, self.cfg.epochs, epoch_metrics)

            if epoch_metrics["val_auc"] >= best_val_auc:
                best_val_auc = epoch_metrics["val_auc"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.cfg.__dict__,
                        "metrics": epoch_metrics,
                    },
                    ckpt_path,
                )

            if self.progress_callback:
                self.progress_callback(self.run_id, epoch, self.cfg.epochs, epoch_metrics)

        final_metrics = {
            "best_val_auc": round(best_val_auc, 4),
            "best_epoch": best_epoch,
            "history": history,
            "total_epochs": self.cfg.epochs,
            "n_train": len(self.train_data),
            "n_val": len(self.val_data),
        }
        return ckpt_path, final_metrics

    # ------------------------------------------------------------------

    def _run_epoch(self, split_data: list[dict], train: bool) -> dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        n_skipped = 0

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for item in split_data:
                feats = load_features_from_h5(item["features_local_path"])
                if feats is None or feats.size(0) == 0:
                    n_skipped += 1
                    continue

                feats = feats.to(self.device)
                label = torch.tensor(item["label"], dtype=torch.long, device=self.device)

                out = self.model(feats, label=label, instance_eval=train)
                logits = out["logits"]

                bag_loss = F.cross_entropy(logits, label.unsqueeze(0))
                inst_loss = out.get("instance_loss", torch.tensor(0.0, device=self.device))
                loss = self.cfg.bag_weight * bag_loss + (1 - self.cfg.bag_weight) * inst_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                all_labels.append(item["label"])
                all_preds.append(out["Y_hat"].item())
                if self.cfg.n_classes == 2:
                    all_probs.append(out["Y_prob"][0, 1].item())

        n = max(1, len(split_data) - n_skipped)
        metrics = {
            "loss": total_loss / n,
            "acc": accuracy_score(all_labels, all_preds) if all_labels else 0.0,
        }
        if self.cfg.n_classes == 2 and len(set(all_labels)) == 2 and all_probs:
            try:
                metrics["auc"] = roc_auc_score(all_labels, all_probs)
            except Exception:
                metrics["auc"] = 0.0
        else:
            metrics["auc"] = 0.0

        return metrics
