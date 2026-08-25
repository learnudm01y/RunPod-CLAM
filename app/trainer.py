"""
CLAM — Clustering-constrained Attention Multiple Instance Learning
Reference: Lu et al., Nature Biomedical Engineering 2021
GitHub:    https://github.com/mahmoodlab/CLAM

Implements:
  - CLAM_SB  (single-branch attention MIL)
  - CLAM_MB  (multi-branch attention MIL)
  - Training loop with bag-level CE + optional instance-level clustering loss
  - Optional HIERARCHICAL supervision: an auxiliary coarse head (disease family)
    trained jointly with the fine head (exact disease name), plus
    hierarchy-consistent decoding at evaluation time.

Label semantics
---------------
Every bag carries exactly one fine label (the exact disease entity). When the
run is hierarchical it also carries one coarse label (its family / category).
This is single-label classification at two granularities — NOT multi-label.
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
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

    When cfg.is_hierarchical, a second linear head predicts the coarse class
    from the same pooled embedding. Sharing the embedding is what makes the
    coarse signal act as a regulariser for the fine head: the family-level
    decision is easy and well-populated, and it shapes the representation the
    hard leaf-level decision has to use.
    """

    def __init__(self, cfg: TrainingConfig):
        super().__init__()
        self.attention = Attention(cfg.in_dim, cfg.hidden_dim, cfg.dropout)
        self.classifier = nn.Linear(cfg.in_dim, cfg.n_classes)
        self.parent_classifier = (
            nn.Linear(cfg.in_dim, cfg.n_parent_classes) if cfg.is_hierarchical else None
        )
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

        result = {"logits": logits, "A": A, "M": M}

        if self.parent_classifier is not None:
            result["parent_logits"] = self.parent_classifier(M)   # [1, n_parent_classes]

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
        cls_idx = int(label.item())
        classifier = self.instance_classifiers[cls_idx]

        # Positive instances: top-k attention patches → pseudo label 1
        # Negative instances: bottom-k attention patches → pseudo label 0
        _, top_idx = torch.topk(A.squeeze(-1), k)
        _, bot_idx = torch.topk(-A.squeeze(-1), k)

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

    For fine-grained disease naming this is usually the stronger head: each
    entity gets to attend to its own morphology instead of sharing one
    attention map across every diagnosis.
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
        self.parent_classifier = (
            nn.Linear(cfg.in_dim, cfg.n_parent_classes) if cfg.is_hierarchical else None
        )
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
        M_all = []
        for i in range(self.n_classes):
            M_i, A_i = self.attention_branches[i](features)
            logit_i = self.classifiers[i](M_i)   # [1, 1]
            logits.append(logit_i)
            A_all.append(A_i)
            M_all.append(M_i)

        logits = torch.cat(logits, dim=1)          # [1, n_classes]

        result = {"logits": logits, "A": A_all}

        if self.parent_classifier is not None:
            # Mean over the per-class embeddings — a slide-level summary that
            # every branch contributes to.
            M_mean = torch.stack(M_all, dim=0).mean(dim=0)   # [1, D]
            result["M"] = M_mean
            result["parent_logits"] = self.parent_classifier(M_mean)

        if instance_eval and label is not None and self.use_instance_loss:
            inst_loss = self._instance_loss(features, A_all, label)
            result["instance_loss"] = inst_loss

        return result

    def _instance_loss(
        self, features: Tensor, A_all: list[Tensor], label: Tensor
    ) -> Tensor:
        k = min(8, max(1, features.size(0) // 8))
        cls_idx = int(label.item())
        classifier = self.instance_classifiers[cls_idx]
        A = A_all[cls_idx]

        _, top_idx = torch.topk(A.squeeze(-1), k)
        _, bot_idx = torch.topk(-A.squeeze(-1), k)

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
                "label": int,               # fine (exact disease) class index
                "parent_label": int,        # coarse class index, -1 when unknown
                "features_local_path": str, # local HDF5 path
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, int]:
        item = self.samples[idx]
        feats = load_features_from_h5(item["features_local_path"])
        if feats is None:
            # Return empty bag — will be skipped in training loop
            feats = torch.zeros(1, 1)
        if self.bag_size > 0 and feats.size(0) > self.bag_size:
            perm = torch.randperm(feats.size(0))[: self.bag_size]
            feats = feats[perm]
        label = torch.tensor(item["label"], dtype=torch.long)
        parent = torch.tensor(item.get("parent_label", -1), dtype=torch.long)
        return feats, label, parent, item["sample_id"]


# ─── Metric helpers ───────────────────────────────────────────────────────────

def macro_ovr_auc(
    y_true: list[int], probs: np.ndarray, n_classes: int
) -> tuple[float, dict[int, float]]:
    """
    One-vs-rest AUC averaged over every class that is actually evaluable.

    sklearn's multi_class='ovr' raises as soon as one class is absent from
    y_true, which happens constantly on small pathology validation splits.
    Computing it per class and skipping the degenerate ones keeps the metric
    defined (and comparable across epochs) instead of collapsing it to 0.0.
    """
    per_class: dict[int, float] = {}
    y_arr = np.asarray(y_true)

    for c in range(n_classes):
        y_bin = (y_arr == c).astype(int)
        if y_bin.min() == y_bin.max():      # class absent, or the only class present
            continue
        try:
            per_class[c] = float(roc_auc_score(y_bin, probs[:, c]))
        except Exception:
            continue

    macro = float(np.mean(list(per_class.values()))) if per_class else 0.0
    return macro, per_class


def hierarchy_consistent_probs(
    fine_probs: np.ndarray,
    parent_probs: Optional[np.ndarray],
    child_to_parent: list,
) -> np.ndarray:
    """
    Rescale each fine probability by how much the coarse head believes in that
    fine class's family, then renormalise.

        p'(leaf) ∝ p(leaf) · p(family(leaf))

    A leaf whose family the coarse head has ruled out is suppressed even when
    the fine head liked its morphology — which is exactly the failure mode of a
    flat fine-grained classifier on a small dataset.
    """
    if parent_probs is None or not child_to_parent:
        return fine_probs

    n_classes = fine_probs.shape[1]
    factor = np.ones_like(fine_probs)

    for c in range(n_classes):
        p = int(child_to_parent[c]) if c < len(child_to_parent) else -1
        if 0 <= p < parent_probs.shape[1]:
            factor[:, c] = parent_probs[:, p]

    adjusted = fine_probs * factor
    row_sum = adjusted.sum(axis=1, keepdims=True)
    # Fall back to the raw fine probabilities where the product underflowed.
    safe = row_sum.squeeze(-1) > 1e-12
    out = fine_probs.copy()
    out[safe] = adjusted[safe] / row_sum[safe]
    return out


# ─── Training Loop ────────────────────────────────────────────────────────────

class ClamTrainer:
    """Orchestrates the complete CLAM training run."""

    def __init__(
        self,
        cfg: TrainingConfig,
        sample_data: list[dict],
        run_id: int,
        progress_callback=None,
        # Explicit splits provided by Laravel (preferred).
        # If None, a random 80/20 train/val split is applied as fallback.
        train_data: Optional[list[dict]] = None,
        val_data:   Optional[list[dict]] = None,
        test_data:  Optional[list[dict]] = None,
        # Human-readable class names, stored in the checkpoint for inference.
        label_map: Optional[list[str]] = None,
        parent_label_map: Optional[list[str]] = None,
    ):
        self.cfg = cfg
        self.run_id = run_id
        self.progress_callback = progress_callback  # callable(run_id, epoch, total, metrics)
        self.label_map = label_map or []
        self.parent_label_map = parent_label_map or []

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        logger.info("CLAM trainer: device=%s  run_id=%s", self.device, run_id)

        if train_data is not None:
            # ── Explicit split from caller ─────────────────────────────────────
            self.train_data = train_data
            self.val_data   = val_data or []
            self.test_data  = test_data or []
            logger.info(
                "Using explicit split — train=%d  val=%d  test=%d",
                len(self.train_data), len(self.val_data), len(self.test_data),
            )
        else:
            # ── Fallback: random 80/20 train/val split (no test set) ──────────
            logger.warning(
                "No explicit split provided for run_id=%d — applying random 80/20 split", run_id
            )
            n = len(sample_data)
            train_n = max(1, int(n * 0.8))
            random.shuffle(sample_data)
            self.train_data = sample_data[:train_n]
            self.val_data   = sample_data[train_n:]
            self.test_data  = []

        if cfg.is_hierarchical:
            logger.info(
                "Hierarchical run — %d fine classes over %d coarse classes  "
                "hier_weight=%.3f  consistent_inference=%s",
                cfg.n_classes, cfg.n_parent_classes,
                cfg.hier_weight, cfg.hierarchy_consistent_inference,
            )
        else:
            logger.info("Flat run — %d classes", cfg.n_classes)

        # ── Class weights from the TRAIN split only ───────────────────────────
        self.class_weight_tensor = self._compute_class_weights()

        self.model = build_model(cfg).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------

    def _compute_class_weights(self) -> Optional[Tensor]:
        """
        Inverse-frequency weights, normalised to mean 1 so the loss scale — and
        therefore the usable learning rate — does not change with the number of
        classes. Fine-grained disease sets are always long-tailed; without this
        the head collapses onto the two or three most common entities.
        """
        if not self.cfg.use_class_weights:
            return None

        if self.cfg.class_weights and len(self.cfg.class_weights) == self.cfg.n_classes:
            weights = np.asarray(self.cfg.class_weights, dtype=np.float32)
        else:
            counts = np.zeros(self.cfg.n_classes, dtype=np.float64)
            for item in self.train_data:
                c = int(item["label"])
                if 0 <= c < self.cfg.n_classes:
                    counts[c] += 1
            if counts.sum() == 0:
                return None
            # Classes absent from train get weight 0 rather than +inf.
            weights = np.zeros(self.cfg.n_classes, dtype=np.float32)
            present = counts > 0
            weights[present] = (counts[present].sum() / (present.sum() * counts[present])).astype(np.float32)
            if weights[present].mean() > 0:
                weights[present] = weights[present] / weights[present].mean()

        self.cfg.class_weights = [float(w) for w in weights]
        logger.info(
            "Class weights: %s",
            {i: round(float(w), 3) for i, w in enumerate(weights)},
        )
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------

    def _monitor_value(self, metrics: dict[str, Any]) -> float:
        key = {
            "macro_auc": "macro_auc",
            "balanced_acc": "balanced_acc",
            "acc": "acc",
        }.get(self.cfg.monitor, "macro_auc")
        value = metrics.get(key)
        if value is None or (key == "macro_auc" and value == 0.0):
            # Undefined AUC (e.g. a single class present in val) — fall back to
            # balanced accuracy so model selection never degenerates into
            # "keep the last epoch", which is what the old `>=` on a constant
            # 0.0 AUC effectively did.
            return float(metrics.get("balanced_acc", metrics.get("acc", 0.0)))
        return float(value)

    # ------------------------------------------------------------------

    def train(self) -> tuple[str, dict[str, Any]]:
        best_score = -1.0
        best_epoch = 0
        best_metrics: dict[str, Any] = {}
        history = []
        epochs_without_improvement = 0

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
                "val_balanced_acc": round(val_metrics["balanced_acc"], 4),
                "val_macro_auc": round(val_metrics["macro_auc"], 4),
                # Kept so existing dashboards that read `val_auc` keep working.
                "val_auc": round(val_metrics["macro_auc"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
            }
            if self.cfg.is_hierarchical:
                epoch_metrics["val_parent_acc"] = round(val_metrics.get("parent_acc", 0.0), 4)
                epoch_metrics["val_coarse_head_acc"] = round(val_metrics.get("coarse_head_acc", 0.0), 4)

            history.append(epoch_metrics)
            logger.info("Epoch %d/%d — %s", epoch, self.cfg.epochs, epoch_metrics)

            score = self._monitor_value(val_metrics)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = val_metrics
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.cfg.__dict__,
                        "label_map": self.label_map,
                        "parent_label_map": self.parent_label_map,
                        "child_to_parent": list(self.cfg.child_to_parent),
                        "metrics": epoch_metrics,
                        "monitor": self.cfg.monitor,
                        "monitor_value": score,
                    },
                    ckpt_path,
                )
                logger.info("New best checkpoint at epoch %d (%s=%.4f)", epoch, self.cfg.monitor, score)
            else:
                epochs_without_improvement += 1

            if self.progress_callback:
                self.progress_callback(self.run_id, epoch, self.cfg.epochs, epoch_metrics)

            if (
                self.cfg.early_stopping_patience > 0
                and epochs_without_improvement >= self.cfg.early_stopping_patience
            ):
                logger.info(
                    "Early stopping at epoch %d — no improvement for %d epochs",
                    epoch, epochs_without_improvement,
                )
                break

        # ── Final test-set evaluation using best checkpoint ──────────────────
        test_metrics: dict = {}
        if self.test_data and os.path.exists(ckpt_path):
            logger.info("Loading best checkpoint (epoch %d) for test-set evaluation …", best_epoch)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            raw_test = self._run_epoch(self.test_data, train=False)
            test_metrics = {
                "test_loss":         round(raw_test["loss"], 4),
                "test_acc":          round(raw_test["acc"], 4),
                "test_balanced_acc": round(raw_test["balanced_acc"], 4),
                "test_macro_auc":    round(raw_test["macro_auc"], 4),
                "test_auc":          round(raw_test["macro_auc"], 4),   # legacy key
                "test_macro_f1":     round(raw_test["macro_f1"], 4),
                "test_per_class":    raw_test["per_class"],
                "test_confusion_matrix": raw_test["confusion_matrix"],
            }
            if self.cfg.is_hierarchical:
                test_metrics["test_parent_acc"] = round(raw_test.get("parent_acc", 0.0), 4)
                test_metrics["test_coarse_head_acc"] = round(raw_test.get("coarse_head_acc", 0.0), 4)
            logger.info("Test-set results: acc=%.4f balanced_acc=%.4f macro_auc=%.4f",
                        raw_test["acc"], raw_test["balanced_acc"], raw_test["macro_auc"])
        elif self.test_data:
            logger.warning("No checkpoint written — skipping test evaluation.")
        else:
            logger.info("No test set provided — skipping final test evaluation.")

        final_metrics = {
            "best_epoch": best_epoch,
            "monitor": self.cfg.monitor,
            "best_monitor_value": round(float(best_score), 4) if best_score >= 0 else 0.0,
            "best_val_macro_auc": round(float(best_metrics.get("macro_auc", 0.0)), 4),
            "best_val_balanced_acc": round(float(best_metrics.get("balanced_acc", 0.0)), 4),
            "best_val_acc": round(float(best_metrics.get("acc", 0.0)), 4),
            "best_val_macro_f1": round(float(best_metrics.get("macro_f1", 0.0)), 4),
            # Legacy key kept for the existing dashboard/accessor.
            "best_val_auc": round(float(best_metrics.get("macro_auc", 0.0)), 4),
            "val_per_class": best_metrics.get("per_class", {}),
            "val_confusion_matrix": best_metrics.get("confusion_matrix", []),
            "history": history,
            "total_epochs": len(history),
            "planned_epochs": self.cfg.epochs,
            "n_train": len(self.train_data),
            "n_val": len(self.val_data),
            "n_test": len(self.test_data),
            "n_classes": self.cfg.n_classes,
            "label_map": self.label_map,
            "hierarchical": self.cfg.is_hierarchical,
            "class_weights": self.cfg.class_weights,
            **test_metrics,   # test_* (if a test set exists)
        }
        if self.cfg.is_hierarchical:
            final_metrics["n_parent_classes"] = self.cfg.n_parent_classes
            final_metrics["parent_label_map"] = self.parent_label_map
            final_metrics["best_val_parent_acc"] = round(float(best_metrics.get("parent_acc", 0.0)), 4)

        return ckpt_path, final_metrics

    # ------------------------------------------------------------------

    def _subsample(self, feats: Tensor, train: bool) -> Tensor:
        """
        Cap the bag at cfg.bag_size patches.

        Applied on the training path only: a random subset each epoch acts as
        augmentation and bounds GPU memory on 100k-patch slides, while
        evaluation always sees the complete bag so val/test numbers are
        deterministic and comparable across epochs.

        (This is what `bag_size` was always meant to do — it previously lived
        only on BagDataset, which the training loop never used.)
        """
        if train and self.cfg.bag_size > 0 and feats.size(0) > self.cfg.bag_size:
            perm = torch.randperm(feats.size(0), device=feats.device)[: self.cfg.bag_size]
            return feats[perm]
        return feats

    # ------------------------------------------------------------------

    def _run_epoch(self, split_data: list[dict], train: bool) -> dict[str, Any]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        all_labels: list[int] = []
        all_parent_labels: list[int] = []
        all_fine_probs: list[np.ndarray] = []
        all_parent_probs: list[np.ndarray] = []
        n_skipped = 0

        order = list(range(len(split_data)))
        if train:
            random.shuffle(order)   # bag order must not be a learnable signal

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            for i in order:
                item = split_data[i]
                feats = load_features_from_h5(item["features_local_path"])
                if feats is None or feats.size(0) == 0:
                    n_skipped += 1
                    continue

                feats = feats.to(self.device)
                feats = self._subsample(feats, train)

                label = torch.tensor(item["label"], dtype=torch.long, device=self.device)
                parent_idx = int(item.get("parent_label", -1))

                out = self.model(feats, label=label, instance_eval=train)
                logits = out["logits"]

                bag_loss = F.cross_entropy(
                    logits,
                    label.unsqueeze(0),
                    weight=self.class_weight_tensor,
                    label_smoothing=self.cfg.label_smoothing,
                )
                inst_loss = out.get("instance_loss", torch.tensor(0.0, device=self.device))

                loss = self.cfg.bag_weight * bag_loss + (1 - self.cfg.bag_weight) * inst_loss

                # ── Auxiliary coarse (family) supervision ──────────────────
                parent_logits = out.get("parent_logits")
                if parent_logits is not None and parent_idx >= 0:
                    parent_target = torch.tensor(
                        parent_idx, dtype=torch.long, device=self.device
                    )
                    parent_loss = F.cross_entropy(parent_logits, parent_target.unsqueeze(0))
                    loss = loss + self.cfg.hier_weight * parent_loss

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.cfg.grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                total_loss += float(loss.item())
                all_labels.append(int(item["label"]))
                all_parent_labels.append(parent_idx)
                all_fine_probs.append(
                    F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                )
                if parent_logits is not None:
                    all_parent_probs.append(
                        F.softmax(parent_logits, dim=1).detach().cpu().numpy()[0]
                    )

        n = max(1, len(split_data) - n_skipped)
        if not all_labels:
            return {
                "loss": total_loss / n, "acc": 0.0, "balanced_acc": 0.0,
                "macro_auc": 0.0, "macro_f1": 0.0, "auc": 0.0,
                "per_class": {}, "confusion_matrix": [], "n_skipped": n_skipped,
            }

        fine_probs = np.vstack(all_fine_probs)
        parent_probs = np.vstack(all_parent_probs) if all_parent_probs else None

        # Hierarchy-consistent decoding — evaluation only. Training keeps the
        # raw fine head so the two heads stay independently supervised.
        decode_probs = fine_probs
        if (
            not train
            and self.cfg.is_hierarchical
            and self.cfg.hierarchy_consistent_inference
            and parent_probs is not None
        ):
            decode_probs = hierarchy_consistent_probs(
                fine_probs, parent_probs, self.cfg.child_to_parent
            )

        preds = decode_probs.argmax(axis=1).tolist()

        macro_auc, per_class_auc = macro_ovr_auc(all_labels, decode_probs, self.cfg.n_classes)

        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, preds,
            labels=list(range(self.cfg.n_classes)),
            zero_division=0,
        )

        per_class = {}
        for c in range(self.cfg.n_classes):
            name = self.label_map[c] if c < len(self.label_map) else f"class_{c}"
            per_class[str(c)] = {
                "label":     name,
                "precision": round(float(precision[c]), 4),
                "recall":    round(float(recall[c]), 4),
                "f1":        round(float(f1[c]), 4),
                "support":   int(support[c]),
                "auc":       round(per_class_auc[c], 4) if c in per_class_auc else None,
            }

        metrics: dict[str, Any] = {
            "loss": total_loss / n,
            "acc": float(accuracy_score(all_labels, preds)),
            "balanced_acc": float(balanced_accuracy_score(all_labels, preds)),
            "macro_auc": macro_auc,
            "auc": macro_auc,                       # legacy alias
            "macro_f1": float(np.mean(f1)),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(
                all_labels, preds, labels=list(range(self.cfg.n_classes))
            ).tolist(),
            "n_skipped": n_skipped,
        }

        # ── Hierarchical diagnostics ──────────────────────────────────────────
        if self.cfg.is_hierarchical:
            c2p = self.cfg.child_to_parent
            true_parents, pred_parents = [], []
            for true_c, pred_c, pl in zip(all_labels, preds, all_parent_labels):
                tp = pl if pl >= 0 else (int(c2p[true_c]) if true_c < len(c2p) else -1)
                pp = int(c2p[pred_c]) if pred_c < len(c2p) else -1
                if tp >= 0 and pp >= 0:
                    true_parents.append(tp)
                    pred_parents.append(pp)
            # How often the predicted disease is at least in the right family —
            # a wrong leaf inside the right family is a much cheaper error
            # clinically than a wrong family.
            metrics["parent_acc"] = (
                float(accuracy_score(true_parents, pred_parents)) if true_parents else 0.0
            )

            if parent_probs is not None:
                coarse_preds = parent_probs.argmax(axis=1).tolist()
                pairs = [(t, p) for t, p in zip(all_parent_labels, coarse_preds) if t >= 0]
                metrics["coarse_head_acc"] = (
                    float(accuracy_score([t for t, _ in pairs], [p for _, p in pairs]))
                    if pairs else 0.0
                )

        return metrics
