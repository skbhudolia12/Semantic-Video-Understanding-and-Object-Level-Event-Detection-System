"""
NeuralComplianceChecker
========================
A lightweight online-learning MLP that replaces the hard-coded
``detect_helmet_violations`` logic in ``compliance.py``.

Motivation
----------
The original rule:  Person + NO Helmet → Violation
Works for one domain but fails to generalise (e.g. "Person without safety
vest", "Forklift without spotter", "Machine door open during operation").
A learned checker can discover these patterns from detection history without
requiring a new hand-crafted rule per scenario.

Architecture
------------
  Linear(F) → LayerNorm → ReLU → Dropout(0.2)
  → Linear(64) → LayerNorm → ReLU
  → Linear(1) → Sigmoid

  F = number of input features (computed from class vocabulary size).

Input features (per frame)
  1. Multi-hot class presence vector   — which labels appear in this frame
  2. Pairwise-IoU proximity features   — interaction graph (flattened, max 5×5)
  3. Rolling temporal context          — class presence in last 5 frames

Training
--------
  Cold start: pseudo-labels are generated from the existing rule-based
  ``check_compliance`` function.  The MLP trains on these labels in a
  background pass.  Once ``n_samples >= cold_start_threshold`` the MLP
  predictions are used; until then the rule-based checker is the fallback.

  ``fit_batch(X, y)`` is called after every video to fine-tune.

  Training data (feature matrix + labels) is saved to disk so subsequent
  runs warm-start from prior knowledge.

Persistence
-----------
  Checkpoint path: ``models/neural_compliance_checkpoint.pt``
  Training data:   ``models/neural_compliance_data.npz``
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE = Path(__file__).parent.parent.parent / "models"
_CHECKPOINT_PATH = _BASE / "neural_compliance_checkpoint.pt"
_DATA_PATH = _BASE / "neural_compliance_data.npz"

# COCO-80 + custom classes; extend as needed
_DEFAULT_VOCAB = [
    "person", "helmet", "vest", "gloves", "boots",
    "bicycle", "car", "truck", "forklift", "motorcycle",
    "fire extinguisher", "cone", "barrier", "ladder", "scaffold",
    "bottle", "bag", "box", "machine", "door",
    # Fallback — index 20 reserved for any unknown class
    "__other__",
]


# ---------------------------------------------------------------------------
# MLP Model
# ---------------------------------------------------------------------------
class _ComplianceMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F)
        return self.net(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NeuralComplianceChecker:
    """
    Online-learning MLP compliance checker.

    Parameters
    ----------
    vocab : List[str]
        Class vocabulary.  Defines the size of the multi-hot feature vector.
    cold_start_threshold : int
        Minimum labelled samples required before switching from rule-based
        to MLP predictions.
    temporal_window : int
        Number of past frames to include in the rolling context vector.
    lr : float
        SGD learning rate for online fine-tuning.
    checkpoint_path : Path | None
        Where to save/load model weights.
    data_path : Path | None
        Where to save/load accumulated (X, y) training data for warm-start.
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        cold_start_threshold: int = 50,
        temporal_window: int = 5,
        lr: float = 1e-3,
        checkpoint_path: Optional[Path] = None,
        data_path: Optional[Path] = None,
    ):
        self._vocab = vocab or _DEFAULT_VOCAB
        self._vocab_idx = {label: i for i, label in enumerate(self._vocab)}
        self._V = len(self._vocab)

        self._temporal_window = temporal_window
        self._cold_start_threshold = cold_start_threshold

        # Max objects considered in pairwise IoU feature
        self._max_obj = 5
        self._pairwise_dim = self._max_obj * self._max_obj

        self._input_dim = (
            self._V                     # multi-hot class presence
            + self._pairwise_dim        # pairwise IoU
            + self._V                   # rolling temporal context
        )

        self._model = _ComplianceMLP(self._input_dim)
        self._optimizer = optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.BCELoss()

        # Training data accumulation
        self._X_buf: List[np.ndarray] = []
        self._y_buf: List[float] = []

        # Rolling frame context
        self._frame_context: List[np.ndarray] = []  # circular buffer of multi-hot vecs

        # Persistence
        self._checkpoint_path = checkpoint_path or _CHECKPOINT_PATH
        self._data_path = data_path or _DATA_PATH
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _multi_hot(self, detections: List[dict]) -> np.ndarray:
        """Build a multi-hot vector for the classes present in a frame."""
        vec = np.zeros(self._V, dtype=np.float32)
        for det in detections:
            label = det.get("label", "__other__").lower()
            idx = self._vocab_idx.get(label, self._vocab_idx.get("__other__", self._V - 1))
            vec[idx] = 1.0
        return vec

    def _pairwise_iou_features(self, detections: List[dict]) -> np.ndarray:
        """
        Build a flattened max_obj×max_obj IoU matrix.
        Captures spatial interactions between up to ``_max_obj`` objects.
        This lets the model learn "person close to helmet" without explicit coding.
        """
        M = self._max_obj
        mat = np.zeros((M, M), dtype=np.float32)
        boxes = [det["bbox"] for det in detections[:M]]
        for i, b1 in enumerate(boxes):
            for j, b2 in enumerate(boxes):
                if i != j:
                    mat[i, j] = self._iou(b1, b2)
        return mat.flatten()  # (M*M,)

    @staticmethod
    def _iou(a: list, b: list) -> float:
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-6)

    def _temporal_context(self) -> np.ndarray:
        """Aggregate rolling class presence over the last N frames (OR logic)."""
        if not self._frame_context:
            return np.zeros(self._V, dtype=np.float32)
        stack = np.stack(self._frame_context[-self._temporal_window:])
        return stack.max(axis=0)  # (V,) — 1 if seen in any recent frame

    def extract_features(self, detections: List[dict]) -> np.ndarray:
        """Extract the full flat feature vector for one frame."""
        mh = self._multi_hot(detections)

        # Update rolling context
        self._frame_context.append(mh.copy())
        if len(self._frame_context) > self._temporal_window:
            self._frame_context.pop(0)

        pw = self._pairwise_iou_features(detections)
        tc = self._temporal_context()
        return np.concatenate([mh, pw, tc])  # (F,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of labelled training samples accumulated so far."""
        return len(self._y_buf)

    def predict(
        self, detections: List[dict]
    ) -> Tuple[bool, float]:
        """
        Predict whether the current frame constitutes a compliance violation.

        Returns
        -------
        (is_violation, confidence)
            ``is_violation`` — True if the MLP (or rule baseline) flags a
            violation.
            ``confidence`` — probability in [0, 1]; rule-based returns 1.0
            when triggered.
        """
        # Cold-start: fall back to rule-based check
        if self.n_samples < self._cold_start_threshold:
            rb_violation = self._rule_based_check(detections)
            # Accumulate a pseudo-labelled sample for future training
            features = self.extract_features(detections)
            self._add_sample(features, float(rb_violation))
            return rb_violation, 1.0 if rb_violation else 0.0

        # MLP prediction
        features = self.extract_features(detections)
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            prob = float(self._model(x).item())

        # Continue accumulating with pseudo-label from rule baseline
        rb_label = float(self._rule_based_check(detections))
        self._add_sample(features, rb_label)

        return prob >= 0.5, prob

    def fit_batch(self, X: np.ndarray, y: np.ndarray, epochs: int = 5) -> float:
        """
        Fine-tune the MLP on a batch of (features, labels).

        Parameters
        ----------
        X : np.ndarray (N, F)
        y : np.ndarray (N,) — binary labels (0 or 1)
        epochs : int

        Returns
        -------
        float : Final epoch loss.
        """
        self._model.train()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        final_loss = 0.0
        for _ in range(epochs):
            self._optimizer.zero_grad()
            preds = self._model(X_t)
            loss = self._loss_fn(preds, y_t)
            loss.backward()
            self._optimizer.step()
            final_loss = float(loss.item())
        logger.info("NeuralCompliance fit: final loss=%.4f (N=%d)", final_loss, len(y))
        return final_loss

    def train_from_buffer(self, epochs: int = 10) -> Optional[float]:
        """
        Train (or fine-tune) the MLP on the accumulated pseudo-labelled buffer.
        Called automatically at end-of-video in HybridPipeline.
        """
        if self.n_samples < self._cold_start_threshold:
            logger.info(
                "Not enough samples (%d / %d) to train MLP yet.",
                self.n_samples,
                self._cold_start_threshold,
            )
            return None
        X = np.stack(self._X_buf)
        y = np.array(self._y_buf, dtype=np.float32)
        loss = self.fit_batch(X, y, epochs=epochs)
        self._save_state()
        return loss

    # ------------------------------------------------------------------
    # Rule-based baseline (kept for cold-start and pseudo-labelling)
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_check(detections: List[dict]) -> bool:
        """
        Reproduce the original helmet-violation rule.
        Used as a pseudo-label oracle during the MLP cold-start phase.
        Returns True if a violation is detected.
        """
        labels = [d.get("label", "").lower() for d in detections]
        has_person = any(lbl == "person" for lbl in labels)
        has_helmet = any("helmet" in lbl for lbl in labels)
        # Original rule: Person + no Helmet = Violation
        return has_person and not has_helmet

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _add_sample(self, features: np.ndarray, label: float) -> None:
        self._X_buf.append(features)
        self._y_buf.append(label)

    def _save_state(self) -> None:
        """Save model weights and accumulated training data to disk."""
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "n_samples": self.n_samples,
            },
            self._checkpoint_path,
        )
        np.savez(
            self._data_path,
            X=np.stack(self._X_buf) if self._X_buf else np.empty((0, self._input_dim)),
            y=np.array(self._y_buf),
        )
        logger.info(
            "NeuralCompliance state saved (%d samples).", self.n_samples
        )

    def _load_state(self) -> None:
        """Warm-start from saved checkpoint and data if available."""
        if self._checkpoint_path.exists():
            checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
            self._model.load_state_dict(checkpoint["model_state"])
            logger.info(
                "NeuralCompliance: loaded checkpoint (prev_samples=%d).",
                checkpoint.get("n_samples", 0),
            )
        if self._data_path.exists():
            data = np.load(self._data_path)
            X, y = data["X"], data["y"]
            if len(y) > 0:
                self._X_buf = list(X)
                self._y_buf = list(y)
                logger.info(
                    "NeuralCompliance: warm-started from %d saved samples.", len(y)
                )
