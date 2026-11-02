"""Pairwise (Bradley-Terry) learning — EXPERIMENTAL ONLY, does not drive ranking.

Measured 2026-10-28 (offline experiment, calibration labels only):
- Pair construction: all keeper-vs-reject pairs (25x25 = 625); only 3 of
  them are within-duplicate-group pairs, so within-group override signal
  is nearly absent from the calibration set.
- Primary evaluation (leak-free): 50 stratified frame-level splits of the
  50 labeled frames (40 train / 10 test); both models fit on train frames
  only and scored on pairs from test frames only.
  - Pointwise (live model): held-out pair accuracy 0.502 (SEM 0.021).
  - Pairwise (this module, C=1.0): held-out pair accuracy 0.500 (SEM 0.022).
  - Difference: -0.002 (SEM 0.002); pairwise wins 3/50 splits, ties 41.
- Verdict: NON-RESULT. Pairwise learning does not beat the pointwise
  model, and neither generalizes beyond chance on held-out frames — the
  live +0.072 keeper/reject separation is in-sample fit on the same 50
  labels, not generalization. Do not present separation as generalization.
- The weights below are therefore exposed for inspection only (like
  blended_score). Do NOT sort by them without a pairwise signal that
  first beats the pointwise model on held-out pairs.

Method: logistic regression on standardized feature DIFFERENCES
(keeper minus reject, both orientations so sklearn sees 2 classes),
no intercept (a proper Bradley-Terry model: swapping the pair must
complement the probability), C=1.0 — the same default as TasteModel.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from taste_model import FEATURE_NAMES

# Offline-measured evaluation snapshot (see module docstring). These are
# measured numbers from the 2026-10-28 experiment, not live-computed.
PAIRWISE_LEARNED_EVAL = {
    "measured": "2026-10-28, 50 stratified frame-level splits, calibration labels only",
    "held_out_pairwise_accuracy": 0.500,
    "held_out_pointwise_accuracy": 0.502,
    "mean_difference": -0.002,
    "pairwise_wins_splits": "3/50 (ties 41)",
    "within_group_labeled_pairs": 3,
    "verdict": (
        "non-result: pairwise learning does not beat the pointwise model "
        "on held-out pairs; experimental only, does not drive ranking"
    ),
}


def fit_pairwise_weights(
    vectors: list[np.ndarray], labels: list[int], C: float = 1.0
) -> dict | None:
    """Fit Bradley-Terry weights on feature differences.

    Returns {feature_name: weight} in standardized space, or None when
    the labels carry fewer than 2 classes (same fallback rule as
    TasteModel.fit).
    """
    if len(set(labels)) < 2 or len(labels) < 4:
        return None
    X = np.vstack(vectors)
    scaler = StandardScaler()
    scaler.fit(X)
    Z = scaler.transform(X)
    keep_idx = [i for i, lab in enumerate(labels) if lab == 1]
    rej_idx = [i for i, lab in enumerate(labels) if lab == 0]
    diffs = []
    labels = []
    for ki in keep_idx:
        for ri in rej_idx:
            d = Z[ki] - Z[ri]
            # Labels are appended ALONGSIDE their diffs: each pair
            # contributes (d, 1) and (-d, 0). A blocked layout
            # ([1]*n + [0]*n) would misalign with interleaved diffs.
            diffs.append(d)
            labels.append(1)
            diffs.append(-d)
            labels.append(0)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=1000, C=C, fit_intercept=False)
    clf.fit(np.vstack(diffs), y)
    return {
        name: round(float(w), 3)
        for name, w in zip(FEATURE_NAMES, clf.coef_[0])
    }
