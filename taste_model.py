"""Taste-learning layer: logistic regression over the per-frame feature vector.

Feature vector (in order):
  0. sharpness_log, standardized
  1. exposure_quality, standardized
  2. composition (Muse Spark vision score; NEUTRAL 0.5 while pending)
  3. moment (Muse Spark vision score; NEUTRAL 0.5 while pending)

While vision is pending, features 2-3 are constant, so they carry ~zero
weight after standardization — the model honestly learns only from real
signals (sharpness/exposure) until the vision pass goes live.

Training data = calibration labels (keeper=1, reject=0) + the user's
keep/reject feedback. Refit is cheap (tens of rows), so every label
action refits and re-ranks immediately.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from vision import NEUTRAL_VISION_SCORE

FEATURE_NAMES = ["sharpness_log", "exposure_quality", "composition", "moment"]


def feature_vector(feat: dict, vision: dict) -> np.ndarray:
    comp = vision.get("composition")
    mom = vision.get("moment")
    return np.array(
        [
            feat["sharpness_log"],
            feat["exposure_quality"],
            NEUTRAL_VISION_SCORE if comp is None else comp,
            NEUTRAL_VISION_SCORE if mom is None else mom,
        ],
        dtype=float,
    )


class TasteModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000)
        self.trained = False
        self.n_train = 0

    def fit(self, vectors: list[np.ndarray], labels: list[int]) -> None:
        """Fit on >=2 classes; with a single class, fall back to untrained."""
        if len(set(labels)) < 2 or len(labels) < 4:
            self.trained = False
            self.n_train = len(labels)
            return
        X = np.vstack(vectors)
        self.scaler.fit(X)
        self.clf.fit(self.scaler.transform(X), labels)
        self.trained = True
        self.n_train = len(labels)

    def rank_scores(self, vectors: list[np.ndarray]) -> list[float]:
        """P(keep) per frame; 0.5 (neutral) when untrained."""
        if not self.trained:
            return [0.5] * len(vectors)
        X = self.scaler.transform(np.vstack(vectors))
        proba = self.clf.predict_proba(X)
        keep_idx = list(self.clf.classes_).index(1)
        return [float(p[keep_idx]) for p in proba]

    def weights(self) -> dict:
        if not self.trained:
            return {name: None for name in FEATURE_NAMES}
        return {
            name: round(float(w), 3)
            for name, w in zip(FEATURE_NAMES, self.clf.coef_[0])
        }
