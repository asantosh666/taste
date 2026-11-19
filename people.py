"""People detection for Taste — tiled EfficientDet-Lite0, fully offline.

Contract
--------
`PeoplePass.detect(path)` returns a dict:
  - has_person: True/False once detected, None when not yet detected
  - person_count: number of person detections across all tiles (None if undetected)
  - max_conf: best detection confidence, or None
  - detected: True once a result exists (cached or fresh)
  - status: "done" | "unavailable" (mediapipe/model missing)
  - note: human-readable explanation

Method (validated 2026-11-12 in ~/workspace/taste-culling-pretest):
MediaPipe ObjectDetector + EfficientDet-Lite0, COCO `person` only,
score threshold 0.3, run over the full image plus 3x3 overlapping 50%
crops (origins 0/25/50% of width/height). On the 50-photo pre-test sample
this scored 98% accuracy, 100% recall vs corrected ground truth
(28 with people, 22 without; the single false positive was a sheep).
Untiled, the same detector scored only 88% / 81% — the tiles are what
make small/distant figures visible, so do not drop them.

Honesty / offline rules (mirror vision.py):
- No API key, no network: the model file lives next to this app
  (efficientdet_lite0.tflite) and everything runs on CPU (~0.6 s/photo).
- Results are cached on disk keyed by the file's sha256, so a photo is
  only ever detected once. A failed detection is never cached: the entry
  stays undetected so a later run retries it.
- The detector is created lazily on first use, inside the background
  worker thread only (MediaPipe detectors are not shared across threads).
  If mediapipe or the model file is missing, detection reports
  "unavailable" instead of failing the app — the rest of Taste keeps
  working.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from features import file_sha256

CONF_THRESHOLD = 0.3
MODEL_NAME = "efficientdet_lite0 tiled 3x3+full"


def tiles_for(im: Image.Image) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Full image + 3x3 overlapping 50% crops (origins 0/25/50%).

    Identical to detect_tiled3.py in the pre-test — the pre-test numbers
    only hold for this exact tiling, so do not change it.
    """
    w, h = im.size
    tiles = [("full", (0, 0, w, h))]
    tw, th = int(w * 0.5), int(h * 0.5)
    for ox in (0, int(w * 0.25), int(w * 0.5)):
        for oy in (0, int(h * 0.25), int(h * 0.5)):
            tiles.append(
                (f"crop_{ox}_{oy}", (ox, oy, min(w, ox + tw), min(h, oy + th)))
            )
    return tiles


class PeoplePass:
    def __init__(self, cache_dir: Path, model_path: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self._detector = None
        self._detector_error: str | None = None
        self._lock = threading.Lock()
        self._job: dict = {
            "status": "idle",  # idle | running | done | unavailable
            "done": 0,
            "total": 0,
            "with_people": 0,
            "without_people": 0,
            "set_key": None,
            "note": "",
            "started_at": None,
            "seconds": None,
        }
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ cache
    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"

    def cached(self, digest: str) -> dict | None:
        p = self._cache_path(digest)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            if isinstance(data.get("has_person"), bool):
                return data
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _save_cache(self, digest: str, data: dict) -> None:
        tmp = self._cache_path(digest).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._cache_path(digest))

    # --------------------------------------------------------- detector
    def _get_detector(self):
        if self._detector is not None:
            return self._detector
        if self._detector_error is not None:
            return None
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(
                model_asset_path=str(self.model_path)
            )
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                max_results=20,
                score_threshold=CONF_THRESHOLD,
                category_allowlist=["person"],
            )
            self._detector = vision.ObjectDetector.create_from_options(options)
            self._mp = mp
        except Exception as exc:  # noqa: BLE001 - any failure = unavailable
            self._detector_error = (
                f"People detector unavailable ({exc}); "
                "install mediapipe and keep efficientdet_lite0.tflite "
                "next to the app to enable offline people detection."
            )
        return self._detector

    def override(self, digest: str, has_person: bool) -> dict:
        """Manual correction of one photo's people result.

        Writes the same cache format detect() writes, so the correction
        survives restarts and shows up in GET /api/people, the filter
        counts, and the separate dry-run immediately. person_count is
        1/0 (a flip is a yes/no verdict, not a recount); max_conf is
        kept from the previous entry when there is one. The
        "overridden" marker records that a human, not the detector,
        set this value.
        """
        old = self.cached(digest)
        data = {
            "has_person": bool(has_person),
            "person_count": 1 if has_person else 0,
            "max_conf": old["max_conf"] if old else None,
            "model": MODEL_NAME,
            "conf": CONF_THRESHOLD,
            "overridden": True,
        }
        self._save_cache(digest, data)
        return data

    # -------------------------------------------------------- detection
    def detect(self, path: Path) -> dict:
        """Detect people in one photo, using the disk cache when present."""
        digest = file_sha256(path)
        hit = self.cached(digest)
        if hit is not None:
            return {**hit, "detected": True, "status": "done", "note": ""}
        detector = self._get_detector()
        if detector is None:
            return {
                "has_person": None,
                "person_count": None,
                "max_conf": None,
                "detected": False,
                "status": "unavailable",
                "note": self._detector_error or "detector unavailable",
            }
        try:
            im = Image.open(path).convert("RGB")
            boxes = []
            for _name, box in tiles_for(im):
                crop = im.crop(box)
                mp_img = self._mp.Image(
                    image_format=self._mp.ImageFormat.SRGB,
                    data=np.asarray(crop),
                )
                res = detector.detect(mp_img)
                for d in res.detections:
                    boxes.append(round(float(d.categories[0].score), 3))
        except Exception as exc:  # noqa: BLE001 - never cache a failure
            return {
                "has_person": None,
                "person_count": None,
                "max_conf": None,
                "detected": False,
                "status": "unavailable",
                "note": f"Detection failed for {path.name}: {exc}",
            }
        data = {
            "has_person": len(boxes) > 0,
            "person_count": len(boxes),
            "max_conf": max(boxes) if boxes else None,
            "model": MODEL_NAME,
            "conf": CONF_THRESHOLD,
        }
        self._save_cache(digest, data)
        return {**data, "detected": True, "status": "done", "note": ""}

    # ---------------------------------------------------- background job
    @staticmethod
    def _public(job: dict) -> dict:
        """Status safe for API responses (no set_key / timestamps)."""
        return {
            k: v
            for k, v in job.items()
            if k not in ("set_key", "started_at")
        }

    def _set_key(self, items: list[tuple[str, Path, str]]) -> str:
        return "|".join(sha for _, _, sha in items)

    def start(self, items: list[tuple[str, Path, str]]) -> dict:
        """Start a background run over a snapshot of (id, path, sha).

        Idempotent: a running job for the same set is left alone; a job
        for a different set replaces it. Already-cached photos are
        skipped instantly, so re-running a finished set is near-free.
        Returns the current job status dict.
        """
        key = self._set_key(items)
        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and self._job["set_key"] == key
            ):
                return self._public(self._job)
            if self._thread is not None and self._thread.is_alive():
                # A run for another set is in flight; do not stack jobs.
                return self._public(self._job)
            self._job = {
                "status": "running",
                "done": 0,
                "total": len(items),
                "with_people": 0,
                "without_people": 0,
                "set_key": key,
                "note": "",
                "started_at": time.time(),
                "seconds": None,
            }
            self._thread = threading.Thread(
                target=self._run, args=(items,), daemon=True
            )
            self._thread.start()
            return self._public(self._job)

    def _run(self, items: list[tuple[str, Path, str]]) -> None:
        with_people = without_people = done = 0
        for _item_id, path, _sha in items:
            res = self.detect(path)
            done += 1
            if res["status"] == "unavailable":
                with self._lock:
                    self._job.update(
                        {
                            "status": "unavailable",
                            "done": done,
                            "note": res["note"],
                        }
                    )
                return
            if res["has_person"]:
                with_people += 1
            else:
                without_people += 1
            with self._lock:
                self._job.update(
                    {
                        "done": done,
                        "with_people": with_people,
                        "without_people": without_people,
                    }
                )
        with self._lock:
            self._job.update(
                {
                    "status": "done",
                    "seconds": round(time.time() - self._job["started_at"], 1),
                }
            )

    def job_status(self) -> dict:
        with self._lock:
            return dict(self._job)

    def reset(self) -> None:
        """Forget the current job (e.g. a new working set was loaded).

        A running thread is left to finish its snapshot harmlessly; its
        results are still cached by sha, so nothing is lost. The next
        start() begins a fresh job.
        """
        with self._lock:
            self._job = {
                "status": "idle",
                "done": 0,
                "total": 0,
                "with_people": 0,
                "without_people": 0,
                "set_key": None,
                "note": "",
                "started_at": None,
                "seconds": None,
            }

    def status_for(self, items: list[tuple[str, Path, str]]) -> dict:
        """Job status for this set, falling back to cache counts.

        If a job ran (or is running) for exactly this set, its status
        wins. Otherwise the disk cache speaks: a fully-cached set is
        "done", anything else is "idle" (nothing started yet).
        """
        key = self._set_key(items)
        job = self.job_status()
        summary = self.summary_for(items)
        if job["set_key"] == key and job["status"] in (
            "running",
            "done",
            "unavailable",
        ):
            merged = self._public(job)
            merged.update(summary)
            merged["status"] = job["status"]
            merged["note"] = job["note"]
            merged["seconds"] = job["seconds"]
            return merged
        if summary["done"] == summary["total"] and summary["total"] > 0:
            return {"status": "done", **summary, "note": "", "seconds": None}
        return {"status": "idle", **summary, "note": "", "seconds": None}

    def summary_for(self, items: list[tuple[str, Path, str]]) -> dict:
        """Counts for the given set from the disk cache (no detection)."""
        done = with_people = without_people = 0
        for _item_id, _path, sha in items:
            hit = self.cached(sha)
            if hit is None:
                continue
            done += 1
            if hit["has_person"]:
                with_people += 1
            else:
                without_people += 1
        return {
            "done": done,
            "total": len(items),
            "with_people": with_people,
            "without_people": without_people,
        }
