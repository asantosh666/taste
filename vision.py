"""Vision pass interface for Taste (Muse Spark multimodal + SAM 3.1).

Contract
--------
`VisionPass.score(path)` returns a `VisionScores` object:
  - composition, moment: float 0..1, or None when no real model score exists
  - subject_sharpness: float 0..1, or None (SAM 3.1 subject-mask sharpness)
  - status: "live" | "pending_api_key" | "pending_configuration"
  - note: human-readable explanation of the status
  - sam_done: True once a SAM 3.1 call has completed for this file
    (including a completed empty result: no subject found)
  - sam_mask_returned: True/False when sam_done, else None
  - sam_mask_count: number of subject masks, when sam_done, else None

Honesty rules (non-negotiable):
- Without META_MODEL_API_KEY in the environment, fresh Spark scores are
  None and status is "pending_api_key". The UI must show "vision pass
  pending API key" and must never present neutral filler as model output.
  (SAM merges for already-cached live entries do NOT need the env key:
  the SAM helper resolves the credential through the vault surrogate.)
- Scores are cached on disk keyed by the file's sha256, so a frame is only
  ever scored once (budget guardrail: re-ranking never re-calls the API).
  Only *live* scores are cached. Pending results are never cached: D1
  cached pending results for all 152 demo frames, which would have made
  the pass stay "pending" forever even after the key arrived, because the
  stale cache would be returned first. Fixed 2026-10-27 (D2 prep).
- The live Spark client uses the Meta Model API's OpenAI-compatible chat
  completions surface (base https://api.meta.ai/v1, model muse-spark-1.3
  by default, Bearer key) and asks the model to return JSON with
  composition/moment in 0..1. Any failure degrades to pending, never to
  invented scores.
- SAM 3.1 subject sharpness is wired since 2026-10-28 via a subprocess
  call to the connector helper `sam_score.py`, which calls
  POST https://api.meta.ai/v1/responses (model sam-3.1, prompt "person")
  through the vault surrogate — the app process never handles the API
  key for SAM. The output_text lane is parsed with the official
  meta-sam-parser (0.1.0 required: 0.0.5 rejects records carrying the
  `c=` confidence field the live API now emits) and masks are decoded
  with its decode_mask_to_raster. A completed empty result (no subject
  found) is cached as sam_done with subject_sharpness None — that None
  means "no subject", not "pending". A failed SAM call is never cached:
  the entry stays sam_done=False so a later rank retries it.
- subject_sharpness itself is a ratio: mask-restricted Laplacian
  variance / (mask variance + whole-frame variance) on one pixel array.
  0.5 means "subject as sharp as its own frame". It is NOT comparable
  across frames as an absolute sharpness number — only as a ratio.

Ranking fallback: when scores are pending, the ranker substitutes the
neutral value 0.5 for composition/moment/subject_sharpness *inside the
model only*, and the UI labels those columns "pending". See taste_model.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from features import file_sha256

NEUTRAL_VISION_SCORE = 0.5

# SAM helper (connector bin). Overridable for tests via SAM_SCORE_HELPER.
SAM_SCORE_HELPER = Path(
    os.environ.get(
        "SAM_SCORE_HELPER",
        str(
            Path.home()
            / "workspace"
            / "custom-connectors"
            / "custom.meta-model-api"
            / "bin"
            / "sam_score.py"
        ),
    )
)
SAM_PROMPT = "person"
SAM_TIMEOUT_S = 180


@dataclass
class VisionScores:
    composition: float | None
    moment: float | None
    subject_sharpness: float | None
    status: str  # "live" | "pending_api_key" | "pending_configuration"
    note: str
    sam_done: bool = False
    sam_mask_returned: bool | None = None
    sam_mask_count: int | None = None

    def as_dict(self) -> dict:
        return {
            "composition": self.composition,
            "moment": self.moment,
            "subject_sharpness": self.subject_sharpness,
            "status": self.status,
            "note": self.note,
            "sam_done": self.sam_done,
            "sam_mask_returned": self.sam_mask_returned,
            "sam_mask_count": self.sam_mask_count,
        }


DEFAULT_API_URL = "https://api.meta.ai/v1/chat/completions"
DEFAULT_MODEL = "muse-spark-1.3"


class VisionPass:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("META_MODEL_API_KEY", "").strip()
        # META_MODEL_API_URL overrides the default; it is no longer required
        # for the pass to attempt a call, since the default is researched.
        self.api_url = (
            os.environ.get("META_MODEL_API_URL", "").strip() or DEFAULT_API_URL
        )
        self.model = os.environ.get("META_MODEL_API_MODEL", "").strip() or DEFAULT_MODEL

    def _cache_path(self, digest: str) -> Path:
        return self.cache_dir / f"{digest}.json"

    def score(self, path: Path) -> VisionScores:
        digest = file_sha256(path)
        cached = self._cache_path(digest)
        if cached.exists():
            try:
                data = json.loads(cached.read_text())
                result = VisionScores(**data)
                # Only live scores are trustworthy cache entries. A stale
                # pending entry (all of D1's cache) must not block a later
                # live call once the key exists.
                if result.status == "live":
                    if result.sam_done:
                        # Backfill (2026-10-28): D2 cache entries carry a
                        # stale note saying SAM is "not yet wired". The
                        # scores are unaffected; only the label is fixed.
                        if "not yet wired" in result.note:
                            result.note = (
                                "Scored by Muse Spark via the Meta Model API. "
                                "Subject sharpness via SAM 3.1."
                            )
                            self._cache_path(digest).write_text(
                                json.dumps(result.as_dict())
                            )
                        return result
                    # Cached live Spark entry predating SAM wiring: merge
                    # SAM once, then cache the merged entry. The SAM helper
                    # uses the vault surrogate, so this needs no env key.
                    merged = self._merge_sam(path, result)
                    if merged is not None:
                        self._cache_path(digest).write_text(
                            json.dumps(merged.as_dict())
                        )
                        return merged
                    # SAM call failed: return the Spark entry with
                    # subject_sharpness None and leave the cache untouched,
                    # so a later rank retries SAM instead of freezing it.
                    return result
            except (json.JSONDecodeError, TypeError):
                pass  # corrupt cache entry: fall through and re-score

        if not self.api_key:
            return VisionScores(
                None, None, None, "pending_api_key",
                "Vision pass pending API key: set META_MODEL_API_KEY to enable "
                "Muse Spark composition/moment scores and SAM 3.1 subject masks.",
            )

        result = self._call_api(path)

        # Cache live scores only; pending results must be retried, not frozen.
        if result.status == "live":
            merged = self._merge_sam(path, result)
            if merged is not None:
                result = merged
            # If SAM failed, cache the Spark-only entry WITHOUT sam_done so
            # the next rank retries SAM; never cache a fake sharpness.
            self._cache_path(digest).write_text(json.dumps(result.as_dict()))
        return result

    def _merge_sam(self, path: Path, result: VisionScores) -> VisionScores | None:
        """Run the SAM helper once and fold its result into `result`.

        Returns the merged VisionScores, or None if the SAM call failed
        (caller keeps the unmerged entry and retries later). A completed
        empty result (no subject found) is a success: subject_sharpness
        stays None with sam_done True — "no subject", not "pending".
        """
        sam = self._sam_subject_sharpness(path)
        if sam is None:
            return None
        result.subject_sharpness = sam["subject_sharpness"]
        result.sam_done = True
        result.sam_mask_returned = sam["mask_returned"]
        result.sam_mask_count = sam["mask_count"]
        result.note = (
            "Scored by Muse Spark via the Meta Model API. "
            "Subject sharpness via SAM 3.1."
        )
        return result

    def _sam_subject_sharpness(self, path: Path) -> dict | None:
        """Subject sharpness via the connector SAM helper (subprocess).

        The helper resolves the API credential through the vault
        surrogate; this process never sees the key. Any failure (missing
        helper, nonzero exit, bad JSON, out-of-range value) returns None
        and the caller degrades to pending — never to a filler score.
        """
        if not SAM_SCORE_HELPER.is_file():
            print(f"SAM helper not found: {SAM_SCORE_HELPER}", file=sys.stderr)
            return None
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SAM_SCORE_HELPER),
                    str(path),
                    "--prompt",
                    SAM_PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=SAM_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"SAM helper failed for {path.name}: {exc}", file=sys.stderr)
            return None
        if proc.returncode != 0:
            print(
                f"SAM helper exit {proc.returncode} for {path.name}",
                file=sys.stderr,
            )
            return None
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            sharp = data["subject_sharpness"]
            if sharp is not None:
                sharp = float(sharp)
                if not 0.0 <= sharp <= 1.0:
                    raise ValueError("subject_sharpness outside 0..1")
            return {
                "subject_sharpness": sharp,
                "mask_returned": bool(data["mask_returned"]),
                "mask_count": int(data["mask_count"]),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(
                f"SAM helper bad output for {path.name}: {exc}",
                file=sys.stderr,
            )
            return None

    @staticmethod
    def _parse_scores(text: str) -> tuple[float, float]:
        """Extract composition/moment in 0..1 from model text.

        Accepts bare JSON or JSON embedded in prose/code fences. Anything
        else raises, and the caller degrades to pending — never to a guess.
        """
        import re

        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in model reply")
        data = json.loads(match.group(0))
        comp = float(data["composition"])
        mom = float(data["moment"])
        if not (0.0 <= comp <= 1.0 and 0.0 <= mom <= 1.0):
            raise ValueError("scores outside 0..1")
        return comp, mom

    def _call_api(self, path: Path) -> VisionScores:
        """Muse Spark vision via the OpenAI-compatible chat completions API.

        Sends the frame as a base64 data-URI image and asks for JSON scores
        only. SAM 3.1 is not called here: score() merges it afterwards via
        the subprocess helper, so this returns subject_sharpness None with
        sam_done False. Any failure degrades to pending, never to invented
        scores.
        """
        try:
            import base64
            import mimetypes

            mime, _ = mimetypes.guess_type(str(path))
            if mime is None or not mime.startswith("image/"):
                mime = "image/jpeg"
            b64 = base64.b64encode(path.read_bytes()).decode()
            payload = json.dumps(
                {
                    "model": self.model,
                    "temperature": 0,
                    # Token budget: Muse Spark spends most of its completion
                    # budget on hidden reasoning (observed 198 reasoning
                    # tokens on a first live call). With max_tokens=200 the
                    # reply itself got zero tokens, content came back null,
                    # and finish_reason was "length". 1024 leaves ample room
                    # for the short JSON reply after reasoning.
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Score this photograph for photo culling. "
                                        "Return ONLY a JSON object with two numbers "
                                        "between 0 and 1: "
                                        '{"composition": <framing, balance, leading lines>, '
                                        '"moment": <decisive moment, expression, gesture>}. '
                                        "No other text."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{b64}"
                                    },
                                },
                            ],
                        }
                    ],
                }
            ).encode()
            req = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            if isinstance(text, list):  # some clients return content blocks
                text = "".join(
                    str(part.get("text", "")) for part in text
                )
            comp, mom = self._parse_scores(str(text))
            return VisionScores(
                comp, mom, None, "live",
                "Scored by Muse Spark via the Meta Model API. "
                "Subject sharpness merged separately via SAM 3.1.",
            )
        except Exception as exc:  # noqa: BLE001 - any failure = pending, never fake
            return VisionScores(
                None, None, None, "pending_configuration",
                f"Vision API call failed ({exc}); scores stay pending.",
            )
