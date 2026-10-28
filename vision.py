"""Vision pass interface for Taste (Muse Spark multimodal + SAM 3.1).

Contract
--------
`VisionPass.score(path)` returns a `VisionScores` object:
  - composition, moment: float 0..1, or None when no real model score exists
  - subject_sharpness: float 0..1, or None (SAM 3.1 subject-mask sharpness)
  - status: "live" | "pending_api_key" | "pending_configuration"
  - note: human-readable explanation of the status

Honesty rules (non-negotiable):
- Without META_MODEL_API_KEY in the environment, every score is None and
  status is "pending_api_key". The UI must show "vision pass pending API key"
  and must never present neutral filler as model output.
- Scores are cached on disk keyed by the file's sha256, so a frame is only
  ever scored once (budget guardrail: re-ranking never re-calls the API).
  Only *live* scores are cached. Pending results are never cached: D1
  cached pending results for all 152 demo frames, which would have made
  the pass stay "pending" forever even after the key arrived, because the
  stale cache would be returned first. Fixed 2026-10-27 (D2 prep).
- The live client uses the Meta Model API's OpenAI-compatible chat
  completions surface (base https://api.meta.ai/v1, model muse-spark-1.3
  by default, Bearer key) and asks the model to return JSON with
  composition/moment in 0..1. That surface is documented by third-party
  guides, not yet confirmed against Meta's own docs in this run, so the
  first live call is the confirmation: any failure degrades to pending,
  never to invented scores. SAM 3.1 subject-mask sharpness is NOT wired
  yet — its request contract is unconfirmed, so subject_sharpness stays
  None rather than being guessed. Nothing here pretends a model call
  happened.

Ranking fallback: when scores are pending, the ranker substitutes the
neutral value 0.5 for composition/moment *inside the model only*, and the
UI labels those columns "pending". See taste_model.py.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from features import file_sha256

NEUTRAL_VISION_SCORE = 0.5


@dataclass
class VisionScores:
    composition: float | None
    moment: float | None
    subject_sharpness: float | None
    status: str  # "live" | "pending_api_key" | "pending_configuration"
    note: str

    def as_dict(self) -> dict:
        return {
            "composition": self.composition,
            "moment": self.moment,
            "subject_sharpness": self.subject_sharpness,
            "status": self.status,
            "note": self.note,
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
            self._cache_path(digest).write_text(json.dumps(result.as_dict()))
        return result

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
        only. SAM 3.1 is not called here: its mask contract is unconfirmed,
        so subject_sharpness stays None. Any failure degrades to pending,
        never to invented scores.
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
                "Subject sharpness (SAM 3.1) not yet wired.",
            )
        except Exception as exc:  # noqa: BLE001 - any failure = pending, never fake
            return VisionScores(
                None, None, None, "pending_configuration",
                f"Vision API call failed ({exc}); scores stay pending.",
            )
