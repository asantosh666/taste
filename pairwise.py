"""Within-group pairwise re-rank for Taste (Muse Spark, one call per group).

For each near-duplicate group, the connector helper `pairwise_rank.py`
sends the group's frames together in one Muse Spark call and returns a
forced-choice ranking (best to worst), a winner, a margin, and a
one-clause deciding difference. Independent 0-10 scores compress; a
forced choice cannot tie.

Contract
--------
`PairwisePass.rank_group(members)` returns a dict:
  - status: "live" | "pending" (any failure degrades to pending, never
    to an invented rank)
  - ranking: member ids best to worst (live only)
  - winner: winning member id (live only)
  - margin: 0..1 confidence gap winner vs runner-up (live only)
  - deciding: one clause stating what decides the winner (live only)
  - size: group size

Honesty rules (same as vision.py, non-negotiable):
- Only *live* pairwise results are cached, keyed by the group's content
  hash (sha256 of the sorted member file hashes). A failed call is never
  cached: the next rank retries it.
- The cache stores member hashes in call order plus the label ranking;
  a cache hit is only used when the current members' hashes match the
  stored order exactly, so a changed group never inherits a stale rank.
- The app process never handles the API key: the helper resolves the
  credential through the vault surrogate (see pairwise_rank.py).

Blend rule (stated plainly, fixed before measuring):
  blended = 0.7 * model_score + 0.3 * position_score
  position_score = (n - rank_in_group + 1) / n   (winner 1.0, last 1/n)
  for grouped frames with a live pairwise result; ungrouped frames and
  pending groups keep blended = model_score. The original model score
  stays in every API item as `score`, and the pairwise object is shown
  alongside it, so the re-rank is inspectable, never silent.

MEASURED OUTCOME (2026-10-28, do not re-litigate without new evidence):
sorting by blended REGRESSED the global ranking (separation +0.072 ->
+0.017, top-25 6 -> 4) because pairwise winner accuracy on calibration
groups is ~52% and the sharpness-first criterion mismatches the user's
moment-led taste. main.py therefore sorts by the model score; `blend()`
is kept only for the experimental `blended_score` field.
"""
from __future__ import annotations

import hashlib
import json
import os
import string
import subprocess
import sys
from pathlib import Path

# Pairwise helper (connector bin). Overridable for tests via PAIRWISE_HELPER.
PAIRWISE_HELPER = Path(
    os.environ.get(
        "PAIRWISE_HELPER",
        str(
            Path.home()
            / "workspace"
            / "custom-connectors"
            / "custom.meta-model-api"
            / "bin"
            / "pairwise_rank.py"
        ),
    )
)
PAIRWISE_TIMEOUT_S = 240

# Blend weights (fixed a priori; see module docstring).
BLEND_MODEL_WEIGHT = 0.7
BLEND_PAIRWISE_WEIGHT = 0.3


def group_key(member_shas: list[str]) -> str:
    """Content hash for a group: sha256 of the sorted member file hashes."""
    return hashlib.sha256("\n".join(sorted(member_shas)).encode()).hexdigest()


def position_score(rank_in_group: int, size: int) -> float:
    """1-based rank -> position score: winner 1.0, last 1/n."""
    return (size - rank_in_group + 1) / size


def blend(model_score: float, rank_in_group: int | None, size: int | None) -> float:
    """Blended score per the fixed rule in the module docstring."""
    if rank_in_group is None or size is None:
        return model_score
    return (
        BLEND_MODEL_WEIGHT * model_score
        + BLEND_PAIRWISE_WEIGHT * position_score(rank_in_group, size)
    )


class PairwisePass:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def rank_group(self, members: list[dict]) -> dict:
        """Rank one group. members: [{id, path, sha}] in call order.

        Returns the contract dict; on any failure, status "pending" with
        no ranking, and nothing is cached.
        """
        shas = [m["sha"] for m in members]
        key = group_key(shas)
        cached = self._cache_path(key)
        if cached.exists():
            try:
                data = json.loads(cached.read_text())
                if data.get("member_shas") == shas:
                    return self._from_cache(data, members)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # corrupt cache entry: fall through and re-call

        result = self._call_helper(members)
        if result is None:
            return {"status": "pending", "size": len(members)}
        labels = list(string.ascii_uppercase[: len(members)])
        ranking_ids = [members[labels.index(l)]["id"] for l in result["ranking"]]
        data = {
            "member_shas": shas,
            "ranking": result["ranking"],
            "margin": result["margin"],
            "deciding": result["deciding"],
            "size": len(members),
        }
        self._cache_path(key).write_text(json.dumps(data))
        return {
            "status": "live",
            "ranking": ranking_ids,
            "winner": ranking_ids[0],
            "margin": result["margin"],
            "deciding": result["deciding"],
            "size": len(members),
        }

    @staticmethod
    def _from_cache(data: dict, members: list[dict]) -> dict:
        labels = list(string.ascii_uppercase[: len(members)])
        ranking_ids = [members[labels.index(l)]["id"] for l in data["ranking"]]
        return {
            "status": "live",
            "ranking": ranking_ids,
            "winner": ranking_ids[0],
            "margin": data["margin"],
            "deciding": data["deciding"],
            "size": len(members),
        }

    def _call_helper(self, members: list[dict]) -> dict | None:
        """Run the pairwise helper once. None on any failure (pending)."""
        if not PAIRWISE_HELPER.is_file():
            print(f"Pairwise helper not found: {PAIRWISE_HELPER}", file=sys.stderr)
            return None
        try:
            proc = subprocess.run(
                [sys.executable, str(PAIRWISE_HELPER)]
                + [str(m["path"]) for m in members],
                capture_output=True,
                text=True,
                timeout=PAIRWISE_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"Pairwise helper failed: {exc}", file=sys.stderr)
            return None
        if proc.returncode != 0:
            print(
                f"Pairwise helper exit {proc.returncode} for "
                f"{len(members)}-member group",
                file=sys.stderr,
            )
            return None
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            labels = list(string.ascii_uppercase[: len(members)])
            ranking = data["ranking"]
            margin = float(data["margin"])
            deciding = str(data["deciding"]).strip()
            if sorted(ranking) != sorted(labels):
                raise ValueError("ranking is not a permutation of labels")
            if not 0.0 <= margin <= 1.0 or not deciding:
                raise ValueError("margin/deciding invalid")
            return {"ranking": ranking, "margin": margin, "deciding": deciding}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            print(f"Pairwise helper bad output: {exc}", file=sys.stderr)
            return None
