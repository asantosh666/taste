"""Taste MVP — FastAPI backend.

Run:  ./run.sh            (or: .venv/bin/python -m uvicorn main:app --port 8000)
UI:    http://127.0.0.1:8000/

Working set: either the built-in demo set (the user's own Nepal frames in
../files/taste-photo-set, referenced, never copied) or a user upload batch.
State is in-memory; user labels persist in data/labels.json keyed by file
hash so demo labels survive restarts. Feature/vision caches live in
.cache/ keyed by file hash, so re-ranking never recomputes or re-calls.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

import features
from pairwise import PairwisePass, blend
from pairwise_learn import PAIRWISE_LEARNED_EVAL, fit_pairwise_weights
from people import PeoplePass
from taste_model import TasteModel, feature_vector
from vision import SAM_PROMPT, SAM_SCORE_HELPER, VisionPass

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DEMO_DIR = APP_DIR.parent / "files" / "taste-photo-set"
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
CACHE_DIR = APP_DIR / ".cache"
THUMB_DIR = CACHE_DIR / "thumbs"
FEATURE_CACHE_FILE = CACHE_DIR / "features.json"
LABELS_FILE = DATA_DIR / "labels.json"
# The working-set folder chosen via POST /api/source/set, persisted so
# it survives restarts. This supersedes the old TASTE_DEMO_DIR env
# patch (v2 Mac setup): the folder is now chosen in the app, not in
# run.sh, and the file below is the only place it is remembered.
WORKING_DIR_FILE = DATA_DIR / "working-dir.json"
# Photos deleted to the OS Trash, keyed by SOURCE PATH (not content
# sha): byte-identical duplicates share a sha, so sha-keying would
# hide a photo's twin when only one copy was deleted. A path key
# deletes exactly the file the user pointed at.
DELETED_FILE = DATA_DIR / "deleted.json"

for d in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR, CACHE_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_persisted_demo_dir() -> Path:
    """Working-set folder from data/working-dir.json, if it still
    exists with a manifest.csv; otherwise the built-in demo set."""
    try:
        data = json.loads(WORKING_DIR_FILE.read_text())
        p = Path(data["demo_dir"]).expanduser()
        if p.is_dir() and (p / "manifest.csv").is_file():
            return p
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    return DEFAULT_DEMO_DIR


# Mutable: POST /api/source/set reassigns this (and persists it).
DEMO_DIR = _load_persisted_demo_dir()


def load_deleted_paths() -> set[str]:
    if DELETED_FILE.exists():
        try:
            data = json.loads(DELETED_FILE.read_text())
            if isinstance(data, list):
                return {str(s) for s in data}
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def save_deleted_paths(paths: set[str]) -> None:
    DELETED_FILE.write_text(json.dumps(sorted(paths), indent=1))

app = FastAPI(title="Taste — AI photo culling that learns your eye")
vision_pass = VisionPass(CACHE_DIR / "vision")
people_pass = PeoplePass(CACHE_DIR / "people", APP_DIR / "efficientdet_lite0.tflite")
pairwise_pass = PairwisePass(CACHE_DIR / "pairwise")
model = TasteModel()
# Pairwise-learned (Bradley-Terry) weights, computed on each refit for
# inspection only. Measured 2026-10-28 as a NON-RESULT (held-out pair
# accuracy 0.500 vs pointwise 0.502): these weights never drive ranking.
pairwise_learned_weights: dict | None = None


# ---------------------------------------------------------------- state
class Item:
    def __init__(self, source: Path, set_name: str, calib_label: str | None,
                 manifest_source: str = ""):
        self.source = source
        self.set_name = set_name
        self.calib_label = calib_label  # "keeper" | "reject" | None
        self.manifest_source = manifest_source  # manifest `source` col:
        # absolute path of the original file this working-set file came
        # from (a RAW for extracted previews, the JPEG itself for
        # linked originals). Used by the delete path to check a
        # symlink target is really this photo's original.
        self.user_label: str | None = None  # "keep" | "reject" | None
        self.sha = ""
        self.feat: dict = {}
        self.vis: dict = {}

    @property
    def filename(self) -> str:
        return self.source.name


class WorkingSet:
    def __init__(self, name: str, items: list[Item]):
        self.name = name
        self.items = items
        self.ranked: list[dict] | None = None
        self.groups: list[list[str]] = []

    def by_id(self, item_id: str) -> Item | None:
        try:
            idx = int(item_id)
        except ValueError:
            return None
        if idx < 0 or idx >= len(self.items):
            return None
        return self.items[idx]


working: WorkingSet | None = None


def load_feature_cache() -> dict:
    if FEATURE_CACHE_FILE.exists():
        try:
            return json.loads(FEATURE_CACHE_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_feature_cache(cache: dict) -> None:
    FEATURE_CACHE_FILE.write_text(json.dumps(cache))


def load_persisted_labels() -> dict:
    if LABELS_FILE.exists():
        try:
            return json.loads(LABELS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_persisted_labels(labels: dict) -> None:
    LABELS_FILE.write_text(json.dumps(labels, indent=1))


def load_demo_set() -> WorkingSet:
    """Build the working set from DEMO_DIR/manifest.csv. Read-only: no copies."""
    manifest = DEMO_DIR / "manifest.csv"
    try:
        manifest_file = open(manifest, newline="")
    except FileNotFoundError:
        # Code-only checkout (no seed photos): a friendly empty set,
        # not a 500 — the repo ships code-only by decision.
        return WorkingSet("demo set not included (code-only repo)", [])
    items: list[Item] = []
    with manifest_file as f:
        for row in csv.DictReader(f):
            rel = row["file"]
            src = DEMO_DIR / rel
            if not src.exists():
                continue
            label = row["label"] if row["label"] in ("keeper", "reject") else None
            items.append(Item(src, row["set"], label, row.get("source", "") or ""))
    if DEMO_DIR.resolve() == DEFAULT_DEMO_DIR.resolve():
        name = "demo — your Nepal frames (152)"
    else:
        name = f"{DEMO_DIR.name} ({len(items)} frames)"
    ws = WorkingSet(name, items)
    _apply_persisted_labels(ws)
    return ws


def _apply_persisted_labels(ws: WorkingSet) -> None:
    persisted = load_persisted_labels()
    for it in ws.items:
        it.sha = features.file_sha256(it.source)
        if it.sha in persisted:
            it.user_label = persisted[it.sha]


def ensure_ranked() -> list[dict]:
    """Compute features/vision, fit the taste model, rank. Cached in memory."""
    global working
    if working is None:
        working = load_demo_set()
    if working.ranked is not None:
        return working.ranked

    feat_cache = load_feature_cache()
    deleted = load_deleted_paths()
    train_vecs, train_labels = [], []
    ranked = []
    for idx, it in enumerate(working.items):
        if not it.sha:
            it.sha = features.file_sha256(it.source)
        if str(it.source) in deleted:
            # Trashed via /api/dupes/delete: excluded from the ranking,
            # the groups, the people lists and the dry-runs. The item
            # stays in working.items so ids stay stable within a set.
            # Keyed by source path, not sha: byte-identical duplicates
            # share a sha, and deleting one must not hide its twin.
            continue
        if it.sha in feat_cache:
            it.feat = feat_cache[it.sha]
        else:
            try:
                it.feat = features.extract_features(it.source)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                # Unreadable file (e.g. a non-image that slipped into a
                # set): skip it with a warning instead of failing the
                # whole rank with a 500. Upload validation (400) is the
                # primary guard; this is the defense in depth.
                print(f"Skipping unreadable file {it.source.name}: {exc}", file=sys.stderr)
                continue
            feat_cache[it.sha] = it.feat
        it.vis = vision_pass.score(it.source).as_dict()
        vec = feature_vector(it.feat, it.vis)
        # Training labels: calibration labels + the user's own keep/reject.
        # "maybe" is a deliberate non-label: it excludes the frame from
        # training entirely (a user override must supersede the calib
        # label, not fall through to it).
        lab = None
        if it.user_label == "keep":
            lab = 1
        elif it.user_label == "reject":
            lab = 0
        elif it.user_label == "maybe":
            lab = None
        elif it.calib_label == "keeper":
            lab = 1
        elif it.calib_label == "reject":
            lab = 0
        if lab is not None:
            train_vecs.append(vec)
            train_labels.append(lab)
        ranked.append({"idx": idx, "vec": vec})
    save_feature_cache(feat_cache)

    model.fit(train_vecs, train_labels)
    # Pairwise-learned weights for inspection only (see pairwise_learn.py:
    # measured non-result, never used for sorting).
    global pairwise_learned_weights
    pairwise_learned_weights = fit_pairwise_weights(train_vecs, train_labels)
    scores = model.rank_scores([r["vec"] for r in ranked])

    hashes = {str(r["idx"]): working.items[r["idx"]].feat["dhash"] for r in ranked}
    group_list = features.group_near_dupes(hashes)
    group_of = {}
    for gi, grp in enumerate(group_list):
        for member in grp:
            group_of[member] = gi
    working.groups = group_list

    # Within-group pairwise re-rank (Muse, one call per group, cached by
    # group content hash). Pending groups keep the model score; the
    # original model score stays in every item as `score`, so the
    # re-rank is inspectable, never silent.
    pairwise_by_id: dict[str, dict] = {}
    for grp in group_list:
        members = [
            {
                "id": m,
                "path": working.items[int(m)].source,
                "sha": working.items[int(m)].sha,
            }
            for m in sorted(grp, key=int)
        ]
        res = pairwise_pass.rank_group(members)
        if res["status"] == "live":
            for pos, mid in enumerate(res["ranking"], start=1):
                pairwise_by_id[mid] = {
                    "status": "live",
                    "rank_in_group": pos,
                    "is_winner": pos == 1,
                    "margin": res["margin"],
                    "deciding": res["deciding"],
                    "group_size": res["size"],
                }
        else:
            for m in members:
                pairwise_by_id[m["id"]] = {
                    "status": "pending",
                    "rank_in_group": None,
                    "is_winner": False,
                    "margin": None,
                    "deciding": None,
                    "group_size": res["size"],
                }

    out = []
    for r, score in zip(ranked, scores):
        it = working.items[r["idx"]]
        pw = pairwise_by_id.get(
            str(r["idx"]),
            {
                "status": "none",
                "rank_in_group": None,
                "is_winner": False,
                "margin": None,
                "deciding": None,
                "group_size": None,
            },
        )
        blended = blend(score, pw["rank_in_group"], pw["group_size"])
        out.append(
            {
                "id": str(r["idx"]),
                "filename": it.filename,
                "set": it.set_name,
                "calib_label": it.calib_label,
                "user_label": it.user_label,
                "score": score,
                "blended_score": blended,
                "sharpness": round(it.feat["sharpness_log"], 2),
                "exposure_quality": round(it.feat["exposure_quality"], 3),
                "exposure_clipped": round(it.feat["exposure_clipped"], 3),
                "vision": it.vis,
                "group": group_of.get(str(r["idx"])),
                "pairwise": pw,
            }
        )
    out.sort(key=lambda d: (-d["score"], d["filename"]))
    # Round only for display: sorting on rounded scores created
    # 18 rounded-tie groups on the demo set (no metric impact there,
    # but raw-score order is the honest order).
    for d in out:
        d["score"] = round(d["score"], 3)
        d["blended_score"] = round(d["blended_score"], 3)
    # NOTE (measured 2026-10-28): sorting by the blended score REGRESSED
    # the global ranking — separation fell from +0.072 to +0.017 and
    # top-25 from 6 to 4, because pairwise winner accuracy on calibration
    # groups is ~52% (reject-avoid 11/21): the sharpness-first pairwise
    # criterion mismatches the user's moment-led taste, and many groups
    # are visually identical (margin 0.0), so forced choice is noise
    # there. The global sort therefore stays on the model score; the
    # pairwise result remains visible per group (winner badge, rank,
    # deciding clause) and `blended_score` is kept as an experimental
    # field only. Do not re-enable blended sorting without a pairwise
    # signal that beats chance on calibration groups first.
    working.ranked = out
    return out


def invalidate() -> None:
    if working is not None:
        working.ranked = None


def _people_items() -> list[tuple[str, Path, str]]:
    """Current working set as (id, source, sha) for the people pass.

    Deleted (trashed) photos are excluded here, so they never appear
    in /api/people, the detection job, or the separate dry-runs."""
    assert working is not None
    deleted = load_deleted_paths()
    out = []
    for idx, it in enumerate(working.items):
        if not it.sha:
            it.sha = features.file_sha256(it.source)
        if str(it.source) in deleted:
            continue
        out.append((str(idx), it.source, it.sha))
    return out


# ------------------------------------------------- separate into folders
RAW_EXTS = [".RW2", ".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF"]
RAW_EXT_SET = {e.lower() for e in RAW_EXTS}


class SeparateError(Exception):
    """The separate plan cannot be built; carries the message, an HTTP
    status, and any extra fields for the API response."""

    def __init__(self, message: str, status: int = 400, extra: dict | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra or {}


def _read_raw_dir() -> Path:
    """RAW folder from DEMO_DIR/source.json {"raw_dir": "..."}.

    Written by the preview-extraction step. Never guess a path: a
    missing file, a bad file, or a non-existent folder is an error
    telling the user the RAW folder is unknown.
    """
    src_file = DEMO_DIR / "source.json"
    if not src_file.exists():
        raise SeparateError(
            "RAW folder unknown — no source.json in the working-set folder. "
            "Run the preview-extraction step first so it can record the RAW folder."
        )
    try:
        data = json.loads(src_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise SeparateError(
            f"Cannot read source.json in the working-set folder: {exc}"
        )
    raw_dir = data.get("raw_dir") if isinstance(data, dict) else None
    if not raw_dir or not isinstance(raw_dir, str):
        raise SeparateError(
            "source.json in the working-set folder has no usable raw_dir — "
            "run the preview-extraction step first."
        )
    p = Path(raw_dir).expanduser()
    if not p.is_dir():
        raise SeparateError(f"RAW folder does not exist: {raw_dir}")
    return p


def _index_raw_dir(raw_dir: Path) -> dict[str, list[Path]]:
    """RAW files in raw_dir keyed by filename stem (lowercased).

    Flat scan only — the RAW folder is one folder of originals.
    Within a stem, files sort by the RAW_EXTS order above, then name,
    so the match is deterministic when several extensions share a stem.
    """
    ext_order = {e.lower(): i for i, e in enumerate(RAW_EXTS)}
    index: dict[str, list[Path]] = {}
    for p in raw_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in RAW_EXT_SET:
            continue
        index.setdefault(p.stem.lower(), []).append(p)
    for paths in index.values():
        paths.sort(key=lambda p: (ext_order.get(p.suffix.lower(), 99), p.name))
    return index


def _separate_plan(require_complete: bool = True) -> dict:
    """Match every working-set item with a detection result to its RAW original.

    with_people / without_people are lists of {id, preview, raw_src,
    has_person}; unmatched / no_result are preview filenames. With
    require_complete=True (dry-run), any item without a detection
    result raises SeparateError carrying the no-result list; with
    False (run), those items are reported as skipped instead.
    """
    global working
    if working is None:
        working = load_demo_set()
    raw_dir = _read_raw_dir()
    index = _index_raw_dir(raw_dir)
    parent = raw_dir.parent
    plan = {
        "raw_dir": str(raw_dir),
        "dest_with": str(parent / "with-people"),
        "dest_without": str(parent / "without-people"),
        "with_people": [],
        "without_people": [],
        "unmatched": [],
        "no_result": [],
    }
    for item_id, src, sha in _people_items():
        hit = people_pass.cached(sha)
        if hit is None:
            plan["no_result"].append(src.name)
            continue
        raws = index.get(src.stem.lower())
        if not raws:
            plan["unmatched"].append(src.name)
            continue
        entry = {
            "id": item_id,
            "preview": src.name,
            "raw_src": str(raws[0]),
            "has_person": bool(hit["has_person"]),
        }
        (plan["with_people"] if entry["has_person"] else plan["without_people"]).append(entry)
    if require_complete and plan["no_result"]:
        raise SeparateError(
            "People detection is not complete — "
            f"{len(plan['no_result'])} photo(s) have no result yet. "
            "Run Detect people first.",
            status=400,
            extra={"no_result": plan["no_result"]},
        )
    return plan


def _dedup_dest_name(dest_dir: Path, name: str, used: set[str]) -> str:
    """Dest filename that never overwrites: the same _2/_3 rule as
    /api/export, plus a check against files already in the folder
    (export's dirs are always fresh; separate's dest folders may
    already hold files from an earlier run)."""
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    n = 2
    while candidate in used or (dest_dir / candidate).exists():
        candidate = f"{stem}_{n}{suffix}"
        n += 1
    used.add(candidate)
    return candidate


def _move_raw_files(entries: list[dict], dest_for) -> tuple[list[dict], list[str]]:
    """Move matched RAW files; shared by people-separate and best-shots.

    entries: [{preview, raw_src, ...}] where raw_src is the current
    location. dest_for(entry) -> destination directory Path. Dedupes
    dest names with _2/_3 (also against files already in the folder).
    A RAW that is already gone (same stem matched by two previews —
    the first move took it) is reported as skipped_missing, not an
    error. Nothing is deleted. Returns (moves, skipped_missing);
    each move is its entry plus raw_dest.
    """
    moves: list[dict] = []
    skipped_missing: list[str] = []
    used: dict[Path, set[str]] = {}
    for entry in entries:
        dest_dir = dest_for(entry)
        dest_dir.mkdir(parents=True, exist_ok=True)
        used.setdefault(dest_dir, set())
        src = Path(entry["raw_src"])
        if not src.exists():
            skipped_missing.append(entry["preview"])
            continue
        dest_name = _dedup_dest_name(dest_dir, src.name, used[dest_dir])
        dest = dest_dir / dest_name
        shutil.move(str(src), str(dest))
        moves.append({**entry, "raw_dest": str(dest)})
    return moves, skipped_missing


def _write_move_log(log: dict, log_filename: str, copy_dirs: list[Path],
                    copy_name: str) -> Path:
    """Write a move log to data/ plus a copy in each dest folder."""
    log_path = DATA_DIR / log_filename
    log_path.write_text(json.dumps(log, indent=1))
    for dest_dir in copy_dirs:
        (dest_dir / copy_name).write_text(json.dumps(log, indent=1))
    return log_path


# ---------------------------------------------------------------- API
class LabelBody(BaseModel):
    id: str
    label: str | None  # "keep" | "reject" | "maybe" | null (clear)


class ExportBody(BaseModel):
    ids: list[str] | None = None


class KeepSetBody(BaseModel):
    n: int = 20


@app.get("/api/state")
def state():
    ranked = ensure_ranked()
    assert working is not None
    vision_status = "pending_api_key"
    if any(r["vision"]["status"] == "live" for r in ranked):
        vision_status = "live"
    elif any(r["vision"]["status"] == "pending_configuration" for r in ranked):
        vision_status = "pending_configuration"
    return {
        "set_name": working.name,
        "total": len(ranked),
        "keeps": sum(1 for r in ranked if r["user_label"] == "keep"),
        "rejects": sum(1 for r in ranked if r["user_label"] == "reject"),
        "maybes": sum(1 for r in ranked if r["user_label"] == "maybe"),
        "vision_status": vision_status,
        "vision_note": ranked[0]["vision"]["note"] if ranked else "",
        "model_trained": model.trained,
        "model_n_train": model.n_train,
        "model_weights": model.weights(),
        "pairwise_learned": {
            "weights": pairwise_learned_weights,
            "evaluation": PAIRWISE_LEARNED_EVAL,
        },
        "dupe_groups": len(working.groups),
        "dupe_frames": sum(len(g) for g in working.groups),
        "people": people_pass.status_for(_people_items()),
        "pairwise_groups_total": len(working.groups),
        "pairwise_groups_live": len(
            {r["group"] for r in ranked if r["pairwise"]["status"] == "live"}
        ),
        "pairwise_groups_pending": len(
            {r["group"] for r in ranked if r["pairwise"]["status"] == "pending"}
        ),
    }


@app.get("/api/rank")
def rank():
    return {"items": ensure_ranked()}


@app.post("/api/people/start")
def people_start():
    """Start background people detection over the current working set.

    Returns immediately with the job status; the run proceeds in a
    worker thread (~0.6 s/photo on CPU) and never blocks the UI.
    Poll GET /api/people for progress. Fully offline — no API key.
    """
    global working
    if working is None:
        working = load_demo_set()
    return people_pass.start(_people_items())


@app.get("/api/people")
def people():
    """People flags for the current working set + job status.

    Undetected photos report has_person null (never a guessed value);
    the job status says whether a run is idle, running, done, or
    unavailable (e.g. mediapipe/model missing).
    """
    global working
    if working is None:
        working = load_demo_set()
    items = _people_items()
    status = people_pass.status_for(items)
    out = []
    for item_id, _src, sha in items:
        hit = people_pass.cached(sha)
        it = working.items[int(item_id)]
        out.append(
            {
                "id": item_id,
                "filename": it.filename,
                "has_person": hit["has_person"] if hit else None,
                "person_count": hit["person_count"] if hit else None,
                "max_conf": hit["max_conf"] if hit else None,
                "detected": hit is not None,
            }
        )
    return {**status, "items": out}


class PeopleOverrideBody(BaseModel):
    id: str
    has_person: bool


@app.post("/api/people/override")
def people_override(body: PeopleOverrideBody):
    """Flip one photo's people result (manual correction).

    Updates the disk cache, so GET /api/people, the filter counts,
    and the separate dry-run reflect the change immediately.
    """
    global working
    if working is None:
        working = load_demo_set()
    it = working.by_id(body.id)
    if it is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    if not it.sha:
        it.sha = features.file_sha256(it.source)
    data = people_pass.override(it.sha, body.has_person)
    return {
        "ok": True,
        "id": body.id,
        "filename": it.filename,
        "has_person": data["has_person"],
        "person_count": data["person_count"],
        "max_conf": data["max_conf"],
    }


@app.get("/api/people/separate/dry-run")
def people_separate_dry_run():
    """What the separate run would move. Moves nothing.

    Requires detection done for every item; otherwise 400 with the
    no-result list. with_people / without_people are preview filenames
    (the RAW originals are what would move).
    """
    global working
    if working is None:
        working = load_demo_set()
    try:
        plan = _separate_plan(require_complete=True)
    except SeparateError as exc:
        return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)
    return {
        "with_people": [e["preview"] for e in plan["with_people"]],
        "without_people": [e["preview"] for e in plan["without_people"]],
        "unmatched": plan["unmatched"],
        "no_result": plan["no_result"],
        "dest_with": plan["dest_with"],
        "dest_without": plan["dest_without"],
        "counts": {
            "with_people": len(plan["with_people"]),
            "without_people": len(plan["without_people"]),
            "unmatched": len(plan["unmatched"]),
            "no_result": len(plan["no_result"]),
        },
    }


class SeparateRunBody(BaseModel):
    confirmed: bool = False


@app.post("/api/people/separate/run")
def people_separate_run(body: SeparateRunBody):
    """Move matched RAW originals into with-people / without-people
    folders next to the RAW folder. Only the RAWs move — previews
    stay where they are. Refuses without {"confirmed": true}.
    Items with no result or no matching RAW are skipped and reported.
    Nothing is deleted; every move is written to a log in data/ and
    a copy in each destination folder.
    """
    if not body.confirmed:
        return JSONResponse(
            {
                "error": "refusing to move files — review the dry-run first, "
                'then call with {"confirmed": true}'
            },
            status_code=400,
        )
    global working
    if working is None:
        working = load_demo_set()
    try:
        plan = _separate_plan(require_complete=False)
    except SeparateError as exc:
        return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest_with = Path(plan["dest_with"])
    dest_without = Path(plan["dest_without"])
    entries = plan["with_people"] + plan["without_people"]
    raw_moves, skipped_missing = _move_raw_files(
        entries, lambda e: dest_with if e["has_person"] else dest_without
    )
    moves = [
        {
            "preview": m["preview"],
            "raw_src": m["raw_src"],
            "raw_dest": m["raw_dest"],
            "has_person": m["has_person"],
        }
        for m in raw_moves
    ]
    log = {
        "timestamp": ts,
        "raw_dir": plan["raw_dir"],
        "dest_with": plan["dest_with"],
        "dest_without": plan["dest_without"],
        "moves": moves,
        "skipped_no_result": plan["no_result"],
        "skipped_unmatched": plan["unmatched"],
        "skipped_missing": skipped_missing,
        "counts": {
            "moved_with": sum(1 for m in moves if m["has_person"]),
            "moved_without": sum(1 for m in moves if not m["has_person"]),
            "skipped_no_result": len(plan["no_result"]),
            "skipped_unmatched": len(plan["unmatched"]),
            "skipped_missing": len(skipped_missing),
        },
    }
    log_path = _write_move_log(
        log, f"separate-log-{ts}.json",
        [dest_with, dest_without], f"_move-log-{ts}.json",
    )
    return {
        "ok": True,
        "moved_with": log["counts"]["moved_with"],
        "moved_without": log["counts"]["moved_without"],
        "skipped_no_result": plan["no_result"],
        "skipped_unmatched": plan["unmatched"],
        "skipped_missing": skipped_missing,
        "dest_with": plan["dest_with"],
        "dest_without": plan["dest_without"],
        "log": str(log_path),
        "moves": moves,
    }


# ---------------------------------------------------------------- best shots
BEST_TOP_N = (10, 25, 50, 100)


def _best_picks(top_n: int) -> tuple[list[dict], list[dict]]:
    """One best shot per similar-group, in global rank order.

    Walks ensure_ranked() top-down (same ranking as /api/rank); the
    first photo of each dHash dupe group is a pick, the rest of that
    group are skipped dupes. Ungrouped photos pass through. The user
    filters/rejects picks in the UI before any move. Returns
    (picks, skipped_dupes); each pick carries score, sharpness,
    exposure_quality and group_size ("best of N similar").
    """
    ranked = ensure_ranked()
    assert working is not None
    group_size = {}
    for grp in working.groups:
        for m in grp:
            group_size[m] = len(grp)
    picks, skipped = [], []
    seen_groups: set[int] = set()
    for r in ranked:
        g = r["group"]
        if g is not None:
            if g in seen_groups:
                skipped.append(
                    {
                        "id": r["id"],
                        "filename": r["filename"],
                        "group_size": group_size.get(r["id"], 1),
                    }
                )
                continue
            seen_groups.add(g)
        picks.append(
            {
                "id": r["id"],
                "filename": r["filename"],
                "score": r["score"],
                "sharpness": r["sharpness"],
                "exposure_quality": r["exposure_quality"],
                "group_size": group_size.get(r["id"], 1),
            }
        )
        if len(picks) >= top_n:
            break
    return picks, skipped


def _best_dest() -> tuple[Path, Path]:
    """(raw_dir, best-shots dest) — same raw_dir source as the people
    separation; the same clear error when the RAW folder is unknown."""
    raw_dir = _read_raw_dir()
    return raw_dir, raw_dir.parent / "best-shots"


@app.get("/api/best/dry-run")
def best_dry_run(top_n: int = 10):
    """What the best-shots run would move. Moves nothing.

    Picks are one best per dupe group (see _best_picks). `unmatched`
    lists picks with no RAW original to move — those are reported,
    never guessed at.
    """
    if top_n not in BEST_TOP_N:
        return JSONResponse(
            {"error": f"top_n must be one of {list(BEST_TOP_N)}"},
            status_code=400,
        )
    try:
        raw_dir, dest_dir = _best_dest()
    except SeparateError as exc:
        return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)
    picks, skipped = _best_picks(top_n)
    index = _index_raw_dir(raw_dir)
    unmatched = [
        p["filename"]
        for p in picks
        if Path(p["filename"]).stem.lower() not in index
    ]
    return {
        "picks": picks,
        "skipped_dupes": skipped,
        "unmatched": unmatched,
        "dest": str(dest_dir),
        "top_n": top_n,
    }


class BestRunBody(BaseModel):
    confirmed: bool = False
    top_n: int = 10
    excluded: list[str] | None = None


@app.post("/api/best/separate/run")
def best_separate_run(body: BestRunBody):
    """Move the best-shots picks' RAW originals into a best-shots
    folder next to the RAW folder. Picks are recomputed server-side
    (same rule as the dry-run); `excluded` ids — the picks the user
    rejected in the UI — are dropped before matching. Only the RAWs
    move; nothing is deleted. Refuses without {"confirmed": true}.
    """
    if not body.confirmed:
        return JSONResponse(
            {
                "error": "refusing to move files — review the dry-run first, "
                'then call with {"confirmed": true}'
            },
            status_code=400,
        )
    if body.top_n not in BEST_TOP_N:
        return JSONResponse(
            {"error": f"top_n must be one of {list(BEST_TOP_N)}"},
            status_code=400,
        )
    excluded = set(body.excluded or [])
    try:
        raw_dir, dest_dir = _best_dest()
    except SeparateError as exc:
        return JSONResponse({"error": exc.message, **exc.extra}, status_code=exc.status)
    picks, skipped = _best_picks(body.top_n)
    index = _index_raw_dir(raw_dir)
    entries, unmatched = [], []
    for p in picks:
        if p["id"] in excluded:
            continue
        raws = index.get(Path(p["filename"]).stem.lower())
        if not raws:
            unmatched.append(p["filename"])
            continue
        entries.append(
            {
                "id": p["id"],
                "preview": p["filename"],
                "raw_src": str(raws[0]),
                "score": p["score"],
            }
        )
    ts = time.strftime("%Y%m%d-%H%M%S")
    raw_moves, skipped_missing = _move_raw_files(entries, lambda e: dest_dir)
    moves = [
        {
            "id": m["id"],
            "preview": m["preview"],
            "raw_src": m["raw_src"],
            "raw_dest": m["raw_dest"],
            "score": m["score"],
        }
        for m in raw_moves
    ]
    log = {
        "timestamp": ts,
        "raw_dir": str(raw_dir),
        "dest": str(dest_dir),
        "top_n": body.top_n,
        "moves": moves,
        "excluded": sorted(excluded),
        "skipped_dupes": skipped,
        "skipped_unmatched": unmatched,
        "skipped_missing": skipped_missing,
        "counts": {
            "moved": len(moves),
            "excluded": len(excluded),
            "skipped_dupes": len(skipped),
            "skipped_unmatched": len(unmatched),
            "skipped_missing": len(skipped_missing),
        },
    }
    log_path = _write_move_log(
        log, f"best-shots-log-{ts}.json",
        [dest_dir], f"_best-shots-log-{ts}.json",
    )
    return {
        "ok": True,
        "moved": len(moves),
        "excluded": sorted(excluded),
        "skipped_unmatched": unmatched,
        "skipped_missing": skipped_missing,
        "dest": str(dest_dir),
        "log": str(log_path),
        "moves": moves,
    }


# ---------------------------------------------------------------- source (feed any folder)
class SourceSetBody(BaseModel):
    demo_dir: str


@app.get("/api/source")
def source_get():
    """Current working-set folder, without triggering a rank.

    total = manifest items; visible = items still shown (deleted
    photos excluded)."""
    global working
    if working is None:
        working = load_demo_set()
    return {
        "demo_dir": str(DEMO_DIR),
        "set_name": working.name,
        "total": len(working.items),
        "visible": len(ensure_ranked()),
        "manifest": (DEMO_DIR / "manifest.csv").is_file(),
        "persisted": WORKING_DIR_FILE.exists(),
    }


@app.post("/api/source/set")
def source_set(body: SourceSetBody):
    """Switch the working set to an already-ingested folder (see
    tools/ingest.py). The folder must contain manifest.csv, else 400
    with a clear message. No restart: the in-memory rank is cleared,
    the manifest is re-read, and people-detection state resets for
    the new set. The choice is persisted to data/working-dir.json so
    it survives restarts."""
    global DEMO_DIR, working
    p = Path(body.demo_dir).expanduser()
    if not p.is_dir():
        return JSONResponse(
            {"error": f"folder does not exist: {body.demo_dir}"}, status_code=400
        )
    if not (p / "manifest.csv").is_file():
        return JSONResponse(
            {
                "error": f"no manifest.csv in {body.demo_dir} — run "
                "tools/ingest.py on a photo folder first"
            },
            status_code=400,
        )
    DEMO_DIR = p
    WORKING_DIR_FILE.write_text(json.dumps({"demo_dir": str(p)}, indent=1))
    working = load_demo_set()
    people_pass.reset()
    return {
        "demo_dir": str(DEMO_DIR),
        "set_name": working.name,
        "total": len(working.items),
        "persisted": True,
    }


# ---------------------------------------------------------------- similar cleanup (delete to Trash)
def _trash_dir() -> Path:
    """OS Trash folder: macOS ~/.Trash, Linux ~/.local/share/Trash/files."""
    if sys.platform == "darwin":
        return Path.home() / ".Trash"
    base = Path.home() / ".local" / "share" / "Trash"
    (base / "files").mkdir(parents=True, exist_ok=True)
    (base / "info").mkdir(parents=True, exist_ok=True)
    return base / "files"


def _move_to_trash(src: Path) -> Path:
    """Move a file to the OS Trash — never a permanent delete.

    Dest names are deduped (_2/_3). On Linux a .trashinfo file is
    written alongside so file managers can restore it.
    """
    trash = _trash_dir()
    trash.mkdir(parents=True, exist_ok=True)
    dest_name = _dedup_dest_name(trash, src.name, set())
    dest = trash / dest_name
    shutil.move(str(src), str(dest))
    if sys.platform != "darwin":
        info_dir = trash.parent / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        (info_dir / f"{dest_name}.trashinfo").write_text(
            f"[Trash Info]\nPath={src}\nDeletionDate={stamp}\n"
        )
    return dest


def _delete_source_for(it: Item) -> tuple[Path, Path | None]:
    """The file to trash for one photo, plus a symlink to unlink after
    (or None).

    - Matched RAW in raw_dir? Trash the RAW; the preview stays.
    - Else the photo file itself. If it is a symlink, trash its
      TARGET — but only when the target is this photo's recorded
      original (the manifest `source` column), and unlink the link
      itself afterwards. A symlink whose target is not the recorded
      original is refused: trashing it could delete the wrong file.
    - A regular file must live inside the working set or the RAW
      folder, else it is refused (delete it by hand).
    A preview file is never trashed on its own while its RAW exists.
    """
    raw_dir: Path | None = None
    try:
        raw_dir = _read_raw_dir()
        index = _index_raw_dir(raw_dir)
        raws = index.get(it.source.stem.lower())
        if raws:
            return raws[0], None
    except SeparateError:
        raw_dir = None
    src = it.source
    if src.is_symlink():
        target = src.resolve()
        recorded = Path(it.manifest_source).expanduser() if it.manifest_source else None
        if recorded is not None and target == recorded.resolve():
            return target, src
        raise SeparateError(
            f"cannot delete {src.name}: its link target is not the "
            "recorded original — delete it by hand"
        )
    resolved = src.resolve()
    allowed = [DEMO_DIR.resolve()]
    if raw_dir is not None:
        allowed.append(raw_dir.resolve())
    if not any(resolved == a or a in resolved.parents for a in allowed):
        raise SeparateError(
            f"cannot delete {src.name}: it lives outside the working "
            "set — delete it by hand"
        )
    return src, None


def _append_delete_log(entries: list[dict]) -> Path:
    """One delete-log file per delete operation, in data/."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = DATA_DIR / f"delete-log-{ts}.json"
    log_path.write_text(
        json.dumps({"timestamp": ts, "entries": entries}, indent=1)
    )
    return log_path


def _delete_one(item_id: str, it: Item, write_log: bool = True) -> dict:
    """Trash one photo's source file and record the deletion.

    The photo's source path is added to data/deleted.json, so it is excluded
    from /api/rank, /api/groups, /api/people and the separation
    dry-runs from then on. A source file that is already gone (its
    RAW was moved by an earlier separation) is recorded with
    trash_dest null and a note — the photo is still hidden, and
    nothing crashes. `restored_from` is null: restore is a manual
    move back from the OS Trash (the log names the exact trash
    location); no automatic restore is built.
    """
    if not it.sha:
        it.sha = features.file_sha256(it.source)
    src, extra_unlink = _delete_source_for(it)
    if src.exists() or src.is_symlink():
        trash_dest = _move_to_trash(src)
        note = ""
    else:
        trash_dest = None
        note = "source file was already gone"
    if extra_unlink is not None:
        extra_unlink.unlink(missing_ok=True)
    entry = {
        "id": item_id,
        "filename": it.filename,
        "src": str(src),
        "trash_dest": str(trash_dest) if trash_dest else None,
        "restored_from": None,
        "note": note,
    }
    if write_log:
        _append_delete_log([entry])
    deleted = load_deleted_paths()
    deleted.add(str(it.source))
    save_deleted_paths(deleted)
    invalidate()
    return entry


class DeleteBody(BaseModel):
    id: str


@app.post("/api/dupes/delete")
def dupes_delete(body: DeleteBody):
    """Move one photo's source file to the OS Trash (never permanent
    delete). The photo then disappears from the rank, groups, people
    lists and dry-runs. The log entry names the exact trash location
    for a manual restore."""
    global working
    if working is None:
        working = load_demo_set()
    it = working.by_id(body.id)
    if it is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    try:
        entry = _delete_one(body.id, it)
    except SeparateError as exc:
        return JSONResponse({"error": exc.message}, status_code=exc.status)
    return {"ok": True, **entry}


class DeleteGroupBody(BaseModel):
    group_id: int
    keep_id: str
    confirmed: bool = False


@app.post("/api/dupes/delete-group")
def dupes_delete_group(body: DeleteGroupBody):
    """Delete every non-keeper member of one dupe group (same Trash +
    log mechanism as a single delete, one log entry per file, one
    log file for the operation). Requires {"confirmed": true} and a
    keep_id that is a member of the group."""
    if not body.confirmed:
        return JSONResponse(
            {
                "error": "refusing to delete — review the group first, "
                'then call with {"confirmed": true}'
            },
            status_code=400,
        )
    global working
    if working is None:
        working = load_demo_set()
    groups = working.groups if working else []
    if body.group_id < 0 or body.group_id >= len(groups):
        return JSONResponse({"error": "unknown group"}, status_code=400)
    members = groups[body.group_id]
    if body.keep_id not in members:
        return JSONResponse(
            {"error": "keep_id is not a member of this group"}, status_code=400
        )
    entries = []
    for mid in members:
        if mid == body.keep_id:
            continue
        it = working.by_id(mid)
        if it is None:
            continue
        try:
            entries.append(_delete_one(mid, it, write_log=False))
        except SeparateError as exc:
            entries.append(
                {"id": mid, "filename": it.filename, "error": exc.message}
            )
    # One log file for the whole operation.
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = DATA_DIR / f"delete-log-{ts}.json"
    log_path.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "group_id": body.group_id,
                "kept": body.keep_id,
                "entries": entries,
            },
            indent=1,
        )
    )
    return {
        "ok": True,
        "kept": body.keep_id,
        "deleted": entries,
        "log": str(log_path),
    }


@app.post("/api/load-demo")
def load_demo():
    """Back to the built-in demo set (also clears a persisted
    source choice)."""
    global DEMO_DIR, working
    DEMO_DIR = DEFAULT_DEMO_DIR
    WORKING_DIR_FILE.unlink(missing_ok=True)
    working = load_demo_set()
    people_pass.reset()
    return {"set_name": working.name, "total": len(working.items)}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    global working
    staged: list[tuple[Path, str]] = []
    for up in files:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(up.filename or "upload").name)
        # uuid suffix: a second-resolution timestamp alone lets two
        # same-name files in one request overwrite each other.
        dest = UPLOAD_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
        staged.append((dest, up.filename or dest.name))
    # Validate every staged file: reject the whole batch with 400 if any
    # file is not a readable image (previously a text file returned 200
    # and then broke /api/state and /api/rank with 500s).
    bad: list[str] = []
    for dest, name in staged:
        try:
            with Image.open(dest) as im:
                im.verify()
        except Exception:
            bad.append(name)
    if bad:
        for dest, _ in staged:
            dest.unlink(missing_ok=True)
        return JSONResponse({"error": "not an image", "files": bad}, status_code=400)
    items = [Item(dest, "upload", None) for dest, _ in staged]
    working = WorkingSet(f"upload batch ({len(items)} frames)", items)
    _apply_persisted_labels(working)
    people_pass.reset()
    return {"set_name": working.name, "total": len(items)}


@app.post("/api/label")
def label(body: LabelBody):
    if working is None:
        return JSONResponse({"error": "no working set"}, status_code=400)
    it = working.by_id(body.id)
    if it is None or body.label not in ("keep", "reject", "maybe", None):
        return JSONResponse({"error": "bad id or label"}, status_code=400)
    it.user_label = body.label
    persisted = load_persisted_labels()
    if body.label is None:
        persisted.pop(it.sha, None)
    else:
        persisted[it.sha] = body.label
    save_persisted_labels(persisted)
    invalidate()
    return {"ok": True, "id": body.id, "label": body.label}


@app.get("/api/groups")
def groups():
    ranked = ensure_ranked()
    by_id = {r["id"]: r for r in ranked}
    out = []
    for grp in working.groups if working else []:
        members = [dict(by_id[m]) for m in grp if m in by_id]
        if members:
            # Suggested keeper: the highest-score member of the group.
            keeper = max(members, key=lambda m: m["score"])
            for m in members:
                m["is_suggested_keeper"] = m["id"] == keeper["id"]
        out.append(members)
    return {"groups": out}


@app.get("/api/thumb/{item_id}")
def thumb(item_id: str):
    if working is None:
        return JSONResponse({"error": "no working set"}, status_code=400)
    it = working.by_id(item_id)
    if it is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    if not it.sha:
        it.sha = features.file_sha256(it.source)
    dest = THUMB_DIR / f"{it.sha}.jpg"
    if not dest.exists():
        features.make_thumb(it.source, dest)
    return FileResponse(dest, media_type="image/jpeg")


@app.get("/api/image/{item_id}")
def image(item_id: str):
    if working is None:
        return JSONResponse({"error": "no working set"}, status_code=400)
    it = working.by_id(item_id)
    if it is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    return FileResponse(it.source, media_type="image/jpeg")


@app.post("/api/export")
def export(body: ExportBody):
    ranked = ensure_ranked()
    assert working is not None
    by_id = {r["id"]: r for r in ranked}
    if body.ids is not None:
        chosen = [by_id[i] for i in body.ids if i in by_id]
        mode = "selection"
    else:
        chosen = [r for r in ranked if r["user_label"] == "keep"]
        mode = "user keeps"
        if not chosen:
            chosen = ranked[:20]
            mode = "top 20 by rank (no user keeps yet)"
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = EXPORT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    names = []
    used: set[str] = set()
    for r in chosen:
        src = working.items[int(r["id"])].source
        # Dedup dest names: two chosen frames sharing a basename must
        # not silently overwrite each other (count said N, files said N-1).
        dest_name = src.name
        if dest_name in used:
            n = 2
            while f"{src.stem}_{n}{src.suffix}" in used:
                n += 1
            dest_name = f"{src.stem}_{n}{src.suffix}"
        used.add(dest_name)
        dest = out_dir / dest_name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        names.append(dest_name)
    # Contact sheet: 4-column montage of keeper thumbs with filenames.
    from PIL import Image, ImageDraw

    cols = 4
    thumb_w, thumb_h, label_h = 320, 240, 28
    rows = (len(chosen) + cols - 1) // cols or 1
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(chosen):
        src = working.items[int(r["id"])].source
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + label_h)
            sheet.paste(im, (x + (thumb_w - im.width) // 2, y))
            draw.text((x + 4, y + thumb_h + 4), r["filename"][:40], fill="black")
    sheet_path = out_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, "JPEG", quality=88)
    return {
        "export_dir": str(out_dir),
        "mode": mode,
        "count": len(chosen),
        "files": names,
        "contact_sheet": f"/api/export-file/{ts}/contact_sheet.jpg",
    }


@app.get("/api/export-file/{ts}/{filename}")
def export_file(ts: str, filename: str):
    # ts must be exactly the format export() generates; anything else
    # (e.g. "..") is a traversal attempt, not an export dir.
    if not re.fullmatch(r"\d{8}-\d{6}", ts):
        return JSONResponse({"error": "not found"}, status_code=404)
    path = EXPORT_DIR / ts / filename
    if not path.exists() or path.parent.resolve() != (EXPORT_DIR / ts).resolve():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/evaluate")
def evaluate():
    """Live evaluation harness. In-sample metrics are recomputed from
    the current ranking on every call; held-out metrics come from fresh
    deterministic stratified splits fit on train frames only and scored
    on test frames only. The two are labeled separately and must never
    be conflated: in-sample separation is fit on the same labels the
    model trained on — it is NOT generalization (both models score
    ~0.50 on unseen frames). This endpoint only reads; it never
    mutates the global model, labels, or ranking."""
    import random

    ranked = ensure_ranked()
    assert working is not None

    def train_lab(r):
        if r["user_label"] == "keep":
            return 1
        if r["user_label"] == "reject":
            return 0
        if r["user_label"] == "maybe":
            return None
        if r["calib_label"] == "keeper":
            return 1
        if r["calib_label"] == "reject":
            return 0
        return None

    train = [(r, train_lab(r)) for r in ranked]
    train = [(r, lab) for r, lab in train if lab is not None]
    keep_scores = [r["score"] for r, lab in train if lab == 1]
    rej_scores = [r["score"] for r, lab in train if lab == 0]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    keep_mean, rej_mean = mean(keep_scores), mean(rej_scores)
    in_sample = {
        "n_labels": len(train),
        "keeper_mean": round(keep_mean, 3) if keep_mean is not None else None,
        "reject_mean": round(rej_mean, 3) if rej_mean is not None else None,
        "separation": round(keep_mean - rej_mean, 3)
        if keep_mean is not None and rej_mean is not None
        else None,
        "top25_keepers": len(
            [r for r in ranked[:25] if r["calib_label"] == "keeper"]
        ),
        "top50_keepers": len(
            [r for r in ranked[:50] if r["calib_label"] == "keeper"]
        ),
        "label_agreement": round(
            sum(1 for r, lab in train if (1 if r["score"] > 0.5 else 0) == lab)
            / len(train),
            3,
        )
        if train
        else None,
    }

    held_out = None
    keep_idx = [i for i, (_, lab) in enumerate(train) if lab == 1]
    rej_idx = [i for i, (_, lab) in enumerate(train) if lab == 0]
    if len(keep_idx) >= 5 and len(rej_idx) >= 5:
        rng = random.Random(20261028)
        accs = []
        for _ in range(20):
            ks = keep_idx[:]
            rs = rej_idx[:]
            rng.shuffle(ks)
            rng.shuffle(rs)
            nkt = max(1, len(ks) // 5)
            nrt = max(1, len(rs) // 5)
            test = ks[:nkt] + rs[:nrt]
            tr = ks[nkt:] + rs[nrt:]

            def vec(i):
                it = working.items[int(train[i][0]["id"])]
                return feature_vector(it.feat, it.vis)

            m = TasteModel()
            m.fit([vec(i) for i in tr], [train[i][1] for i in tr])
            if not m.trained:
                continue
            smap = dict(zip(test, m.rank_scores([vec(i) for i in test])))
            pairs = [(smap[k], smap[r]) for k in ks[:nkt] for r in rs[:nrt]]
            accs.append(
                sum(
                    1.0 if k > r else 0.5 if k == r else 0.0 for k, r in pairs
                )
                / len(pairs)
            )
        if accs:
            mu = sum(accs) / len(accs)
            var = sum((a - mu) ** 2 for a in accs) / len(accs)
            held_out = {
                "splits": len(accs),
                "pair_accuracy": round(mu, 3),
                "sem": round((var**0.5) / (len(accs) ** 0.5), 3),
                "note": "Fresh stratified frame-level splits (80/20), fit on "
                "train frames only, scored on test frames only. "
                "Deterministic seed.",
            }
    return {
        "in_sample": in_sample,
        "held_out": held_out,
        "pairwise_learned_snapshot": PAIRWISE_LEARNED_EVAL,
        "honesty_note": "In-sample separation is fit on the same labels the "
        "model trained on — it is NOT generalization. Held-out accuracy "
        "on unseen frames is the generalization number.",
    }


@app.post("/api/keep-set")
def keep_set(body: KeepSetBody):
    """Coverage-aware keep-set: one winner per duplicate group first
    (highest model score in the group), then fill by score skipping
    frames whose group is already represented; if the request exceeds
    groups + singletons, allow second members by score. A 'scene' is a
    duplicate group or a standalone frame — that definition is stated
    in the UI so the counts are honest."""
    ranked = ensure_ranked()
    assert working is not None
    n = max(0, body.n)
    group_of: dict = {}
    for r in ranked:
        if r["group"] is not None:
            group_of.setdefault(r["group"], []).append(r)
    winners = [max(grp, key=lambda d: d["score"]) for grp in group_of.values()]
    winners.sort(key=lambda d: -d["score"])
    selected, seen = [], set()
    for r in winners:
        if len(selected) >= n:
            break
        selected.append(r)
        seen.add(r["id"])
    if len(selected) < n:
        represented = {r["group"] for r in selected if r["group"] is not None}
        for r in ranked:
            if len(selected) >= n:
                break
            if r["id"] in seen or (
                r["group"] is not None and r["group"] in represented
            ):
                continue
            selected.append(r)
            seen.add(r["id"])
            if r["group"] is not None:
                represented.add(r["group"])
    if len(selected) < n:
        for r in ranked:
            if len(selected) >= n:
                break
            if r["id"] not in seen:
                selected.append(r)
                seen.add(r["id"])
    groups_covered = len(
        {r["group"] for r in selected if r["group"] is not None}
    )
    scenes = groups_covered + sum(
        1 for r in selected if r["group"] is None
    )
    return {
        "n_requested": body.n,
        "count": len(selected),
        "ids": [r["id"] for r in selected],
        "groups_covered": groups_covered,
        "scenes": scenes,
        "members": selected,
    }


@app.get("/api/subject-crop/{item_id}")
def subject_crop(item_id: str):
    """Zoomed crop of the largest SAM 'person' box for one frame.

    Box geometry is NOT in the vision cache (only aggregates are), so
    the first view of a frame runs one SAM call via the connector
    helper and caches boxes + crop by file sha; repeat views are free.
    Cost: ~$2.50/1,000 images, about $0.0025 per newly cropped frame.
    Any failure (no subject, helper error) returns 404 JSON — the UI
    shows a fallback note, never a fake crop."""
    if working is None:
        return JSONResponse({"error": "no working set"}, status_code=400)
    it = working.by_id(item_id)
    if it is None:
        return JSONResponse({"error": "unknown id"}, status_code=404)
    if not it.sha:
        it.sha = features.file_sha256(it.source)
    box_dir = CACHE_DIR / "subject_boxes"
    crop_dir = CACHE_DIR / "subject_crops"
    box_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{it.sha}.jpg"
    if crop_path.exists():
        return FileResponse(crop_path, media_type="image/jpeg")
    boxes, sent_w, sent_h = None, None, None
    box_path = box_dir / f"{it.sha}.json"
    if box_path.exists():
        try:
            meta = json.loads(box_path.read_text())
            boxes, sent_w, sent_h = (
                meta["boxes"],
                meta["sent_width"],
                meta["sent_height"],
            )
        except (json.JSONDecodeError, KeyError):
            boxes = None
    if boxes is None:
        if not SAM_SCORE_HELPER.is_file():
            return JSONResponse({"error": "sam helper missing"}, status_code=404)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SAM_SCORE_HELPER),
                    str(it.source),
                    "--prompt",
                    SAM_PROMPT,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            return JSONResponse({"error": "sam call failed"}, status_code=404)
        if proc.returncode != 0:
            return JSONResponse({"error": "sam call failed"}, status_code=404)
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
            boxes, sent_w, sent_h = (
                data["boxes"],
                data["sent_width"],
                data["sent_height"],
            )
        except (json.JSONDecodeError, KeyError, IndexError):
            return JSONResponse({"error": "sam parse failed"}, status_code=404)
        box_path.write_text(
            json.dumps(
                {"boxes": boxes, "sent_width": sent_w, "sent_height": sent_h}
            )
        )
    if not boxes:
        return JSONResponse({"error": "no subject"}, status_code=404)
    try:
        with Image.open(it.source) as im:
            im = im.convert("RGB")
            ow, oh = im.size
            sx, sy = ow / sent_w, oh / sent_h
            box = max(
                boxes,
                key=lambda b: (b["right"] - b["left"])
                * (b["bottom"] - b["top"]),
            )
            x0 = int(box["left"] * sx)
            y0 = int(box["top"] * sy)
            x1 = int(box["right"] * sx)
            y1 = int(box["bottom"] * sy)
            mx = int((x1 - x0) * 0.2)
            my = int((y1 - y0) * 0.2)
            x0 = max(0, x0 - mx)
            y0 = max(0, y0 - my)
            x1 = min(ow, x1 + mx)
            y1 = min(oh, y1 + my)
            im.crop((x0, y0, x1, y1)).save(crop_path, "JPEG", quality=88)
    except Exception:
        return JSONResponse({"error": "crop failed"}, status_code=404)
    return FileResponse(crop_path, media_type="image/jpeg")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")
