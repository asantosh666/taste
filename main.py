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
from taste_model import TasteModel, feature_vector
from vision import SAM_PROMPT, SAM_SCORE_HELPER, VisionPass

APP_DIR = Path(__file__).resolve().parent
DEMO_DIR = APP_DIR.parent / "files" / "taste-photo-set"
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
EXPORT_DIR = DATA_DIR / "exports"
CACHE_DIR = APP_DIR / ".cache"
THUMB_DIR = CACHE_DIR / "thumbs"
FEATURE_CACHE_FILE = CACHE_DIR / "features.json"
LABELS_FILE = DATA_DIR / "labels.json"

for d in (DATA_DIR, UPLOAD_DIR, EXPORT_DIR, CACHE_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Taste — AI photo culling that learns your eye")
vision_pass = VisionPass(CACHE_DIR / "vision")
pairwise_pass = PairwisePass(CACHE_DIR / "pairwise")
model = TasteModel()
# Pairwise-learned (Bradley-Terry) weights, computed on each refit for
# inspection only. Measured 2026-10-28 as a NON-RESULT (held-out pair
# accuracy 0.500 vs pointwise 0.502): these weights never drive ranking.
pairwise_learned_weights: dict | None = None


# ---------------------------------------------------------------- state
class Item:
    def __init__(self, source: Path, set_name: str, calib_label: str | None):
        self.source = source
        self.set_name = set_name
        self.calib_label = calib_label  # "keeper" | "reject" | None
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
    """Build the demo working set from the manifest. Read-only: no copies."""
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
            items.append(Item(src, row["set"], label))
    ws = WorkingSet("demo — your Nepal frames (152)", items)
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
    train_vecs, train_labels = [], []
    ranked = []
    for idx, it in enumerate(working.items):
        if not it.sha:
            it.sha = features.file_sha256(it.source)
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


@app.post("/api/load-demo")
def load_demo():
    global working
    working = load_demo_set()
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
        out.append([by_id[m] for m in grp if m in by_id])
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
