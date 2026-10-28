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
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import features
from taste_model import TasteModel, feature_vector
from vision import VisionPass

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
model = TasteModel()


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
            return self.items[int(item_id)]
        except (ValueError, IndexError):
            return None


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
    items: list[Item] = []
    with open(manifest, newline="") as f:
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
            it.feat = features.extract_features(it.source)
            feat_cache[it.sha] = it.feat
        it.vis = vision_pass.score(it.source).as_dict()
        vec = feature_vector(it.feat, it.vis)
        # Training labels: calibration labels + the user's own keep/reject.
        lab = None
        if it.user_label == "keep":
            lab = 1
        elif it.user_label == "reject":
            lab = 0
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
    scores = model.rank_scores([r["vec"] for r in ranked])

    hashes = {str(r["idx"]): working.items[r["idx"]].feat["dhash"] for r in ranked}
    group_list = features.group_near_dupes(hashes)
    group_of = {}
    for gi, grp in enumerate(group_list):
        for member in grp:
            group_of[member] = gi
    working.groups = group_list

    out = []
    for r, score in zip(ranked, scores):
        it = working.items[r["idx"]]
        out.append(
            {
                "id": str(r["idx"]),
                "filename": it.filename,
                "set": it.set_name,
                "calib_label": it.calib_label,
                "user_label": it.user_label,
                "score": round(score, 3),
                "sharpness": round(it.feat["sharpness_log"], 2),
                "exposure_quality": round(it.feat["exposure_quality"], 3),
                "exposure_clipped": round(it.feat["exposure_clipped"], 3),
                "vision": it.vis,
                "group": group_of.get(str(r["idx"])),
            }
        )
    out.sort(key=lambda d: (-d["score"], d["filename"]))
    working.ranked = out
    return out


def invalidate() -> None:
    if working is not None:
        working.ranked = None


# ---------------------------------------------------------------- API
class LabelBody(BaseModel):
    id: str
    label: str | None  # "keep" | "reject" | null (clear)


class ExportBody(BaseModel):
    ids: list[str] | None = None


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
        "vision_status": vision_status,
        "vision_note": ranked[0]["vision"]["note"] if ranked else "",
        "model_trained": model.trained,
        "model_n_train": model.n_train,
        "model_weights": model.weights(),
        "dupe_groups": len(working.groups),
        "dupe_frames": sum(len(g) for g in working.groups),
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
    items: list[Item] = []
    for up in files:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(up.filename or "upload").name)
        dest = UPLOAD_DIR / f"{int(time.time())}_{safe}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
        items.append(Item(dest, "upload", None))
    working = WorkingSet(f"upload batch ({len(items)} frames)", items)
    _apply_persisted_labels(working)
    return {"set_name": working.name, "total": len(items)}


@app.post("/api/label")
def label(body: LabelBody):
    if working is None:
        return JSONResponse({"error": "no working set"}, status_code=400)
    it = working.by_id(body.id)
    if it is None or body.label not in ("keep", "reject", None):
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
    if body.ids:
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
    for r in chosen:
        src = working.items[int(r["id"])].source
        dest = out_dir / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        names.append(src.name)
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
    path = EXPORT_DIR / ts / filename
    if not path.exists() or path.parent.resolve() != (EXPORT_DIR / ts).resolve():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(APP_DIR / "static" / "index.html")
