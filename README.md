# Taste — MVP scaffold (D1)

AI photo-culling assistant that learns the photographer's eye.
FastAPI backend + vanilla HTML/JS UI. Meta Global AI Developer Hackathon,
indie track. Build window Oct 26 – Nov 4, 2026; submission due Nov 4, 6 PM PT.

## Run

```sh
./run.sh
# → http://127.0.0.1:8000/
```

`run.sh` creates `.venv/` on first run and installs `requirements.txt`.
Or manually:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

## What it does

- **Demo set**: the built-in set is the user's own 152 Nepal frames in
  `../files/taste-photo-set/` (25 calibration keepers, 25 rejects,
  102 pool). It is *referenced*, never copied into `app/`.
- **Upload**: drag-drop a batch (or the file picker) to cull a new shoot.
- **Features** (`features.py`, real math): sharpness = Laplacian variance,
  exposure = luminance histogram stats (clipped fraction, mean/std),
  near-duplicates = hand-rolled 64-bit dHash, Hamming ≤ 8, union-find.
- **Taste model** (`taste_model.py`): logistic regression over
  [sharpness_log, exposure_quality, composition, moment], seeded with the
  50 calibration labels. Every keep/reject refits and re-ranks.
- **Keys**: `←`/`→` navigate, `J` reject, `K` clear, `L` keep.
- **Compare**: `dup` badges and the groups view show near-duplicates
  side by side with per-frame keep/reject.
- **Export**: copies the keep-set (user keeps, else top 20 by rank) to
  `data/exports/<timestamp>/` plus a `contact_sheet.jpg` montage.

## Vision pass — pending API key (read this)

`vision.py` defines the interface for two later calls: Muse Spark
multimodal vision (composition + moment) and SAM 3.1 subject masks
(sharpness-by-subject). Today it is **pending, not stubbed with fake
scores**:

- No `META_MODEL_API_KEY` in the environment → all vision scores are
  `None`, status `pending_api_key`, and the UI shows a banner:
  "vision pass pending API key". Ranking substitutes neutral 0.5 for
  composition/moment *inside the model only*; the UI labels them pending.
- Scores are cached on disk under `.cache/vision/<sha256>.json`, so a
  frame is scored at most once (budget guardrail for the $150 credits).
- When the key arrives, set `META_MODEL_API_KEY` (and
  `META_MODEL_API_URL` for the endpoint) — see `.env.example`. Never
  put a real key in code or git; `.env` is gitignored.

## Layout

- `main.py` — FastAPI app + API (`/api/state`, `/api/rank`, `/api/label`,
  `/api/groups`, `/api/upload`, `/api/export`, `/api/thumb|image/{id}`)
- `features.py`, `vision.py`, `taste_model.py` — pipeline layers
- `static/` — `index.html`, `app.js`, `style.css`
- `data/` — uploads, `labels.json` (persisted user labels by file hash),
  exports (gitignored)
- `.cache/` — feature cache (`features.json`), thumbnails, vision cache
  (gitignored)

## Verified 2026-10-26 (D1)

Server started; demo set ranked (152 frames); upload of 3 seed frames
worked; keep/reject labels refit and moved scores; 35 dupe groups found
(max size 6); export produced a keep folder + contact sheet; UI served 200
and `app.js` passes `node --check`. See the D1 report to the parent agent
for the honest caveat: with vision pending, calibration keepers vs rejects
separate only weakly (mean rank 0.506 vs 0.494) — the vision pass is the
missing signal, not a tuning problem in the heuristics.
