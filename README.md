# Taste

AI photo-culling assistant that shows its work.
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
- **Keys**: `←`/`→` navigate, `P`/`L` keep, `X`/`J` reject, `U`/`K` clear,
  `M` maybe, `Z` 100% zoom. Keyboard flagging auto-advances.
- **Compare**: `dup` badges and the groups view show near-duplicates
  side by side with per-frame keep/reject, synced pan, and a delta table.
- **Subject strip**: SAM 3.1 subject crops, shown in the detail view and
  as a small badge on grid cards (subject sharpness is visible info; it
  did not lift the ranking).
- **Keep-set**: coverage-aware — "give me 20" returns one winner per
  duplicate group first, then fills by rank.
- **Reason cards + Maybe bucket**: deterministic reason cards per frame;
  Maybe excludes a frame from training (clearing restores it).
- **Evaluation** (`/api/evaluate`): in-sample vs held-out results shown
  explicitly, with the honesty note that the model does not generalize
  beyond the calibration labels.
- **Export**: copies the keep-set (user keeps, else top 20 by rank) to
  `data/exports/<timestamp>/` plus a `contact_sheet.jpg` montage.

## Vision pass — live (updated 2026-10-28)

`vision.py` scores frames with Muse Spark multimodal vision
(composition + moment) via the Meta Model API, and SAM 3.1 subject
masks (sharpness-by-subject). The $150 hackathon credits funded the
live passes; scores are cached on disk under `.cache/vision/<sha256>.json`,
so a frame is scored at most once. Never put a real key in code or git;
`.env` is gitignored (see `.env.example` for the placeholder).

## Layout

- `main.py` — FastAPI app + API (`/api/state`, `/api/rank`, `/api/label`,
  `/api/groups`, `/api/upload`, `/api/export`, `/api/thumb|image/{id}`,
  `/api/evaluate`, `/api/keep-set`, `/api/subject-crop/{id}`)
- `features.py`, `vision.py`, `taste_model.py`, `pairwise.py`,
  `pairwise_learn.py` — pipeline layers
- `static/` — `index.html`, `app.js`, `style.css`
- `data/` — uploads, `labels.json` (persisted user labels by file hash),
  exports (gitignored)
- `.cache/` — feature cache (`features.json`), thumbnails, vision cache
  (gitignored)

## Verified 2026-10-28 (final demo state)

Live vision pass complete: calibration keepers vs rejects separate by
+0.072 (keepers 0.536 vs rejects 0.464), with 6 keepers in the top 25 and
10 in the top 50 — a real but modest separation. Held-out accuracy is
about chance (~0.46–0.50), and a fresh-photo test of the
lowest-luminance-contrast rule failed (3/11 vs ~3.45 chance), so the demo
claims UX + honesty, not generalization. Three further ranking levers
were tried and did not move the ranking: SAM subject sharpness (no lift),
pairwise within-group re-rank (worse, reverted), and pairwise learning
from keep/reject overrides (0.500 held-out vs 0.502 current model,
experimental only, does not drive ranking). Bug-hunt fixes verified
live: same-name uploads no longer overwrite, non-image uploads are
rejected cleanly, the export traversal hole returns 404, and the
sticky-header scroll fix is intact. Labels, weights, top-25/top-50 and
35/35 pairwise coverage unchanged through the fixes.

## Verified 2026-10-26 (D1)

Server started; demo set ranked (152 frames); upload of 3 seed frames
worked; keep/reject labels refit and moved scores; 35 dupe groups found
(max size 6); export produced a keep folder + contact sheet; UI served 200
and `app.js` passes `node --check`. See the D1 report to the parent agent
for the honest caveat: with vision pending, calibration keepers vs rejects
separate only weakly (mean rank 0.506 vs 0.494) — the vision pass is the
missing signal, not a tuning problem in the heuristics.
