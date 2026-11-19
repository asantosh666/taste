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

## People split (merged into the submitted app 2026-11-18)

Tiled person detection, fully offline, no API key. `people.py` runs
MediaPipe EfficientDet-Lite0 (COCO `person`, conf 0.3) over the full
image plus 3×3 overlapping 50% crops (origins 0/25/50%). Results are
cached in `.cache/people/<sha256>.json` (has_person, person_count,
max_conf) — each photo is detected once, ~0.6 s/photo on CPU.

The model file `efficientdet_lite0.tflite` (~14 MB) is downloaded
automatically by `run.sh` on first start from the public MediaPipe model
bucket (`storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite`,
md5 `ca669b1df0ebe9a75ddcde57fe07a0d0`); it is NOT committed to the repo
(binary). If the download fails, the app still starts — people detection
just reports "unavailable" (see `people.py`).

- UI: **People: All / With people / Without people** filter with counts,
  **Detect people (offline)** button with progress, per-card 👤 people /
  no-people badge, People line in the detail view.
- API: `POST /api/people/start` (background run, returns immediately),
  `GET /api/people` (flags + job status), people summary inside
  `GET /api/state`.

Verified 2026-11-12/13 on the 50-photo pre-test sample vs corrected
ground truth (28 with people, 22 without): **accuracy 0.98, recall
1.00, TP 28, FP 1 (a sheep), FN 0, TN 21** — passes the ≥0.90/≥0.90
bar. Full demo set: 72 with people, 80 without (152 frames).
Untiled YOLOv8n on the same sample scored 0.88/0.81 and failed the
bar; tiled detection is what makes small/distant figures visible.
See `~/workspace/taste-culling-pretest/REPORT.md` and
`verify_copy_people.py` / `verify_copy_ui.py` there.

### Separate into folders

Moves the RAW originals into `with-people/` and `without-people/`
folders next to the RAW folder, using the detection results — so a
1,127-photo shoot gets split without hand-sorting.

- **source.json contract**: the preview-extraction step writes
  `source.json` into the working-set folder (the previews folder):
  `{"raw_dir": "/Volumes/LUMIX/1Nepal/pics"}`. The endpoints read
  the RAW folder from there and never guess a path. If `source.json`
  is missing, or `raw_dir` does not exist, the endpoints return a
  clear error saying the RAW folder is unknown.
- **Matching**: a preview `<stem>.jpg` matches a RAW with the same
  stem in `raw_dir`, extension `.RW2/.ARW/.DNG/.NEF/.CR2/.CR3/.RAF/.ORF`
  (case-insensitive, flat scan of that one folder). Only the RAW
  originals move; the previews stay where they are.
- **Flow**: click **Separate into folders** (enabled only once
  detection is done) → dry-run shows counts, destination paths, and
  the unmatched / no-result lists → **Move the RAW files** confirms
  and moves. The dry-run moves nothing; the run refuses without
  `{"confirmed": true}`. Items with no result or no matching RAW are
  skipped and listed, never moved.
- **API**: `POST /api/people/override` `{id, has_person}` flips one
  photo's result (also a "Flip to people / no people" button in the
  detail view — it updates the disk cache, so badges, filter counts
  and the dry-run change immediately); `GET
  /api/people/separate/dry-run`; `POST /api/people/separate/run`
  `{"confirmed": true}`.
- **Safety**: destination filenames are deduped like `/api/export`
  (`_2`, `_3`, …) and never overwrite — the check also covers files
  already in the destination folder. Nothing is deleted.
- **Undo**: every run writes a move log
  `data/separate-log-<timestamp>.json` and a copy
  `_move-log-<timestamp>.json` into each destination folder, listing
  every move `{preview, raw_src, raw_dest, has_person}` plus the
  skipped lists. There is no automatic undo — move a file back by
  hand using its log entry (each entry names the exact source and
  destination).

First load is slow on a cold cache: the pairwise pass makes one live
Muse call per duplicate group (~35 groups, ~40 s each, ~20 min total),
then it is cached. That is pre-existing app behaviour, not this feature.

## v3 additions — best shots, similar cleanup, feed any folder (2026-11-13; merged into the submitted app 2026-11-18)

All three reuse the separation move/dedup/log code (`_move_raw_files`,
`_write_move_log` in `main.py`). Nothing is ever permanently deleted;
RAWs move, previews stay.

### Best shots

Auto-score proposes, the user filters, the survivors move to a folder.

- **Ranking**: the same global ranking as `/api/rank` (sharpness +
  exposure only while vision is pending; held-out accuracy ~0.50, so
  the list is a proposal, not a verdict).
- **One best per similar-group**: walk the ranking top-down, take the
  first photo of each dHash dupe group, skip the rest of that group;
  ungrouped photos pass through.
- **API**: `GET /api/best/dry-run?top_n=10` (top_n in 10/25/50/100,
  default 10) returns `{picks, skipped_dupes, dest, top_n}`; `POST
  /api/best/separate/run {"confirmed": true, "top_n": 10,
  "excluded": [ids]}` recomputes the picks server-side, drops the
  excluded ids, moves matched RAWs only into
  `<parent-of-raw-dir>/best-shots`, log
  `data/best-shots-log-<ts>.json` + a copy `_best-shots-log-<ts>.json`
  in the dest folder. Unconfirmed → 400, moves nothing. Unknown
  `raw_dir` → the same clear "RAW folder unknown" error as people
  separation.
- **UI**: **Best shots** section next to the People bar — Top N
  selector, **Preview best shots**, pick list with score, sharpness,
  exposure and a "best of N similar" / "no duplicates" label, a
  **Reject ✕** (toggles to Restore) per pick, dry-run counts + dest,
  **Move best shots (N)** confirms.

### Similar cleanup — delete with a way back

- **`GET /api/groups`** now marks the suggested keeper of each group —
  the highest-score member — with `is_suggested_keeper: true` (exactly
  one per group).
- **`POST /api/dupes/delete {"id"}`**: moves that photo's SOURCE file
  to the OS Trash (macOS `~/.Trash`, Linux
  `~/.local/share/Trash/files`), never a permanent delete. The source
  file = the matched RAW when a `raw_dir` match exists, else the photo
  file itself. A preview is never trashed on its own while its RAW
  exists; a symlinked JPEG/PNG is only trashed when its target equals
  the manifest `source` (otherwise 400). Appends to
  `data/delete-log-<ts>.json` `{id, filename, src, trash_dest,
  restored_from}` and records the source path in `data/deleted.json`;
  deleted photos vanish from `/api/rank`, `/api/groups`, `/api/people`
  and the separation dry-runs thereafter. **Deleted is keyed by source
  path, not content sha** — byte-identical duplicates share a sha, and
  deleting one must not hide its twin (verified: deleting H leaves D,
  its byte-identical twin, in the ranking).
- **`POST /api/dupes/delete-group {"group_id", "keep_id",
  "confirmed": true}`**: deletes every non-keeper member of that dupe
  group (same Trash + log mechanism, one log file for the operation).
  Requires `confirmed: true` and a `keep_id` that is a member of the
  group, else 400.
- **Restore**: no automatic restore — the delete log names the exact
  Trash location for every file; move it back by hand from the OS
  Trash (`restored_from` stays null in the log). TIF-converted
  previews: deleting trashes the converted JPEG preview, the TIF
  original survives.
- **UI**: groups view shows the ★ suggested keeper badge, a **Delete**
  button per photo (inline "Move to Trash? [Yes, delete] [Cancel]"
  confirm), and per group a **Delete the rest (keep …)** button that
  shows a confirm panel (count + filenames) before posting. After a
  delete the photo disappears from the lists.

### Feed any folder — ingest + switch

- **`tools/ingest.py`**: `python tools/ingest.py <photo-folder>
  [--raw-dir <raw-dir>] [--out <working-dir>]`. Scans recursively for
  RAW + JPG/JPEG/PNG/TIF (skips `._` AppleDouble and dotfiles). RAWs
  → embedded JPEG preview via `exiftool -b -PreviewImage`, fallback
  `dcraw -e` (stdout, so no file lands next to the RAW), else warn +
  skip that file, never fail the whole run; extracted bytes must start
  `FF D8` and open in Pillow. JPG/PNG → symlinked into the working
  dir (never copied). TIF → converted to a JPEG preview via Pillow
  (the TIF stays put). Dest-name collisions get `_2` suffixes. Writes
  `manifest.csv` (columns `file,set,label,basis,source,width,height`;
  `set=shoot`, label blank) + `source.json` (`{"raw_dir": ...}` — the
  `--raw-dir` wins, else the photo folder itself when it holds RAWs,
  else **no `source.json`**, so separate/best-shots return the clear
  "RAW folder unknown" error). Never writes into the source folder,
  never modifies a source file. Prints a summary (counts, skipped
  files with reasons, the DEMO_DIR path to use). Needs `exiftool`
  and/or `dcraw` on PATH for RAW previews.
- **Working-set switch (no restart)**: `GET /api/source` returns
  `{demo_dir, set_name, total, visible, manifest, persisted}` —
  total = manifest items, visible = items still shown (deleted
  excluded). `POST /api/source/set {"demo_dir"}` validates the dir
  exists and contains `manifest.csv` (else 400 with a clear message),
  switches the working set, reloads the manifest, resets people
  detection, and persists the choice in `data/working-dir.json` (it
  survives a restart; a missing/unreadable persisted folder falls back
  to the built-in demo set). `POST /api/load-demo` resets to the
  built-in set and clears the persisted choice. This supersedes the
  Mac-only `TASTE_DEMO_DIR` env patch in `run.sh`/`main.py`, which v3
  removes.
- **UI**: **Folder** switcher under the People bar — shows the current
  folder and `visible/total frames`, an input + **Switch folder**
  button; errors (e.g. no `manifest.csv`) show in the status line.

## Verified 2026-11-18 (merged app, fixtures in `hidden_files/`)

- **API suite** `verify_merged_api.py`: 45/45 checks pass on the merged
  `app/` — state/rank/groups (152 frames, 35 dupe groups, 1 suggested
  keeper each), people start → done with live detection **72 with / 80
  without** (matches the 2026-11-13 v3 numbers exactly), override flip +
  flip-back, source get/set/errors, separate/best dry-runs (demo set →
  clear "RAW folder unknown" error; 6-photo fixture set → separate
  dry-run 6/0 matched, best dry-run 6 picks), evaluate, keep-set,
  label round-trip, thumb/image, delete-group error paths (unconfirmed
  / unknown group → 400; no delete endpoint was ever called).
- **UI suite** `verify_merged_ui.py` (Playwright, headless Chromium):
  21/21 checks pass — grid 152, filter buttons All (152) / With (72) /
  Without (80), people badges 72/72 and 80/80 under the filters, detail
  People line, groups view with suggested-keeper marks, best/separate
  error banners on the demo set, keyboard L keep + clear of that one id.
  Screenshot: `hidden_files/ui-merged.png`.
- Environment note: this VM was missing the system libraries mediapipe
  needs (`libEGL.so.1`, `libGLESv2.so.2`); they were installed from the
  Ubuntu 24.04 archive (libegl1 + libegl-mesa0 + libgbm1 +
  mesa-libgallium 25.2.8-0ubuntu0.24.04.5, libgles2 1.7.0-1build1).
  Without them the detector reports "unavailable" and the rest of the
  app keeps working — by design, not a merge defect. `run.sh` only
  installs the Python venv; system libs are outside the repo.

## Verified 2026-11-13 (v3, fixtures under /tmp/v3-test)

- **API suite** `verify_api.py`: 129/129 checks pass — best-shots
  dry-run (all top_n values, invalid top_n → 400, unknown raw_dir →
  clear error), best run unconfirmed → 400/moves nothing, confirmed
  run with an excluded pick + dedup `_2` re-run, people-separation
  regression (dry-run + confirmed move, unchanged behaviour),
  delete single (RAW-match and no-RAW), delete-group (all error paths:
  unconfirmed, non-member keep, unknown group), delete exclusion from
  rank/groups/people/dry-runs, source get/set/errors/persistence +
  load-demo reset, ingest (stub exiftool, JPEG-only folder, `--raw-dir`,
  nonexistent folder).
- **UI suite** `verify_ui.py` (Playwright, headless Chromium):
  37/37 checks pass — best-shots preview → reject one pick → move
  (8 picks − 1 excluded − 2 unmatched = 5 moved) → summary + disk
  state; groups view keeper badges → per-photo delete → Delete-the-rest
  confirm panel; source switch to a 3-photo set and back; source
  switch error names `manifest.csv`. Screenshots read by hand:
  `ui-best-shots.png`, `ui-groups.png` — panels render cleanly.
- Real bug found and fixed by the fixtures: deletions were keyed by
  content sha, which hid a byte-identical twin; now keyed by source
  path.
