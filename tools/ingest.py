#!/usr/bin/env python3
"""Ingest a photo folder into a Taste working set.

Usage:
    python tools/ingest.py <photo-folder> [--raw-dir <raw-dir>] [--out <working-dir>]

What it does:
- Scans <photo-folder> for images (RAW + JPG/JPEG/PNG/TIF).
- For each RAW, extracts the embedded JPEG preview into the working
  dir (exiftool `-b -PreviewImage`, fallback `dcraw -e -c`, else warn
  and skip that file — never fail the whole run).
- For each JPG/PNG, symlinks the original into the working dir
  (no copy, no duplication). For TIF, converts to a JPEG preview
  with Pillow (the TIF itself stays put).
- Writes manifest.csv (file,set,label,basis,source,width,height;
  set=shoot, label blank) and source.json ({"raw_dir": ...}) into
  the working dir.

Safety: never writes into the source folder, never modifies a
source file. exiftool only reads; dcraw runs with `-c` (stdout),
so no thumbnail file lands next to the RAW.

source.json: {"raw_dir": <raw-dir>} when --raw-dir is given, else the
photo folder itself when it holds RAWs. A folder with no RAWs gets
no raw_dir — the separate/best-shots endpoints then return their
clear "RAW folder unknown" error instead of moving nothing silently.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RAW_EXTS = {".RW2", ".ARW", ".DNG", ".NEF", ".CR2", ".CR3", ".RAF", ".ORF"}
IMG_EXTS = {".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"}
ALL_EXTS = RAW_EXTS | IMG_EXTS


def _run(cmd: list[str], timeout: int = 120) -> bytes | None:
    """Run a helper, return stdout bytes, or None on any failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def extract_preview(raw_path: Path) -> tuple[bytes | None, str]:
    """Embedded JPEG preview bytes for one RAW, or (None, reason)."""
    tool = shutil.which("exiftool")
    if tool:
        out = _run([tool, "-b", "-PreviewImage", str(raw_path)])
        if out and out[:2] == b"\xff\xd8":
            return out, "exiftool"
        if out:
            return None, "exiftool returned non-JPEG bytes"
    tool = shutil.which("dcraw")
    if tool:
        # -e extracts the thumbnail, -c sends it to stdout (no file
        # is written next to the RAW).
        out = _run([tool, "-e", "-c", str(raw_path)])
        if out and out[:2] == b"\xff\xd8":
            return out, "dcraw"
        if out:
            return None, "dcraw returned non-JPEG bytes"
    return None, "no preview extractor (need exiftool or dcraw)"


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) of JPEG bytes, or None if unreadable."""
    try:
        from PIL import Image
        import io

        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("photo_folder", help="folder of photos to ingest")
    ap.add_argument("--raw-dir", default=None,
                    help="RAW originals folder (defaults to the photo folder when it holds RAWs)")
    ap.add_argument("--out", default=None,
                    help="working dir to create (default: <photo-folder-name>-taste next to the folder)")
    args = ap.parse_args()

    src_dir = Path(args.photo_folder).expanduser().resolve()
    if not src_dir.is_dir():
        print(f"error: folder does not exist: {src_dir}", file=sys.stderr)
        return 2
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else src_dir.parent / f"{src_dir.name}-taste"
    )
    if out_dir.resolve() == src_dir:
        print("error: --out must differ from the photo folder", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in src_dir.rglob("*")
        if p.is_file()
        and not p.name.startswith("._")
        and not p.name.startswith(".")
        and p.suffix.upper() in ALL_EXTS
    )
    raws = [p for p in files if p.suffix.upper() in RAW_EXTS]
    print(f"found {len(files)} image(s) in {src_dir} ({len(raws)} RAW)")

    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    used_names: set[str] = set()

    def dest_name(stem: str, suffix: str) -> str:
        name = f"{stem}{suffix}"
        n = 2
        while name.lower() in used_names:
            name = f"{stem}_{n}{suffix}"
            n += 1
        used_names.add(name.lower())
        return name

    for p in files:
        stem = p.stem
        if p.suffix.upper() in RAW_EXTS:
            data, how = extract_preview(p)
            if data is None:
                skipped.append((p.name, how))
                print(f"  skip {p.name}: {how}")
                continue
            size = jpeg_size(data)
            if size is None:
                skipped.append((p.name, "extracted preview is not a readable JPEG"))
                print(f"  skip {p.name}: extracted preview is not a readable JPEG")
                continue
            name = dest_name(stem, ".jpg")
            (out_dir / name).write_bytes(data)
            rows.append({
                "file": name, "set": "shoot", "label": "",
                "basis": f"embedded preview from {p.name} ({how})",
                "source": str(p), "width": size[0], "height": size[1],
            })
        elif p.suffix.upper() in (".TIF", ".TIFF"):
            try:
                from PIL import Image

                with Image.open(p) as im:
                    im = im.convert("RGB")
                    size = im.size
                    name = dest_name(stem, ".jpg")
                    im.save(out_dir / name, "JPEG", quality=90)
            except Exception as exc:  # noqa: BLE001 - one bad file skips
                skipped.append((p.name, f"TIF conversion failed: {exc}"))
                print(f"  skip {p.name}: TIF conversion failed: {exc}")
                continue
            rows.append({
                "file": name, "set": "shoot", "label": "",
                "basis": f"converted from {p.name} (Pillow, not an embedded preview)",
                "source": str(p), "width": size[0], "height": size[1],
            })
        else:  # JPG / PNG: link the original, never copy it
            name = dest_name(stem, p.suffix.lower())
            dest = out_dir / name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            os.symlink(p, dest)
            try:
                from PIL import Image

                with Image.open(p) as im:
                    size = im.size
            except Exception:
                size = (0, 0)
            rows.append({
                "file": name, "set": "shoot", "label": "",
                "basis": f"original {p.suffix[1:].upper()} (symlinked)",
                "source": str(p), "width": size[0], "height": size[1],
            })

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "set", "label", "basis", "source", "width", "height"]
        )
        w.writeheader()
        w.writerows(rows)

    if args.raw_dir:
        raw_dir = str(Path(args.raw_dir).expanduser().resolve())
    elif raws:
        raw_dir = str(src_dir)
    else:
        raw_dir = ""
    if raw_dir:
        (out_dir / "source.json").write_text(
            json.dumps({"raw_dir": raw_dir}, indent=1) + "\n"
        )
        print(f"wrote source.json: raw_dir = {raw_dir}")
    else:
        print("no RAWs found: no source.json written (separate/best-shots "
              "will report the RAW folder as unknown)")

    print(f"\ningested {len(rows)} photo(s) into {out_dir}")
    print(f"skipped {len(skipped)} file(s)")
    for name, reason in skipped:
        print(f"  - {name}: {reason}")
    print(f"\nUse it with:  DEMO_DIR={out_dir}")
    print(f"  or in the app: POST /api/source/set {{\"demo_dir\": \"{out_dir}\"}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
