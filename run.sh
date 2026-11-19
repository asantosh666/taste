#!/bin/sh
# Taste MVP dev server. Creates the venv on first run.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
# People detection model (EfficientDet-Lite0, ~14 MB). Downloaded once
# from the public MediaPipe model bucket if missing — the file is NOT
# committed to the repo (binary). If the download fails the app still
# starts; people detection just reports "unavailable" (see people.py).
if [ ! -f efficientdet_lite0.tflite ]; then
  echo "Downloading efficientdet_lite0.tflite (~14 MB) for offline people detection..."
  curl -sL -o efficientdet_lite0.tflite \
    https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite \
    || { echo "Model download failed — people detection will be unavailable."; rm -f efficientdet_lite0.tflite; }
fi
exec .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
