"""Fetch the CrowdHuman-trained YOLOv5m head/person detector checkpoint used
by head_detector.py (and therefore gaze_target_pipeline.py). Not tracked in
git (169MB, public model, downloaded via a community mirror since the
original repo only links Google Drive).

Run once after cloning:
    .venv/bin/pip install gdown
    .venv/bin/python3 download_weights.py

Gaze-LLE's own weights (DINOv2 backbone + gazelle head) are fetched
automatically via torch.hub the first time gaze_target_pipeline.py runs, and
cached in ~/.cache/torch/hub — no manual step needed for those.
"""
from __future__ import annotations

from pathlib import Path

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "crowdhuman_yolov5m.pt"
GDRIVE_FILE_ID = "1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb"


def main() -> None:
    if WEIGHTS_PATH.exists():
        print(f"Already present: {WEIGHTS_PATH}")
        return

    try:
        import gdown
    except ImportError:
        raise SystemExit("Missing dependency: run `pip install gdown` first, then re-run this script.")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading crowdhuman_yolov5m.pt -> {WEIGHTS_PATH}")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", str(WEIGHTS_PATH), quiet=False)


if __name__ == "__main__":
    main()
