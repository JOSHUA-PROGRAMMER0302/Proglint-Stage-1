"""Phase 1: YOLO person detection on Video A."""
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "video_a.mp4"
OUT = ROOT / "outputs" / "detect_a"


def main():
    if not VIDEO.exists():
        raise SystemExit(f"Missing video: {VIDEO}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    model.predict(
        source=str(VIDEO),
        classes=[0],
        device=device,
        save=True,
        project=str(OUT.parent),
        name=OUT.name,
        exist_ok=True,
    )
    print(f"Done. Annotated video under: {OUT}")


if __name__ == "__main__":
    main()
