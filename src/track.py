"""Phase 2 baseline track (no Global ID). Prefer: python src/run_video_a.py"""
import csv
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "video_a.mp4"
OUT_DIR = ROOT / "outputs" / "track_a"
CSV_PATH = ROOT / "outputs" / "tracks_a.csv"
TRACKER = ROOT / "configs" / "botsort.yaml"


def main():
    if not VIDEO.exists():
        raise SystemExit(f"Missing video: {VIDEO}")

    device = "0" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    results = model.track(
        source=str(VIDEO),
        classes=[0],
        tracker=str(TRACKER),
        persist=True,
        device=device,
        stream=True,
        save=True,
        project=str(OUT_DIR.parent),
        name=OUT_DIR.name,
        exist_ok=True,
    )

    rows = []
    for frame_id, r in enumerate(results, start=1):
        timestamp = (frame_id - 1) / fps
        if r.boxes is None or r.boxes.id is None:
            continue
        ids = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for tid, box, conf in zip(ids, xyxy, confs):
            x1, y1, x2, y2 = box.tolist()
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(timestamp, 4),
                    "global_id": int(tid),
                    "track_id": int(tid),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                    "confidence": round(float(conf), 4),
                }
            )

    fields = [
        "frame_id",
        "timestamp",
        "global_id",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Tracks: {len(rows)} rows → {CSV_PATH}")


if __name__ == "__main__":
    main()
