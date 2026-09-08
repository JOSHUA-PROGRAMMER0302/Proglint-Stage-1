"""Develop on Video A: YOLO + BoT-SORT + Global Re-ID → video + CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from global_id import GlobalIdManager  # noqa: E402

TRACKER = ROOT / "configs" / "botsort.yaml"


def pick_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "video_a.mp4"
    if not video.is_absolute():
        video = (Path.cwd() / video).resolve()
    if not video.exists():
        raise SystemExit(f"Missing video: {video}\nUsage: python src/run_video_a.py [path/to/video.mp4]")

    out_dir = ROOT / "outputs" / video.stem
    csv_path = out_dir / "tracks.csv"
    video_out = out_dir / "annotated.mp4"

    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device} video={video}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    model = YOLO("yolov8n.pt")
    gid_mgr = GlobalIdManager(similarity_thresh=0.72, max_age_frames=int(fps * 10))
    rows: list[dict] = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        result = model.track(
            frame,
            classes=[0],
            tracker=str(TRACKER),
            persist=True,
            device=device,
            verbose=False,
        )[0]

        active: set[int] = set()
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for tid, box, conf in zip(ids, xyxy, confs):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                crop = frame[y1:y2, x1:x2]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                gid = gid_mgr.assign(int(tid), crop, frame_id, (cx, cy))
                active.add(int(tid))

                rows.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": round((frame_id - 1) / fps, 4),
                        "global_id": gid,
                        "track_id": int(tid),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": round(float(conf), 4),
                    }
                )

                color = (0, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {gid}"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

        gid_mgr.mark_missing(active, frame_id)
        writer.write(frame)
        if frame_id % 50 == 0:
            print(f"frame {frame_id}")

    cap.release()
    writer.release()

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
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)

    n_ids = len({r["global_id"] for r in rows})
    print(f"frames={frame_id} detections={len(rows)} unique_global_ids={n_ids}")
    print(f"video → {video_out}")
    print(f"csv   → {csv_path}")


if __name__ == "__main__":
    main()
