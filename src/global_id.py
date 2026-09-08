"""Map local tracker IDs → stable global person IDs using appearance."""
from __future__ import annotations

import numpy as np


def appearance_embedding(bgr_crop: np.ndarray) -> np.ndarray | None:
    """Compact HSV histogram embedding. No extra model needed."""
    import cv2

    if bgr_crop is None or bgr_crop.size == 0:
        return None
    h, w = bgr_crop.shape[:2]
    if h < 8 or w < 8:
        return None
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    n = np.linalg.norm(hist)
    if n < 1e-6:
        return None
    return hist / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class GlobalIdManager:
    # ponytail: histogram Re-ID; swap for OSNet if false merges show up on Video A
    def __init__(
        self,
        similarity_thresh: float = 0.72,
        max_age_frames: int = 300,
    ):
        self.similarity_thresh = similarity_thresh
        self.max_age_frames = max_age_frames
        self._next_gid = 1
        self._track_to_gid: dict[int, int] = {}
        self._memory: dict[int, dict] = {}  # gid -> embedding, last_seen, last_xy

    def assign(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_id: int,
        center_xy: tuple[float, float],
    ) -> int:
        emb = appearance_embedding(crop)

        if track_id in self._track_to_gid:
            gid = self._track_to_gid[track_id]
            self._touch(gid, emb, frame_id, center_xy)
            return gid

        # New local track — try restore from lost identities
        best_gid, best_sim = None, -1.0
        if emb is not None:
            for gid, info in self._memory.items():
                if gid in self._track_to_gid.values():
                    continue  # already active under another track
                age = frame_id - info["last_seen"]
                if age < 1 or age > self.max_age_frames:
                    continue
                if info["embedding"] is None:
                    continue
                sim = cosine(emb, info["embedding"])
                if sim > best_sim:
                    best_sim, best_gid = sim, gid

        if best_gid is not None and best_sim >= self.similarity_thresh:
            gid = best_gid
        else:
            gid = self._next_gid
            self._next_gid += 1

        self._track_to_gid[track_id] = gid
        self._touch(gid, emb, frame_id, center_xy)
        return gid

    def mark_missing(self, active_track_ids: set[int], frame_id: int) -> None:
        gone = [tid for tid in list(self._track_to_gid) if tid not in active_track_ids]
        for tid in gone:
            del self._track_to_gid[tid]
        # expire very old memory
        expired = [
            gid
            for gid, info in self._memory.items()
            if frame_id - info["last_seen"] > self.max_age_frames
        ]
        for gid in expired:
            del self._memory[gid]

    def _touch(self, gid, emb, frame_id, center_xy) -> None:
        prev = self._memory.get(gid)
        if emb is not None and prev and prev["embedding"] is not None:
            # EMA update appearance
            emb = 0.7 * prev["embedding"] + 0.3 * emb
            n = np.linalg.norm(emb)
            emb = emb / n if n > 1e-6 else emb
        self._memory[gid] = {
            "embedding": emb if emb is not None else (prev["embedding"] if prev else None),
            "last_seen": frame_id,
            "last_xy": center_xy,
        }
