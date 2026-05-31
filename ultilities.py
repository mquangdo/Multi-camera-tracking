import cv2
import numpy as np
import json
from collections import defaultdict

# ---------- utils ----------
def get_color(track_id: int):
    rng = np.random.default_rng(int(track_id) * 10007 + 12345)
    b, g, r = rng.integers(60, 255, size=3).tolist()
    return int(b), int(g), int(r)

def clamp_tracks(tracks, w, h):
    if tracks is None or len(tracks) == 0:
        return tracks
    t = tracks.copy()
    t[:, 0] = np.clip(t[:, 0], 0, w - 1)
    t[:, 2] = np.clip(t[:, 2], 0, w - 1)
    t[:, 1] = np.clip(t[:, 1], 0, h - 1)
    t[:, 3] = np.clip(t[:, 3], 0, h - 1)
    return t

def load_tracks_jsonl_grouped(jsonl_path):
    tmp = defaultdict(lambda: defaultdict(list))
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            frame = int(r["frame"])
            cam = str(r["cam"])
            x1, y1, x2, y2 = r["bbox_xyxy"]
            tid = int(r["id"])
            tmp[cam][frame].append([x1, y1, x2, y2, tid])

    grouped = {}
    for cam, frames in tmp.items():
        grouped[cam] = {}
        for fr, rows in frames.items():
            grouped[cam][fr] = np.asarray(rows, dtype=np.float32)
    return grouped

# ---------- homography ----------
def project_points(H, pts_xy):
    if pts_xy is None or len(pts_xy) == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    proj = (H @ pts_h.T).T
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-6, 1e-6, w)
    return (proj[:, :2] / w).astype(np.float32)

def draw_dots_on_minimap(minimap_bgr, tracks, H, dot_radius=6):
    if tracks is None or len(tracks) == 0:
        return
    tr = np.asarray(tracks, dtype=np.float32)
    pts = np.stack([(tr[:,0]+tr[:,2])*0.5, tr[:,3]], axis=1)
    tids = tr[:,4].astype(np.int32)

    uv = project_points(H, pts)
    h, w = minimap_bgr.shape[:2]
    for (u, v), tid in zip(uv, tids):
        x, y = int(u), int(v)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(minimap_bgr, (x,y), dot_radius, get_color(tid), -1)

# ---------- drawing ----------
def draw_tracks(frame, tracks, thickness=2):
    if tracks is None:
        return frame
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return frame

    for trk in arr:
        x1, y1, x2, y2, tid = trk
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        tid = int(tid)

        color = get_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(frame, f"ID:{tid}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return frame

def build_click_boxes(tracks):
    boxes = []
    if tracks is None:
        return boxes
    arr = np.asarray(tracks, dtype=np.float32)
    if arr.size == 0:
        return boxes
    for trk in arr:
        x1, y1, x2, y2, tid = trk
        boxes.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
    return boxes