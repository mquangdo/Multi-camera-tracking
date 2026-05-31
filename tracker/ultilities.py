import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json

def get_color(track_id):
    """Deterministic color per track ID."""
    np.random.seed(int(track_id) * 7 + 13)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def draw_tracks(frame, tracks, gallery_ids=None, thickness=2):
    """
    Draw bounding boxes and IDs. If gallery_ids is provided,
    tracks whose ID is in gallery_ids get a ★ re-ID marker.

    Args:
        frame:       BGR image (modified in-place)
        tracks:      (K, 5) [x1,y1,x2,y2, track_id]
        gallery_ids: set of IDs from cross-camera gallery (for labeling)
        thickness:   line thickness
    """
    for trk in tracks:
        x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
        color = get_color(tid)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label: mark re-identified tracks with ★
        if gallery_ids is not None and tid in gallery_ids:
            label = f"ID:{tid}"
        else:
            label = f"ID:{tid}"

        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def append_tracks_jsonl(fp, frame_idx, cam_name, tracks):
    """
    tracks: (K,5) [x1,y1,x2,y2,tid]
    """
    if tracks is None:
        return
    for trk in tracks:
        rec = {
            "frame": int(frame_idx),
            "cam": str(cam_name),
            "id": int(trk[4]),
            "bbox_xyxy": [float(trk[0]), float(trk[1]), float(trk[2]), float(trk[3])],
        }
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
