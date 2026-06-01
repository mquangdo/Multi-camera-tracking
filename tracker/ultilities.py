import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import json

def get_color(track_id):
    '''
    Deterministic color per track ID.
    
    Args:
        track_id: int — track ID to generate color for
        
    Returns:
        (b, g, r) tuple of ints in [0, 255]
    '''
    np.random.seed(int(track_id) * 7 + 13)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))


def draw_tracks(frame, tracks, gallery_ids=None, thickness=2):
    '''
    Draw bounding boxes and track IDs on the frame.
    
    Args:
        frame: np.ndarray (H, W, 3) - input image
        tracks: np.ndarray (K, 5) [x1,y1,x2,y2, track_id] - tracks to draw
        gallery_ids: set of track IDs that are in the cross-camera gallery (optional)
        thickness: int - thickness of bounding box lines

    Returns:
        np.ndarray - image with drawn tracks
    '''
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
    '''
    Append track records to a JSONL file for later analysis or visualization.
    
    Args:
        fp: file-like object opened for appending text
        frame_idx: int - current frame index
        cam_name: str - camera name or ID
        tracks: np.ndarray (K, 5) [x1,y1,x2,y2, track_id] - tracks to record
    
    Returns:
        None (writes to file)
    '''
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
