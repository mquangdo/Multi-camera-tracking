import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
from ultralytics import YOLO
from collections import deque
import json
import time

# Each tracked object maintains a Kalman filter for motion prediction
# and a gallery of appearance features for cross-camera re-identification.
#
# State vector: [cx, cy, area, aspect_ratio, vx, vy, vs]
# Measurement:  [cx, cy, area, aspect_ratio]

class KalmanBoxTracker:
    """
    Tracks a single object using a Kalman Filter for motion
    and stores appearance features for re-identification.
    """
    count = 0  # global ID counter (shared across cameras when not re-IDing)

    def __init__(self, bbox, feature=None, track_id=None):
        """
        Args:
            bbox:     [x1, y1, x2, y2]
            feature:  np.ndarray (512,) appearance embedding, or None
            track_id: int — if provided, use this ID (for cross-camera re-ID);
                      otherwise auto-increment
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # Constant velocity model
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self._bbox_to_z(bbox)

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # Assign ID
        if track_id is not None:
            self.id = track_id
        else:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1

        # Appearance feature gallery for this track
        self.features = deque(maxlen=100)
        if feature is not None:
            self.features.append(feature)

    def _bbox_to_z(self, bbox):
        """[x1,y1,x2,y2] → [cx, cy, area, aspect_ratio]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        area = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([[cx], [cy], [area], [r]])

    def _z_to_bbox(self, z):
        """[cx, cy, area, aspect_ratio] → [x1, y1, x2, y2]"""
        w = np.sqrt(max(z[2] * z[3], 0))
        h = z[2] / w if w > 0 else 0
        return np.array([
            z[0] - w / 2.0,
            z[1] - h / 2.0,
            z[0] + w / 2.0,
            z[1] + h / 2.0
        ]).flatten()

    def update(self, bbox, feature=None):
        """Correct state with a matched detection."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        if feature is not None:
            self.features.append(feature)

    def predict(self):
        """Advance state one timestep; return predicted bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        """Current bbox estimate [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.kf.x[:4].flatten())

    def get_feature(self):
        """Mean of stored appearance features."""
        if len(self.features) == 0:
            return None
        return np.mean(self.features, axis=0)

    def get_all_features(self):
        """Return all stored features as an array."""
        if len(self.features) == 0:
            return None
        return np.array(list(self.features))


# Cell 5: Cost Matrix & Hungarian Matching
#
# DeepSORT matching logic:
#   cost = λ_iou × (1 - IoU) + λ_app × cosine_distance
# Solved via the Hungarian algorithm with an IoU gate.

def iou_batch(bb_test, bb_gt):
    """
    Pairwise IoU between two bbox arrays.

    Args:
        bb_test: (N, 4) [x1,y1,x2,y2]
        bb_gt:   (M, 4) [x1,y1,x2,y2]
    Returns:
        iou: (N, M)
    """
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt   = np.expand_dims(bb_gt, 0)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_t = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_g = (bb_gt[..., 2]   - bb_gt[..., 0])   * (bb_gt[..., 3]   - bb_gt[..., 1])
    union  = area_t + area_g - inter

    return inter / np.maximum(union, 1e-6)


def cosine_distance_matrix(feats_a, feats_b):
    """
    Cosine distance between two feature matrices.
    Returns: (N, M) in [0, 2].
    """
    # Ensure L2 normalized
    norms_a = np.linalg.norm(feats_a, axis=1, keepdims=True).clip(min=1e-6)
    norms_b = np.linalg.norm(feats_b, axis=1, keepdims=True).clip(min=1e-6)
    feats_a = feats_a / norms_a
    feats_b = feats_b / norms_b
    return 1.0 - np.dot(feats_a, feats_b.T)


def combined_cost_matrix(trackers, detections, det_features,
                         lambda_iou=0.5, lambda_app=0.5):
    """
    Build cost = λ_iou × (1 - IoU) + λ_app × cosine_dist.
    """
    N = len(trackers)
    M = len(detections)
    if N == 0 or M == 0:
        return np.empty((N, M))

    pred_bboxes = np.array([t.get_state() for t in trackers])
    iou_cost = 1.0 - iou_batch(pred_bboxes, detections)

    track_features = []
    for t in trackers:
        feat = t.get_feature()
        if feat is None:
            feat = np.zeros(det_features.shape[1])
        track_features.append(feat)
    track_features = np.array(track_features)

    app_cost = cosine_distance_matrix(track_features, det_features)

    return lambda_iou * iou_cost + lambda_app * app_cost


def associate_detections_to_trackers(trackers, detections, det_features,
                                     iou_threshold=0.3,
                                     lambda_iou=0.5, lambda_app=0.5):
    """
    Hungarian matching with IoU gating.

    Returns:
        matches:              (K, 2) [tracker_idx, detection_idx]
        unmatched_detections: list of detection indices
        unmatched_trackers:   list of tracker indices
    """
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                list(range(len(detections))),
                [])

    cost = combined_cost_matrix(trackers, detections, det_features,
                                lambda_iou, lambda_app)

    row_idx, col_idx = linear_sum_assignment(cost)

    matched = []
    unmatched_dets = list(range(len(detections)))
    unmatched_trks = list(range(len(trackers)))

    for r, c in zip(row_idx, col_idx):
        pred_bbox = trackers[r].get_state()
        det_bbox  = detections[c]
        iou_val   = iou_batch(pred_bbox.reshape(1, 4), det_bbox.reshape(1, 4))[0, 0]
        if iou_val < iou_threshold:
            continue
        matched.append([r, c])
        if c in unmatched_dets:
            unmatched_dets.remove(c)
        if r in unmatched_trks:
            unmatched_trks.remove(r)

    matched = np.array(matched) if len(matched) > 0 else np.empty((0, 2), dtype=int)
    return matched, unmatched_dets, unmatched_trks