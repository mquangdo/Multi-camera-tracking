import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
from tracker.kalman_filter import KalmanBoxTracker, cosine_distance_matrix

class CrossCameraGallery:
    """
    Stores and manages appearance features for cross-camera re-identification.

    The gallery maps global track IDs → arrays of appearance embeddings.
    When a new unmatched detection appears in a new camera view,
    we compare its feature against all gallery entries to find the best match.
    """

    def __init__(self, reid_threshold=0.4, max_features_per_id=50):
        """
        Args:
            reid_threshold:      max cosine distance to accept a re-ID match.
                                 Lower = stricter (fewer false positives).
                                 Typical range: 0.3–0.5 for OSNet.
            max_features_per_id: max number of feature vectors stored per ID.
        """
        self.gallery = {}           # { global_id: np.ndarray (K, 512) }
        self.reid_threshold = reid_threshold
        self.max_features_per_id = max_features_per_id
        self.id_source = {}         # id -> cam_name tạo đầu tiên
        self.id_active = {}         # id -> cam_name đang giữ ID (toàn cục)

    def add_track(self, track_id, features, cam_name=None):
        """
        Add or update a track's features in the gallery.

        Args:
            track_id: int — global track ID
            features: np.ndarray (K, D) or (D,) — appearance features
        """
        if features is None:
            return
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if track_id in self.gallery:
            existing = self.gallery[track_id]
            combined = np.vstack([existing, features])
            # Keep only the most recent features
            if len(combined) > self.max_features_per_id:
                combined = combined[-self.max_features_per_id:]
            self.gallery[track_id] = combined
        else:
            self.gallery[track_id] = features[-self.max_features_per_id:]

        self.id_source.setdefault(track_id, cam_name)

    # ---------- Active-ID helpers (to avoid one ID active in multiple cams) ----------
    def mark_active(self, track_id, cam_name):
        '''Mark a track ID as currently active in a specific camera.'''
        if track_id is not None:
            self.id_active[track_id] = cam_name

    def mark_inactive(self, track_id):
        if track_id is not None:
            self.id_active.pop(track_id, None)

    def is_active(self, track_id):
        '''Check if a track ID is currently active in any camera.'''
        return track_id in self.id_active

    def active_cam(self, track_id):
        '''Return the camera name where this track ID is currently active, or None if inactive.'''
        return self.id_active.get(track_id)

    def query(self, feature):
        """
        Find the best matching gallery ID for a query feature.

        Args:
            feature: np.ndarray (D,) — L2-normalized query feature

        Returns:
            (best_id, best_distance) or (None, None) if no match below threshold
        """
        if len(self.gallery) == 0:
            return None, None

        feature = feature.reshape(1, -1)
        best_id = None
        best_dist = float('inf')

        for gid, gallery_feats in self.gallery.items():
            # Compute cosine distance to each gallery feature, take minimum
            dists = cosine_distance_matrix(feature, gallery_feats)  # (1, K)
            min_dist = dists.min()
            if min_dist < best_dist:
                best_dist = min_dist
                best_id = gid

        if best_dist < self.reid_threshold:
            return best_id, best_dist
        return None, None

    def save(self, filepath):
        """Save the gallery to disk."""
        data = {
            'gallery': self.gallery,
            'reid_threshold': self.reid_threshold,
            'max_features_per_id': self.max_features_per_id,
            'next_id': KalmanBoxTracker.count,
            'id_source': self.id_source,
            'id_active': self.id_active,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Gallery saved: {filepath}")
        print(f"  Total IDs: {len(self.gallery)}")
        total_feats = sum(len(v) for v in self.gallery.values())
        print(f"  Total features: {total_feats}")

    def load(self, filepath):
        """Load the gallery from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.gallery = data['gallery']
        self.reid_threshold = data['reid_threshold']
        self.max_features_per_id = data['max_features_per_id']
        KalmanBoxTracker.count = data['next_id']
        self.id_source = data.get('id_source', {})
        self.id_active = data.get('id_active', {})
        print(f"Gallery loaded: {filepath}")
        print(f"  Total IDs: {len(self.gallery)}")
        print(f"  Next ID counter: {KalmanBoxTracker.count}")

    def get_source(self, track_id):
        '''Return the camera source that originally created this track ID, or None if unknown.'''
        return self.id_source.get(track_id)

    def get_all_ids(self):
        """Return all stored track IDs."""
        return list(self.gallery.keys())

    def __len__(self):
        return len(self.gallery)

    def __repr__(self):
        total_feats = sum(len(v) for v in self.gallery.values())
        return (f"CrossCameraGallery(ids={len(self.gallery)}, "
                f"features={total_feats}, "
                f"threshold={self.reid_threshold})")

