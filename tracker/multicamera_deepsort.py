import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import time
from tracker.kalman_filter import KalmanBoxTracker, associate_detections_to_trackers

class MultiCameraDeepSORT:
    """
    DeepSORT tracker with cross-camera re-identification support.

    Args:
        feature_extractor:  OSNetFeatureExtractor instance
        gallery:            CrossCameraGallery instance
        gallery_mode:       "build" — Camera 1: build the gallery
                            "query" — Camera 2+: query gallery for re-ID
                            "both"  — query existing gallery AND add new IDs
        max_age:            frames before a lost track is deleted
        min_hits:           consecutive hits to confirm a track
        iou_threshold:      IoU gate for within-frame matching
        lambda_iou:         IoU cost weight
        lambda_app:         appearance cost weight
    """

    def __init__(self, feature_extractor, gallery,
                 gallery_mode="build",
                 max_age=30, min_hits=3,
                 iou_threshold=0.3,
                 lambda_iou=0.4, lambda_app=0.6, cam_name='cam_1', save_dir="my_image"):
        self.feature_extractor = feature_extractor
        self.gallery = gallery
        self.gallery_mode = gallery_mode
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lambda_iou = lambda_iou
        self.lambda_app = lambda_app
        self.trackers = []
        self.frame_count = 0
        self.cam_name = cam_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Track which global IDs have been assigned via re-ID in this session
        self.reid_log = {}  # { local_tracker_idx: (global_id, distance) }

    def _save_crop(self, frame_bgr, bbox, track_id):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame_bgr[y1:y2, x1:x2]
        ts = int(time.time() * 1000)
        fname = f"{self.cam_name}_id{track_id}_t{self.frame_count}_ts{ts}.jpg"
        cv2.imwrite(os.path.join(self.save_dir, fname), crop)
        

    def _try_reid(self, feature):
        """
        Attempt to re-identify a detection using the cross-camera gallery.

        Returns:
            global_id (int or None), distance (float or None)
        """
        if self.gallery_mode in ("query", "both"):
            gid, dist = self.gallery.query(feature)
            if gid is not None:
                return gid, dist
        return None, None


    def update(self, detections, frame_bgr):
        """
        Process one frame.
    
        Args:
            detections: np.ndarray (N, 5) — [x1, y1, x2, y2, confidence]
            frame_bgr:  np.ndarray (H, W, 3) BGR image
    
        Returns:
            outputs: np.ndarray (K, 5) — [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        det_bboxes = detections[:, :4] if len(detections) > 0 else np.empty((0, 4))
    
        # --- Extract appearance features ---
        if len(det_bboxes) > 0:
            det_features = self.feature_extractor.extract(frame_bgr, det_bboxes)
        else:
            det_features = np.empty((0, self.feature_extractor.feature_dim))
    
        # --- Predict step for all existing trackers ---
        for t in self.trackers:
            t.predict()
    
        # --- Associate detections to existing trackers ---
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            self.trackers, det_bboxes, det_features,
            self.iou_threshold, self.lambda_iou, self.lambda_app
        )
    
        # --- Update matched trackers ---
        for trk_idx, det_idx in matched:
            self.trackers[trk_idx].update(det_bboxes[det_idx], det_features[det_idx])
            # đánh dấu ID đang active toàn cục
            self.gallery.mark_active(self.trackers[trk_idx].id, self.cam_name)
    
        # 🔥 Lưu feature ngay khi track đã ổn định
        if self.gallery_mode in ("build", "both"):
            for trk_idx, det_idx in matched:
                t = self.trackers[trk_idx]
                if t.hits >= self.min_hits:
                    feat = det_features[det_idx]
                    self.gallery.add_track(t.id, feat, cam_name=self.cam_name)
    
        # --- Handle unmatched detections ---
        for i in unmatched_dets:
            feat = det_features[i] if len(det_features) > 0 else None
            print(f"[{self.cam_name}] new detection idx={i} bbox={det_bboxes[i].astype(int).tolist()}")
    
            assigned_id = None
            best_id = best_dist = best_src = None
            active_elsewhere = None
    
            # Try cross-camera re-identification
            if feat is not None and self.gallery_mode in ("query", "both"):
                gid, dist = self._try_reid(feat)
                best_id, best_dist = gid, dist
                best_src = self.gallery.get_source(gid) if gid is not None else None
                active_elsewhere = self.gallery.active_cam(gid) if gid is not None else None
                if gid is not None:
                    print(f"[{self.cam_name}] re-id best_id={gid} dist={dist:.4f} from_cam={best_src} active_cam={active_elsewhere}")
                    # chỉ gán nếu ID chưa active ở bất kỳ camera nào
                    if not self.gallery.is_active(gid):
                        assigned_id = gid
                else:
                    print(f"[{self.cam_name}] re-id no match (best_dist={dist})")
    
            new_tracker = KalmanBoxTracker(det_bboxes[i], feature=feat, track_id=assigned_id)
            self.trackers.append(new_tracker)
            self.gallery.mark_active(new_tracker.id, self.cam_name)
    
            if assigned_id is not None:
                print(f"[{self.cam_name}] assigned existing global ID {assigned_id} (from_cam={best_src})")
            else:
                print(f"[{self.cam_name}] created new ID {new_tracker.id}")
    
        # --- Remove dead trackers & update gallery ---
        surviving = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                surviving.append(t)
            else:
                if self.gallery_mode in ("build", "both") and t.hits >= self.min_hits:
                    all_feats = t.get_all_features()
                    if all_feats is not None:
                        self.gallery.add_track(t.id, all_feats, cam_name=self.cam_name)
                        print(f"[{self.cam_name}] add_track id={t.id} feats={len(t.features)} gallery_ids={len(self.gallery)}")
                # giải phóng trạng thái active toàn cục
                self.gallery.mark_inactive(t.id)
        self.trackers = surviving
    
        # --- Collect outputs (confirmed tracks only) ---
        outputs = []
        for t in self.trackers:
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                bbox = t.get_state()
                outputs.append([bbox[0], bbox[1], bbox[2], bbox[3], t.id])
    
        return np.array(outputs) if len(outputs) > 0 else np.empty((0, 5))

    def finalize(self):
        """
        Call at the end of a video to flush all active tracks into the gallery.
        Critical for Camera 1 — ensures all tracked people are saved.
        """
        if self.gallery_mode in ("build", "both"):
            saved_count = 0
            for t in self.trackers:
                all_feats = t.get_all_features()
                if all_feats is not None and t.hits >= self.min_hits:
                    self.gallery.add_track(t.id, all_feats)
                    saved_count += 1
            print(f"  Finalized: saved {saved_count} active tracks to gallery")