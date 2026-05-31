import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from tracker.detector import YOLOv8Detector
from tracker.multicamera_deepsort import MultiCameraDeepSORT
from tracker.ultilities import draw_tracks, append_tracks_jsonl


def run(
    cam1_video_path, cam2_video_path,
    feature_extractor, gallery,
    cam1_yolo_model="yolov8l.pt", cam2_yolo_model="yolov8l.pt",
    target_classes=None,
    conf_threshold=0.75, nms_iou_threshold=0.4,
    max_age=30, min_hits=3,
    iou_threshold=0.3, lambda_iou=0.4, lambda_app=0.6,
    export_jsonl_path=None,
):
    det1 = YOLOv8Detector(cam1_yolo_model, conf_threshold, nms_iou_threshold, target_classes)
    det2 = YOLOv8Detector(cam2_yolo_model, conf_threshold, nms_iou_threshold, target_classes)

    trk1 = MultiCameraDeepSORT(
        feature_extractor, gallery,
        gallery_mode="both",
        cam_name='cam1',
        max_age=max_age, min_hits=min_hits,
        iou_threshold=iou_threshold, lambda_iou=lambda_iou, lambda_app=lambda_app
    )
    trk2 = MultiCameraDeepSORT(
        feature_extractor, gallery,
        gallery_mode="both",
        cam_name='cam2',
        max_age=max_age, min_hits=min_hits,
        iou_threshold=iou_threshold, lambda_iou=lambda_iou, lambda_app=lambda_app
    )

    cap1, cap2 = cv2.VideoCapture(cam1_video_path), cv2.VideoCapture(cam2_video_path)
    if not cap1.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cam1_video_path}")
    if not cap2.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cam2_video_path}")

    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    export_fp = open(export_jsonl_path, "w", encoding="utf-8") if export_jsonl_path else None

    frame_idx = 0
    try:
        while True:
            ret1, f1 = cap1.read()
            ret2, f2 = cap2.read()
            if not ret1 and not ret2:
                break

            if not ret1:
                h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
                f1 = np.zeros((h1, w1, 3), dtype=np.uint8)
            if not ret2:
                h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                f2 = np.zeros((h2, w2, 3), dtype=np.uint8)

            dets1 = det1.detect(f1)
            tracks1 = trk1.update(dets1, f1)
            if export_fp:
                append_tracks_jsonl(export_fp, frame_idx, "cam1", tracks1)

            dets2 = det2.detect(f2)
            tracks2 = trk2.update(dets2, f2)
            if export_fp:
                append_tracks_jsonl(export_fp, frame_idx, "cam2", tracks2)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"Processed frame {frame_idx} "
                      f"({min(frame_idx, total1)}/{total1} cam1, {min(frame_idx, total2)}/{total2} cam2)")
    finally:
        cap1.release()
        cap2.release()
        if export_fp:
            export_fp.close()

    trk1.finalize()
    trk2.finalize()

    print(f"Done. Frames processed: {frame_idx}")
    if export_jsonl_path:
        print(f"Exported tracks JSONL: {export_jsonl_path}")
    print(f"Gallery summary: {gallery}")
