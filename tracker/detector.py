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
import pickle
import time

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot",
    52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
    66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

class YOLOv8Detector:
    """YOLOv8 person detector."""

    def __init__(self, model_name="yolov8n.pt",
                 conf_threshold=0.5, iou_threshold=0.4,
                 target_classes=None):
        print(f"Loading YOLOv8 model: {model_name} ...")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        if target_classes is not None:
            name_to_id = {v: k for k, v in COCO_NAMES.items()}
            self.target_class_ids = set()
            for cls_name in target_classes:
                if cls_name in name_to_id:
                    self.target_class_ids.add(name_to_id[cls_name])
                else:
                    print(f"  Warning: '{cls_name}' not in COCO, skipping.")
        else:
            self.target_class_ids = None

        print("  YOLOv8 detector ready.")

    def detect(self, frame_bgr):
        """
        Returns: np.ndarray (N, 5) — [x1, y1, x2, y2, confidence]
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf   = float(boxes.conf[i].item())
                if self.target_class_ids is not None and cls_id not in self.target_class_ids:
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf])

        return np.array(detections) if len(detections) > 0 else np.empty((0, 5))