import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.detector import YOLOv8Detector
from tracker.multicamera_deepsort import MultiCameraDeepSORT
from tracker.ultilities import draw_tracks, append_tracks_jsonl
from tracker.kalman_filter import KalmanBoxTracker, associate_detections_to_trackers
from tracker.gallery import CrossCameraGallery
from tracker.feature_extractor import CustomReIDFeatureExtractor
from tracker.run_pipeline import run

from reid.model import ReIDModel

def run_pipeline():
    KalmanBoxTracker.count = 0

    feat = CustomReIDFeatureExtractor(
            checkpoint_path="weights/reid/model01.pth",
            num_classes=2,     # đúng như lúc train
            model_cls=ReIDModel, # class bạn đã định nghĩa
            input_size=(256, 128),
            feature_dim=2048
    )
    gallery = CrossCameraGallery(reid_threshold=0.7)
    
    run(
        "videos/video1_crop.avi",
        "videos/video2_crop.avi",
        cam1_yolo_model="weights/detector/yolov8m.pt",
        cam2_yolo_model="weights/detector/yolov8m.pt",
        feature_extractor=feat, gallery=gallery,
        max_age=15, target_classes=['person'],
        export_jsonl_path="results/tracks.jsonl"
    )
    
    
if __name__ == "__main__":
    run_pipeline()


