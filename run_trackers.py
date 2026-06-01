import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracker.kalman_filter import KalmanBoxTracker, associate_detections_to_trackers
from tracker.gallery import CrossCameraGallery
from tracker.feature_extractor import CustomReIDFeatureExtractor
from tracker.run_pipeline import run
from reid.model import ReIDModel
from configs.load_config import load_run_config

def run_pipeline():
    cfg = load_run_config(path="configs/main_config.yaml")
    KalmanBoxTracker.count = 0

    feat = CustomReIDFeatureExtractor(
        checkpoint_path=cfg["weights"]["reid"],
        num_classes=cfg["reid"]["num_classes"],
        model_cls=ReIDModel,
        input_size=tuple(cfg["reid"]["input_size"]),
        feature_dim=cfg["reid"]["feature_dim"],
    )
    gallery = CrossCameraGallery(reid_threshold=cfg["reid"]["threshold"])

    run(
        cfg["videos"]["cam1"],
        cfg["videos"]["cam2"],
        cam1_yolo_model=cfg["weights"]["detector"],
        cam2_yolo_model=cfg["weights"]["detector"],
        feature_extractor=feat, gallery=gallery,
        max_age=cfg["tracker"]["max_age"],
        target_classes=cfg["tracker"]["target_classes"],
        export_jsonl_path=cfg["results"]["jsonl"],
    )


if __name__ == "__main__":
    run_pipeline()

