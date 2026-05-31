# Multi-Camera Tracking

A cross-camera multi-person tracking system using YOLOv8 detection, DeepSORT tracking, and a custom ReID (Re-Identification) model with ResNet50 + BNNeck architecture.

## Pipeline

```
YOLOv8 Detection → Feature Extraction (ReID) → Kalman Filter Tracking → Cross-Camera Gallery Matching
```

Each camera runs YOLOv8 for person detection, extracts appearance features via a custom ReID model, and tracks individuals with a Kalman filter. A cross-camera gallery stores features from all cameras and performs re-identification when a person leaves one view and enters another.

## Project Structure

| Module | Description |
|---|---|
| `main.py` | Entry point — runs the tracking pipeline on two videos |
| `visualize.py` | Interactive visualization with minimap, track overlay, and click-to-focus |
| `ultilities.py` | Drawing helpers, homography projection, JSONL track loading |
| `tracker/detector.py` | YOLOv8 detector (COCO classes, filterable by target class) |
| `tracker/kalman_filter.py` | Kalman filter per-track state estimation + Hungarian matching (IoU + cosine cost) |
| `tracker/multicamera_deepsort.py` | DeepSORT with gallery build/query modes for cross-camera ID assignment |
| `tracker/gallery.py` | Cross-camera feature gallery — stores embeddings, queries by cosine distance |
| `tracker/feature_extractor.py` | Custom ReID feature extractor wrapper (ResNet50 + BNNeck) |
| `tracker/run_pipeline.py` | Two-camera pipeline orchestrator (detect → track → export JSONL) |
| `tracker/ultilities.py` | Drawing, color generation, JSONL export utilities |
| `reid/model.py` | ReIDModel — ResNet50 backbone + BNNeck, returns L2-normalized features |
| `reid/training.py` | Training script (Warmup LR, Label Smoothing, Triplet + Center Loss) |
| `reid/sampler.py` | RandomIdentitySampler (P identities × K images per batch) |
| `reid/loss.py` | LabelSmoothingCE, TripletLossHardMining, CenterLoss |
| `tools/mapping_src_dest.py` | Interactive tool to select source/destination points for homography calibration |

## ReID Model

The re-identification model is a ResNet50 (ImageNet pretrained) with:

- **Last stride = 1** for higher spatial resolution (16×8)
- **BNNeck** — separates features for triplet loss (`ft`) vs. classification (`fi`)
- **Inference** returns L2-normalized `fi` features for cosine similarity matching

Trained with: Warmup LR schedule, Label Smoothing CE, Hard-mining Triplet Loss, Center Loss.

## Usage

```bash
python main.py
```

Two video files (`videos/video1_crop.avi`, `videos/video2_crop.avi`) are processed frame-by-frame. Track results are exported to `results/tracks.jsonl`.

### Training ReID

Configure paths in `reid/training.py` and run:

```bash
python -c "from reid.training import train; train()"
```

### Visualization

```bash
python visualize.py
```

Interactive viewer with:
- Side-by-side camera views with bounding boxes and track IDs
- Minimap with projected track positions (via homography)
- Click to focus on a specific track ID across all views
- Space to pause/resume, R to reset focus

### Homography Calibration

```bash
python tools/mapping_src_dest.py
```

Select 4 corresponding points per camera → minimap to compute perspective transforms.

## Output Format

Tracks are exported as JSONL (one JSON object per detection per frame):

```json
{"frame": 0, "cam": "cam1", "id": 3, "bbox_xyxy": [100.0, 200.0, 150.0, 350.0]}
```