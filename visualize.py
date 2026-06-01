import cv2
import numpy as np
import yaml
from ultilities import load_tracks_jsonl_grouped, draw_dots_on_minimap, draw_tracks, build_click_boxes
from configs.load_config import load_run_config, load_map_config


def run_visualization(
    cam1_video_path,
    cam2_video_path,
    tracks_jsonl_path,
    minimap_image_path,
    cam1_src_pts, cam1_dst_pts,
    cam2_src_pts, cam2_dst_pts,
):
    grouped      = load_tracks_jsonl_grouped(tracks_jsonl_path)
    minimap_base = cv2.imread(minimap_image_path)
    H1           = cv2.getPerspectiveTransform(cam1_src_pts, cam1_dst_pts)
    H2           = cv2.getPerspectiveTransform(cam2_src_pts, cam2_dst_pts)

    cap1 = cv2.VideoCapture(cam1_video_path)
    cap2 = cv2.VideoCapture(cam2_video_path)

    state = {
        "frame_idx":  0,
        "focus_id":   None,
        "paused":     False,
        "boxes_cam1": [],
        "boxes_cam2": [],
    }

    def click_handler(which_cam):
        def _cb(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            boxes = state["boxes_cam1"] if which_cam == "cam1" else state["boxes_cam2"]
            for (x1, y1, x2, y2, tid) in reversed(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    state["focus_id"] = tid
                    return
            state["focus_id"] = None
        return _cb

    cv2.namedWindow("cam1")
    cv2.namedWindow("cam2")
    cv2.namedWindow("minimap")
    cv2.setMouseCallback("cam1", click_handler("cam1"))
    cv2.setMouseCallback("cam2", click_handler("cam2"))

    last_v1, last_v2, last_mm = None, None, None

    while True:
        if not state["paused"]:
            r1, f1 = cap1.read()
            r2, f2 = cap2.read()
            if not r1 or not r2:
                break

            fr = state["frame_idx"]
            t1 = grouped.get("cam1", {}).get(fr, np.zeros((0, 5)))
            t2 = grouped.get("cam2", {}).get(fr, np.zeros((0, 5)))

            focus = state["focus_id"]
            if focus is not None:
                t1 = t1[t1[:, 4] == focus] if len(t1) else t1
                t2 = t2[t2[:, 4] == focus] if len(t2) else t2

            last_v1 = draw_tracks(f1.copy(), t1)
            last_v2 = draw_tracks(f2.copy(), t2)
            last_mm = minimap_base.copy()
            draw_dots_on_minimap(last_mm, t1, H1)
            draw_dots_on_minimap(last_mm, t2, H2)

            state["boxes_cam1"] = build_click_boxes(t1)
            state["boxes_cam2"] = build_click_boxes(t2)
            state["frame_idx"] += 1

        if last_v1 is not None:
            cv2.imshow("cam1",    last_v1)
            cv2.imshow("cam2",    last_v2)
            cv2.imshow("minimap", last_mm)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            state["paused"] = not state["paused"]
        if key == ord("r"):
            state["focus_id"] = None

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = load_run_config("configs/main_config.yaml")
    minimap_path, results_path, cam1_src, cam1_dst, cam2_src, cam2_dst  = load_map_config("configs/mapping_config.yaml")

    run_visualization(
        cam1_video_path=cfg["videos"]["cam1"],
        cam2_video_path=cfg["videos"]["cam2"],
        tracks_jsonl_path=results_path,
        minimap_image_path=minimap_path,
        cam1_src_pts=cam1_src, cam1_dst_pts=cam1_dst,
        cam2_src_pts=cam2_src, cam2_dst_pts=cam2_dst,
    )