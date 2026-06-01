import sys, os 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from configs.load_config import load_config

# Config
cfg = load_config(path="configs/main_config.yaml")
map_cfg = load_config(path="configs/mapping_config.yaml")

cam1_video = cfg['videos']['cam1']
cam2_video = cfg['videos']['cam2']
minimap_path = map_cfg['minimap']


MODES = ["cam1_src", "cam1_dst", "cam2_src", "cam2_dst"]


# State
state = {
    "points": {m: [] for m in MODES},
    "mode_idx": 0,
}

# Load first frame of videos and minimap
def load_data(cam1_path, cam2_path, map_path):
    '''
    Load the first frame of each video and the minimap.

    Args:
        cam1_path: str — path to the first camera video
        cam2_path: str — path to the second camera video
        map_path: str — path to the minimap image

    Returns:
        tuple — (frame1, frame2, minimap)
    '''
    cap1 = cv2.VideoCapture(cam1_path)
    cap2 = cv2.VideoCapture(cam2_path)

    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    cap1.set(cv2.CAP_PROP_POS_FRAMES, total1 // 2)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, total2 // 2)

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    minimap = cv2.imread(map_path)

    cap1.release()
    cap2.release()

    if not ret1 or not ret2 or minimap is None:
        raise FileNotFoundError("Lỗi load dữ liệu — kiểm tra lại đường dẫn video/minimap")

    return frame1, frame2, minimap

# Draw
def draw_points(img, pts, color):
    '''
    Draw points with indices on an image.
    
    Args:
        img: np.ndarray — image to draw on
        pts: list — list of points to draw
        color: tuple — color of the points
    
    Returns:        
        None (modifies img in-place)
    '''
    
    for i, (x, y) in enumerate(pts):
        cv2.circle(img, (x, y), 6, color, -1)
        cv2.putText(img, str(i), (x + 6, y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_labels(img1, img2, imgm, current_mode):
    '''
    Draw mode and labels on the images.
    
    Args:
        img1: np.ndarray — first camera frame
        img2: np.ndarray — second camera frame
        imgm: np.ndarray — minimap image
        current_mode: str — current mode for labeling
    
    Returns:
        None (modifies images in-place)
    '''
    
    for img in [img1, img2]:
        cv2.putText(img, f"MODE: {current_mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(imgm, f"MODE: {current_mode}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(imgm, "CAM1_DST", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(imgm, "CAM2_DST", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def render(frame1, frame2, minimap, points, current_mode):
    '''
    Render the current state of point selection on the images.
    
    Args:
        frame1: np.ndarray — first camera frame
        frame2: np.ndarray — second camera frame
        minimap: np.ndarray — minimap image
        points: dict — dictionary of selected points for each mode
        current_mode: str — current mode for labeling
    
    Returns:
        None (displays images in windows)
    '''
    img1, img2, imgm = frame1.copy(), frame2.copy(), minimap.copy()

    draw_points(img1, points["cam1_src"], (0, 255, 0))
    draw_points(img2, points["cam2_src"], (0, 0, 255))
    draw_points(imgm, points["cam1_dst"], (0, 255, 0))
    draw_points(imgm, points["cam2_dst"], (0, 0, 255))
    draw_labels(img1, img2, imgm, current_mode)

    cv2.imshow("cam1", img1)
    cv2.imshow("cam2", img2)
    cv2.imshow("minimap", imgm)

# Mouse click handler factory
def make_click_handler(points, state):
    def click(event, x, y, flags, param):
        current_mode = MODES[state["mode_idx"]]
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points[current_mode]) < 4:
                points[current_mode].append((x, y))
                print(f"{current_mode}: {(x, y)}")
    return click

# Print results
def print_result(points):
    def to_np(arr):
        return "np.array([" + ",".join([f"[{x},{y}]" for x, y in arr]) + "], dtype=np.float32)"

    print("\n=========================\n")
    print("cam1_src_pts =", to_np(points["cam1_src"]))
    print("cam1_dst_pts =", to_np(points["cam1_dst"]))
    print("cam2_src_pts =", to_np(points["cam2_src"]))
    print("cam2_dst_pts =", to_np(points["cam2_dst"]))
    print("\n=========================\n")

# Handle key presses
def handle_key(key, state):
    """Return True nếu muốn thoát"""
    current_mode = MODES[state["mode_idx"]]

    if key == 27:  # ESC
        return True

    elif key == ord('n'):
        state["mode_idx"] = (state["mode_idx"] + 1) % 4
        print(f"👉 Switch to {MODES[state['mode_idx']]}")

    elif key == ord('r'):
        state["points"][current_mode] = []
        print(f"Reset {current_mode}")

    elif key == ord('p'):
        print_result(state["points"])
        return True

    return False

# Set up windows and mouse callbacks
def setup_windows(points, state):
    cv2.namedWindow("cam1")
    cv2.namedWindow("cam2")
    cv2.namedWindow("minimap")

    handler = make_click_handler(points, state)
    cv2.setMouseCallback("cam1", handler)
    cv2.setMouseCallback("cam2", handler)
    cv2.setMouseCallback("minimap", handler)

# Main
def map():
    frame1, frame2, minimap = load_data(cam1_video, cam2_video, minimap_path)
    setup_windows(state["points"], state)

    print("""
HƯỚNG DẪN:
1. cam1_src  → chọn 4 điểm trên cam1
2. nhấn 'n'
3. cam1_dst  → chọn 4 điểm trên minimap
4. nhấn 'n'
5. cam2_src  → chọn 4 điểm trên cam2
6. nhấn 'n'
7. cam2_dst  → chọn 4 điểm trên minimap
8. nhấn 'p' để in kết quả

Phím:
- n: next mode
- r: reset mode hiện tại
- p: print kết quả
- ESC: thoát
""")

    while True:
        current_mode = MODES[state["mode_idx"]]
        render(frame1, frame2, minimap, state["points"], current_mode)

        key = cv2.waitKey(1) & 0xFF
        if handle_key(key, state):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    map()