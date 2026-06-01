import cv2
import numpy as np
import streamlit as st
import os
from ultilities import load_tracks_jsonl_grouped, draw_dots_on_minimap, draw_tracks

# Config
cam1_src_pts = np.array([[3,549],[190,451],[352,539],[109,717]], dtype=np.float32)
cam1_dst_pts = np.array([[175,263],[371,262],[384,403],[184,409]], dtype=np.float32)
cam2_src_pts = np.array([[37,412],[340,399],[386,670],[21,668]], dtype=np.float32)
cam2_dst_pts = np.array([[445,180],[442,44],[673,44],[667,185]], dtype=np.float32)

# Combine windows
def render_combined_video(
    cam1_path, cam2_path, jsonl_path, minimap_path, output_path
):
    grouped      = load_tracks_jsonl_grouped(jsonl_path)
    minimap_base = cv2.imread(minimap_path)
    H1 = cv2.getPerspectiveTransform(cam1_src_pts, cam1_dst_pts)
    H2 = cv2.getPerspectiveTransform(cam2_src_pts, cam2_dst_pts)

    cap1 = cv2.VideoCapture(cam1_path)
    cap2 = cv2.VideoCapture(cam2_path)

    fps   = int(cap1.get(cv2.CAP_PROP_FPS)) or 30
    w1    = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1    = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2    = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2    = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mh, mw = minimap_base.shape[:2]

    target_h = max(h1, h2)
    mm_w     = int(mw * target_h / mh)
    out_w    = w1 + w2 + mm_w

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, target_h))

    def pad(img, h):
        dh = h - img.shape[0]
        if dh <= 0:
            return img
        return cv2.copyMakeBorder(img, dh//2, dh - dh//2, 0, 0, cv2.BORDER_CONSTANT)

    total = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    frame_idx = 0
    progress  = st.progress(0, text="Đang render...")

    while True:
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not r1 or not r2:
            break

        t1 = grouped.get("cam1", {}).get(frame_idx, np.zeros((0, 5)))
        t2 = grouped.get("cam2", {}).get(frame_idx, np.zeros((0, 5)))

        v1 = draw_tracks(f1.copy(), t1)
        v2 = draw_tracks(f2.copy(), t2)

        mm = minimap_base.copy()
        draw_dots_on_minimap(mm, t1, H1)
        draw_dots_on_minimap(mm, t2, H2)
        mm = cv2.resize(mm, (mm_w, target_h))

        combined = np.concatenate([pad(v1, target_h), pad(v2, target_h), mm], axis=1)
        out.write(combined)

        frame_idx += 1
        progress.progress(min(frame_idx / total, 1.0), text=f"Render frame {frame_idx}/{total}")

    cap1.release()
    cap2.release()
    out.release()
    progress.empty()


# UI
st.set_page_config(page_title="Multi-Camera Tracking Viewer", layout="wide")
st.title("🎥 Multi-Camera Tracking Viewer")

with st.sidebar:
    st.header("📂 Đường dẫn")
    cam1_path    = st.text_input("CAM1 video",       "videos/vid1_2.avi")
    cam2_path    = st.text_input("CAM2 video",       "videos/vid2_2.avi")
    jsonl_path   = st.text_input("Tracks JSONL",     "results/tracks2.jsonl")
    minimap_path = st.text_input("Minimap image",    "map/mymapv10.png")
    output_path  = st.text_input("Output video",     "results/combined.mp4")

    render_btn = st.button("🎬 Render Video", type="primary", use_container_width=True)

if render_btn:
    with st.spinner("Đang render..."):
        render_combined_video(cam1_path, cam2_path, jsonl_path, minimap_path, output_path)
    st.success("✅ Render xong!")

if os.path.exists(output_path):
    st.video(output_path)
else:
    st.info("👈 Nhập đường dẫn và bấm **Render Video** để bắt đầu.")