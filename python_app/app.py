"""
app.py — VisionTrack AI
Simple real-time object detection & tracking using YOLOv8 + Streamlit.

Run:  streamlit run app.py
"""

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from src.detector import ObjectDetector
from src.tracker import get_tracker_config, list_tracker_names
from src.video_processor import VideoProcessor
from src.utils import (
    draw_detections,
    draw_hud,
    compute_class_stats,
    log_to_csv,
    get_csv_path,
)
from src.ai_assistant import is_ai_available, generate_scene_analysis
from src.supabase_client import (
    is_supabase_configured,
    log_detection as log_to_supabase,
    flush_buffer as flush_supabase,
    log_session_summary as log_session_to_supabase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("visiontrack")

# ── Page Setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisionTrack AI", page_icon="🎯", layout="wide")

# Auto-redirect from raw network IP to localhost to eliminate browser 'Not secure' flag
components.html(
    """
    <script>
    try {
        if (window.top.location.hostname !== 'localhost' && window.top.location.hostname !== '127.0.0.1') {
            window.top.location.href = 'http://localhost:' + (window.top.location.port || '8501');
        }
    } catch (e) {
        if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            window.location.href = 'http://localhost:' + (window.location.port || '8501');
        }
    }
    </script>
    """,
    height=0,
    width=0,
)

# ── Session State Defaults ───────────────────────────────────────────────────
defaults = {
    "running": False,
    "detector": None,
    "vp": None,
    "session_id": str(uuid.uuid4())[:8],
    "model_name": "yolov8n",
    "tracker_name": "ByteTrack",
    "conf": 0.50,
    "show_ids": True,
    "show_conf": True,
    "show_hud": True,
    "save_csv": False,
    "fps": 0.0,
    "total": 0,
    "people": 0,
    "vehicles": 0,
    "animals": 0,
    "frames_count": 0,
    "error_msg": "",
    "last_annotated": None,
    "sync_supabase": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

ss = st.session_state


# ── Helpers ──────────────────────────────────────────────────────────────────
def get_detector(model_name):
    if ss.detector is None or ss.detector.model_name != model_name:
        with st.spinner(f"Loading {model_name}..."):
            ss.detector = ObjectDetector(model_name=model_name)
    return ss.detector


def stop():
    ss.running = False
    if ss.vp:
        ss.vp.release()
        ss.vp = None
    if ss.get("sync_supabase"):
        try:
            flush_supabase()
            log_session_to_supabase(
                ss.session_id,
                {"total": ss.total, "person": ss.people, "vehicle": ss.vehicles, "animal": ss.animals, "frames": ss.frames_count},
                model_name=ss.model_name
            )
        except Exception:
            pass


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 VisionTrack AI")

    st.markdown("**Input Source**")
    source = st.radio("Source", ["📷 Webcam", "🎥 Video File"], horizontal=True, label_visibility="collapsed")
    uploaded = None
    if "Video" in source:
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

    st.divider()
    st.markdown("**Model**")
    ss.model_name = st.selectbox("YOLO Model", ["yolov8n", "yolov8s", "yolov8m", "yolov8l"], label_visibility="collapsed")
    ss.tracker_name = st.selectbox("Tracker", list_tracker_names(), label_visibility="collapsed")

    st.divider()
    st.markdown("**Settings**")
    ss.conf = st.slider("Confidence", 0.10, 0.95, ss.conf, 0.05)
    ss.show_ids = st.checkbox("Show IDs", ss.show_ids)
    ss.show_conf = st.checkbox("Show Confidence", ss.show_conf)
    ss.show_hud = st.checkbox("Show HUD", ss.show_hud)
    ss.save_csv = st.checkbox("Log to CSV", ss.save_csv)
    if is_supabase_configured():
        ss.sync_supabase = st.checkbox("Sync to Supabase", ss.sync_supabase)
    else:
        st.checkbox("Sync to Supabase", value=False, disabled=True, help="Set SUPABASE_URL and SUPABASE_KEY in .env to enable")
    st.divider()
    col1, col2 = st.columns(2)
    start = col1.button("▶ Start", use_container_width=True, type="primary")
    stop_btn = col2.button("⏹ Stop", use_container_width=True)

    if start:
        ss.error_msg = ""
        stop()
        detector = get_detector(ss.model_name)
        vp = VideoProcessor()

        if "Webcam" in source:
            ok = vp.open_webcam()
            if not ok:
                ss.error_msg = "❌ Webcam not available. Ensure camera permissions are granted and no other application is using it."
        else:
            if uploaded is None:
                ss.error_msg = "❌ Please upload a video file before clicking Start."
                ok = False
            else:
                try:
                    uploaded.seek(0)
                    file_bytes = uploaded.read()
                    ok = vp.open_file_bytes(file_bytes, suffix=Path(uploaded.name).suffix)
                    if not ok:
                        ss.error_msg = f"❌ Could not open video file '{uploaded.name}'."
                except Exception as upload_err:
                    ss.error_msg = f"❌ Error reading uploaded file: {upload_err}"
                    ok = False

        if ok:
            ss.vp = vp
            ss.running = True
            ss.frames_count = 0
            ss.session_id = str(uuid.uuid4())[:8]
            st.rerun()

    if stop_btn:
        stop()
        st.rerun()


# ── Main Area ────────────────────────────────────────────────────────────────
st.markdown("# 🎯 VisionTrack AI — Object Detection & Tracking")

if ss.error_msg:
    st.error(ss.error_msg)

# Stats row with dynamic placeholder slots
c1, c2, c3, c4, c5, c6 = st.columns(6)
m_total = c1.empty()
m_people = c2.empty()
m_vehicles = c3.empty()
m_animals = c4.empty()
m_fps = c5.empty()
m_frames = c6.empty()

m_total.metric("Objects", ss.total)
m_people.metric("People", ss.people)
m_vehicles.metric("Vehicles", ss.vehicles)
m_animals.metric("Animals", ss.animals)
m_fps.metric("FPS", f"{ss.fps:.1f}")
m_frames.metric("Frames", ss.frames_count)

st.divider()

if ss.running:
    col_status, col_stop = st.columns([4, 1])
    col_status.success("🟢 Video stream active")
    if col_stop.button("⏹ Stop Detection", key="main_stop", use_container_width=True):
        stop()
        st.rerun()

video_area = st.empty()

if not ss.running:
    video_area.info("Select a source and press ▶ Start")

# CSV & Screenshot download
col_d1, col_d2 = st.columns(2)
csv_path = get_csv_path(ss.session_id)
if csv_path.exists():
    with open(csv_path, "rb") as f:
        col_d1.download_button("⬇ Download Detections CSV", f, file_name=f"detections_{ss.session_id}.csv", use_container_width=True)

if ss.last_annotated is not None:
    is_success, buffer = cv2.imencode(".png", ss.last_annotated)
    if is_success:
        col_d2.download_button("📸 Download Frame Snapshot", buffer.tobytes(), file_name=f"snapshot_{ss.session_id}.png", mime="image/png", use_container_width=True)

if is_ai_available():
    with st.expander("🤖 AI Scene Intelligence & Summary", expanded=False):
        st.caption("Generate automated scene intelligence and density summary powered by Vision AI.")
        if st.button("Generate AI Scene Report", use_container_width=True):
            with st.spinner("Analyzing detection metrics with Vision AI..."):
                current_stats = {
                    "total": ss.total,
                    "person": ss.people,
                    "vehicle": ss.vehicles,
                    "animal": ss.animals,
                    "frames": ss.frames_count,
                }
                report = generate_scene_analysis(current_stats, session_id=ss.session_id)
                st.markdown(report)

# ── Detection Loop ───────────────────────────────────────────────────────────
if ss.running and ss.vp:
    detector = get_detector(ss.model_name)
    tracker_cfg = get_tracker_config(ss.tracker_name)

    try:
        while ss.running:
            ok, frame = ss.vp.read_frame()
            if not ok:
                if ss.vp and "Webcam" in str(getattr(ss.vp, "_source", "")):
                    retried = False
                    for _ in range(5):
                        time.sleep(0.04)
                        ok, frame = ss.vp.read_frame()
                        if ok and frame is not None and frame.size > 0:
                            retried = True
                            break
                    if not retried:
                        stop()
                        st.warning("⚠️ Camera stream ended.")
                        break
                else:
                    stop()
                    st.info("✅ Video playback completed.")
                    break

            small = VideoProcessor.resize_for_inference(frame)
            dets = detector.detect_and_track(small, conf_threshold=ss.conf, tracker_config=tracker_cfg)

            # Scale boxes back if frame was resized
            if small.shape != frame.shape:
                sx = frame.shape[1] / small.shape[1]
                sy = frame.shape[0] / small.shape[0]
                for d in dets:
                    x1, y1, x2, y2 = d["bbox"]
                    d["bbox"] = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))

            annotated = draw_detections(frame, dets, show_ids=ss.show_ids, show_conf=ss.show_conf)
            stats = compute_class_stats(dets)

            if ss.show_hud:
                annotated = draw_hud(annotated, fps=ss.vp.fps, total=stats["total"],
                                     people=stats["person"], vehicles=stats["vehicle"],
                                     animals=stats["animal"], model_name=ss.model_name)

            if ss.save_csv:
                log_to_csv(dets, session_id=ss.session_id)

            if ss.sync_supabase:
                log_to_supabase(dets, session_id=ss.session_id)

            ss.last_annotated = annotated
            ss.fps = ss.vp.fps
            ss.total = stats["total"]
            ss.people = stats["person"]
            ss.vehicles = stats["vehicle"]
            ss.animals = stats["animal"]
            ss.frames_count += 1

            if ss.frames_count % 3 == 0:
                m_total.metric("Objects", ss.total)
                m_people.metric("People", ss.people)
                m_vehicles.metric("Vehicles", ss.vehicles)
                m_animals.metric("Animals", ss.animals)
                m_fps.metric("FPS", f"{ss.fps:.1f}")
                m_frames.metric("Frames", ss.frames_count)

            video_area.image(VideoProcessor.bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

    except Exception as e:
        logger.exception("Error: %s", e)
        ss.error_msg = f"❌ Error: {e}"
        stop()
        st.error(ss.error_msg)


# ── Vercel Function compatibility export ─────────────────────────────────────
# If inspected by Vercel serverless runtime:
try:
    from http.server import BaseHTTPRequestHandler

    class handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"VisionTrack Streamlit desktop app is configured for local running.")
    app = handler
except Exception:
    app = None

