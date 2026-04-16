"""
Streamlit Dashboard - Professional AI Surveillance Interface

Clean, demo-ready UI for real-time surveillance monitoring.

Usage:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import cv2
import tempfile
import time

from src.input_module.input_handler import InputModule
from src.detection_module.detector import DetectionModule
from src.tracking_module.tracker import TrackingModule
from src.behavior_module.behavior_analyzer import BehaviorAnalyzer
from src.output_module.output_handler import OutputModule


# Page configuration
st.set_page_config(
    page_title="AI Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Initialize session state
if "surveillance" not in st.session_state:
    st.session_state.surveillance = {
        "running": False,
        "pipeline": None,
        "source_type": "Webcam",
        "uploaded_file": None,
        "temp_video_path": None
    }


def init_pipeline(source=0, confidence=0.5):
    """Initialize the surveillance pipeline."""
    detector = DetectionModule(
        confidence_threshold=confidence,
        classes=[0]
    )
    detector.warmup()

    input_module = InputModule(source=source)
    if not input_module.open():
        raise RuntimeError(f"Failed to open video source: {source}")

    return {
        "input": input_module,
        "detector": detector,
        "tracker": TrackingModule(),
        "behavior": BehaviorAnalyzer(),
        "output": OutputModule(log_dir="logs"),
    }


def process_frame(pipeline):
    """Process a single frame through the pipeline."""
    frame_obj = pipeline["input"].read()
    if frame_obj is None:
        return None, None

    frame = frame_obj.image
    detections = pipeline["detector"].detect(frame)
    person_detections = [d for d in detections if d.class_name == "person"]
    tracks = pipeline["tracker"].update(person_detections, frame)
    behaviors = pipeline["behavior"].analyze(tracks)
    output_frame = pipeline["output"].process(frame, tracks, behaviors)
    metrics = pipeline["output"].get_metrics()

    return output_frame, metrics


def stop_pipeline():
    """Stop the surveillance pipeline."""
    if st.session_state.surveillance["pipeline"]:
        st.session_state.surveillance["pipeline"]["input"].close()
        st.session_state.surveillance["pipeline"] = None
    st.session_state.surveillance["running"] = False


# Header section
st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        🛡️ AI Surveillance System
    </h1>
    <p style='text-align: center; color: #666; margin-top: 0;'>
        Real-time person tracking & behavior analysis
    </p>
""", unsafe_allow_html=True)

st.divider()

# Main layout: Video (center) | Sidebar (controls + metrics)
video_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    # Controls Section
    st.header("⚙️ Controls")

    # Source selection
    source_type = st.segmented_control(
        "Video Source",
        options=["Webcam", "Upload"],
        default="Webcam"
    )

    st.session_state.surveillance["source_type"] = source_type

    # Video upload
    uploaded_file = None
    if source_type == "Upload":
        uploaded_file = st.file_uploader(
            "Choose video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for analysis"
        )

    # Detection confidence
    confidence = st.slider(
        "Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Detection confidence threshold"
    )

    st.divider()

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button(
            "▶️ Start",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.surveillance["running"]
        )
    with col2:
        stop_clicked = st.button(
            "⏹️ Stop",
            type="secondary",
            use_container_width=True,
            disabled=not st.session_state.surveillance["running"]
        )

    st.divider()

    # Metrics Section (only shown when running)
    if st.session_state.surveillance["running"]:
        st.header("📊 Status")
        metrics_container = st.container()
    else:
        st.info("Click **Start** to begin monitoring")

    # Alerts Section
    st.header("🚨 Alerts")
    alerts_container = st.container()

# Video Feed Section (Main Area)
with video_col:
    video_placeholder = st.empty()

    # Show placeholder when not running
    if not st.session_state.surveillance["running"]:
        video_placeholder.markdown("""
            <div style='
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 100px 20px;
                text-align: center;
                border: 2px dashed #444;
            '>
                <h2 style='color: #888; margin: 0;'>📹 Surveillance Feed</h2>
                <p style='color: #666;'>Select video source and click Start</p>
            </div>
        """, unsafe_allow_html=True)

# Store uploaded file in session state (don't start automatically)
if uploaded_file is not None:
    # Only save if new file or different from previous
    if (st.session_state.surveillance.get("uploaded_file_name") != uploaded_file.name):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.session_state.surveillance["temp_video_path"] = tfile.name
        st.session_state.surveillance["uploaded_file_name"] = uploaded_file.name
        st.toast(f"✓ File '{uploaded_file.name}' ready")

# Handle Start button
if start_clicked:
    try:
        # Determine source
        if source_type == "Webcam":
            source = 0
        elif st.session_state.surveillance.get("temp_video_path"):
            source = st.session_state.surveillance["temp_video_path"]
        else:
            st.error("Please upload a video file first")
            st.stop()

        # Initialize pipeline
        st.session_state.surveillance["pipeline"] = init_pipeline(source, confidence)
        st.session_state.surveillance["running"] = True
        st.rerun()

    except Exception as e:
        st.error(f"Failed to start: {e}")
        stop_pipeline()

# Handle Stop button
if stop_clicked:
    stop_pipeline()
    # Clean up temp file
    if st.session_state.surveillance["temp_video_path"]:
        try:
            Path(st.session_state.surveillance["temp_video_path"]).unlink(missing_ok=True)
        except:
            pass
        st.session_state.surveillance["temp_video_path"] = None
    st.rerun()

# Main processing loop
if st.session_state.surveillance["running"] and st.session_state.surveillance["pipeline"]:
    try:
        output_frame, metrics = process_frame(st.session_state.surveillance["pipeline"])

        if output_frame is None:
            stop_pipeline()
            st.warning("Video stream ended")
            st.rerun()
        else:
            # Update video feed
            rgb_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                rgb_frame,
                channels="RGB",
                use_container_width=True
            )

            # Update metrics
            with metrics_container:
                fps = metrics.get('fps', 0)
                tracks = metrics.get('active_tracks', 0)
                alerts = metrics.get('active_alerts', 0)

                col1, col2 = st.columns(2)
                col1.metric("FPS", f"{fps:.1f}")
                col2.metric("Tracks", tracks)

                alert_color = "🔴" if alerts > 0 else "🟢"
                st.metric("Alerts", f"{alert_color} {alerts}")

            # Update alerts
            with alerts_container:
                recent_alerts = st.session_state.surveillance["pipeline"]["output"].get_recent_alerts(count=5)

                if not recent_alerts:
                    st.caption("No alerts")
                else:
                    for alert in recent_alerts[:5]:
                        severity = alert.get('severity', 'medium')
                        alert_type = alert.get('alert_type', 'UNKNOWN').replace('_', ' ')
                        track_id = alert.get('track_id', '-')

                        if severity == 'high':
                            st.error(f"⚠️ {alert_type} (ID: {track_id})", icon="🔴")
                        elif severity == 'medium':
                            st.warning(f"⚡ {alert_type} (ID: {track_id})", icon="🟡")
                        else:
                            st.info(f"ℹ️ {alert_type} (ID: {track_id})", icon="🟢")

            # Small delay to prevent high CPU usage
            time.sleep(0.01)
            st.rerun()

    except Exception as e:
        stop_pipeline()
        st.error(f"Pipeline error: {e}")
        st.rerun()

# Footer
st.divider()
st.caption("AI Surveillance System v1.0 | Phase 1 Complete")
