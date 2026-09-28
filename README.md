# VisionTrack AI — Object Detection & Tracking

Simple real-time object detection and tracking using YOLOv8 + Streamlit.

## Quick Start

```bash
cd python_app
pip install -r requirements.txt
streamlit run app.py
```

## What It Does

1. **Open a webcam** or **upload a video file**
2. **YOLOv8** detects objects (people, vehicles, animals, etc.)
3. **ByteTrack / BoT-SORT** assigns tracking IDs across frames
4. Bounding boxes, labels, and a HUD overlay are drawn on the video
5. Optionally **log detections to CSV**

## Project Structure

```
python_app/
├── app.py                 ← Main app (run this)
├── requirements.txt       ← Python dependencies
├── outputs/               ← Screenshots & CSV logs
└── src/
    ├── detector.py        ← YOLOv8 wrapper
    ├── tracker.py         ← Tracker config (ByteTrack / BoT-SORT)
    ├── video_processor.py ← Video capture & FPS calculation
    └── utils.py           ← Drawing, stats, CSV logging
```

## Requirements

- Python 3.10+
- Webcam (optional — can use video files instead)
- GPU optional (CPU works fine with yolov8n)
