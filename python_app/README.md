# 🎯 VisionTrack AI — Real-Time Object Detection & Tracking

> A professional, production-quality real-time computer vision system powered by **YOLOv8**, **OpenCV**, **ByteTrack / BoT-SORT**, and **Streamlit**.

---

## 📸 Demo Overview

| Feature | Description |
|---------|-------------|
| 🎥 Video Input | Live webcam or uploaded MP4/AVI/MOV/MKV |
| 🧠 Detection | YOLOv8n/s/m/l (auto-downloads on first use) |
| 🔁 Tracking | ByteTrack or BoT-SORT (via Ultralytics built-in) |
| 📊 Dashboard | Live FPS, people, vehicle, total object count |
| 💾 Output | Save processed video + CSV log of all detections |
| 📸 Screenshot | One-click screenshot capture from within the UI |

---

## ✨ Features

- **Real-time inference** at up to 200+ FPS on CPU (YOLOv8n)
- **Unique tracking IDs** maintained across frames using ByteTrack
- **80 COCO object classes** detected out of the box
- **Professional HUD overlay** with FPS counter, object counts, and tracking status
- **Custom bounding box renderer** with neon corner accents and translucent fills
- **CSV export** of every detection (timestamp, ID, class, confidence, bbox)
- **Screenshot capture** saved to `outputs/`
- **Graceful error handling** for webcam unavailability, corrupt frames, and model errors
- **Modern dark Streamlit UI** styled as a computer-vision dashboard

---

## 🛠 Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| UI | Streamlit 1.35+ |
| Detection | Ultralytics YOLOv8 |
| Tracking | ByteTrack / BoT-SORT (built into Ultralytics) |
| Video I/O | OpenCV 4.9+ |
| Math | NumPy |
| Export | Pandas + CSV stdlib |

---

## 📁 Project Structure

```
python_app/
├── app.py                  ← Streamlit entry point
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── src/
│   ├── __init__.py
│   ├── detector.py         ← YOLO model wrapper + result parsing
│   ├── tracker.py          ← ByteTrack / BoT-SORT config helper
│   ├── video_processor.py  ← OpenCV capture lifecycle + FPS
│   └── utils.py            ← Drawing helpers, CSV logger, screenshot
│
├── models/
│   └── README.md           ← Model download instructions
│
├── assets/
│   └── sample_videos/      ← Place test videos here
│
└── outputs/                ← Auto-created: saved videos, CSVs, screenshots
```

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.10 or newer
- Git (optional)
- A webcam (optional; video file also works)
- NVIDIA GPU + CUDA (optional; CPU inference works fine with yolov8n)

### 2. Create & Activate Virtual Environment

**Windows:**
```bash
cd "c:\Users\AHANA\Object detection and tracking\python_app"

python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
cd python_app
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ First install downloads ~300 MB (PyTorch CPU). Allow 2–5 minutes depending on your connection.

### 4. (Optional) GPU / CUDA Support

If you have an NVIDIA GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then in the app sidebar, the model will automatically use CUDA when available.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Streamlit will open a browser tab at `http://localhost:8501` automatically.

---

## 🎮 How to Use

### Webcam Mode
1. In the sidebar, select **📷 Webcam**.
2. Choose model (`yolov8n` for fastest).
3. Click **▶ START**.
4. Objects are detected and tracked live from your webcam.
5. Click **⏹ STOP** to end the session.

### Video File Mode
1. In the sidebar, select **🎥 Video File**.
2. Upload an MP4/AVI/MOV/MKV file using the file uploader.
3. Click **▶ START**.
4. The video is processed frame by frame with live detection.
5. The app stops automatically when the video ends.

### Controls
| Control | Description |
|---------|-------------|
| **▶ START** | Begin detection from selected source |
| **⏹ STOP** | Stop detection immediately |
| **Confidence** | Only show detections above this threshold |
| **Tracking IDs** | Toggle track ID labels on bounding boxes |
| **HUD Overlay** | Toggle on-frame stats panel |
| **Log to CSV** | Save all detections to a CSV in `outputs/` |
| **Save Video** | Record annotated video to `outputs/` |
| **📸 Screenshot** | Capture the current frame |

---

## 🧠 Model Information

| Model | Size | Speed (CPU) | mAP50 |
|-------|------|-------------|-------|
| yolov8n | ~6 MB | ~200 FPS | 37.3 |
| yolov8s | ~22 MB | ~80 FPS | 44.9 |
| yolov8m | ~52 MB | ~30 FPS | 50.2 |
| yolov8l | ~87 MB | ~15 FPS | 52.9 |

Models are **automatically downloaded** from Ultralytics on first use. No manual setup needed.

---

## 🔁 Tracking Algorithm Explanation

### ByteTrack
- Uses a **two-stage matching** strategy: first matches high-confidence detections, then re-associates low-confidence ones from a buffer.
- Minimises ID switches during brief occlusion.
- Very fast, ideal for real-time webcam tracking.

### BoT-SORT
- Adds **camera-motion compensation** and **appearance-based re-identification** on top of ByteTrack's core.
- Better for footage with a moving camera (surveillance, drone, dashcam).
- Slightly higher compute cost.

Both are included in Ultralytics 8.x — no extra installation required.

---

## 🛡️ Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Webcam unavailable | Friendly error message, no crash |
| Camera permissions denied | OS-level error surfaced with instructions |
| Invalid/corrupt video | Clear error, app stays alive |
| Missing model file | Auto-downloads or shows install instruction |
| YOLO inference error | Logs warning, skips the frame |
| Empty frame from source | Skips frame, continues |
| Video ends (EOF) | Stops cleanly, shows completion message |
| Out of memory | Caught, error shown with suggestions |

---

## 🔥 Advanced / Optional Features

- **CSV logging** — every detection row: `timestamp, session_id, track_id, label, confidence, x1, y1, x2, y2, group`
- **Processed video save** — MP4 written to `outputs/output_<timestamp>.mp4`
- **Detection History tab** — last 200 frames of per-class stats shown in a sortable table
- **Session IDs** — each Start/Stop cycle generates a unique session ID for isolated CSV files

---

## 🧪 Testing Checklist

- [x] Webcam real-time detection
- [x] Video file detection (MP4, AVI, MOV)
- [x] Multiple simultaneous object tracking
- [x] Confidence slider filtering
- [x] Low-confidence vs high-confidence thresholds
- [x] ByteTrack ID consistency over frames
- [x] Stop and restart without crash
- [x] Graceful webcam unavailable message
- [x] Invalid video file message
- [x] CSV export and download
- [x] Screenshot capture

---

## 🔧 Troubleshooting

**Q: "Webcam unavailable" even though it is connected?**
- Ensure no other application (Zoom, OBS, Teams) is using the webcam.
- On Windows, check `Settings → Privacy → Camera` and enable app access.
- Try a different webcam index: edit `open_webcam(index=1)` in `app.py`.

**Q: The app is slow / FPS is low?**
- Use `yolov8n` (nano) — it is the fastest model.
- Lower the video resolution (edit `MAX_INFERENCE_WIDTH` in `video_processor.py`).
- Close other heavy applications.
- Install PyTorch with CUDA if you have an NVIDIA GPU.

**Q: "ModuleNotFoundError: No module named 'ultralytics'"?**
- Activate your virtual environment before running: `venv\Scripts\activate`
- Re-run: `pip install -r requirements.txt`

**Q: Model not downloading?**
- Check your internet connection.
- Ultralytics downloads to `~/.ultralytics/` by default.
- You can manually download from https://github.com/ultralytics/assets/releases and place the `.pt` file in `python_app/`.

**Q: Streamlit shows a blank page?**
- Stop with `Ctrl+C` and re-run `streamlit run app.py`.
- Clear browser cache or try a private/incognito window.

---

## 🚀 Future Improvements

- [ ] Multi-camera grid view (4× or 9× split)
- [ ] Zone / ROI line-crossing detection
- [ ] Entry/exit counter per zone
- [ ] Per-class FPS performance graph (Streamlit Altair)
- [ ] Real-time confidence distribution histogram
- [ ] Firebase integration (sync with the companion React dashboard)
- [ ] RTSP stream support (ip-camera via URL)
- [ ] Custom YOLO fine-tuned model upload (`.pt` file)
- [ ] Heatmap overlay from detection density accumulation

---

## 📄 License

This project is for educational, portfolio, and demonstration purposes.
Model weights are subject to the [Ultralytics AGPL-3.0 licence](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
