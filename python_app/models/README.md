# Model Weights

YOLO model weights (`.pt` files) are **downloaded automatically** on first run.

No manual download is required. Ultralytics fetches them from its CDN and caches them at:

- **Windows:** `C:\Users\<you>\.ultralytics\`
- **Linux/macOS:** `~/.ultralytics/`

## Model Options

| Model File     | Size   | Device  | Use Case                   |
|----------------|--------|---------|----------------------------|
| `yolov8n.pt`   | ~6 MB  | CPU     | Real-time webcam (default) |
| `yolov8s.pt`   | ~22 MB | CPU     | Better accuracy            |
| `yolov8m.pt`   | ~52 MB | GPU     | High accuracy              |
| `yolov8l.pt`   | ~87 MB | GPU     | Highest accuracy           |

## Manual Download (offline environments)

If your machine has no internet access, download the weights from:

https://github.com/ultralytics/assets/releases

Then place the `.pt` file in this `models/` folder and pass the full path to
`ObjectDetector(model_name="models/yolov8n.pt")` in `app.py`.
