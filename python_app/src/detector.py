"""
detector.py
-----------
YOLO model wrapper for object detection and tracking.
Uses Ultralytics YOLOv8 with built-in ByteTrack / BoT-SORT support.
"""

import logging

import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# ── COCO class groupings for dashboard statistics ────────────────────────────
PERSON_CLASSES = {"person"}

VEHICLE_CLASSES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat"
}

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
}

# Human-readable, color-coded bounding-box palette (BGR for OpenCV)
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person":     (57, 255, 20),    # Neon green
    "car":        (0, 200, 255),    # Cyan
    "truck":      (0, 140, 255),    # Orange
    "bus":        (0, 60, 255),     # Red-orange
    "motorcycle": (180, 60, 255),   # Purple
    "bicycle":    (255, 200, 0),    # Yellow
    "airplane":   (255, 100, 100),  # Light red
    "boat":       (255, 180, 0),    # Gold
    "_default":   (0, 240, 255),    # Cyan fallback
}


class ObjectDetector:
    """
    Wraps Ultralytics YOLO with track() support.

    Parameters
    ----------
    model_name : str
        One of: 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'.
        The weight file is downloaded automatically on first use.
    device : str
        'cpu', 'cuda', or 'mps'. Defaults to 'cpu' for widest compatibility.
    """

    def __init__(self, model_name: str = "yolov8n", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._class_names: list[str] = []
        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load or download the YOLO model weights."""
        try:
            from ultralytics import YOLO  # deferred import for speed
            weight_file = f"{self.model_name}.pt"
            logger.info("Loading model: %s on device: %s", weight_file, self.device)
            self._model = YOLO(weight_file)
            self._class_names = self._model.names  # dict[int, str]
            logger.info("Model loaded successfully. Classes: %d", len(self._class_names))
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics package not found. Run: pip install ultralytics"
            ) from exc
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", self.model_name, exc)
            raise RuntimeError(f"Model load failed: {exc}") from exc

    # ── Detection + Tracking ─────────────────────────────────────────────────

    def detect_and_track(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        tracker_config: str = "bytetrack.yaml",
        iou_threshold: float = 0.45,
        max_det: int = 300,
    ) -> list[dict]:
        """
        Run YOLO detection + tracking on a single BGR frame.

        Returns
        -------
        list[dict]  Each dict contains:
            id          (int)   Unique tracking ID  (-1 if tracking disabled)
            label       (str)   Class name
            class_id    (int)   COCO class index
            confidence  (float) 0–1
            bbox        (tuple) (x1, y1, x2, y2) in pixel coords
            color       (tuple) BGR color for drawing
            group       (str)   'person' | 'vehicle' | 'animal' | 'other'
        """
        if self._model is None:
            return []

        if frame is None or frame.size == 0:
            logger.warning("detect_and_track received an empty frame.")
            return []

        try:
            results = self._model.track(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                tracker=tracker_config,
                max_det=max_det,
                device=self.device,
                verbose=False,
                persist=True,          # keep tracker state between calls
            )
        except Exception as exc:
            logger.error("YOLO inference error: %s", exc)
            return []

        return self._parse_results(results)

    def detect_only(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.50,
        iou_threshold: float = 0.45,
    ) -> list[dict]:
        """Run detection without tracking (no IDs assigned)."""
        if self._model is None or frame is None or frame.size == 0:
            return []
        try:
            results = self._model.predict(
                source=frame,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("YOLO predict error: %s", exc)
            return []
        return self._parse_results(results, has_ids=False)

    # ── Result Parsing ───────────────────────────────────────────────────────

    def _parse_results(self, results, has_ids: bool = True) -> list[dict]:
        """Convert raw Ultralytics Results objects to clean dicts."""
        detections: list[dict] = []

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                try:
                    # Bounding box (xyxy, absolute pixel coords)
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # Class
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    label = self._class_names.get(cls_id, f"class_{cls_id}")

                    # Confidence
                    conf = float(boxes.conf[i].cpu().numpy())

                    # Tracking ID
                    track_id = -1
                    if has_ids and boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())

                    # Color and group
                    color = CLASS_COLORS.get(label, CLASS_COLORS["_default"])
                    group = self._get_group(label)

                    detections.append({
                        "id":         track_id,
                        "label":      label,
                        "class_id":   cls_id,
                        "confidence": conf,
                        "bbox":       (x1, y1, x2, y2),
                        "color":      color,
                        "group":      group,
                    })
                except Exception as exc:
                    logger.debug("Skipping detection %d due to parse error: %s", i, exc)
                    continue

        return detections

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_group(label: str) -> str:
        if label in PERSON_CLASSES:
            return "person"
        if label in VEHICLE_CLASSES:
            return "vehicle"
        if label in ANIMAL_CLASSES:
            return "animal"
        return "other"

    def reload(self, model_name: str) -> None:
        """Hot-swap model at runtime (called when user changes model in sidebar)."""
        if model_name != self.model_name:
            self.model_name = model_name
            self._load_model()

    @property
    def class_names(self) -> dict:
        return self._class_names
