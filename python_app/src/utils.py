"""
utils.py
--------
Drawing helpers, statistics computation, screenshot capture, and CSV logging.
All drawing uses OpenCV so no Pillow dependency is required at runtime.
"""

import csv
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Ensure outputs directory exists next to this package
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Visual constants ─────────────────────────────────────────────────────────
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.55
FONT_THICK    = 1
BOX_THICK     = 2
LABEL_PADDING = 6          # px padding inside label banner
CORNER_LEN    = 14         # px length of corner accent marks
CORNER_THICK  = 3


# ── Drawing ──────────────────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    show_ids: bool = True,
    show_conf: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes, labels, tracking IDs, and confidence scores
    on the frame (in-place copy returned).

    Parameters
    ----------
    frame      : BGR NumPy array
    detections : list of dicts from ObjectDetector.detect_and_track()
    show_ids   : draw tracking ID on label
    show_conf  : draw confidence % on label
    show_labels: draw class name on label

    Returns
    -------
    Annotated BGR frame (new array, original unchanged).
    """
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det.get("color", (0, 240, 255))
        label = det.get("label", "obj")
        conf  = det.get("confidence", 0.0)
        tid   = det.get("id", -1)

        # ── Filled semi-transparent mask ─────────────────────────────────
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)

        # ── Main bounding box ────────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICK)

        # ── Corner accent marks ──────────────────────────────────────────
        _draw_corners(out, x1, y1, x2, y2, color)

        # ── Label string ─────────────────────────────────────────────────
        parts: list[str] = []
        if show_labels:
            parts.append(label.capitalize())
        if show_ids and tid >= 0:
            parts.append(f"ID:{tid:02d}")
        if show_conf:
            parts.append(f"{conf * 100:.0f}%")

        if parts:
            text = " | ".join(parts)
            _draw_label(out, text, x1, y1, color)

    return out


def draw_hud(
    frame: np.ndarray,
    fps: float,
    total: int,
    people: int,
    vehicles: int,
    animals: int,
    status: str = "Active",
    model_name: str = "yolov8n",
) -> np.ndarray:
    """
    Overlay a small translucent HUD panel in the top-left corner of the frame.
    Shows FPS, object counts, and tracking status.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    lines = [
        f"  MODEL : {model_name.upper()}",
        f"  FPS   : {fps:.1f}",
        f"  TOTAL : {total}",
        f"  PEOPLE: {people}",
        f"  CARS  : {vehicles}",
        f"  ANIMALS: {animals}",
        f"  STATUS: {status.upper()}",
    ]

    panel_h = len(lines) * 20 + 16
    panel_w = 200

    # Semi-transparent dark panel
    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (15, 19, 27), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Cyan border
    cv2.rectangle(out, (8, 8), (8 + panel_w, 8 + panel_h), (0, 240, 255), 1)

    # Header bar
    cv2.rectangle(out, (8, 8), (8 + panel_w, 26), (0, 240, 255), -1)
    cv2.putText(out, "VISIONTRACK AI HUD", (14, 21),
                FONT, 0.42, (15, 19, 27), 1, cv2.LINE_AA)

    # Data lines
    for i, line in enumerate(lines):
        y = 8 + 26 + i * 20 + 13
        color = (0, 255, 128) if "STATUS" in line else (200, 200, 200)
        cv2.putText(out, line, (12, y), FONT, 0.42, color, 1, cv2.LINE_AA)

    return out


# ── Statistics ───────────────────────────────────────────────────────────────

def compute_class_stats(detections: list[dict]) -> dict:
    """
    Aggregate per-group counts from a list of detection dicts.

    Returns
    -------
    dict with keys: 'total', 'person', 'vehicle', 'animal', 'other'
    """
    stats = {"total": 0, "person": 0, "vehicle": 0, "animal": 0, "other": 0}
    for det in detections:
        stats["total"] += 1
        group = det.get("group", "other")
        stats[group] = stats.get(group, 0) + 1
    return stats


# ── Persistence ──────────────────────────────────────────────────────────────

def save_screenshot(frame: np.ndarray, prefix: str = "screenshot") -> Optional[str]:
    """
    Save the current frame as a PNG to the outputs/ folder.

    Returns
    -------
    Absolute file path string on success, None on failure.
    """
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = OUTPUTS_DIR / f"{prefix}_{ts}.png"
        cv2.imwrite(str(filename), frame)
        logger.info("Screenshot saved: %s", filename)
        return str(filename)
    except Exception as exc:
        logger.error("Failed to save screenshot: %s", exc)
        return None


def log_to_csv(
    detections: list[dict],
    timestamp: Optional[str] = None,
    session_id: str = "default",
) -> None:
    """
    Append detection rows to a per-session CSV file in outputs/.

    CSV columns: timestamp, session_id, track_id, label, confidence, x1, y1, x2, y2, group
    """
    if not detections:
        return

    csv_path = OUTPUTS_DIR / f"detections_{session_id}.csv"
    file_exists = csv_path.exists()
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "timestamp", "session_id", "track_id", "label",
                    "confidence", "x1", "y1", "x2", "y2", "group"
                ])
            for det in detections:
                x1, y1, x2, y2 = det.get("bbox", (0, 0, 0, 0))
                writer.writerow([
                    ts,
                    session_id,
                    det.get("id", -1),
                    det.get("label", ""),
                    f"{det.get('confidence', 0):.4f}",
                    x1, y1, x2, y2,
                    det.get("group", "other"),
                ])
    except Exception as exc:
        logger.error("CSV log error: %s", exc)


def get_csv_path(session_id: str = "default") -> Path:
    return OUTPUTS_DIR / f"detections_{session_id}.csv"


def save_video_writer(
    frame: np.ndarray,
    output_path: Optional[str] = None,
    fps: float = 25.0,
) -> cv2.VideoWriter:
    """
    Create an OpenCV VideoWriter for the given frame shape.

    Returns
    -------
    cv2.VideoWriter (caller must call .write() and .release())
    """
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUTS_DIR / f"output_{ts}.mp4")

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    logger.info("VideoWriter opened: %s (%d×%d @ %.1f FPS)", output_path, w, h, fps)
    return writer


def save_screenshot(frame: np.ndarray, output_path: Optional[str] = None) -> str:
    """Save an annotated frame as a PNG image in outputs/."""
    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(OUTPUTS_DIR / f"screenshot_{ts}.png")
    cv2.imwrite(output_path, frame)
    logger.info("Screenshot saved: %s", output_path)
    return output_path


# ── Private Helpers ──────────────────────────────────────────────────────────

def _draw_corners(img, x1, y1, x2, y2, color):
    """Draw corner accent brackets inside the bounding box."""
    cl = CORNER_LEN
    ct = CORNER_THICK

    # Top-left
    cv2.line(img, (x1, y1), (x1 + cl, y1), color, ct)
    cv2.line(img, (x1, y1), (x1, y1 + cl), color, ct)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - cl, y1), color, ct)
    cv2.line(img, (x2, y1), (x2, y1 + cl), color, ct)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + cl, y2), color, ct)
    cv2.line(img, (x1, y2), (x1, y2 - cl), color, ct)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - cl, y2), color, ct)
    cv2.line(img, (x2, y2), (x2, y2 - cl), color, ct)


def _draw_label(img, text, x1, y1, color):
    """Draw a filled banner label above (or inside if near top) the box."""
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    banner_x1 = x1
    banner_x2 = x1 + tw + LABEL_PADDING * 2
    banner_y1 = y1 - th - LABEL_PADDING * 2
    banner_y2 = y1

    # If there's no space above, place label inside the box
    if banner_y1 < 0:
        banner_y1 = y1
        banner_y2 = y1 + th + LABEL_PADDING * 2

    cv2.rectangle(img, (banner_x1, banner_y1), (banner_x2, banner_y2), color, -1)
    cv2.putText(
        img, text,
        (banner_x1 + LABEL_PADDING, banner_y2 - LABEL_PADDING),
        FONT, FONT_SCALE, (10, 10, 10), FONT_THICK, cv2.LINE_AA,
    )
