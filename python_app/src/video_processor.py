"""
video_processor.py
------------------
Manages OpenCV video capture lifecycle, frame reading, resizing,
and real-time FPS calculation.
"""

import time
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum resolution sent to YOLO (wider than this is downscaled first)
MAX_INFERENCE_WIDTH = 1280


class VideoProcessor:
    """
    Wraps cv2.VideoCapture for both webcam and video-file sources.

    Usage
    -----
        vp = VideoProcessor()
        vp.open_webcam(index=0)
        ok, frame = vp.read_frame()
        vp.release()
    """

    def __init__(self, fps_window: int = 30):
        """
        Parameters
        ----------
        fps_window : int
            Number of recent frame timestamps to average for the FPS display.
        """
        self._cap: Optional[cv2.VideoCapture] = None
        self._source: Optional[str] = None
        self._is_open: bool = False

        # Rolling FPS calculation
        self._timestamps: deque = deque(maxlen=fps_window)
        self._fps: float = 0.0
        self._temp_file: Optional[str] = None  # temp file from open_file_bytes

        # Source metadata
        self.source_width: int = 0
        self.source_height: int = 0
        self.source_fps: float = 0.0
        self.total_frames: int = 0   # 0 for live streams

    # ── Source Opening ───────────────────────────────────────────────────────

    def open_webcam(self, index: int = 0) -> bool:
        """
        Open the default (or indexed) webcam.
        Tries DirectShow on Windows to avoid MSMF capture errors, with fallback.
        """
        import sys
        self.release()
        try:
            cap = None
            # On Windows, try DirectShow (CAP_DSHOW) first to prevent MSMF grabFrame errors (-1072875772)
            if sys.platform.startswith("win"):
                try:
                    c = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if c.isOpened():
                        ret, test_frame = c.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap = c
                        else:
                            c.release()
                except Exception as dshow_err:
                    logger.debug("CAP_DSHOW attempt failed: %s", dshow_err)

            # Fallback to default backend
            if cap is None:
                c = cv2.VideoCapture(index)
                if c.isOpened():
                    cap = c

            if cap is None or not cap.isOpened():
                logger.error(
                    "Webcam at index %d is unavailable. "
                    "Check that no other application is using it, "
                    "and that camera permissions are granted.",
                    index,
                )
                return False

            # Prefer higher resolution when available
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            self._cap = cap
            self._source = f"Webcam (index {index})"
            self._is_open = True
            self._cache_source_meta()
            logger.info("Webcam opened: %s × %s @ %.1f FPS",
                        self.source_width, self.source_height, self.source_fps)
            return True

        except Exception as exc:
            logger.error("Unexpected error opening webcam: %s", exc)
            return False

    def open_file(self, file_path: str) -> bool:
        """
        Open a local video file.

        Returns True on success, False on failure.
        """
        self.release()
        path = Path(file_path)

        if not path.exists():
            logger.error("Video file not found: %s", file_path)
            return False

        supported = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}
        if path.suffix.lower() not in supported:
            logger.error(
                "Unsupported video format '%s'. Supported: %s",
                path.suffix, ", ".join(supported)
            )
            return False

        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                logger.error("OpenCV could not open file: %s — file may be corrupted.", file_path)
                return False

            self._cap = cap
            self._source = str(path)
            self._is_open = True
            self._cache_source_meta()
            logger.info(
                "Video file opened: '%s' | %d × %d @ %.1f FPS | %d frames",
                path.name, self.source_width, self.source_height,
                self.source_fps, self.total_frames
            )
            return True

        except Exception as exc:
            logger.error("Unexpected error opening video file '%s': %s", file_path, exc)
            return False

    def open_file_bytes(self, file_bytes: bytes, suffix: str = ".mp4") -> bool:
        """
        Open a video from Streamlit's UploadedFile bytes (written to a temp file).
        """
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(file_bytes)
            tmp.flush()
            tmp.close()
            success = self.open_file(tmp.name)
            if success:
                self._temp_file = tmp.name  # cleaned up in release()
            else:
                os.unlink(tmp.name)
            return success
        except Exception as exc:
            logger.error("Failed to write temp video file: %s", exc)
            try:
                tmp.close()
                os.unlink(tmp.name)
            except OSError:
                pass
            return False

    # ── Frame Reading ────────────────────────────────────────────────────────

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next frame.

        Returns
        -------
        (ok: bool, frame: ndarray | None)
            ok=False when stream is exhausted or the source is not open.
        """
        if not self._is_open or self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            # For video files this is normal EOF; for webcam it signals an error
            logger.debug("read_frame: cap.read() returned False (EOF or error).")
            return False, None

        if frame is None or frame.size == 0:
            logger.warning("read_frame: received corrupt/empty frame.")
            return False, None

        # Update FPS rolling window
        now = time.monotonic()
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            self._fps = (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0

        return True, frame

    # ── Frame Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def resize_for_inference(frame: np.ndarray, max_width: int = MAX_INFERENCE_WIDTH) -> np.ndarray:
        """
        Downscale frame to max_width while preserving aspect ratio.
        Returns the original frame unchanged if it is already within bounds.
        """
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR (OpenCV default) → RGB (Streamlit/Pillow default)."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Seek / Position ──────────────────────────────────────────────────────

    def seek_to_start(self) -> None:
        """Rewind the video to frame 0 (no-op for live webcam)."""
        if self._cap and self.total_frames > 0:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.debug("Seeked to frame 0.")

    def get_position_ratio(self) -> float:
        """Return playback position as 0.0–1.0 (0.0 for live streams)."""
        if self._cap and self.total_frames > 0:
            pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            return min(pos / self.total_frames, 1.0)
        return 0.0

    # ── Release ──────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Release the underlying capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp_file is not None:
            import os
            try:
                os.unlink(self._temp_file)
            except OSError:
                pass
            self._temp_file = None
        self._is_open = False
        self._timestamps.clear()
        self._fps = 0.0
        self.source_width = 0
        self.source_height = 0
        self.source_fps = 0.0
        self.total_frames = 0
        logger.debug("VideoProcessor released.")

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def fps(self) -> float:
        """Current measured FPS (rolling average)."""
        return round(self._fps, 1)

    @property
    def source(self) -> Optional[str]:
        return self._source

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _cache_source_meta(self) -> None:
        if self._cap is None:
            return
        self.source_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.source_fps    = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
