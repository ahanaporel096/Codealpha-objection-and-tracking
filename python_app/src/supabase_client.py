"""
supabase_client.py — Supabase Cloud Database Client for VisionTrack AI.

Handles buffered batch logging of detections and session statistics
to Supabase using REST API (PostgREST) with zero external dependency requirements.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger("visiontrack.supabase")

# In-memory buffer to batch inserts and protect real-time video FPS
_BUFFER: List[dict] = []
_LAST_FLUSH_TIME: float = time.time()
BUFFER_SIZE_THRESHOLD = 40      # Max records before flushing
FLUSH_INTERVAL_SECONDS = 2.5    # Seconds between flushes


def _load_env_file():
    """Load configuration from .env if present."""
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_path in candidate_paths:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip('"').strip("'")
                            if key not in os.environ:
                                os.environ[key] = val
                break
            except Exception as e:
                logger.debug("Could not read %s: %s", env_path, e)


_load_env_file()


def get_supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    if not url:
        _load_env_file()
        url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    return url


def get_supabase_key() -> str:
    key = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_ANON_KEY", "")).strip()
    if not key:
        _load_env_file()
        key = os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_ANON_KEY", "")).strip()
    return key


def is_supabase_configured() -> bool:
    """Return True if both URL and Key are configured."""
    return bool(get_supabase_url() and get_supabase_key())


def _make_request(endpoint: str, method: str = "POST", data: Optional[list | dict] = None) -> Tuple[bool, str]:
    """Execute authenticated REST request to Supabase PostgREST endpoint."""
    url = f"{get_supabase_url()}/rest/v1/{endpoint.lstrip('/')}"
    key = get_supabase_key()

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True, "OK"
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_msg = e.read().decode("utf-8")
        except Exception:
            err_msg = str(e)
        logger.error("Supabase HTTP Error (%s): %s", e.code, err_msg)
        return False, f"HTTP {e.code}: {err_msg}"
    except Exception as e:
        logger.error("Supabase request failed: %s", e)
        return False, str(e)


def test_connection() -> Tuple[bool, str]:
    """Test connection to Supabase by checking the detections table."""
    if not is_supabase_configured():
        return False, "SUPABASE_URL or SUPABASE_KEY missing in .env"
    
    # Try reading 1 record from detections table
    url = f"{get_supabase_url()}/rest/v1/detections?select=id&limit=1"
    key = get_supabase_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return True, "Connected successfully to Supabase."
    except urllib.error.HTTPError as e:
        if e.code == 404 or "relation \"public.detections\" does not exist" in str(e.read().decode("utf-8", "ignore")):
            return False, "Connected to Supabase, but 'detections' table is not created yet. Please run the SQL setup script."
        return False, f"Supabase error ({e.code})"
    except Exception as e:
        return False, f"Connection failed: {e}"


def log_detection(detections: List[dict], session_id: str, force_flush: bool = False) -> None:
    """
    Buffer detections and flush to Supabase periodically to preserve video FPS.
    """
    global _BUFFER, _LAST_FLUSH_TIME

    if not is_supabase_configured() or not detections:
        return

    now_iso = datetime.utcnow().isoformat() + "Z"
    for d in detections:
        x1, y1, x2, y2 = d.get("bbox", (0, 0, 0, 0))
        record = {
            "session_id": session_id,
            "track_id": int(d.get("id", -1)),
            "label": str(d.get("label", "")),
            "confidence": float(round(d.get("confidence", 0.0), 4)),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "group_name": str(d.get("group", "other")),
            "created_at": now_iso
        }
        _BUFFER.append(record)

    # Check if buffer should flush
    time_elapsed = time.time() - _LAST_FLUSH_TIME
    if force_flush or len(_BUFFER) >= BUFFER_SIZE_THRESHOLD or time_elapsed >= FLUSH_INTERVAL_SECONDS:
        flush_buffer()


def flush_buffer() -> bool:
    """Flush any pending records to Supabase."""
    global _BUFFER, _LAST_FLUSH_TIME

    if not _BUFFER:
        return True

    records_to_send = list(_BUFFER)
    _BUFFER = []
    _LAST_FLUSH_TIME = time.time()

    ok, msg = _make_request("detections", method="POST", data=records_to_send)
    if not ok:
        logger.warning("Failed to flush %d records to Supabase: %s", len(records_to_send), msg)
    return ok


def log_session_summary(session_id: str, stats: dict, model_name: str = "yolov8n") -> bool:
    """Save high-level session summary metrics to Supabase sessions table."""
    if not is_supabase_configured():
        return False

    # First flush any remaining detection logs
    flush_buffer()

    record = {
        "id": session_id,
        "total_objects": int(stats.get("total", 0)),
        "people_count": int(stats.get("person", stats.get("people", 0))),
        "vehicle_count": int(stats.get("vehicle", stats.get("vehicles", 0))),
        "animal_count": int(stats.get("animal", stats.get("animals", 0))),
        "frames_count": int(stats.get("frames", stats.get("frames_count", 0))),
        "model_name": model_name,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    # Upsert session
    url = f"{get_supabase_url()}/rest/v1/sessions"
    key = get_supabase_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }
    req = urllib.request.Request(url, data=json.dumps(record).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return True
    except Exception as e:
        logger.error("Failed to log session summary to Supabase: %s", e)
        return False
