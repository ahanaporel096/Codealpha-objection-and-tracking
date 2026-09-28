"""
ai_assistant.py — AI Vision Intelligence Assistant.
Provides automated scene analytics, activity reports, and summaries.
Keeps all underlying API configurations and credentials completely hidden.
"""

import json
import logging
import os
from pathlib import Path
import urllib.request
import urllib.error

logger = logging.getLogger("visiontrack.ai")

DEFAULT_MODEL = "openai/gpt-oss-20b"
API_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"


def _load_env_file():
    """Load configuration from .env file if present."""
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


# Attempt initial env load
_load_env_file()


def _get_api_key() -> str:
    """Retrieve API key from environment."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        _load_env_file()
        key = os.environ.get("GROQ_API_KEY", "").strip()
    return key


def is_ai_available() -> bool:
    """Check if AI intelligence is enabled."""
    return bool(_get_api_key())


# Alias for backward compatibility
is_groq_available = is_ai_available


def _get_model_name() -> str:
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def query_ai(messages: list, model: str = None, temperature: float = 0.4, max_tokens: int = 350) -> str:
    """
    Execute AI request in the background.
    """
    api_key = _get_api_key()
    if not api_key:
        raise ValueError("AI API key is not configured.")

    selected_model = model or _get_model_name()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "VisionTrack-AI/1.0 (Windows NT 10.0; Win64; x64)"
    }

    payload = {
        "model": selected_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(API_ENDPOINT, data=data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "").strip()
            return ""
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_data = json.loads(e.read().decode("utf-8"))
            err_msg = err_data.get("error", {}).get("message", str(e))
        except Exception:
            err_msg = str(e)
        logger.error("AI HTTP Error (%s): %s", e.code, err_msg)
        raise RuntimeError(f"AI Service Error ({e.code}): {err_msg}")
    except Exception as e:
        logger.error("AI service communication failed: %s", e)
        raise RuntimeError(f"Connection to AI service failed: {e}")


def generate_scene_analysis(stats: dict, session_id: str = "", extra_context: str = "") -> str:
    """
    Generate an AI insights report based on detected objects and tracking statistics.
    """
    prompt = (
        f"You are the VisionTrack AI computer vision intelligence analyst.\n"
        f"Analyze the following object detection and tracking session data:\n"
        f"- Session ID: {session_id or 'Active'}\n"
        f"- Total Objects Detected: {stats.get('total', 0)}\n"
        f"- People Count: {stats.get('person', stats.get('people', 0))}\n"
        f"- Vehicles Count: {stats.get('vehicle', stats.get('vehicles', 0))}\n"
        f"- Animals Count: {stats.get('animal', stats.get('animals', 0))}\n"
        f"- Other Classes Breakdown: {stats.get('classes', {})}\n"
    )
    if extra_context:
        prompt += f"- Additional Observations: {extra_context}\n"

    prompt += (
        "\nProvide a crisp 3-part intelligence summary:\n"
        "1. 📊 Scene Overview & Density Assessment\n"
        "2. ⚠️ Activity / Anomaly / Safety Highlights\n"
        "3. 💡 Recommended Next Action or Monitoring Focus\n"
        "Important Rule: Never mention vendor names, provider names, or API backends in your response."
    )

    messages = [
        {"role": "system", "content": "You are an expert video analytics and computer vision intelligence assistant. Provide structured, actionable, professional insights. Never mention any vendor or provider names."},
        {"role": "user", "content": prompt}
    ]

    return query_ai(messages=messages, temperature=0.3, max_tokens=400)
