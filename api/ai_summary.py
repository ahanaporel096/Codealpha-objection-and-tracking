"""
api/ai_summary.py — Vercel Serverless Function for AI Scene Intelligence
Uses Groq API via server-side environment variables. Zero heavy ML dependencies.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def _get_api_key():
    return os.environ.get("GROQ_API_KEY", "").strip()


def _get_model():
    return os.environ.get("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"

        try:
            data = json.loads(body)
        except Exception:
            data = {}

        api_key = _get_api_key()
        if not api_key:
            self._send_json(500, {
                "error": "GROQ_API_KEY is not configured in Vercel environment variables."
            })
            return

        stats = data.get("stats", {})
        session_id = data.get("session_id", "web-session")

        prompt = (
            f"You are the VisionTrack AI computer vision intelligence analyst.\n"
            f"Analyze the following object detection and tracking session data:\n"
            f"- Session ID: {session_id}\n"
            f"- Total Objects Detected: {stats.get('total', 0)}\n"
            f"- People Count: {stats.get('person', 0)}\n"
            f"- Vehicles Count: {stats.get('vehicle', 0)}\n"
            f"- Animals Count: {stats.get('animal', 0)}\n"
            f"- Frame Count: {stats.get('frames', 0)}\n"
            f"- Class Breakdown: {stats.get('classes', {})}\n\n"
            "Provide a crisp 3-part intelligence summary:\n"
            "1. 📊 Scene Overview & Density Assessment\n"
            "2. ⚠️ Activity / Anomaly / Safety Highlights\n"
            "3. 💡 Recommended Next Action or Monitoring Focus\n"
            "Note: Format in Markdown. Do not mention vendor or API provider names."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert video analytics and computer vision intelligence assistant. Provide structured, actionable, professional insights. Never mention any vendor or provider names like Groq."
            },
            {"role": "user", "content": prompt}
        ]

        payload = {
            "model": _get_model(),
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 400
        }

        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VisionTrack-Web/1.0"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                content = choices[0]["message"]["content"] if choices else "No summary generated."
                self._send_json(200, {"summary": content})
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            self._send_json(e.code, {"error": f"AI Service Error ({e.code}): {err_msg}"})
        except Exception as e:
            self._send_json(500, {"error": f"Connection to AI service failed: {e}"})

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
