"""
api/sync_supabase.py — Vercel Serverless Function for Supabase Logging
Safely writes detection logs and session summaries using server-side credentials.
"""

from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error


def _get_supabase_url():
    return os.environ.get("SUPABASE_URL", "").strip().rstrip("/")


def _get_supabase_key():
    return (os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or "").strip()


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
            payload = json.loads(body)
        except Exception:
            payload = {}

        supabase_url = _get_supabase_url()
        supabase_key = _get_supabase_key()

        if not supabase_url or not supabase_key:
            self._send_json(500, {
                "error": "SUPABASE_URL or SUPABASE_KEY is missing from environment variables."
            })
            return

        action = payload.get("action", "detections")

        if action == "session":
            # Session summary insert/upsert
            session_data = payload.get("data", {})
            target_url = f"{supabase_url}/rest/v1/sessions"
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates"
            }
            req_data = session_data
        else:
            # Batch detections insert
            records = payload.get("records", [])
            if not records:
                self._send_json(200, {"status": "ok", "inserted": 0})
                return
            target_url = f"{supabase_url}/rest/v1/detections"
            headers = {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal"
            }
            req_data = records

        req = urllib.request.Request(
            target_url,
            data=json.dumps(req_data).encode("utf-8"),
            headers=headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self._send_json(200, {"status": "ok", "message": "Synced to Supabase successfully."})
        except urllib.error.HTTPError as e:
            err_msg = ""
            try:
                err_msg = e.read().decode("utf-8")
            except Exception:
                err_msg = str(e)
            self._send_json(e.code, {"error": f"Supabase HTTP Error ({e.code}): {err_msg}"})
        except Exception as e:
            self._send_json(500, {"error": f"Failed to sync with Supabase: {e}"})

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
