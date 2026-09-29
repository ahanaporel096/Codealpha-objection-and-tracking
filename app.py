"""
app.py — Top-level WSGI Entrypoint for Vercel Deployment.
Serves static frontend assets (index.html, styles.css, app.js)
and provides endpoints for AI summaries, Supabase sync, and health checks.
"""

import json
import os
import urllib.error
import urllib.request
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")


# ── Static Frontend Routes ───────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/styles.css")
def serve_styles():
    return send_from_directory(BASE_DIR, "styles.css")


@app.route("/app.js")
def serve_script():
    return send_from_directory(BASE_DIR, "app.js")


# ── API Health Check ─────────────────────────────────────────────────────────
@app.route("/api")
@app.route("/api/")
@app.route("/api/health")
@app.route("/api/health/")
def health():
    return jsonify({
        "status": "healthy",
        "service": "visiontrack-ai",
        "platform": "vercel"
    })


# ── AI Scene Intelligence Proxy Endpoint ─────────────────────────────────────
@app.route("/api/ai_summary", methods=["POST", "OPTIONS"])
def api_ai_summary():
    if request.method == "OPTIONS":
        return "", 200, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "error": "GROQ_API_KEY is not configured in Vercel environment variables."
        }), 500

    data = request.get_json(silent=True) or {}
    stats = data.get("stats", {})
    session_id = data.get("session_id", "web-session")
    model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip() or "openai/gpt-oss-20b"

    prompt = (
        f"You are the VisionTrack AI computer vision intelligence analyst.\n"
        f"Analyze the following object detection and tracking session data:\n"
        f"- Session ID: {session_id}\n"
        f"- Total Objects Detected: {stats.get('total', 0)}\n"
        f"- People Count: {stats.get('person', 0)}\n"
        f"- Vehicles Count: {stats.get('vehicle', 0)}\n"
        f"- Animals Count: {stats.get('animal', 0)}\n"
        f"- Frame Count: {stats.get('frames', 0)}\n\n"
        "Provide a crisp 3-part intelligence summary:\n"
        "1. 📊 Scene Overview & Density Assessment\n"
        "2. ⚠️ Activity / Anomaly / Safety Highlights\n"
        "3. 💡 Recommended Next Action or Monitoring Focus\n"
        "Note: Format in Markdown. Do not mention vendor or API provider names."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert video analytics and computer vision intelligence assistant. Provide structured, actionable, professional insights. Never mention any vendor or provider names like Groq."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 400
    }

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
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
            return jsonify({"summary": content})
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_msg = json.loads(e.read().decode("utf-8")).get("error", {}).get("message", str(e))
        except Exception:
            err_msg = str(e)
        return jsonify({"error": f"AI Service Error ({e.code}): {err_msg}"}), e.code
    except Exception as e:
        return jsonify({"error": f"Connection to AI service failed: {e}"}), 500


# ── Supabase Cloud Sync Endpoint ─────────────────────────────────────────────
@app.route("/api/sync_supabase", methods=["POST", "OPTIONS"])
def api_sync_supabase():
    if request.method == "OPTIONS":
        return "", 200, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }

    supabase_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    supabase_key = (os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or "").strip()

    if not supabase_url or not supabase_key:
        return jsonify({
            "error": "SUPABASE_URL or SUPABASE_KEY is missing from environment variables."
        }), 500

    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "detections")

    if action == "session":
        target_url = f"{supabase_url}/rest/v1/sessions"
        req_data = payload.get("data", {})
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }
    else:
        records = payload.get("records", [])
        if not records:
            return jsonify({"status": "ok", "inserted": 0})
        target_url = f"{supabase_url}/rest/v1/detections"
        req_data = records
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }

    req = urllib.request.Request(
        target_url,
        data=json.dumps(req_data).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return jsonify({"status": "ok", "message": "Synced to Supabase successfully."})
    except urllib.error.HTTPError as e:
        err_msg = ""
        try:
            err_msg = e.read().decode("utf-8")
        except Exception:
            err_msg = str(e)
        return jsonify({"error": f"Supabase HTTP Error ({e.code}): {err_msg}"}), e.code
    except Exception as e:
        return jsonify({"error": f"Failed to sync with Supabase: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
