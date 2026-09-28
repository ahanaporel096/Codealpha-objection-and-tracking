import os
from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/")
@app.route("/api")
@app.route("/api/")
def api_root():
    return jsonify({
        "status": "ok",
        "message": "Object Detection & Tracking API is running on Vercel"
    })


@app.route("/health")
@app.route("/health/")
@app.route("/api/health")
@app.route("/api/health/")
def health():
    return jsonify({
        "status": "healthy",
        "service": "object-detection-api",
        "platform": "vercel"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
