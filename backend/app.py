"""
Guardian AI - Main Flask Application
Multi-Modal Threat Detection API
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import uuid
import logging
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

from audio_analyzer import AudioAnalyzer
from image_analyzer import ImageAnalyzer
from video_analyzer import VideoAnalyzer
from text_analyzer import TextAnalyzer
from utils import (
    allowed_file,
    save_upload,
    cleanup_file,
    format_response,
    UPLOAD_FOLDER,
    ALLOWED_AUDIO,
    ALLOWED_IMAGE,
    ALLOWED_VIDEO,
)

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("guardian-ai")

# ─────────────────────────────────────────────
# Initialize Analyzers
# ─────────────────────────────────────────────

audio_analyzer = AudioAnalyzer()
image_analyzer = ImageAnalyzer()
video_analyzer = VideoAnalyzer()
text_analyzer = TextAnalyzer()

# ─────────────────────────────────────────────
# Root Route (So browser doesn’t show 404)
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "🛡️ Guardian AI Backend Running",
        "version": "1.1.0",
        "available_endpoints": [
            "/api/health",
            "/api/analyze/audio",
            "/api/analyze/image",
            "/api/analyze/video",
            "/api/analyze/text",
            "/api/analyze/batch",
            "/api/stats",
            "/routes"
        ]
    })


# ─────────────────────────────────────────────
# List All Routes (Developer Tool)
# ─────────────────────────────────────────────

@app.route("/routes", methods=["GET"])
def list_routes():
    return jsonify({
        "routes": [str(rule) for rule in app.url_map.iter_rules()]
    })


# ─────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Guardian AI",
        "timestamp": datetime.utcnow().isoformat(),
        "analyzers": {
            "audio": audio_analyzer.is_ready(),
            "image": image_analyzer.is_ready(),
            "video": video_analyzer.is_ready(),
            "text": text_analyzer.is_ready(),
        },
    })


# ─────────────────────────────────────────────
# TEXT ANALYSIS (GET + POST)
# ─────────────────────────────────────────────

@app.route("/api/analyze/text", methods=["GET", "POST"])
def analyze_text():
    request_id = str(uuid.uuid4())[:8]

    try:
        # GET for quick browser testing
        if request.method == "GET":
            return jsonify({
                "message": "Use POST with JSON body: { 'text': 'your message' }",
                "example": {
                    "text": "Congratulations! You won a prize. Click now!"
                }
            })

        data = request.get_json(silent=True) or {}
        text_input = data.get("text", "").strip()

        if not text_input:
            text_input = request.form.get("text", "").strip()

        if not text_input:
            return jsonify({"error": "No text provided", "request_id": request_id}), 400

        result = text_analyzer.analyze(text_input)
        return jsonify(format_response("text", result, request_id))

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Text analysis failed", "detail": str(e)}), 500


# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────

@app.route("/api/analyze/audio", methods=["POST"])
def analyze_audio():
    request_id = str(uuid.uuid4())[:8]
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        file = request.files["file"]

        if not allowed_file(file.filename, ALLOWED_AUDIO):
            return jsonify({"error": "Unsupported audio format"}), 400

        filepath = save_upload(file, UPLOAD_FOLDER)
        result = audio_analyzer.analyze(filepath)
        return jsonify(format_response("audio", result, request_id))

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Audio analysis failed", "detail": str(e)}), 500

    finally:
        if filepath:
            cleanup_file(filepath)


# ─────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────

@app.route("/api/analyze/image", methods=["POST"])
def analyze_image():
    request_id = str(uuid.uuid4())[:8]
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["file"]

        if not allowed_file(file.filename, ALLOWED_IMAGE):
            return jsonify({"error": "Unsupported image format"}), 400

        filepath = save_upload(file, UPLOAD_FOLDER)
        result = image_analyzer.analyze(filepath)
        return jsonify(format_response("image", result, request_id))

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Image analysis failed", "detail": str(e)}), 500

    finally:
        if filepath:
            cleanup_file(filepath)


# ─────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────

@app.route("/api/analyze/video", methods=["POST"])
def analyze_video():
    request_id = str(uuid.uuid4())[:8]
    filepath = None

    try:
        if "file" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        file = request.files["file"]

        if not allowed_file(file.filename, ALLOWED_VIDEO):
            return jsonify({"error": "Unsupported video format"}), 400

        filepath = save_upload(file, UPLOAD_FOLDER)
        result = video_analyzer.analyze(filepath)
        return jsonify(format_response("video", result, request_id))

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Video analysis failed", "detail": str(e)}), 500

    finally:
        if filepath:
            cleanup_file(filepath)


# ─────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────

@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "total_analyzed": {
            "audio": audio_analyzer.get_analysis_count(),
            "image": image_analyzer.get_analysis_count(),
            "video": video_analyzer.get_analysis_count(),
            "text": text_analyzer.get_analysis_count(),
        },
        "uptime": datetime.utcnow().isoformat(),
    })


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🛡️ Guardian AI starting on port {port}...")
    app.run(host="0.0.0.0", port=port)