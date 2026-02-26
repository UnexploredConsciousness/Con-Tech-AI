"""
Guardian AI - Flask Backend Server
===================================
REST API for multi-modal threat detection:
  POST /api/analyze/audio  → Audio scam detection
  POST /api/analyze/image  → Image deepfake detection
  POST /api/analyze/video  → Video deepfake detection
  GET  /api/health         → Health check
"""

import os
import sys
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import (
    logger, validate_file, save_temp_file, cleanup_temp_file,
    format_response, ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS
)
from audio_analyzer import analyze_audio
from image_analyzer import analyze_image
from video_analyzer import analyze_video

# ─── App Configuration ────────────────────────────────────────────

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Determine frontend path
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')


# ─── Static File Serving (Frontend) ──────────────────────────────

@app.route('/')
def serve_index():
    """Serve the frontend index.html."""
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve frontend static files."""
    return send_from_directory(FRONTEND_DIR, filename)


# ─── Health Check ─────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify(format_response(True, {
        'status': 'ok',
        'service': 'Guardian AI Backend',
        'version': '1.0.0',
        'capabilities': {
            'audio_analysis': True,
            'image_analysis': True,
            'video_analysis': True
        }
    }))


# ─── Audio Analysis Endpoint ─────────────────────────────────────

@app.route('/api/analyze/audio', methods=['POST'])
def analyze_audio_endpoint():
    """
    Analyze an audio file for scam indicators.
    Expects multipart/form-data with 'file' field.
    """
    if 'file' not in request.files:
        return jsonify(format_response(False, error='No file provided')), 400

    file = request.files['file']
    is_valid, error = validate_file(file, ALLOWED_AUDIO_EXTENSIONS)
    if not is_valid:
        return jsonify(format_response(False, error=error)), 400

    filepath = None
    try:
        filepath = save_temp_file(file)
        start_time = time.time()

        result = analyze_audio(filepath)
        result['processing_time'] = round(time.time() - start_time, 2)

        logger.info(f"Audio analysis completed in {result['processing_time']}s")
        return jsonify(format_response(True, data=result))

    except Exception as e:
        logger.error(f"Audio analysis failed: {e}")
        return jsonify(format_response(False, error=f'Analysis failed: {str(e)}')), 500

    finally:
        if filepath:
            cleanup_temp_file(filepath)


# ─── Image Analysis Endpoint ─────────────────────────────────────

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image_endpoint():
    """
    Analyze an image file for deepfake/AI-generation indicators.
    Expects multipart/form-data with 'file' field.
    """
    if 'file' not in request.files:
        return jsonify(format_response(False, error='No file provided')), 400

    file = request.files['file']
    is_valid, error = validate_file(file, ALLOWED_IMAGE_EXTENSIONS)
    if not is_valid:
        return jsonify(format_response(False, error=error)), 400

    filepath = None
    try:
        filepath = save_temp_file(file)
        start_time = time.time()

        result = analyze_image(filepath)
        result['processing_time'] = round(time.time() - start_time, 2)

        logger.info(f"Image analysis completed in {result['processing_time']}s")
        return jsonify(format_response(True, data=result))

    except Exception as e:
        logger.error(f"Image analysis failed: {e}")
        return jsonify(format_response(False, error=f'Analysis failed: {str(e)}')), 500

    finally:
        if filepath:
            cleanup_temp_file(filepath)


# ─── Video Analysis Endpoint ─────────────────────────────────────

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video_endpoint():
    """
    Analyze a video file for deepfake indicators.
    Expects multipart/form-data with 'file' field.
    """
    if 'file' not in request.files:
        return jsonify(format_response(False, error='No file provided')), 400

    file = request.files['file']
    is_valid, error = validate_file(file, ALLOWED_VIDEO_EXTENSIONS)
    if not is_valid:
        return jsonify(format_response(False, error=error)), 400

    filepath = None
    try:
        filepath = save_temp_file(file)
        start_time = time.time()

        result = analyze_video(filepath)
        result['processing_time'] = round(time.time() - start_time, 2)

        logger.info(f"Video analysis completed in {result['processing_time']}s")
        return jsonify(format_response(True, data=result))

    except Exception as e:
        logger.error(f"Video analysis failed: {e}")
        return jsonify(format_response(False, error=f'Analysis failed: {str(e)}')), 500

    finally:
        if filepath:
            cleanup_temp_file(filepath)


# ─── Error Handlers ───────────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify(format_response(False, error='File too large. Maximum size is 50MB.')), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify(format_response(False, error='Endpoint not found')), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify(format_response(False, error='Internal server error')), 500


# ─── Run Server ───────────────────────────────────────────────────

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("  🛡️  Guardian AI Backend Server Starting...")
    logger.info("=" * 60)
    logger.info(f"  Frontend: {FRONTEND_DIR}")
    logger.info(f"  API: http://localhost:5000/api/health")
    logger.info("=" * 60)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
