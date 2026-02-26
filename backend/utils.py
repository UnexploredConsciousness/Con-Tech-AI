"""
Guardian AI - Shared Utilities
==============================
File validation, temp file management, response formatting, and logging.
"""

import os
import uuid
import logging
import tempfile
from datetime import datetime

# ─── Logging Config ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('GuardianAI')

# ─── Constants ────────────────────────────────────────────────────
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a', 'wma', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp', 'tiff'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv'}

MAX_FILE_SIZE_MB = 50
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'guardian_ai_uploads')


def ensure_upload_folder():
    """Create upload folder if it doesn't exist."""
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    return UPLOAD_FOLDER


def get_file_extension(filename):
    """Extract file extension (lowercase, no dot)."""
    if '.' in filename:
        return filename.rsplit('.', 1)[1].lower()
    return ''


def validate_file(file, allowed_extensions):
    """
    Validate uploaded file.
    Returns (is_valid, error_message).
    """
    if not file or file.filename == '':
        return False, 'No file selected'

    ext = get_file_extension(file.filename)
    if ext not in allowed_extensions:
        return False, f'Invalid file type .{ext}. Allowed: {", ".join(sorted(allowed_extensions))}'

    return True, None


def save_temp_file(file):
    """
    Save uploaded file to temp directory with a unique name.
    Returns the temp file path.
    """
    ensure_upload_folder()
    ext = get_file_extension(file.filename)
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)
    logger.info(f"Saved temp file: {filepath} ({os.path.getsize(filepath)} bytes)")
    return filepath


def cleanup_temp_file(filepath):
    """Remove a temp file safely."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup {filepath}: {e}")


def format_response(success, data=None, error=None):
    """Standard API response format."""
    response = {
        'success': success,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    }
    if data is not None:
        response['data'] = data
    if error is not None:
        response['error'] = error
    return response


def classify_threat_level(score):
    """
    Classify a 0-100 threat score into a threat level.
    Returns (level, color, description).
    """
    if score >= 70:
        return 'CRITICAL', '#ef4444', 'High probability of threat detected. Immediate action recommended.'
    elif score >= 50:
        return 'HIGH', '#f97316', 'Significant indicators of potential threat. Exercise extreme caution.'
    elif score >= 30:
        return 'MEDIUM', '#eab308', 'Some suspicious indicators detected. Proceed with caution.'
    else:
        return 'LOW', '#22c55e', 'No significant threats detected. Content appears genuine.'


def clamp(value, min_val=0, max_val=100):
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
