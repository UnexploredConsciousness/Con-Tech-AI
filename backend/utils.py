"""
Guardian AI - Shared Utilities
Common helpers, constants, and formatting functions.
"""

import os
import uuid
import logging
from datetime import datetime
from werkzeug.utils import secure_filename

logger = logging.getLogger("guardian-ai.utils")

# ─── Configuration ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")

ALLOWED_AUDIO = {"wav", "mp3", "ogg", "flac", "m4a", "aac", "wma", "opus"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp", "tiff", "gif"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm", "flv", "wmv", "mpeg", "3gp"}

# Threat level thresholds
THREAT_THRESHOLDS = {
    "CRITICAL": 70,
    "HIGH": 50,
    "MEDIUM": 30,
    "LOW": 0,
}

# Classification labels
THREAT_COLORS = {
    "CRITICAL": "#FF0000",
    "HIGH": "#FF6600",
    "MEDIUM": "#FFCC00",
    "LOW": "#00CC44",
}

THREAT_DESCRIPTIONS = {
    "CRITICAL": "Highly likely to be malicious. Block immediately.",
    "HIGH": "Strong indicators of fraud. Exercise extreme caution.",
    "MEDIUM": "Suspicious patterns detected. Proceed with caution.",
    "LOW": "Likely genuine. No significant threats detected.",
}


# ─── File Helpers ─────────────────────────────────────────────────────────────
def allowed_file(filename: str, allowed_set: set) -> bool:
    """Check if file extension is in the allowed set."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def save_upload(file_obj, upload_dir: str) -> str:
    """Save an uploaded file with a unique name and return the path."""
    os.makedirs(upload_dir, exist_ok=True)
    ext = file_obj.filename.rsplit(".", 1)[-1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(upload_dir, unique_name)
    file_obj.save(filepath)
    logger.debug(f"Saved upload to {filepath}")
    return filepath


def cleanup_file(filepath: str) -> None:
    """Delete a temporary file, ignoring errors."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Cleaned up {filepath}")
    except Exception as e:
        logger.warning(f"Could not remove {filepath}: {e}")


# ─── Threat Classification ────────────────────────────────────────────────────
def classify_threat(score: float) -> str:
    """Convert a 0-100 threat score to a label."""
    if score >= THREAT_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif score >= THREAT_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= THREAT_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"


def classify_authenticity(score: float) -> str:
    """Convert a 0-100 deepfake score to a label."""
    if score >= 70:
        return "AI_GENERATED"
    elif score >= 40:
        return "SUSPICIOUS"
    else:
        return "GENUINE"


# ─── Response Formatting ─────────────────────────────────────────────────────
def format_response(modality: str, result: dict, request_id: str) -> dict:
    """Wrap an analyzer result in the standard API response envelope."""
    return {
        "request_id": request_id,
        "modality": modality,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result": result,
        "meta": {
            "service": "Guardian AI",
            "version": "1.1.0",
        },
    }


# ─── Score Helpers ────────────────────────────────────────────────────────────
def weighted_average(scores: dict, weights: dict) -> float:
    """
    Compute a weighted average given score and weight dicts with matching keys.
    Both dicts must have the same keys.
    """
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    return sum(scores[k] * weights[k] for k in scores) / total_weight


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


# ─── Text Helpers ─────────────────────────────────────────────────────────────
def truncate_text(text: str, max_len: int = 200) -> str:
    """Truncate text for display/logging."""
    return text[:max_len] + "..." if len(text) > max_len else text