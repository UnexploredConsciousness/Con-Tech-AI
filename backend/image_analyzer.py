"""
Guardian AI - Image Deepfake Detection Analyzer (v2.0)
Upgraded: HuggingFace pretrained deepfake detector replaces custom CNN.
Pipeline: HuggingFace Deepfake Model (primary) + Forensic fallbacks (backup)
"""

import os
import logging
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger("guardian-ai.image")

AI_SOFTWARE_TAGS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e", "firefly",
    "generative", "ai generated", "stable-diffusion", "novelai",
    "dreamstudio", "runway", "imagen", "kandinsky",
]


class ImageAnalyzer:
    """
    Image deepfake detection pipeline (v2.0 with HuggingFace):

    PRIMARY (when model available):
      HuggingFace pretrained deepfake classifier  ──────────────── 60%
      + Metadata forensics (supporting signal)   ──────────────── 20%
      + Noise / FFT analysis  (supporting signal) ────────────── 20%

    FALLBACK (when HuggingFace model unavailable):
      Metadata forensics  30% · Noise 20% · Face 20%
      Compression 15%     · Pixel stats 15%
    """

    # ── Primary HuggingFace model ─────────────────────────────────────────────
    # prithivMLmods/Deep-Fake-Detector-Model  (~80 MB, downloads once)
    HF_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-Model"

    def __init__(self):
        self._analysis_count    = 0
        self._hf_processor      = None
        self._hf_model          = None
        self._hf_id2label       = {}
        self._init_hf_model()

    # ── Model Init ───────────────────────────────────────────────────────────

    def _init_hf_model(self):
        """
        Load pretrained deepfake detection model from HuggingFace.
        Downloads ~80 MB on first run, cached afterwards.
        """
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            logger.info(f"Loading HuggingFace model '{self.HF_MODEL_NAME}'…")
            self._hf_processor = AutoImageProcessor.from_pretrained(self.HF_MODEL_NAME)
            self._hf_model     = AutoModelForImageClassification.from_pretrained(self.HF_MODEL_NAME)
            self._hf_model.eval()
            self._hf_id2label  = self._hf_model.config.id2label
            logger.info(f"✅ HuggingFace deepfake model loaded. Labels: {self._hf_id2label}")
        except ImportError:
            logger.warning("transformers not installed — using forensic fallback pipeline.")
        except Exception as e:
            logger.warning(f"HuggingFace model load failed ({e}) — falling back to forensics.")

    def is_ready(self) -> bool:
        return True

    def get_analysis_count(self) -> int:
        return self._analysis_count

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def analyze(self, filepath: str) -> dict:
        self._analysis_count += 1

        if self._hf_model is not None:
            return self._analyze_with_hf(filepath)
        else:
            return self._analyze_forensic(filepath)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY PATH — HuggingFace pretrained model
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_with_hf(self, filepath: str) -> dict:
        """Run HuggingFace deepfake classifier + supporting forensic checks."""
        scores  = {}
        details = {}

        # Stage 1 — HuggingFace deepfake classifier (60% weight)
        scores["hf_deepfake"], details["hf_deepfake"] = self._hf_predict(filepath)

        # Stage 2 — Metadata forensics (20% weight)
        scores["metadata"],   details["metadata"]   = self._analyze_metadata(filepath)

        # Stage 3 — Noise analysis (20% weight)
        scores["noise"],      details["noise"]      = self._analyze_noise(filepath)

        weights     = {"hf_deepfake": 0.60, "metadata": 0.20, "noise": 0.20}
        final_score = float(np.clip(
            sum(scores[k] * weights[k] for k in scores), 0, 100
        ))

        classification = self._classify(final_score)
        confidence     = self._score_to_confidence(final_score, classification)

        return {
            "classification":    classification,
            "ai_probability":    round(final_score, 2),
            "confidence":        round(confidence, 2),
            "detection_method":  "huggingface-pretrained",
            "model_used":        self.HF_MODEL_NAME,
            "stage_scores":      {k: round(v, 2) for k, v in scores.items()},
            "analysis_details":  details,
            "recommendations":   self._get_recommendations(classification),
            "summary": (
                f"Classification: {classification} "
                f"(AI probability {final_score:.1f}%, "
                f"confidence {confidence:.0f}%). "
                f"Method: HuggingFace pretrained deepfake detector."
            ),
        }

    def _hf_predict(self, filepath: str) -> tuple:
        """
        Run the HuggingFace image classification model.
        Returns (score_0_to_100, detail_dict).
        """
        findings = []
        score    = 0.0
        try:
            import torch
            from PIL import Image

            image   = Image.open(filepath).convert("RGB")
            inputs  = self._hf_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self._hf_model(**inputs)

            probs      = torch.softmax(outputs.logits, dim=1)[0]
            label_probs = {
                self._hf_id2label.get(i, f"class_{i}"): float(p)
                for i, p in enumerate(probs)
            }
            findings.append(f"Raw class probabilities: {label_probs}")

            # Identify which label corresponds to "fake"
            fake_score = 0.0
            for label, prob in label_probs.items():
                label_lower = label.lower()
                if any(k in label_lower for k in ("fake", "ai", "generated", "synthetic", "deepfake")):
                    fake_score = max(fake_score, prob)
                    findings.append(f"Fake-class label '{label}': {prob:.4f}")

            # If model only has 2 classes and neither matched, index 1 = fake by convention
            if fake_score == 0.0 and len(label_probs) == 2:
                fake_score = list(label_probs.values())[1]
                findings.append(f"2-class model — treating class[1] as fake: {fake_score:.4f}")

            score = fake_score * 100
            findings.append(f"Final fake probability: {score:.2f}%")

        except Exception as e:
            logger.warning(f"HuggingFace inference failed: {e}")
            findings.append(f"Inference error: {str(e)}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK PATH — Forensic analysis (no HuggingFace model)
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_forensic(self, filepath: str) -> dict:
        """Full forensic pipeline used when HuggingFace model is unavailable."""
        scores  = {}
        details = {}

        scores["metadata"],     details["metadata"]     = self._analyze_metadata(filepath)
        scores["noise"],        details["noise"]        = self._analyze_noise(filepath)
        scores["face"],         details["face"]         = self._analyze_face_artifacts(filepath)
        scores["compression"],  details["compression"]  = self._analyze_compression(filepath)
        scores["pixel_stats"],  details["pixel_stats"]  = self._analyze_pixel_stats(filepath)

        weights = {
            "metadata": 0.30, "noise": 0.20, "face": 0.20,
            "compression": 0.15, "pixel_stats": 0.15,
        }
        final_score    = float(np.clip(sum(scores[k] * weights[k] for k in scores), 0, 100))
        classification = self._classify(final_score)
        confidence     = self._score_to_confidence(final_score, classification)

        return {
            "classification":    classification,
            "ai_probability":    round(final_score, 2),
            "confidence":        round(confidence, 2),
            "detection_method":  "forensic-fallback",
            "model_used":        "none",
            "stage_scores":      {k: round(v, 2) for k, v in scores.items()},
            "analysis_details":  details,
            "recommendations":   self._get_recommendations(classification),
            "summary": (
                f"Classification: {classification} "
                f"(AI probability {final_score:.1f}%). "
                f"Note: forensic-only mode (install transformers for better accuracy)."
            ),
        }

    # ── Forensic Stages ───────────────────────────────────────────────────────

    def _analyze_metadata(self, filepath: str) -> tuple:
        findings = []
        score    = 0.0
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img          = Image.open(filepath)
            img_format   = img.format or "UNKNOWN"
            width, height = img.size

            exif_data = img._getexif() if hasattr(img, "_getexif") and img._getexif() else {}
            exif_readable = {}
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_readable[str(tag)] = str(value)[:200]

            all_meta = " ".join(str(v).lower() for v in exif_readable.values())
            all_meta += " " + " ".join(
                str(img.info.get(k, "")).lower()
                for k in img.info
                if isinstance(img.info.get(k, ""), str)
            )

            for ai_tag in AI_SOFTWARE_TAGS:
                if ai_tag in all_meta:
                    score += 50
                    findings.append(f"AI software tag: '{ai_tag}'")
                    break

            if not exif_readable.get("Make") and not exif_readable.get("Model"):
                if img_format in ("JPEG", "JPG"):
                    score += 15
                    findings.append("No camera Make/Model in JPEG")

            if width in (512, 768, 1024, 1152, 1280) and height in (512, 768, 1024, 1152, 1280):
                score += 20
                findings.append(f"Standard AI-gen dimensions: {width}x{height}")

            if img_format == "PNG" and not exif_readable:
                score += 10
                findings.append("PNG with no EXIF — common for AI-generated images")

        except Exception as e:
            findings.append(f"Metadata error: {e}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    def _analyze_noise(self, filepath: str) -> tuple:
        findings = []
        score    = 0.0
        try:
            import cv2
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0, {"score": 0.0, "findings": ["Could not load image"]}

            lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
            findings.append(f"Laplacian variance: {lap_var:.2f}")
            if lap_var < 50:
                score += 40
                findings.append("Very low variance — suspiciously uniform sharpness")
            elif lap_var < 100:
                score += 15

            f_transform = np.fft.fft2(img.astype(np.float32))
            magnitude   = 20 * np.log(np.abs(np.fft.fftshift(f_transform)) + 1)
            h, w        = magnitude.shape
            center_e    = magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
            ratio       = center_e / (magnitude.mean() + 1e-8)
            findings.append(f"FFT center/edge ratio: {ratio:.3f}")
            if ratio > 1.5:
                score += 25
                findings.append("Unusual FFT distribution — AI artifact")
            elif ratio > 1.2:
                score += 10

        except Exception as e:
            findings.append(f"Noise error: {e}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    def _analyze_face_artifacts(self, filepath: str) -> tuple:
        findings = []
        score    = 0.0
        try:
            import cv2
            img_color = cv2.imread(filepath)
            if img_color is None:
                return 0.0, {"score": 0.0, "findings": ["Cannot load image"]}

            img_gray      = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            cascade_path  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade  = cv2.CascadeClassifier(cascade_path)
            faces         = face_cascade.detectMultiScale(img_gray, 1.1, 4, minSize=(30, 30))
            findings.append(f"Faces detected: {len(faces)}")

            if len(faces) == 0:
                return 0.0, {"score": 0.0, "findings": findings}

            for i, (x, y, w, h) in enumerate(faces):
                face_region = img_gray[y: y + h, x: x + w]
                left_half   = face_region[:, : w // 2]
                right_half  = np.fliplr(face_region[:, w // 2:])
                min_w       = min(left_half.shape[1], right_half.shape[1])
                sym_diff    = np.abs(left_half[:, :min_w].astype(float) - right_half[:, :min_w].astype(float)).mean()
                findings.append(f"Face {i+1} symmetry diff: {sym_diff:.2f}")
                if sym_diff < 8:
                    score += 30
                    findings.append(f"Face {i+1}: near-perfect symmetry — AI signature")
                elif sym_diff < 15:
                    score += 10

                face_hsv = cv2.cvtColor(img_color[y: y + h, x: x + w], cv2.COLOR_BGR2HSV)
                skin_std = float(face_hsv[:, :, 2].std())
                findings.append(f"Face {i+1} skin value std: {skin_std:.2f}")
                if skin_std < 20:
                    score += 30
                    findings.append(f"Face {i+1}: suspiciously smooth skin")
                elif skin_std < 35:
                    score += 10

        except Exception as e:
            findings.append(f"Face analysis error: {e}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    def _analyze_compression(self, filepath: str) -> tuple:
        findings = []
        score    = 0.0
        try:
            import imagehash
            from PIL import Image
            img    = Image.open(filepath)
            p_hash = imagehash.phash(img)
            a_hash = imagehash.average_hash(img)
            d_hash = imagehash.dhash(img)
            pa_diff = p_hash - a_hash
            pd_diff = p_hash - d_hash
            findings.append(f"pHash-aHash diff: {pa_diff}, pHash-dHash diff: {pd_diff}")
            if pa_diff < 5 and pd_diff < 5:
                score += 35
                findings.append("Very low hash divergence — AI-like uniformity")
            elif pa_diff < 10:
                score += 15
        except ImportError:
            findings.append("imagehash not installed — skipping")
        except Exception as e:
            findings.append(f"Compression error: {e}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    def _analyze_pixel_stats(self, filepath: str) -> tuple:
        """Simple channel statistics as a lightweight deep-learning substitute."""
        findings = []
        score    = 0.0
        try:
            from PIL import Image
            img = Image.open(filepath).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            channel_stds = [arr[:, :, c].std() for c in range(3)]
            avg_std      = float(np.mean(channel_stds))
            findings.append(f"Average channel std: {avg_std:.2f}")
            if avg_std < 40:
                score = 50
                findings.append("Low channel variance — consistent with AI imagery")
            elif avg_std < 60:
                score = 25
            else:
                score = 5
        except Exception as e:
            findings.append(f"Pixel stats error: {e}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ── Shared Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _classify(score: float) -> str:
        if score >= 70: return "AI_GENERATED"
        if score >= 40: return "SUSPICIOUS"
        return "GENUINE"

    @staticmethod
    def _score_to_confidence(score: float, classification: str) -> float:
        if classification == "AI_GENERATED":
            return min(99.0, 50 + score * 0.5)
        if classification == "GENUINE":
            return min(99.0, 50 + (100 - score) * 0.5)
        return 60.0

    @staticmethod
    def _get_recommendations(classification: str) -> list:
        recs = {
            "AI_GENERATED": [
                "🚫 This image is highly likely to be AI-generated.",
                "❌ Do not use for verification or identity purposes.",
                "📢 Report to the platform where you received it.",
                "🔍 Use reverse image search to trace its origin.",
            ],
            "SUSPICIOUS": [
                "⚠️ This image shows suspicious characteristics.",
                "🔬 Conduct additional verification before trusting.",
                "📸 Look for inconsistencies in lighting, shadows, and backgrounds.",
            ],
            "GENUINE": [
                "✅ No significant AI generation artifacts detected.",
                "💡 Always maintain healthy scepticism for sensitive content.",
            ],
        }
        return recs.get(classification, recs["GENUINE"])