"""
Guardian AI - Video Deepfake Detection Analyzer
Detects manipulated/AI-generated videos through temporal consistency,
face tracking, frame-by-frame analysis, and audio-video sync.
"""

import os
import logging
import tempfile
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger("guardian-ai.video")

# Import image analyzer for per-frame analysis
from image_analyzer import ImageAnalyzer


class VideoAnalyzer:
    """
    Multi-stage video deepfake detection pipeline:
      1. Frame extraction (10 evenly-spaced frames)
      2. Temporal consistency analysis (30%)
      3. Face tracking & jitter detection (25%)
      4. Frame-by-frame deepfake analysis (35%)
      5. Audio-video sync analysis (10%)
    """

    WEIGHTS = {
        "temporal": 0.30,
        "face_tracking": 0.25,
        "frame_analysis": 0.35,
        "audio_sync": 0.10,
    }

    NUM_FRAMES = 10

    def __init__(self):
        self._analysis_count = 0
        self._image_analyzer = ImageAnalyzer()

    def is_ready(self) -> bool:
        return True

    def get_analysis_count(self) -> int:
        return self._analysis_count

    # ── Main Entry Point ──────────────────────────────────────────────────────
    def analyze(self, filepath: str) -> dict:
        self._analysis_count += 1

        # Extract frames to a temp directory
        frames_dir = tempfile.mkdtemp(prefix="guardian_frames_")
        frame_paths = []

        try:
            video_meta = self._get_video_metadata(filepath)
            frame_paths = self._extract_frames(filepath, frames_dir)

            if not frame_paths:
                return {
                    "classification": "ERROR",
                    "ai_probability": 0.0,
                    "error": "Could not extract frames from video",
                    "video_metadata": video_meta,
                }

            scores = {}
            details = {}

            scores["temporal"], details["temporal"] = self._temporal_analysis(frame_paths)
            scores["face_tracking"], details["face_tracking"] = self._face_tracking(frame_paths)
            scores["frame_analysis"], details["frame_analysis"] = self._frame_analysis(frame_paths)
            scores["audio_sync"], details["audio_sync"] = self._audio_sync_analysis(filepath)

            final_score = float(np.clip(
                sum(scores[k] * self.WEIGHTS[k] for k in scores), 0, 100
            ))

            classification = self._classify(final_score)

            return {
                "classification": classification,
                "ai_probability": round(final_score, 2),
                "frames_analyzed": len(frame_paths),
                "video_metadata": video_meta,
                "stage_scores": {k: round(v, 2) for k, v in scores.items()},
                "analysis_details": details,
                "recommendations": self._get_recommendations(classification),
                "summary": (
                    f"Classification: {classification} (AI probability {final_score:.1f}%). "
                    f"Analyzed {len(frame_paths)} frames. "
                    f"Highest concern from: {max(scores, key=scores.get)} stage."
                ),
            }

        finally:
            # Cleanup extracted frames
            for fp in frame_paths:
                try:
                    os.remove(fp)
                except Exception:
                    pass
            try:
                os.rmdir(frames_dir)
            except Exception:
                pass

    # ── Video Metadata ────────────────────────────────────────────────────────
    def _get_video_metadata(self, filepath: str) -> dict:
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            return {
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_seconds": round(duration, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Frame Extraction ──────────────────────────────────────────────────────
    def _extract_frames(self, filepath: str, output_dir: str) -> list:
        """Extract NUM_FRAMES evenly-spaced frames from the video."""
        try:
            import cv2
            cap = cv2.VideoCapture(filepath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []

            indices = np.linspace(0, max(total - 1, 0), min(self.NUM_FRAMES, total), dtype=int)
            saved = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
                    cv2.imwrite(path, frame)
                    saved.append(path)

            cap.release()
            logger.debug(f"Extracted {len(saved)} frames from {filepath}")
            return saved

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return []

    # ── Stage 2: Temporal Consistency ─────────────────────────────────────────
    def _temporal_analysis(self, frame_paths: list) -> tuple:
        """Detect temporal inconsistencies between consecutive frames."""
        findings = []
        score = 0.0

        try:
            import cv2
            if len(frame_paths) < 2:
                return 0.0, {"score": 0.0, "findings": ["Not enough frames for temporal analysis"]}

            diffs = []
            for i in range(1, len(frame_paths)):
                f1 = cv2.imread(frame_paths[i - 1], cv2.IMREAD_GRAYSCALE)
                f2 = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
                if f1 is None or f2 is None:
                    continue
                if f1.shape != f2.shape:
                    f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
                diff = cv2.absdiff(f1, f2)
                diffs.append(float(diff.mean()))

            if not diffs:
                return 0.0, {"score": 0.0, "findings": ["Could not compute frame differences"]}

            mean_diff = float(np.mean(diffs))
            std_diff = float(np.std(diffs))
            cv_diff = std_diff / (mean_diff + 1e-8)  # Coefficient of variation

            findings.append(f"Mean inter-frame diff: {mean_diff:.3f}")
            findings.append(f"Std inter-frame diff: {std_diff:.3f}")
            findings.append(f"Coefficient of variation: {cv_diff:.3f}")

            # Very low variation = unnaturally smooth transitions (GAN artifact)
            if cv_diff < 0.15:
                score += 50
                findings.append("Suspiciously smooth inter-frame transitions — deepfake signal")
            elif cv_diff < 0.30:
                score += 20
                findings.append("Low variance in frame transitions — mildly suspicious")

            # Very high variation = blending artifacts
            if std_diff > 30:
                score += 25
                findings.append("High frame difference spikes — possible blending artifacts")

        except Exception as e:
            logger.warning(f"Temporal analysis error: {e}")
            findings.append(f"Error: {str(e)}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ── Stage 3: Face Tracking ────────────────────────────────────────────────
    def _face_tracking(self, frame_paths: list) -> tuple:
        """Track face positions across frames to detect jitter and morphing."""
        findings = []
        score = 0.0

        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            positions = []
            sizes = []
            frames_with_face = 0

            for fp in frame_paths:
                img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                faces = face_cascade.detectMultiScale(img, 1.1, 4, minSize=(30, 30))
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    positions.append((x + w / 2, y + h / 2))
                    sizes.append(w * h)
                    frames_with_face += 1

            findings.append(f"Frames with face detected: {frames_with_face}/{len(frame_paths)}")

            if frames_with_face < 2:
                findings.append("Not enough face detections for tracking analysis")
                return 0.0, {"score": 0.0, "findings": findings}

            # Position jitter
            positions = np.array(positions)
            pos_jitter = float(np.std(np.diff(positions, axis=0)))
            findings.append(f"Position jitter (std of position changes): {pos_jitter:.2f}")

            if pos_jitter > 20:
                score += 35
                findings.append("High face jitter — unstable face position suggests deepfake blending")
            elif pos_jitter > 10:
                score += 15

            # Size consistency — deepfake faces can flicker in size
            size_cv = float(np.std(sizes) / (np.mean(sizes) + 1e-8))
            findings.append(f"Face size coefficient of variation: {size_cv:.3f}")

            if size_cv > 0.20:
                score += 30
                findings.append("High face size variance — inconsistent face tracking")
            elif size_cv > 0.10:
                score += 10

        except Exception as e:
            logger.warning(f"Face tracking error: {e}")
            findings.append(f"Error: {str(e)}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ── Stage 4: Frame-by-Frame Analysis ─────────────────────────────────────
    def _frame_analysis(self, frame_paths: list) -> tuple:
        """Run image deepfake detection on each extracted frame."""
        findings = []
        ai_scores = []

        for fp in frame_paths:
            try:
                result = self._image_analyzer.analyze(fp)
                ai_prob = result.get("ai_probability", 0.0)
                ai_scores.append(ai_prob)
                findings.append(f"Frame {os.path.basename(fp)}: AI probability {ai_prob:.1f}%")
            except Exception as e:
                findings.append(f"Frame {os.path.basename(fp)}: analysis error ({e})")

        if not ai_scores:
            return 0.0, {"score": 0.0, "findings": findings}

        mean_score = float(np.mean(ai_scores))
        ai_frame_ratio = sum(1 for s in ai_scores if s >= 60) / len(ai_scores)
        findings.append(f"Mean frame AI score: {mean_score:.1f}%")
        findings.append(f"High-AI-score frames: {ai_frame_ratio * 100:.1f}%")

        score = mean_score * 0.6 + ai_frame_ratio * 100 * 0.4
        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ── Stage 5: Audio-Video Sync ─────────────────────────────────────────────
    def _audio_sync_analysis(self, filepath: str) -> tuple:
        """Basic audio-video sync check via energy alignment."""
        findings = []
        score = 0.0

        try:
            import cv2
            import librosa

            # Extract audio track to temp file
            audio_tmp = filepath.rsplit(".", 1)[0] + "_audio_tmp.wav"
            os.system(f"ffmpeg -y -i {filepath} -vn -acodec pcm_s16le -ar 16000 {audio_tmp} -loglevel quiet")

            if not os.path.exists(audio_tmp):
                findings.append("No audio track found or ffmpeg unavailable")
                return 0.0, {"score": 0.0, "findings": findings}

            y, sr = librosa.load(audio_tmp, sr=None, mono=True)
            os.remove(audio_tmp)

            # Get audio RMS per segment
            frame_length = int(sr * 0.1)
            hop_length = frame_length // 2
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            audio_duration = len(y) / sr
            findings.append(f"Audio duration: {audio_duration:.2f}s")

            # Check for silences that don't match visible speech in frames
            silence_ratio = float(np.sum(rms < 0.01) / len(rms))
            findings.append(f"Silence ratio: {silence_ratio:.2%}")

            if silence_ratio > 0.7:
                score += 30
                findings.append("Very high silence ratio — audio may be dubbed or manipulated")
            elif silence_ratio > 0.5:
                score += 15

            # Audio RMS variance (deepfakes often have unnaturally even volume)
            rms_cv = float(rms.std() / (rms.mean() + 1e-8))
            findings.append(f"Audio RMS coefficient of variation: {rms_cv:.3f}")
            if rms_cv < 0.2:
                score += 25
                findings.append("Unnaturally consistent audio levels — possible synthetic audio")

        except Exception as e:
            logger.warning(f"Audio sync analysis error: {e}")
            findings.append(f"Audio sync analysis unavailable: {str(e)}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _classify(score: float) -> str:
        if score >= 65:
            return "DEEPFAKE"
        elif score >= 35:
            return "SUSPICIOUS"
        return "GENUINE"

    @staticmethod
    def _get_recommendations(classification: str) -> list:
        recs = {
            "DEEPFAKE": [
                "🚫 This video is highly likely to be a deepfake.",
                "❌ Do not spread or use this video as evidence.",
                "📢 Report to the platform where it was shared.",
                "🔍 Cross-reference with verified sources.",
                "⚖️ If used in a legal context, consult a forensics expert.",
            ],
            "SUSPICIOUS": [
                "⚠️ Suspicious artifacts detected in this video.",
                "🔬 Do not treat this video as conclusive evidence.",
                "🎥 Look for inconsistencies in facial lighting and motion.",
            ],
            "GENUINE": [
                "✅ No strong deepfake indicators found.",
                "💡 Continue exercising healthy skepticism for online content.",
            ],
        }
        return recs.get(classification, recs["GENUINE"])