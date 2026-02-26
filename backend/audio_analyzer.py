"""
Guardian AI - Audio Scam Detection Analyzer (v2.0)
Upgraded: Faster-Whisper (offline STT) replaces Google Speech Recognition.
Pipeline: Faster-Whisper → Keyword Detection → Behavioral Analysis → BERT Fraud
"""

import os
import re
import logging
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
logger = logging.getLogger("guardian-ai.audio")

# ─── Scam Keyword Database ────────────────────────────────────────────────────
SCAM_KEYWORDS = {
    "urgent": {
        "words": [
            "urgent", "immediately", "right now", "expire today", "limited time",
            "act now", "last chance", "final notice", "deadline", "overdue",
            "suspended", "terminated", "arrested", "warrant", "legal action",
        ],
        "weight": 1.5,
    },
    "financial": {
        "words": [
            "bank account", "credit card", "wire transfer", "gift card", "bitcoin",
            "cryptocurrency", "send money", "payment required", "fees", "refund",
            "lottery", "prize", "inheritance", "millions", "investment opportunity",
            "guaranteed return", "irs", "tax", "social security", "ssn",
        ],
        "weight": 2.0,
    },
    "authority": {
        "words": [
            "police", "fbi", "cia", "government", "microsoft", "apple", "amazon",
            "official", "badge number", "officer", "agent", "department",
            "social security administration", "medicare", "immigration",
        ],
        "weight": 1.8,
    },
    "threats": {
        "words": [
            "arrest", "jail", "prison", "lawsuit", "sue", "legal consequences",
            "deport", "criminal charges", "penalty", "fine", "frozen account",
            "hack", "compromised", "virus", "malware",
        ],
        "weight": 2.2,
    },
    "personal_info": {
        "words": [
            "social security number", "date of birth", "mother maiden name",
            "password", "pin", "cvv", "account number", "routing number",
            "verify your identity", "confirm your details",
        ],
        "weight": 2.5,
    },
}

BEHAVIORAL_PATTERNS = [
    (r"\b(do not|don't|never)\s+(tell|inform|contact)\b", 3.0, "Secrecy demand"),
    (r"\b(stay on the line|don't hang up|keep this confidential)\b", 2.5, "Isolation tactic"),
    (r"\b(i am calling from|this is an official call from)\b", 1.5, "Authority impersonation"),
    (r"\b(your computer has|we detected|we found a virus)\b", 2.0, "Tech support scam"),
    (r"\b(you have won|congratulations|selected as winner)\b", 2.0, "Lottery scam"),
    (r"\b(send gift cards|itunes card|google play card|steam card)\b", 3.0, "Gift card scam"),
    (r"\b(press 1|press 2|press [0-9])\b", 1.0, "Robocall indicator"),
    (r"\b(verify now|confirm immediately|validate your)\b", 1.8, "Verification pressure"),
]


class AudioAnalyzer:
    """
    Audio scam detection pipeline (v2.0 with Faster-Whisper):
      1. Preprocess audio to WAV (pydub)
      2. Transcribe with Faster-Whisper (offline, no API key needed)
      3. Extract audio features with librosa
      4. Keyword detection (35%)
      5. Behavioral pattern analysis (40%)
      6. BERT/ML fraud prediction (25%)
    """

    WHISPER_MODEL_SIZE = "base"   # tiny | base | small | medium
    WHISPER_COMPUTE   = "int8"    # int8 (CPU) | float16 (GPU)

    def __init__(self):
        self._analysis_count = 0
        self._whisper        = None
        self._bert_tokenizer = None
        self._bert_model     = None
        self._lr_model       = None
        self._lr_vectorizer  = None
        self._init_whisper()
        self._init_bert()
        self._load_fallback_model()

    # ── Model Initialisation ──────────────────────────────────────────────────

    def _init_whisper(self):
        """Load Faster-Whisper model once at startup."""
        try:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading Faster-Whisper '{self.WHISPER_MODEL_SIZE}' "
                f"(compute={self.WHISPER_COMPUTE})…"
            )
            self._whisper = WhisperModel(
                self.WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type=self.WHISPER_COMPUTE,
            )
            logger.info("✅ Faster-Whisper loaded — fully offline STT ready.")
        except ImportError:
            logger.warning(
                "faster-whisper not installed. "
                "Run: pip install faster-whisper  "
                "Falling back to SpeechRecognition."
            )
        except Exception as e:
            logger.warning(f"Faster-Whisper init failed: {e}")

    def _init_bert(self):
        """Load a small pretrained BERT spam/fraud classifier from HuggingFace."""
        # mrm8488/bert-tiny-finetuned-sms-spam-detection
        # ~17 MB model — downloads once, cached in ~/.cache/huggingface/
        model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            logger.info(f"Loading BERT fraud model '{model_name}'…")
            self._bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._bert_model     = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._bert_model.eval()
            logger.info("✅ BERT fraud classifier loaded.")
        except ImportError:
            logger.warning("transformers / torch not installed — BERT stage unavailable.")
        except Exception as e:
            logger.warning(f"BERT audio init failed: {e}")

    def _load_fallback_model(self):
        """Load a locally-trained logistic regression model if present."""
        model_path = Path(__file__).parent / "models" / "audio_scam_model.pkl"
        vec_path   = Path(__file__).parent / "models" / "audio_vectorizer.pkl"
        try:
            import joblib
            if model_path.exists() and vec_path.exists():
                self._lr_model      = joblib.load(model_path)
                self._lr_vectorizer = joblib.load(vec_path)
                logger.info("Local LR model loaded (used when BERT unavailable).")
        except Exception as e:
            logger.warning(f"Local model load failed: {e}")

    def is_ready(self) -> bool:
        return True

    def get_analysis_count(self) -> int:
        return self._analysis_count

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def analyze(self, filepath: str) -> dict:
        self._analysis_count += 1

        wav_path = self._preprocess(filepath)

        transcription, confidence, language = self._transcribe(wav_path)

        if wav_path != filepath and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass

        audio_features = self._extract_audio_features(filepath)

        keyword_score,    keyword_flags    = self._keyword_detection(transcription)
        behavioral_score, behavioral_flags = self._behavioral_analysis(transcription)
        ml_score,         ml_source        = self._ml_prediction(transcription)

        raw_score   = keyword_score * 0.35 + behavioral_score * 0.40 + ml_score * 0.25
        final_score = float(np.clip(raw_score, 0, 100))
        threat_level = self._classify_threat(final_score)

        return {
            "threat_level":              threat_level,
            "threat_score":              round(final_score, 2),
            "transcription":             transcription,
            "transcription_confidence":  round(confidence, 2),
            "detected_language":         language,
            "stt_engine":                "faster-whisper" if self._whisper else "google-fallback",
            "audio_features":            audio_features,
            "scores": {
                "keyword":    round(keyword_score, 2),
                "behavioral": round(behavioral_score, 2),
                "ml_model":   round(ml_score, 2),
            },
            "ml_source":           ml_source,
            "detected_keywords":   keyword_flags,
            "behavioral_patterns": behavioral_flags,
            "recommendations":     self._get_recommendations(threat_level, keyword_flags, behavioral_flags),
            "summary":             self._build_summary(threat_level, final_score, keyword_flags, behavioral_flags),
        }

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────

    def _preprocess(self, filepath: str) -> str:
        if filepath.lower().endswith(".wav"):
            return filepath
        try:
            from pydub import AudioSegment
            audio    = AudioSegment.from_file(filepath)
            audio    = audio.set_frame_rate(16000).set_channels(1)
            wav_path = filepath.rsplit(".", 1)[0] + "_converted.wav"
            audio.export(wav_path, format="wav")
            return wav_path
        except Exception as e:
            logger.warning(f"Audio conversion failed ({e}) — using original.")
            return filepath

    # ── Stage 2: Transcription (Faster-Whisper) ───────────────────────────────

    def _transcribe(self, wav_path: str) -> tuple:
        """
        Primary: Faster-Whisper (offline, no API key, multilingual).
        Fallback: SpeechRecognition (Google API, requires internet).
        Returns (text, confidence, language).
        """
        # ── Faster-Whisper ──────────────────────────────────────────────────
        if self._whisper is not None:
            try:
                segments, info = self._whisper.transcribe(
                    wav_path,
                    beam_size=5,
                    language=None,       # auto-detect language
                    vad_filter=True,     # skip silent parts
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                transcript = " ".join(seg.text.strip() for seg in segments).lower()
                avg_conf   = float(info.duration_after_vad / max(info.duration, 0.001))
                language   = getattr(info, "language", "unknown")
                logger.debug(f"Whisper: lang={language}, conf≈{avg_conf:.2f}, text='{transcript[:80]}'")
                return transcript, min(0.99, avg_conf), language
            except Exception as e:
                logger.warning(f"Faster-Whisper transcription failed: {e}")

        # ── SpeechRecognition fallback ──────────────────────────────────────
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text.lower(), 0.80, "en"
        except ImportError:
            logger.warning("SpeechRecognition not installed either.")
        except Exception as e:
            logger.warning(f"STT fallback also failed: {e}")

        return "", 0.0, "unknown"

    # ── Stage 3: Audio Feature Extraction ────────────────────────────────────

    def _extract_audio_features(self, filepath: str) -> dict:
        try:
            import librosa
            y, sr = librosa.load(filepath, sr=None, mono=True, duration=60)
            mfccs             = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            zcr               = librosa.feature.zero_crossing_rate(y)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            rms               = librosa.feature.rms(y=y)
            tempo, _          = librosa.beat.beat_track(y=y, sr=sr)
            return {
                "duration_seconds":       round(float(len(y) / sr), 2),
                "sample_rate":            int(sr),
                "tempo_bpm":              round(float(tempo), 2),
                "mean_zcr":               round(float(np.mean(zcr)), 4),
                "mean_rms_energy":        round(float(np.mean(rms)), 4),
                "mean_spectral_centroid": round(float(np.mean(spectral_centroid)), 2),
                "mfcc_mean":              [round(float(m), 4) for m in np.mean(mfccs, axis=1)],
            }
        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}")
            return {"error": "Feature extraction unavailable"}

    # ── Stage 4a: Keyword Detection ───────────────────────────────────────────

    def _keyword_detection(self, text: str) -> tuple:
        if not text:
            return 0.0, []
        flags       = []
        total_score = 0.0
        for category, info in SCAM_KEYWORDS.items():
            for word in info["words"]:
                if word in text:
                    total_score += info["weight"] * 4.0
                    flags.append({"keyword": word, "category": category, "severity": info["weight"]})
        return float(np.clip(total_score * 2.5, 0, 100)), flags

    # ── Stage 4b: Behavioral Analysis ────────────────────────────────────────

    def _behavioral_analysis(self, text: str) -> tuple:
        if not text:
            return 0.0, []
        flags        = []
        total_weight = 0.0
        for pattern, weight, label in BEHAVIORAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append({"pattern": label, "severity": weight})
                total_weight += weight
        if len(flags) >= 3:
            total_weight *= 1.3
        return float(np.clip(total_weight * 10.0, 0, 100)), flags

    # ── Stage 4c: ML Prediction ───────────────────────────────────────────────

    def _ml_prediction(self, text: str) -> tuple:
        """
        Priority order:
          1. BERT tiny spam model (HuggingFace) — most accurate
          2. Local logistic regression model
          3. Keyword-count heuristic
        """
        if not text:
            return 0.0, "no_text"

        # BERT
        if self._bert_tokenizer and self._bert_model:
            try:
                import torch
                inputs = self._bert_tokenizer(
                    text[:512],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )
                with torch.no_grad():
                    outputs = self._bert_model(**inputs)
                probs      = torch.softmax(outputs.logits, dim=1)
                spam_prob  = float(probs[0][1].item())
                return spam_prob * 100, "bert-tiny-sms-spam"
            except Exception as e:
                logger.warning(f"BERT prediction error: {e}")

        # Local LR model
        if self._lr_model and self._lr_vectorizer:
            try:
                features = self._lr_vectorizer.transform([text])
                prob     = float(self._lr_model.predict_proba(features)[0][1])
                return prob * 100, "local-logistic-regression"
            except Exception as e:
                logger.warning(f"LR prediction error: {e}")

        # Heuristic fallback
        all_words  = [w for cat in SCAM_KEYWORDS.values() for w in cat["words"]]
        hit_count  = sum(1 for w in all_words if w in text)
        return min(100.0, hit_count * 5.0), "keyword-heuristic"

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_threat(score: float) -> str:
        if score >= 70: return "CRITICAL"
        if score >= 50: return "HIGH"
        if score >= 30: return "MEDIUM"
        return "LOW"

    @staticmethod
    def _get_recommendations(threat_level, kw_flags, beh_flags):
        base = {
            "CRITICAL": [
                "❌ Hang up immediately — this is almost certainly a scam.",
                "🚫 Do NOT provide any personal or financial information.",
                "📞 Report the number to your carrier and local authorities.",
                "🔒 If you shared account details, contact your bank immediately.",
            ],
            "HIGH": [
                "⚠️ Strong scam indicators detected. Do not share personal info.",
                "🔍 Independently verify the caller's identity via official channels.",
                "📵 Consider blocking this number.",
            ],
            "MEDIUM": [
                "⚡ Suspicious patterns found. Proceed with caution.",
                "✅ Verify the caller before providing any information.",
            ],
            "LOW": [
                "✅ No major threats detected.",
                "💡 Always verify unexpected callers independently.",
            ],
        }
        tips       = list(base.get(threat_level, base["LOW"]))
        beh_labels = [b["pattern"] for b in beh_flags]
        if "Gift card scam" in beh_labels:
            tips.append("🎁 Legitimate organisations NEVER ask for gift cards as payment.")
        if "Tech support scam" in beh_labels:
            tips.append("💻 Microsoft/Apple will NEVER cold-call you about viruses.")
        return tips

    @staticmethod
    def _build_summary(threat_level, score, kw_flags, beh_flags):
        categories = list({f["category"] for f in kw_flags})
        cat_str    = ", ".join(categories) if categories else "none"
        return (
            f"Threat Level: {threat_level} (score {score:.1f}/100). "
            f"Detected {len(kw_flags)} scam keyword(s) across categories: {cat_str}. "
            f"Found {len(beh_flags)} behavioural pattern(s)."
        )