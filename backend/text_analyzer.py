"""
Guardian AI - Text & SMS Scam Detection Analyzer (v2.0)
Upgraded: BERT pretrained spam/fraud classifier replaces TF-IDF logistic regression.
Pipeline: BERT SMS Spam Model (primary) + Rule-based forensics (supporting)
"""

import re
import logging
import warnings
import numpy as np
from pathlib import Path
from urllib.parse import urlparse

warnings.filterwarnings("ignore")
logger = logging.getLogger("guardian-ai.text")

# ─── Scam Keyword Database ─────────────────────────────────────────────────────
TEXT_SCAM_KEYWORDS = {
    "urgency": {
        "words": [
            "urgent", "immediately", "act now", "respond within", "expires today",
            "last chance", "final warning", "account suspended", "verify now",
            "24 hours", "48 hours", "limited time", "don't delay", "time sensitive",
        ],
        "weight": 1.4,
    },
    "financial_lure": {
        "words": [
            "you have won", "winner", "prize", "lottery", "unclaimed funds",
            "free gift", "free money", "cash reward", "earn money", "get rich",
            "inheritance", "investment opportunity", "double your money",
            "no risk", "guaranteed profit", "forex", "crypto opportunity",
        ],
        "weight": 2.0,
    },
    "credential_harvest": {
        "words": [
            "click here to verify", "confirm your account", "update your information",
            "login to continue", "verify your identity", "re-enter your password",
            "account will be closed", "security alert", "unauthorized access",
            "suspicious activity", "click the link below", "tap to verify",
        ],
        "weight": 2.2,
    },
    "personal_info_request": {
        "words": [
            "social security", "date of birth", "mother maiden name",
            "bank account number", "credit card", "cvv", "routing number",
            "passport number", "pin number", "otp", "one-time password",
            "verification code", "authentication code",
        ],
        "weight": 2.8,
    },
    "authority_impersonation": {
        "words": [
            "irs", "fbi", "cia", "police", "government", "microsoft support",
            "apple support", "amazon", "paypal", "your bank", "visa", "mastercard",
            "social security administration", "medicare", "healthcare.gov",
            "official notice", "legal department", "compliance team",
        ],
        "weight": 1.8,
    },
    "threat_coercion": {
        "words": [
            "legal action", "arrest warrant", "lawsuit", "court order",
            "criminal charges", "debt collection", "overdue payment",
            "penalty", "fine", "prosecution", "will be sued",
            "report to authorities", "consequences",
        ],
        "weight": 2.0,
    },
    "gift_card_scam": {
        "words": [
            "gift card", "itunes card", "google play card", "amazon gift",
            "steam card", "target gift card", "walmart gift card",
            "buy gift cards", "send card codes",
        ],
        "weight": 3.0,
    },
    "tech_support_scam": {
        "words": [
            "your device has been hacked", "virus detected", "call immediately",
            "tech support", "windows defender", "norton", "mcafee alert",
            "your ip has been compromised", "remote access", "download now",
            "install this app",
        ],
        "weight": 2.4,
    },
}

STRUCTURAL_PATTERNS = [
    (
        r"(https?://)?([a-z0-9\-]+\.)?(bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|short\.io|rb\.gy|cutt\.ly)[/\w\-?=%&]+",
        2.5, "Shortened URL detected",
    ),
    (
        r"(https?://)?(\d{1,3}\.){3}\d{1,3}[/\w]*",
        3.0, "Direct IP address URL — phishing indicator",
    ),
    (r"\b([A-Z]{2,}\s+){3,}", 1.0, "Excessive capitalisation (urgency signal)"),
    (r"(!{2,}|\?{2,})", 0.8, "Excessive punctuation (urgency signal)"),
    (r"reply\s+(stop|yes|no|1|2)\s+to\b", 1.5, "Suspicious reply instruction"),
    (r"\bfree\b.{0,30}\bno\s+cost\b|\bno\s+cost\b.{0,30}\bfree\b", 1.8, "Double free/no-cost emphasis"),
]

SUSPICIOUS_DOMAIN_PATTERNS = [
    r"[a-z]+-[a-z]+\.(tk|ml|ga|cf|gq|pw|xyz|top|click|download|work|party)$",
    r"(paypa1|paypai|gooog|amaz0n|faceb00k|micros0ft|app1e)",
    r"(login|verify|secure|account|update|confirm)\.[a-z0-9\-]+\.(com|net|org)",
    r"[a-z]{20,}\.com",
]


class TextAnalyzer:
    """
    Text scam detection pipeline (v2.0 with BERT):

    PRIMARY path (BERT available):
      BERT SMS-spam classifier          ──── 50%
      Keyword detection                 ──── 20%
      Structural patterns               ──── 15%
      URL forensics                     ──── 15%

    FALLBACK path (BERT unavailable):
      Keyword detection                 ──── 30%
      Structural patterns               ──── 25%
      URL forensics                     ──── 20%
      Linguistic analysis               ──── 15%
      Local ML / heuristic              ──── 10%
    """

    # ── Primary model: BERT tiny fine-tuned on SMS spam ───────────────────────
    # mrm8488/bert-tiny-finetuned-sms-spam-detection  (~17 MB, downloads once)
    BERT_MODEL_NAME = "mrm8488/bert-tiny-finetuned-sms-spam-detection"

    def __init__(self):
        self._analysis_count = 0
        self._bert_tokenizer = None
        self._bert_model     = None
        self._lr_model       = None
        self._lr_vectorizer  = None
        self._init_bert()
        self._load_fallback_model()

    # ── Model Init ───────────────────────────────────────────────────────────

    def _init_bert(self):
        """Load pretrained BERT SMS spam detection model from HuggingFace."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info(f"Loading BERT model '{self.BERT_MODEL_NAME}'…")
            self._bert_tokenizer = AutoTokenizer.from_pretrained(self.BERT_MODEL_NAME)
            self._bert_model     = AutoModelForSequenceClassification.from_pretrained(self.BERT_MODEL_NAME)
            self._bert_model.eval()
            logger.info(
                f"✅ BERT text model loaded. "
                f"Labels: {self._bert_model.config.id2label}"
            )
        except ImportError:
            logger.warning("transformers / torch not installed — using rule-based fallback.")
        except Exception as e:
            logger.warning(f"BERT text init failed ({e}) — using fallback.")

    def _load_fallback_model(self):
        model_path = Path(__file__).parent / "models" / "text_scam_model.pkl"
        vec_path   = Path(__file__).parent / "models" / "text_vectorizer.pkl"
        try:
            import joblib
            if model_path.exists() and vec_path.exists():
                self._lr_model      = joblib.load(model_path)
                self._lr_vectorizer = joblib.load(vec_path)
                logger.info("Local text LR model loaded.")
        except Exception as e:
            logger.warning(f"Could not load local text model: {e}")

    def is_ready(self) -> bool:
        return True

    def get_analysis_count(self) -> int:
        return self._analysis_count

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        self._analysis_count += 1
        text_lower = text.lower()

        if self._bert_model is not None:
            return self._analyze_with_bert(text, text_lower)
        else:
            return self._analyze_fallback(text, text_lower)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMARY PATH — BERT
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_with_bert(self, text: str, text_lower: str) -> dict:
        scores  = {}
        details = {}

        scores["bert"],       details["bert"]       = self._bert_predict(text)
        scores["keyword"],    details["keyword"]    = self._keyword_detection(text_lower)
        scores["structural"], details["structural"] = self._structural_analysis(text)
        scores["url"],        details["url"]        = self._url_analysis(text)

        weights = {"bert": 0.50, "keyword": 0.20, "structural": 0.15, "url": 0.15}
        final_score  = float(np.clip(sum(scores[k] * weights[k] for k in scores), 0, 100))
        threat_level = self._classify_threat(final_score)
        scam_type    = self._detect_scam_type(details)

        return {
            "threat_level":      threat_level,
            "threat_score":      round(final_score, 2),
            "scam_type":         scam_type,
            "detection_method":  "bert-pretrained",
            "model_used":        self.BERT_MODEL_NAME,
            "stage_scores":      {k: round(v, 2) for k, v in scores.items()},
            "analysis_details":  details,
            "text_length":       len(text),
            "word_count":        len(text.split()),
            "recommendations":   self._get_recommendations(threat_level, scam_type),
            "summary":           self._build_summary(threat_level, final_score, scam_type, details),
        }

    def _bert_predict(self, text: str) -> tuple:
        """Run BERT SMS spam classifier. Returns (score_0_to_100, detail_dict)."""
        findings = []
        score    = 0.0
        try:
            import torch
            inputs = self._bert_tokenizer(
                text[:512],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            with torch.no_grad():
                outputs = self._bert_model(**inputs)

            probs      = torch.softmax(outputs.logits, dim=1)[0]
            id2label   = self._bert_model.config.id2label
            label_probs = {id2label.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)}
            findings.append(f"Class probabilities: {label_probs}")

            # Identify spam class
            spam_score = 0.0
            for label, prob in label_probs.items():
                if any(k in label.lower() for k in ("spam", "scam", "fake", "fraud", "ham")):
                    # "ham" = 0 (legit), "spam" = 1 (scam) in SMS datasets
                    if "ham" in label.lower():
                        spam_score = max(spam_score, 1.0 - prob)
                        findings.append(f"Ham label '{label}': {prob:.4f} → spam={1-prob:.4f}")
                    else:
                        spam_score = max(spam_score, prob)
                        findings.append(f"Spam label '{label}': {prob:.4f}")

            # Generic fallback: higher of index 1
            if spam_score == 0.0 and len(label_probs) == 2:
                spam_score = list(label_probs.values())[1]
                findings.append(f"2-class fallback: treating class[1] as spam ({spam_score:.4f})")

            score = spam_score * 100
            findings.append(f"Final spam probability: {score:.2f}%")

        except Exception as e:
            logger.warning(f"BERT text prediction error: {e}")
            findings.append(f"Inference error: {str(e)}")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    # ══════════════════════════════════════════════════════════════════════════
    # FALLBACK PATH — Rule-based + local ML
    # ══════════════════════════════════════════════════════════════════════════

    def _analyze_fallback(self, text: str, text_lower: str) -> dict:
        scores  = {}
        details = {}

        scores["keyword"],    details["keyword"]    = self._keyword_detection(text_lower)
        scores["structural"], details["structural"] = self._structural_analysis(text)
        scores["url"],        details["url"]        = self._url_analysis(text)
        scores["linguistic"], details["linguistic"] = self._linguistic_analysis(text, text_lower)
        scores["ml"],         details["ml"]         = self._local_ml_predict(text)

        weights      = {"keyword": 0.30, "structural": 0.25, "url": 0.20, "linguistic": 0.15, "ml": 0.10}
        final_score  = float(np.clip(sum(scores[k] * weights[k] for k in scores), 0, 100))
        threat_level = self._classify_threat(final_score)
        scam_type    = self._detect_scam_type(details)

        return {
            "threat_level":      threat_level,
            "threat_score":      round(final_score, 2),
            "scam_type":         scam_type,
            "detection_method":  "rule-based-fallback",
            "model_used":        "none (install transformers for BERT)",
            "stage_scores":      {k: round(v, 2) for k, v in scores.items()},
            "analysis_details":  details,
            "text_length":       len(text),
            "word_count":        len(text.split()),
            "recommendations":   self._get_recommendations(threat_level, scam_type),
            "summary":           self._build_summary(threat_level, final_score, scam_type, details),
        }

    # ── Rule-Based Stages ─────────────────────────────────────────────────────

    def _keyword_detection(self, text: str) -> tuple:
        flags       = []
        total_score = 0.0
        for category, info in TEXT_SCAM_KEYWORDS.items():
            hits = [w for w in info["words"] if w in text]
            for hit in hits:
                total_score += info["weight"] * 3.5
                flags.append({"keyword": hit, "category": category, "weight": info["weight"]})
        return float(np.clip(total_score * 1.8, 0, 100)), {"score": round(total_score * 1.8, 2), "flags": flags}

    def _structural_analysis(self, text: str) -> tuple:
        flags        = []
        total_weight = 0.0
        for pattern, weight, label in STRUCTURAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append({"pattern": label, "weight": weight})
                total_weight += weight
        score = float(np.clip(total_weight * 12.0, 0, 100))
        return score, {"score": round(score, 2), "flags": flags}

    def _url_analysis(self, text: str) -> tuple:
        findings = []
        score    = 0.0
        urls     = re.findall(r"https?://[^\s\"\']+", text, re.IGNORECASE)
        findings.append(f"URLs found: {len(urls)}")

        for url in urls:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()

                for pat in SUSPICIOUS_DOMAIN_PATTERNS:
                    if re.search(pat, domain, re.IGNORECASE):
                        score += 35
                        findings.append(f"Suspicious domain: {domain}")
                        break

                if re.match(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", domain):
                    score += 40
                    findings.append(f"Direct IP URL: {url[:80]}")

                if re.search(r"[а-яА-Я]", domain):
                    score += 45
                    findings.append(f"Cyrillic chars in domain — homograph attack")

                if len(url) > 150:
                    score += 15
                    findings.append(f"Abnormally long URL ({len(url)} chars)")

                if domain.count(".") > 3:
                    score += 20
                    findings.append(f"Deep subdomain nesting")
            except Exception:
                pass

        if len(urls) > 2:
            score += 15
            findings.append(f"Multiple URLs ({len(urls)}) — unusual for legitimate SMS")

        return float(np.clip(score, 0, 100)), {
            "score": round(score, 2),
            "findings": findings,
            "urls_found": urls[:5],
        }

    def _linguistic_analysis(self, text: str, text_lower: str) -> tuple:
        findings = []
        score    = 0.0
        words    = text.split()
        if not words:
            return 0.0, {"score": 0.0, "findings": []}

        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)
        findings.append(f"Caps ratio: {caps_ratio:.2%}")
        if caps_ratio > 0.25:
            score += 20
            findings.append("High caps ratio")

        excl_density = text.count("!") / max(len(words), 1)
        if excl_density > 0.2:
            score += 15
            findings.append(f"High exclamation density ({excl_density:.2f}/word)")

        for phrase in ["don't tell anyone","keep this private","once in a lifetime",
                       "you were selected","act before","exclusive offer"]:
            if phrase in text_lower:
                score += 12
                findings.append(f"Manipulation phrase: '{phrase}'")

        return float(np.clip(score, 0, 100)), {"score": round(score, 2), "findings": findings}

    def _local_ml_predict(self, text: str) -> tuple:
        if self._lr_model and self._lr_vectorizer:
            try:
                prob = float(self._lr_model.predict_proba(self._lr_vectorizer.transform([text]))[0][1])
                return prob * 100, {"model": "local-lr", "spam_prob": round(prob, 4)}
            except Exception as e:
                logger.warning(f"Local LR failed: {e}")

        all_words = [w for cat in TEXT_SCAM_KEYWORDS.values() for w in cat["words"]]
        hits      = sum(1 for w in all_words if w in text.lower())
        score     = min(100.0, hits * 4.5)
        return score, {"model": "keyword-heuristic", "keyword_hits": hits}

    # ── Scam Type & Helpers ───────────────────────────────────────────────────

    def _detect_scam_type(self, details: dict) -> str:
        kw_flags    = details.get("keyword", {}).get("flags", [])
        categories  = [f["category"] for f in kw_flags]
        str_flags   = details.get("structural", {}).get("flags", [])
        str_labels  = [f["pattern"] for f in str_flags]

        if "gift_card_scam"           in categories: return "Gift Card Scam"
        if "tech_support_scam"        in categories: return "Tech Support Scam"
        if "credential_harvest"       in categories or "Shortened URL detected" in str_labels: return "Phishing / Smishing"
        if "authority_impersonation"  in categories and "threat_coercion" in categories: return "Government Impersonation"
        if "financial_lure"           in categories: return "Lottery / Prize Scam"
        if "personal_info_request"    in categories: return "Identity Theft Attempt"
        if "threat_coercion"          in categories: return "Threatening / Extortion"
        return "General Scam"

    @staticmethod
    def _classify_threat(score: float) -> str:
        if score >= 70: return "CRITICAL"
        if score >= 50: return "HIGH"
        if score >= 30: return "MEDIUM"
        return "LOW"

    @staticmethod
    def _get_recommendations(threat_level: str, scam_type: str) -> list:
        base = {
            "CRITICAL": [
                f"🚫 SCAM DETECTED: This appears to be a '{scam_type}'.",
                "❌ Do NOT click any links or call any numbers in this message.",
                "🗑️ Delete this message immediately.",
                "📞 Report to your carrier (forward to 7726 'SPAM') and relevant authorities.",
                "🔒 If you've already clicked a link, change your passwords immediately.",
            ],
            "HIGH": [
                f"⚠️ High scam probability — likely '{scam_type}'.",
                "🛑 Do not respond, click links, or provide any information.",
                "🔍 Independently verify through official channels only.",
            ],
            "MEDIUM": [
                "⚡ Suspicious message patterns detected.",
                "✅ Verify the sender's identity before responding.",
                "🔗 Do not click links without verifying the domain.",
            ],
            "LOW": [
                "✅ No major scam indicators detected.",
                "💡 Always be cautious with unsolicited messages.",
            ],
        }
        tips = list(base.get(threat_level, base["LOW"]))
        if scam_type == "Gift Card Scam":
            tips.append("🎁 No legitimate organisation will ever ask for gift card payments.")
        if scam_type == "Tech Support Scam":
            tips.append("💻 Never allow remote access to your device from an unsolicited caller.")
        if "Phishing" in scam_type:
            tips.append("🔐 Check URLs carefully — scammers use look-alike domains.")
        return tips

    @staticmethod
    def _build_summary(threat_level: str, score: float, scam_type: str, details: dict) -> str:
        kw_count  = len(details.get("keyword", {}).get("flags", []))
        url_count = len(details.get("url", {}).get("urls_found", []))
        return (
            f"Threat Level: {threat_level} (score {score:.1f}/100). "
            f"Likely scam type: {scam_type}. "
            f"Detected {kw_count} scam keyword(s) and {url_count} URL(s)."
        )