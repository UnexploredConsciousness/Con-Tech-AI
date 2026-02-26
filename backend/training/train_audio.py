"""
Guardian AI - Audio Scam Detection Model Training
Trains a Logistic Regression classifier on audio transcriptions.
"""

import os
import json
import logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train-audio")

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent / "data"


# ─── Synthetic Training Data ──────────────────────────────────────────────────
# In production, replace with your labeled dataset.
SCAM_TRANSCRIPTS = [
    "this is the irs calling you have an outstanding tax debt you must pay immediately or face arrest",
    "your computer has been hacked we detected a virus call microsoft support immediately at 1800",
    "congratulations you have won our lottery prize send your bank account details to claim",
    "this is your final notice your social security number has been suspended call now",
    "your amazon account has been compromised verify your credit card to restore access",
    "act now limited time offer send 500 in gift cards and receive 5000 back",
    "we are calling from your bank your account is suspended due to suspicious activity",
    "you owe back taxes and will be arrested unless you pay with gift cards today",
    "your nephew has been arrested in mexico wire transfer the bail money immediately",
    "we have compromised your webcam pay bitcoin or we will send the videos to your contacts",
    "there is a warrant out for your arrest call our legal department immediately",
    "you have been selected for a government grant no repayment needed verify your details",
    "technical department detected malware on your device do not turn off your computer",
    "press 1 to speak with an agent about your expiring auto warranty",
    "this is social security administration your number has been used in criminal activity",
    "your icloud has been hacked tap the link to verify your apple id credentials",
    "you qualify for a debt consolidation program eliminate your debt today no credit check",
    "final notice unpaid toll charges will result in suspension pay immediately via link",
    "your parcel could not be delivered re-schedule by entering your payment details",
    "you have been selected for an exclusive investment opportunity guaranteed 30 percent returns",
]

LEGITIMATE_TRANSCRIPTS = [
    "hi this is dr johnson calling to confirm your appointment tomorrow at 3pm",
    "hello this is a reminder that your subscription is due for renewal next month",
    "thank you for calling customer support how can i help you today",
    "this is a courtesy call from your dentist office regarding your upcoming cleaning",
    "hi i am calling about the job application you submitted last week",
    "good afternoon this is the pharmacy calling your prescription is ready for pickup",
    "hello this is a school calling regarding your child attendance today",
    "this is an automated reminder your package has shipped and will arrive thursday",
    "hi calling from the library to let you know your requested book is available",
    "good morning this is your financial advisor calling about your annual review",
    "hello this is the repair shop your vehicle is ready for pickup",
    "this is your insurance agent calling to discuss your renewal options",
    "hi calling to confirm delivery of your order between 2 and 4pm tomorrow",
    "hello this is the hotel confirming your reservation for next weekend",
    "this is a test call from the emergency broadcast system please disregard",
    "good afternoon calling to follow up on your recent service visit",
    "hello your appointment has been rescheduled please call us to confirm",
    "this is the school district calling with an important update for parents",
    "hi this is the real estate agent calling about the property you viewed",
    "good morning calling from the utility company regarding scheduled maintenance",
]


def create_training_data():
    texts = SCAM_TRANSCRIPTS + LEGITIMATE_TRANSCRIPTS
    labels = [1] * len(SCAM_TRANSCRIPTS) + [0] * len(LEGITIMATE_TRANSCRIPTS)
    return texts, labels


def augment_data(texts, labels, factor=3):
    """Simple augmentation: randomly drop words."""
    import random
    aug_texts, aug_labels = list(texts), list(labels)
    for text, label in zip(texts, labels):
        words = text.split()
        for _ in range(factor):
            if len(words) > 5:
                drop_idx = random.sample(range(len(words)), k=max(1, len(words) // 6))
                new_words = [w for i, w in enumerate(words) if i not in drop_idx]
                aug_texts.append(" ".join(new_words))
                aug_labels.append(label)
    return aug_texts, aug_labels


def train():
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.metrics import classification_report
    import joblib

    logger.info("Loading training data...")
    texts, labels = create_training_data()
    texts, labels = augment_data(texts, labels, factor=5)
    logger.info(f"Total samples: {len(texts)} ({sum(labels)} scam, {len(labels)-sum(labels)} legit)")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        sublinear_tf=True,
        min_df=1,
        stop_words="english",
    )

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)

    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    logger.info("\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Scam"]))

    cv_scores = cross_val_score(model, vectorizer.transform(texts), labels, cv=5, scoring="f1")
    logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "audio_scam_model.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "audio_vectorizer.pkl")
    logger.info(f"Models saved to {MODELS_DIR}")


if __name__ == "__main__":
    train()