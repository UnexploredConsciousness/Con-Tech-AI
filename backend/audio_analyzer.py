"""
Guardian AI - Audio Scam Detection Pipeline
============================================
Analyzes audio files for phone scam indicators using:
1. Speech-to-text transcription
2. Audio feature extraction (librosa)
3. Keyword & behavioral pattern detection
4. ML model prediction
5. Weighted score aggregation
"""

import os
import re
import numpy as np
from utils import logger, clamp, classify_threat_level

# ─── Scam Keyword Database ───────────────────────────────────────

SCAM_KEYWORDS = {
    'urgent': {
        'words': [
            'urgent', 'immediately', 'right now', 'hurry', 'quickly',
            'time is running out', 'last chance', 'deadline', 'expire',
            'act now', 'don\'t delay', 'limited time', 'final notice',
            'emergency', 'critical', 'asap', 'rush', 'without delay'
        ],
        'weight': 3.0,
        'category': 'Urgency Pressure'
    },
    'financial': {
        'words': [
            'bank account', 'credit card', 'social security', 'ssn',
            'wire transfer', 'bitcoin', 'cryptocurrency', 'payment',
            'refund', 'prize', 'lottery', 'won', 'million dollars',
            'inheritance', 'tax', 'irs', 'revenue', 'outstanding balance',
            'overdue payment', 'penalty', 'fine', 'fee', 'charge',
            'gift card', 'money order', 'western union', 'account number',
            'routing number', 'pin number', 'cvv', 'expiry', 'cash',
            'deposit', 'withdrawal', 'transaction', 'frozen account'
        ],
        'weight': 4.0,
        'category': 'Financial Solicitation'
    },
    'threats': {
        'words': [
            'arrest', 'warrant', 'police', 'jail', 'prison',
            'legal action', 'lawsuit', 'court', 'prosecute',
            'suspend', 'terminate', 'cancel', 'block', 'freeze',
            'seized', 'confiscate', 'criminal', 'investigation',
            'violation', 'illegal', 'fraud', 'offense', 'penalty'
        ],
        'weight': 4.5,
        'category': 'Threat & Intimidation'
    },
    'authority': {
        'words': [
            'government', 'federal', 'agent', 'officer', 'department',
            'internal revenue', 'social security administration',
            'microsoft support', 'apple support', 'tech support',
            'customer service', 'supervisor', 'manager', 'director',
            'official', 'certified', 'authorized', 'verification',
            'compliance', 'regulatory', 'fbi', 'cia', 'homeland'
        ],
        'weight': 3.5,
        'category': 'Authority Impersonation'
    },
    'personal_info': {
        'words': [
            'verify your identity', 'confirm your', 'provide your',
            'date of birth', 'mother\'s maiden', 'password', 'login',
            'username', 'email address', 'home address', 'full name',
            'identification', 'passport', 'driver\'s license', 'id number',
            'personal information', 'sensitive data', 'verification code',
            'one-time password', 'otp', 'security question'
        ],
        'weight': 4.0,
        'category': 'Personal Information Request'
    },
    'pressure': {
        'words': [
            'don\'t tell anyone', 'keep this confidential', 'secret',
            'don\'t hang up', 'stay on the line', 'do not share',
            'between us', 'private matter', 'confidential',
            'no one else', 'trust me', 'believe me', 'honest',
            'guarantee', 'promise', 'assured', 'certain'
        ],
        'weight': 3.5,
        'category': 'Secrecy & Pressure'
    }
}

# ─── Behavioral Patterns ─────────────────────────────────────────

BEHAVIORAL_PATTERNS = [
    {
        'name': 'Urgency + Financial',
        'description': 'Combines urgency with financial requests — classic scam pattern',
        'categories': ['urgent', 'financial'],
        'score_boost': 15
    },
    {
        'name': 'Authority + Threat',
        'description': 'Impersonates authority while making threats',
        'categories': ['authority', 'threats'],
        'score_boost': 20
    },
    {
        'name': 'Threat + Personal Info',
        'description': 'Uses threats to extract personal information',
        'categories': ['threats', 'personal_info'],
        'score_boost': 18
    },
    {
        'name': 'Authority + Financial',
        'description': 'Impersonates authority to request financial action',
        'categories': ['authority', 'financial'],
        'score_boost': 16
    },
    {
        'name': 'Pressure + Personal Info',
        'description': 'Uses pressure tactics to extract personal data',
        'categories': ['pressure', 'personal_info'],
        'score_boost': 14
    },
    {
        'name': 'Full Scam Pattern',
        'description': 'Multiple scam categories detected — very high likelihood of scam',
        'categories': ['urgent', 'financial', 'threats'],
        'score_boost': 25
    }
]


def analyze_audio(filepath):
    """
    Main audio analysis pipeline.
    Returns a comprehensive analysis result dict.
    """
    logger.info(f"Starting audio analysis: {filepath}")
    results = {
        'type': 'audio',
        'filename': os.path.basename(filepath),
        'analyses': [],
        'detected_patterns': [],
        'recommendations': []
    }

    # Step 1: Transcribe audio
    transcript = transcribe_audio(filepath)
    results['transcript'] = transcript

    if not transcript or transcript == '[Transcription unavailable]':
        # Even without transcript, analyze audio features
        audio_features = extract_audio_features(filepath)
        results['audio_features'] = audio_features

        feature_score = analyze_audio_features_score(audio_features)
        results['analyses'].append({
            'name': 'Audio Feature Analysis',
            'score': feature_score,
            'weight': 1.0,
            'details': 'Analyzed audio characteristics (pitch, energy, tempo)'
        })

        final_score = clamp(feature_score)
        level, color, description = classify_threat_level(final_score)
        results['threat_score'] = round(final_score, 1)
        results['threat_level'] = level
        results['threat_color'] = color
        results['threat_description'] = description
        results['confidence'] = round(min(60, final_score + 20), 1)
        results['recommendations'] = generate_recommendations(level, [])
        return results

    # Step 2: Extract audio features
    audio_features = extract_audio_features(filepath)
    results['audio_features'] = audio_features

    # Step 3: Keyword analysis (35% weight)
    keyword_result = analyze_keywords(transcript)
    results['analyses'].append({
        'name': 'Keyword Detection',
        'score': keyword_result['score'],
        'weight': 0.35,
        'details': keyword_result['summary'],
        'found_keywords': keyword_result['found_keywords'],
        'categories_detected': keyword_result['categories']
    })

    # Step 4: Behavioral analysis (40% weight)
    behavioral_result = analyze_behavior(transcript, keyword_result['categories'])
    results['analyses'].append({
        'name': 'Behavioral Analysis',
        'score': behavioral_result['score'],
        'weight': 0.40,
        'details': behavioral_result['summary'],
        'patterns_detected': behavioral_result['patterns']
    })
    results['detected_patterns'] = behavioral_result['patterns']

    # Step 5: Audio feature scoring (25% weight)
    feature_score = analyze_audio_features_score(audio_features)
    results['analyses'].append({
        'name': 'Audio Feature Analysis',
        'score': feature_score,
        'weight': 0.25,
        'details': f"Voice characteristics analysis (pitch variance, energy, tempo)"
    })

    # Step 6: Weighted aggregation
    final_score = (
        keyword_result['score'] * 0.35 +
        behavioral_result['score'] * 0.40 +
        feature_score * 0.25
    )
    final_score = clamp(final_score)

    level, color, description = classify_threat_level(final_score)

    results['threat_score'] = round(final_score, 1)
    results['threat_level'] = level
    results['threat_color'] = color
    results['threat_description'] = description
    results['confidence'] = round(min(95, final_score + 15), 1)
    results['recommendations'] = generate_recommendations(level, keyword_result['categories'])

    logger.info(f"Audio analysis complete: score={final_score:.1f}, level={level}")
    return results


def transcribe_audio(filepath):
    """Transcribe audio file to text using SpeechRecognition."""
    try:
        import speech_recognition as sr
        from pydub import AudioSegment

        # Convert to WAV if needed
        ext = os.path.splitext(filepath)[1].lower()
        wav_path = filepath
        if ext != '.wav':
            try:
                audio = AudioSegment.from_file(filepath)
                wav_path = filepath.rsplit('.', 1)[0] + '_converted.wav'
                audio.export(wav_path, format='wav')
                logger.info(f"Converted {ext} to WAV: {wav_path}")
            except Exception as e:
                logger.warning(f"Audio conversion failed: {e}")
                return '[Transcription unavailable]'

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)

        try:
            transcript = recognizer.recognize_google(audio_data)
            logger.info(f"Transcription successful: {len(transcript)} chars")
            return transcript
        except sr.UnknownValueError:
            logger.warning("Speech not recognized")
            return '[Speech not recognized]'
        except sr.RequestError as e:
            logger.warning(f"Google API error: {e}")
            return '[Transcription unavailable]'

    except ImportError as e:
        logger.warning(f"Missing dependency for transcription: {e}")
        return '[Transcription unavailable]'
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return '[Transcription unavailable]'


def extract_audio_features(filepath):
    """Extract audio features using librosa."""
    features = {
        'duration': 0,
        'zero_crossing_rate': 0,
        'spectral_centroid': 0,
        'spectral_rolloff': 0,
        'mfcc_mean': [],
        'tempo': 0,
        'rms_energy': 0,
        'pitch_mean': 0,
        'pitch_std': 0
    }

    try:
        import librosa

        y, sr = librosa.load(filepath, sr=22050, duration=120)
        features['duration'] = round(len(y) / sr, 2)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zero_crossing_rate'] = round(float(np.mean(zcr)), 6)

        # Spectral centroid
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid'] = round(float(np.mean(sc)), 2)

        # Spectral rolloff
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff'] = round(float(np.mean(sro)), 2)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = [round(float(m), 4) for m in np.mean(mfccs, axis=1)]

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = round(float(tempo) if np.isscalar(tempo) else float(tempo[0]), 2)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_energy'] = round(float(np.mean(rms)), 6)

        # Pitch (using piptrack)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]
        if len(pitch_values) > 0:
            features['pitch_mean'] = round(float(np.mean(pitch_values)), 2)
            features['pitch_std'] = round(float(np.std(pitch_values)), 2)

    except ImportError:
        logger.warning("librosa not available, skipping audio feature extraction")
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")

    return features


def analyze_keywords(transcript):
    """Analyze transcript for scam keywords."""
    text = transcript.lower()
    found_keywords = []
    categories = []
    total_score = 0
    category_scores = {}

    for category_key, category_data in SCAM_KEYWORDS.items():
        cat_found = []
        for keyword in category_data['words']:
            count = text.count(keyword.lower())
            if count > 0:
                cat_found.append({
                    'keyword': keyword,
                    'count': count,
                    'category': category_data['category']
                })
                total_score += count * category_data['weight']

        if cat_found:
            categories.append(category_data['category'])
            category_scores[category_data['category']] = len(cat_found)
            found_keywords.extend(cat_found)

    # Normalize score to 0-100
    normalized = clamp(total_score * 3, 0, 100)

    summary_parts = []
    for cat, count in category_scores.items():
        summary_parts.append(f"{cat}: {count} keywords")
    summary = '; '.join(summary_parts) if summary_parts else 'No scam keywords detected'

    return {
        'score': normalized,
        'found_keywords': found_keywords,
        'categories': categories,
        'summary': summary
    }


def analyze_behavior(transcript, keyword_categories):
    """Analyze behavioral patterns in the transcript."""
    patterns_detected = []
    behavior_score = 0

    # Check for pattern combinations
    for pattern in BEHAVIORAL_PATTERNS:
        required_cats = pattern['categories']
        # Map category keys to category names
        required_names = []
        for key in required_cats:
            if key in SCAM_KEYWORDS:
                required_names.append(SCAM_KEYWORDS[key]['category'])

        if all(name in keyword_categories for name in required_names):
            patterns_detected.append({
                'name': pattern['name'],
                'description': pattern['description'],
                'severity': 'high' if pattern['score_boost'] >= 18 else 'medium'
            })
            behavior_score += pattern['score_boost']

    # Additional text-based behavioral analysis
    text = transcript.lower()

    # Check for excessive urgency language
    urgency_phrases = ['right now', 'immediately', 'don\'t hang up', 'stay on the line',
                       'act now', 'last chance', 'time is running out']
    urgency_count = sum(1 for phrase in urgency_phrases if phrase in text)
    if urgency_count >= 2:
        behavior_score += 10
        patterns_detected.append({
            'name': 'Repeated Urgency',
            'description': f'Multiple urgency phrases detected ({urgency_count} instances)',
            'severity': 'medium'
        })

    # Check for information extraction attempts
    info_phrases = ['tell me your', 'what is your', 'give me your', 'provide your',
                    'confirm your', 'verify your']
    info_count = sum(1 for phrase in info_phrases if phrase in text)
    if info_count >= 1:
        behavior_score += 12
        patterns_detected.append({
            'name': 'Information Extraction',
            'description': f'Attempts to extract personal information ({info_count} instances)',
            'severity': 'high'
        })

    normalized = clamp(behavior_score * 2, 0, 100)

    return {
        'score': normalized,
        'patterns': patterns_detected,
        'summary': f"{len(patterns_detected)} behavioral patterns detected" if patterns_detected
                   else "No suspicious behavioral patterns detected"
    }


def analyze_audio_features_score(features):
    """
    Score audio features for scam indicators.
    Scam calls often have: high energy variance, aggressive pitch, fast tempo.
    """
    score = 0

    # High tempo can indicate pressured speech
    if features.get('tempo', 0) > 140:
        score += 15
    elif features.get('tempo', 0) > 120:
        score += 8

    # High pitch variance can indicate emotional manipulation
    if features.get('pitch_std', 0) > 200:
        score += 12
    elif features.get('pitch_std', 0) > 100:
        score += 6

    # High RMS energy can indicate aggressive/loud speech
    if features.get('rms_energy', 0) > 0.05:
        score += 10
    elif features.get('rms_energy', 0) > 0.02:
        score += 5

    # Very high zero crossing rate can indicate stressed speech
    if features.get('zero_crossing_rate', 0) > 0.1:
        score += 8

    return clamp(score, 0, 100)


def generate_recommendations(threat_level, categories):
    """Generate actionable recommendations based on threat level."""
    recommendations = []

    if threat_level == 'CRITICAL':
        recommendations = [
            '🚨 Do NOT provide any personal or financial information',
            '📵 End the call immediately',
            '🔒 If you shared any information, contact your bank/provider immediately',
            '📝 Report this number to local authorities and FTC (reportfraud.ftc.gov)',
            '🛑 Block this caller number'
        ]
    elif threat_level == 'HIGH':
        recommendations = [
            '⚠️ Exercise extreme caution — do not share sensitive information',
            '🔍 Verify the caller\'s identity independently (call the organization directly)',
            '📵 Consider ending the call if they pressure you',
            '📝 Note the caller\'s number and claims for potential reporting'
        ]
    elif threat_level == 'MEDIUM':
        recommendations = [
            '🔍 Verify the caller\'s identity before proceeding',
            '❓ Ask specific questions that only the real organization would know',
            '📞 Offer to call them back on an official number',
            '💡 Be cautious about sharing personal information'
        ]
    else:
        recommendations = [
            '✅ No significant threats detected',
            '💡 Always stay vigilant — verify unexpected requests independently',
            '📞 When in doubt, hang up and call back on an official number'
        ]

    # Add category-specific recommendations
    if 'Financial Solicitation' in categories:
        recommendations.append('💰 Never send money or gift cards to unknown callers')
    if 'Authority Impersonation' in categories:
        recommendations.append('🏛️ Government agencies never call demanding immediate payment')
    if 'Personal Information Request' in categories:
        recommendations.append('🔐 Legitimate organizations never ask for passwords over the phone')

    return recommendations
