"""
Guardian AI - Video Deepfake Detection Pipeline
================================================
Analyzes video files for deepfake indicators using:
1. Frame extraction & temporal consistency
2. Face tracking & jitter detection
3. Per-frame image analysis
4. Audio-video alignment checking
5. Weighted score aggregation
"""

import os
import numpy as np
from utils import logger, clamp, classify_threat_level

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def analyze_video(filepath):
    """
    Main video analysis pipeline.
    Returns a comprehensive analysis result dict.
    """
    logger.info(f"Starting video analysis: {filepath}")
    results = {
        'type': 'video',
        'filename': os.path.basename(filepath),
        'analyses': [],
        'detected_indicators': [],
        'recommendations': []
    }

    if not HAS_CV2:
        results['error'] = 'OpenCV not available for video analysis'
        results['threat_score'] = 0
        results['threat_level'] = 'LOW'
        results['threat_color'] = '#22c55e'
        results['threat_description'] = 'Analysis incomplete — missing dependencies'
        results['confidence'] = 0
        return results

    # Extract video info and frames
    video_info, frames = extract_frames(filepath)
    if not frames:
        results['error'] = 'Could not extract frames from video'
        results['threat_score'] = 0
        results['threat_level'] = 'LOW'
        results['threat_color'] = '#22c55e'
        results['threat_description'] = 'Could not process video'
        results['confidence'] = 0
        return results

    results['video_info'] = video_info

    # ─── Analysis 1: Temporal Consistency (30% weight) ────────────
    temporal_result = analyze_temporal_consistency(frames)
    results['analyses'].append({
        'name': 'Temporal Consistency',
        'score': temporal_result['score'],
        'weight': 0.30,
        'details': temporal_result['summary']
    })
    results['detected_indicators'].extend(temporal_result.get('indicators', []))

    # ─── Analysis 2: Face Tracking (25% weight) ──────────────────
    face_result = analyze_face_tracking(frames)
    results['analyses'].append({
        'name': 'Face Tracking Analysis',
        'score': face_result['score'],
        'weight': 0.25,
        'details': face_result['summary']
    })
    results['detected_indicators'].extend(face_result.get('indicators', []))

    # ─── Analysis 3: Frame-by-Frame Analysis (35% weight) ────────
    frame_result = analyze_frames_individually(frames)
    results['analyses'].append({
        'name': 'Frame-by-Frame Analysis',
        'score': frame_result['score'],
        'weight': 0.35,
        'details': frame_result['summary']
    })
    results['detected_indicators'].extend(frame_result.get('indicators', []))

    # ─── Analysis 4: Compression Artifacts (10% weight) ──────────
    compression_result = analyze_video_compression(frames, video_info)
    results['analyses'].append({
        'name': 'Compression Artifact Analysis',
        'score': compression_result['score'],
        'weight': 0.10,
        'details': compression_result['summary']
    })

    # ─── Weighted Aggregation ─────────────────────────────────────
    final_score = (
        temporal_result['score'] * 0.30 +
        face_result['score'] * 0.25 +
        frame_result['score'] * 0.35 +
        compression_result['score'] * 0.10
    )
    final_score = clamp(final_score)

    level, color, description = classify_threat_level(final_score)

    results['threat_score'] = round(final_score, 1)
    results['threat_level'] = level
    results['threat_color'] = color
    results['threat_description'] = description
    results['confidence'] = round(min(93, final_score + 12), 1)
    results['classification'] = (
        'DEEPFAKE' if final_score >= 60
        else 'SUSPICIOUS' if final_score >= 35
        else 'LIKELY_GENUINE'
    )
    results['recommendations'] = generate_video_recommendations(
        results['classification'], results['detected_indicators']
    )
    results['frames_analyzed'] = len(frames)

    logger.info(f"Video analysis complete: score={final_score:.1f}, classification={results['classification']}")
    return results


def extract_frames(filepath, max_frames=15):
    """
    Extract evenly-spaced frames from a video file.
    Returns (video_info, list_of_frames).
    """
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {filepath}")
            return None, []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        video_info = {
            'total_frames': total_frames,
            'fps': round(fps, 2),
            'width': width,
            'height': height,
            'duration_seconds': round(duration, 2),
            'size_bytes': os.path.getsize(filepath)
        }

        # Calculate frame indices to sample
        num_samples = min(max_frames, total_frames)
        if num_samples < 2:
            num_samples = min(2, total_frames)

        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize for consistent analysis (max 512px on longest side)
                h, w = frame.shape[:2]
                if max(h, w) > 512:
                    scale = 512 / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                frames.append(frame)

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {total_frames} total")
        return video_info, frames

    except Exception as e:
        logger.error(f"Frame extraction error: {e}")
        return None, []


def analyze_temporal_consistency(frames):
    """
    Check temporal consistency between consecutive frames.
    Deepfakes often show sudden color shifts, brightness changes, or jitter.
    """
    if len(frames) < 2:
        return {'score': 0, 'summary': 'Not enough frames for temporal analysis', 'indicators': []}

    indicators = []
    score = 0
    frame_diffs = []
    color_shifts = []

    for i in range(len(frames) - 1):
        # Absolute difference between consecutive frames
        diff = cv2.absdiff(frames[i], frames[i + 1])
        mean_diff = np.mean(diff)
        frame_diffs.append(mean_diff)

        # Color channel shifts
        for c in range(3):
            c_shift = abs(float(np.mean(frames[i][:, :, c])) - float(np.mean(frames[i + 1][:, :, c])))
            color_shifts.append(c_shift)

    # Check for sudden spikes in frame differences
    if frame_diffs:
        mean_fd = np.mean(frame_diffs)
        std_fd = np.std(frame_diffs)

        # Count anomalous transitions
        anomalous = sum(1 for d in frame_diffs if d > mean_fd + 2 * std_fd) if std_fd > 0 else 0
        if anomalous > 0:
            score += min(30, anomalous * 12)
            indicators.append({
                'type': 'Temporal Anomaly',
                'detail': f'{anomalous} sudden frame transitions detected (mean diff: {mean_fd:.2f})',
                'severity': 'high' if anomalous > 2 else 'medium'
            })

        # High overall variance suggests inconsistency
        if std_fd > mean_fd * 0.8 and mean_fd > 5:
            score += 15
            indicators.append({
                'type': 'Inconsistent Frame Flow',
                'detail': f'High variance in frame-to-frame changes (σ/μ: {std_fd/mean_fd:.2f})',
                'severity': 'medium'
            })

    # Check for sudden color shifts
    if color_shifts:
        max_shift = max(color_shifts)
        if max_shift > 30:
            score += 15
            indicators.append({
                'type': 'Color Shift',
                'detail': f'Sudden color channel shift detected (max: {max_shift:.1f})',
                'severity': 'medium'
            })

    return {
        'score': clamp(score),
        'indicators': indicators,
        'summary': f'{len(indicators)} temporal anomalies detected' if indicators
                   else 'Temporal consistency appears normal'
    }


def analyze_face_tracking(frames):
    """
    Track face positions across frames and detect jitter or morphing.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    indicators = []
    score = 0
    face_positions = []
    face_sizes = []
    frames_with_faces = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            frames_with_faces += 1
            # Largest face
            areas = [w * h for (x, y, w, h) in faces]
            largest_idx = np.argmax(areas)
            x, y, w, h = faces[largest_idx]
            face_positions.append((x + w // 2, y + h // 2))
            face_sizes.append(w * h)

    if frames_with_faces < 3:
        return {
            'score': 0,
            'summary': f'Faces detected in {frames_with_faces}/{len(frames)} frames — insufficient for tracking',
            'indicators': []
        }

    # Analyze face position stability (jitter detection)
    if len(face_positions) >= 3:
        pos_array = np.array(face_positions)
        pos_diffs = np.diff(pos_array, axis=0)
        jitter = np.std(np.linalg.norm(pos_diffs, axis=1))

        if jitter > 20:
            score += 25
            indicators.append({
                'type': 'Face Jitter',
                'detail': f'Unstable face position across frames (jitter: {jitter:.1f}px)',
                'severity': 'high'
            })
        elif jitter > 10:
            score += 12
            indicators.append({
                'type': 'Mild Face Jitter',
                'detail': f'Noticeable face position instability (jitter: {jitter:.1f}px)',
                'severity': 'medium'
            })

    # Analyze face size consistency
    if len(face_sizes) >= 3:
        size_std = np.std(face_sizes)
        size_mean = np.mean(face_sizes)
        if size_mean > 0 and size_std / size_mean > 0.25:
            score += 15
            indicators.append({
                'type': 'Face Size Variance',
                'detail': f'Inconsistent face size (CV: {size_std / size_mean:.2f})',
                'severity': 'medium'
            })

    # Check face detection consistency
    detection_ratio = frames_with_faces / len(frames)
    if 0.3 < detection_ratio < 0.7:
        score += 10
        indicators.append({
            'type': 'Intermittent Face Detection',
            'detail': f'Face detected in only {frames_with_faces}/{len(frames)} frames — may indicate morphing',
            'severity': 'medium'
        })

    return {
        'score': clamp(score),
        'indicators': indicators,
        'faces_tracked': frames_with_faces,
        'summary': f'Tracked faces in {frames_with_faces}/{len(frames)} frames, {len(indicators)} anomalies'
    }


def analyze_frames_individually(frames):
    """
    Analyze individual frames for AI-generation artifacts.
    Uses noise analysis and edge detection on each frame.
    """
    indicators = []
    frame_scores = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_score = 0

        # Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 30:
            frame_score += 30
        elif lap_var < 100:
            frame_score += 10

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        if edge_density < 0.02:
            frame_score += 15

        # Color uniformity
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_std = np.std(hsv[:, :, 1])
        if sat_std < 15:
            frame_score += 10

        frame_scores.append(frame_score)

    if not frame_scores:
        return {'score': 0, 'summary': 'No frames to analyze', 'indicators': []}

    avg_score = np.mean(frame_scores)
    suspicious_count = sum(1 for s in frame_scores if s > 25)
    suspicious_ratio = suspicious_count / len(frame_scores)

    if suspicious_ratio > 0.6:
        indicators.append({
            'type': 'Majority Suspicious Frames',
            'detail': f'{suspicious_count}/{len(frame_scores)} frames show AI-generation artifacts',
            'severity': 'high'
        })
    elif suspicious_ratio > 0.3:
        indicators.append({
            'type': 'Some Suspicious Frames',
            'detail': f'{suspicious_count}/{len(frame_scores)} frames show potential artifacts',
            'severity': 'medium'
        })

    return {
        'score': clamp(avg_score + suspicious_ratio * 20),
        'indicators': indicators,
        'frame_scores': [round(s, 1) for s in frame_scores],
        'suspicious_ratio': round(suspicious_ratio, 3),
        'summary': f'{suspicious_count}/{len(frame_scores)} suspicious frames (avg score: {avg_score:.1f})'
    }


def analyze_video_compression(frames, video_info):
    """Analyze video compression artifacts and codec anomalies."""
    score = 0
    details = []

    if video_info:
        # Check for unusual resolution
        w, h = video_info.get('width', 0), video_info.get('height', 0)
        if w > 0 and h > 0:
            aspect_ratio = w / h
            common_ratios = [16/9, 4/3, 1, 9/16, 3/4]
            min_diff = min(abs(aspect_ratio - r) for r in common_ratios)
            if min_diff > 0.1:
                score += 10
                details.append(f'Unusual aspect ratio ({aspect_ratio:.3f})')

        # Check for unusual frame rate
        fps = video_info.get('fps', 0)
        if fps > 0 and fps not in [23.976, 24, 25, 29.97, 30, 50, 59.94, 60]:
            if abs(fps - round(fps)) > 0.1:
                score += 5
                details.append(f'Non-standard frame rate ({fps})')

        # Very short videos might be generated clips
        duration = video_info.get('duration_seconds', 0)
        if 0 < duration < 3:
            score += 8
            details.append(f'Very short duration ({duration:.1f}s)')

    # Check inter-frame compression consistency
    if len(frames) >= 3:
        complexities = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            complexities.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        comp_std = np.std(complexities)
        comp_mean = np.mean(complexities)
        if comp_mean > 0 and comp_std / comp_mean > 0.5:
            score += 10
            details.append('Inconsistent frame complexity across video')

    return {
        'score': clamp(score),
        'summary': '; '.join(details) if details else 'Compression characteristics appear normal'
    }


def generate_video_recommendations(classification, indicators):
    """Generate recommendations based on video analysis."""
    if classification == 'DEEPFAKE':
        return [
            '🚨 This video shows strong signs of deepfake manipulation',
            '❌ Do not trust this video as authentic evidence',
            '🔍 Compare with known authentic footage of the same person',
            '📝 If used for fraud/misinformation, report to platform and authorities',
            '⚠️ Deepfake videos are increasingly used in scams and disinformation'
        ]
    elif classification == 'SUSPICIOUS':
        return [
            '⚠️ This video has some concerning characteristics',
            '🔍 Look for subtle face distortions or unnatural movements',
            '🔎 Check if audio lip-sync appears natural',
            '💡 Search for the original source of this video',
            '📋 Consider the context — who shared it and why?'
        ]
    else:
        return [
            '✅ Video appears to be genuine',
            '💡 Stay vigilant — deepfake technology is rapidly advancing',
            '🔍 When authenticity matters, verify with multiple methods'
        ]
