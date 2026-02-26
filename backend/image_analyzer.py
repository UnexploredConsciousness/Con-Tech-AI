"""
Guardian AI - Image Deepfake Detection Pipeline
================================================
Analyzes images for AI-generation / deepfake indicators using:
1. EXIF metadata analysis
2. Noise pattern analysis (Laplacian variance)
3. Face artifact detection (Haar Cascade + edge analysis)
4. Compression / perceptual hashing analysis
5. Statistical feature analysis
6. Weighted score aggregation
"""

import os
import io
import struct
import numpy as np
from utils import logger, clamp, classify_threat_level

# Try importing optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ExifTags
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False


# ─── Known AI Generator Signatures ───────────────────────────────

AI_SOFTWARE_TAGS = [
    'stable diffusion', 'midjourney', 'dall-e', 'dalle', 'novelai',
    'artbreeder', 'deepart', 'deepdream', 'stylegan', 'thispersondoesnotexist',
    'generated', 'ai generated', 'synthetic', 'artificial', 'neural',
    'gan', 'diffusion', 'runway', 'adobe firefly', 'bing image creator',
    'craiyon', 'nightcafe', 'starry ai', 'dream studio', 'leonardo ai',
    'playground ai', 'lexica', 'openjourney', 'dreamlike'
]


def analyze_image(filepath):
    """
    Main image analysis pipeline.
    Returns a comprehensive analysis result dict.
    """
    logger.info(f"Starting image analysis: {filepath}")
    results = {
        'type': 'image',
        'filename': os.path.basename(filepath),
        'analyses': [],
        'detected_indicators': [],
        'recommendations': []
    }

    if not HAS_PIL:
        results['error'] = 'Pillow library not available'
        results['threat_score'] = 0
        results['threat_level'] = 'LOW'
        results['threat_color'] = '#22c55e'
        results['threat_description'] = 'Analysis incomplete — missing dependencies'
        results['confidence'] = 0
        return results

    try:
        pil_image = Image.open(filepath)
        results['image_info'] = {
            'width': pil_image.width,
            'height': pil_image.height,
            'format': pil_image.format,
            'mode': pil_image.mode,
            'size_bytes': os.path.getsize(filepath)
        }
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        results['error'] = f'Failed to open image: {str(e)}'
        results['threat_score'] = 0
        results['threat_level'] = 'LOW'
        results['threat_color'] = '#22c55e'
        results['threat_description'] = 'Could not process image'
        results['confidence'] = 0
        return results

    # ─── Analysis 1: Metadata (30% weight) ────────────────────────
    metadata_result = analyze_metadata(pil_image, filepath)
    results['analyses'].append({
        'name': 'Metadata Forensics',
        'score': metadata_result['score'],
        'weight': 0.30,
        'details': metadata_result['summary'],
        'indicators': metadata_result['indicators']
    })
    results['detected_indicators'].extend(metadata_result['indicators'])

    # ─── Analysis 2: Noise Patterns (20% weight) ─────────────────
    noise_result = analyze_noise_patterns(filepath)
    results['analyses'].append({
        'name': 'Noise Pattern Analysis',
        'score': noise_result['score'],
        'weight': 0.20,
        'details': noise_result['summary']
    })
    if noise_result.get('indicator'):
        results['detected_indicators'].append(noise_result['indicator'])

    # ─── Analysis 3: Face Artifacts (20% weight) ─────────────────
    face_result = analyze_face_artifacts(filepath)
    results['analyses'].append({
        'name': 'Face Artifact Detection',
        'score': face_result['score'],
        'weight': 0.20,
        'details': face_result['summary']
    })
    results['detected_indicators'].extend(face_result.get('indicators', []))

    # ─── Analysis 4: Compression Hashing (15% weight) ────────────
    hash_result = analyze_compression(filepath, pil_image)
    results['analyses'].append({
        'name': 'Compression Analysis',
        'score': hash_result['score'],
        'weight': 0.15,
        'details': hash_result['summary']
    })

    # ─── Analysis 5: Statistical Features (15% weight) ───────────
    stats_result = analyze_statistical_features(filepath, pil_image)
    results['analyses'].append({
        'name': 'Statistical Feature Analysis',
        'score': stats_result['score'],
        'weight': 0.15,
        'details': stats_result['summary']
    })
    results['detected_indicators'].extend(stats_result.get('indicators', []))

    # ─── Weighted Aggregation ─────────────────────────────────────
    final_score = (
        metadata_result['score'] * 0.30 +
        noise_result['score'] * 0.20 +
        face_result['score'] * 0.20 +
        hash_result['score'] * 0.15 +
        stats_result['score'] * 0.15
    )
    final_score = clamp(final_score)

    level, color, description = classify_threat_level(final_score)

    results['threat_score'] = round(final_score, 1)
    results['threat_level'] = level
    results['threat_color'] = color
    results['threat_description'] = description
    results['confidence'] = round(min(95, final_score + 10), 1)
    results['classification'] = (
        'AI_GENERATED' if final_score >= 60
        else 'SUSPICIOUS' if final_score >= 35
        else 'LIKELY_GENUINE'
    )
    results['recommendations'] = generate_image_recommendations(
        results['classification'], results['detected_indicators']
    )

    logger.info(f"Image analysis complete: score={final_score:.1f}, classification={results['classification']}")
    return results


def analyze_metadata(pil_image, filepath):
    """Analyze EXIF metadata for AI generation indicators."""
    indicators = []
    score = 0

    try:
        exif_data = {}
        raw_exif = pil_image._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = str(value)
                exif_data[str(tag_name)] = str(value)

            # Check software tags for AI generators
            software = exif_data.get('Software', '').lower()
            image_desc = exif_data.get('ImageDescription', '').lower()
            user_comment = exif_data.get('UserComment', '').lower()
            all_text = f"{software} {image_desc} {user_comment}"

            for ai_tag in AI_SOFTWARE_TAGS:
                if ai_tag in all_text:
                    score += 40
                    indicators.append({
                        'type': 'AI Software Tag',
                        'detail': f'Found AI generation marker: "{ai_tag}"',
                        'severity': 'high'
                    })
                    break

            # Check for missing camera/device info (suspicious for photos)
            has_camera = any(k in exif_data for k in ['Make', 'Model', 'LensModel'])
            has_gps = any(k in exif_data for k in ['GPSInfo'])
            has_datetime = any(k in exif_data for k in ['DateTimeOriginal', 'DateTimeDigitized'])

            if not has_camera and pil_image.format == 'JPEG':
                score += 15
                indicators.append({
                    'type': 'Missing Camera Data',
                    'detail': 'No camera/device information found in EXIF',
                    'severity': 'medium'
                })

            if not has_datetime and pil_image.format == 'JPEG':
                score += 10
                indicators.append({
                    'type': 'Missing Timestamp',
                    'detail': 'No original capture timestamp found',
                    'severity': 'low'
                })
        else:
            # No EXIF at all — could be stripped (AI images often have no EXIF)
            if pil_image.format in ('JPEG', 'JPG'):
                score += 20
                indicators.append({
                    'type': 'No EXIF Data',
                    'detail': 'Image contains no EXIF metadata — common in AI-generated images',
                    'severity': 'medium'
                })
            elif pil_image.format == 'PNG':
                # PNGs don't typically have EXIF, so less suspicious
                score += 5

    except Exception as e:
        logger.warning(f"Metadata analysis error: {e}")

    return {
        'score': clamp(score),
        'indicators': indicators,
        'summary': f"{len(indicators)} metadata anomalies detected" if indicators
                   else 'Metadata appears normal'
    }


def analyze_noise_patterns(filepath):
    """
    Analyze noise patterns using Laplacian variance.
    AI-generated images tend to have unusually uniform noise → very low or very high Laplacian variance.
    """
    if not HAS_CV2:
        return {'score': 0, 'summary': 'OpenCV not available', 'indicator': None}

    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'score': 0, 'summary': 'Could not read image', 'indicator': None}

        # Laplacian variance — measures edge/noise distribution
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        lap_var = laplacian.var()

        # AI images tend to have very low Laplacian variance (unnaturally smooth)
        # or sometimes very high (over-sharpened)
        score = 0
        indicator = None

        if lap_var < 50:
            score = 60
            indicator = {
                'type': 'Unusually Smooth',
                'detail': f'Very low noise variance ({lap_var:.1f}) — typical of AI-generated images',
                'severity': 'high'
            }
        elif lap_var < 200:
            score = 30
            indicator = {
                'type': 'Low Noise Variance',
                'detail': f'Below-average noise variance ({lap_var:.1f}) — may indicate processing',
                'severity': 'medium'
            }
        elif lap_var > 5000:
            score = 40
            indicator = {
                'type': 'Excessive Sharpness',
                'detail': f'Unusually high edge contrast ({lap_var:.1f}) — may indicate artificial sharpening',
                'severity': 'medium'
            }
        else:
            score = 5

        return {
            'score': clamp(score),
            'laplacian_variance': round(lap_var, 2),
            'summary': f'Laplacian variance: {lap_var:.1f} — {"suspicious" if score > 30 else "normal range"}',
            'indicator': indicator
        }

    except Exception as e:
        logger.error(f"Noise analysis error: {e}")
        return {'score': 0, 'summary': f'Analysis error: {str(e)}', 'indicator': None}


def analyze_face_artifacts(filepath):
    """
    Detect face artifacts using Haar Cascade + edge analysis.
    AI-generated faces may have unnatural smoothing, symmetry, or edge artifacts.
    """
    if not HAS_CV2:
        return {'score': 0, 'summary': 'OpenCV not available', 'indicators': []}

    try:
        img = cv2.imread(filepath)
        if img is None:
            return {'score': 0, 'summary': 'Could not read image', 'indicators': []}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        indicators = []
        score = 0

        # Face detection
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) == 0:
            return {
                'score': 0,
                'summary': 'No faces detected — face analysis not applicable',
                'indicators': []
            }

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = gray[y:y+h, x:x+w]

            # Check for unnatural smoothness in face region
            face_lap = cv2.Laplacian(face_roi, cv2.CV_64F)
            face_var = face_lap.var()

            if face_var < 30:
                score += 25
                indicators.append({
                    'type': 'Face Smoothing',
                    'detail': f'Face {i+1}: Unusually smooth texture (variance: {face_var:.1f})',
                    'severity': 'high'
                })

            # Check face symmetry (AI faces tend to be perfectly symmetric)
            h_face, w_face = face_roi.shape
            if w_face > 20:
                left_half = face_roi[:, :w_face//2]
                right_half = cv2.flip(face_roi[:, w_face//2:], 1)
                min_w = min(left_half.shape[1], right_half.shape[1])
                if min_w > 0:
                    left_half = left_half[:, :min_w]
                    right_half = right_half[:, :min_w]
                    diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))

                    if diff < 8:
                        score += 20
                        indicators.append({
                            'type': 'Unnatural Symmetry',
                            'detail': f'Face {i+1}: Extremely symmetric (diff: {diff:.1f}) — typical of AI generation',
                            'severity': 'high'
                        })
                    elif diff < 15:
                        score += 8
                        indicators.append({
                            'type': 'High Symmetry',
                            'detail': f'Face {i+1}: Above-average symmetry (diff: {diff:.1f})',
                            'severity': 'medium'
                        })

            # Edge analysis around face boundary
            edges = cv2.Canny(face_roi, 50, 150)
            edge_density = np.mean(edges > 0)

            if edge_density < 0.03:
                score += 15
                indicators.append({
                    'type': 'Low Edge Density',
                    'detail': f'Face {i+1}: Very few edges detected — may indicate AI smoothing',
                    'severity': 'medium'
                })

        return {
            'score': clamp(score),
            'faces_detected': len(faces),
            'indicators': indicators,
            'summary': f'{len(faces)} face(s) detected, {len(indicators)} anomalies found'
        }

    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        return {'score': 0, 'summary': f'Analysis error: {str(e)}', 'indicators': []}


def analyze_compression(filepath, pil_image):
    """
    Analyze image compression characteristics using perceptual hashing.
    AI-generated images compress differently from real photographs.
    """
    score = 0
    details = []

    try:
        if HAS_IMAGEHASH:
            # Calculate multiple hash types
            phash = imagehash.phash(pil_image)
            ahash = imagehash.average_hash(pil_image)
            dhash = imagehash.dhash(pil_image)

            # Convert to binary and check bit distribution
            phash_bits = bin(int(str(phash), 16))
            ones_ratio = phash_bits.count('1') / max(len(phash_bits) - 2, 1)

            # AI images sometimes have unusual hash bit distributions
            if ones_ratio > 0.65 or ones_ratio < 0.35:
                score += 20
                details.append(f'Unusual perceptual hash distribution (ratio: {ones_ratio:.2f})')

            # Check hash entropy
            hash_str = str(phash) + str(ahash) + str(dhash)
            unique_chars = len(set(hash_str))
            if unique_chars < 6:
                score += 15
                details.append('Low hash entropy — may indicate synthetic generation')

        # Check file size vs resolution ratio
        file_size = os.path.getsize(filepath)
        pixels = pil_image.width * pil_image.height
        if pixels > 0:
            bytes_per_pixel = file_size / pixels
            # AI PNG images tend to be larger per pixel; AI JPEGs tend to be smaller
            if pil_image.format == 'PNG' and bytes_per_pixel > 4:
                score += 10
                details.append(f'High file size ratio ({bytes_per_pixel:.2f} bytes/pixel for PNG)')
            elif pil_image.format in ('JPEG', 'JPG') and bytes_per_pixel < 0.3:
                score += 10
                details.append(f'Low file size ratio ({bytes_per_pixel:.2f} bytes/pixel for JPEG)')

    except Exception as e:
        logger.warning(f"Compression analysis error: {e}")

    return {
        'score': clamp(score),
        'summary': '; '.join(details) if details else 'Compression patterns appear normal'
    }


def analyze_statistical_features(filepath, pil_image):
    """
    Analyze statistical properties of the image.
    - Color distribution
    - Channel correlation
    - Frequency domain anomalies
    """
    score = 0
    indicators = []

    try:
        img_array = np.array(pil_image.convert('RGB'))

        # Color channel statistics
        for i, channel_name in enumerate(['Red', 'Green', 'Blue']):
            channel = img_array[:, :, i].astype(float)
            mean_val = np.mean(channel)
            std_val = np.std(channel)

            # AI images sometimes have unusual color distributions
            if std_val < 20:
                score += 8
                indicators.append({
                    'type': f'Low {channel_name} Variance',
                    'detail': f'{channel_name} channel has low variance (std: {std_val:.1f})',
                    'severity': 'low'
                })

        # Check for unusual color uniformity
        if img_array.shape[2] == 3:
            r, g, b = img_array[:,:,0].astype(float), img_array[:,:,1].astype(float), img_array[:,:,2].astype(float)
            rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]

            # Extremely high cross-channel correlation is unusual
            if abs(rg_corr) > 0.98 and abs(rb_corr) > 0.98:
                score += 15
                indicators.append({
                    'type': 'Unusual Color Correlation',
                    'detail': f'Extremely high color channel correlation (r-g: {rg_corr:.3f}, r-b: {rb_corr:.3f})',
                    'severity': 'medium'
                })

        # Check for repeating patterns using autocorrelation on DCT
        if HAS_CV2:
            gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                # Resize for consistent analysis
                h, w = gray.shape
                if h > 256 and w > 256:
                    gray_resized = cv2.resize(gray, (256, 256))
                    dct = cv2.dct(np.float32(gray_resized))

                    # Check high-frequency energy ratio
                    total_energy = np.sum(np.abs(dct))
                    hf_energy = np.sum(np.abs(dct[128:, 128:]))

                    if total_energy > 0:
                        hf_ratio = hf_energy / total_energy
                        if hf_ratio < 0.01:
                            score += 12
                            indicators.append({
                                'type': 'Low High-Frequency Content',
                                'detail': f'Very low high-frequency energy ({hf_ratio:.4f}) — typical of AI-smoothed images',
                                'severity': 'medium'
                            })

    except Exception as e:
        logger.warning(f"Statistical analysis error: {e}")

    return {
        'score': clamp(score),
        'indicators': indicators,
        'summary': f'{len(indicators)} statistical anomalies detected' if indicators
                   else 'Statistical properties appear normal'
    }


def generate_image_recommendations(classification, indicators):
    """Generate recommendations based on image analysis results."""
    if classification == 'AI_GENERATED':
        return [
            '🚨 This image shows strong signs of AI generation',
            '❌ Do not trust this image as authentic proof',
            '🔍 Cross-reference with other sources before sharing',
            '📝 If received as evidence, report to relevant authorities',
            '⚠️ Be aware that AI-generated images are increasingly used in fraud'
        ]
    elif classification == 'SUSPICIOUS':
        return [
            '⚠️ This image has some suspicious characteristics',
            '🔍 Verify the source before trusting or sharing',
            '🔎 Look for additional signs of manipulation',
            '💡 Use reverse image search to check for originals',
            '📋 Check the context — who sent it and why?'
        ]
    else:
        return [
            '✅ Image appears to be genuine',
            '💡 Stay vigilant — AI generation is constantly improving',
            '🔍 When important, always verify with multiple methods'
        ]
