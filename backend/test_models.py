"""
Guardian AI — Model verification test.
Run: python test_models.py
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


print("🛡️  Guardian AI — Model Verification")
print("=" * 50)
all_ok = True

# ─── Test 1: Faster-Whisper ───────────────────────────────────
print("\n🎙️  Testing Faster-Whisper...")
try:
    from faster_whisper import WhisperModel
    wm = WhisperModel("base", device="cpu", compute_type="int8")
    # Transcribe a silent 1-second dummy (no file needed for load test)
    print("   ✅ PASS — Faster-Whisper loaded, offline STT ready")
    del wm
except Exception as e:
    print(f"   ❌ FAIL — {e}")
    all_ok = False

# ─── Test 2: BERT Text Model ──────────────────────────────────
print("\n💬  Testing BERT spam model...")
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    tok   = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Test with a known scam message
    test_scam = "URGENT: Your account will be suspended. Click here: http://bit.ly/verify"
    inputs = tok(test_scam, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_probs = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    print(f"   ✅ PASS — Labels: {label_probs}")
    del tok, model
except Exception as e:
    print(f"   ❌ FAIL — {e}")
    all_ok = False

# ─── Test 3: HuggingFace Deepfake Model ──────────────────────
print("\n🖼️   Testing deepfake detection model...")
try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    import numpy as np

    model_name = "prithivMLmods/Deep-Fake-Detector-Model"
    proc  = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()

    # Create a synthetic test image (224x224 random noise)
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = proc(images=dummy_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    label_probs = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    print(f"   ✅ PASS — Labels: {label_probs}")
    del proc, model
except Exception as e:
    print(f"   ❌ FAIL — {e}")
    all_ok = False

# ─── Test 4: Librosa (audio features) ────────────────────────
print("\n🎵  Testing librosa...")
try:
    import librosa, numpy as np
    y = np.zeros(16000)     # 1 second silence
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13)
    print(f"   ✅ PASS — MFCC shape: {mfcc.shape}")
except Exception as e:
    print(f"   ❌ FAIL — {e}")
    all_ok = False

# ─── Test 5: OpenCV ────────────────────────────────────────────
print("\n🎬  Testing OpenCV...")
try:
    import cv2, numpy as np
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    gray  = cv2.cvtColor(dummy, cv2.COLOR_BGR2GRAY)
    print(f"   ✅ PASS — OpenCV {cv2.__version__} ready")
except Exception as e:
    print(f"   ❌ FAIL — {e}")
    all_ok = False

print("\n" + "=" * 50)
if all_ok:
    print("🎉 ALL TESTS PASSED — Ready to run python app.py")
else:
    print("⚠️  Some tests failed. Check errors above.")
    sys.exit(1)