"""
Guardian AI — Pre-download all pretrained models.
Run this ONCE before starting the server.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("=" * 60)
print("Guardian AI — Downloading Pretrained Models")
print("=" * 60)

# ─── 1. Faster-Whisper (Audio STT) ────────────────────────────
print("\n[1/3] Downloading Faster-Whisper 'base' model (~150 MB)...")
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    # Test it
    print("✅ Faster-Whisper loaded successfully!")
    print(f"   Model size: base | Compute: int8 (CPU-optimised)")
    del model
except ImportError:
    print("❌ faster-whisper not installed. Run: pip install faster-whisper")
except Exception as e:
    print(f"❌ Whisper download failed: {e}")

# ─── 2. BERT SMS Spam Model (Text Analysis) ───────────────────
print("\n[2/3] Downloading BERT SMS spam model (~17 MB)...")
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(f"✅ BERT text model loaded!")
    print(f"   Model: {model_name}")
    print(f"   Labels: {model.config.id2label}")
    del tokenizer, model
except ImportError:
    print("❌ transformers not installed. Run: pip install transformers torch")
except Exception as e:
    print(f"❌ BERT text download failed: {e}")

# ─── 3. HuggingFace Deepfake Detector (Image Analysis) ────────
print("\n[3/3] Downloading deepfake detection model (~80 MB)...")
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model_name = "prithivMLmods/Deep-Fake-Detector-Model"
    processor  = AutoImageProcessor.from_pretrained(model_name)
    model      = AutoModelForImageClassification.from_pretrained(model_name)
    print(f"✅ Deepfake model loaded!")
    print(f"   Model: {model_name}")
    print(f"   Labels: {model.config.id2label}")
    del processor, model
except ImportError:
    print("❌ transformers not installed. Run: pip install transformers torch")
except Exception as e:
    print(f"❌ Deepfake model download failed: {e}")

print("\n" + "=" * 60)
print("✅ Done! All models are cached and ready.")
print("   Cache location: ~/.cache/huggingface/ and ~/.cache/whisper/")
print("   Start server with: python app.py")
print("=" * 60)