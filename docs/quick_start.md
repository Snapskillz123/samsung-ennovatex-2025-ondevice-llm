# Quick Start Guide - Samsung EnnovateX 2025

Get up and running with the On-Device Fine-Tuning Framework in 10 minutes!

## Prerequisites

- **Python 3.9+** with pip
- **CUDA GPU** (recommended for training, RTX 3060+ or better)
- **Samsung Galaxy S24 Emulator** or physical device
- **8GB+ RAM** on development machine
- **20GB+ free disk space**

## Step 1: Setup Environment

### Windows
```cmd
# Run the setup script
tools\setup.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/Mac
```bash
# Run the setup script
chmod +x tools/setup.sh
./tools/setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 2: Prepare Your Data

### Export WhatsApp Chats
1. Open WhatsApp on your phone
2. Go to Settings → Chats → Chat History → Export Chat
3. Choose "Without Media" 
4. Save the `.txt` file to `data/raw/my_whatsapp_export.txt`

### Process the Data
```bash
# Parse WhatsApp export
python src/data_pipeline/parse_whatsapp.py \
    --input data/raw/my_whatsapp_export.txt \
    --output data/processed/whatsapp_parsed.jsonl

# Apply privacy filtering
python src/data_pipeline/privacy_filter.py \
    --input data/processed/whatsapp_parsed.jsonl \
    --output data/processed/whatsapp_filtered.jsonl

# Split into train/validation
head -n -50 data/processed/whatsapp_filtered.jsonl > data/communication/train.jsonl
tail -n 50 data/processed/whatsapp_filtered.jsonl > data/communication/val.jsonl
```

## Step 3: Train Your First Adapter

```bash
# Train communication adapter
python src/training/train_qlora.py \
    --config configs/communication_adapter.json \
    --train-data data/communication/train.jsonl \
    --val-data data/communication/val.jsonl
```

**Expected output:**
```
Loading tokenizer: microsoft/Phi-3-mini-4k-instruct
Loading quantized model...
Setting up LoRA configuration
Trainable parameters: 2,359,296 (0.83%)
Starting QLoRA training
Training: 100%|██████████| 50/50 [15:23<00:00, 18.46s/it]
Adapter saved to: out/comm_v1/adapter
Manifest saved to: out/comm_v1/adapter/adapter.json
```

## Step 4: Create Adapter Manifest

```bash
python src/conversion/create_manifest.py \
    --adapter-id comm_v1 \
    --adapter-dir out/comm_v1/adapter \
    --domain communication \
    --display-name "My Communication Style" \
    --base-model-id phi3-mini-4bit-v1
```

## Step 5: Test the System

### Test Data Pipeline
```python
# Test script
from pathlib import Path
import sys
sys.path.append('src')

from data_pipeline.parse_whatsapp import process_whatsapp_file
from adapters.router import create_router
from adapters.protocol import AdapterMeta

# Test parsing
result = process_whatsapp_file(
    Path("data/raw/sample_whatsapp.txt"),
    Path("data/processed/test_output.jsonl"),
    max_pairs=10
)
print(f"✅ Parsed {result} conversation pairs")

# Test routing
router = create_router()
routing_result = router.route("Hey, how's it going?", ui_context="messaging")
print(f"✅ Router selected: {routing_result.adapter_ids}")

# Test adapter loading
manifest_path = Path("out/comm_v1/adapter/adapter.json")
if manifest_path.exists():
    with open(manifest_path) as f:
        import json
        manifest_data = json.load(f)
    
    adapter_meta = AdapterMeta.from_dict(manifest_data)
    print(f"✅ Adapter loaded: {adapter_meta.display_name}")
else:
    print("⚠️ No adapter manifest found - train an adapter first")
```

## Step 6: Android Integration (Optional)

### Setup Android Development
```bash
# Install Android Studio
# Setup Samsung Galaxy S24 Emulator
# Install NDK for native libraries

# Build for Android (placeholder - requires full Android setup)
# ./gradlew assembleDebug
```

### Test on Emulator
```bash
# Push adapter files to emulator
adb push out/comm_v1/adapter/ /sdcard/adapters/comm_v1/

# Install APK
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Verification Checklist

✅ Python environment setup successfully  
✅ Dependencies installed without errors  
✅ WhatsApp data parsed and privacy-filtered  
✅ QLoRA training completed successfully  
✅ Adapter manifest created and validated  
✅ Router correctly selects adapters based on input  
✅ Memory usage stays within mobile device limits  

## Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size in config file
{
  "training": {
    "batch_size": 1,              # Reduce from 2
    "gradient_accumulation_steps": 8  # Increase to maintain effective batch size
  }
}
```

### Issue: "No module named 'transformers'"
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

pip install -r requirements.txt
```

### Issue: WhatsApp parsing fails
**Solution:**
```bash
# Check file encoding
file data/raw/your_export.txt

# Convert if needed
iconv -f ISO-8859-1 -t UTF-8 data/raw/your_export.txt > data/raw/your_export_utf8.txt
```

### Issue: Model download fails
**Solution:**
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

## Next Steps

1. **Train more adapters**: Use `configs/calendar_adapter.json` and `configs/notes_adapter.json`
2. **Optimize for mobile**: Convert models to GGUF format for deployment
3. **Build Android app**: Integrate with native inference engine
4. **Create demo video**: Show before/after adapter improvements
5. **Write documentation**: Complete technical docs for submission

## Performance Expectations

| Component | Memory Usage | Time |
|-----------|-------------|------|
| Base Model Loading | 2.5-3.0GB | 30-60s |
| Adapter Training | 4-6GB peak | 15-45min |
| Adapter Loading | +120MB | <1s |
| Inference | 3-4GB total | 1-3s first token |

## Getting Help

- **Documentation**: Check `docs/` folder for detailed guides
- **Issues**: Review `docs/troubleshooting.md`  
- **Architecture**: See `docs/architecture.md` for system design
- **API Reference**: Check code comments and docstrings

**🎉 You're now ready to build your Samsung EnnovateX submission!**
