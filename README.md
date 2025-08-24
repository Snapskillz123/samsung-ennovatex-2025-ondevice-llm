# Samsung EnnovateX 2025 AI Challenge Submission

**Problem Statement** - On-Device Fine-Tuning Framework for Billion+ Parameter scale LLMs

**Team name** - [Your Team Name Here]

**Team members (Names)** - Member 1 Name, Member 2 Name, Member 3 Name, Member 4 Name

**Demo Video Link** - [Upload the Demo video on Youtube as a public or unlisted video and share the link]

## Project Artefacts

### Technical Documentation - Docs
All technical details are written in markdown files inside the docs folder in the repo.

### Source Code - Source  
All source code is added to the src folder in the repo. The code is capable of being successfully installed/executed and runs consistently on Samsung Galaxy S23-S25 equivalent devices.

### Models Used
- Microsoft Phi-3-mini-4k-instruct (3.8B parameters) - Base model for fine-tuning
- Hugging Face links: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

### Models Published
[In case you have developed a model as a part of your solution, upload it on Hugging Face and add the link here]

### Datasets Used
- WhatsApp chat exports (personal data, processed with privacy filters)
- PersonaChat dataset (for evaluation) - https://huggingface.co/datasets/bavard/personachat_truecased

### Datasets Published
[Links to all datasets created for the project and published on Hugging Face]

## Quick Start

### Prerequisites
```bash
# Python 3.9+ required
python --version

# CUDA-capable GPU recommended for training
nvidia-smi
```

### Installation
```bash
# Clone and setup
git clone [your-repo-link]
cd Samsung

# Install dependencies
pip install -r requirements.txt

# Set up Android development (optional)
# Install Android Studio and SDK for emulator testing
```

### Usage

#### 1. Data Preprocessing
```bash
# Convert WhatsApp chats to training format
python src/data_pipeline/parse_whatsapp.py --input data/chats.txt --output data/whatsapp_sft.jsonl

# Apply privacy filters
python src/data_pipeline/privacy_filter.py --input data/whatsapp_sft.jsonl --output data/filtered_sft.jsonl
```

#### 2. Model Training
```bash
# Fine-tune with QLoRA
python src/training/train_qlora.py --config configs/communication_adapter.yaml
```

#### 3. Adapter Conversion
```bash
# Create adapter manifest and convert for mobile
python src/conversion/create_manifest.py --adapter_id comm_v1 --adapter_dir out/comm_v1/adapter
python src/conversion/convert_for_android.py --base_model phi3-mini --adapter comm_v1
```

#### 4. Android Integration
```bash
# Push to Android emulator
./tools/push_to_device.sh out/mobile/
```

## Architecture Overview

### Core Components
1. **QLoRA Training Pipeline** - Fine-tune 3-4B models efficiently
2. **Adapter Protocol System** - Hot-swappable model adapters
3. **Resource-Aware Scheduler** - Mobile-optimized background processing
4. **Privacy-First Data Pipeline** - Local processing with PII filtering
5. **Android Integration** - Samsung Galaxy S23-S25 compatible

### Adapter Domains
- **Communication Adapter**: Learns texting and writing style from WhatsApp
- **Calendar Adapter**: Learns scheduling and organizational preferences  
- **Notes Adapter**: Learns user writing style in notes

### Key Innovations
- **Hot-swappable adapters** with memory-efficient loading
- **Resource-aware scheduling** respecting battery/thermal constraints
- **Privacy-preserving local training** with comprehensive PII filtering
- **Multi-domain routing** with intelligent adapter selection

## Attribution
This project builds upon open-source libraries including Transformers, PEFT, and bitsandbytes for QLoRA implementation.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.
