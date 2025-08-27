# Samsung EnnovateX 2025 - On-Device LLM Fine-Tuning Framework

Efficient framework for on-device fine-tuning of 3-4B parameter LLMs using QLoRA on Samsung Galaxy devices.

## Technical Overview

- **Base Model**: Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **Training**: QLoRA with 4-bit quantization  
- **Mobile Runtime**: Hot-swappable adapter system
- **Privacy**: Local-only processing with PII filtering
- **Target**: Samsung Galaxy S23-S25 equivalent devices

## Architecture

### Core Components
1. **QLoRA Training Pipeline** - Fine-tune 3-4B models efficiently
2. **Adapter Protocol System** - Hot-swappable model adapters
3. **Resource-Aware Scheduler** - Mobile-optimized background processing
4. **Privacy-First Data Pipeline** - Local processing with PII filtering
5. **Android Integration** - Samsung Galaxy compatible

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Data Processing
```bash
python src/data_pipeline/parse_whatsapp.py
python src/data_pipeline/privacy_filter.py
```

#### Training
```bash
python train_real_adapter.py
```

#### Inference
```bash
python src/inference/mobile_session.py
```

## Key Features

- **Hot-swappable adapters** with memory-efficient loading
- **Resource-aware scheduling** respecting battery/thermal constraints
- **Privacy-preserving local training** with comprehensive PII filtering
- **Multi-domain routing** with intelligent adapter selection

## License
MIT License
