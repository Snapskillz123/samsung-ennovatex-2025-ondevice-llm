# Samsung EnnovateX 2025 AI Challenge - Project Structure

```
Samsung/
├── .github/
│   └── copilot-instructions.md     # Copilot customization for this project
├── configs/                        # Training configurations
│   ├── communication_adapter.json  # WhatsApp/messaging style adapter
│   ├── calendar_adapter.json       # Calendar/scheduling adapter
│   └── notes_adapter.json         # Note-taking style adapter
├── data/                          # Data directories
│   ├── raw/                       # Raw exported data (WhatsApp, etc.)
│   ├── processed/                 # Processed training data
│   ├── communication/             # Communication-specific datasets
│   ├── calendar/                  # Calendar-specific datasets
│   └── notes/                     # Notes-specific datasets
├── docs/                          # Documentation
│   ├── architecture.md            # Technical architecture guide
│   ├── quick_start.md            # Quick start tutorial
│   ├── adapter_spec.md           # Adapter specification
│   └── api_reference.md          # API documentation
├── models/                        # Model storage
│   ├── base/                     # Base model files (quantized)
│   └── adapters/                 # Deployed adapter files
├── out/                          # Training outputs
│   ├── models/                   # Trained model outputs
│   └── adapters/                 # Generated adapters
├── src/                          # Source code
│   ├── adapters/                 # Adapter protocol system
│   │   ├── protocol.py           # Core adapter interfaces
│   │   └── router.py            # Adapter selection logic
│   ├── conversion/               # Model conversion tools
│   │   ├── create_manifest.py   # Manifest creator
│   │   └── convert_for_android.py # Mobile format converter
│   ├── data_pipeline/           # Data processing
│   │   ├── parse_whatsapp.py    # WhatsApp chat parser
│   │   └── privacy_filter.py    # PII filtering
│   ├── training/                # Training pipeline
│   │   └── train_qlora.py       # QLoRA fine-tuning
│   └── android/                 # Android integration (future)
│       ├── AdapterManager.kt    # Android adapter manager
│       └── NativeInference.java # JNI wrapper
├── tools/                       # Development tools
│   ├── setup.sh                 # Linux/Mac setup script
│   ├── setup.bat               # Windows setup script
│   └── validate_manifest.py    # Manifest validator
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT license
├── README.md                   # Main project README
├── requirements.txt            # Python dependencies
└── quick_start.py             # Development environment tester
```

## Key Files to Start With

### 1. Environment Setup
```bash
# Windows
tools\setup.bat

# Linux/Mac  
chmod +x tools/setup.sh && ./tools/setup.sh

# Test setup
python quick_start.py
```

### 2. Data Processing
```bash
# Parse WhatsApp export
python src/data_pipeline/parse_whatsapp.py \
    --input data/raw/your_export.txt \
    --output data/processed/parsed.jsonl

# Apply privacy filter
python src/data_pipeline/privacy_filter.py \
    --input data/processed/parsed.jsonl \
    --output data/processed/filtered.jsonl
```

### 3. Training
```bash
# Train communication adapter
python src/training/train_qlora.py \
    --config configs/communication_adapter.json \
    --train-data data/processed/filtered.jsonl
```

### 4. Manifest Creation
```bash
python src/conversion/create_manifest.py \
    --adapter-id comm_v1 \
    --adapter-dir out/comm_v1/adapter \
    --domain communication \
    --display-name "Personal Communication Style" \
    --base-model-id phi3-mini-4bit-v1
```

## Configuration Files

Each adapter domain has its own configuration:

- **communication_adapter.json**: For WhatsApp/messaging style learning
- **calendar_adapter.json**: For scheduling preferences  
- **notes_adapter.json**: For note-taking style

## Core Components

1. **Adapter Protocol** (`src/adapters/protocol.py`): Defines how adapters are loaded, managed, and used
2. **Adapter Router** (`src/adapters/router.py`): Intelligently selects which adapters to use based on context
3. **QLoRA Training** (`src/training/train_qlora.py`): Efficient fine-tuning pipeline
4. **Data Pipeline** (`src/data_pipeline/`): Privacy-safe data processing
5. **Conversion Tools** (`src/conversion/`): Convert adapters for mobile deployment

This structure provides a complete framework for the Samsung EnnovateX hackathon, from data processing through training to mobile deployment.
