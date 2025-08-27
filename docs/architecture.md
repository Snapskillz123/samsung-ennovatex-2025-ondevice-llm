# Samsung EnnovateX 2025 AI Challenge - Technical Architecture

## Overview

This document describes the technical architecture for the On-Device Fine-Tuning Framework for Large Language Models, designed to run efficiently on Samsung Galaxy S23-S25 equivalent devices.

## System Architecture

### High-Level Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Android App   │    │  Adapter        │    │   Base Model    │
│   (UI Layer)    │◄──►│  Manager        │◄──►│   (Frozen)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Router      │    │   Adapters      │    │   Inference     │
│  (Selection)    │    │ (Hot-swappable) │    │    Engine       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Resource        │    │ Data Pipeline   │    │  Training       │
│ Monitor         │    │ (Privacy-Safe)  │    │ (QLoRA)         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Principles

1. **Resource Efficiency**: Operate within mobile device constraints (4-8GB RAM, thermal limits)
2. **Privacy First**: All processing happens on-device with comprehensive PII filtering
3. **Hot-Swappable Adapters**: Dynamic loading/unloading of specialized model components
4. **Graceful Degradation**: Always fallback to base model if adapters fail

## Component Details

### 1. Base Model System

**Supported Models:**
- Microsoft Phi-3-mini-4k-instruct (3.8B parameters) - Primary choice
- Qwen2.5-3B-Instruct (3B parameters) - Alternative
- Llama-3.2-3B-Instruct (3B parameters) - Backup option

**Quantization Strategy:**
- 4-bit NF4 quantization using bitsandbytes
- Target memory footprint: ~2.5-3.0GB for base model
- Runtime format: GGUF with Q4_K_M quantization for inference

### 2. Adapter Protocol System

**Adapter Structure:**
```
adapter_directory/
├── adapter.json          # Manifest with metadata
├── adapter_model.safetensors  # LoRA weights
├── vocab_patch.json      # Optional vocabulary extensions
└── README.md            # Human-readable description
```

**Adapter Types:**
- **Communication Adapter**: Learns texting/messaging style from WhatsApp
- **Calendar Adapter**: Learns scheduling preferences and time management
- **Notes Adapter**: Learns note-taking and writing style

**Memory Management:**
- Maximum 3 concurrent adapters loaded
- LRU (Least Recently Used) eviction policy
- Per-adapter memory budget: 100-150MB

### 3. QLoRA Training Pipeline

**Training Configuration:**
```json
{
  "quantization": "4-bit NF4",
  "lora_rank": 8-16,
  "lora_alpha": 16-32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "batch_size": 2,
  "gradient_accumulation": 4
}
```

**Resource Constraints:**
- Only train when device is charging
- Temperature monitoring (halt if >42°C)
- Memory monitoring (halt if <1GB free)
- Background processing only

### 4. Data Pipeline & Privacy

**WhatsApp Processing:**
1. Parse exported chat files
2. Extract conversation turns (6-message sliding window)
3. Apply PII filtering (emails, phones, SSNs, etc.)
4. Convert to supervised fine-tuning format
5. Split into train/validation sets

**Privacy Safeguards:**
- All data processing happens locally
- Comprehensive PII masking with placeholders
- User consent required for each data source
- One-click data deletion capability

### 5. Adapter Router System

**Routing Strategies:**
1. **UI Context**: Direct mapping from app screen to adapter
2. **Keyword Matching**: Pattern matching on prompt content
3. **Hybrid**: Combines UI context + keywords for optimal selection

**Example Routing:**
```python
# UI Context: "messaging" → Communication Adapter
# Keywords: "schedule", "meeting" → Calendar Adapter  
# Keywords: "note", "summary" → Notes Adapter
```

### 6. Android Integration

**Architecture:**
```kotlin
class AdapterManager {
    fun ensureLoaded(adapterId: String): AdapterHandle?
    fun generate(prompt: String, adapters: List<String>): String
    fun unload(adapterId: String)
}

class ResourceMonitor {
    fun getFreeRamMb(): Int
    fun getTemperatureCelsius(): Float
    fun isBatteryCharging(): Boolean
}
```

**JNI Integration:**
- Native libraries for model inference (llama.cpp or MLC-LLM)
- Adapter attachment/detachment through native calls
- Memory-mapped file access for model weights

## Performance Targets

### Memory Usage
- Base model: 2.5-3.0GB
- Single adapter: 100-150MB
- Maximum total: 4.5GB (within 8GB device limit)

### Inference Performance
- First token latency: <2 seconds
- Subsequent tokens: 10-20 tokens/second
- Adapter switching: <500ms

### Training Performance
- Adapter training: 30-60 minutes (communication style)
- Memory during training: <6GB peak
- Temperature rise: <5°C above baseline

## Evaluation Framework

### Metrics
1. **Response Quality**: Human evaluation scores (1-5 scale)
2. **Style Consistency**: Automated comparison with user's historical style
3. **Resource Usage**: Memory, CPU, battery consumption
4. **User Satisfaction**: A/B testing before/after adapter application

### Validation Process
1. Hold-out evaluation set (10% of training data)
2. Automated toxicity screening
3. PII leakage detection
4. Performance regression testing

## Deployment Strategy

### Mobile Packaging
1. Base model conversion: PyTorch → GGUF format
2. Adapter conversion: Safetensors → Mobile-optimized format
3. APK packaging with native libraries
4. Progressive download for large model files

### Update Mechanism
- Over-the-air adapter updates
- A/B testing for new adapter versions
- Automatic rollback on quality regression
- User control over adapter versions

## Risk Mitigation

### Technical Risks
1. **Out of Memory**: Implement memory monitoring and graceful degradation
2. **Thermal Throttling**: Temperature-based workload scheduling
3. **Model Incompatibility**: Robust version checking and fallbacks
4. **Privacy Breach**: Comprehensive PII filtering and audit logs

### Operational Risks
1. **Poor Quality Adapters**: Automated evaluation and human review
2. **User Data Loss**: Secure backup and recovery mechanisms
3. **Device Compatibility**: Extensive testing across Samsung Galaxy models

## Future Enhancements

### Phase 2 Features
- **Federated Learning**: Aggregate improvements across users (privacy-preserving)
- **Multi-Modal Adapters**: Support for image and voice input
- **Dynamic Adapter Fusion**: Real-time combination of multiple adapters
- **Edge Deployment**: Support for Samsung DeX and other edge devices

### Advanced Optimizations
- **Gradient Checkpointing**: Reduce memory during training
- **Mixed Precision**: FP16/BF16 inference for speed improvements  
- **Model Pruning**: Remove unnecessary parameters post-training
- **Knowledge Distillation**: Create smaller, faster specialized models

---

This architecture balances innovation with practical constraints, ensuring the system can deliver personalized AI experiences while respecting the limitations of mobile hardware and user privacy expectations.
