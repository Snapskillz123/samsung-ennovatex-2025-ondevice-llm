<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Samsung EnnovateX 2025 AI Challenge - Copilot Instructions

This is a hackathon project for on-device fine-tuning of Large Language Models using QLoRA on Samsung Galaxy devices.

## Project Context
- **Problem**: Efficient framework for on-device fine-tuning of 3-4B parameter LLMs on mobile devices
- **Target Device**: Samsung Galaxy S23-S25 equivalent smartphones  
- **Key Innovation**: Hot-swappable adapter system with resource-aware scheduling
- **Evaluation Criteria**: Novelty (25%), Technical Implementation (25%), Demo Video (25%), UI/UX (15%), Ethics/Scalability (10%)

## Technical Stack
- **Base Models**: Microsoft Phi-3-mini-4k-instruct (3.8B parameters)
- **Training**: QLoRA with bitsandbytes 4-bit quantization
- **Mobile Runtime**: GGUF format with llama.cpp or MLC LLM
- **Android**: Kotlin/Java with JNI for model inference
- **Data Pipeline**: WhatsApp chat processing with privacy filtering

## Code Generation Guidelines
1. **Memory Efficiency**: Always consider mobile RAM constraints (4-8GB available)
2. **Battery Awareness**: Include thermal and battery checks before heavy operations
3. **Privacy First**: Implement PII filtering and local-only processing
4. **Error Handling**: Robust fallbacks for mobile constraints (OOM, thermal throttling)
5. **Resource Monitoring**: Track CPU/GPU/memory usage for optimization

## Architecture Patterns
- Use adapter pattern for hot-swappable model components
- Implement observer pattern for resource monitoring
- Use factory pattern for adapter creation and validation
- Apply strategy pattern for routing between different adapters

## File Naming Conventions
- Training scripts: `train_*.py`
- Adapter protocols: `*_adapter.py` or `*_protocol.py`
- Android integration: `*Manager.kt` or `*Service.kt`
- Conversion tools: `convert_*.py`
- Evaluation scripts: `eval_*.py`

Please prioritize code that can run efficiently on mobile devices and follows the project's privacy-first approach.
