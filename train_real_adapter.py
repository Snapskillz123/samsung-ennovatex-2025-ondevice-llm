"""
QLoRA Training Pipeline
Production training script for Samsung EnnovateX 2025
"""

import sys
import os
from pathlib import Path
import json
import logging

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_training():
    """Run QLoRA training on processed data."""
    
    # Check training data
    data_dir = project_root / "data" / "processed"
    training_file = data_dir / "filtered.jsonl"
    
    if not training_file.exists():
        # Process data pipeline
        try:
            from data_pipeline.parse_whatsapp import process_whatsapp_file
            from data_pipeline.privacy_filter import process_jsonl_file, PrivacyFilter
            
            raw_file = project_root / "data" / "raw" / "sample_whatsapp.txt"
            parsed_file = data_dir / "parsed.jsonl" 
            
            # Parse and filter
            num_pairs = process_whatsapp_file(raw_file, parsed_file, max_pairs=100)
            privacy_filter = PrivacyFilter()
            stats = process_jsonl_file(parsed_file, training_file, privacy_filter)
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return False
    
    if not training_file.exists():
        logger.error(f"Training file not found: {training_file}")
        return False
    
    # Load training data
    with open(training_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]
    
    # Training configuration
    config = {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "adapter_name": "communication_v1",
        "train_data": str(training_file),
        "output_dir": str(project_root / "adapters" / "comm_v1"),
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_length": 512
    }
    
    # Save config
    config_file = project_root / "configs" / "communication_adapter.json"
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        from training.train_qlora import QLoRATrainer
        
        # Initialize and train
        trainer = QLoRATrainer(config)
        trainer.train()
        
        logger.info(f"Training completed. Adapter saved to: {config['output_dir']}")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Run the training pipeline."""
    success = run_training()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
