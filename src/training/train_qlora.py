"""
QLoRA Fine-tuning for Samsung EnnovateX 2025 AI Challenge
Efficient fine-tuning of 3-4B parameter models for on-device deployment
"""

import os
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes import BitsAndBytesConfig
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "phi3-mini": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 2048,
        "base_tag": "phi3-mini-4bit-v1"
    },
    "qwen2.5-3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct", 
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "max_length": 2048,
        "base_tag": "qwen25-3b-4bit-v1"
    }
}

class QLoRATrainer:
    """QLoRA trainer for efficient fine-tuning."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = MODEL_CONFIGS[config["base_model"]]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Initialize model with quantization
        self.model = self._load_quantized_model()
        
        # Setup LoRA
        self._setup_lora()
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        logger.info(f"Loading tokenizer: {self.model_config['model_id']}")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["model_id"], 
            use_fast=True,
            trust_remote_code=False
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _load_quantized_model(self) -> AutoModelForCausalLM:
        """Load model with 4-bit quantization."""
        logger.info(f"Loading quantized model: {self.model_config['model_id']}")
        
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model_id"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.bfloat16
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        return model
    
    def _setup_lora(self):
        """Setup LoRA configuration."""
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.config.get("lora_r", 16),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            target_modules=self.model_config["target_modules"],
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _load_dataset(self, data_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL dataset."""
        logger.info(f"Loading dataset: {data_path}")
        
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    data.append(example)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def _format_example(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Format example for training."""
        # Build conversation format for Phi-3
        if "phi3" in self.config["base_model"]:
            prompt = f"<|user|>\n{example['instruction']}\n{example['input']}\n<|assistant|>\n"
        else:
            # Generic format
            prompt = f"### Instruction:\n{example['instruction']}\n### Input:\n{example['input']}\n### Response:\n"
        
        target = example["output"]
        
        # Tokenize
        prompt_tokens = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.model_config["max_length"] - 256
        )
        target_tokens = self.tokenizer(
            target, 
            truncation=True, 
            max_length=256
        )
        
        # Combine and create labels
        input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"] + [self.tokenizer.eos_token_id]
        labels = [-100] * len(prompt_tokens["input_ids"]) + target_tokens["input_ids"] + [self.tokenizer.eos_token_id]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels)
        }
    
    def _create_data_collator(self):
        """Create data collator for dynamic padding."""
        def collate_fn(batch):
            max_len = max(len(example["input_ids"]) for example in batch)
            
            input_ids = []
            labels = []
            attention_mask = []
            
            for example in batch:
                # Pad sequences
                pad_length = max_len - len(example["input_ids"])
                
                input_ids.append(
                    example["input_ids"].tolist() + 
                    [self.tokenizer.pad_token_id] * pad_length
                )
                
                labels.append(
                    example["labels"].tolist() + 
                    [-100] * pad_length
                )
                
                attention_mask.append(
                    [1] * len(example["input_ids"]) + 
                    [0] * pad_length
                )
            
            return {
                "input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention_mask)
            }
        
        return collate_fn
    
    def train(self, train_data_path: Path, val_data_path: Path = None) -> str:
        """Train the model and return adapter directory."""
        logger.info("Starting QLoRA training")
        
        # Load and format data
        train_data = self._load_dataset(train_data_path)
        train_dataset = [self._format_example(ex) for ex in train_data]
        
        val_dataset = None
        if val_data_path and val_data_path.exists():
            val_data = self._load_dataset(val_data_path)
            val_dataset = [self._format_example(ex) for ex in val_data]
        
        # Training arguments
        output_dir = Path(f"out/{self.config['adapter_id']}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=self.config.get("batch_size", 2),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 4),
            num_train_epochs=self.config.get("num_epochs", 1),
            learning_rate=self.config.get("learning_rate", 2e-4),
            logging_steps=10,
            save_strategy="no",
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=100 if val_dataset else None,
            bf16=True,
            optim="paged_adamw_8bit",
            warmup_steps=self.config.get("warmup_steps", 50),
            report_to=[]
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self._create_data_collator()
        )
        
        # Train
        trainer.train()
        
        # Save adapter
        adapter_dir = output_dir / "adapter"
        self.model.save_pretrained(str(adapter_dir), safe_serialization=True)
        self.tokenizer.save_pretrained(str(adapter_dir))
        
        logger.info(f"Adapter saved to: {adapter_dir}")
        return str(adapter_dir)

def create_adapter_manifest(adapter_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create adapter manifest with metadata."""
    model_config = MODEL_CONFIGS[config["base_model"]]
    weights_file = adapter_dir / "adapter_model.safetensors"
    
    # Calculate checksum
    hasher = hashlib.sha256()
    with open(weights_file, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            hasher.update(chunk)
    checksum = "sha256:" + hasher.hexdigest()
    
    manifest = {
        "adapter_id": config["adapter_id"],
        "domain": config["domain"],
        "display_name": config["display_name"],
        "base_model_id": model_config["base_tag"],
        "adapter_format": "safetensors-v1",
        "size_bytes": weights_file.stat().st_size,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checksum": checksum,
        "signature": "",
        "privacy_scope": config.get("privacy_scope", ["personal_texts"]),
        "activation_memory_mb": config.get("activation_memory_mb", 120),
        "recommended_layers": {
            "inject_every": 0,
            "target_modules": model_config["target_modules"],
            "alpha": config.get("lora_alpha", 32),
            "rank": config.get("lora_r", 16)
        },
        "validation_score": 0.0,  # TODO: Add evaluation
        "version": "1.0.0",
        "safe_to_share": False,
        "training_meta": {
            "qlora": True,
            "base_quant": "nf4",
            "compute_dtype": "bfloat16",
            "epochs": config.get("num_epochs", 1),
            "learning_rate": config.get("learning_rate", 2e-4)
        }
    }
    
    # Save manifest
    manifest_path = adapter_dir / "adapter.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Manifest saved to: {manifest_path}")
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Train QLoRA adapter")
    parser.add_argument("--config", required=True, help="Training configuration file")
    parser.add_argument("--train-data", required=True, help="Training JSONL file")
    parser.add_argument("--val-data", help="Validation JSONL file")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize trainer
    trainer = QLoRATrainer(config)
    
    # Train
    adapter_dir = trainer.train(
        Path(args.train_data),
        Path(args.val_data) if args.val_data else None
    )
    
    # Create manifest
    create_adapter_manifest(Path(adapter_dir), config)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
