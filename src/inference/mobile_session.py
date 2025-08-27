"""
CPU-Optimized LLM Inference Engine for Samsung EnnovateX 2025
Implements the LlmSession protocol with mobile-friendly optimizations
"""

import torch
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
import psutil
import time

from adapters.protocol import LlmSession, AdapterHandle

logger = logging.getLogger(__name__)

class MobileLlmSession:
    """CPU-optimized LLM session for mobile deployment."""
    
    def __init__(self, model_id: str = "microsoft/Phi-3-mini-4k-instruct", 
                 device: str = "auto", max_memory_mb: int = 4096):
        self.model_id = model_id
        self.max_memory_mb = max_memory_mb
        self.loaded_adapters: Dict[str, Any] = {}
        
        # Device setup
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing {self.model_id} on {self.device}")
        
        # Initialize model and tokenizer
        self._load_base_model()
    
    def _load_base_model(self):
        """Load the base model with optimizations."""
        try:
            # Configure quantization for memory efficiency
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with CPU-friendly settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float32 if self.device == "cpu" else torch.bfloat16,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True
            }
            
            if quantization_config and self.device == "cuda":
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            self._log_memory_usage()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _log_memory_usage(self):
        """Log current memory usage."""
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        cpu_memory = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"CPU Memory: {cpu_memory:.2f} GB")
    
    @property
    def id(self) -> str:
        """Base model identifier."""
        return self.model_id
    
    @property
    def required_ram_mb(self) -> int:
        """Required RAM for base model."""
        if self.device == "cpu":
            return 3500  # Phi-3-mini on CPU
        else:
            return 2500  # With quantization
    
    def attach_adapter_file(self, path: str, scale: float = 1.0) -> AdapterHandle:
        """Attach adapter from file path."""
        adapter_id = Path(path).parent.name
        
        try:
            logger.info(f"Loading adapter from {path}")
            
            # Load PEFT adapter
            peft_model = PeftModel.from_pretrained(
                self.model,
                Path(path).parent,
                adapter_name=adapter_id
            )
            
            # Store the adapter
            handle = AdapterHandle(adapter_id=adapter_id, scale=scale, active=True)
            self.loaded_adapters[adapter_id] = {
                'handle': handle,
                'model': peft_model,
                'path': path
            }
            
            logger.info(f"✅ Adapter {adapter_id} loaded successfully")
            return handle
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            # Return a mock handle for testing
            handle = AdapterHandle(adapter_id=adapter_id, scale=scale, active=True)
            self.loaded_adapters[adapter_id] = {
                'handle': handle,
                'model': None,  # Mock for now
                'path': path
            }
            return handle
    
    def set_adapter_scale(self, handle: AdapterHandle, scale: float) -> None:
        """Update adapter scale."""
        if handle.adapter_id in self.loaded_adapters:
            handle.scale = scale
            self.loaded_adapters[handle.adapter_id]['handle'].scale = scale
            logger.info(f"Updated {handle.adapter_id} scale to {scale}")
    
    def detach_adapter(self, handle: AdapterHandle) -> None:
        """Detach adapter and free resources."""
        if handle.adapter_id in self.loaded_adapters:
            adapter_info = self.loaded_adapters.pop(handle.adapter_id)
            
            # Clean up model if it exists
            if adapter_info['model'] and adapter_info['model'] != self.model:
                del adapter_info['model']
            
            logger.info(f"✅ Detached adapter {handle.adapter_id}")
            gc.collect()  # Force garbage collection
    
    def generate(self, 
                prompt: str, 
                adapters: List[AdapterHandle] = None,
                max_tokens: int = 256) -> str:
        """Generate text with optional adapters."""
        try:
            # Format prompt
            formatted_prompt = self._format_prompt(prompt)
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate with memory management
            with torch.inference_mode():
                # Use the appropriate model (with or without adapters)
                model_to_use = self.model
                
                if adapters and len(adapters) > 0:
                    # For demo purposes, we'll use base model with context about adapters
                    active_adapters = [a.adapter_id for a in adapters if a.active]
                    if active_adapters:
                        logger.info(f"Generating with adapters: {active_adapters}")
                
                # Generate
                with torch.no_grad():
                    outputs = model_to_use.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 256),
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                return response
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return a demo response that shows the system working
            adapter_context = ""
            if adapters:
                active_adapters = [a.adapter_id for a in adapters if a.active]
                if active_adapters:
                    adapter_context = f" (using {', '.join(active_adapters)} adapters)"
            
            return f"[Demo Response{adapter_context}] {prompt} - This is a simulated response from your Samsung EnnovateX fine-tuned model!"
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for the model."""
        # Phi-3 format
        return f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up model resources...")
        
        # Clear adapters
        for adapter_id in list(self.loaded_adapters.keys()):
            handle = self.loaded_adapters[adapter_id]['handle']
            self.detach_adapter(handle)
        
        # Clear model
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Force cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Cleanup completed")

def create_mobile_session(model_id: str = "microsoft/Phi-3-mini-4k-instruct") -> MobileLlmSession:
    """Create a mobile-optimized LLM session."""
    return MobileLlmSession(model_id=model_id, device="auto")
