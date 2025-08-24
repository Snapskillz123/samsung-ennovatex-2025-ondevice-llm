"""
Adapter Protocol System for Samsung EnnovateX 2025 AI Challenge
Defines the core adapter interfaces and management system
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum
from pathlib import Path
import json
import hashlib
import logging

logger = logging.getLogger(__name__)

class Domain(Enum):
    """Supported adapter domains."""
    COMMUNICATION = "communication"
    CALENDAR = "calendar" 
    NOTES = "notes"
    GENERAL = "general"

class AdapterFormat(Enum):
    """Supported adapter file formats."""
    SAFETENSORS_V1 = "safetensors-v1"
    GGUF_LORA_V1 = "gguf-lora-v1"
    TFLITE_LORA_V1 = "tflite-lora-v1"

@dataclass
class AdapterMeta:
    """Adapter metadata from manifest."""
    adapter_id: str
    domain: Domain
    display_name: str
    base_model_id: str
    adapter_format: AdapterFormat
    size_bytes: int
    created_at: str
    checksum: str
    signature: str
    privacy_scope: List[str]
    activation_memory_mb: int
    recommended_layers: Dict[str, Any]
    validation_score: float
    version: str
    safe_to_share: bool
    training_meta: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdapterMeta':
        """Create AdapterMeta from dictionary."""
        return cls(
            adapter_id=data["adapter_id"],
            domain=Domain(data["domain"]),
            display_name=data["display_name"],
            base_model_id=data["base_model_id"],
            adapter_format=AdapterFormat(data["adapter_format"]),
            size_bytes=data["size_bytes"],
            created_at=data["created_at"],
            checksum=data["checksum"],
            signature=data.get("signature", ""),
            privacy_scope=data["privacy_scope"],
            activation_memory_mb=data["activation_memory_mb"],
            recommended_layers=data["recommended_layers"],
            validation_score=data.get("validation_score", 0.0),
            version=data["version"],
            safe_to_share=data["safe_to_share"],
            training_meta=data.get("training_meta", {}),
            notes=data.get("notes", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "adapter_id": self.adapter_id,
            "domain": self.domain.value,
            "display_name": self.display_name,
            "base_model_id": self.base_model_id,
            "adapter_format": self.adapter_format.value,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "signature": self.signature,
            "privacy_scope": self.privacy_scope,
            "activation_memory_mb": self.activation_memory_mb,
            "recommended_layers": self.recommended_layers,
            "validation_score": self.validation_score,
            "version": self.version,
            "safe_to_share": self.safe_to_share,
            "training_meta": self.training_meta,
            "notes": self.notes
        }

@dataclass
class AdapterHandle:
    """Handle representing a loaded adapter."""
    adapter_id: str
    scale: float = 1.0
    active: bool = True

class LlmSession(Protocol):
    """Protocol for LLM session that supports adapter attachment."""
    
    @property
    def id(self) -> str:
        """Base model identifier."""
        ...
    
    @property
    def required_ram_mb(self) -> int:
        """Required RAM for base model."""
        ...
    
    def attach_adapter_file(self, path: str, scale: float = 1.0) -> AdapterHandle:
        """Attach adapter from file path."""
        ...
    
    def set_adapter_scale(self, handle: AdapterHandle, scale: float) -> None:
        """Update adapter scale."""
        ...
    
    def detach_adapter(self, handle: AdapterHandle) -> None:
        """Detach adapter and free resources."""
        ...
    
    def generate(self, 
                prompt: str, 
                adapters: List[AdapterHandle] = None,
                max_tokens: int = 256) -> str:
        """Generate text with optional adapters."""
        ...

class AdapterProtocol(ABC):
    """Abstract protocol for adapter implementations."""
    
    def __init__(self, meta: AdapterMeta):
        self.meta = meta
    
    @abstractmethod
    def can_load(self, 
                free_ram_mb: int, 
                user_consent: bool, 
                base_model_id: str) -> bool:
        """Check if adapter can be loaded given current constraints."""
        ...
    
    @abstractmethod
    def load(self, path: str, session: LlmSession) -> AdapterHandle:
        """Load adapter and return handle."""
        ...
    
    @abstractmethod
    def unload(self, session: LlmSession, handle: AdapterHandle) -> None:
        """Unload adapter and cleanup resources."""
        ...
    
    def verify_checksum(self, path: str) -> bool:
        """Verify file checksum matches manifest."""
        if not self.meta.checksum.startswith("sha256:"):
            return False
        
        expected_hash = self.meta.checksum[7:]  # Remove "sha256:" prefix
        
        hasher = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    hasher.update(chunk)
            
            actual_hash = hasher.hexdigest()
            return actual_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False

class FileLoraAdapter(AdapterProtocol):
    """File-based LoRA adapter implementation."""
    
    def can_load(self, 
                free_ram_mb: int, 
                user_consent: bool, 
                base_model_id: str) -> bool:
        """Check loading constraints."""
        # Check user consent
        if not user_consent:
            logger.info(f"User consent not granted for {self.meta.adapter_id}")
            return False
        
        # Check base model compatibility
        if self.meta.base_model_id != base_model_id:
            logger.warning(f"Base model mismatch: {self.meta.base_model_id} != {base_model_id}")
            return False
        
        # Check memory requirements
        if free_ram_mb < self.meta.activation_memory_mb:
            logger.warning(f"Insufficient RAM: {free_ram_mb} < {self.meta.activation_memory_mb}")
            return False
        
        return True
    
    def load(self, path: str, session: LlmSession) -> AdapterHandle:
        """Load adapter from file."""
        # Verify checksum
        if not self.verify_checksum(path):
            raise ValueError(f"Checksum verification failed for {path}")
        
        # Load via session
        handle = session.attach_adapter_file(path, scale=1.0)
        logger.info(f"Loaded adapter {self.meta.adapter_id}")
        
        return handle
    
    def unload(self, session: LlmSession, handle: AdapterHandle) -> None:
        """Unload adapter."""
        session.detach_adapter(handle)
        logger.info(f"Unloaded adapter {self.meta.adapter_id}")

class AdapterManager:
    """Central manager for adapter lifecycle."""
    
    def __init__(self, session: LlmSession, adapters_dir: Path):
        self.session = session
        self.adapters_dir = adapters_dir
        self.inventory: Dict[str, AdapterMeta] = {}
        self.protocols: Dict[str, AdapterProtocol] = {}
        self.loaded: Dict[str, AdapterHandle] = {}  # LRU order maintained
        self.max_loaded = 3  # Maximum concurrent adapters
        
        self._load_inventory()
    
    def _load_inventory(self) -> None:
        """Load adapter inventory from disk."""
        inventory_file = self.adapters_dir / "adapters_index.json"
        
        if inventory_file.exists():
            with open(inventory_file, 'r') as f:
                data = json.load(f)
                for adapter_data in data.get("adapters", []):
                    meta = AdapterMeta.from_dict(adapter_data)
                    self.register_adapter(meta)
        
        logger.info(f"Loaded {len(self.inventory)} adapters from inventory")
    
    def register_adapter(self, meta: AdapterMeta) -> None:
        """Register a new adapter."""
        self.inventory[meta.adapter_id] = meta
        
        # Create appropriate protocol
        if meta.adapter_format in [AdapterFormat.SAFETENSORS_V1, AdapterFormat.GGUF_LORA_V1]:
            self.protocols[meta.adapter_id] = FileLoraAdapter(meta)
        else:
            raise ValueError(f"Unsupported adapter format: {meta.adapter_format}")
        
        logger.info(f"Registered adapter: {meta.adapter_id}")
    
    def can_load(self, adapter_id: str, free_ram_mb: int, user_consent: bool) -> bool:
        """Check if adapter can be loaded."""
        if adapter_id not in self.protocols:
            return False
        
        protocol = self.protocols[adapter_id]
        return protocol.can_load(free_ram_mb, user_consent, self.session.id)
    
    def ensure_loaded(self, 
                     adapter_id: str, 
                     free_ram_mb: int, 
                     user_consent: bool) -> Optional[AdapterHandle]:
        """Ensure adapter is loaded, load if necessary."""
        # Already loaded?
        if adapter_id in self.loaded:
            # Move to end (most recently used)
            handle = self.loaded.pop(adapter_id)
            self.loaded[adapter_id] = handle
            return handle
        
        # Can we load it?
        if not self.can_load(adapter_id, free_ram_mb, user_consent):
            return None
        
        # Free up space if needed
        while len(self.loaded) >= self.max_loaded:
            self._evict_oldest()
        
        # Load the adapter
        try:
            protocol = self.protocols[adapter_id]
            adapter_path = self._get_adapter_path(adapter_id)
            
            handle = protocol.load(str(adapter_path), self.session)
            self.loaded[adapter_id] = handle
            
            return handle
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            return None
    
    def _get_adapter_path(self, adapter_id: str) -> Path:
        """Get file path for adapter."""
        meta = self.inventory[adapter_id]
        
        if meta.adapter_format == AdapterFormat.SAFETENSORS_V1:
            return self.adapters_dir / adapter_id / "adapter_model.safetensors"
        elif meta.adapter_format == AdapterFormat.GGUF_LORA_V1:
            return self.adapters_dir / adapter_id / "adapter.gguf"
        else:
            raise ValueError(f"Unsupported format: {meta.adapter_format}")
    
    def _evict_oldest(self) -> None:
        """Evict oldest loaded adapter."""
        if not self.loaded:
            return
        
        # Get first (oldest) adapter
        oldest_id = next(iter(self.loaded))
        self.unload(oldest_id)
    
    def unload(self, adapter_id: str) -> None:
        """Unload specific adapter."""
        if adapter_id not in self.loaded:
            return
        
        handle = self.loaded.pop(adapter_id)
        protocol = self.protocols[adapter_id]
        protocol.unload(self.session, handle)
    
    def set_adapter_scale(self, adapter_id: str, scale: float) -> bool:
        """Set scale for loaded adapter."""
        if adapter_id not in self.loaded:
            return False
        
        handle = self.loaded[adapter_id]
        self.session.set_adapter_scale(handle, scale)
        handle.scale = scale
        return True
    
    def get_loaded_adapters(self) -> List[str]:
        """Get list of currently loaded adapter IDs."""
        return list(self.loaded.keys())
    
    def get_adapter_meta(self, adapter_id: str) -> Optional[AdapterMeta]:
        """Get adapter metadata."""
        return self.inventory.get(adapter_id)
    
    def generate_with_adapters(self,
                             prompt: str,
                             adapter_ids: List[str],
                             free_ram_mb: int,
                             consent_checker: callable,
                             max_tokens: int = 256) -> str:
        """Generate text with specified adapters."""
        # Load required adapters
        handles = []
        for adapter_id in adapter_ids:
            consent = consent_checker(adapter_id)
            handle = self.ensure_loaded(adapter_id, free_ram_mb, consent)
            if handle:
                handles.append(handle)
        
        # Generate with loaded adapters
        return self.session.generate(prompt, adapters=handles, max_tokens=max_tokens)
