"""
Adapter Manifest Creator for Samsung EnnovateX 2025 AI Challenge
Creates and validates adapter manifest files
"""

import json
import hashlib
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..adapters.protocol import AdapterMeta, Domain, AdapterFormat

logger = logging.getLogger(__name__)

class ManifestCreator:
    """Creates adapter manifests with proper validation."""
    
    def __init__(self):
        self.domain_mappings = {
            "communication": Domain.COMMUNICATION,
            "comm": Domain.COMMUNICATION,
            "messaging": Domain.COMMUNICATION,
            "calendar": Domain.CALENDAR,
            "cal": Domain.CALENDAR,
            "schedule": Domain.CALENDAR,
            "notes": Domain.NOTES,
            "note": Domain.NOTES,
            "memo": Domain.NOTES
        }
    
    def create_manifest(self,
                       adapter_id: str,
                       adapter_dir: Path,
                       domain: str,
                       display_name: str,
                       base_model_id: str,
                       **kwargs) -> Dict[str, Any]:
        """Create adapter manifest."""
        
        logger.info(f"Creating manifest for adapter: {adapter_id}")
        
        # Validate inputs
        domain_enum = self._validate_domain(domain)
        adapter_format = self._detect_format(adapter_dir)
        weights_file = self._find_weights_file(adapter_dir, adapter_format)
        
        # Calculate file properties
        size_bytes = weights_file.stat().st_size
        checksum = self._calculate_checksum(weights_file)
        
        # Build manifest
        manifest = {
            "adapter_id": adapter_id,
            "domain": domain_enum.value,
            "display_name": display_name,
            "base_model_id": base_model_id,
            "adapter_format": adapter_format.value,
            "size_bytes": size_bytes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checksum": checksum,
            "signature": kwargs.get("signature", ""),
            "privacy_scope": kwargs.get("privacy_scope", ["personal_texts"]),
            "activation_memory_mb": kwargs.get("activation_memory_mb", 120),
            "recommended_layers": self._get_default_layers(kwargs),
            "validation_score": kwargs.get("validation_score", 0.0),
            "version": kwargs.get("version", "1.0.0"),
            "safe_to_share": kwargs.get("safe_to_share", False),
            "training_meta": kwargs.get("training_meta", {}),
            "notes": kwargs.get("notes", "")
        }
        
        # Validate manifest
        self._validate_manifest(manifest)
        
        # Save manifest
        manifest_path = adapter_dir / "adapter.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Manifest saved to: {manifest_path}")
        return manifest
    
    def _validate_domain(self, domain: str) -> Domain:
        """Validate and convert domain string to enum."""
        domain_lower = domain.lower()
        if domain_lower not in self.domain_mappings:
            raise ValueError(f"Invalid domain: {domain}. Must be one of: {list(self.domain_mappings.keys())}")
        
        return self.domain_mappings[domain_lower]
    
    def _detect_format(self, adapter_dir: Path) -> AdapterFormat:
        """Detect adapter format from files in directory."""
        # Look for safetensors files
        if (adapter_dir / "adapter_model.safetensors").exists():
            return AdapterFormat.SAFETENSORS_V1
        
        # Look for GGUF files
        gguf_files = list(adapter_dir.glob("*.gguf"))
        if gguf_files:
            return AdapterFormat.GGUF_LORA_V1
        
        # Look for TFLite files
        tflite_files = list(adapter_dir.glob("*.tflite"))
        if tflite_files:
            return AdapterFormat.TFLITE_LORA_V1
        
        raise ValueError(f"No supported adapter files found in {adapter_dir}")
    
    def _find_weights_file(self, adapter_dir: Path, format: AdapterFormat) -> Path:
        """Find the main weights file for the adapter."""
        if format == AdapterFormat.SAFETENSORS_V1:
            weights_file = adapter_dir / "adapter_model.safetensors"
            if not weights_file.exists():
                raise FileNotFoundError(f"Expected safetensors file not found: {weights_file}")
            return weights_file
        
        elif format == AdapterFormat.GGUF_LORA_V1:
            gguf_files = list(adapter_dir.glob("*.gguf"))
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {adapter_dir}")
            return gguf_files[0]  # Take first GGUF file
        
        elif format == AdapterFormat.TFLITE_LORA_V1:
            tflite_files = list(adapter_dir.glob("*.tflite"))
            if not tflite_files:
                raise FileNotFoundError(f"No TFLite files found in {adapter_dir}")
            return tflite_files[0]  # Take first TFLite file
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file."""
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                hasher.update(chunk)
        
        return f"sha256:{hasher.hexdigest()}"
    
    def _get_default_layers(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get default layer configuration."""
        return {
            "inject_every": kwargs.get("inject_every", 0),
            "target_modules": kwargs.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            "alpha": kwargs.get("lora_alpha", 32),
            "rank": kwargs.get("lora_r", 16)
        }
    
    def _validate_manifest(self, manifest: Dict[str, Any]) -> None:
        """Validate manifest structure and values."""
        required_fields = [
            "adapter_id", "domain", "display_name", "base_model_id",
            "adapter_format", "size_bytes", "created_at", "checksum",
            "privacy_scope", "activation_memory_mb", "recommended_layers",
            "version", "safe_to_share"
        ]
        
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate specific fields
        if manifest["size_bytes"] <= 0:
            raise ValueError("size_bytes must be positive")
        
        if manifest["activation_memory_mb"] <= 0:
            raise ValueError("activation_memory_mb must be positive")
        
        if not manifest["checksum"].startswith("sha256:"):
            raise ValueError("checksum must be in format 'sha256:...'")
        
        logger.info("Manifest validation passed")

def load_manifest(manifest_path: Path) -> AdapterMeta:
    """Load manifest from file and convert to AdapterMeta."""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return AdapterMeta.from_dict(data)

def validate_manifest_file(manifest_path: Path) -> bool:
    """Validate manifest file structure."""
    try:
        manifest = load_manifest(manifest_path)
        logger.info(f"Manifest validation passed: {manifest.adapter_id}")
        return True
    except Exception as e:
        logger.error(f"Manifest validation failed: {e}")
        return False

def create_adapters_index(adapters_dir: Path) -> Dict[str, Any]:
    """Create index of all adapters in directory."""
    index = {
        "version": "1.0.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapters": []
    }
    
    # Find all adapter directories
    for adapter_dir in adapters_dir.iterdir():
        if not adapter_dir.is_dir():
            continue
        
        manifest_file = adapter_dir / "adapter.json"
        if not manifest_file.exists():
            continue
        
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                index["adapters"].append(manifest)
            
        except Exception as e:
            logger.warning(f"Failed to load manifest from {adapter_dir}: {e}")
            continue
    
    # Save index
    index_file = adapters_dir / "adapters_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created adapters index with {len(index['adapters'])} adapters")
    return index

def main():
    parser = argparse.ArgumentParser(description="Create adapter manifest")
    parser.add_argument("--adapter-id", required=True, help="Adapter identifier")
    parser.add_argument("--adapter-dir", required=True, type=Path, help="Adapter directory")
    parser.add_argument("--domain", required=True, help="Adapter domain")
    parser.add_argument("--display-name", required=True, help="Display name")
    parser.add_argument("--base-model-id", required=True, help="Base model identifier")
    parser.add_argument("--activation-memory-mb", type=int, default=120, help="Activation memory")
    parser.add_argument("--validation-score", type=float, default=0.0, help="Validation score")
    parser.add_argument("--version", default="1.0.0", help="Version")
    parser.add_argument("--notes", default="", help="Additional notes")
    
    args = parser.parse_args()
    
    if not args.adapter_dir.exists():
        print(f"Error: Adapter directory does not exist: {args.adapter_dir}")
        return 1
    
    creator = ManifestCreator()
    
    try:
        manifest = creator.create_manifest(
            adapter_id=args.adapter_id,
            adapter_dir=args.adapter_dir,
            domain=args.domain,
            display_name=args.display_name,
            base_model_id=args.base_model_id,
            activation_memory_mb=args.activation_memory_mb,
            validation_score=args.validation_score,
            version=args.version,
            notes=args.notes
        )
        
        print("Manifest created successfully!")
        print(json.dumps(manifest, indent=2))
        
    except Exception as e:
        print(f"Error creating manifest: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
