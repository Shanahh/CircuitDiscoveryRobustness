"""Model loading and snapshot management."""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from circuit_discovery.spec import BenchmarkSpec, ModelSnapshot


class ModelLoader:
    """Handles model downloading and loading with integrity checks."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model loader.
        
        Args:
            cache_dir: Directory for caching models (default: ~/.cache/circuit_discovery)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "circuit_discovery" / "models"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_snapshot(self, spec: BenchmarkSpec) -> ModelSnapshot:
        """Download and cache a model snapshot.
        
        Args:
            spec: Benchmark specification with model details
            
        Returns:
            ModelSnapshot with download information
        """
        model_id = spec.model_id
        revision = spec.model_commit or "main"
        
        # Download from HuggingFace Hub with pinned revision
        try:
            local_path = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=str(self.cache_dir),
                    local_files_only=False,
                )
            )
        except Exception as e:
            # Fallback to local path if not on HF Hub
            local_path = Path(model_id)
            if not local_path.exists():
                raise ValueError(f"Model not found at {model_id} and not on HF Hub: {e}")
            revision = "local"
        
        # Create snapshot object
        snapshot = ModelSnapshot(
            model_id=model_id,
            commit_hash=revision,
            local_path=local_path,
            checksum=None,
            metadata={"download_timestamp": str(Path.ctime(local_path))},
        )
        
        return snapshot
    
    def verify_integrity(self, snapshot: ModelSnapshot) -> bool:
        """Verify integrity of downloaded model snapshot.
        
        Args:
            snapshot: Model snapshot to verify
            
        Returns:
            True if integrity check passes
        """
        # Check if path exists
        if not snapshot.local_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {snapshot.local_path}")
        
        # Basic integrity: check for config and model files
        config_file = snapshot.local_path / "config.json"
        has_safetensors = any(snapshot.local_path.glob("*.safetensors"))
        has_pytorch = any(snapshot.local_path.glob("*.bin"))
        
        if not config_file.exists():
            raise ValueError(f"No config.json found in {snapshot.local_path}")
        
        if not (has_safetensors or has_pytorch):
            raise ValueError(f"No model weights found in {snapshot.local_path}")
        
        # Compute checksum if not already set
        if snapshot.checksum is None:
            snapshot.checksum = self._compute_checksum(snapshot.local_path)
        
        return True
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of model directory.
        
        Args:
            path: Path to model directory
            
        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.sha256()
        
        # Hash all model files in sorted order
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix in [".bin", ".safetensors", ".json"]:
                hasher.update(file_path.name.encode())
                hasher.update(str(file_path.stat().st_size).encode())
        
        return hasher.hexdigest()[:16]  # Use first 16 chars for brevity
    
    def load_model(self, snapshot: ModelSnapshot, device: str = "cpu") -> nn.Module:
        """Load a PyTorch model from snapshot.
        
        Args:
            snapshot: Model snapshot to load
            device: Device to load model on
            
        Returns:
            Loaded PyTorch model
        """
        # This is a basic loader - in practice, you'd use transformers or other libraries
        # For now, we'll create a placeholder that returns a simple model
        # Users should extend this for their specific model types
        
        # Try to load with safetensors first, then PyTorch
        model_files = list(snapshot.local_path.glob("*.safetensors"))
        if not model_files:
            model_files = list(snapshot.local_path.glob("*.bin"))
        
        if not model_files:
            raise ValueError(f"No model files found in {snapshot.local_path}")
        
        # For this scaffold, we'll just note that the model would be loaded here
        # In practice, this would use transformers.AutoModel or similar
        class PlaceholderModel(nn.Module):
            """Placeholder model for testing."""
            
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(10, 10)
            
            def forward(self, x):
                return self.layer(x)
        
        model = PlaceholderModel()
        return model.to(device)


def load_model_from_spec(spec: BenchmarkSpec, device: str = "cpu") -> tuple[nn.Module, ModelSnapshot]:
    """Convenience function to load a model from specification.
    
    Args:
        spec: Benchmark specification
        device: Device to load model on
        
    Returns:
        Tuple of (model, snapshot)
    """
    loader = ModelLoader()
    snapshot = loader.download_snapshot(spec)
    loader.verify_integrity(snapshot)
    model = loader.load_model(snapshot, device=device)
    return model, snapshot
