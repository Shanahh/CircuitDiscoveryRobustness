"""Results storage with versioning for replay."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from circuit_discovery.algorithms import CircuitDiscoveryResult
from circuit_discovery.execution import ActivationCache, GradientCache
from circuit_discovery.spec import BenchmarkSpec


class ResultStorage:
    """Manages storage and retrieval of benchmark results."""
    
    def __init__(self, base_dir: Path):
        """Initialize result storage.
        
        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Path to run directory
        """
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def save_spec(self, run_id: str, spec: BenchmarkSpec) -> None:
        """Save benchmark specification.
        
        Args:
            run_id: Run identifier
            spec: Benchmark specification
        """
        run_dir = self._get_run_dir(run_id)
        spec_path = run_dir / "spec.yaml"
        spec.to_yaml(spec_path)
    
    def load_spec(self, run_id: str) -> BenchmarkSpec:
        """Load benchmark specification.
        
        Args:
            run_id: Run identifier
            
        Returns:
            BenchmarkSpec
        """
        run_dir = self._get_run_dir(run_id)
        spec_path = run_dir / "spec.yaml"
        return BenchmarkSpec.from_yaml(spec_path)
    
    def save_result(
        self,
        run_id: str,
        result: CircuitDiscoveryResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save circuit discovery result.
        
        Args:
            run_id: Run identifier
            result: Circuit discovery result
            metadata: Additional metadata
        """
        run_dir = self._get_run_dir(run_id)
        result_path = run_dir / "result.json"
        
        # Add version information
        result_data = result.to_dict()
        result_data["_metadata"] = metadata or {}
        result_data["_metadata"]["timestamp"] = datetime.now().isoformat()
        result_data["_metadata"]["pytorch_version"] = torch.__version__
        
        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)
    
    def load_result(self, run_id: str) -> Dict[str, Any]:
        """Load circuit discovery result.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Result dictionary
        """
        run_dir = self._get_run_dir(run_id)
        result_path = run_dir / "result.json"
        
        with open(result_path, "r") as f:
            return json.load(f)
    
    def save_activations(
        self,
        run_id: str,
        activation_cache: ActivationCache,
        compress: bool = True,
    ) -> None:
        """Save activation cache.
        
        Args:
            run_id: Run identifier
            activation_cache: Activation cache to save
            compress: Whether to compress the file
        """
        run_dir = self._get_run_dir(run_id)
        cache_path = run_dir / "activations.pt"
        
        # Convert to saveable format
        cache_data = {
            "activations": {
                node_id: torch.stack(acts) if acts else torch.empty(0)
                for node_id, acts in activation_cache.activations.items()
            },
            "metadata": activation_cache.metadata,
        }
        
        torch.save(cache_data, cache_path)
    
    def load_activations(self, run_id: str) -> ActivationCache:
        """Load activation cache.
        
        Args:
            run_id: Run identifier
            
        Returns:
            ActivationCache
        """
        run_dir = self._get_run_dir(run_id)
        cache_path = run_dir / "activations.pt"
        
        cache_data = torch.load(cache_path, weights_only=False)
        
        cache = ActivationCache(metadata=cache_data["metadata"])
        for node_id, stacked in cache_data["activations"].items():
            if stacked.numel() > 0:
                # Convert back to list of tensors
                cache.activations[node_id] = [t for t in stacked]
        
        return cache
    
    def save_gradients(
        self,
        run_id: str,
        gradient_cache: GradientCache,
        compress: bool = True,
    ) -> None:
        """Save gradient cache.
        
        Args:
            run_id: Run identifier
            gradient_cache: Gradient cache to save
            compress: Whether to compress the file
        """
        run_dir = self._get_run_dir(run_id)
        cache_path = run_dir / "gradients.pt"
        
        # Convert to saveable format
        cache_data = {
            "gradients": {
                node_id: torch.stack(grads) if grads else torch.empty(0)
                for node_id, grads in gradient_cache.gradients.items()
            },
            "metadata": gradient_cache.metadata,
        }
        
        torch.save(cache_data, cache_path)
    
    def load_gradients(self, run_id: str) -> GradientCache:
        """Load gradient cache.
        
        Args:
            run_id: Run identifier
            
        Returns:
            GradientCache
        """
        run_dir = self._get_run_dir(run_id)
        cache_path = run_dir / "gradients.pt"
        
        cache_data = torch.load(cache_path, weights_only=False)
        
        cache = GradientCache(metadata=cache_data["metadata"])
        for node_id, stacked in cache_data["gradients"].items():
            if stacked.numel() > 0:
                # Convert back to list of tensors
                cache.gradients[node_id] = [t for t in stacked]
        
        return cache
    
    def list_runs(self) -> list[str]:
        """List all available runs.
        
        Returns:
            List of run IDs
        """
        if not self.base_dir.exists():
            return []
        
        runs = []
        for path in self.base_dir.iterdir():
            if path.is_dir():
                runs.append(path.name)
        
        return sorted(runs)
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """Get information about a run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with run information
        """
        run_dir = self._get_run_dir(run_id)
        
        info = {
            "run_id": run_id,
            "path": str(run_dir),
            "has_spec": (run_dir / "spec.yaml").exists(),
            "has_result": (run_dir / "result.json").exists(),
            "has_activations": (run_dir / "activations.pt").exists(),
            "has_gradients": (run_dir / "gradients.pt").exists(),
        }
        
        # Add spec info if available
        if info["has_spec"]:
            spec = self.load_spec(run_id)
            info["spec"] = {
                "model_id": spec.model_id,
                "task": spec.task,
                "algorithm": spec.algorithm,
                "seed": spec.seed,
            }
        
        # Add result metadata if available
        if info["has_result"]:
            result = self.load_result(run_id)
            if "_metadata" in result:
                info["metadata"] = result["_metadata"]
        
        return info


def create_storage(base_dir: Path) -> ResultStorage:
    """Convenience function to create result storage.
    
    Args:
        base_dir: Base directory for storage
        
    Returns:
        ResultStorage instance
    """
    return ResultStorage(base_dir)
