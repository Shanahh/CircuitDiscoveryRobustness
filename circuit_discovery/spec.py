"""Specification for circuit discovery benchmarks."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class BenchmarkSpec(BaseModel):
    """Specification for a circuit discovery benchmark run.
    
    Attributes:
        model_id: HuggingFace model ID or path
        model_commit: Git commit hash or tag for reproducibility
        task: Task name (e.g., 'ioi', 'greaterthan', 'tracr')
        algorithm: Algorithm name (e.g., 'attribution_patching', 'acdc', 'eap')
        seed: Random seed for reproducibility
        batch_size: Number of samples per batch (B)
        sequence_length: Maximum sequence length (S)
        target_nodes: Target nodes/layers to analyze
        cache_activations: Whether to cache activations
        cache_gradients: Whether to cache gradients
        output_dir: Directory for results storage
        metadata: Additional metadata
    """
    
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    model_commit: Optional[str] = Field(None, description="Git commit hash for model version")
    task: str = Field(..., description="Task name")
    algorithm: str = Field(..., description="Algorithm name")
    seed: int = Field(42, description="Random seed")
    batch_size: int = Field(8, ge=1, description="Batch size (B)")
    sequence_length: int = Field(512, ge=1, description="Sequence length (S)")
    target_nodes: List[str] = Field(default_factory=list, description="Target nodes to analyze")
    cache_activations: bool = Field(True, description="Cache activations")
    cache_gradients: bool = Field(False, description="Cache gradients")
    output_dir: Path = Field(Path("results"), description="Output directory")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('output_dir', mode='before')
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert output_dir to Path."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkSpec":
        """Load specification from YAML file.
        
        Args:
            path: Path to YAML specification file
            
        Returns:
            BenchmarkSpec instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSpec":
        """Load specification from dictionary.
        
        Args:
            data: Dictionary with specification parameters
            
        Returns:
            BenchmarkSpec instance
        """
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save specification to YAML file.
        
        Args:
            path: Path to save YAML file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # Convert to dict and handle Path objects
            data = self.model_dump()
            data['output_dir'] = str(data['output_dir'])
            yaml.safe_dump(data, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        data = self.model_dump()
        data['output_dir'] = str(data['output_dir'])
        return data


@dataclass
class ModelSnapshot:
    """Information about a model snapshot.
    
    Attributes:
        model_id: Model identifier
        commit_hash: Commit hash
        local_path: Local path to cached model
        checksum: Integrity checksum
        metadata: Additional metadata
    """
    
    model_id: str
    commit_hash: str
    local_path: Path
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
