"""Tests for specification module."""
import tempfile
from pathlib import Path

import pytest

from circuit_discovery.spec import BenchmarkSpec, ModelSnapshot


class TestBenchmarkSpec:
    """Test BenchmarkSpec class."""
    
    def test_create_spec(self):
        """Test creating a specification."""
        spec = BenchmarkSpec(
            model_id="gpt2",
            task="ioi",
            algorithm="attribution_patching",
            seed=42,
            batch_size=8,
            sequence_length=128,
        )
        
        assert spec.model_id == "gpt2"
        assert spec.task == "ioi"
        assert spec.algorithm == "attribution_patching"
        assert spec.seed == 42
        assert spec.batch_size == 8
        assert spec.sequence_length == 128
        assert spec.cache_activations is True
        assert spec.cache_gradients is False
    
    def test_spec_validation(self):
        """Test specification validation."""
        # Valid spec
        spec = BenchmarkSpec(
            model_id="model",
            task="task",
            algorithm="algo",
            batch_size=1,
            sequence_length=1,
        )
        assert spec.batch_size == 1
        
        # Invalid batch size
        with pytest.raises(Exception):
            BenchmarkSpec(
                model_id="model",
                task="task",
                algorithm="algo",
                batch_size=0,  # Invalid
                sequence_length=1,
            )
    
    def test_yaml_serialization(self):
        """Test YAML serialization/deserialization."""
        spec = BenchmarkSpec(
            model_id="gpt2",
            task="ioi",
            algorithm="acdc",
            seed=123,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.yaml"
            
            # Save
            spec.to_yaml(path)
            assert path.exists()
            
            # Load
            loaded = BenchmarkSpec.from_yaml(path)
            assert loaded.model_id == spec.model_id
            assert loaded.task == spec.task
            assert loaded.algorithm == spec.algorithm
            assert loaded.seed == spec.seed
    
    def test_dict_conversion(self):
        """Test dictionary conversion."""
        spec = BenchmarkSpec(
            model_id="model",
            task="task",
            algorithm="algo",
        )
        
        # To dict
        data = spec.to_dict()
        assert isinstance(data, dict)
        assert data["model_id"] == "model"
        assert data["task"] == "task"
        
        # From dict
        spec2 = BenchmarkSpec.from_dict(data)
        assert spec2.model_id == spec.model_id
        assert spec2.task == spec.task


class TestModelSnapshot:
    """Test ModelSnapshot class."""
    
    def test_create_snapshot(self):
        """Test creating a model snapshot."""
        snapshot = ModelSnapshot(
            model_id="gpt2",
            commit_hash="abc123",
            local_path=Path("/tmp/model"),
            checksum="def456",
        )
        
        assert snapshot.model_id == "gpt2"
        assert snapshot.commit_hash == "abc123"
        assert snapshot.local_path == Path("/tmp/model")
        assert snapshot.checksum == "def456"
    
    def test_snapshot_metadata(self):
        """Test snapshot metadata."""
        snapshot = ModelSnapshot(
            model_id="model",
            commit_hash="hash",
            local_path=Path("/tmp"),
            metadata={"key": "value"},
        )
        
        assert snapshot.metadata["key"] == "value"
