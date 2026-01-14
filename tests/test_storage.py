"""Tests for storage module."""
import tempfile
from pathlib import Path

import pytest
import torch

from circuit_discovery.algorithms import CircuitDiscoveryResult
from circuit_discovery.execution import ActivationCache, GradientCache
from circuit_discovery.spec import BenchmarkSpec
from circuit_discovery.storage import ResultStorage, create_storage


class TestResultStorage:
    """Test ResultStorage class."""
    
    def test_create_storage(self):
        """Test creating storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            assert storage.base_dir.exists()
    
    def test_save_load_spec(self):
        """Test saving and loading spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            spec = BenchmarkSpec(
                model_id="gpt2",
                task="ioi",
                algorithm="acdc",
            )
            
            run_id = "test_run_001"
            storage.save_spec(run_id, spec)
            
            loaded_spec = storage.load_spec(run_id)
            assert loaded_spec.model_id == spec.model_id
            assert loaded_spec.task == spec.task
            assert loaded_spec.algorithm == spec.algorithm
    
    def test_save_load_result(self):
        """Test saving and loading result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            result = CircuitDiscoveryResult(
                algorithm="test",
                important_nodes=["n1", "n2"],
                node_scores={"n1": 0.8, "n2": 0.6},
            )
            
            run_id = "test_run_002"
            storage.save_result(run_id, result, metadata={"key": "value"})
            
            loaded_result = storage.load_result(run_id)
            assert loaded_result["algorithm"] == "test"
            assert "n1" in loaded_result["important_nodes"]
            assert loaded_result["_metadata"]["key"] == "value"
    
    def test_save_load_activations(self):
        """Test saving and loading activations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            cache = ActivationCache()
            cache.store("n1", torch.randn(2, 10))
            cache.store("n1", torch.randn(2, 10))
            cache.store("n2", torch.randn(2, 5))
            
            run_id = "test_run_003"
            storage.save_activations(run_id, cache)
            
            loaded_cache = storage.load_activations(run_id)
            assert "n1" in loaded_cache.activations
            assert "n2" in loaded_cache.activations
            assert len(loaded_cache.activations["n1"]) == 2
    
    def test_save_load_gradients(self):
        """Test saving and loading gradients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            cache = GradientCache()
            cache.store("n1", torch.randn(2, 10))
            cache.store("n2", torch.randn(2, 5))
            
            run_id = "test_run_004"
            storage.save_gradients(run_id, cache)
            
            loaded_cache = storage.load_gradients(run_id)
            assert "n1" in loaded_cache.gradients
            assert "n2" in loaded_cache.gradients
    
    def test_list_runs(self):
        """Test listing runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            # Create some runs
            spec = BenchmarkSpec(
                model_id="model",
                task="task",
                algorithm="algo",
            )
            
            storage.save_spec("run1", spec)
            storage.save_spec("run2", spec)
            
            runs = storage.list_runs()
            assert "run1" in runs
            assert "run2" in runs
    
    def test_get_run_info(self):
        """Test getting run information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = ResultStorage(Path(tmpdir))
            
            spec = BenchmarkSpec(
                model_id="gpt2",
                task="ioi",
                algorithm="acdc",
            )
            result = CircuitDiscoveryResult(algorithm="acdc")
            
            run_id = "test_run_005"
            storage.save_spec(run_id, spec)
            storage.save_result(run_id, result)
            
            info = storage.get_run_info(run_id)
            assert info["run_id"] == run_id
            assert info["has_spec"] is True
            assert info["has_result"] is True
            assert info["spec"]["model_id"] == "gpt2"
    
    def test_create_storage_convenience(self):
        """Test convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = create_storage(Path(tmpdir))
            assert isinstance(storage, ResultStorage)
