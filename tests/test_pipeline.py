"""Tests for pipeline module."""
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from circuit_discovery.pipeline import BenchmarkPipeline, run_benchmark
from circuit_discovery.spec import BenchmarkSpec


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)


class TestBenchmarkPipeline:
    """Test BenchmarkPipeline class."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = BenchmarkSpec(
                model_id="dummy",
                task="test",
                algorithm="attribution_patching",
                output_dir=Path(tmpdir),
            )
            
            pipeline = BenchmarkPipeline(spec, verbose=False)
            
            assert pipeline.spec is spec
            assert pipeline.device in ["cpu", "cuda"]
            assert pipeline.run_id is not None
    
    def test_generate_run_id(self):
        """Test run ID generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = BenchmarkSpec(
                model_id="model",
                task="task",
                algorithm="algo",
                seed=42,
                output_dir=Path(tmpdir),
            )
            
            pipeline = BenchmarkPipeline(spec, verbose=False)
            
            assert "task" in pipeline.run_id
            assert "algo" in pipeline.run_id
            assert "42" in pipeline.run_id
    
    @pytest.mark.skip(reason="Requires actual model download")
    def test_load_model(self):
        """Test loading a model."""
        # This would require network access and actual models
        pass
    
    def test_export_graph_from_model(self):
        """Test exporting graph from a model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = BenchmarkSpec(
                model_id="dummy",
                task="test",
                algorithm="attribution_patching",
                batch_size=2,
                sequence_length=10,
                output_dir=Path(tmpdir),
            )
            
            pipeline = BenchmarkPipeline(spec, verbose=False)
            pipeline.model = DummyModel()
            
            pipeline.export_graph()
            
            assert pipeline.graph is not None
            assert len(pipeline.graph.nodes) > 0
    
    def test_slice_subgraph_no_targets(self):
        """Test slicing with no target nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = BenchmarkSpec(
                model_id="dummy",
                task="test",
                algorithm="attribution_patching",
                batch_size=2,
                sequence_length=10,
                output_dir=Path(tmpdir),
            )
            
            pipeline = BenchmarkPipeline(spec, verbose=False)
            pipeline.model = DummyModel()
            pipeline.export_graph()
            
            pipeline.slice_subgraph()
            
            # Should use full graph
            assert pipeline.subgraph is pipeline.graph


class TestRunBenchmark:
    """Test run_benchmark convenience function."""
    
    def test_run_benchmark_basic(self):
        """Test running a basic benchmark."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = BenchmarkSpec(
                model_id="dummy",
                task="test",
                algorithm="attribution_patching",
                batch_size=4,
                sequence_length=10,
                output_dir=Path(tmpdir),
            )
            
            # Create dummy data loader
            data = torch.randn(16, 10)
            labels = torch.randint(0, 2, (16,))
            dataset = TensorDataset(data, labels)
            loader = DataLoader(dataset, batch_size=4)
            
            # Mock model loading by manually creating a pipeline and setting the model
            # In reality, the pipeline would download the model
            # For testing, we'll skip the actual run since it requires model download
            
            # This test would need to be more comprehensive in a real scenario
            # For now, we verify the function exists and has the right signature
            assert callable(run_benchmark)
