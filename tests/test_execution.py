"""Tests for execution module."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from circuit_discovery.execution import (
    ActivationCache,
    GradientCache,
    BatchExecutor,
    create_executor,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestActivationCache:
    """Test ActivationCache class."""
    
    def test_create_cache(self):
        """Test creating an activation cache."""
        cache = ActivationCache()
        assert len(cache.activations) == 0
    
    def test_store_activation(self):
        """Test storing activations."""
        cache = ActivationCache()
        tensor = torch.randn(2, 10)
        
        cache.store("node1", tensor)
        assert "node1" in cache.activations
        assert len(cache.activations["node1"]) == 1
    
    def test_get_activation(self):
        """Test retrieving activations."""
        cache = ActivationCache()
        tensor = torch.randn(2, 10)
        
        cache.store("node1", tensor)
        retrieved = cache.get("node1")
        
        assert retrieved is not None
        assert len(retrieved) == 1
    
    def test_get_stacked(self):
        """Test getting stacked activations."""
        cache = ActivationCache()
        
        cache.store("node1", torch.randn(2, 10))
        cache.store("node1", torch.randn(2, 10))
        
        stacked = cache.get_stacked("node1")
        assert stacked is not None
        assert stacked.shape[0] == 2  # 2 batches
    
    def test_clear_cache(self):
        """Test clearing cache."""
        cache = ActivationCache()
        cache.store("node1", torch.randn(2, 10))
        
        cache.clear()
        assert len(cache.activations) == 0


class TestGradientCache:
    """Test GradientCache class."""
    
    def test_create_cache(self):
        """Test creating a gradient cache."""
        cache = GradientCache()
        assert len(cache.gradients) == 0
    
    def test_store_gradient(self):
        """Test storing gradients."""
        cache = GradientCache()
        tensor = torch.randn(2, 10)
        
        cache.store("node1", tensor)
        assert "node1" in cache.gradients
        assert len(cache.gradients["node1"]) == 1


class TestBatchExecutor:
    """Test BatchExecutor class."""
    
    def test_create_executor(self):
        """Test creating a batch executor."""
        model = SimpleModel()
        executor = BatchExecutor(model, cache_activations=True)
        
        assert executor.model is model
        assert executor.cache_activations is True
        assert executor.cache_gradients is False
    
    def test_run_single_batch(self):
        """Test running a single batch."""
        model = SimpleModel()
        executor = BatchExecutor(model, cache_activations=True)
        
        inputs = torch.randn(4, 10)
        outputs, act_cache, grad_cache = executor.run_single(inputs)
        
        assert outputs.shape == (4, 10)
        assert len(act_cache.activations) > 0
    
    def test_run_batches(self):
        """Test running multiple batches."""
        model = SimpleModel()
        executor = BatchExecutor(model, cache_activations=True)
        
        # Create data loader
        data = torch.randn(20, 10)
        labels = torch.randint(0, 2, (20,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=4)
        
        act_cache, grad_cache = executor.run_batches(
            loader,
            num_batches=3,
            show_progress=False,
        )
        
        assert len(act_cache.activations) > 0
    
    def test_cache_gradients(self):
        """Test caching gradients."""
        model = SimpleModel()
        executor = BatchExecutor(
            model,
            cache_activations=True,
            cache_gradients=True,
        )
        
        # Create data loader
        data = torch.randn(8, 10)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=4)
        
        # Define loss function
        def loss_fn(outputs, targets):
            return outputs.sum()
        
        act_cache, grad_cache = executor.run_batches(
            loader,
            num_batches=2,
            loss_fn=loss_fn,
            show_progress=False,
        )
        
        # Should have gradients
        assert len(grad_cache.gradients) > 0
    
    def test_create_executor_convenience(self):
        """Test convenience function."""
        model = SimpleModel()
        executor = create_executor(model, cache_activations=True)
        
        assert isinstance(executor, BatchExecutor)
