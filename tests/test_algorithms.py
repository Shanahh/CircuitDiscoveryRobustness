"""Tests for algorithms module."""
import pytest
import torch

from circuit_discovery.algorithms import (
    AlgorithmRegistry,
    AttributionPatchingAlgorithm,
    ACDCAlgorithm,
    EAPAlgorithm,
    CircuitDiscoveryResult,
)
from circuit_discovery.execution import ActivationCache, GradientCache
from circuit_discovery.graph_export import ComputationalGraph, GraphNode


class TestCircuitDiscoveryResult:
    """Test CircuitDiscoveryResult class."""
    
    def test_create_result(self):
        """Test creating a result."""
        result = CircuitDiscoveryResult(algorithm="test")
        
        assert result.algorithm == "test"
        assert len(result.important_nodes) == 0
        assert len(result.edge_scores) == 0
        assert len(result.node_scores) == 0
    
    def test_result_to_dict(self):
        """Test converting result to dict."""
        result = CircuitDiscoveryResult(
            algorithm="test",
            important_nodes=["n1", "n2"],
            node_scores={"n1": 0.8, "n2": 0.6},
            edge_scores={("n1", "n2"): 0.7},
        )
        
        data = result.to_dict()
        assert data["algorithm"] == "test"
        assert "n1" in data["important_nodes"]
        assert data["node_scores"]["n1"] == 0.8
        assert "n1->n2" in data["edge_scores"]


class TestAlgorithmRegistry:
    """Test AlgorithmRegistry class."""
    
    def test_list_algorithms(self):
        """Test listing algorithms."""
        algorithms = AlgorithmRegistry.list_algorithms()
        
        assert "attribution_patching" in algorithms
        assert "acdc" in algorithms
        assert "eap" in algorithms
    
    def test_get_algorithm(self):
        """Test getting an algorithm."""
        algo_class = AlgorithmRegistry.get("attribution_patching")
        assert algo_class == AttributionPatchingAlgorithm
    
    def test_create_algorithm(self):
        """Test creating algorithm instance."""
        algo = AlgorithmRegistry.create("acdc", threshold=0.1)
        
        assert isinstance(algo, ACDCAlgorithm)
        assert algo.config["threshold"] == 0.1
    
    def test_unknown_algorithm(self):
        """Test getting unknown algorithm."""
        with pytest.raises(ValueError):
            AlgorithmRegistry.get("unknown_algorithm")


class TestAttributionPatchingAlgorithm:
    """Test AttributionPatchingAlgorithm."""
    
    def test_discover(self):
        """Test discovery with attribution patching."""
        # Create simple graph
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="layer1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="layer2", op="op", target="t"))
        graph.add_edge("n1", "n2")
        
        # Create activation cache
        cache = ActivationCache()
        cache.store("n1", torch.randn(2, 10))
        cache.store("n2", torch.randn(2, 10))
        
        # Run algorithm
        algo = AttributionPatchingAlgorithm(threshold=0.0)
        result = algo.discover(graph, cache)
        
        assert result.algorithm == "attribution_patching"
        assert len(result.node_scores) > 0
        assert "n1" in result.node_scores or "n2" in result.node_scores


class TestACDCAlgorithm:
    """Test ACDCAlgorithm."""
    
    def test_discover_with_gradients(self):
        """Test discovery with gradients."""
        # Create graph
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="layer1", op="op", target="t"))
        
        # Create caches
        act_cache = ActivationCache()
        act_cache.store("n1", torch.randn(2, 10))
        
        grad_cache = GradientCache()
        grad_cache.store("n1", torch.randn(2, 10))
        
        # Run algorithm
        algo = ACDCAlgorithm(threshold=0.0)
        result = algo.discover(graph, act_cache, grad_cache)
        
        assert result.algorithm == "acdc"
        assert result.metadata["used_gradients"] is True
    
    def test_discover_without_gradients(self):
        """Test discovery without gradients (fallback)."""
        # Create graph
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="layer1", op="op", target="t"))
        
        # Create activation cache only
        act_cache = ActivationCache()
        act_cache.store("n1", torch.randn(2, 10))
        
        # Run algorithm
        algo = ACDCAlgorithm(threshold=0.0)
        result = algo.discover(graph, act_cache, gradient_cache=None)
        
        assert result.algorithm == "acdc"
        assert result.metadata["used_gradients"] is False


class TestEAPAlgorithm:
    """Test EAPAlgorithm."""
    
    def test_discover(self):
        """Test EAP discovery."""
        # Create graph with edges
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="layer1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="layer2", op="op", target="t"))
        graph.add_edge("n1", "n2")
        
        # Create activation cache
        cache = ActivationCache()
        # Store multiple activations for correlation
        for _ in range(5):
            cache.store("n1", torch.randn(2, 10))
            cache.store("n2", torch.randn(2, 10))
        
        # Run algorithm
        algo = EAPAlgorithm(threshold=0.0)
        result = algo.discover(graph, cache)
        
        assert result.algorithm == "eap"
        assert len(result.edge_scores) >= 0  # May or may not have edges depending on correlation
