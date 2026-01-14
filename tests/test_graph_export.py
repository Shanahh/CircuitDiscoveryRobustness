"""Tests for graph export module."""
import pytest
import torch
import torch.nn as nn

from circuit_discovery.graph_export import (
    GraphExporter,
    GraphExportMethod,
    GraphNode,
    ComputationalGraph,
    export_graph,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestGraphNode:
    """Test GraphNode class."""
    
    def test_create_node(self):
        """Test creating a graph node."""
        node = GraphNode(
            node_id="node_1",
            name="fc1",
            op="call_module",
            target="fc1",
        )
        
        assert node.node_id == "node_1"
        assert node.name == "fc1"
        assert node.op == "call_module"
    
    def test_node_hash(self):
        """Test node hashing."""
        node1 = GraphNode(node_id="node_1", name="n1", op="op", target="t")
        node2 = GraphNode(node_id="node_1", name="n1", op="op", target="t")
        node3 = GraphNode(node_id="node_2", name="n2", op="op", target="t")
        
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)


class TestComputationalGraph:
    """Test ComputationalGraph class."""
    
    def test_create_graph(self):
        """Test creating a computational graph."""
        graph = ComputationalGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_node(self):
        """Test adding nodes."""
        graph = ComputationalGraph()
        node = GraphNode(node_id="n1", name="node1", op="op", target="t")
        
        graph.add_node(node)
        assert len(graph.nodes) == 1
        assert "n1" in graph.nodes
    
    def test_add_edge(self):
        """Test adding edges."""
        graph = ComputationalGraph()
        node1 = GraphNode(node_id="n1", name="node1", op="op", target="t")
        node2 = GraphNode(node_id="n2", name="node2", op="op", target="t")
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge("n1", "n2")
        
        assert "n2" in graph.edges["n1"]
    
    def test_get_children_parents(self):
        """Test getting children and parents."""
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="n1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="n2", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n3", name="n3", op="op", target="t"))
        
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        # Test children
        assert graph.get_children("n1") == ["n2"]
        assert graph.get_children("n2") == ["n3"]
        assert graph.get_children("n3") == []
        
        # Test parents
        assert graph.get_parents("n1") == []
        assert graph.get_parents("n2") == ["n1"]
        assert graph.get_parents("n3") == ["n2"]


class TestGraphExporter:
    """Test GraphExporter class."""
    
    def test_export_with_fx(self):
        """Test graph export using FX."""
        model = SimpleModel()
        exporter = GraphExporter(method=GraphExportMethod.FX)
        
        example_inputs = torch.randn(2, 10)
        graph = exporter.export(model, example_inputs)
        
        assert isinstance(graph, ComputationalGraph)
        assert graph.method == GraphExportMethod.FX
        assert len(graph.nodes) > 0
    
    def test_export_with_hooks(self):
        """Test graph export using hooks."""
        model = SimpleModel()
        exporter = GraphExporter(method=GraphExportMethod.HOOKS)
        
        example_inputs = torch.randn(2, 10)
        graph = exporter.export(model, example_inputs)
        
        assert isinstance(graph, ComputationalGraph)
        assert graph.method == GraphExportMethod.HOOKS
        assert len(graph.nodes) > 0
    
    def test_auto_export(self):
        """Test automatic method selection."""
        model = SimpleModel()
        exporter = GraphExporter(method=None)
        
        example_inputs = torch.randn(2, 10)
        graph = exporter.export(model, example_inputs)
        
        assert isinstance(graph, ComputationalGraph)
        assert len(graph.nodes) > 0
    
    def test_export_convenience_function(self):
        """Test convenience function."""
        model = SimpleModel()
        example_inputs = torch.randn(2, 10)
        
        graph = export_graph(model, example_inputs)
        
        assert isinstance(graph, ComputationalGraph)
        assert len(graph.nodes) > 0
