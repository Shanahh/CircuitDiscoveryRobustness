"""Tests for subgraph module."""
import pytest

from circuit_discovery.graph_export import ComputationalGraph, GraphNode
from circuit_discovery.subgraph import (
    SubgraphSlicer,
    CanonicalNodeMapper,
    slice_graph,
)


class TestSubgraphSlicer:
    """Test SubgraphSlicer class."""
    
    def test_create_slicer(self):
        """Test creating a subgraph slicer."""
        graph = ComputationalGraph()
        slicer = SubgraphSlicer(graph)
        assert slicer.graph is graph
    
    def test_slice_from_targets_no_dependencies(self):
        """Test slicing without dependencies."""
        # Create graph
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="n1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="n2", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n3", name="n3", op="op", target="t"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        slicer = SubgraphSlicer(graph)
        subgraph = slicer.slice_from_targets(["n3"], include_dependencies=False)
        
        assert len(subgraph.nodes) == 1
        assert "n3" in subgraph.nodes
    
    def test_slice_from_targets_with_dependencies(self):
        """Test slicing with dependencies."""
        # Create linear graph: n1 -> n2 -> n3
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="n1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="n2", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n3", name="n3", op="op", target="t"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        slicer = SubgraphSlicer(graph)
        subgraph = slicer.slice_from_targets(["n3"], include_dependencies=True)
        
        # Should include all nodes in the path
        assert len(subgraph.nodes) == 3
        assert "n1" in subgraph.nodes
        assert "n2" in subgraph.nodes
        assert "n3" in subgraph.nodes
    
    def test_slice_forward(self):
        """Test forward slicing."""
        # Create graph: n1 -> n2 -> n3
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="n1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="n2", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n3", name="n3", op="op", target="t"))
        graph.add_edge("n1", "n2")
        graph.add_edge("n2", "n3")
        
        slicer = SubgraphSlicer(graph)
        subgraph = slicer.slice_forward(["n1"])
        
        # Should include all nodes reachable from n1
        assert len(subgraph.nodes) == 3
        assert "n1" in subgraph.nodes
        assert "n2" in subgraph.nodes
        assert "n3" in subgraph.nodes
    
    def test_slice_between(self):
        """Test slicing between source and target."""
        # Create graph with branching
        graph = ComputationalGraph()
        for i in range(1, 6):
            graph.add_node(GraphNode(node_id=f"n{i}", name=f"n{i}", op="op", target="t"))
        
        # n1 -> n2 -> n4
        # n1 -> n3 -> n5
        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")
        graph.add_edge("n2", "n4")
        graph.add_edge("n3", "n5")
        
        slicer = SubgraphSlicer(graph)
        subgraph = slicer.slice_between(["n1"], ["n4"])
        
        # Should include n1, n2, n4 (not n3, n5)
        assert "n1" in subgraph.nodes
        assert "n2" in subgraph.nodes
        assert "n4" in subgraph.nodes
        # n3 and n5 should not be in the path from n1 to n4
    
    def test_slice_graph_convenience(self):
        """Test convenience function."""
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="n1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="n2", op="op", target="t"))
        graph.add_edge("n1", "n2")
        
        subgraph = slice_graph(graph, ["n2"], include_dependencies=True)
        
        assert isinstance(subgraph, ComputationalGraph)
        assert len(subgraph.nodes) == 2


class TestCanonicalNodeMapper:
    """Test CanonicalNodeMapper class."""
    
    def test_create_mapper(self):
        """Test creating a mapper."""
        mapper = CanonicalNodeMapper()
        assert len(mapper.name_to_canonical) == 0
    
    def test_register_mapping(self):
        """Test registering mappings."""
        mapper = CanonicalNodeMapper()
        mapper.register_mapping("layer1", "canonical_1")
        
        assert mapper.get_canonical("layer1") == "canonical_1"
        assert mapper.get_name("canonical_1") == "layer1"
    
    def test_get_canonical_unknown(self):
        """Test getting canonical ID for unknown name."""
        mapper = CanonicalNodeMapper()
        # Should return the name itself if not found
        assert mapper.get_canonical("unknown") == "unknown"
    
    def test_from_graph(self):
        """Test creating mapper from graph."""
        graph = ComputationalGraph()
        graph.add_node(GraphNode(node_id="n1", name="layer1", op="op", target="t"))
        graph.add_node(GraphNode(node_id="n2", name="layer2", op="op", target="t"))
        
        mapper = CanonicalNodeMapper.from_graph(graph)
        
        assert mapper.get_canonical("layer1") == "n1"
        assert mapper.get_canonical("layer2") == "n2"
        assert mapper.get_name("n1") == "layer1"
