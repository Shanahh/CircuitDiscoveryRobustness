"""Subgraph slicing and canonical node operations."""
from typing import List, Set

from circuit_discovery.graph_export import ComputationalGraph, GraphNode


class SubgraphSlicer:
    """Slices computational graphs to extract relevant subgraphs."""
    
    def __init__(self, graph: ComputationalGraph):
        """Initialize subgraph slicer.
        
        Args:
            graph: Full computational graph
        """
        self.graph = graph
    
    def slice_from_targets(
        self,
        target_node_ids: List[str],
        include_dependencies: bool = True,
    ) -> ComputationalGraph:
        """Slice subgraph starting from target nodes.
        
        Args:
            target_node_ids: List of target node IDs to start from
            include_dependencies: Whether to include all dependencies
            
        Returns:
            Sliced ComputationalGraph
        """
        if not include_dependencies:
            # Just return nodes themselves
            subgraph = ComputationalGraph(
                method=self.graph.method,
                metadata={**self.graph.metadata, "sliced": True, "targets": target_node_ids}
            )
            for node_id in target_node_ids:
                node = self.graph.get_node(node_id)
                if node:
                    subgraph.add_node(node)
            return subgraph
        
        # Backward slice: find all dependencies
        visited = set()
        to_visit = list(target_node_ids)
        
        while to_visit:
            node_id = to_visit.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Add parents to visit
            parents = self.graph.get_parents(node_id)
            to_visit.extend(parents)
        
        # Build subgraph with all visited nodes
        subgraph = ComputationalGraph(
            method=self.graph.method,
            metadata={**self.graph.metadata, "sliced": True, "targets": target_node_ids}
        )
        
        for node_id in visited:
            node = self.graph.get_node(node_id)
            if node:
                subgraph.add_node(node)
                
                # Add edges within subgraph
                for child_id in self.graph.get_children(node_id):
                    if child_id in visited:
                        subgraph.add_edge(node_id, child_id)
        
        return subgraph
    
    def slice_forward(
        self,
        source_node_ids: List[str],
    ) -> ComputationalGraph:
        """Forward slice: find all nodes that depend on sources.
        
        Args:
            source_node_ids: List of source node IDs
            
        Returns:
            Sliced ComputationalGraph
        """
        visited = set()
        to_visit = list(source_node_ids)
        
        while to_visit:
            node_id = to_visit.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Add children to visit
            children = self.graph.get_children(node_id)
            to_visit.extend(children)
        
        # Build subgraph
        subgraph = ComputationalGraph(
            method=self.graph.method,
            metadata={**self.graph.metadata, "sliced": True, "sources": source_node_ids}
        )
        
        for node_id in visited:
            node = self.graph.get_node(node_id)
            if node:
                subgraph.add_node(node)
                
                # Add edges within subgraph
                for child_id in self.graph.get_children(node_id):
                    if child_id in visited:
                        subgraph.add_edge(node_id, child_id)
        
        return subgraph
    
    def slice_between(
        self,
        source_node_ids: List[str],
        target_node_ids: List[str],
    ) -> ComputationalGraph:
        """Slice to include only paths from sources to targets.
        
        Args:
            source_node_ids: List of source node IDs
            target_node_ids: List of target node IDs
            
        Returns:
            Sliced ComputationalGraph
        """
        # Backward slice from targets
        backward_visited = set()
        to_visit = list(target_node_ids)
        
        while to_visit:
            node_id = to_visit.pop()
            if node_id in backward_visited:
                continue
            backward_visited.add(node_id)
            
            parents = self.graph.get_parents(node_id)
            to_visit.extend(parents)
        
        # Forward slice from sources
        forward_visited = set()
        to_visit = list(source_node_ids)
        
        while to_visit:
            node_id = to_visit.pop()
            if node_id in forward_visited:
                continue
            forward_visited.add(node_id)
            
            children = self.graph.get_children(node_id)
            to_visit.extend(children)
        
        # Intersection is the path between
        path_nodes = backward_visited & forward_visited
        
        # Build subgraph
        subgraph = ComputationalGraph(
            method=self.graph.method,
            metadata={
                **self.graph.metadata,
                "sliced": True,
                "sources": source_node_ids,
                "targets": target_node_ids,
            }
        )
        
        for node_id in path_nodes:
            node = self.graph.get_node(node_id)
            if node:
                subgraph.add_node(node)
                
                # Add edges within subgraph
                for child_id in self.graph.get_children(node_id):
                    if child_id in path_nodes:
                        subgraph.add_edge(node_id, child_id)
        
        return subgraph


class CanonicalNodeMapper:
    """Maps between different node naming schemes to canonical IDs."""
    
    def __init__(self):
        """Initialize canonical node mapper."""
        self.name_to_canonical: dict[str, str] = {}
        self.canonical_to_name: dict[str, str] = {}
    
    def register_mapping(self, name: str, canonical_id: str) -> None:
        """Register a mapping between name and canonical ID.
        
        Args:
            name: Original node name
            canonical_id: Canonical node ID
        """
        self.name_to_canonical[name] = canonical_id
        self.canonical_to_name[canonical_id] = name
    
    def get_canonical(self, name: str) -> str:
        """Get canonical ID from name.
        
        Args:
            name: Original node name
            
        Returns:
            Canonical ID (or name if not found)
        """
        return self.name_to_canonical.get(name, name)
    
    def get_name(self, canonical_id: str) -> str:
        """Get original name from canonical ID.
        
        Args:
            canonical_id: Canonical node ID
            
        Returns:
            Original name (or canonical_id if not found)
        """
        return self.canonical_to_name.get(canonical_id, canonical_id)
    
    @classmethod
    def from_graph(cls, graph: ComputationalGraph) -> "CanonicalNodeMapper":
        """Create mapper from a computational graph.
        
        Args:
            graph: Computational graph
            
        Returns:
            CanonicalNodeMapper instance
        """
        mapper = cls()
        for node_id, node in graph.nodes.items():
            mapper.register_mapping(node.name, node_id)
        return mapper


def slice_graph(
    graph: ComputationalGraph,
    target_nodes: List[str],
    include_dependencies: bool = True,
) -> ComputationalGraph:
    """Convenience function to slice a graph.
    
    Args:
        graph: Full computational graph
        target_nodes: Target node IDs
        include_dependencies: Whether to include dependencies
        
    Returns:
        Sliced subgraph
    """
    slicer = SubgraphSlicer(graph)
    return slicer.slice_from_targets(target_nodes, include_dependencies)
