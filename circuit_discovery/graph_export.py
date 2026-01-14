"""Graph extraction using FX, torch.export, or hooks."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule


class GraphExportMethod(Enum):
    """Methods for exporting computational graphs."""
    
    FX = "fx"
    TORCH_EXPORT = "torch_export"
    HOOKS = "hooks"


@dataclass
class GraphNode:
    """Representation of a graph node with canonical ID.
    
    Attributes:
        node_id: Canonical node identifier
        name: Human-readable name
        op: Operation type
        target: Target function/module
        args: Node arguments
        metadata: Additional metadata
    """
    
    node_id: str
    name: str
    op: str
    target: Any
    args: tuple = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id


@dataclass
class ComputationalGraph:
    """Computational graph representation.
    
    Attributes:
        nodes: Dictionary of node_id -> GraphNode
        edges: Dictionary of source_id -> list of target_ids
        method: Export method used
        metadata: Graph-level metadata
    """
    
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)
    method: GraphExportMethod = GraphExportMethod.HOOKS
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = []
    
    def add_edge(self, source_id: str, target_id: str) -> None:
        """Add an edge to the graph."""
        if source_id not in self.edges:
            self.edges[source_id] = []
        self.edges[source_id].append(target_id)
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[str]:
        """Get children of a node."""
        return self.edges.get(node_id, [])
    
    def get_parents(self, node_id: str) -> List[str]:
        """Get parents of a node."""
        parents = []
        for parent_id, children in self.edges.items():
            if node_id in children:
                parents.append(parent_id)
        return parents


class GraphExporter:
    """Exports computational graphs from PyTorch models."""
    
    def __init__(self, method: Optional[GraphExportMethod] = None):
        """Initialize graph exporter.
        
        Args:
            method: Preferred export method (will try in order if None)
        """
        self.method = method
    
    def export(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> ComputationalGraph:
        """Export computational graph from model.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs for tracing
            
        Returns:
            ComputationalGraph representation
        """
        if self.method == GraphExportMethod.FX:
            return self._export_fx(model, example_inputs)
        elif self.method == GraphExportMethod.TORCH_EXPORT:
            return self._export_torch_export(model, example_inputs)
        elif self.method == GraphExportMethod.HOOKS:
            return self._export_hooks(model, example_inputs)
        else:
            # Try methods in order
            try:
                return self._export_fx(model, example_inputs)
            except Exception:
                try:
                    return self._export_torch_export(model, example_inputs)
                except Exception:
                    return self._export_hooks(model, example_inputs)
    
    def _export_fx(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> ComputationalGraph:
        """Export using torch.fx symbolic tracing.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            
        Returns:
            ComputationalGraph
        """
        # Symbolic trace the model
        traced = symbolic_trace(model)
        
        graph = ComputationalGraph(method=GraphExportMethod.FX)
        
        # Convert FX nodes to canonical representation
        for idx, node in enumerate(traced.graph.nodes):
            node_id = f"fx_{idx}_{node.name}"
            graph_node = GraphNode(
                node_id=node_id,
                name=node.name,
                op=node.op,
                target=node.target,
                args=tuple(arg.name if hasattr(arg, 'name') else str(arg) for arg in node.args),
                metadata={
                    'fx_node': str(node),
                    'type': str(type(node.target)),
                }
            )
            graph.add_node(graph_node)
            
            # Add edges based on args
            for arg in node.args:
                if hasattr(arg, 'name'):
                    # Find parent node
                    for parent_idx, parent_node in enumerate(traced.graph.nodes):
                        if parent_node.name == arg.name:
                            parent_id = f"fx_{parent_idx}_{parent_node.name}"
                            graph.add_edge(parent_id, node_id)
                            break
        
        return graph
    
    def _export_torch_export(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> ComputationalGraph:
        """Export using torch.export (PyTorch 2.x).
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs (required)
            
        Returns:
            ComputationalGraph
        """
        if example_inputs is None:
            raise ValueError("example_inputs required for torch.export")
        
        # Try to use torch.export if available (PyTorch 2.1+)
        try:
            from torch.export import export
            exported = export(model, (example_inputs,))
            
            graph = ComputationalGraph(method=GraphExportMethod.TORCH_EXPORT)
            
            # Convert exported nodes to canonical representation
            for idx, node in enumerate(exported.graph.nodes):
                node_id = f"export_{idx}_{node.name}"
                graph_node = GraphNode(
                    node_id=node_id,
                    name=node.name,
                    op=node.op,
                    target=node.target,
                    metadata={'export_node': str(node)}
                )
                graph.add_node(graph_node)
            
            return graph
            
        except ImportError:
            raise RuntimeError("torch.export not available in this PyTorch version")
    
    def _export_hooks(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None,
    ) -> ComputationalGraph:
        """Export using forward hooks to capture computation.
        
        Args:
            model: PyTorch model
            example_inputs: Example inputs
            
        Returns:
            ComputationalGraph
        """
        graph = ComputationalGraph(method=GraphExportMethod.HOOKS)
        execution_order = []
        
        def make_hook(name: str):
            def hook(module, input, output):
                execution_order.append((name, module, input, output))
            return hook
        
        # Register hooks on all modules
        handles = []
        for name, module in model.named_modules():
            if name:  # Skip root module
                handle = module.register_forward_hook(make_hook(name))
                handles.append(handle)
        
        # Run forward pass to capture execution
        if example_inputs is not None:
            with torch.no_grad():
                model(example_inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Build graph from execution order
        prev_id = None
        for idx, (name, module, input, output) in enumerate(execution_order):
            node_id = f"hook_{idx}_{name}"
            graph_node = GraphNode(
                node_id=node_id,
                name=name,
                op="module",
                target=type(module).__name__,
                metadata={
                    'module_type': str(type(module)),
                    'execution_order': idx,
                }
            )
            graph.add_node(graph_node)
            
            # Create sequential edges
            if prev_id is not None:
                graph.add_edge(prev_id, node_id)
            prev_id = node_id
        
        return graph


def export_graph(
    model: nn.Module,
    example_inputs: Optional[torch.Tensor] = None,
    method: Optional[GraphExportMethod] = None,
) -> ComputationalGraph:
    """Convenience function to export computational graph.
    
    Args:
        model: PyTorch model
        example_inputs: Example inputs for tracing
        method: Preferred export method
        
    Returns:
        ComputationalGraph
    """
    exporter = GraphExporter(method=method)
    return exporter.export(model, example_inputs)
