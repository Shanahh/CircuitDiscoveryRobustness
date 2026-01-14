"""Circuit discovery algorithm interface and registry."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import torch

from circuit_discovery.execution import ActivationCache, GradientCache
from circuit_discovery.graph_export import ComputationalGraph


@dataclass
class CircuitDiscoveryResult:
    """Result from a circuit discovery algorithm.
    
    Attributes:
        algorithm: Algorithm name
        important_nodes: List of important node IDs
        edge_scores: Dictionary of edge -> importance score
        node_scores: Dictionary of node_id -> importance score
        metadata: Additional algorithm-specific metadata
    """
    
    algorithm: str
    important_nodes: List[str] = field(default_factory=list)
    edge_scores: Dict[tuple[str, str], float] = field(default_factory=dict)
    node_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "important_nodes": self.important_nodes,
            "edge_scores": {f"{k[0]}->{k[1]}": v for k, v in self.edge_scores.items()},
            "node_scores": self.node_scores,
            "metadata": self.metadata,
        }


class CircuitDiscoveryAlgorithm(ABC):
    """Abstract base class for circuit discovery algorithms."""
    
    def __init__(self, **kwargs):
        """Initialize algorithm with configuration.
        
        Args:
            **kwargs: Algorithm-specific configuration
        """
        self.config = kwargs
    
    @abstractmethod
    def discover(
        self,
        graph: ComputationalGraph,
        activation_cache: ActivationCache,
        gradient_cache: Optional[GradientCache] = None,
        **kwargs,
    ) -> CircuitDiscoveryResult:
        """Run circuit discovery algorithm.
        
        Args:
            graph: Computational graph
            activation_cache: Cached activations
            gradient_cache: Cached gradients (optional)
            **kwargs: Additional algorithm-specific arguments
            
        Returns:
            CircuitDiscoveryResult
        """
        pass
    
    @classmethod
    def name(cls) -> str:
        """Get algorithm name."""
        return cls.__name__.lower().replace("algorithm", "")


class AttributionPatchingAlgorithm(CircuitDiscoveryAlgorithm):
    """Attribution patching algorithm for circuit discovery."""
    
    def discover(
        self,
        graph: ComputationalGraph,
        activation_cache: ActivationCache,
        gradient_cache: Optional[GradientCache] = None,
        threshold: float = 0.1,
        **kwargs,
    ) -> CircuitDiscoveryResult:
        """Discover circuits using attribution patching.
        
        Args:
            graph: Computational graph
            activation_cache: Cached activations
            gradient_cache: Cached gradients
            threshold: Threshold for importance
            **kwargs: Additional arguments
            
        Returns:
            CircuitDiscoveryResult
        """
        result = CircuitDiscoveryResult(algorithm="attribution_patching")
        
        # Compute node importance based on activation magnitudes
        for node_id, activations in activation_cache.activations.items():
            if len(activations) > 0:
                # Stack activations and compute mean absolute value
                stacked = torch.stack(activations)
                importance = stacked.abs().mean().item()
                result.node_scores[node_id] = importance
                
                if importance > threshold:
                    result.important_nodes.append(node_id)
        
        # Compute edge importance (simplified)
        for source_id, targets in graph.edges.items():
            source_score = result.node_scores.get(source_id, 0.0)
            for target_id in targets:
                target_score = result.node_scores.get(target_id, 0.0)
                edge_score = (source_score + target_score) / 2.0
                result.edge_scores[(source_id, target_id)] = edge_score
        
        result.metadata = {
            "threshold": threshold,
            "total_nodes": len(graph.nodes),
            "important_nodes": len(result.important_nodes),
        }
        
        return result
    
    @classmethod
    def name(cls) -> str:
        return "attribution_patching"


class ACDCAlgorithm(CircuitDiscoveryAlgorithm):
    """Automatic Circuit Discovery (ACDC) algorithm."""
    
    def discover(
        self,
        graph: ComputationalGraph,
        activation_cache: ActivationCache,
        gradient_cache: Optional[GradientCache] = None,
        threshold: float = 0.05,
        **kwargs,
    ) -> CircuitDiscoveryResult:
        """Discover circuits using ACDC.
        
        Args:
            graph: Computational graph
            activation_cache: Cached activations
            gradient_cache: Cached gradients
            threshold: Threshold for importance
            **kwargs: Additional arguments
            
        Returns:
            CircuitDiscoveryResult
        """
        result = CircuitDiscoveryResult(algorithm="acdc")
        
        # Simplified ACDC: use gradient information if available
        if gradient_cache and gradient_cache.gradients:
            for node_id, gradients in gradient_cache.gradients.items():
                if len(gradients) > 0:
                    stacked = torch.stack(gradients)
                    importance = stacked.abs().mean().item()
                    result.node_scores[node_id] = importance
                    
                    if importance > threshold:
                        result.important_nodes.append(node_id)
        else:
            # Fallback to activation-based importance
            for node_id, activations in activation_cache.activations.items():
                if len(activations) > 0:
                    stacked = torch.stack(activations)
                    importance = stacked.abs().mean().item()
                    result.node_scores[node_id] = importance
                    
                    if importance > threshold:
                        result.important_nodes.append(node_id)
        
        # Compute edge scores
        for source_id, targets in graph.edges.items():
            source_score = result.node_scores.get(source_id, 0.0)
            for target_id in targets:
                target_score = result.node_scores.get(target_id, 0.0)
                edge_score = (source_score * target_score) ** 0.5  # Geometric mean
                result.edge_scores[(source_id, target_id)] = edge_score
        
        result.metadata = {
            "threshold": threshold,
            "total_nodes": len(graph.nodes),
            "important_nodes": len(result.important_nodes),
            "used_gradients": gradient_cache is not None and len(gradient_cache.gradients) > 0,
        }
        
        return result
    
    @classmethod
    def name(cls) -> str:
        return "acdc"


class EAPAlgorithm(CircuitDiscoveryAlgorithm):
    """Edge Attribution Patching (EAP) algorithm."""
    
    def discover(
        self,
        graph: ComputationalGraph,
        activation_cache: ActivationCache,
        gradient_cache: Optional[GradientCache] = None,
        threshold: float = 0.1,
        **kwargs,
    ) -> CircuitDiscoveryResult:
        """Discover circuits using EAP.
        
        Args:
            graph: Computational graph
            activation_cache: Cached activations
            gradient_cache: Cached gradients
            threshold: Threshold for importance
            **kwargs: Additional arguments
            
        Returns:
            CircuitDiscoveryResult
        """
        result = CircuitDiscoveryResult(algorithm="eap")
        
        # Edge-focused approach
        for source_id, targets in graph.edges.items():
            source_acts = activation_cache.get(source_id)
            
            for target_id in targets:
                target_acts = activation_cache.get(target_id)
                
                # Compute correlation-based edge importance
                if source_acts and target_acts and len(source_acts) > 0 and len(target_acts) > 0:
                    source_tensor = torch.stack(source_acts).flatten(1).mean(1)
                    target_tensor = torch.stack(target_acts).flatten(1).mean(1)
                    
                    # Compute correlation
                    if len(source_tensor) == len(target_tensor) and len(source_tensor) > 1:
                        correlation = torch.corrcoef(
                            torch.stack([source_tensor, target_tensor])
                        )[0, 1].abs().item()
                    else:
                        correlation = 0.0
                    
                    result.edge_scores[(source_id, target_id)] = correlation
        
        # Aggregate to node scores
        for node_id in graph.nodes.keys():
            # Incoming edge scores
            incoming = [
                score for (src, tgt), score in result.edge_scores.items()
                if tgt == node_id
            ]
            # Outgoing edge scores
            outgoing = [
                score for (src, tgt), score in result.edge_scores.items()
                if src == node_id
            ]
            
            all_scores = incoming + outgoing
            if all_scores:
                result.node_scores[node_id] = sum(all_scores) / len(all_scores)
                
                if result.node_scores[node_id] > threshold:
                    result.important_nodes.append(node_id)
        
        result.metadata = {
            "threshold": threshold,
            "total_edges": len(result.edge_scores),
            "important_nodes": len(result.important_nodes),
        }
        
        return result
    
    @classmethod
    def name(cls) -> str:
        return "eap"


class AlgorithmRegistry:
    """Registry for circuit discovery algorithms."""
    
    _algorithms: Dict[str, Type[CircuitDiscoveryAlgorithm]] = {}
    
    @classmethod
    def register(cls, algorithm_class: Type[CircuitDiscoveryAlgorithm]) -> None:
        """Register an algorithm.
        
        Args:
            algorithm_class: Algorithm class to register
        """
        name = algorithm_class.name()
        cls._algorithms[name] = algorithm_class
    
    @classmethod
    def get(cls, name: str) -> Type[CircuitDiscoveryAlgorithm]:
        """Get an algorithm by name.
        
        Args:
            name: Algorithm name
            
        Returns:
            Algorithm class
        """
        if name not in cls._algorithms:
            raise ValueError(f"Unknown algorithm: {name}. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[name]
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """List all registered algorithms.
        
        Returns:
            List of algorithm names
        """
        return list(cls._algorithms.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> CircuitDiscoveryAlgorithm:
        """Create an algorithm instance.
        
        Args:
            name: Algorithm name
            **kwargs: Algorithm configuration
            
        Returns:
            Algorithm instance
        """
        algorithm_class = cls.get(name)
        return algorithm_class(**kwargs)


# Register built-in algorithms
AlgorithmRegistry.register(AttributionPatchingAlgorithm)
AlgorithmRegistry.register(ACDCAlgorithm)
AlgorithmRegistry.register(EAPAlgorithm)
