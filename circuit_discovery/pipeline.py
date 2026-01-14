"""Main pipeline orchestration for circuit discovery benchmarks."""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from circuit_discovery.algorithms import AlgorithmRegistry, CircuitDiscoveryResult
from circuit_discovery.execution import BatchExecutor, ActivationCache, GradientCache
from circuit_discovery.graph_export import GraphExportMethod, export_graph, ComputationalGraph
from circuit_discovery.models import load_model_from_spec, ModelSnapshot
from circuit_discovery.spec import BenchmarkSpec
from circuit_discovery.storage import ResultStorage
from circuit_discovery.subgraph import slice_graph


class BenchmarkPipeline:
    """Orchestrates the full circuit discovery benchmark pipeline."""
    
    def __init__(
        self,
        spec: BenchmarkSpec,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize benchmark pipeline.
        
        Args:
            spec: Benchmark specification
            device: Device to run on (auto-detect if None)
            verbose: Whether to print progress messages
        """
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # Pipeline components
        self.model: Optional[nn.Module] = None
        self.snapshot: Optional[ModelSnapshot] = None
        self.graph: Optional[ComputationalGraph] = None
        self.subgraph: Optional[ComputationalGraph] = None
        self.activation_cache: Optional[ActivationCache] = None
        self.gradient_cache: Optional[GradientCache] = None
        self.result: Optional[CircuitDiscoveryResult] = None
        
        # Storage
        self.storage = ResultStorage(spec.output_dir)
        self.run_id = self._generate_run_id()
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID.
        
        Returns:
            Run ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.spec.task}_{self.spec.algorithm}_{self.spec.seed}_{timestamp}"
    
    def _log(self, message: str) -> None:
        """Log a message if verbose.
        
        Args:
            message: Message to log
        """
        if self.verbose:
            print(f"[Pipeline] {message}")
    
    def load_model(self) -> None:
        """Load model and download snapshot with integrity check."""
        self._log(f"Loading model: {self.spec.model_id}")
        
        # Download pinned snapshot
        self.model, self.snapshot = load_model_from_spec(self.spec, device=self.device)
        
        self._log(f"Model loaded from: {self.snapshot.local_path}")
        self._log(f"Commit hash: {self.snapshot.commit_hash}")
        self._log(f"Integrity checksum: {self.snapshot.checksum}")
    
    def export_graph(self, method: Optional[GraphExportMethod] = None) -> None:
        """Export computational graph.
        
        Args:
            method: Export method (auto-select if None)
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded first")
        
        self._log(f"Exporting computational graph...")
        
        # Create example inputs for tracing
        example_inputs = torch.randn(
            self.spec.batch_size,
            self.spec.sequence_length,
            device=self.device,
        )
        
        # Export graph
        self.graph = export_graph(self.model, example_inputs, method=method)
        
        self._log(f"Graph exported using {self.graph.method.value}")
        self._log(f"Total nodes: {len(self.graph.nodes)}")
        self._log(f"Total edges: {sum(len(v) for v in self.graph.edges.values())}")
    
    def slice_subgraph(self) -> None:
        """Slice subgraph from target nodes."""
        if self.graph is None:
            raise RuntimeError("Graph must be exported first")
        
        if not self.spec.target_nodes:
            self._log("No target nodes specified, using full graph")
            self.subgraph = self.graph
            return
        
        self._log(f"Slicing subgraph from {len(self.spec.target_nodes)} targets")
        
        # Resolve target node IDs (handle both names and canonical IDs)
        target_ids = []
        for target in self.spec.target_nodes:
            # Check if it's a canonical ID
            if target in self.graph.nodes:
                target_ids.append(target)
            else:
                # Try to find by name
                for node_id, node in self.graph.nodes.items():
                    if node.name == target:
                        target_ids.append(node_id)
                        break
        
        if not target_ids:
            self._log("Warning: No valid target nodes found, using full graph")
            self.subgraph = self.graph
            return
        
        # Slice subgraph
        self.subgraph = slice_graph(self.graph, target_ids, include_dependencies=True)
        
        self._log(f"Subgraph nodes: {len(self.subgraph.nodes)}")
        self._log(f"Subgraph edges: {sum(len(v) for v in self.subgraph.edges.values())}")
    
    def run_batches(self, data_loader: Any, num_batches: Optional[int] = None) -> None:
        """Run batches with activation and gradient caching.
        
        Args:
            data_loader: DataLoader providing batches
            num_batches: Maximum number of batches (None for all)
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded first")
        
        self._log(f"Running batches (cache_activations={self.spec.cache_activations}, "
                  f"cache_gradients={self.spec.cache_gradients})")
        
        # Create executor
        executor = BatchExecutor(
            model=self.model,
            graph=self.subgraph or self.graph,
            cache_activations=self.spec.cache_activations,
            cache_gradients=self.spec.cache_gradients,
            device=self.device,
        )
        
        # Run batches
        self.activation_cache, self.gradient_cache = executor.run_batches(
            data_loader=data_loader,
            num_batches=num_batches,
            show_progress=self.verbose,
        )
        
        if self.spec.cache_activations:
            self._log(f"Cached activations for {len(self.activation_cache.activations)} nodes")
        if self.spec.cache_gradients:
            self._log(f"Cached gradients for {len(self.gradient_cache.gradients)} nodes")
    
    def run_algorithm(self, **algo_kwargs) -> None:
        """Run circuit discovery algorithm.
        
        Args:
            **algo_kwargs: Algorithm-specific arguments
        """
        if self.activation_cache is None:
            raise RuntimeError("Must run batches first")
        
        graph = self.subgraph or self.graph
        if graph is None:
            raise RuntimeError("Graph must be exported first")
        
        self._log(f"Running algorithm: {self.spec.algorithm}")
        
        # Create algorithm instance
        algorithm = AlgorithmRegistry.create(self.spec.algorithm, **algo_kwargs)
        
        # Run discovery
        self.result = algorithm.discover(
            graph=graph,
            activation_cache=self.activation_cache,
            gradient_cache=self.gradient_cache if self.spec.cache_gradients else None,
        )
        
        self._log(f"Discovery complete: {len(self.result.important_nodes)} important nodes found")
    
    def save_results(self, save_caches: bool = False) -> None:
        """Save results and metadata.
        
        Args:
            save_caches: Whether to save activation/gradient caches
        """
        if self.result is None:
            raise RuntimeError("Must run algorithm first")
        
        self._log(f"Saving results to: {self.spec.output_dir / self.run_id}")
        
        # Save specification
        self.storage.save_spec(self.run_id, self.spec)
        
        # Save result with metadata
        metadata = {
            "run_id": self.run_id,
            "device": self.device,
            "pytorch_version": torch.__version__,
            "model_commit": self.snapshot.commit_hash if self.snapshot else None,
            "model_checksum": self.snapshot.checksum if self.snapshot else None,
            "graph_method": self.graph.method.value if self.graph else None,
            "graph_nodes": len(self.graph.nodes) if self.graph else None,
            "subgraph_nodes": len(self.subgraph.nodes) if self.subgraph else None,
        }
        self.storage.save_result(self.run_id, self.result, metadata=metadata)
        
        # Optionally save caches
        if save_caches:
            if self.activation_cache:
                self.storage.save_activations(self.run_id, self.activation_cache)
            if self.gradient_cache and self.spec.cache_gradients:
                self.storage.save_gradients(self.run_id, self.gradient_cache)
        
        self._log(f"Results saved successfully")
    
    def run(
        self,
        data_loader: Any,
        num_batches: Optional[int] = None,
        save_caches: bool = False,
        graph_method: Optional[GraphExportMethod] = None,
        **algo_kwargs,
    ) -> CircuitDiscoveryResult:
        """Run the complete pipeline.
        
        Args:
            data_loader: DataLoader providing batches
            num_batches: Maximum number of batches
            save_caches: Whether to save caches
            graph_method: Graph export method
            **algo_kwargs: Algorithm-specific arguments
            
        Returns:
            CircuitDiscoveryResult
        """
        # Set random seed
        torch.manual_seed(self.spec.seed)
        
        # Run pipeline stages
        self.load_model()
        self.export_graph(method=graph_method)
        self.slice_subgraph()
        self.run_batches(data_loader, num_batches=num_batches)
        self.run_algorithm(**algo_kwargs)
        self.save_results(save_caches=save_caches)
        
        return self.result


def run_benchmark(
    spec: BenchmarkSpec,
    data_loader: Any,
    num_batches: Optional[int] = None,
    save_caches: bool = False,
    device: Optional[str] = None,
    verbose: bool = True,
    **algo_kwargs,
) -> CircuitDiscoveryResult:
    """Convenience function to run a complete benchmark.
    
    Args:
        spec: Benchmark specification
        data_loader: DataLoader providing batches
        num_batches: Maximum number of batches
        save_caches: Whether to save caches
        device: Device to run on
        verbose: Whether to print progress
        **algo_kwargs: Algorithm-specific arguments
        
    Returns:
        CircuitDiscoveryResult
    """
    pipeline = BenchmarkPipeline(spec, device=device, verbose=verbose)
    return pipeline.run(data_loader, num_batches=num_batches, save_caches=save_caches, **algo_kwargs)
