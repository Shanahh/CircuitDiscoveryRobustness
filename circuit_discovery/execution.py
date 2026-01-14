"""Batch execution with activation and gradient caching."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from circuit_discovery.graph_export import ComputationalGraph


@dataclass
class ActivationCache:
    """Cache for storing activations during forward passes.
    
    Attributes:
        activations: Dictionary of node_id -> list of activation tensors
        metadata: Additional metadata about cached activations
    """
    
    activations: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def store(self, node_id: str, activation: torch.Tensor) -> None:
        """Store an activation tensor.
        
        Args:
            node_id: Node identifier
            activation: Activation tensor
        """
        if node_id not in self.activations:
            self.activations[node_id] = []
        # Detach and clone to avoid memory issues
        self.activations[node_id].append(activation.detach().clone())
    
    def get(self, node_id: str) -> Optional[List[torch.Tensor]]:
        """Get stored activations for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of activation tensors or None
        """
        return self.activations.get(node_id)
    
    def clear(self) -> None:
        """Clear all cached activations."""
        self.activations.clear()
    
    def get_stacked(self, node_id: str) -> Optional[torch.Tensor]:
        """Get activations stacked into a single tensor.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Stacked tensor or None
        """
        activations = self.get(node_id)
        if activations is None or len(activations) == 0:
            return None
        return torch.stack(activations)


@dataclass
class GradientCache:
    """Cache for storing gradients during backward passes.
    
    Attributes:
        gradients: Dictionary of node_id -> list of gradient tensors
        metadata: Additional metadata about cached gradients
    """
    
    gradients: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def store(self, node_id: str, gradient: torch.Tensor) -> None:
        """Store a gradient tensor.
        
        Args:
            node_id: Node identifier
            gradient: Gradient tensor
        """
        if node_id not in self.gradients:
            self.gradients[node_id] = []
        # Detach and clone to avoid memory issues
        self.gradients[node_id].append(gradient.detach().clone())
    
    def get(self, node_id: str) -> Optional[List[torch.Tensor]]:
        """Get stored gradients for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of gradient tensors or None
        """
        return self.gradients.get(node_id)
    
    def clear(self) -> None:
        """Clear all cached gradients."""
        self.gradients.clear()
    
    def get_stacked(self, node_id: str) -> Optional[torch.Tensor]:
        """Get gradients stacked into a single tensor.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Stacked tensor or None
        """
        gradients = self.get(node_id)
        if gradients is None or len(gradients) == 0:
            return None
        return torch.stack(gradients)


class BatchExecutor:
    """Executes batches through a model with caching."""
    
    def __init__(
        self,
        model: nn.Module,
        graph: Optional[ComputationalGraph] = None,
        cache_activations: bool = True,
        cache_gradients: bool = False,
        device: str = "cpu",
    ):
        """Initialize batch executor.
        
        Args:
            model: PyTorch model
            graph: Computational graph (optional, for selective caching)
            cache_activations: Whether to cache activations
            cache_gradients: Whether to cache gradients
            device: Device to run on
        """
        self.model = model.to(device)
        self.graph = graph
        self.cache_activations = cache_activations
        self.cache_gradients = cache_gradients
        self.device = device
        
        self.activation_cache = ActivationCache()
        self.gradient_cache = GradientCache()
        
        self._hooks = []
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks for caching."""
        self._clear_hooks()
        
        for name, module in self.model.named_modules():
            if not name:  # Skip root
                continue
            
            # Forward hook for activations
            if self.cache_activations:
                def make_forward_hook(node_name):
                    def hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            self.activation_cache.store(node_name, output)
                    return hook
                
                handle = module.register_forward_hook(make_forward_hook(name))
                self._hooks.append(handle)
            
            # Backward hook for gradients
            if self.cache_gradients:
                def make_backward_hook(node_name):
                    def hook(module, grad_input, grad_output):
                        if grad_output[0] is not None:
                            self.gradient_cache.store(node_name, grad_output[0])
                    return hook
                
                handle = module.register_full_backward_hook(make_backward_hook(name))
                self._hooks.append(handle)
    
    def _clear_hooks(self) -> None:
        """Clear all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
    
    def run_batches(
        self,
        data_loader: Any,
        num_batches: Optional[int] = None,
        loss_fn: Optional[Callable] = None,
        show_progress: bool = True,
    ) -> Tuple[ActivationCache, GradientCache]:
        """Run batches through the model.
        
        Args:
            data_loader: DataLoader or iterable providing batches
            num_batches: Maximum number of batches (None for all)
            loss_fn: Loss function for gradient computation
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (activation_cache, gradient_cache)
        """
        # Register hooks
        self._register_hooks()
        
        # Clear previous caches
        self.activation_cache.clear()
        self.gradient_cache.clear()
        
        self.model.train() if self.cache_gradients else self.model.eval()
        
        # Iterate through batches
        iterator = enumerate(data_loader)
        if show_progress:
            total = num_batches if num_batches else len(data_loader) if hasattr(data_loader, '__len__') else None
            iterator = tqdm(iterator, total=total, desc="Processing batches")
        
        for batch_idx, batch in iterator:
            if num_batches and batch_idx >= num_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                inputs = batch[0].to(self.device)
                targets = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                inputs = batch.to(self.device)
                targets = None
            
            # Forward pass
            if self.cache_gradients and loss_fn is not None:
                outputs = self.model(inputs)
                
                if targets is not None:
                    loss = loss_fn(outputs, targets)
                else:
                    # Dummy loss if no targets
                    loss = outputs.sum()
                
                # Backward pass
                loss.backward()
                
                # Clear gradients
                self.model.zero_grad()
            else:
                with torch.no_grad():
                    outputs = self.model(inputs)
        
        # Clear hooks
        self._clear_hooks()
        
        return self.activation_cache, self.gradient_cache
    
    def run_single(
        self,
        inputs: torch.Tensor,
        compute_gradients: bool = False,
        loss_fn: Optional[Callable] = None,
    ) -> Tuple[torch.Tensor, ActivationCache, GradientCache]:
        """Run a single batch.
        
        Args:
            inputs: Input tensor
            compute_gradients: Whether to compute gradients
            loss_fn: Loss function
            
        Returns:
            Tuple of (outputs, activation_cache, gradient_cache)
        """
        # Register hooks
        self._register_hooks()
        
        # Clear previous caches
        self.activation_cache.clear()
        self.gradient_cache.clear()
        
        inputs = inputs.to(self.device)
        
        if compute_gradients and loss_fn is not None:
            self.model.train()
            outputs = self.model(inputs)
            loss = loss_fn(outputs)
            loss.backward()
            self.model.zero_grad()
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(inputs)
        
        # Clear hooks
        self._clear_hooks()
        
        return outputs, self.activation_cache, self.gradient_cache


def create_executor(
    model: nn.Module,
    graph: Optional[ComputationalGraph] = None,
    cache_activations: bool = True,
    cache_gradients: bool = False,
    device: str = "cpu",
) -> BatchExecutor:
    """Convenience function to create a batch executor.
    
    Args:
        model: PyTorch model
        graph: Computational graph
        cache_activations: Whether to cache activations
        cache_gradients: Whether to cache gradients
        device: Device to run on
        
    Returns:
        BatchExecutor instance
    """
    return BatchExecutor(
        model=model,
        graph=graph,
        cache_activations=cache_activations,
        cache_gradients=cache_gradients,
        device=device,
    )
