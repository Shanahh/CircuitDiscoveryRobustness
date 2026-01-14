# Usage Guide

This guide provides detailed examples of using the Circuit Discovery Benchmark Runner.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Python API](#python-api)
3. [CLI Usage](#cli-usage)
4. [Custom Algorithms](#custom-algorithms)
5. [Advanced Features](#advanced-features)

## Quick Start

### 1. Install the Package

```bash
git clone https://github.com/Shanahh/CircuitDiscoveryRobustness.git
cd CircuitDiscoveryRobustness
pip install -e .
```

### 2. Run a Simple Benchmark

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from circuit_discovery import BenchmarkSpec, run_benchmark

# Create specification
spec = BenchmarkSpec(
    model_id="gpt2",
    task="ioi",
    algorithm="attribution_patching",
    seed=42,
    batch_size=8,
)

# Create data loader (replace with your task data)
data = torch.randn(100, 128)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=8)

# Run benchmark
result = run_benchmark(spec, data_loader, num_batches=10)

print(f"Found {len(result.important_nodes)} important nodes")
```

## Python API

### Creating a Specification

```python
from circuit_discovery.spec import BenchmarkSpec
from pathlib import Path

# From Python
spec = BenchmarkSpec(
    model_id="gpt2",
    model_commit="main",  # Git commit or tag
    task="ioi",
    algorithm="acdc",
    seed=42,
    batch_size=16,
    sequence_length=256,
    target_nodes=["transformer.h.4", "transformer.h.5"],
    cache_activations=True,
    cache_gradients=True,
    output_dir=Path("results"),
)

# Save to YAML
spec.to_yaml(Path("my_spec.yaml"))

# Load from YAML
spec = BenchmarkSpec.from_yaml(Path("my_spec.yaml"))
```

### Using the Pipeline

```python
from circuit_discovery.pipeline import BenchmarkPipeline

pipeline = BenchmarkPipeline(spec, device="cuda", verbose=True)

# Run individual stages
pipeline.load_model()
pipeline.export_graph()
pipeline.slice_subgraph()
pipeline.run_batches(data_loader, num_batches=50)
pipeline.run_algorithm(threshold=0.05)
pipeline.save_results(save_caches=True)

# Access results
print(f"Run ID: {pipeline.run_id}")
print(f"Important nodes: {pipeline.result.important_nodes}")
print(f"Node scores: {pipeline.result.node_scores}")
```

### Working with Graphs

```python
from circuit_discovery.graph_export import export_graph, GraphExportMethod
from circuit_discovery.subgraph import slice_graph
import torch.nn as nn

# Export graph
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
)
graph = export_graph(model, example_inputs=torch.randn(4, 10))

# Explore graph structure
print(f"Total nodes: {len(graph.nodes)}")
print(f"Total edges: {sum(len(v) for v in graph.edges.values())}")

# Slice subgraph
subgraph = slice_graph(graph, target_nodes=["2"], include_dependencies=True)
print(f"Subgraph nodes: {len(subgraph.nodes)}")
```

### Caching Activations and Gradients

```python
from circuit_discovery.execution import BatchExecutor
from torch.utils.data import DataLoader

executor = BatchExecutor(
    model=model,
    graph=graph,
    cache_activations=True,
    cache_gradients=True,
    device="cuda",
)

# Run batches
act_cache, grad_cache = executor.run_batches(
    data_loader=data_loader,
    num_batches=100,
    loss_fn=torch.nn.functional.cross_entropy,
    show_progress=True,
)

# Access cached data
activations = act_cache.get("layer_name")
gradients = grad_cache.get("layer_name")

# Get stacked tensors
stacked_acts = act_cache.get_stacked("layer_name")
```

### Running Algorithms

```python
from circuit_discovery.algorithms import AlgorithmRegistry

# List available algorithms
algorithms = AlgorithmRegistry.list_algorithms()
print(algorithms)  # ['attribution_patching', 'acdc', 'eap']

# Create and run algorithm
algo = AlgorithmRegistry.create("attribution_patching", threshold=0.1)
result = algo.discover(
    graph=graph,
    activation_cache=act_cache,
    gradient_cache=grad_cache,
)

# Examine results
print(f"Algorithm: {result.algorithm}")
print(f"Important nodes: {result.important_nodes}")
print(f"Node scores: {result.node_scores}")
print(f"Edge scores: {result.edge_scores}")
```

### Storage and Retrieval

```python
from circuit_discovery.storage import ResultStorage
from pathlib import Path

storage = ResultStorage(Path("results"))

# Save results
storage.save_spec(run_id, spec)
storage.save_result(run_id, result, metadata={"note": "experiment 1"})
storage.save_activations(run_id, act_cache)
storage.save_gradients(run_id, grad_cache)

# Load results
loaded_spec = storage.load_spec(run_id)
loaded_result = storage.load_result(run_id)
loaded_acts = storage.load_activations(run_id)

# List runs
runs = storage.list_runs()
for run_id in runs:
    info = storage.get_run_info(run_id)
    print(f"{run_id}: {info}")
```

## CLI Usage

### Creating Specifications

```bash
# Interactive creation
circuit-bench create-spec \
  --model-id gpt2 \
  --task ioi \
  --algorithm attribution_patching \
  --batch-size 16 \
  --sequence-length 256 \
  --seed 42 \
  --output my_spec.yaml

# Edit the YAML file as needed
vim my_spec.yaml
```

### Running Benchmarks

```bash
# Basic run
circuit-bench run --spec my_spec.yaml

# With options
circuit-bench run \
  --spec my_spec.yaml \
  --num-batches 100 \
  --save-caches \
  --device cuda \
  --verbose
```

### Viewing Results

```bash
# List all runs
circuit-bench list-runs --results-dir results

# Show specific run
circuit-bench show-result RUN_ID \
  --results-dir results \
  --show-nodes 20

# Compare runs
circuit-bench compare-runs \
  --results-dir results \
  --format table

# JSON output for programmatic access
circuit-bench compare-runs \
  --results-dir results \
  --format json > comparison.json
```

### Algorithm Discovery

```bash
# List available algorithms
circuit-bench list-algorithms
```

## Custom Algorithms

### Creating a Custom Algorithm

```python
from circuit_discovery.algorithms import CircuitDiscoveryAlgorithm, CircuitDiscoveryResult, AlgorithmRegistry
import torch

class MyCustomAlgorithm(CircuitDiscoveryAlgorithm):
    """My custom circuit discovery algorithm."""
    
    def discover(self, graph, activation_cache, gradient_cache=None, **kwargs):
        """Discover circuits using custom logic."""
        result = CircuitDiscoveryResult(algorithm="my_custom")
        
        # Your custom logic here
        for node_id, activations in activation_cache.activations.items():
            if activations:
                # Compute custom importance metric
                stacked = torch.stack(activations)
                importance = stacked.std().item()  # Example: use std dev
                
                result.node_scores[node_id] = importance
                
                # Threshold for important nodes
                if importance > kwargs.get('threshold', 0.1):
                    result.important_nodes.append(node_id)
        
        # Compute edge scores
        for source_id, targets in graph.edges.items():
            for target_id in targets:
                source_score = result.node_scores.get(source_id, 0.0)
                target_score = result.node_scores.get(target_id, 0.0)
                result.edge_scores[(source_id, target_id)] = min(source_score, target_score)
        
        result.metadata = {
            'threshold': kwargs.get('threshold', 0.1),
            'total_nodes': len(graph.nodes),
        }
        
        return result
    
    @classmethod
    def name(cls):
        return "my_custom"

# Register your algorithm
AlgorithmRegistry.register(MyCustomAlgorithm)

# Use it
spec = BenchmarkSpec(
    model_id="gpt2",
    task="test",
    algorithm="my_custom",  # Your algorithm name
)
```

## Advanced Features

### Graph Export Methods

```python
from circuit_discovery.graph_export import GraphExportMethod, export_graph

# Try FX first (fastest, most detailed)
graph = export_graph(model, inputs, method=GraphExportMethod.FX)

# Try torch.export (PyTorch 2.x, good compatibility)
graph = export_graph(model, inputs, method=GraphExportMethod.TORCH_EXPORT)

# Use hooks (fallback, works with any model)
graph = export_graph(model, inputs, method=GraphExportMethod.HOOKS)

# Auto-select (tries in order)
graph = export_graph(model, inputs, method=None)
```

### Target Node Selection

```python
# Specify target nodes by name
spec = BenchmarkSpec(
    model_id="gpt2",
    task="ioi",
    algorithm="acdc",
    target_nodes=[
        "transformer.h.0.attn",
        "transformer.h.1.attn",
        "transformer.h.2.attn",
    ],
)

# Or use canonical IDs from graph
graph = export_graph(model, inputs)
target_ids = [node_id for node_id in graph.nodes.keys() if "attn" in node_id]
```

### Canonical Node Mapping

```python
from circuit_discovery.subgraph import CanonicalNodeMapper

# Create mapper from graph
mapper = CanonicalNodeMapper.from_graph(graph)

# Convert between names and canonical IDs
canonical_id = mapper.get_canonical("transformer.h.0.attn")
name = mapper.get_name(canonical_id)
```

### Batch Processing

```python
from circuit_discovery.execution import BatchExecutor

executor = BatchExecutor(
    model=model,
    cache_activations=True,
    cache_gradients=True,
    device="cuda",
)

# Process in batches with custom loss
def custom_loss(outputs, targets):
    return (outputs - targets).pow(2).mean()

act_cache, grad_cache = executor.run_batches(
    data_loader=data_loader,
    num_batches=None,  # Process all batches
    loss_fn=custom_loss,
    show_progress=True,
)
```

### Subgraph Slicing Strategies

```python
from circuit_discovery.subgraph import SubgraphSlicer

slicer = SubgraphSlicer(graph)

# Backward slice (dependencies of targets)
subgraph = slicer.slice_from_targets(
    target_node_ids=["output_layer"],
    include_dependencies=True,
)

# Forward slice (what depends on sources)
subgraph = slicer.slice_forward(
    source_node_ids=["input_layer"],
)

# Slice between source and target
subgraph = slicer.slice_between(
    source_node_ids=["layer_0"],
    target_node_ids=["layer_5"],
)
```

## Tips and Best Practices

1. **Start Small**: Test with small models and datasets first
2. **Use Caching Wisely**: Activation caching uses memory; monitor usage
3. **Reproducibility**: Always set a seed and use pinned model commits
4. **Device Management**: Use `device="cuda"` for large models
5. **Incremental Development**: Test each pipeline stage independently
6. **Version Control**: Track spec files in git for experiment tracking
7. **Storage Management**: Clean up old runs periodically to save space

## Troubleshooting

### Graph Export Fails

If FX tracing fails, the system automatically falls back to hooks. You can also force a specific method:

```python
graph = export_graph(model, inputs, method=GraphExportMethod.HOOKS)
```

### Memory Issues

For large models, reduce batch size or number of batches:

```python
spec.batch_size = 4
result = run_benchmark(spec, data_loader, num_batches=10)
```

### Model Download Issues

If HuggingFace downloads fail, use a local model:

```python
spec = BenchmarkSpec(
    model_id="/path/to/local/model",
    model_commit="local",
    ...
)
```
