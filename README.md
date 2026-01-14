# Circuit Discovery Robustness

A comprehensive benchmark runner for evaluating circuit discovery algorithms on neural networks with swappable models, tasks, and algorithms.

## Overview

This project provides a modular pipeline for running circuit discovery experiments on PyTorch models. It supports:

- **Swappable Components**: Models, tasks, and algorithms are fully configurable
- **Graph Export**: Multiple methods (FX, torch.export, hooks) with automatic fallback
- **Reproducibility**: Pinned model versions, checksums, and comprehensive versioning
- **Caching**: Efficient activation and gradient caching for large-scale experiments
- **Storage**: Complete results storage with replay capability

## Installation

```bash
# Clone the repository
git clone https://github.com/Shanahh/CircuitDiscoveryRobustness.git
cd CircuitDiscoveryRobustness

# Install the package
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Create a Specification

Create a YAML specification file or use the CLI:

```bash
circuit-bench create-spec \
  --model-id gpt2 \
  --task ioi \
  --algorithm attribution_patching \
  --output my_spec.yaml
```

Or manually create a `spec.yaml`:

```yaml
model_id: "gpt2"
task: "ioi"
algorithm: "attribution_patching"
seed: 42
batch_size: 8
sequence_length: 128
cache_activations: true
cache_gradients: false
output_dir: "results"
```

### 2. Run a Benchmark

```bash
circuit-bench run --spec my_spec.yaml
```

### 3. View Results

```bash
# List all runs
circuit-bench list-runs --results-dir results

# Show specific run results
circuit-bench show-result RUN_ID --results-dir results

# Compare multiple runs
circuit-bench compare-runs --results-dir results
```

## Pipeline Architecture

The benchmark pipeline consists of the following stages:

1. **Load Spec**: Parse YAML specification with model ID, task, algorithm, and parameters
2. **Download Snapshot**: Download pinned model version from HuggingFace Hub
3. **Integrity Gate**: Verify checksums and model completeness
4. **Export Graph**: Extract computational graph using FX, torch.export, or hooks
5. **Canonical Node IDs**: Create consistent node identifiers with metadata
6. **Slice Subgraph**: Extract relevant subgraph from target nodes
7. **Run Batches**: Execute batches with activation/gradient caching
8. **Run Algorithm**: Execute circuit discovery algorithm
9. **Store Results**: Save results with full versioning for replay

## Components

### Models (`circuit_discovery.models`)

Handles model downloading, caching, and integrity verification:

```python
from circuit_discovery.models import load_model_from_spec
from circuit_discovery.spec import BenchmarkSpec

spec = BenchmarkSpec(model_id="gpt2", task="ioi", algorithm="acdc")
model, snapshot = load_model_from_spec(spec, device="cuda")
```

### Graph Export (`circuit_discovery.graph_export`)

Exports computational graphs using multiple methods:

```python
from circuit_discovery.graph_export import export_graph, GraphExportMethod

# Auto-select method
graph = export_graph(model, example_inputs)

# Force specific method
graph = export_graph(model, example_inputs, method=GraphExportMethod.FX)
```

### Subgraph Slicing (`circuit_discovery.subgraph`)

Slice graphs to extract relevant circuits:

```python
from circuit_discovery.subgraph import slice_graph

# Backward slice from targets
subgraph = slice_graph(graph, target_nodes=["layer.4", "layer.5"])
```

### Batch Execution (`circuit_discovery.execution`)

Run batches with efficient caching:

```python
from circuit_discovery.execution import BatchExecutor

executor = BatchExecutor(
    model=model,
    cache_activations=True,
    cache_gradients=True,
    device="cuda"
)

act_cache, grad_cache = executor.run_batches(data_loader)
```

### Algorithms (`circuit_discovery.algorithms`)

Built-in algorithms:
- **Attribution Patching**: Activation-based importance scoring
- **ACDC** (Automatic Circuit Discovery): Gradient-based discovery
- **EAP** (Edge Attribution Patching): Edge-focused circuit discovery

```python
from circuit_discovery.algorithms import AlgorithmRegistry

# List available algorithms
algorithms = AlgorithmRegistry.list_algorithms()

# Create algorithm instance
algo = AlgorithmRegistry.create("attribution_patching", threshold=0.1)

# Run discovery
result = algo.discover(graph, activation_cache, gradient_cache)
```

### Custom Algorithms

Register your own algorithms:

```python
from circuit_discovery.algorithms import CircuitDiscoveryAlgorithm, AlgorithmRegistry

class MyAlgorithm(CircuitDiscoveryAlgorithm):
    def discover(self, graph, activation_cache, gradient_cache=None, **kwargs):
        # Your implementation
        result = CircuitDiscoveryResult(algorithm="my_algorithm")
        # ... compute node and edge scores
        return result
    
    @classmethod
    def name(cls):
        return "my_algorithm"

# Register
AlgorithmRegistry.register(MyAlgorithm)
```

### Storage (`circuit_discovery.storage`)

Persistent storage with versioning:

```python
from circuit_discovery.storage import ResultStorage

storage = ResultStorage(Path("results"))

# Save results
storage.save_spec(run_id, spec)
storage.save_result(run_id, result, metadata)
storage.save_activations(run_id, activation_cache)

# Load results
spec = storage.load_spec(run_id)
result = storage.load_result(run_id)
```

## Python API

### Complete Pipeline Example

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
    sequence_length=128,
)

# Create data loader (example)
data = torch.randn(100, 128)
labels = torch.randint(0, 2, (100,))
dataset = TensorDataset(data, labels)
data_loader = DataLoader(dataset, batch_size=8)

# Run benchmark
result = run_benchmark(
    spec=spec,
    data_loader=data_loader,
    num_batches=10,
    save_caches=True,
    device="cuda",
)

print(f"Found {len(result.important_nodes)} important nodes")
```

### Step-by-Step Pipeline

```python
from circuit_discovery.pipeline import BenchmarkPipeline

pipeline = BenchmarkPipeline(spec, device="cuda", verbose=True)

# Run individual stages
pipeline.load_model()
pipeline.export_graph()
pipeline.slice_subgraph()
pipeline.run_batches(data_loader, num_batches=10)
pipeline.run_algorithm(threshold=0.1)
pipeline.save_results(save_caches=True)

# Access results
print(pipeline.result.important_nodes)
print(pipeline.run_id)
```

## CLI Commands

### `circuit-bench run`
Run a benchmark from a specification file.

```bash
circuit-bench run --spec spec.yaml --num-batches 100 --save-caches --device cuda
```

### `circuit-bench create-spec`
Create a specification file interactively.

```bash
circuit-bench create-spec -m gpt2 -t ioi -a acdc -o spec.yaml
```

### `circuit-bench list-algorithms`
List available circuit discovery algorithms.

```bash
circuit-bench list-algorithms
```

### `circuit-bench list-runs`
List all benchmark runs in a directory.

```bash
circuit-bench list-runs --results-dir results
```

### `circuit-bench show-result`
Display detailed results for a specific run.

```bash
circuit-bench show-result RUN_ID --results-dir results --show-nodes 20
```

### `circuit-bench compare-runs`
Compare results across multiple runs.

```bash
circuit-bench compare-runs --results-dir results --format table
```

## Examples

See the `examples/` directory for sample specifications:

- `ioi_attribution.yaml` - Indirect Object Identification with Attribution Patching
- `greaterthan_acdc.yaml` - Greater-Than task with ACDC
- `tracr_eap.yaml` - TRACR task with Edge Attribution Patching

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black circuit_discovery/
ruff check circuit_discovery/
```

### Type Checking

```bash
mypy circuit_discovery/
```

## Architecture Details

### Graph Export Methods

1. **FX (torch.fx)**: Symbolic tracing for most PyTorch models
2. **torch.export**: PyTorch 2.x export API for better compatibility
3. **Hooks**: Fallback using forward/backward hooks for any model

The pipeline automatically tries methods in order and falls back as needed.

### Canonical Node IDs

Each graph node receives a canonical ID based on the export method:
- FX: `fx_{index}_{name}`
- Export: `export_{index}_{name}`
- Hooks: `hook_{index}_{module_path}`

### Caching Strategy

- **Activations**: Stored per-node across batches
- **Gradients**: Optional, stored when needed for gradient-based algorithms
- **Memory**: Tensors are detached and cloned to prevent memory leaks
- **Storage**: Optional persistent storage using PyTorch's save/load

### Versioning

All results include:
- PyTorch version
- Model commit hash
- Model checksum
- Algorithm metadata
- Timestamp

This enables exact replay of experiments.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this benchmark runner in your research, please cite:

```bibtex
@software{circuit_discovery_robustness,
  title={Circuit Discovery Robustness Benchmark},
  author={Your Name},
  year={2024},
  url={https://github.com/Shanahh/CircuitDiscoveryRobustness}
}
```
