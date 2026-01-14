#!/usr/bin/env python3
"""
Demo script showing the circuit discovery benchmark runner in action.

This script demonstrates a complete pipeline run with a simple model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from circuit_discovery import BenchmarkSpec, run_benchmark
from circuit_discovery.pipeline import BenchmarkPipeline


class DemoModel(nn.Module):
    """A simple demo model for testing."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    """Run the demo."""
    print("=" * 70)
    print("Circuit Discovery Benchmark Runner Demo")
    print("=" * 70)
    
    # 1. Create a specification
    print("\n1. Creating benchmark specification...")
    spec = BenchmarkSpec(
        model_id="demo_model",
        task="demo_task",
        algorithm="attribution_patching",
        seed=42,
        batch_size=4,
        sequence_length=10,
        cache_activations=True,
        cache_gradients=False,
        output_dir="demo_results",
    )
    print(f"   Model: {spec.model_id}")
    print(f"   Task: {spec.task}")
    print(f"   Algorithm: {spec.algorithm}")
    print(f"   Batch size: {spec.batch_size}")
    
    # 2. Create demo data
    print("\n2. Creating demo dataset...")
    num_samples = 20
    data = torch.randn(num_samples, spec.sequence_length)
    labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=spec.batch_size, shuffle=False)
    print(f"   Created {num_samples} samples")
    
    # 3. Create pipeline and manually set model (since we're using a demo model)
    print("\n3. Initializing pipeline...")
    pipeline = BenchmarkPipeline(spec, device="cpu", verbose=True)
    
    # Override model loading with our demo model
    print("\n4. Loading demo model...")
    pipeline.model = DemoModel()
    print("   Demo model loaded")
    
    # 5. Run remaining pipeline stages
    print("\n5. Exporting computational graph...")
    pipeline.export_graph()
    
    print("\n6. Slicing subgraph...")
    pipeline.slice_subgraph()
    
    print("\n7. Running batches...")
    pipeline.run_batches(data_loader, num_batches=5)
    
    print("\n8. Running algorithm...")
    pipeline.run_algorithm(threshold=0.1)
    
    print("\n9. Saving results...")
    pipeline.save_results(save_caches=True)
    
    # 6. Display results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Run ID: {pipeline.run_id}")
    print(f"Algorithm: {pipeline.result.algorithm}")
    print(f"Total nodes in graph: {len(pipeline.graph.nodes)}")
    print(f"Important nodes found: {len(pipeline.result.important_nodes)}")
    print(f"Edge scores computed: {len(pipeline.result.edge_scores)}")
    
    if pipeline.result.node_scores:
        print("\nTop 5 nodes by importance score:")
        sorted_nodes = sorted(
            pipeline.result.node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for node_id, score in sorted_nodes:
            print(f"  {node_id}: {score:.4f}")
    
    print(f"\nResults saved to: {spec.output_dir}/{pipeline.run_id}")
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
