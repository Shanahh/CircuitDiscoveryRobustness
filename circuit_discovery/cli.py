"""Command-line interface for circuit discovery benchmarks."""
from pathlib import Path
from typing import Optional

import click
import torch
from torch.utils.data import DataLoader, TensorDataset

from circuit_discovery import __version__
from circuit_discovery.algorithms import AlgorithmRegistry
from circuit_discovery.pipeline import BenchmarkPipeline
from circuit_discovery.spec import BenchmarkSpec
from circuit_discovery.storage import ResultStorage


@click.group()
@click.version_option(version=__version__)
def main():
    """Circuit Discovery Benchmark Runner - Evaluate circuit discovery algorithms."""
    pass


@main.command()
@click.option("--spec", "-s", required=True, type=click.Path(exists=True), help="Path to YAML spec file")
@click.option("--num-batches", "-n", type=int, default=None, help="Number of batches to run")
@click.option("--save-caches", is_flag=True, help="Save activation/gradient caches")
@click.option("--device", "-d", default=None, help="Device to run on (cuda/cpu)")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run(
    spec: str,
    num_batches: Optional[int],
    save_caches: bool,
    device: Optional[str],
    verbose: bool,
):
    """Run a circuit discovery benchmark from a specification file."""
    # Load specification
    spec_path = Path(spec)
    benchmark_spec = BenchmarkSpec.from_yaml(spec_path)
    
    click.echo(f"Running benchmark: {benchmark_spec.task} / {benchmark_spec.algorithm}")
    click.echo(f"Model: {benchmark_spec.model_id}")
    click.echo(f"Seed: {benchmark_spec.seed}")
    
    # Create dummy data loader for testing
    # In practice, this would load task-specific data
    num_samples = num_batches * benchmark_spec.batch_size if num_batches else 100
    dummy_data = torch.randn(num_samples, benchmark_spec.sequence_length)
    dummy_labels = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    data_loader = DataLoader(dataset, batch_size=benchmark_spec.batch_size, shuffle=False)
    
    # Run pipeline
    pipeline = BenchmarkPipeline(benchmark_spec, device=device, verbose=verbose)
    result = pipeline.run(data_loader, num_batches=num_batches, save_caches=save_caches)
    
    # Display results
    click.echo("\n" + "=" * 60)
    click.echo("RESULTS")
    click.echo("=" * 60)
    click.echo(f"Run ID: {pipeline.run_id}")
    click.echo(f"Important nodes: {len(result.important_nodes)}")
    click.echo(f"Edge scores: {len(result.edge_scores)}")
    click.echo(f"Results saved to: {benchmark_spec.output_dir / pipeline.run_id}")
    
    if verbose and result.important_nodes:
        click.echo("\nTop important nodes:")
        # Sort by node score
        sorted_nodes = sorted(
            result.node_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        for node_id, score in sorted_nodes:
            click.echo(f"  {node_id}: {score:.4f}")


@main.command()
@click.option("--model-id", "-m", required=True, help="Model ID or path")
@click.option("--task", "-t", required=True, help="Task name")
@click.option("--algorithm", "-a", required=True, help="Algorithm name")
@click.option("--output", "-o", type=click.Path(), default="spec.yaml", help="Output file path")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size")
@click.option("--sequence-length", "-l", type=int, default=512, help="Sequence length")
def create_spec(
    model_id: str,
    task: str,
    algorithm: str,
    output: str,
    seed: int,
    batch_size: int,
    sequence_length: int,
):
    """Create a benchmark specification file."""
    spec = BenchmarkSpec(
        model_id=model_id,
        task=task,
        algorithm=algorithm,
        seed=seed,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    
    output_path = Path(output)
    spec.to_yaml(output_path)
    
    click.echo(f"Specification saved to: {output_path}")
    click.echo(f"  Model: {model_id}")
    click.echo(f"  Task: {task}")
    click.echo(f"  Algorithm: {algorithm}")


@main.command()
def list_algorithms():
    """List available circuit discovery algorithms."""
    algorithms = AlgorithmRegistry.list_algorithms()
    
    click.echo("Available algorithms:")
    for algo in algorithms:
        click.echo(f"  - {algo}")


@main.command()
@click.option("--results-dir", "-r", type=click.Path(exists=True), required=True, help="Results directory")
def list_runs(results_dir: str):
    """List all benchmark runs in a results directory."""
    storage = ResultStorage(Path(results_dir))
    runs = storage.list_runs()
    
    if not runs:
        click.echo("No runs found.")
        return
    
    click.echo(f"Found {len(runs)} run(s):")
    for run_id in runs:
        info = storage.get_run_info(run_id)
        click.echo(f"\n{run_id}:")
        if "spec" in info:
            click.echo(f"  Model: {info['spec']['model_id']}")
            click.echo(f"  Task: {info['spec']['task']}")
            click.echo(f"  Algorithm: {info['spec']['algorithm']}")
        click.echo(f"  Has result: {info['has_result']}")
        click.echo(f"  Has activations: {info['has_activations']}")
        click.echo(f"  Has gradients: {info['has_gradients']}")


@main.command()
@click.argument("run-id")
@click.option("--results-dir", "-r", type=click.Path(exists=True), required=True, help="Results directory")
@click.option("--show-nodes", "-n", type=int, default=10, help="Number of top nodes to show")
def show_result(run_id: str, results_dir: str, show_nodes: int):
    """Show results for a specific run."""
    storage = ResultStorage(Path(results_dir))
    
    try:
        result = storage.load_result(run_id)
    except FileNotFoundError:
        click.echo(f"Run not found: {run_id}")
        return
    
    click.echo(f"Results for run: {run_id}")
    click.echo("=" * 60)
    click.echo(f"Algorithm: {result['algorithm']}")
    click.echo(f"Important nodes: {len(result['important_nodes'])}")
    click.echo(f"Edge scores: {len(result['edge_scores'])}")
    
    if "_metadata" in result:
        click.echo("\nMetadata:")
        for key, value in result["_metadata"].items():
            click.echo(f"  {key}: {value}")
    
    if result['node_scores']:
        click.echo(f"\nTop {show_nodes} nodes by score:")
        sorted_nodes = sorted(
            result['node_scores'].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:show_nodes]
        for node_id, score in sorted_nodes:
            click.echo(f"  {node_id}: {score:.4f}")


@main.command()
@click.option("--results-dir", "-r", type=click.Path(exists=True), required=True, help="Results directory")
@click.option("--format", "-f", type=click.Choice(["table", "json"]), default="table", help="Output format")
def compare_runs(results_dir: str, format: str):
    """Compare results across multiple runs."""
    storage = ResultStorage(Path(results_dir))
    runs = storage.list_runs()
    
    if not runs:
        click.echo("No runs found.")
        return
    
    if format == "json":
        import json
        comparison = []
        for run_id in runs:
            info = storage.get_run_info(run_id)
            if info["has_result"]:
                result = storage.load_result(run_id)
                comparison.append({
                    "run_id": run_id,
                    "algorithm": result.get("algorithm"),
                    "important_nodes": len(result.get("important_nodes", [])),
                    "edge_scores": len(result.get("edge_scores", {})),
                })
        click.echo(json.dumps(comparison, indent=2))
    else:
        # Table format
        click.echo(f"{'Run ID':<40} {'Algorithm':<20} {'Nodes':<10} {'Edges':<10}")
        click.echo("=" * 82)
        for run_id in runs:
            info = storage.get_run_info(run_id)
            if info["has_result"]:
                result = storage.load_result(run_id)
                algo = result.get("algorithm", "N/A")
                nodes = len(result.get("important_nodes", []))
                edges = len(result.get("edge_scores", {}))
                click.echo(f"{run_id:<40} {algo:<20} {nodes:<10} {edges:<10}")


if __name__ == "__main__":
    main()
