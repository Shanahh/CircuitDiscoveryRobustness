"""Circuit Discovery Benchmark Runner package."""

__version__ = "0.1.0"

from circuit_discovery.spec import BenchmarkSpec
from circuit_discovery.pipeline import run_benchmark

__all__ = ["BenchmarkSpec", "run_benchmark", "__version__"]
