"""
OpenCog Multi-Agent Orchestration Example using Python API

This example demonstrates how to use the OpenCog backend for multi-agent
orchestration of model benchmarking with the Python API.
"""

from optimum_benchmark import (
    Benchmark,
    BenchmarkConfig,
    InferenceConfig,
    OpenCogConfig,
    ProcessConfig,
)
from optimum_benchmark.logging_utils import setup_logging

setup_logging(level="INFO")

if __name__ == "__main__":
    # Configure the OpenCog multi-agent backend
    backend_config = OpenCogConfig(
        model="bert-base-uncased",
        device="cpu",
        no_weights=True,
        # Multi-agent orchestration settings
        num_agents=3,
        agent_coordination="sequential",
        agent_memory_sharing=True,
        # Cognitive reasoning settings
        reasoning_engine="pattern_matcher",
        atomspace_size=10000,
        enable_attention_allocation=True,
        # Orchestration workflow
        workflow_strategy="adaptive",
        enable_self_optimization=True,
        # Integration settings
        delegate_backend="pytorch",
        # Performance tuning
        load_balancing="round_robin",
    )

    # Configure the inference scenario
    scenario_config = InferenceConfig(
        latency=True,
        memory=True,
        warmup_runs=10,
        input_shapes={"batch_size": 2, "sequence_length": 128},
    )

    # Configure the launcher
    launcher_config = ProcessConfig()

    # Create the benchmark configuration
    benchmark_config = BenchmarkConfig(
        name="opencog_multi_agent_bert",
        scenario=scenario_config,
        launcher=launcher_config,
        backend=backend_config,
    )

    # Run the benchmark
    print("\n" + "=" * 80)
    print("Running OpenCog Multi-Agent Orchestration Benchmark")
    print("=" * 80 + "\n")
    
    benchmark_report = Benchmark.launch(benchmark_config)

    # Print orchestration metrics
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80 + "\n")
    
    print(f"Latency (mean): {benchmark_report.latency.mean:.4f} seconds")
    print(f"Throughput: {benchmark_report.latency.throughput:.2f} samples/second")
    
    if hasattr(benchmark_report, 'memory') and benchmark_report.memory:
        print(f"Peak Memory: {benchmark_report.memory.max_ram / (1024**3):.2f} GB")
    
    # Save results
    benchmark_config.save_json("opencog_benchmark_config.json")
    benchmark_report.save_json("opencog_benchmark_report.json")
    
    print("\nResults saved to opencog_benchmark_*.json")
    print("\n" + "=" * 80)
