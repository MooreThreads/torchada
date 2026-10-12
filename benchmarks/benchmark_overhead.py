#!/usr/bin/env python
"""
Benchmark to measure torchada patching overhead.

This script measures the overhead of torchada's runtime patching for common
torch.cuda.* API calls that are frequently used in sglang and similar projects.

Usage:
    python benchmarks/benchmark_overhead.py           # Run benchmarks and print results
    python benchmarks/benchmark_overhead.py --save    # Run and save results to history
"""

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "benchmark_history.json"


def benchmark_function(func, name, iterations=100000, warmup=1000):
    """Benchmark a function and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)

    return {
        "name": name,
        "iterations": iterations,
        "mean_ns": statistics.mean(times),
        "median_ns": statistics.median(times),
        "stdev_ns": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ns": min(times),
        "max_ns": max(times),
    }


def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks for all wrapper classes."""
    import torch

    import torchada

    results = []

    print("=" * 80)
    print("COMPREHENSIVE TORCHADA OVERHEAD ANALYSIS")
    print("=" * 80)
    print(f"Platform: {'MUSA' if torchada.is_musa_platform() else 'CUDA'}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # === _CudaModuleWrapper (torch.cuda.*) ===
    print("1. _CudaModuleWrapper (torch.cuda.* access)")
    print("-" * 60)

    results.append(
        benchmark_function(lambda: torch.cuda.device_count(), "torch.cuda.device_count()")
    )

    if torch.cuda.device_count() > 0:
        results.append(
            benchmark_function(lambda: torch.cuda.current_device(), "torch.cuda.current_device()")
        )

    # TORCHADA_FORCE_CUDA_AVAILABLE redirects this attribute to torch.musa, so
    # keep the reported name honest about which implementation was measured.
    is_available_redirected = torch.cuda.is_available is torch.musa.is_available
    results.append(
        benchmark_function(
            lambda: torch.cuda.is_available(),
            (
                "torch.cuda.is_available() [redirected]"
                if is_available_redirected
                else "torch.cuda.is_available() [NOT redirected]"
            ),
        )
    )

    results.append(benchmark_function(lambda: torch.cuda.Stream, "torch.cuda.Stream (attr)"))

    results.append(benchmark_function(lambda: torch.cuda.Event, "torch.cuda.Event (attr)"))

    # === _CudartWrapper (torch.cuda.cudart()) ===
    print("\n2. _CudartWrapper (torch.cuda.cudart())")
    print("-" * 60)

    try:
        cudart = torch.cuda.cudart()
        # First access (uncached)
        results.append(
            benchmark_function(lambda: cudart.cudaHostRegister, "cudart.cudaHostRegister (attr)")
        )
    except Exception as e:
        print(f"  Skipping cudart benchmarks: {e}")

    # === DeviceFactoryWrapper (torch.device) ===
    print("\n3. DeviceFactoryWrapper (torch.device)")
    print("-" * 60)

    results.append(benchmark_function(lambda: torch.device("cuda"), "torch.device('cuda')"))

    results.append(benchmark_function(lambda: torch.device("cuda:0"), "torch.device('cuda:0')"))

    results.append(benchmark_function(lambda: torch.device("cuda", 0), "torch.device('cuda', 0)"))

    # === tensor.is_cuda property ===
    print("\n4. Tensor.is_cuda property")
    print("-" * 60)

    t_cpu = torch.zeros(1)
    results.append(benchmark_function(lambda: t_cpu.is_cuda, "cpu_tensor.is_cuda (property)"))

    if torch.cuda.device_count() > 0:
        try:
            t_gpu = torch.zeros(1, device="cuda")
            results.append(
                benchmark_function(lambda: t_gpu.is_cuda, "gpu_tensor.is_cuda (property)")
            )
        except RuntimeError as e:
            print(f"  Skipping GPU tensor: {e}")

    # === _translate_device function (internal) ===
    print("\n5. _translate_device (internal)")
    print("-" * 60)

    from torchada._patch import _translate_device

    results.append(
        benchmark_function(lambda: _translate_device("cuda"), "_translate_device('cuda')")
    )

    results.append(
        benchmark_function(lambda: _translate_device("cuda:0"), "_translate_device('cuda:0')")
    )

    # === torch.backends.cuda ===
    print("\n6. torch.backends.cuda")
    print("-" * 60)

    results.append(
        benchmark_function(lambda: torch.backends.cuda.is_built(), "torch.backends.cuda.is_built()")
    )

    # === Print Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Operation':<45} {'Mean (ns)':<12} {'Median (ns)':<12} {'Min (ns)':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<45} {r['mean_ns']:<12.1f} {r['median_ns']:<12.1f} {r['min_ns']:<10}")

    print()
    print("Analysis:")
    print("-" * 40)

    # Categorize results
    fast = [r for r in results if r["mean_ns"] < 200]
    medium = [r for r in results if 200 <= r["mean_ns"] < 800]
    slow = [r for r in results if r["mean_ns"] >= 800]

    if fast:
        print(f"✅ Fast (<200ns): {len(fast)} operations - OPTIMIZED")
        for r in fast:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    if medium:
        print(f"⚠️  Medium (200-800ns): {len(medium)} operations")
        for r in medium:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    if slow:
        print(f"❌ Slow (>800ns): {len(slow)} operations - NEEDS OPTIMIZATION?")
        for r in slow:
            print(f"   - {r['name']}: {r['mean_ns']:.0f}ns")

    print()
    print("Note: 1 microsecond = 1000 nanoseconds")
    print("      Typical GPU kernel launch: 5,000-20,000 ns")
    print()

    return results


def run_micro_benchmarks():
    """Run micro-benchmarks to identify remaining optimization opportunities."""
    import torch

    from torchada._platform import Platform, detect_platform, is_musa_platform

    print("=" * 80)
    print("MICRO-BENCHMARKS FOR OPTIMIZATION ANALYSIS")
    print("=" * 80)
    print()

    results = []

    # Platform detection overhead
    print("1. Platform Detection")
    print("-" * 60)

    results.append(benchmark_function(lambda: detect_platform(), "detect_platform() [lru_cached]"))

    results.append(benchmark_function(lambda: is_musa_platform(), "is_musa_platform()"))

    results.append(
        benchmark_function(
            lambda: detect_platform() == Platform.MUSA, "detect_platform() == Platform.MUSA"
        )
    )

    # Test global variable access vs function call
    _cached_result = is_musa_platform()
    results.append(benchmark_function(lambda: _cached_result, "cached global variable access"))

    # _translate_device internals
    print("\n2. _translate_device internals")
    print("-" * 60)

    from torchada._patch import _device_str_cache, _translate_device

    # Pre-populate cache
    _translate_device("cuda")
    _translate_device("cuda:0")

    results.append(
        benchmark_function(
            lambda: "cuda" in _device_str_cache, "'cuda' in _device_str_cache (dict lookup)"
        )
    )

    results.append(
        benchmark_function(lambda: _translate_device("cuda"), "_translate_device('cuda') [cached]")
    )

    results.append(
        benchmark_function(lambda: _translate_device("cpu"), "_translate_device('cpu') [non-cuda]")
    )

    results.append(benchmark_function(lambda: _translate_device(None), "_translate_device(None)"))

    # isinstance checks
    print("\n3. isinstance checks")
    print("-" * 60)

    results.append(benchmark_function(lambda: isinstance("cuda", str), "isinstance('cuda', str)"))

    results.append(
        benchmark_function(
            lambda: isinstance("cuda", (str, torch.device)),
            "isinstance('cuda', (str, torch.device))",
        )
    )

    dev = torch.device("musa" if is_musa_platform() else "cuda")
    results.append(
        benchmark_function(lambda: isinstance(dev, torch.device), "isinstance(dev, torch.device)")
    )

    # String operations
    print("\n4. String operations")
    print("-" * 60)

    results.append(
        benchmark_function(lambda: "cuda".startswith("cuda:"), "'cuda'.startswith('cuda:')")
    )

    results.append(
        benchmark_function(
            lambda: "cuda:0".replace("cuda", "musa"), "'cuda:0'.replace('cuda', 'musa')"
        )
    )

    # Tensor operations
    print("\n5. Tensor operations (hot paths)")
    print("-" * 60)

    cpu_tensor = torch.randn(10, 10)
    results.append(benchmark_function(lambda: cpu_tensor.is_cuda, "cpu_tensor.is_cuda"))

    # Test hasattr overhead
    results.append(
        benchmark_function(lambda: hasattr(cpu_tensor, "musa"), "hasattr(tensor, 'musa')")
    )

    results.append(
        benchmark_function(lambda: hasattr(cpu_tensor, "is_musa"), "hasattr(tensor, 'is_musa')")
    )

    results.append(
        benchmark_function(
            lambda: getattr(cpu_tensor, "is_musa", False), "getattr(tensor, 'is_musa', False)"
        )
    )

    # Test device.type access
    results.append(benchmark_function(lambda: cpu_tensor.device, "tensor.device"))

    results.append(benchmark_function(lambda: cpu_tensor.device.type, "tensor.device.type"))

    results.append(
        benchmark_function(lambda: cpu_tensor.device.type == "musa", "tensor.device.type == 'musa'")
    )

    # Test try/except vs getattr
    def try_is_musa():
        try:
            return cpu_tensor.is_musa
        except AttributeError:
            return False

    results.append(benchmark_function(try_is_musa, "try: tensor.is_musa except: False"))

    # Print summary
    print("\n" + "=" * 80)
    print("MICRO-BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Operation':<50} {'Mean (ns)':<12} {'Min (ns)':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<50} {r['mean_ns']:<12.1f} {r['min_ns']:<10}")

    print()
    print("Key insights:")
    print("-" * 40)

    # Find the slowest operations
    sorted_results = sorted(results, key=lambda x: x["mean_ns"], reverse=True)
    print("Slowest operations:")
    for r in sorted_results[:5]:
        print(f"  - {r['name']}: {r['mean_ns']:.0f}ns")

    print()


def save_results_to_history(results):
    """Save benchmark results to the history file."""
    import torch

    import torchada

    # Get version info
    try:
        from torchada import __version__ as torchada_version
    except ImportError:
        torchada_version = "unknown"

    torch_musa_version = "N/A"
    if torchada.is_musa_platform():
        try:
            import torch_musa

            torch_musa_version = getattr(torch_musa, "__version__", "unknown")
        except ImportError:
            pass

    # Build operations dict
    operations = {}
    for r in results:
        operations[r["name"]] = {
            "mean_ns": round(r["mean_ns"]),
            "median_ns": round(r["median_ns"]),
            "min_ns": r["min_ns"],
        }

    # Count fast/medium operations
    fast_count = len([r for r in results if r["mean_ns"] < 200])
    medium_count = len([r for r in results if 200 <= r["mean_ns"] < 800])

    # Create new result entry
    new_entry = {
        "version": torchada_version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "platform": "MUSA" if torchada.is_musa_platform() else "CUDA",
        "pytorch_version": torch.__version__,
        "torch_musa_version": torch_musa_version,
        "operations": operations,
        "summary": {
            "fast_ops_count": fast_count,
            "medium_ops_count": medium_count,
            "notes": "",
        },
    }

    # Load existing history
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = {
            "schema_version": 1,
            "description": "Historical benchmark results",
            "results": [],
        }

    # Check if we already have a result for this version
    existing_idx = None
    for i, entry in enumerate(history["results"]):
        if entry["version"] == torchada_version and entry["platform"] == new_entry["platform"]:
            existing_idx = i
            break

    if existing_idx is not None:
        # Update existing entry
        history["results"][existing_idx] = new_entry
        print(f"\n✅ Updated benchmark results for version {torchada_version}")
    else:
        # Add new entry
        history["results"].append(new_entry)
        print(f"\n✅ Added benchmark results for version {torchada_version}")

    # Save history
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"   Saved to: {HISTORY_FILE}")

    # Print comparison with previous version if available
    if len(history["results"]) > 1:
        prev_entry = history["results"][-2]
        print(f"\n📊 Comparison with v{prev_entry['version']}:")
        print("-" * 60)
        for op_name, op_data in new_entry["operations"].items():
            if op_name in prev_entry["operations"]:
                prev_mean = prev_entry["operations"][op_name]["mean_ns"]
                curr_mean = op_data["mean_ns"]
                diff = curr_mean - prev_mean
                pct = (diff / prev_mean) * 100 if prev_mean > 0 else 0
                symbol = "🔺" if diff > 0 else "🔻" if diff < 0 else "➡️"
                print(
                    f"  {op_name[:40]:<40} {prev_mean:>6}ns → {curr_mean:>6}ns ({symbol} {pct:+.1f}%)"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark torchada overhead")
    parser.add_argument(
        "--save", action="store_true", help="Save results to benchmark_history.json"
    )
    args = parser.parse_args()

    results = run_comprehensive_benchmarks()
    print("\n\n")
    run_micro_benchmarks()

    if args.save:
        save_results_to_history(results)
