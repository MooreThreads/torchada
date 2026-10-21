"""Benchmark the TorchADA MUSA joint RoPE provider."""

import argparse
import json
import statistics

import torch
import torch_musa

import torchada


def make_case(
    *,
    tokens: int,
    query_heads: int,
    key_heads: int,
    head_dimension: int,
    rope_dimension: int,
    is_neox: bool,
) -> dict:
    query_storage = torch.randn(
        tokens, query_heads, head_dimension, device="musa", dtype=torch.bfloat16
    )
    key_storage = torch.randn(
        tokens, key_heads, head_dimension, device="musa", dtype=torch.bfloat16
    )
    query = query_storage[..., :rope_dimension]
    key = key_storage[..., :rope_dimension]
    angles = torch.randn(8192, rope_dimension // 2, device="musa", dtype=torch.float32)
    return {
        "name": "partial" if rope_dimension < head_dimension else "full",
        "query": query,
        "key": key,
        "cos_sin_cache": torch.cat((angles.cos(), angles.sin()), dim=-1),
        "positions": torch.arange(tokens, device="musa", dtype=torch.int64),
        "is_neox": is_neox,
        "tokens": tokens,
        "query_heads": query_heads,
        "key_heads": key_heads,
        "head_dimension": head_dimension,
        "rope_dimension": rope_dimension,
    }


def benchmark(case: dict, *, warmup: int, iterations: int, repeats: int) -> dict:
    arguments = (
        case["query"],
        case["key"],
        case["cos_sin_cache"],
        case["positions"],
    )
    keywords = {"is_neox": case["is_neox"]}
    for _ in range(warmup):
        torchada.apply_rope_inplace(*arguments, **keywords)
    torch.musa.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            torchada.apply_rope_inplace(*arguments, **keywords)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / iterations)

    return {
        "name": case["name"],
        "tokens": case["tokens"],
        "query_heads": case["query_heads"],
        "key_heads": case["key_heads"],
        "head_dimension": case["head_dimension"],
        "rope_dimension": case["rope_dimension"],
        "is_neox": case["is_neox"],
        "dtype": "bfloat16",
        "positions_dtype": "int64",
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "samples_ms": samples_ms,
        "median_latency_ms": statistics.median(samples_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")
    torch.manual_seed(0)
    cases = (
        make_case(
            tokens=args.tokens,
            query_heads=32,
            key_heads=8,
            head_dimension=128,
            rope_dimension=64,
            is_neox=False,
        ),
        make_case(
            tokens=args.tokens,
            query_heads=32,
            key_heads=8,
            head_dimension=128,
            rope_dimension=128,
            is_neox=True,
        ),
    )
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "torch_musa": torch_musa.__version__,
                "device": torch.musa.get_device_name(0),
                "results": [
                    benchmark(
                        case,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        repeats=args.repeats,
                    )
                    for case in cases
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
