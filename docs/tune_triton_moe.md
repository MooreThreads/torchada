# tune_moe — MoE Kernel Tuning & Benchmarking Tool

## Overview

`tune_moe.py` is an autotuning and benchmarking tool for the fused MoE (Mixture of Experts) Triton kernels used in torchada. It searches over a large space of GPU kernel configurations (tiling sizes, warp counts, pipeline stages) to find the optimal parameters for each model architecture, batch size, and quantization mode.

The tool supports two modes:

- **Tuning mode** (`--tune`): Searches the full configuration space to find the best kernel parameters for each model × batch size combination, then saves the results to disk for later use.
- **Benchmark mode** (default): Runs the kernel using previously tuned (or default) configurations and reports performance.

> **Note:** This tool requires a **CUDA or MUSA GPU** with the **Triton** compiler installed.

---

## Quick Start

### 1. Tune a single model

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model deepseek-ai/DeepSeek-V2 \
    --tp 8 --ep 8 \
    --dtype auto \
    --tune
```

### 2. Tune multiple models from a config file

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config models.json \
    --tune
```

### 3. Benchmark after tuning

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config models.json \
    --output benchmark_results.json
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `models.json` | Path to JSON config file listing models (used if `--model` not given). |
| `--model` | `str` | `None` | Single HuggingFace model path (overrides `--config`). |
| `--tune` | flag | `False` | Run tuning mode (search for best configs). Without this flag, runs benchmarking. |
| `--materialize` | flag | `False` | Write the recipe's `pinned_rows` into their tables and exit (no GPU work); use it for rows an end-to-end measurement decided. |
| `--merge-configs` | flag | `False` | Merge the tuned buckets into the existing config file instead of replacing it, so a re-derived bucket keeps the other rows of a curated table. |
| `--batch-size` | `str` | *all defaults* | Batch size(s). Can specify multiple times or comma-separated, e.g. `--batch-size 1,2,4 --batch-size 8`. If omitted, uses a full default list. |
| `--tp-size` / `--tp` | `int` | `2` | Tensor parallelism (TP) size. |
| `--ep-size` / `--ep` | `int` | `1` | Expert parallelism (EP) size. |
| `--dtype` | `str` | `auto` | Quantization data type. Choices: `auto`, `fp8_w8a8`, `int8_w8a16`, `int8_w8a8`, `int4_w4a16`. |
| `--per-channel-quant` | flag | `False` | Use per-channel scaling. |
| `--seed` | `int` | `0` | Random seed for reproducibility. |
| `--disable-shared-experts-fusion` | flag | `False` | Disable shared experts fusion (relevant for DeepSeekV2/V3, GLM, etc.). |
| `--output` | `str` | `benchmark_results.json` | Path to output JSON file (benchmark mode only). |

---

## Config File Format (`models.json`)

The config file is a JSON array where each entry describes a model configuration to tune or benchmark.

```json
[
    {
        "model": "Qwen/Qwen3-30B-A3B",
        "tp_size": [1, 2, 4, 8, 2, 4, 8],
        "ep_size": [1, 1, 1, 1, 2, 4, 8],
        "dtype": "auto",
        "per_channel_quant": false,
        "disable_shared_experts_fusion": false
    },
    {
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "tp_size": [1, 2, 2, 4],
        "ep_size": [1, 1, 2, 4],
        "dtype": "fp8_w8a8"
    }
]
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `model` | ✅ (or a shape) | HuggingFace model name/path. |
| `tp_size` | ❌ (default: `--tp` arg) | TP size(s). Can be a single int or a list. |
| `ep_size` | ❌ (default: `--ep` arg) | EP size(s). Can be a single int or a list. |
| `dtype` | ❌ (default: `--dtype` arg) | Quantization dtype. |
| `per_channel_quant` | ❌ | Per-channel quantization flag. |
| `disable_shared_experts_fusion` | ❌ | Disable shared expert fusion. |
| `batch_sizes` | ❌ (default: `--batch-size` arg) | Buckets to tune for this entry only. Pins the bucket set a shipped table was tuned with. |
| `pinned_rows` | ❌ | `{bucket: {"config": {...}, "source": "..."}}` — rows chosen outside the grid, written verbatim by `--materialize`. `source` is required and records the measurement behind the row. |

> When `tp_size` and `ep_size` are both lists, they are zipped pairwise. If one list is shorter, it is broadcast to match the longer one.

**Shape entries (no model checkout):** an entry without `model` describes the kernel shape directly,
using the numbers a model config would resolve to. This is what makes a shipped table re-derivable
from the recipe alone — no model directory, no hub access:

```json
[
    {
        "name": "s5000_nemotron35_a3b_tiny_m",
        "num_experts": 128,
        "hidden_size": 2688,
        "shard_intermediate_size": 1856,
        "topk": 6,
        "is_gated": false,
        "activation": "relu2_no_mul",
        "dtype": "bf16",
        "tp_size": 1,
        "ep_size": 1,
        "batch_sizes": [16]
    }
]
```

| Shape field | Required | Description |
|---|---|---|
| `num_experts`, `hidden_size`, `shard_intermediate_size`, `topk` | ✅ | The kernel shape. `shard_intermediate_size` is the w1 output width: doubled for gated projections, the plain intermediate size otherwise. |
| `is_gated` | ❌ (default: `true`) | `false` for single-projection MoEs such as Nemotron-H (`relu2`). |
| `activation` | ❌ (default: `silu`) | Matches the model's MoE activation. |
| `name` | ❌ | Label used in logs and benchmark output. |

---

## How Tuning Works
### Configuration Search Space

The tuner searches over all combinations of these parameters:

```python
# From get_configs_compute_bound() in common_utils.py
configs = []
for num_stages in [1]:                       # Always 1 on Moore Threads
    for block_m in [32, 64, 128]:            # Block size along M (token) dimension
        for block_k in [32, 64, 128]:        # Block size along K (hidden) dimension
            for block_n in [32, 64, 128]:    # Block size along N (intermediate) dimension
                for num_warps in [4, 8, 16]: # Number of warps per block
                    for group_size in [1, 16, 32, 64]:  # Group size for GROUP_SIZE_M
                        configs.append({...})
# Additionally: rectangular block shapes (16,64) and (64,16)
```

**Total search space:** ~4,860 configurations per (model, batch_size) pair.

### Multiprocessing Model

The tuner uses Python `multiprocessing` (spawn method) to distribute the workload across **all available GPUs**.

- Each worker process is pinned to a single GPU via `torch.cuda.set_device(gpu_id)`.
- Configurations are split into chunks per worker.
- A shared `mp.Queue` distributes tasks and collects results via a producer-consumer pattern.
- A progress bar (`tqdm`) shows overall completion.

### Compatibility Filtering

Before benchmarking, each configuration is checked for compatibility with the model's quantization settings using `is_config_compatible()`. Configs that would cause assertion failures (e.g., `hidden_size % block_k != 0` for block-quantized models) are silently skipped.

---

## Output Files

### Tuned Configurations

Tuning results are saved as JSON files in:

```
import os
import triton
import torchada
base = torchada.__path__[0]
triton_version = tirton.__version__
config_path = os.path.join(base, "triton/autotune/fused_moe/configs/{triton_version}/")
print(config_path)

# Default moe configs dir: {base}/triton/autotune/fused_moe
print(os.environ.get("SGLANG_MOE_CONFIG_DIR"))
print(os.environ.get("VLLM_TUNED_CONFIG_FOLDER"))
```

The filename encodes the kernel signature:

```
E={num_experts},N={intermediate_size},device_name={GPU},
    dtype={dtype},block_shape=[{block_n},{block_k}].json
```

Each file contains a dictionary mapping batch sizes (as string keys) to best kernel configurations:

```json
{
    "1": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1
    },
    "32": {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 16,
        "num_warps": 8,
        "num_stages": 1
    },
    "4096": {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 64,
        "num_warps": 16,
        "num_stages": 1
    }
}
```

**Configuration directory priority:**
1. `SGLANG_MOE_CONFIG_DIR` environment variable (if set)
2. `VLLM_TUNED_CONFIG_FOLDER` environment variable (if set)
3. Default: `src/torchada/triton/autotune/fused_moe/configs/triton_{version}/`

On MUSA (Moore Threads) GPUs, the device name is typically `MTT_S5000` or similar.

### Benchmark Results

When running in benchmark mode (without `--tune`), results are saved to the file specified by `--output` (default: `benchmark_results.json`):

```json
[
    {
        "model": "Qwen/Qwen3-30B-A3B",
        "tp_size": 8,
        "ep_size": 8,
        "benchmarks": {
            "1": {"time_us": 123.45, "status": "success"},
            "2": {"time_us": 156.78, "status": "success"},
            "4": {"time_us": 210.34, "status": "success"},
            "2048": {"status": "failed", "error": "OOM during ..."}
        }
    }
]
```

---

## Quantization Support

The tool supports multiple quantization modes:

| `--dtype` | Weight | Activation | Scale Type | Notes |
|---|---|---|---|---|
| `auto` | fp16/bf16 | fp16/bf16 | None | Native precision from model config |
| `fp8_w8a8` | fp8-e4m3 | fp8-e4m3 | Per-expert or per-block | Requires H100 or MUSA FP8 support |
| `int8_w8a8` | int8 | int8 | Per-expert or per-block | Symmetric quantization |
| `int8_w8a16` | int8 | fp16/bf16 | Per-channel | Weight-only quantization |
| `int4_w4a16` | int4 | fp16/bf16 | Per-block | Weight-only quantization, higher compression |

### Block Quantization (@per-channel-quant)

When `--per-channel-quant` is set (or `block_shape` is found in the model config), the tool uses per-block quantization with the block shape specified in the model's HuggingFace configuration (e.g., `[128, 128]` for weight_block_size).

**Compatibility check:** The tuner automatically skips configurations where `hidden_size % block_k != 0` or `intermediate_size % block_k != 0` to avoid assertion failures.

---

## Practical Usage

### Typical Tuning Workflow

```bash
# 1. Start with a single model to verify the setup
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model Qwen/Qwen3-30B-A3B \
    --tp 8 --ep 8 \
    --tune

# 2. Tune all models defined in models.json
# (this can take many hours depending on GPU count)
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config models.json \
    --tune

# 3. Optionally benchmark to verify performance
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config models.json \
    --output benchmark_results.json
```

### Custom Batch Sizes

```bash
# Focus on inference-critical batch sizes
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model Qwen/Qwen3-30B-A3B \
    --tp 8 --ep 8 \
    --batch-size 1 --batch-size 8 --batch-size 32 --batch-size 64 \
    --tune
```

### Reproducing a shipped configuration

A shipped row has to be *derivable*, not typed in. `ci/shapes.json` holds the recipes for the
tables that were re-derived this way; the checked-in example is the Nemotron-3.5-Lightning-30B-A3B
tiny-M bucket on MTT_S5000. A recipe marks each bucket with the path that actually decided it:

**Measured rows** (`pinned_rows`) — a bucket whose winner came from an end-to-end measurement,
where the uniform-routing sweep is not representative. The recipe records the configuration *and*
the measurement it came from, and materializes it:

```bash
# Deterministic, no GPU: writes the pinned rows and keeps the rest of the table.
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config src/torchada/triton/autotune/fused_moe/ci/shapes.json \
    --materialize --merge-configs

# The result is the shipped table, byte for byte.
python -m pytest tests/test_tune_moe_recipe.py
```

The tool writes the indented JSON style the other tables use; this table is hand-formatted
(one row per line), so refreshing it in place reformats the file without changing its content.

**Swept rows** — a bucket the search grid can answer. Pin its bucket set in the recipe, then tune
into a scratch directory first; `--merge-configs` keeps the rows a run did not measure:

```bash
SGLANG_MOE_CONFIG_DIR=/tmp/retune python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --config src/torchada/triton/autotune/fused_moe/ci/shapes.json --tune
diff /tmp/retune/configs/triton_3_2_0/E=128,N=1856,device_name=MTT_S5000.json \
     src/torchada/triton/autotune/fused_moe/configs/triton_3_2_0/E=128,N=1856,device_name=MTT_S5000.json
```

Why the Nemotron bucket is pinned rather than swept: for `M = 16` the sweep ranks
BM64/BN128/BK64/GSM1/w8/s1, and it does not even agree with itself between two runs of the same
grid; the end-to-end measurement on the serving path prefers BM16/BN64/BK128/GSM1/w4/s1 — 5.5659 ms
against 6.6207 ms TPOT over 5 repeats at 4096-in/1000-out. The bucket also has to stay the only one
listed: the other seven rows of that table carry `num_stages` 2-4 from an earlier sweep, so a
full-table re-run would replace measured rows with unmeasured ones. A candidate the kernel cannot
launch is logged and skipped instead of aborting the sweep.

### FP8 Tuning

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model Qwen/Qwen3-30B-A3B-FP8 \
    --tp 8 --ep 8 \
    --dtype fp8_w8a8 \
    --tune
```

### DeepSeek V2/V3 with Shared Experts

```bash
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model deepseek-ai/DeepSeek-V2 \
    --tp 8 --ep 8 \
    --tune

# Disable shared expert fusion if desired
python src/torchada/triton/autotune/fused_moe/tune_moe.py \
    --model deepseek-ai/DeepSeek-V2 \
    --tp 8 --ep 8 \
    --disable-shared-experts-fusion \
    --tune
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `SGLANG_MOE_CONFIG_DIR` | Directory where tuned configs are saved/loaded (overrides default path). |
| `VLLM_TUNED_CONFIG_FOLDER` | Alternative config directory (checked if `SGLANG_MOE_CONFIG_DIR` is not set). |

These variables are automatically set by `torchada/triton/autotune/fused_moe.__init__` to point to the package's config directory unless already defined.

---
