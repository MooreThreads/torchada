# Adapted from https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_moe.py

import argparse
import json
import logging
import multiprocessing as mp
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import torch
import triton
from tqdm import tqdm

# Custom MoE kernels and utilities (assumed to be available)
from torchada.triton.autotune.fused_moe.utils import (
    BenchmarkConfig,
    get_config_filename,
    get_configs_compute_bound,
    get_default_batch_sizes,
    get_model_config,
    save_configs,
    sort_config,
)
from torchada.triton.runtime.fused_moe.config import (
    get_config_dtype_str,
    get_default_config,
    get_moe_configs,
    override_config,
)
from torchada.triton.runtime.fused_moe.fused_moe import fused_moe
from torchada.triton.runtime.fused_moe.router import TopKConfig, select_experts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_dtype_str(requested_dtype: str, params: Dict) -> str:
    if requested_dtype != "auto":
        return requested_dtype
    quant_dtype_str = params.get("quant_dtype_str")
    if quant_dtype_str and quant_dtype_str != "auto":
        return quant_dtype_str
    dtype = params.get("dtype")
    if dtype is not None:
        return str(dtype).replace("torch.", "")
    raise ValueError(
        "Unable to resolve dtype='auto'. Set dtype explicitly in the tune config "
        "or provide model config dtype/quantization_config metadata."
    )


def _resolve_torch_dtype(dtype_str: str, params: Dict) -> torch.dtype:
    dtype_aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    dtype = params.get("dtype")
    if dtype is not None:
        if isinstance(dtype, str):
            dtype = dtype.replace("torch.", "")
            try:
                return dtype_aliases[dtype]
            except KeyError as exc:
                raise ValueError(f"Unsupported model config dtype {dtype!r}") from exc
        return dtype
    if dtype_str in {"fp8_w8a8", "int8_w8a8", "int8_w8a16", "int4_w4a16"}:
        return torch.bfloat16
    try:
        return dtype_aliases[dtype_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_str!r}") from exc


def _is_quant_dtype(dtype_str: str, quant_dtype: str) -> bool:
    return dtype_str == quant_dtype


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    Unspecified = 6


@dataclass
class MoeRunnerConfig:
    num_experts: Optional[int] = None
    num_local_experts: Optional[int] = None
    hidden_size: Optional[int] = None
    intermediate_size_per_partition: Optional[int] = None
    layer_id: Optional[int] = None
    top_k: Optional[int] = None
    num_fused_shared_experts: Optional[int] = None
    params_dtype: Optional[torch.dtype] = None
    routing_method_type: Optional[RoutingMethodType] = None
    activation: str = "silu"
    is_gated: bool = True
    apply_router_weight_on_input: bool = False
    inplace: bool = True
    no_combine: bool = False
    routed_scaling_factor: Optional[float] = None
    gemm1_alpha: Optional[float] = None
    gemm1_clamp_limit: Optional[float] = None


@dataclass
class ModelEntry:
    """Holds all information about a model configuration to be tuned/benchmarked."""

    path: str
    tp_size: int
    ep_size: int
    disable_shared_fusion: bool
    dtype_str: str
    per_channel_quant: bool
    # Derived fields (filled by get_model_config)
    num_experts: int = 0
    hidden_size: int = 0
    shard_intermediate_size: int = 0
    topk: int = 0
    num_fused_shared_experts: int = 0
    dtype: torch.dtype = torch.float16
    block_shape: Optional[Tuple[int, int]] = None

    @property
    def use_fp8(self) -> bool:
        return _is_quant_dtype(self.dtype_str, "fp8_w8a8")

    @property
    def use_int8(self) -> bool:
        return _is_quant_dtype(self.dtype_str, "int8_w8a8")

    @property
    def use_int8a16(self) -> bool:
        return _is_quant_dtype(self.dtype_str, "int8_w8a16")

    @property
    def use_int4(self) -> bool:
        return _is_quant_dtype(self.dtype_str, "int4_w4a16")

    @property
    def unique_key(self) -> Tuple:
        return (
            self.num_experts,
            self.hidden_size,
            self.shard_intermediate_size,
            self.topk,
            self.num_fused_shared_experts,
            str(self.dtype),
            self.use_fp8,
            self.use_int8,
            self.use_int8a16,
            self.use_int4,
            self.per_channel_quant,
            self.block_shape,
        )


# ------------------------------------------------------------------------------
# Model validation and warning generation (unchanged)
# ------------------------------------------------------------------------------
def validate_and_log_entries(entries: List[ModelEntry]) -> None:
    """Print each model entry and emit warnings for suspicious configurations."""
    logger.info("Validating %d model entries:", len(entries))
    warn_once: Set[str] = set()

    for i, e in enumerate(entries):
        logger.info(
            "[%d] model=%s tp=%d ep=%d experts=%d hidden=%d "
            "intermediate=%d topk=%d shared=%d dtype=%s block=%s",
            i,
            e.path,
            e.tp_size,
            e.ep_size,
            e.num_experts,
            e.hidden_size,
            e.shard_intermediate_size,
            e.topk,
            e.num_fused_shared_experts,
            e.dtype_str,
            e.block_shape,
        )

        # Warning: ep_size > num_experts
        if e.ep_size > e.num_experts:
            warn_key = f"ep_exceeds_{e.path}"
            if warn_key not in warn_once:
                logger.warning(
                    "Model %s: ep_size (%d) > num_experts (%d). This may cause expert replication or waste.",
                    e.path,
                    e.ep_size,
                    e.num_experts,
                )
                warn_once.add(warn_key)

        # Warning: block quantization divisibility
        if e.block_shape is not None and (e.use_fp8 or e.use_int8):
            block_k = e.block_shape[1]
            if e.hidden_size % block_k != 0:
                warn_key = f"hidden_div_{e.path}"
                if warn_key not in warn_once:
                    logger.warning(
                        "Model %s: hidden_size=%d not divisible by block_k=%d. "
                        "Quantization will fail on tp_size=%d ep_size=%d.",
                        e.path,
                        e.hidden_size,
                        block_k,
                        e.tp_size,
                        e.ep_size,
                    )
                    warn_once.add(warn_key)
            if e.shard_intermediate_size % block_k != 0:
                warn_key = f"interm_div_{e.path}"
                if warn_key not in warn_once:
                    logger.warning(
                        "Model %s: shard_intermediate_size=%d not divisible by block_k=%d. "
                        "Weight quantization may fail on tp_size=%d ep_size=%d.",
                        e.path,
                        e.shard_intermediate_size,
                        block_k,
                        e.tp_size,
                        e.ep_size,
                    )
                    warn_once.add(warn_key)

    logger.info("Validation complete.")


# Configuration compatibility check (skip invalid configs before benchmarking)
_warned_incompat: Set[Tuple] = set()


def is_config_compatible(
    config: BenchmarkConfig,
    entry: ModelEntry,
    block_shape: Optional[List[int]],
) -> bool:
    """
    Return True if the configuration is compatible with the model's shapes
    and quantization settings. Skip configs that would cause `per_token_group_quant_fp8`
    assertion failures (e.g., hidden_size % block_k != 0).
    """
    # If block quantization is not used, everything is compatible
    if block_shape is None:
        return True

    # Only FP8 or INT8 per‑block quantization require divisibility by block_k
    if not (entry.use_fp8 or entry.use_int8):
        return True

    block_k = block_shape[1]  # group size for quantization
    # Input (x) must have hidden_size divisible by block_k
    if entry.hidden_size % block_k != 0:
        key = ("hidden_div", entry.path, block_k)
        if key not in _warned_incompat:
            logger.warning(
                "Skipping config with block_k=%d because hidden_size=%d not divisible by block_k, model=%s, tp_size=%d, ep_size=%d",
                block_k,
                entry.hidden_size,
                entry.path,
                entry.tp_size,
                entry.ep_size,
            )
            _warned_incompat.add(key)
        return False
    # Weights (intermediate dimension) also need to be divisible by block_k
    if entry.shard_intermediate_size % block_k != 0:
        key = ("interm_div", entry.path, block_k)
        if key not in _warned_incompat:
            logger.warning(
                "Skipping config with block_k=%d because shard_intermediate_size=%d not divisible by block_k, model=%s, tp_size=%d, ep_size=%d",
                block_k,
                entry.shard_intermediate_size,
                entry.path,
                entry.tp_size,
                entry.ep_size,
            )
            _warned_incompat.add(key)
        return False
    return True


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: List[int] = None,
    num_fused_shared_experts: int = 0,
    num_iters: int = 100,
) -> float:
    """Run the fused MoE kernel and return latency in microseconds."""
    device = "cuda"
    torch.set_default_device(device)
    init_dtype = torch.float16 if use_fp8_w8a8 else dtype
    num_routed_experts = num_experts - num_fused_shared_experts
    assert num_routed_experts > 0
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # Create random weights based on quantization type
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (num_experts, shard_intermediate_size, hidden_size),
            dtype=torch.int8,
            device=device,
        )
        w2 = torch.randint(
            -127,
            127,
            (num_experts, hidden_size, shard_intermediate_size // 2),
            dtype=torch.int8,
            device=device,
        )
    elif use_int4_w4a16:
        w1 = torch.randint(
            0,
            255,
            (num_experts, shard_intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2 = torch.randint(
            0,
            255,
            (num_experts, hidden_size, shard_intermediate_size // 4),
            dtype=torch.uint8,
            device=device,
        )
    else:
        w1 = torch.randn(
            num_experts,
            shard_intermediate_size,
            hidden_size,
            dtype=init_dtype,
            device=device,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            shard_intermediate_size // 2,
            dtype=init_dtype,
            device=device,
        )

    gating_output = torch.randn(
        num_iters,
        num_tokens,
        num_routed_experts,
        dtype=torch.float32,
        device=device,
    )

    # Scales for quantized paths
    w1_scale = w2_scale = a1_scale = a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size),
            dtype=torch.float32,
            device=device,
        )
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
    if use_int4_w4a16:
        block_n = 1 if (block_shape[0] == 0) else block_shape[0]
        block_k = block_shape[1]
        n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
        n_tiles_w2 = (hidden_size + block_n - 1) // block_n
        k_tiles_w1 = (hidden_size + block_k - 1) // block_k
        k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
        w1_scale = torch.randn(
            (num_experts, n_tiles_w1, k_tiles_w1),
            dtype=torch.bfloat16,
            device=device,
        )
        w2_scale = torch.randn(
            (num_experts, n_tiles_w2, k_tiles_w2),
            dtype=torch.bfloat16,
            device=device,
        )
    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32, device=device
            )
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)
        else:
            block_n, block_k = block_shape[0], block_shape[1]
            n_tiles_w1 = (shard_intermediate_size + block_n - 1) // block_n
            n_tiles_w2 = (hidden_size + block_n - 1) // block_n
            k_tiles_w1 = (hidden_size + block_k - 1) // block_k
            k_tiles_w2 = (shard_intermediate_size // 2 + block_k - 1) // block_k
            w1_scale = torch.rand(
                (num_experts, n_tiles_w1, k_tiles_w1),
                dtype=torch.float32,
                device=device,
            )
            w2_scale = torch.rand(
                (num_experts, n_tiles_w2, k_tiles_w2),
                dtype=torch.float32,
                device=device,
            )

    if use_fp8_w8a8:
        w1 = w1.to(torch.float8_e4m3fn)
        w2 = w2.to(torch.float8_e4m3fn)

    input_gating = torch.randn(num_tokens, num_routed_experts, dtype=torch.float32, device=device)
    topk_config = TopKConfig(
        top_k=topk,
        renormalize=True,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    topk_output = select_experts(x, input_gating, topk_config)

    def prepare(i: int):
        new_topk_output = select_experts(x, gating_output[i], topk_config)
        topk_output.topk_weights.copy_(new_topk_output.topk_weights)
        topk_output.topk_ids.copy_(new_topk_output.topk_ids)
        topk_output.router_logits.copy_(new_topk_output.router_logits)

    def run():
        moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=shard_intermediate_size,
            top_k=topk,
            num_fused_shared_experts=num_fused_shared_experts,
            inplace=True,
        )
        with override_config(config):
            fused_moe(
                x,
                w1,
                w2,
                topk_output,
                moe_runner_config=moe_runner_config,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape,
            )

    # Warmup & JIT
    run()
    torch.cuda.synchronize()

    use_graph = _env_flag("TORCHADA_TUNE_USE_GRAPH") and hasattr(torch.cuda, "CUDAGraph")
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                run()
        torch.cuda.synchronize()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
    else:
        for _ in range(5):
            run()
        torch.cuda.synchronize()

    # Flush L2 cache
    cache_flush = torch.empty(int(256e6 // 4), dtype=torch.int, device=device)
    cache_flush.zero_()

    # Timing loop
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for i in range(num_iters):
        prepare(i)
        start_events[i].record()
        if use_graph:
            graph.replay()
        else:
            run()
        end_events[i].record()
    torch.cuda.synchronize()

    latencies = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
    avg_us = sum(latencies) / num_iters * 1000
    if use_graph:
        graph.reset()
    return avg_us


# Build model entries from command line arguments (with error skipping)
def build_model_entries(args: argparse.Namespace) -> List[ModelEntry]:
    entries: List[ModelEntry] = []

    if args.model is not None:
        entry = ModelEntry(
            path=args.model,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            disable_shared_fusion=args.disable_shared_experts_fusion,
            dtype_str=args.dtype,
            per_channel_quant=args.per_channel_quant,
        )
        try:
            params = get_model_config(
                entry.path, entry.tp_size, entry.ep_size, entry.disable_shared_fusion
            )
            entry.num_experts = params["num_experts"]
            entry.hidden_size = params["hidden_size"]
            entry.shard_intermediate_size = params["shard_intermediate_size"]
            entry.topk = params["topk"]
            entry.num_fused_shared_experts = params.get("num_fused_shared_experts", 0)
            entry.dtype_str = _resolve_dtype_str(entry.dtype_str, params)
            entry.dtype = _resolve_torch_dtype(entry.dtype_str, params)
            entry.block_shape = tuple(params["block_shape"]) if params["block_shape"] else None
            entries.append(entry)
        except Exception as e:
            logger.error(f"Failed to load config for {entry.path}: {e}")
    else:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        with open(args.config, "r") as f:
            raw_entries = json.load(f)

        for raw in raw_entries:
            model_path = raw["model"]
            disable_fusion = raw.get(
                "disable_shared_experts_fusion", args.disable_shared_experts_fusion
            )
            dtype_str = raw.get("dtype", args.dtype)
            per_channel = raw.get("per_channel_quant", args.per_channel_quant)

            tp_sizes = raw.get("tp_size", args.tp_size)
            ep_sizes = raw.get("ep_size", args.ep_size)
            if not isinstance(tp_sizes, list):
                tp_sizes = [tp_sizes]
            if not isinstance(ep_sizes, list):
                ep_sizes = [ep_sizes]

            max_len = max(len(tp_sizes), len(ep_sizes))
            if len(tp_sizes) == 1 and max_len > 1:
                tp_sizes = tp_sizes * max_len
            if len(ep_sizes) == 1 and max_len > 1:
                ep_sizes = ep_sizes * max_len
            if len(tp_sizes) != len(ep_sizes):
                raise ValueError(
                    "tp_size and ep_size lists must have same length after broadcasting"
                )

            for tp, ep in zip(tp_sizes, ep_sizes):
                entry = ModelEntry(
                    path=model_path,
                    tp_size=tp,
                    ep_size=ep,
                    disable_shared_fusion=disable_fusion,
                    dtype_str=dtype_str,
                    per_channel_quant=per_channel,
                )
                try:
                    params = get_model_config(
                        entry.path, entry.tp_size, entry.ep_size, entry.disable_shared_fusion
                    )
                    entry.num_experts = params["num_experts"]
                    entry.hidden_size = params["hidden_size"]
                    entry.shard_intermediate_size = params["shard_intermediate_size"]
                    entry.topk = params["topk"]
                    entry.num_fused_shared_experts = params.get("num_fused_shared_experts", 0)
                    entry.dtype_str = _resolve_dtype_str(entry.dtype_str, params)
                    entry.dtype = _resolve_torch_dtype(entry.dtype_str, params)
                    entry.block_shape = (
                        tuple(params["block_shape"]) if params["block_shape"] else None
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Skipping {model_path} tp={tp} ep={ep}: {e}")

    return entries


def _tune_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)

    while True:
        task = task_queue.get()
        if task is None:
            break
        key, batch_size, entry, config_chunk = task

        best_config = None
        best_time = float("inf")
        logger.debug("GPU %d: tuning bs=%d, %d configs", gpu_id, batch_size, len(config_chunk))
        for idx, config in enumerate(config_chunk):
            # Skip configs that are known to be incompatible with block quantization
            if not is_config_compatible(
                config, entry, list(entry.block_shape) if entry.block_shape else None
            ):
                continue
            try:
                kernel_time = benchmark_config(
                    config,
                    batch_size,
                    entry.num_experts,
                    entry.shard_intermediate_size,
                    entry.hidden_size,
                    entry.topk,
                    entry.dtype,
                    entry.use_fp8,
                    entry.use_int8,
                    entry.use_int8a16,
                    entry.use_int4,
                    entry.per_channel_quant,
                    list(entry.block_shape) if entry.block_shape else None,
                    entry.num_fused_shared_experts,
                    num_iters=10,
                )
            except (triton.runtime.autotuner.OutOfResources, RuntimeError, AssertionError):
                # Silently skip invalid or unsupported configs
                continue

            if kernel_time < best_time:
                best_time = kernel_time
                best_config = config

            if (idx + 1) % max(1, len(config_chunk) // 10) == 0:
                logger.debug("GPU %d: processed %d/%d configs", gpu_id, idx + 1, len(config_chunk))

        result_queue.put((key, batch_size, best_config, best_time))

    result_queue.put(None)


def run_tuning(entries: List[ModelEntry], batch_sizes: List[int], args: argparse.Namespace) -> None:
    key_to_entry = {}
    for e in entries:
        key_to_entry[e.unique_key] = e

    all_configs = get_configs_compute_bound()
    first_block_shape = next((e.block_shape for e in entries if e.block_shape is not None), None)
    if first_block_shape is not None:
        block_k = first_block_shape[1]
        all_configs = [c for c in all_configs if block_k % c["BLOCK_SIZE_K"] == 0]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found")
    logger.info(
        "Tuning mode: %d unique kernel shapes, batch sizes %s", len(key_to_entry), batch_sizes
    )
    logger.info("Search space size: %d configs, using %d GPUs", len(all_configs), num_gpus)

    # Create shared task queue and result queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    total_tasks = 0

    # Enqueue all (key, bs, entry, config_chunk) tasks
    chunk_size = max(1, len(all_configs) // num_gpus)  # Keep chunk size moderate
    for key, entry in key_to_entry.items():
        for bs in batch_sizes:
            # Split all_configs into chunks
            chunks = [
                all_configs[i : i + chunk_size] for i in range(0, len(all_configs), chunk_size)
            ]
            for chunk in chunks:
                task_queue.put((key, bs, entry, chunk))
                total_tasks += 1

    # Place sentinel values (one per worker) to signal termination
    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=_tune_worker, args=(gpu_id, task_queue, result_queue, args.seed))
        p.start()
        workers.append(p)

    temp_results: Dict[Tuple, Dict[int, List[Tuple[BenchmarkConfig, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    active_workers = num_gpus
    with tqdm(total=total_tasks, desc="Tuning overall", unit="chunk") as pbar:
        while active_workers > 0:
            res = result_queue.get()
            if res is None:
                active_workers -= 1
                continue
            key, bs, best_config, best_time = res
            if best_config is not None:
                temp_results[key][bs].append((best_config, best_time))
            pbar.update(1)

    for p in workers:
        p.join()

    # Merge best per (key, batch_size)
    final_configs = {}
    for key, bs_dict in temp_results.items():
        for bs, candidates in bs_dict.items():
            best_cfg, _ = min(candidates, key=lambda x: x[1])
            final_configs.setdefault(key, {})[bs] = best_cfg

    # Save configs
    for key, bs_to_config in final_configs.items():
        entry = key_to_entry[key]
        dtype_str = get_config_dtype_str(
            entry.dtype,
            use_int8_w8a16=entry.use_int8a16,
            use_fp8_w8a8=entry.use_fp8,
            use_int4_w4a16=entry.use_int4,
        )
        filename = get_config_filename(
            entry.num_experts,
            entry.shard_intermediate_size,
            entry.hidden_size,
            entry.topk,
            dtype_str,
            entry.use_fp8,
            entry.use_int8,
            entry.use_int8a16,
            entry.use_int4,
            entry.per_channel_quant,
            entry.block_shape,
        )
        sorted_batches = sorted(bs_to_config.keys())
        best_configs = {bs: sort_config(bs_to_config[bs]) for bs in sorted_batches}

        default_config_dir = os.path.dirname(os.path.realpath(__file__))
        config_dir = os.environ.get("SGLANG_MOE_CONFIG_DIR", default_config_dir)
        triton_version = triton.__version__
        version_dir = f"triton_{triton_version.replace('.', '_')}"
        config_dir = os.path.join(config_dir, "configs", version_dir)
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, filename)
        save_configs(best_configs, config_path)
        logger.info("Saved best configs for key %s to %s", key, config_path)


def _benchmark_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    torch.cuda.set_device(gpu_id)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)

    while True:
        task = task_queue.get()
        if task is None:
            break
        entry, batch_size = task

        try:
            dtype_str = get_config_dtype_str(
                entry.dtype,
                use_int8_w8a16=entry.use_int8a16,
                use_fp8_w8a8=entry.use_fp8,
                use_int4_w4a16=entry.use_int4,
            )
            block_n = entry.block_shape[0] if entry.block_shape else 0
            block_k = entry.block_shape[1] if entry.block_shape else 0
            N = entry.shard_intermediate_size // 2
            if entry.use_int4:
                N = N // 2
            op_config = get_moe_configs(
                entry.num_experts, N, dtype_str, block_n, block_k, entry.per_channel_quant
            )
            if op_config is None:
                config = get_default_config(
                    batch_size,
                    entry.num_experts,
                    entry.shard_intermediate_size,
                    entry.hidden_size,
                    entry.topk,
                    dtype_str,
                    False,
                    list(entry.block_shape) if entry.block_shape else None,
                )
            else:
                closest_bs = min(op_config.keys(), key=lambda x: abs(x - batch_size))
                config = op_config[closest_bs]

            if not is_config_compatible(
                config, entry, list(entry.block_shape) if entry.block_shape else None
            ):
                result_queue.put(
                    (entry, batch_size, float("inf"), "Incompatible config (divisibility)")
                )
                continue

            kernel_time = benchmark_config(
                config,
                batch_size,
                entry.num_experts,
                entry.shard_intermediate_size,
                entry.hidden_size,
                entry.topk,
                entry.dtype,
                entry.use_fp8,
                entry.use_int8,
                entry.use_int8a16,
                entry.use_int4,
                entry.per_channel_quant,
                list(entry.block_shape) if entry.block_shape else None,
                entry.num_fused_shared_experts,
            )
            result_queue.put((entry, batch_size, kernel_time, None))
        except Exception as e:
            logger.exception(
                f"Benchmark failed for {entry.path} tp={entry.tp_size} ep={entry.ep_size} bs={batch_size}"
            )
            result_queue.put((entry, batch_size, float("inf"), str(e)))

    result_queue.put(None)


def run_benchmark(
    entries: List[ModelEntry], batch_sizes: List[int], args: argparse.Namespace
) -> None:
    # Build task list
    tasks = []
    for e in entries:
        for bs in batch_sizes:
            tasks.append((e, bs))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices found")
    logger.info("Benchmark mode: %d model configs, batch sizes %s", len(entries), batch_sizes)
    logger.info("Using %d GPUs", num_gpus)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(num_gpus):
        task_queue.put(None)

    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=_benchmark_worker, args=(gpu_id, task_queue, result_queue, args.seed))
        p.start()
        workers.append(p)

    # Collect results
    results: Dict[Tuple[str, int, int], Dict[int, Tuple[float, Optional[str]]]] = {}
    active_workers = num_gpus
    with tqdm(total=len(tasks), desc="Benchmarking overall", unit="task") as pbar:
        while active_workers > 0:
            res = result_queue.get()
            if res is None:
                active_workers -= 1
                continue
            entry, bs, time_us, error_msg = res
            key = (entry.path, entry.tp_size, entry.ep_size)
            if key not in results:
                results[key] = {}
            results[key][bs] = (time_us, error_msg)
            pbar.update(1)

    for p in workers:
        p.join()

    # Convert to JSON-serializable structure
    output_data = []
    for (path, tp, ep), bs_dict in results.items():
        entry_json = {"model": path, "tp_size": tp, "ep_size": ep, "benchmarks": {}}
        for bs, (time_us, error_msg) in sorted(bs_dict.items()):
            if error_msg is not None:
                entry_json["benchmarks"][str(bs)] = {"status": "failed", "error": error_msg}
            else:
                entry_json["benchmarks"][str(bs)] = {"time_us": time_us, "status": "success"}
        output_data.append(entry_json)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Benchmark results saved to %s", args.output)


def main(args: argparse.Namespace) -> None:
    mp.set_start_method("spawn", force=True)

    entries = build_model_entries(args)

    # Validate and print warnings about model and ep/tp configuration
    validate_and_log_entries(entries)

    if args.batch_size is None:
        batch_sizes = get_default_batch_sizes()
        batch_sizes.sort(reverse=True)
    else:
        batch_sizes = []
        for bs_str in args.batch_size:
            batch_sizes.extend(int(x) for x in bs_str.split(","))
        batch_sizes = sorted(set(batch_sizes))

    if args.tune:
        run_tuning(entries, batch_sizes, args)
    else:
        run_benchmark(entries, batch_sizes, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune or benchmark MoE kernels.")
    parser.add_argument(
        "--config",
        type=str,
        default="./ci/models.json",
        help="Path to JSON config file (used if --model not given).",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Single model path (overrides --config)."
    )
    parser.add_argument(
        "--topk-ids-dir", type=str, default=None, help="(Unused) Directory containing topk ids."
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        action="append",
        help="Batch size(s), e.g., --batch-size 1,2,4 or --batch-size 8 --batch-size 16",
    )
    parser.add_argument("--tune", action="store_true", help="Run tuning (search for best configs).")
    parser.add_argument("--tp-size", "--tp", type=int, default=2, help="Tensor parallelism size.")
    parser.add_argument("--ep-size", "--ep", type=int, default=1, help="Expert parallelism size.")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"],
        default="auto",
        help="Quantization dtype.",
    )
    parser.add_argument("--per-channel-quant", action="store_true", help="Use per‑channel scaling.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--disable-shared-experts-fusion",
        action="store_true",
        help="Disable shared experts fusion.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./benchmarks/results.json",
        help="Path to output JSON file with benchmark results (only for non‑tune mode).",
    )
    args = parser.parse_args()

    main(args)
