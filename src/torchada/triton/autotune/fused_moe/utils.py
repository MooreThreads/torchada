import json
import os
from typing import Dict, List, TypedDict

import torch
from transformers import AutoConfig

# Register SGLang custom configs, e.g. qwen3_5_moe, when SGLang is available.
try:
    import sglang.srt.utils.hf_transformers_utils  # noqa: F401
except ImportError:
    pass

from torchada.triton.runtime.fused_moe.config import get_config_dtype_str, get_config_file_name


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def calculate_shard_intermediate_size(
    intermediate_size: int, tp_size: int, ep_size: int = 1
) -> int:
    assert tp_size % ep_size == 0
    moe_tp_size = tp_size // ep_size
    assert intermediate_size % moe_tp_size == 0
    return 2 * intermediate_size // moe_tp_size


def get_num_shared_experts(config, disable_shared_experts_fusion: bool) -> int:
    if disable_shared_experts_fusion:
        return 0
    if getattr(config, "shared_expert_intermediate_size", None) is not None:
        return 1
    return getattr(config, "n_shared_experts", 0) or getattr(config, "num_shared_experts", 0) or 0


def infer_quant_dtype_str(config) -> str:
    quant_config = getattr(config, "quantization_config", None) or {}
    if not quant_config:
        return "auto"

    quant_method = str(quant_config.get("quant_method", "")).lower()
    if "fp8" in quant_method or quant_config.get("weight_block_size") is not None:
        return "fp8_w8a8"

    config_groups = quant_config.get("config_groups") or {}
    first_group = next(iter(config_groups.values()), {})
    weights_config = first_group.get("weights", {})
    activations_config = (
        first_group.get("input_activations") or first_group.get("activations") or {}
    )
    weight_bits = weights_config.get("num_bits")
    activation_bits = activations_config.get("num_bits")
    if weight_bits == 4:
        return "int4_w4a16"
    if weight_bits == 8 and activation_bits == 8:
        return "int8_w8a8"
    if weight_bits == 8:
        return "int8_w8a16"

    if "int4" in quant_method or "w4a16" in quant_method:
        return "int4_w4a16"
    if "int8" in quant_method or "w8a8" in quant_method:
        return "int8_w8a8" if "w8a8" in quant_method else "int8_w8a16"

    return "auto"


def _load_config_from_modelscope(model_name: str):
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "ModelScope is required to fetch model configs when Hugging Face "
            "AutoConfig cannot load the model."
        ) from exc

    cache_dir = (
        os.environ.get("MODELSCOPE_CACHE")
        or os.environ.get("MODELSCOPE_CACHE_DIR")
        or os.environ.get("MODELSCOPE_CACHE_HOME")
    )
    snapshot_kwargs = {
        "model_id": model_name,
        "allow_file_pattern": [
            "config.json",
            "configuration.json",
            "configuration*.py",
            "generation_config.json",
        ],
        "ignore_file_pattern": [
            "*.bin",
            "*.safetensors",
            "*.pt",
            "*.pth",
            "*.onnx",
            "*.msgpack",
            "*.gguf",
        ],
    }
    if cache_dir:
        snapshot_kwargs["cache_dir"] = cache_dir

    local_path = snapshot_download(**snapshot_kwargs)
    return AutoConfig.from_pretrained(local_path, trust_remote_code=True)


def _load_model_config(model_name: str):
    if os.path.exists(model_name):
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    try:
        return _load_config_from_modelscope(model_name)
    except Exception:
        try:
            return AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        except Exception as hf_error:
            raise RuntimeError(
                f"Failed to load model config for {model_name!r} from both "
                "Hugging Face and ModelScope."
            ) from hf_error


def get_model_config(
    model_name: str,
    tp_size: int,
    ep_size: int = 1,
    disable_shared_experts_fusion: bool = False,
    topk_ids_dir: str = None,
) -> Dict:
    config = _load_model_config(model_name)

    architecture = config.architectures[0]
    quant_dtype_str = infer_quant_dtype_str(config)
    block_shape = None
    if hasattr(config, "quantization_config") and "weight_block_size" in config.quantization_config:
        block_shape = config.quantization_config["weight_block_size"]
        assert len(block_shape) == 2

    if hasattr(config, "quantization_config") and "config_groups" in config.quantization_config:
        config_groups = config.quantization_config["config_groups"]
        # Get group_size from the first group's weights config
        first_group = next(iter(config_groups.values()), {})
        weights_config = first_group.get("weights", {})
        group_size = weights_config.get("group_size")
        block_shape = [0, group_size]
        assert len(block_shape) == 2
    # Replace config with text_config for encoder-decoder models after getting block_shape and architecture
    if hasattr(config, "text_config"):
        config = config.get_text_config()

    hidden_size = config.hidden_size
    num_fused_shared_experts = 0
    if architecture == "DbrxForCausalLM":
        E = config.ffn_config.moe_num_experts // ep_size
        topk = config.ffn_config.moe_top_k
        intermediate_size = config.ffn_config.ffn_hidden_size
    elif architecture == "JambaForCausalLM":
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Qwen2MoeForCausalLM",
        "Qwen3MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ]:
        num_fused_shared_experts = get_num_shared_experts(config, disable_shared_experts_fusion)
        E = config.num_experts // ep_size + num_fused_shared_experts
        topk = config.num_experts_per_tok + num_fused_shared_experts
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "Glm4MoeForCausalLM",
        "GlmMoeDsaForCausalLM",
        "MistralLarge3ForCausalLM",
    ]:
        E = (config.n_routed_experts // ep_size) + (
            0
            if disable_shared_experts_fusion
            or architecture
            not in [
                "DeepseekV3ForCausalLM",
                "DeepseekV32ForCausalLM",
                "Glm4MoeForCausalLM",
                "GlmMoeDsaForCausalLM",
                "MistralLarge3ForCausalLM",
            ]
            else 1
        )
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.moe_intermediate_size
    elif architecture == "Llama4ForConditionalGeneration":
        E = config.num_local_experts // ep_size + (0 if disable_shared_experts_fusion else 1)
        topk = config.num_experts_per_tok + (
            0 if disable_shared_experts_fusion or topk_ids_dir is None else 1
        )
        intermediate_size = config.intermediate_size
    elif architecture in [
        "Grok1ForCausalLM",
        "Grok1ImgGen",
        "Grok1AForCausalLM",
    ]:
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture in [
        "BailingMoEForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
    ]:
        E = config.num_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
    elif architecture == "NemotronHForCausalLM":
        E = config.n_routed_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.moe_intermediate_size
        hidden_size = getattr(config, "moe_latent_size", None) or hidden_size
    elif architecture == "Gemma4ForConditionalGeneration":
        E = config.num_experts // ep_size
        topk = config.top_k_experts
        intermediate_size = config.moe_intermediate_size
    else:
        # Default: Mixtral
        E = config.num_local_experts // ep_size
        topk = config.num_experts_per_tok
        intermediate_size = config.intermediate_size

    shard_intermediate_size = calculate_shard_intermediate_size(intermediate_size, tp_size, ep_size)

    return {
        "num_experts": E,
        "topk": topk,
        "hidden_size": hidden_size,
        "shard_intermediate_size": shard_intermediate_size,
        "dtype": config.torch_dtype,
        "block_shape": block_shape,
        "architecture": architecture,
        "num_fused_shared_experts": num_fused_shared_experts,
        "quant_dtype_str": quant_dtype_str,
    }


def get_configs_compute_bound() -> List[Dict[str, int]]:
    configs: List[BenchmarkConfig] = []
    for num_stages in [1]:
        for block_m in [32, 64, 128]:
            for block_k in [32, 64, 128]:
                for block_n in [32, 64, 128]:
                    for num_warps in [4, 8, 16]:
                        for group_size in [1, 16, 32, 64]:
                            configs.append(
                                {
                                    "BLOCK_SIZE_M": block_m,
                                    "BLOCK_SIZE_N": block_n,
                                    "BLOCK_SIZE_K": block_k,
                                    "GROUP_SIZE_M": group_size,
                                    "num_warps": num_warps,
                                    "num_stages": num_stages,
                                }
                            )
    for num_stages in [1]:
        for block_m, block_n in [(16, 64), (64, 16)]:
            for block_k in [32, 64, 128]:
                for num_warps in [4, 8, 16]:
                    for group_size in [1, 16, 32, 64]:
                        configs.append(
                            {
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "BLOCK_SIZE_K": block_k,
                                "GROUP_SIZE_M": group_size,
                                "num_warps": num_warps,
                                "num_stages": num_stages,
                            }
                        )
    return configs


def sort_config(config: BenchmarkConfig) -> BenchmarkConfig:
    return {
        "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
        **({"waves_per_eu": config["waves_per_eu"]} if "waves_per_eu" in config else {}),
        **({"USE_TMA": config["USE_TMA"]} if "USE_TMA" in config else {}),
    }


def save_configs(
    configs: Dict[int, BenchmarkConfig],
    filename: str,
) -> None:
    print(f"Writing best config to {filename}...")
    with open(filename, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def get_config_filename(
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
    block_shape: List[int],
) -> str:
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int4_w4a16=use_int4_w4a16,
    )

    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    N = shard_intermediate_size // 2
    if use_int4_w4a16:
        N = N // 2

    filename = get_config_file_name(
        num_experts,
        N,
        dtype_str,
        list(block_shape) if block_shape else block_shape,
        per_channel_quant,
    )

    return filename


def get_default_batch_sizes() -> List[int]:
    return [
        1,
        2,
        4,
        8,
        16,
        24,
        32,
        48,
        64,
        96,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
    ]
