import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torchada.triton.autotune.fused_moe.utils import (
    calculate_shard_intermediate_size,
    get_config_filename,
    get_model_config,
    infer_moe_activation,
)


def test_nemotron_relu2_uses_non_gated_layout():
    config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        mlp_hidden_act="relu2",
    )

    assert infer_moe_activation(config) == ("relu2_no_mul", False)
    assert calculate_shard_intermediate_size(1856, tp_size=1, is_gated=False) == 1856


def test_other_models_keep_historical_gated_layout():
    config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"], hidden_act="silu")

    assert infer_moe_activation(config) == ("silu", True)
    assert calculate_shard_intermediate_size(512, tp_size=2) == 512


def test_nemotron_model_config_exposes_projection_metadata(monkeypatch):
    config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        mlp_hidden_act="relu2",
        hidden_size=6144,
        moe_latent_size=2688,
        n_routed_experts=128,
        num_experts_per_tok=6,
        moe_intermediate_size=1856,
        torch_dtype=torch.bfloat16,
    )
    monkeypatch.setattr(
        "torchada.triton.autotune.fused_moe.utils._load_model_config",
        lambda _: config,
    )

    params = get_model_config("nemotron", tp_size=1)

    assert params["num_experts"] == 128
    assert params["topk"] == 6
    assert params["hidden_size"] == 2688
    assert params["shard_intermediate_size"] == 1856
    assert params["activation"] == "relu2_no_mul"
    assert params["is_gated"] is False


def test_config_filename_uses_second_gemm_width():
    gated = get_config_filename(
        128,
        3712,
        2688,
        6,
        torch.bfloat16,
        False,
        False,
        False,
        False,
        False,
        None,
    )
    non_gated = get_config_filename(
        128,
        1856,
        2688,
        6,
        torch.bfloat16,
        False,
        False,
        False,
        False,
        False,
        None,
        is_gated=False,
    )

    assert "E=128,N=1856" in gated
    assert "E=128,N=1856" in non_gated


@pytest.mark.parametrize("is_gated,expected_w2", [(True, 1856), (False, 1856)])
def test_benchmark_allocates_matching_second_projection(monkeypatch, is_gated, expected_w2):
    """w2's K dimension follows the actual post-activation width."""
    from torchada.triton.autotune.fused_moe import tune_moe

    seen = []

    def fake_randn(*shape, **kwargs):
        seen.append(tuple(shape))
        # Stop after x, w1 and w2 allocation; this test only checks shapes.
        if len(seen) == 3:
            raise RuntimeError("stop after expert weights")
        return torch.empty((1,), dtype=kwargs.get("dtype", torch.float32))

    monkeypatch.setattr(tune_moe.torch, "set_default_device", lambda *_: None)
    monkeypatch.setattr(tune_moe.torch, "randn", fake_randn)
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 1,
        "num_stages": 1,
    }

    with pytest.raises(RuntimeError, match="stop after expert weights"):
        tune_moe.benchmark_config(
            config,
            1,
            128,
            3712 if is_gated else 1856,
            2688,
            6,
            torch.bfloat16,
            False,
            False,
            False,
            False,
            False,
            activation="silu" if is_gated else "relu2_no_mul",
            is_gated=is_gated,
        )

    assert seen[1] == (128, 3712 if is_gated else 1856, 2688)
    assert seen[2] == (128, 2688, expected_w2)


def test_s5000_nemotron_config_adds_decode_mtp_bucket():
    path = (
        Path(__file__).parents[1]
        / "src/torchada/triton/autotune/fused_moe/configs/triton_3_2_0"
        / "E=128,N=1856,device_name=MTT_S5000.json"
    )
    config = json.loads(path.read_text())

    assert set(config) == {"16", "49", "80", "113", "144", "511", "512", "513"}
    assert config["16"] == {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 16,
        "SPLIT_K": 1,
        "num_warps": 8,
        "num_stages": 2,
    }
    assert config["49"] == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 4,
    }
    assert config["113"] == {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 3,
    }
    assert config["512"] == config["16"]
    assert config["513"]["BLOCK_SIZE_M"] == 128


def test_s5000_nemotron_config_routes_runtime_m_buckets(monkeypatch):
    from torchada.triton.runtime.fused_moe import config as moe_config

    measured = {
        16: {"BLOCK_SIZE_M": 16},
        49: {"BLOCK_SIZE_M": 49},
        80: {"BLOCK_SIZE_M": 80},
        113: {"BLOCK_SIZE_M": 113},
        144: {"BLOCK_SIZE_M": 144},
        511: {"BLOCK_SIZE_M": 511},
        512: {"BLOCK_SIZE_M": 512},
        513: {"BLOCK_SIZE_M": 513},
    }
    fallback = {"BLOCK_SIZE_M": 999}
    monkeypatch.setattr(moe_config, "get_config", lambda: None)
    monkeypatch.setattr(moe_config, "get_moe_configs", lambda *args, **kwargs: measured)
    monkeypatch.setattr(moe_config, "get_default_config", lambda *args, **kwargs: fallback)

    def resolve(M):
        return moe_config.try_get_optimal_moe_config(
            (128, 2688, 1856),
            (128, 2688, 1856),
            6,
            "bf16",
            M,
        )

    assert resolve(115) == measured[113]
    assert resolve(120) == measured[113]
    assert resolve(512) == measured[512]
    assert resolve(514) == fallback


def test_config_map_falls_back_for_unmeasured_large_m(monkeypatch):
    from torchada.triton.runtime.fused_moe import config as moe_config

    measured = {
        1: {"BLOCK_SIZE_M": 16},
        16: {"BLOCK_SIZE_M": 32},
    }
    fallback = {"BLOCK_SIZE_M": 999}
    monkeypatch.setattr(moe_config, "get_config", lambda: None)
    monkeypatch.setattr(moe_config, "get_moe_configs", lambda *args, **kwargs: measured)
    monkeypatch.setattr(moe_config, "get_default_config", lambda *args, **kwargs: fallback)

    assert (
        moe_config.try_get_optimal_moe_config(
            (128, 1856, 2688),
            (128, 1856, 2688),
            6,
            "bf16",
            16,
        )
        == measured[16]
    )
    assert (
        moe_config.try_get_optimal_moe_config(
            (128, 1856, 2688),
            (128, 1856, 2688),
            6,
            "bf16",
            4096,
        )
        == fallback
    )
