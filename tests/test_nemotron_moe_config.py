from types import SimpleNamespace

import torch

from torchada.triton.autotune.fused_moe.utils import (
    calculate_shard_intermediate_size,
    get_config_filename,
    infer_moe_activation,
)


def test_nemotron_relu2_uses_non_gated_layout():
    config = SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        mlp_hidden_act="relu2",
    )
    assert infer_moe_activation(config) == ("relu2_no_mul", False)
    assert calculate_shard_intermediate_size(1856, tp_size=1, is_gated=False) == 1856


def test_other_models_keep_gated_layout():
    config = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"], hidden_act="silu")
    assert infer_moe_activation(config) == ("silu", True)
    assert calculate_shard_intermediate_size(512, tp_size=2) == 512


def test_nemotron_config_filename_uses_w2_width():
    gated = get_config_filename(
        128, 3712, 2688, 6, torch.bfloat16, False, False, False, False, False, None
    )
    nongated = get_config_filename(
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
    assert "E=128,N=1856" in nongated


def test_nemotron_benchmark_uses_single_projection_width(monkeypatch):
    """The BF16 Nemotron benchmark must allocate w2 with N, rather than N/2."""
    from torchada.triton.autotune.fused_moe import tune_moe

    seen = []

    def fake_randn(*shape, **kwargs):
        seen.append(tuple(shape))
        # Stop before the benchmark starts routing/timing.  This keeps the
        # shape assertion CPU-only and avoids allocating the full checkpoint
        # dimensions.
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
    try:
        tune_moe.benchmark_config(
            config,
            1,
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
            activation="relu2_no_mul",
            is_gated=False,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after expert weights"
    assert seen[1] == (128, 1856, 2688)
    assert seen[2] == (128, 2688, 1856)
