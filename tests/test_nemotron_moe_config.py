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
