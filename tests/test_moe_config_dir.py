"""Tests for the fused_moe default config dir resolution (vLLM env redirect)."""

import importlib.metadata
import json
import os

import pytest

from torchada.triton.autotune import fused_moe


SPACED = "E=128,N=1536,device_name=MTT_S5000,dtype=fp8_w8a8,block_shape=[128, 128].json"
SPACE_FREE = SPACED.replace(" ", "")
PLAIN = "E=32,N=768,device_name=MTT_S5000.json"


@pytest.fixture
def config_tree(tmp_path):
    """A fake torchada fused_moe package dir with one triton config subdir."""
    triton_dir = tmp_path / "configs" / "triton_3_2_0"
    triton_dir.mkdir(parents=True)
    for name in (SPACED, PLAIN):
        (triton_dir / name).write_text(json.dumps({"1": {"BLOCK_SIZE_M": 16}}))
    return tmp_path


def test_resolves_triton_subdir_and_aliases(config_tree):
    resolved = fused_moe._vllm_tuned_config_dir(str(config_tree))
    assert resolved == str(config_tree / "configs" / "triton_3_2_0")
    # vLLM's flat, space-free lookup must now hit both configs.
    assert os.path.exists(os.path.join(resolved, SPACE_FREE))
    assert os.path.exists(os.path.join(resolved, PLAIN))
    # SGLang's spaced lookup is untouched.
    assert os.path.exists(os.path.join(resolved, SPACED))


def test_alias_is_idempotent(config_tree):
    first = fused_moe._vllm_tuned_config_dir(str(config_tree))
    before = sorted(os.listdir(first))
    second = fused_moe._vllm_tuned_config_dir(str(config_tree))
    assert first == second
    assert sorted(os.listdir(second)) == before


def test_alias_content_matches_source(config_tree):
    resolved = fused_moe._vllm_tuned_config_dir(str(config_tree))
    with open(os.path.join(resolved, SPACE_FREE)) as f:
        assert json.load(f) == {"1": {"BLOCK_SIZE_M": 16}}


def test_exact_triton_version_dir_wins(tmp_path, monkeypatch):
    for ver in ("triton_3_1_0", "triton_3_2_0"):
        (tmp_path / "configs" / ver).mkdir(parents=True)
    monkeypatch.setattr(fused_moe, "_installed_triton_version", lambda: "3.1.0")
    resolved = fused_moe._vllm_tuned_config_dir(str(tmp_path))
    assert resolved == str(tmp_path / "configs" / "triton_3_1_0")


@pytest.mark.parametrize(
    "raw",
    ["3.2.0", "3.2.0.post1", "3.2.0rc1", "3.2.0+git9d8d5e91", "3.2.0.dev20260601"],
)
def test_pep440_variants_match_release_dir(tmp_path, monkeypatch, raw):
    """post/rc/dev/local-version suffixes must still exact-match triton_3_2_0
    (not silently fall back to the newest directory)."""
    for ver in ("triton_3_1_0", "triton_3_2_0"):
        (tmp_path / "configs" / ver).mkdir(parents=True)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: raw)
    resolved = fused_moe._vllm_tuned_config_dir(str(tmp_path))
    assert resolved == str(tmp_path / "configs" / "triton_3_2_0")


def test_newest_dir_fallback_when_version_unknown(tmp_path, monkeypatch):
    for ver in ("triton_3_1_0", "triton_3_2_0"):
        (tmp_path / "configs" / ver).mkdir(parents=True)
    monkeypatch.setattr(fused_moe, "_installed_triton_version", lambda: None)
    resolved = fused_moe._vllm_tuned_config_dir(str(tmp_path))
    assert resolved == str(tmp_path / "configs" / "triton_3_2_0")


def test_newest_dir_fallback_when_no_exact_match(tmp_path, monkeypatch):
    for ver in ("triton_3_1_0", "triton_3_2_0"):
        (tmp_path / "configs" / ver).mkdir(parents=True)
    monkeypatch.setattr(fused_moe, "_installed_triton_version", lambda: "9.9.9")
    resolved = fused_moe._vllm_tuned_config_dir(str(tmp_path))
    assert resolved == str(tmp_path / "configs" / "triton_3_2_0")


def test_missing_configs_root_returns_none(tmp_path):
    assert fused_moe._vllm_tuned_config_dir(str(tmp_path)) is None


def test_env_redirect_set_at_import():
    """`import torchada` (done by conftest) must leave the env pointing at a
    real directory whose space-containing configs all have space-free twins."""
    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
    assert folder
    if os.path.dirname(fused_moe.__file__) not in folder:
        pytest.skip("VLLM_TUNED_CONFIG_FOLDER overridden by the environment")
    assert os.path.isdir(folder)
    spaced = [n for n in os.listdir(folder) if " " in n and n.endswith(".json")]
    for name in spaced:
        assert os.path.exists(os.path.join(folder, name.replace(" ", "")))
