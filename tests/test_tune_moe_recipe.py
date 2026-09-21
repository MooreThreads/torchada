"""The tuning recipe is what makes a shipped table row reproducible.

These tests cover the pieces a re-derivation needs: a recipe entry that pins its bucket set,
shape entries that describe a kernel without a model checkout, the merge behaviour that keeps the
rows a run did not measure, and the constexpr map the tuned rows are written with.
"""

import argparse
import json
import queue
import shutil
from pathlib import Path

import pytest
import torch

from torchada.triton.autotune.fused_moe import tune_moe
from torchada.triton.autotune.fused_moe.utils import (
    get_config_filename,
    get_configs_compute_bound,
    merge_configs,
    sort_config,
)
from torchada.triton.runtime.fused_moe.config import get_config_dtype_str

REPO = Path(__file__).parents[1]
RECIPE_PATH = REPO / "src/torchada/triton/autotune/fused_moe/ci/shapes.json"
SHIPPED_TABLE = (
    REPO
    / "src/torchada/triton/autotune/fused_moe/configs/triton_3_2_0"
    / "E=128,N=1856,device_name=MTT_S5000.json"
)

TINY_M_ROW = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 1,
}


def _args(**overrides):
    defaults = dict(tp_size=1, ep_size=1, dtype="auto", per_channel_quant=False)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _filename_for(entry):
    dtype_str = get_config_dtype_str(
        entry.dtype,
        use_int8_w8a16=entry.use_int8a16,
        use_fp8_w8a8=entry.use_fp8,
        use_int4_w4a16=entry.use_int4,
    )
    return get_config_filename(
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
        is_gated=entry.is_gated,
    )


def test_recipe_batch_sizes_accepts_int_list_and_rejects_junk():
    assert tune_moe._recipe_batch_sizes({}) is None
    assert tune_moe._recipe_batch_sizes({"batch_sizes": 16}) == [16]
    assert tune_moe._recipe_batch_sizes({"batch_sizes": [16, 16, 7]}) == [7, 16]
    for bad in (0, -1, True, "16", [1, "2"], [None]):
        with pytest.raises(ValueError):
            tune_moe._recipe_batch_sizes({"batch_sizes": bad})


def test_shape_entry_describes_the_kernel_without_a_model():
    raw = {
        "name": "recipe-shape",
        "num_experts": 128,
        "hidden_size": 2688,
        "shard_intermediate_size": 1856,
        "topk": 6,
        "is_gated": False,
        "activation": "relu2_no_mul",
        "dtype": "bf16",
        "batch_sizes": [16],
    }

    entry = tune_moe._entry_from_shape(raw, _args(), tune_moe._recipe_batch_sizes(raw))

    assert entry.num_experts == 128
    assert entry.hidden_size == 2688
    assert entry.shard_intermediate_size == 1856
    assert entry.topk == 6
    assert entry.is_gated is False
    assert entry.activation == "relu2_no_mul"
    assert entry.dtype == torch.bfloat16
    assert entry.batch_sizes == [16]
    assert entry.path == "recipe-shape"


def test_shape_entry_requires_the_shape_fields():
    with pytest.raises(ValueError, match="missing"):
        tune_moe._entry_from_shape({"num_experts": 128}, _args(), None)


def test_shape_entry_rejects_dtype_auto():
    raw = {
        "num_experts": 128,
        "hidden_size": 2688,
        "shard_intermediate_size": 1856,
        "topk": 6,
        "is_gated": False,
        "dtype": "auto",
    }
    with pytest.raises(ValueError, match="auto"):
        tune_moe._entry_from_shape(raw, _args(), None)


def test_checked_in_recipe_targets_the_shipped_table():
    recipe = json.loads(RECIPE_PATH.read_text())
    entry = tune_moe._entry_from_shape(
        recipe[0], _args(), tune_moe._recipe_batch_sizes(recipe[0])
    )

    # The recipe has to resolve to the file the row was shipped in...
    assert _filename_for(entry) == SHIPPED_TABLE.name
    assert SHIPPED_TABLE.exists()
    # ...and to the tiny-M bucket it was derived for.
    assert entry.batch_sizes == [16]
    assert json.loads(SHIPPED_TABLE.read_text())["16"] == TINY_M_ROW


def test_merge_configs_keeps_unmeasured_rows(tmp_path):
    path = tmp_path / "table.json"
    shipped_row = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "SPLIT_K": 1}
    path.write_text(json.dumps({"49": shipped_row, "512": shipped_row}))

    merged = merge_configs({16: TINY_M_ROW}, str(path))

    assert sorted(merged) == [16, 49, 512]
    assert merged[16] == TINY_M_ROW
    assert merged[512] == shipped_row
    # A re-derived bucket must not take the other rows with it.
    assert json.loads(path.read_text()).keys() == {"49", "512"}


def test_merge_configs_on_absent_file_returns_the_tuned_rows(tmp_path):
    merged = merge_configs({16: TINY_M_ROW}, str(tmp_path / "missing.json"))
    assert merged == {16: TINY_M_ROW}


def test_pinned_rows_require_a_source():
    with pytest.raises(ValueError, match="source"):
        tune_moe._recipe_pinned_rows({"pinned_rows": {"16": {"config": TINY_M_ROW}}})


def test_pinned_rows_reject_incomplete_configs():
    spec = {"config": {"BLOCK_SIZE_M": 16}, "source": "measured"}
    with pytest.raises(ValueError, match="missing"):
        tune_moe._recipe_pinned_rows({"pinned_rows": {"16": spec}})


def test_materialize_reproduces_the_shipped_row(tmp_path, monkeypatch):
    """The shipped row is a function of the recipe: materialize the table and compare it."""

    recipe = json.loads(RECIPE_PATH.read_text())
    entry = tune_moe._entry_from_shape(recipe[0], _args(), tune_moe._recipe_batch_sizes(recipe[0]))
    assert entry.pinned_rows

    monkeypatch.setenv("SGLANG_MOE_CONFIG_DIR", str(tmp_path))
    target = Path(tune_moe.config_dir()) / tune_moe.config_filename(entry)
    assert target.name == SHIPPED_TABLE.name
    # Seed the table the way it ships, then let the recipe write its pinned row into it.
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SHIPPED_TABLE, target)

    written = tune_moe.materialize_pinned_rows([entry], argparse.Namespace(merge_configs=True))

    assert written == 1
    produced = Path(tune_moe.config_dir()) / tune_moe.config_filename(entry)
    assert produced.name == SHIPPED_TABLE.name

    got = json.loads(produced.read_text())
    want = json.loads(SHIPPED_TABLE.read_text())
    # Compared on content, not bytes: the shipped file is hand-formatted (one row per line) while
    # the tool writes the indented JSON style the other tables use.
    assert list(got) == list(want)
    for bucket in want:
        assert list(got[bucket]) == list(want[bucket])
    assert got == want
    assert got["16"] == TINY_M_ROW


def test_grid_candidates_survive_sort_config():
    configs = get_configs_compute_bound()

    assert configs
    # sort_config is what turns a measurement into a row; it must not drop or reorder what the
    # kernel is launched with.
    assert all(sort_config(c) == c for c in configs)
    tiny_m = [c for c in configs if c["BLOCK_SIZE_M"] == 16 and c["BLOCK_SIZE_N"] == 64]
    assert tiny_m and all(c["num_stages"] == 1 for c in tiny_m)


def test_worker_skips_an_unsupported_candidate(monkeypatch):
    """One candidate the kernel cannot run must not take the tuning run down with it."""

    entry = tune_moe._entry_from_shape(
        {
            "num_experts": 128,
            "hidden_size": 2688,
            "shard_intermediate_size": 1856,
            "topk": 6,
            "is_gated": False,
            "activation": "relu2_no_mul",
            "dtype": "bf16",
        },
        _args(),
        [16],
    )
    unsupported = dict(TINY_M_ROW, num_warps=8)
    good = dict(TINY_M_ROW)

    def fake_benchmark(config, *args, **kwargs):
        if config["num_warps"] == 8:
            # What Triton raises for a keyword its kernel does not declare.
            raise KeyError("Keyword argument SPLIT_K was specified but unrecognised")
        return 42.0

    monkeypatch.setattr(tune_moe, "benchmark_config", fake_benchmark)
    monkeypatch.setattr(tune_moe.torch.cuda, "set_device", lambda *a, **k: None)
    monkeypatch.setattr(tune_moe.torch.cuda, "manual_seed_all", lambda *a, **k: None)

    tasks, results = queue.Queue(), queue.Queue()
    tasks.put(("key", 16, entry, [unsupported, good]))
    tasks.put(None)
    tune_moe._tune_worker(0, tasks, results, seed=0)

    key, batch_size, best_config, best_time = results.get_nowait()
    assert results.get_nowait() is None
    assert best_config == good
    assert best_time == 42.0


def test_dead_worker_aborts_instead_of_hanging(monkeypatch):
    """A worker that dies leaves the parent waiting forever unless it checks liveness."""

    entry = tune_moe._entry_from_shape(
        {
            "num_experts": 128,
            "hidden_size": 2688,
            "shard_intermediate_size": 1856,
            "topk": 6,
            "is_gated": False,
            "activation": "relu2_no_mul",
            "dtype": "bf16",
        },
        _args(),
        [16],
    )

    class DeadProcess:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

    class SilentResults(queue.Queue):
        def get(self, block=True, timeout=None):
            raise queue.Empty

    monkeypatch.setattr(tune_moe.mp, "Process", DeadProcess)
    monkeypatch.setattr(tune_moe.mp, "Queue", SilentResults)
    monkeypatch.setattr(tune_moe.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(tune_moe, "get_configs_compute_bound", lambda: [dict(TINY_M_ROW)])

    with pytest.raises(RuntimeError, match="all workers died"):
        tune_moe.run_tuning(
            [entry], [16], argparse.Namespace(seed=0, merge_configs=False)
        )
