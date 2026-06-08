"""Tests for transparent CUDA-graph executable rotation.

The MUSA driver caps live ``musaGraphExec_t`` at ~2048 per process. torchada's
rotation keeps graph templates alive, LRU-caps live executables, and
re-instantiates an evicted graph's executable from its template on replay. The
capture/replay tests run on MUSA hardware only; the env-gating test is pure Python.
"""

import pytest
import torch


def _musa_ready() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def test_is_enabled_respects_env(monkeypatch):
    """Pure-Python: the master switch honors TORCHADA_GRAPH_ROTATION."""
    from torchada import _graph_rotation as rot

    monkeypatch.setenv("TORCHADA_GRAPH_ROTATION", "0")
    assert rot.is_enabled() is False
    monkeypatch.setenv("TORCHADA_GRAPH_ROTATION", "1")
    assert rot.is_enabled() is True


def test_cap_env_parsing(monkeypatch):
    """Pure-Python: cap parsing falls back on bad input and rejects non-positive."""
    from torchada import _graph_rotation as rot

    monkeypatch.setenv("TORCHADA_GRAPH_EXEC_CAP", "1234")
    assert rot._read_cap() == 1234
    monkeypatch.setenv("TORCHADA_GRAPH_EXEC_CAP", "garbage")
    assert rot._read_cap() == rot._DEFAULT_CAP
    monkeypatch.setenv("TORCHADA_GRAPH_EXEC_CAP", "0")
    assert rot._read_cap() == rot._DEFAULT_CAP


def test_resolve_cap_priority(monkeypatch):
    """Pure-Python: explicit cap wins; otherwise default (no probe unless opted in)."""
    from torchada import _graph_rotation as rot

    monkeypatch.setenv("TORCHADA_GRAPH_EXEC_CAP", "1500")
    monkeypatch.delenv("TORCHADA_GRAPH_AUTOPROBE", raising=False)
    assert rot._resolve_cap() == 1500

    monkeypatch.delenv("TORCHADA_GRAPH_EXEC_CAP", raising=False)
    monkeypatch.delenv("TORCHADA_GRAPH_AUTOPROBE", raising=False)
    assert rot._resolve_cap() == rot._DEFAULT_CAP


@pytest.mark.musa
def test_rotation_installed_on_import():
    import torchada  # noqa: F401  (apply_patches installs the rotation)

    if not _musa_ready():
        pytest.skip("MUSA-only test")
    from torchada import _graph_rotation as rot

    assert rot._installed, "rotation should install during torchada.apply_patches()"
    assert rot._rotation is not None


@pytest.mark.musa
def test_rotation_captures_past_cap_and_replays_correctly():
    """Capturing past the cap evicts executables (keeping templates); an evicted
    graph must re-instantiate from its template and replay correctly."""
    import torchada  # noqa: F401

    if not _musa_ready():
        pytest.skip("MUSA-only test")
    from torchada import _graph_rotation as rot

    if not rot._installed or rot._rotation is None:
        pytest.skip("rotation not installed (TORCHADA_GRAPH_ROTATION=0?)")
    r = rot._rotation

    old_cap, old_evicting = r.cap, r._evicting
    r._live.clear()
    r._evicting = False
    for key in r.stats:
        r.stats[key] = 0
    r.cap = 40
    try:
        dev = "musa"
        pool = torch.musa.graph_pool_handle()

        # a correctness graph captured first -> guaranteed evicted by the burst below
        sout = torch.zeros(4, device=dev, dtype=torch.float32)
        const = torch.full((4,), 7.0, device=dev, dtype=torch.float32)
        g0 = torch.musa.MUSAGraph()
        with torch.musa.graph(g0, pool=pool):
            sout.copy_(const)

        # capture well past the cap (40) to force eviction of g0's executable
        inp = torch.randn(64, 64, device=dev, dtype=torch.bfloat16)
        _ = inp + 1.0
        torch.musa.synchronize()
        held = []
        for _ in range(200):
            g = torch.musa.MUSAGraph()
            with torch.musa.graph(g, pool=pool):
                _y = inp + 1.0
            held.append(g)

        snap = r.snapshot()
        assert snap["evict"] > 0, "expected evictions once past the cap"
        assert snap["live"] <= r.cap, "live executables must stay within the cap"

        # g0 was evicted long ago; replay must auto-reinstantiate from its template
        sout.zero_()
        g0.replay()
        torch.musa.synchronize()
        assert sout.tolist() == [7.0, 7.0, 7.0, 7.0], "evicted graph replayed incorrectly"
        assert r.snapshot()["reinstantiate"] > 0, "expected a re-instantiation on replay"
    finally:
        r.cap, r._evicting = old_cap, old_evicting
        r._live.clear()


@pytest.mark.musa
def test_probe_live_exec_limit_isolated():
    """The auto-probe measures a positive driver limit in a throwaway subprocess
    and leaves the caller's RNG generator uncorrupted (the in-process probe would
    raise 'Offset increment outside graph capture' on later RNG)."""
    import torchada  # noqa: F401

    if not _musa_ready():
        pytest.skip("MUSA-only test")
    from torchada import _graph_rotation as rot

    limit = rot._probe_live_exec_limit(max_probe=256)
    assert limit is None or (isinstance(limit, int) and limit > 0)

    # RNG inside a captured graph, then RNG outside -> raises if the probe corrupted
    # this process's generator. Subprocess isolation must prevent that.
    pool = torch.musa.graph_pool_handle()
    out = torch.empty(8, device="musa", dtype=torch.float32)
    g = torch.musa.MUSAGraph()
    with torch.musa.graph(g, pool=pool):
        out.copy_(torch.randn(8, device="musa", dtype=torch.float32))
    g.replay()
    torch.musa.synchronize()
    _ = torch.randn(8, device="musa", dtype=torch.float32)
    torch.musa.synchronize()


@pytest.mark.musa
def test_shallow_workload_never_evicts():
    """Under-cap capture must not evict (and the replay fast path stays correct)."""
    import torchada  # noqa: F401

    if not _musa_ready():
        pytest.skip("MUSA-only test")
    from torchada import _graph_rotation as rot

    if not rot._installed or rot._rotation is None:
        pytest.skip("rotation not installed")
    r = rot._rotation

    old_cap = r.cap
    r._live.clear()
    evict0 = r.stats["evict"]
    r.cap = 100_000  # effectively unbounded -> never evict
    try:
        dev = "musa"
        pool = torch.musa.graph_pool_handle()
        sout = torch.zeros(4, device=dev, dtype=torch.float32)
        const = torch.full((4,), 3.0, device=dev, dtype=torch.float32)
        g = torch.musa.MUSAGraph()
        with torch.musa.graph(g, pool=pool):
            sout.copy_(const)
        held = [g]
        inp = torch.randn(16, 16, device=dev, dtype=torch.bfloat16)
        _ = inp + 1.0
        torch.musa.synchronize()
        for _ in range(50):
            gx = torch.musa.MUSAGraph()
            with torch.musa.graph(gx, pool=pool):
                _y = inp + 1.0
            held.append(gx)

        assert r.stats["evict"] == evict0, "no eviction expected under the cap"
        sout.zero_()
        g.replay()
        torch.musa.synchronize()
        assert sout.tolist() == [3.0, 3.0, 3.0, 3.0]
    finally:
        r.cap = old_cap
        r._live.clear()
