"""Transparent MUSA CUDA-graph executable rotation.

The MUSA driver caps the number of *live* ``musaGraphExec_t`` (instantiated CUDA
graphs) at ~2048 per process; the next ``musaGraphInstantiate`` then throws an
illegal-memory-access at ``capture_end``. vLLM/SGLang piecewise CUDA graphs
instantiate ``capture_sizes * num_layers`` executables, so models deeper than
~40 layers blow past the cap and cannot use piecewise CUDA graphs at all.

This works around it transparently. Graph *templates* (``musaGraph_t``) do **not**
count toward the cap and re-instantiating an executable from an existing template is
cheap (~0.3 ms) and needs no forward re-run. So we keep every template alive,
LRU-cap the live *executables* (default cap 1900, just under the ~2043 driver
limit), and re-instantiate an evicted graph's executable from its template on the
next replay.

It is **zero-cost until the cap is exceeded**: shallow models never evict, never
build the aux extension, and pay only a dict insert per capture plus a single bool
check per replay. Backed by a small aux ``.so`` JIT-built against torch_musa's
*installed* headers -- it does not rebuild torch_musa.

Environment variables:
  * ``TORCHADA_GRAPH_ROTATION=0`` -- disable rotation entirely.
  * ``TORCHADA_GRAPH_EXEC_CAP=<n>`` -- set the live-executable cap explicitly.
  * ``TORCHADA_GRAPH_AUTOPROBE=1`` -- measure the real per-process limit at startup
    and use ``limit - margin`` (adaptive across drivers; adds a one-time probe).
  * ``TORCHADA_GRAPH_EXEC_MARGIN=<n>`` -- margin under the probed limit (default 128).
"""

from __future__ import annotations

import collections
import logging
import os
import threading
import weakref
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Safe-max default, just under the observed MUSA-driver per-process live-executable
# limit (~2043 on MTT S5000 / driver 3.3.5). Bigger cap → larger working set fits →
# fewer re-instantiations. Override with TORCHADA_GRAPH_EXEC_CAP, or set
# TORCHADA_GRAPH_AUTOPROBE=1 to measure the real limit at startup and use limit−margin.
_DEFAULT_CAP = 1900
_DEFAULT_MARGIN = 128


def _read_cap() -> int:
    raw = os.environ.get("TORCHADA_GRAPH_EXEC_CAP", str(_DEFAULT_CAP))
    try:
        cap = int(raw)
    except ValueError:
        logger.warning("invalid TORCHADA_GRAPH_EXEC_CAP=%r; using %d", raw, _DEFAULT_CAP)
        return _DEFAULT_CAP
    return cap if cap > 0 else _DEFAULT_CAP


_PROBE_SCRIPT = """
import torch, torch_musa
pool = torch.musa.graph_pool_handle()
inp = torch.zeros(8, device="musa", dtype=torch.float32)
_ = inp + 1.0
torch.musa.synchronize()
n = 0
keep = []
for i in range({max_probe}):
    g = torch.musa.MUSAGraph()
    try:
        with torch.musa.graph(g, pool=pool):
            y = inp + 1.0
        keep.append(g)
        n = i + 1
    except Exception:
        break
print("TORCHADA_PROBE_LIMIT=%d" % n)
"""


def _probe_live_exec_limit(max_probe: int = 4096) -> Optional[int]:
    """Measure the MUSA driver's per-process live-executable limit, in an ISOLATED
    subprocess.

    Capturing CUDA graphs to the failure point corrupts the process's RNG-generator
    capture state ("Offset increment outside graph capture"), which would break the
    serving process's later RNG. So the probe runs in a throwaway subprocess: it
    captures trivial graphs until the ``capture_end`` illegal-memory-access, prints
    the count that succeeded (= the limit), and exits — any corruption dies with it.
    Returns the limit, or ``None`` if the probe could not run.
    """
    import subprocess
    import sys

    code = _PROBE_SCRIPT.format(max_probe=max_probe)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("TORCHADA_PROBE_LIMIT="):
                return int(line.split("=", 1)[1])
        logger.warning(
            "graph-exec probe subprocess produced no limit; stderr tail: %s",
            (proc.stderr or "")[-200:])
    except Exception as exc:  # noqa: BLE001
        logger.warning("graph-exec probe subprocess failed: %r", exc)
    return None


def _resolve_cap() -> int:
    """Pick the live-executable cap. Priority: explicit env > auto-probe > default."""
    if "TORCHADA_GRAPH_EXEC_CAP" in os.environ:
        return _read_cap()
    if os.environ.get("TORCHADA_GRAPH_AUTOPROBE", "0") == "1":
        limit = _probe_live_exec_limit()
        if limit and limit > 0:
            margin = _DEFAULT_MARGIN
            try:
                margin = int(os.environ.get("TORCHADA_GRAPH_EXEC_MARGIN", str(_DEFAULT_MARGIN)))
            except ValueError:
                pass
            cap = max(64, limit - margin)
            logger.warning(
                "torchada graph-exec auto-probe: driver live-exec limit ≈ %d, "
                "cap set to %d (margin %d).", limit, cap, margin)
            return cap
        logger.warning("torchada graph-exec auto-probe failed; using default cap %d", _DEFAULT_CAP)
    return _DEFAULT_CAP


class _Rotation:
    """LRU rotation of live CUDA-graph executables, keyed by ``id(graph)``.

    ``_live`` maps ``id(graph) -> weakref(graph)`` in LRU order and holds only the
    graphs whose executable is currently instantiated. Weak references are used so
    the rotation never keeps a user's graph alive beyond its own ownership.
    """

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._lock = threading.RLock()
        self._live: "collections.OrderedDict[int, weakref.ReferenceType]" = (
            collections.OrderedDict()
        )
        self._aux: Optional[Any] = None
        self._aux_failed = False
        self._evicting = False
        self.stats: Dict[str, int] = {
            "register": 0, "evict": 0, "reinstantiate": 0, "build_failed": 0
        }

    def _ensure_aux(self) -> Optional[Any]:
        if self._aux is not None:
            return self._aux
        if self._aux_failed:
            return None
        from ._cpp_ops import load_graph_rotation_ops

        aux = load_graph_rotation_ops()
        if aux is None:
            self._aux_failed = True
            self.stats["build_failed"] += 1
            logger.warning(
                "torchada graph-exec rotation unavailable (aux ops failed to build); "
                "deep models may hit the MUSA ~2048 CUDA-graph cap.")
            return None
        self._aux = aux
        logger.warning(
            "torchada graph-exec rotation engaged (cap=%d): live CUDA-graph "
            "executables exceeded the cap; rotating via re-instantiation.", self.cap)
        return self._aux

    def _evict_locked(self, keep_id: int) -> None:
        aux: Optional[Any] = None
        while len(self._live) > self.cap:
            key, wref = next(iter(self._live.items()))
            if key == keep_id:                      # never evict the graph we just touched
                self._live.move_to_end(key)
                break
            graph = wref()
            if graph is None:                       # already GC'd -> exec freed by its destructor
                self._live.pop(key, None)
                continue
            if aux is None:
                aux = self._ensure_aux()
                if aux is None:                     # aux unavailable -> cannot rotate; stop
                    return
            try:
                aux.free_exec(graph)                # destroy exec (keep template) FIRST
            except Exception as exc:                # noqa: BLE001
                # free failed -> the exec is still live; keep it tracked and stop, so
                # _live stays an honest count of live executables (no false eviction).
                logger.warning("torchada rotation free_exec failed: %r", exc)
                return
            self._live.pop(key, None)               # untrack only after the exec is freed
            self._evicting = True
            self.stats["evict"] += 1

    def register(self, graph: Any) -> None:
        with self._lock:
            key = id(graph)
            self._live[key] = weakref.ref(graph)
            self._live.move_to_end(key)
            self.stats["register"] += 1
            if len(self._live) > self.cap:
                self._evict_locked(key)

    def on_replay(self, graph: Any) -> None:
        if not self._evicting:                      # fast path: nothing evicted -> exec is live
            return
        with self._lock:
            aux = self._aux
            if aux is None:
                return
            key = id(graph)
            if key not in self._live or not aux.has_exec(graph):
                aux.inst_exec(graph)                # re-instantiate from the kept template
                self._live[key] = weakref.ref(graph)
                self.stats["reinstantiate"] += 1
            self._live.move_to_end(key)
            self._evict_locked(key)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.stats, live=len(self._live), cap=self.cap,
                        evicting=self._evicting, aux_failed=self._aux_failed)


_rotation: Optional[_Rotation] = None
_installed = False
_lock = threading.Lock()


def is_enabled() -> bool:
    return os.environ.get("TORCHADA_GRAPH_ROTATION", "1") != "0"


def stats() -> Optional[Dict[str, Any]]:
    """Diagnostics snapshot, or ``None`` if rotation is not installed."""
    return _rotation.snapshot() if _rotation is not None else None


def install() -> bool:
    """Patch ``torch.musa.MUSAGraph.{capture_end,replay}`` to rotate executables.

    Idempotent and safe to call from ``torchada.apply_patches()``. Returns True if
    the patches were installed. No-op if disabled, if torch.musa is unavailable, or
    if already installed.
    """
    global _installed, _rotation
    with _lock:
        if _installed:
            return True
        if not is_enabled():
            return False
        import torch

        graph_cls = getattr(getattr(torch, "musa", None), "MUSAGraph", None)
        if graph_cls is None:
            return False

        rot = _Rotation(_resolve_cap())
        orig_capture_end = graph_cls.capture_end
        orig_replay = graph_cls.replay

        def capture_end(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = orig_capture_end(self, *args, **kwargs)
            try:
                rot.register(self)
            except Exception as exc:                # noqa: BLE001
                logger.warning("torchada rotation register failed: %r", exc)
            return result

        def replay(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                rot.on_replay(self)
            except Exception as exc:                # noqa: BLE001
                logger.warning("torchada rotation on_replay failed: %r", exc)
            return orig_replay(self, *args, **kwargs)

        graph_cls.capture_end = capture_end
        graph_cls.replay = replay
        _rotation = rot
        _installed = True
        return True
