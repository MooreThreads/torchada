"""Tests for torchada's FSDP2 device dispatch (``torchada._fsdp2``).

CPU tests run against a fake torch_musa FSDP2 layer (``_fsdp2_fakes``): in process
for the dispatch mechanics, and through the multi-process gloo worker
(``_fsdp2_mesh_worker``) for whole forward and backward runs. MUSA tests run the same
worker against the real torch_musa.
"""

import contextlib
import functools
import json
import logging
import os
import pickle
import signal
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from types import FunctionType, ModuleType, SimpleNamespace
from typing import Callable, Dict, List, Optional

import pytest
import torch

import torchada
from torchada import _fsdp2, _patch

from . import _fsdp2_fakes as fakes
from . import _fsdp2_mesh_worker as worker

WORKER = str(Path(worker.__file__).resolve())
TORCHADA_SRC = str(Path(torchada.__file__).resolve().parents[1])
TARGET_IDS = [f"{t.cls}.{t.name}" for t in _fsdp2.TARGETS]
CPU_STREAM = "torch.cpu.Stream"
# Seconds each worker run of the MUSA-mesh comparisons may take.
MUSA_MESH_TIMEOUT = float(os.environ.get("TORCHADA_TEST_FSDP2_TIMEOUT", "180"))


def _require_cpu_simulation() -> None:
    if torchada.is_musa_platform() or "torch_musa" in sys.modules:
        pytest.skip("simulates torch_musa in process; a real torch_musa is loaded")


def _require_musa(devices: int = 0) -> None:
    if not torchada.is_musa_platform():
        pytest.skip("MUSA platform required")
    if devices and torch.musa.device_count() < devices:
        pytest.skip(f"{devices} MUSA devices required")


# ------------------------------------------------------------------ in-process setup


@pytest.fixture
def fsdp(monkeypatch):
    """FSDP2 modules; everything the fake layer or install() changes is restored."""
    _require_cpu_simulation()
    import torch.distributed.fsdp as api

    fakes.upstream_methods()  # PyTorch's methods are in place
    for target in _fsdp2.TARGETS:
        cls = getattr(sys.modules[target.module], target.cls)
        monkeypatch.setattr(cls, target.name, vars(cls)[target.name])
    collectives = sys.modules[fakes.COLLECTIVES]
    gather = collectives.foreach_all_gather
    monkeypatch.setattr(gather, "__code__", gather.__code__)
    monkeypatch.setattr(api, "fully_shard", api.fully_shard)
    param_group = sys.modules[fakes.PARAM_GROUP]
    state = sys.modules[fakes.STATE]
    return SimpleNamespace(
        monkeypatch=monkeypatch,
        api=api,
        package=sys.modules["torch.distributed.fsdp._fully_shard"],
        param_group=param_group,
        state=state,
        CommContext=param_group.FSDPCommContext,
        ParamGroup=param_group.FSDPParamGroup,
        State=state.FSDPState,
        originals={
            t.name: vars(getattr(sys.modules[t.module], t.cls))[t.name] for t in _fsdp2.TARGETS
        },
    )


def _fake_layer(fsdp, **kwargs) -> SimpleNamespace:
    def register(name: str, module: ModuleType) -> None:
        fsdp.monkeypatch.setitem(sys.modules, name, module)

    return fakes.build(register, **kwargs)


def _snapshot(fsdp) -> List[object]:
    """Every object install() may replace."""
    objects: List[object] = [fsdp.api.fully_shard]
    for target in _fsdp2.TARGETS:
        objects.append(vars(getattr(sys.modules[target.module], target.cls))[target.name])
    for module, name in fakes.HOOK_GLOBALS + fakes.IMPORT_GLOBALS:
        objects.append(getattr(sys.modules.get(module), name, None))
    return objects


def _assert_same(before: List[object], after: List[object]) -> None:
    assert len(before) == len(after)
    for old, new in zip(before, after):
        assert old is new


def _streams(ctx) -> List[object]:
    return [getattr(ctx, name) for name in worker.COMM_STREAMS]


def _spy_fully_shard(fsdp) -> SimpleNamespace:
    """Replace PyTorch's fully_shard (before the fake layer wraps it) with a spy."""
    calls = []

    def fully_shard(*args, **kwargs):
        calls.append((args, kwargs))
        return "sharded"

    fully_shard.state = object()
    fsdp.monkeypatch.setattr(fsdp.package, "fully_shard", fully_shard)
    return SimpleNamespace(fn=fully_shard, calls=calls)


# ------------------------------------------------- recovering PyTorch's own methods


@pytest.mark.parametrize("target", _fsdp2.TARGETS, ids=TARGET_IDS)
def test_upstream_method_returns_the_live_method(fsdp, target):
    module = sys.modules[target.module]
    cls = getattr(module, target.cls)
    assert _fsdp2.upstream_method(module, cls, target.name) is vars(cls)[target.name]


@pytest.mark.parametrize("target", _fsdp2.TARGETS, ids=TARGET_IDS)
def test_upstream_method_rebuilds_a_replaced_method(fsdp, target):
    module = sys.modules[target.module]
    cls = getattr(module, target.cls)
    original = vars(cls)[target.name]

    def foreign(self, *args, **kwargs):
        raise AssertionError("not PyTorch's")

    fsdp.monkeypatch.setattr(cls, target.name, foreign)
    rebuilt = _fsdp2.upstream_method(module, cls, target.name)
    assert isinstance(rebuilt, FunctionType) and rebuilt is not original
    assert rebuilt.__code__ == original.__code__
    assert rebuilt.__globals__ is module.__dict__
    for attr in ("__name__", "__qualname__", "__module__", "__doc__", "__defaults__"):
        assert getattr(rebuilt, attr) == getattr(original, attr), attr
    assert rebuilt.__kwdefaults__ is None and rebuilt.__closure__ is None


_PROBE_SOURCE = """
def keep(fn):
    return fn


def helper():
    return 0


class Probe:
    def sibling(self):
        return 1

    def target(self):
        return helper()
"""


def _probe_module(monkeypatch, tmp_path, source: str) -> ModuleType:
    import importlib.util

    path = tmp_path / "torchada_fsdp2_probe.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("torchada_fsdp2_probe", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _torch_musa_function() -> FunctionType:
    namespace = {"__name__": "torch_musa.distributed.helpers"}
    exec("def helper():\n    return 0\n", namespace)
    return namespace["helper"]


_FAIL_CLOSED = {
    "sibling-edited": lambda path, module: path.write_text(
        _PROBE_SOURCE.replace("return 1", "return 3")
    ),
    "file-missing": lambda path, module: path.unlink(),
    "torch-musa-global": lambda path, module: setattr(module, "helper", _torch_musa_function()),
    "decorated": "    @keep\n    def target(self):",
    "default-argument": "    def target(self, value=1):",
    "keyword-only-default": "    def target(self, *, value=1):",
    "duplicate-class": None,
    "closure": "    def target(self):\n        return super().__init__\n\n    def unused(self):",
}


@pytest.mark.parametrize("case", sorted(_FAIL_CLOSED))
def test_upstream_method_fails_closed(monkeypatch, tmp_path, case):
    change = _FAIL_CLOSED[case]
    source = _PROBE_SOURCE
    if isinstance(change, str):
        source = source.replace("    def target(self):", change)
    elif change is None:
        source = source + source[source.index("class Probe") :]
    module = _probe_module(monkeypatch, tmp_path, source)
    if callable(change):
        change(tmp_path / "torchada_fsdp2_probe.py", module)
    assert _fsdp2.upstream_method(module, module.Probe, "target") is None


def test_upstream_method_of_an_unchanged_probe(monkeypatch, tmp_path):
    module = _probe_module(monkeypatch, tmp_path, _PROBE_SOURCE)
    assert _fsdp2.upstream_method(module, module.Probe, "target") is vars(module.Probe)["target"]


def test_install_changes_nothing_when_a_method_cannot_be_recovered(fsdp, tmp_path, caplog):
    _fake_layer(fsdp)
    fsdp.monkeypatch.setattr(fsdp.state, "__file__", str(tmp_path / "missing.py"))
    before = _snapshot(fsdp)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        assert _fsdp2.install() is False
    _assert_same(before, _snapshot(fsdp))
    assert "FSDPState._root_post_backward_final_callback" in caplog.text


@pytest.mark.parametrize("kind", ["owned", "code-swapped"])
def test_install_changes_nothing_when_pytorch_reads_a_torch_musa_global(fsdp, caplog, kind):
    layer = _fake_layer(fsdp)
    # FSDPCommContext.lazy_init reads the module global _get_device_handle.
    if kind == "owned":
        replacement = _torch_musa_function()
    else:
        code = compile(
            "def _get_device_handle(device_type):\n    return None\n",
            os.path.join(fakes.FAKE_DIR, "distributed", "helpers.py"),
            "exec",
        )
        namespace = {"__name__": fsdp.param_group.__name__}
        exec(code, namespace)
        replacement = namespace["_get_device_handle"]
        assert layer.modules["torch_musa"].__file__.startswith(fakes.FAKE_DIR)
    fsdp.monkeypatch.setattr(fsdp.param_group, "_get_device_handle", replacement)
    before = _snapshot(fsdp)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        assert _fsdp2.install() is False
    _assert_same(before, _snapshot(fsdp))
    assert "FSDPCommContext.lazy_init" in caplog.text


# ------------------------------------------------------------------------ dispatch


def _device(device_type: str) -> SimpleNamespace:
    return SimpleNamespace(type=device_type)


def test_dispatcher_routes_on_the_device_type():
    calls = []

    def musa_fn(self, *args, **kwargs):
        calls.append(("musa", self, args, kwargs))
        return "musa"

    def upstream_fn(self, *args, **kwargs):
        calls.append(("upstream", self, args, kwargs))
        return "upstream"

    dispatcher = _fsdp2.dispatch(musa_fn, upstream_fn, lambda self, *a, **k: self.device.type)
    on_musa = SimpleNamespace(device=_device("musa"))
    on_cpu = SimpleNamespace(device=_device("cpu"))
    assert dispatcher(on_musa, 1, flag=2) == "musa"
    assert dispatcher(on_cpu, 3, flag=4) == "upstream"
    assert calls == [("musa", on_musa, (1,), {"flag": 2}), ("upstream", on_cpu, (3,), {"flag": 4})]
    assert dispatcher.__wrapped__ is musa_fn
    assert getattr(dispatcher, _fsdp2.UPSTREAM_ATTR) is upstream_fn
    assert dispatcher.__qualname__ == musa_fn.__qualname__
    assert _fsdp2.is_dispatch(dispatcher) and not _fsdp2.is_dispatch(musa_fn)


def test_dispatcher_runs_torch_musa_when_the_device_type_is_unknown():
    def device_type(self, *args, **kwargs):
        raise AttributeError("no device")

    dispatcher = _fsdp2.dispatch(lambda self: "musa", lambda self: "upstream", device_type)
    assert dispatcher(object()) == "musa"


def test_target_device_types_read_the_call_arguments():
    keys = {t.name: t.device_type for t in _fsdp2.TARGETS}
    cpu = torch.device("cpu")
    assert keys["lazy_init"](object(), cpu) == "cpu"
    assert keys["lazy_init"](object(), device=cpu) == "cpu"
    assert keys["lazy_init"](object(), cpu, "added", flag=True) == "cpu"
    group = SimpleNamespace(device=cpu)
    assert keys["wait_for_unshard"](group) == "cpu"
    assert keys["wait_for_unshard"](group, "added", flag=True) == "cpu"
    assert keys["post_backward"](group, "unused", "args") == "cpu"
    state = SimpleNamespace(_device=_device("musa"))
    assert keys["_root_post_backward_final_callback"](state) == "musa"
    assert keys["_root_post_backward_final_callback"](state, "added") == "musa"


def test_lazy_init_dispatches_on_the_device_type(fsdp):
    layer = _fake_layer(fsdp)
    assert _fsdp2.install()
    layer.patch._setup_fsdp2_patches()  # torch_musa's first fully_shard call

    ctx = fsdp.CommContext()
    ctx.lazy_init(torch.device("cpu"))
    streams = _streams(ctx)
    assert all(type(s) is torch.cpu.Stream for s in streams)
    assert len({id(s) for s in streams}) == 4
    assert layer.state.calls == []

    ctx = fsdp.CommContext()
    ctx.lazy_init(_device("musa"))
    assert all(s is layer.overlap.CURRENT_STREAM for s in _streams(ctx))
    assert layer.state.calls == ["comm_context_lazy_init"]


# ------------------------------------------------------------- torch_musa's hooks


def _assert_dispatchers(fsdp, wrapped: Dict[str, object]) -> None:
    for target in _fsdp2.TARGETS:
        attr = vars(getattr(sys.modules[target.module], target.cls))[target.name]
        assert _fsdp2.is_dispatch(attr), target.name
        assert attr.__wrapped__ is wrapped[target.name], target.name
        assert not _fsdp2.is_dispatch(attr.__wrapped__)


def test_torch_musa_setup_after_install_assigns_dispatchers(fsdp):
    layer = _fake_layer(fsdp)
    wrapped = {
        "lazy_init": layer.overlap.comm_context_lazy_init,
        "wait_for_unshard": layer.patch.wait_for_unshard_non_overlap,
        "post_backward": layer.patch.post_backward_non_overlap,
        "_root_post_backward_final_callback": layer.fsdp_state._root_post_backward_final_callback,
    }
    assert _fsdp2.install()
    assert not _fsdp2.is_dispatch(fsdp.CommContext.lazy_init)
    for module, name in fakes.HOOK_GLOBALS:
        assert _fsdp2.is_dispatch(getattr(sys.modules[module], name)), name
    layer.patch._setup_fsdp2_patches()
    _assert_dispatchers(fsdp, wrapped)
    for name in ("lazy_init", "wait_for_unshard"):
        attr = getattr(fsdp.CommContext if name == "lazy_init" else fsdp.ParamGroup, name)
        assert getattr(attr, _fsdp2.UPSTREAM_ATTR) is fsdp.originals[name]


def test_install_after_torch_musa_setup_rebuilds_pytorch_methods(fsdp):
    layer = _fake_layer(fsdp)
    layer.patch._setup_fsdp2_patches()
    wrapped = {t.name: vars(getattr(sys.modules[t.module], t.cls))[t.name] for t in _fsdp2.TARGETS}
    assert all(_fsdp2.is_torch_musa(fn) for fn in wrapped.values())
    assert _fsdp2.install()
    _assert_dispatchers(fsdp, wrapped)
    for target in _fsdp2.TARGETS:
        module = sys.modules[target.module]
        attr = vars(getattr(module, target.cls))[target.name]
        upstream = getattr(attr, _fsdp2.UPSTREAM_ATTR)
        assert upstream.__code__ == fsdp.originals[target.name].__code__
        assert upstream.__globals__ is module.__dict__
        assert upstream.__qualname__ == f"{target.cls}.{target.name}"
    ctx = fsdp.CommContext()
    ctx.lazy_init(torch.device("cpu"))
    assert all(type(s) is torch.cpu.Stream for s in _streams(ctx))
    assert layer.state.calls == []


def _assert_pickles_by_reference() -> None:
    objects = [vars(getattr(sys.modules[t.module], t.cls))[t.name] for t in _fsdp2.TARGETS]
    objects += [getattr(sys.modules[module], name) for module, name in fakes.HOOK_GLOBALS]
    for obj in objects:
        assert pickle.loads(pickle.dumps(obj)) is obj, obj.__qualname__


@pytest.mark.parametrize("order", ["install-first", "setup-first"])
def test_dispatchers_pickle_by_reference(fsdp, order):
    layer = _fake_layer(fsdp)
    if order == "setup-first":
        layer.patch._setup_fsdp2_patches()
    assert _fsdp2.install()
    if order == "install-first":
        # torch_musa's import-time replacement, until its first fully_shard call.
        assert vars(fsdp.ParamGroup)["post_backward"] is layer.param_group.post_backward
        assert _fsdp2.is_dispatch(layer.fsdp_state._root_post_backward_final_callback)
        _assert_pickles_by_reference()
        layer.patch._setup_fsdp2_patches()
    assert vars(fsdp.ParamGroup)["post_backward"] is layer.patch.post_backward_non_overlap
    _assert_pickles_by_reference()


# ------------------------------------------------------------------------- router


@pytest.fixture
def routed(fsdp):
    spy = _spy_fully_shard(fsdp)
    layer = _fake_layer(fsdp)
    wrapper = fsdp.api.fully_shard
    assert _fsdp2.install()
    return SimpleNamespace(
        fsdp=fsdp, layer=layer, spy=spy, wrapper=wrapper, router=fsdp.api.fully_shard
    )


def test_router_sends_non_musa_meshes_to_pytorch(routed):
    mesh = SimpleNamespace(device_type="cpu")
    module = object()
    assert routed.router(module, mesh=mesh, reshard_after_forward=True) == "sharded"
    assert routed.spy.calls == [((module,), {"mesh": mesh, "reshard_after_forward": True})]
    state = routed.layer.state
    assert (state.arch, state.custom_comm, state.calls) == (0, 0, [])
    assert not _fsdp2.is_dispatch(routed.fsdp.CommContext.lazy_init)


# Parameter ids avoid the bare "musa" keyword, which conftest skips off MUSA.
@pytest.mark.parametrize("mesh", ["musa-mesh", "no-mesh", "unknown-mesh"])
def test_router_keeps_torch_musa_for_musa_meshes(routed, mesh):
    kwargs = {}
    if mesh == "musa-mesh":
        kwargs["mesh"] = SimpleNamespace(device_type="musa")
    elif mesh == "unknown-mesh":
        kwargs["mesh"] = SimpleNamespace()  # computing the device type raises
    module = object()
    assert routed.router(module, **kwargs) == "sharded"
    assert routed.spy.calls == [((module,), kwargs)]
    assert (routed.layer.state.arch, routed.layer.state.custom_comm) == (1, 1)
    lazy_init = routed.fsdp.CommContext.lazy_init
    assert _fsdp2.is_dispatch(lazy_init)
    assert lazy_init.__wrapped__.__name__ == "comm_context_lazy_init"


def test_router_follows_torch_device_translation(routed, monkeypatch):
    mesh = SimpleNamespace(device_type="cuda")
    assert _fsdp2._mesh_device_type(mesh) == "cuda"
    monkeypatch.setattr(
        torch, "device", lambda kind, *args: _device("musa" if kind == "cuda" else kind)
    )
    assert _fsdp2._mesh_device_type(mesh) == "musa"
    routed.router(object(), mesh=mesh)
    assert routed.layer.state.arch == 1


def test_router_presents_torch_musa_wrapper(routed):
    router = routed.router
    assert getattr(router, _fsdp2.ROUTER_ATTR)
    assert router.state is routed.spy.fn.state
    assert router.__wrapped__ is routed.wrapper
    assert router.__wrapped__.__wrapped__ is routed.spy.fn


def test_router_wraps_torch_musa_methods_under_other_names(fsdp):
    spy = _spy_fully_shard(fsdp)
    layer = _fake_layer(fsdp, renamed=True)
    assert _fsdp2.install()
    for module, name in fakes.HOOK_GLOBALS:
        assert not hasattr(sys.modules[module], name)
    fsdp.api.fully_shard(object(), mesh=SimpleNamespace(device_type="musa"))
    lazy_init = fsdp.CommContext.lazy_init
    assert _fsdp2.is_dispatch(lazy_init)
    assert lazy_init.__wrapped__.__name__ == "comm_context_lazy_init" + fakes.RENAMED_SUFFIX
    assert _fsdp2.is_dispatch(fsdp.ParamGroup.wait_for_unshard)
    assert _fsdp2.is_dispatch(fsdp.ParamGroup.post_backward)

    fsdp.api.fully_shard(object(), mesh=SimpleNamespace(device_type="cpu"))
    ctx = fsdp.CommContext()
    ctx.lazy_init(torch.device("cpu"))
    assert all(type(s) is torch.cpu.Stream for s in _streams(ctx))
    assert layer.state.calls == [] and len(spy.calls) == 2


def test_install_warns_when_fully_shard_cannot_be_routed(fsdp, caplog):
    _fake_layer(fsdp)
    wrapper = fsdp.api.fully_shard

    @functools.wraps(wrapper)
    def outer(*args, **kwargs):
        return wrapper(*args, **kwargs)

    fsdp.monkeypatch.setattr(fsdp.api, "fully_shard", outer)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        assert _fsdp2.install()
    assert fsdp.api.fully_shard is outer
    assert "bind or wrap fully_shard after importing torchada" in caplog.text
    assert _fsdp2.is_dispatch(fsdp.ParamGroup.post_backward)


# ----------------------------------------------------------------- install and shim


def test_install_is_idempotent(fsdp, monkeypatch):
    layer = _fake_layer(fsdp)
    assert _fsdp2.install()
    installed = _snapshot(fsdp)
    assert _fsdp2.install()
    _assert_same(installed, _snapshot(fsdp))

    monkeypatch.setattr(_patch, "is_musa_platform", lambda: True)
    monkeypatch.setattr(_patch, "_fsdp2_dispatch_installed", False)
    _patch._patch_fsdp2_device_dispatch()
    _patch._patch_fsdp2_device_dispatch()
    _assert_same(installed, _snapshot(fsdp))

    layer.patch._setup_fsdp2_patches()
    after_setup = _snapshot(fsdp)
    assert _fsdp2.install()
    _assert_same(after_setup, _snapshot(fsdp))


def test_install_without_torch_musa_changes_nothing(fsdp):
    before = _snapshot(fsdp)
    assert _fsdp2.install() is False
    _assert_same(before, _snapshot(fsdp))


@pytest.mark.parametrize("decorated", [True, False])
def test_all_gather_guard(fsdp, caplog, decorated):
    _fake_layer(fsdp, decorated_non_overlap=decorated)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        assert _fsdp2.install()
    assert ("foreach_all_gather" in caplog.text) is not decorated


def _replacement_warnings(caplog) -> List[str]:
    return [r.getMessage() for r in caplog.records if "replacements of" in r.getMessage()]


def test_install_warns_about_torch_musa_replacements_outside_the_targets(fsdp, caplog):
    _fake_layer(fsdp)
    fsdp.monkeypatch.setattr(fsdp.CommContext, "get_all_gather_streams", _torch_musa_function())
    # An addition under a name PyTorch's class does not define is never called by PyTorch.
    fsdp.monkeypatch.setattr(fsdp.ParamGroup, "musa_only_helper", _torch_musa_function(), False)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        assert _fsdp2.install()
    (message,) = _replacement_warnings(caplog)
    assert "FSDPCommContext.get_all_gather_streams" in message
    assert "musa_only_helper" not in message


def test_router_warns_once_about_replacements_from_torch_musa_setup(routed, caplog):
    fsdp = routed.fsdp
    with caplog.at_level(logging.WARNING, logger="torchada"):
        routed.router(object(), mesh=SimpleNamespace(device_type="musa"))
        assert _replacement_warnings(caplog) == []
        fsdp.monkeypatch.setattr(fsdp.State, "_lazy_init", _torch_musa_function())
        routed.router(object(), mesh=SimpleNamespace(device_type="cpu"))
        routed.router(object(), mesh=SimpleNamespace(device_type="musa"))
    (message,) = _replacement_warnings(caplog)
    assert "FSDPState._lazy_init" in message


def _shim_setup(monkeypatch, musa: bool, install) -> None:
    _require_cpu_simulation()
    monkeypatch.setitem(sys.modules, "torch_musa", ModuleType("torch_musa"))
    monkeypatch.setattr(_patch, "is_musa_platform", lambda: musa)
    monkeypatch.setattr(_patch, "_fsdp2_dispatch_installed", False)
    monkeypatch.setattr(_fsdp2, "install", install)


def test_shim_does_nothing_off_musa(monkeypatch):
    calls = []
    _shim_setup(monkeypatch, musa=False, install=lambda: calls.append(1))
    _patch._patch_fsdp2_device_dispatch()
    assert calls == []


def test_shim_logs_and_continues_when_install_raises(monkeypatch, caplog):
    def install():
        raise RuntimeError("broken")

    _shim_setup(monkeypatch, musa=True, install=install)
    with caplog.at_level(logging.WARNING, logger="torchada"):
        _patch._patch_fsdp2_device_dispatch()
    assert "FSDP2 device dispatch was not installed" in caplog.text
    assert _patch._fsdp2_dispatch_installed


_IMPORT_PROBE = """
import sys
from types import ModuleType

from torchada import _patch

assert "torch_musa" not in sys.modules
assert "torch.distributed.fsdp" not in sys.modules
attempts = []


class Recorder:
    def find_spec(self, name, path=None, target=None):
        attempts.append(name)


sys.meta_path.insert(0, Recorder())
sys.modules["torch_musa"] = ModuleType("torch_musa")
_patch.is_musa_platform = lambda: True
_patch._patch_fsdp2_device_dispatch()
assert _patch._fsdp2_dispatch_installed
imported = [n for n in attempts if n.startswith(("torch_musa.", "torch.distributed.fsdp"))]
assert not imported, imported
assert "torch.distributed.fsdp" not in sys.modules
print("OK")
"""


def test_shim_imports_neither_fsdp_nor_torch_musa_modules():
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=_sim_env("visible"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout.splitlines()


# ------------------------------------------------------- multi-process gloo runs


def _env(updates: Dict[str, Optional[str]]) -> Dict[str, str]:
    env = dict(os.environ)
    for key, value in updates.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [TORCHADA_SRC, env.get("PYTHONPATH")]))
    return env


def _sim_env(flavour: str) -> Dict[str, str]:
    """A CPU simulation: real torch_musa stays unloaded, torchada patches nothing."""
    return _env(
        {
            "TORCHADA_PLATFORM": "cpu",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "MUSA_VISIBLE_DEVICES": "" if flavour == "hidden" else None,
        }
    )


def _raise_interrupt(signum: int, frame) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


@contextlib.contextmanager
def _signals_interrupt():
    """Make SIGTERM and SIGHUP raise KeyboardInterrupt while the block runs.

    Only signals left at their default action are changed, and only in the main thread;
    the previous handlers are restored afterwards.
    """
    previous = {}
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is not None and signal.getsignal(signum) == signal.SIG_DFL:
                previous[signum] = signal.signal(signum, _raise_interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _reap(proc: subprocess.Popen, drain: float) -> None:
    try:
        proc.wait(timeout=drain)
    except subprocess.TimeoutExpired:
        pass


def _drain(proc: subprocess.Popen, drain: float):
    """The output of a killed worker; waits at most ``drain`` seconds for the pipes."""
    try:
        return proc.communicate(timeout=drain)
    except subprocess.TimeoutExpired as exc:
        partial = [exc.stdout, exc.stderr]
        proc.stdout.close()
        proc.stderr.close()
        _reap(proc, drain)
        return [p.decode(errors="replace") if isinstance(p, bytes) else p or "" for p in partial]


def _worker(
    env: Dict[str, str], *args: str, timeout: float = 600, drain: float = 30
) -> SimpleNamespace:
    """Run the worker in its own process group.

    On a timeout, an interrupt, SIGTERM or SIGHUP the whole group is killed, so no rank is
    left holding a device or waiting in a collective; SIGTERM and SIGHUP then raise
    KeyboardInterrupt. After the kill the output is collected for at most ``drain``
    seconds, because a process outside the group may still hold the pipes.
    """
    with _signals_interrupt():
        proc = subprocess.Popen(
            [sys.executable, WORKER, *args],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            timed_out = False
        except BaseException as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if not isinstance(exc, subprocess.TimeoutExpired):
                _reap(proc, drain)
                raise
            stdout, stderr = _drain(proc, drain)
            timed_out = True
    return SimpleNamespace(
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        timeout=timeout,
    )


def _run(out: Path, env: Dict[str, str], *args: str, timeout: float = 600) -> SimpleNamespace:
    return _worker(env, "run", "--out", str(out), *args, timeout=timeout)


def _failure(result: SimpleNamespace) -> Optional[str]:
    """None for a passing run, else its timeout or the exception that ended it."""
    if result.timed_out:
        return f"timeout after {result.timeout:g} s"
    if result.returncode == 0:
        return None
    for text in (result.stderr, result.stdout):
        signature = worker.failure_signature(text or "")
        if signature is not None:
            return signature
    return f"exit status {result.returncode}"


def _passed(result: SimpleNamespace, note: str = "") -> None:
    assert result.returncode == 0 and not result.timed_out, "\n".join(
        filter(None, [note, str(_failure(result)), result.stdout[-4000:], result.stderr[-4000:]])
    )


def _record(out: Path, rank: int = 0) -> dict:
    with open(out / f"rank{rank}.json") as f:
        return json.load(f)


def _trace_lines(out: Path) -> List[str]:
    with open(out / "trace.json") as f:
        return [line for lines in json.load(f).values() for line in lines]


@pytest.fixture(scope="module")
def upstream_sim(tmp_path_factory):
    """PyTorch's own FSDP2 run of the worker per mesh shape, computed once."""
    runs: Dict[str, Path] = {}

    def run(shape: str) -> Path:
        if shape not in runs:
            out = tmp_path_factory.mktemp("upstream")
            _passed(_run(out, _sim_env("visible"), "--arm", "upstream", "--shape", shape))
            runs[shape] = out
        return runs[shape]

    return run


_SIM_CASES = [
    ("visible", "1,2"),
    ("hidden", "1,2"),
    pytest.param("visible", "2,2", marks=pytest.mark.slow),
    pytest.param("visible", "1,4", marks=pytest.mark.slow),
]


@pytest.mark.parametrize("flavour,shape", _SIM_CASES)
def test_cpu_mesh_runs_pytorch_fsdp2(tmp_path, upstream_sim, flavour, shape):
    _passed(_run(tmp_path, _sim_env(flavour), "--arm", "fake-torchada", "--shape", shape))
    assert worker.compare(str(upstream_sim(shape)), str(tmp_path)) == []
    for rank in range(len(worker.load(str(tmp_path))[0])):
        record = _record(tmp_path, rank)
        assert record["fake"] == {"calls": [], "arch": 0, "custom_comm": 0}
        assert record["meshes"][0]["stream_types"] == [CPU_STREAM] * 4
        assert record["overlap_level"] == "FSDP2OverlapLevel.NO_OVERLAP"
    assert not any("torch_musa/" in line for line in _trace_lines(tmp_path))


@pytest.mark.parametrize(
    "flavour,signature",
    [
        ("visible", "AttributeError: 'NoneType' object has no attribute 'wait'"),
        ("hidden", "RuntimeError: No MUSA GPUs are available"),
    ],
)
def test_fake_layer_reproduces_the_cpu_mesh_failure(tmp_path, flavour, signature):
    result = _run(tmp_path, _sim_env(flavour), "--arm", "fake", "--steps", "0")
    assert _failure(result) == signature, result.stderr[-4000:]


_SPAWN_FAILURE = """\
W1008 10:11:29.668000 3837 torch/multiprocessing/spawn.py:165] Terminating process 3988
Traceback (most recent call last):
  File "_fsdp2_mesh_worker.py", line 432, in _run
torch.multiprocessing.spawn.ProcessRaisedException:

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "torch/distributed/distributed_c10d.py", line 3080, in all_reduce
    work = group.allreduce([tensor], opts)
{}
"""
_MCCL = """\
RuntimeError: MCCL error in: ProcessGroupMCCL.cpp:3604, unhandled musa error
unhandled musa error (run with MCCL_DEBUG=INFO for details)
"""
_CHAIN = """\
ValueError: inner

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "_fsdp2_mesh_worker.py", line 400, in _rank_body
RuntimeError: outer
"""
_SHUTDOWN_NOISE = """\
Exception ignored in: <function _Stream.__del__ at 0x7f0000000000>
Traceback (most recent call last):
  File "stream.py", line 9, in __del__
TypeError: 'NoneType' object is not callable
"""
_PARENT_FAILURE = """\
Traceback (most recent call last):
  File "torch/multiprocessing/spawn.py", line 204, in join
torch.multiprocessing.spawn.ProcessExitedException: process 1 terminated with signal SIGKILL
"""
_MCCL_LINE = "RuntimeError: MCCL error in: ProcessGroupMCCL.cpp:3604, unhandled musa error"


@pytest.mark.parametrize(
    "stderr,expected",
    [
        (_SPAWN_FAILURE.format(_MCCL), _MCCL_LINE),
        (_SPAWN_FAILURE.format(_MCCL) + _SHUTDOWN_NOISE, _MCCL_LINE),
        (_SPAWN_FAILURE.format("Exception: cause A"), "Exception: cause A"),
        (_SPAWN_FAILURE.format("KeyboardInterrupt"), "KeyboardInterrupt"),
        (_SPAWN_FAILURE.format(_CHAIN), "RuntimeError: outer"),
        (_PARENT_FAILURE + _SHUTDOWN_NOISE, _PARENT_FAILURE.splitlines()[-1]),
        ("fsdp2-worker: cannot write the output or rendezvous files: [Errno 28] ...", None),
        ("Error: no traceback\nException ignored in: <function f>\n", "Error: no traceback"),
    ],
)
def test_failure_names_the_rank_exception(stderr, expected):
    stdout = "2026-10-08 10:09:59 | _patch | 1 | INFO : backport armed\n"
    result = SimpleNamespace(returncode=1, stdout=stdout, stderr=stderr, timed_out=False)
    assert _failure(result) == (expected or "exit status 1")
    assert _failure(SimpleNamespace(**{**vars(result), "returncode": 0})) is None
    timed_out = {**vars(result), "returncode": -9, "timed_out": True, "timeout": 180}
    assert _failure(SimpleNamespace(**timed_out)) == "timeout after 180 s"


def _gone(pid: int) -> bool:
    """Whether ``pid`` has exited (an unreaped zombie counts as exited)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _wait_for(predicate: Callable[[], bool], timeout: float = 30) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


_STUB_GROUP = """\
import os, subprocess, sys, time
sleepers = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"]) for _ in "ab"]
with open(sys.argv[1] + ".tmp", "w") as f:
    f.write(" ".join(str(p) for p in [os.getpid()] + [s.pid for s in sleepers]))
os.rename(sys.argv[1] + ".tmp", sys.argv[1])
time.sleep(120)
"""


def test_worker_group_is_killed_on_sigterm(tmp_path, monkeypatch):
    if not os.path.isdir("/proc/self") or not hasattr(os, "killpg"):
        pytest.skip("needs POSIX process groups and /proc")
    if signal.getsignal(signal.SIGTERM) != signal.SIG_DFL:
        pytest.skip("SIGTERM already has a handler")
    stub, pids = tmp_path / "stub.py", tmp_path / "pids"
    stub.write_text(_STUB_GROUP)
    monkeypatch.setitem(globals(), "WORKER", str(stub))

    def terminate() -> None:
        if _wait_for(pids.exists):  # the stub and its two children are running
            os.kill(os.getpid(), signal.SIGTERM)

    thread = threading.Thread(target=terminate, daemon=True)
    thread.start()
    with pytest.raises(KeyboardInterrupt):
        _worker(_env({}), str(pids), timeout=60)
    thread.join()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    left = [pid for pid in map(int, pids.read_text().split()) if not _wait_for(lambda: _gone(pid))]
    assert left == []


_STUB_ESCAPE = """\
import os, sys, time
if os.fork() == 0:
    os.setsid()  # leaves the worker's process group and keeps its stdout and stderr
    with open(sys.argv[1] + ".tmp", "w") as f:
        f.write(str(os.getpid()))
    os.rename(sys.argv[1] + ".tmp", sys.argv[1])
    time.sleep(120)
    os._exit(0)
time.sleep(120)
"""


def test_worker_output_drain_is_bounded(tmp_path, monkeypatch):
    if not hasattr(os, "fork") or not hasattr(os, "killpg"):
        pytest.skip("needs POSIX process groups")
    stub, escaped = tmp_path / "stub.py", tmp_path / "escaped"
    stub.write_text(_STUB_ESCAPE)
    monkeypatch.setitem(globals(), "WORKER", str(stub))
    start = time.monotonic()
    try:
        result = _worker(_env({}), str(escaped), timeout=1, drain=1)
        elapsed = time.monotonic() - start
    finally:
        if _wait_for(escaped.exists, 10):
            with contextlib.suppress(ProcessLookupError):
                os.kill(int(escaped.read_text()), signal.SIGKILL)
    assert _failure(result) == "timeout after 1 s"
    assert elapsed < 20


@pytest.mark.parametrize("flavour", ["visible", "hidden"])
def test_consumer_bound_before_install(tmp_path, upstream_sim, flavour):
    _passed(_run(tmp_path, _sim_env(flavour), "--arm", "fake-torchada", "--early-bound"))
    assert worker.compare(str(upstream_sim("1,2")), str(tmp_path), trace="run") == []
    record = _record(tmp_path)
    # torch_musa's wrapper still runs its setup and custom-comm hook for this consumer.
    assert record["fake"]["calls"] == []
    assert record["fake"]["arch"] >= 1 and record["fake"]["custom_comm"] == 3


def test_structure_probe_on_the_fake_layer(tmp_path):
    args = ["--device-types", "cpu", "--after-lazy-setup", "--out", str(tmp_path)]
    result = _worker(_sim_env("visible"), "structure", "--arm", "fake-torchada", *args, timeout=120)
    _passed(result)
    _assert_structure(worker.load_structure(str(tmp_path)), allowed_extra=())


# ---------------------------------------------------------------------- MUSA hosts


def _assert_structure(probe: dict, allowed_extra=worker.TORCH_MUSA_ADDITIONS) -> None:
    allowed = {t.name for t in _fsdp2.TARGETS} | set(allowed_extra)
    for when, record in probe.items():
        if when == "torchada_warnings":
            assert record == []
            continue
        assert record["router"] and record["router_has_state"], when
        assert record["router_wraps_torch_musa"] and record["router_chain_to_upstream"], when
        for name, hook in record["hook_globals"].items():
            assert hook == {"present": True, "dispatch": True}, (when, name)
        for name, target in record["targets"].items():
            assert target["upstream_matches_source"], (when, name)
            assert target["upstream"][0].startswith("torch.distributed.fsdp."), (when, name)
            if target["dispatch"]:
                assert target["torch_musa"][0].startswith("torch_musa."), (when, name)
        after_setup = when == "after_lazy_setup"
        dispatched = {n.split(".")[1] for n, t in record["targets"].items() if t["dispatch"]}
        expected = {"post_backward", "_root_post_backward_final_callback"}
        if after_setup:
            expected |= {"lazy_init", "wait_for_unshard"}
        assert dispatched == expected, when
        for cls_name, names in record["torch_musa_owned"].items():
            assert set(names) <= allowed, (when, cls_name, names)
        for device_type, (via_device, via_mesh) in record["device_types"].items():
            assert via_device == via_mesh, (when, device_type)


_TORCH_MUSA_FSDP2_ENV = (
    "TORCH_MUSA_FSDP2_OVERLAP_LEVEL",
    "TORCH_MUSA_FSDP2_COMM_TYPE",
    "TORCH_MUSA_FSDP2_DISABLE_OVERLAP",
)


def _musa_env(updates: Dict[str, Optional[str]]) -> Dict[str, str]:
    """The host environment with only the given torch_musa FSDP2 settings."""
    return _env({**{key: None for key in _TORCH_MUSA_FSDP2_ENV}, **updates})


def _stderr_after_imports(out: Path, rank: int) -> str:
    text = (out / f"rank{rank}.log").read_text(errors="replace")
    return text.split(worker.IMPORTS_DONE, 1)[-1]


_HOST_ENVS = {
    "visible": {"MUSA_VISIBLE_DEVICES": None},
    "hidden": {"MUSA_VISIBLE_DEVICES": ""},
    "overlap-level-4": {"TORCH_MUSA_FSDP2_OVERLAP_LEVEL": "4"},
    "comm-type-1": {"TORCH_MUSA_FSDP2_COMM_TYPE": "1"},
}
# torch_musa's overlap level set at import; comm-type-1's depends on the MUSA arch.
_HOST_OVERLAP_LEVELS = {
    "visible": "FSDP2OverlapLevel.NO_OVERLAP",
    "hidden": "FSDP2OverlapLevel.NO_OVERLAP",
    "overlap-level-4": "FSDP2OverlapLevel.OVERLAP_HSDP_COMM",
}


def test_musa_structure(tmp_path):
    _require_musa(devices=1)
    args = ["--arm", "torchada", "--after-lazy-setup", "--out", str(tmp_path)]
    _passed(_worker(_musa_env({}), "structure", *args))
    probe = worker.load_structure(str(tmp_path))
    _assert_structure(probe)
    assert probe["before"]["device_types"]["cuda"] == ["musa", "musa"]


@pytest.mark.parametrize("shape", ["1,2", pytest.param("1,4", marks=pytest.mark.slow)])
@pytest.mark.parametrize("case", sorted(_HOST_ENVS))
def test_musa_host_cpu_mesh_runs_pytorch_fsdp2(tmp_path, case, shape):
    _require_musa()
    env = dict(_HOST_ENVS[case])
    upstream, patched = tmp_path / "upstream", tmp_path / "torchada"
    upstream_env = _musa_env({**env, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    _passed(_run(upstream, upstream_env, "--arm", "upstream", "--shape", shape))
    _passed(_run(patched, _musa_env(env), "--arm", "torchada", "--shape", shape))
    assert worker.compare(str(upstream), str(patched)) == []
    for rank in range(len(worker.load(str(patched))[0])):
        record = _record(patched, rank)
        assert record["meshes"][0]["stream_types"] == [CPU_STREAM] * 4
        if case in _HOST_OVERLAP_LEVELS:
            assert record["overlap_level"] == _HOST_OVERLAP_LEVELS[case]
        states = set(record["musa_initialized"].values())
        assert len(states) == 1, record["musa_initialized"]  # unchanged after the imports
        if case == "visible":
            assert states == {False}
        tail = _stderr_after_imports(patched, rank)
        assert "get_devie_properties failed" not in tail
        assert "The overlapping of FSDP2 was disabled" not in tail


@pytest.mark.parametrize(
    "case,signature",
    [
        ("visible", "AttributeError: 'NoneType' object has no attribute 'wait'"),
        ("hidden", "No MUSA GPUs are available"),
    ],
)
def test_musa_host_cpu_mesh_fails_with_torch_musa_alone(tmp_path, case, signature):
    _require_musa()
    result = _run(tmp_path, _musa_env(_HOST_ENVS[case]), "--arm", "torch-musa", "--steps", "0")
    assert result.returncode != 0
    assert signature in result.stderr


@pytest.mark.parametrize("flavour", ["visible", "hidden"])
def test_musa_host_consumer_bound_before_torchada(tmp_path, flavour):
    _require_musa()
    env = _HOST_ENVS[flavour]
    upstream, patched = tmp_path / "upstream", tmp_path / "torchada"
    upstream_env = _musa_env({**env, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    _passed(_run(upstream, upstream_env, "--arm", "upstream"))
    _passed(_run(patched, _musa_env(env), "--arm", "torchada", "--early-bound"))
    assert worker.compare(str(upstream), str(patched), trace="run") == []


_MUSA_ENVS = {
    "default": {},
    "overlap-level-4": {"TORCH_MUSA_FSDP2_OVERLAP_LEVEL": "4"},
    "comm-type-1": {"TORCH_MUSA_FSDP2_COMM_TYPE": "1"},
    "comm-type-2": {"TORCH_MUSA_FSDP2_COMM_TYPE": "2"},
}


@pytest.fixture(scope="module")
def torch_musa_runs(tmp_path_factory):
    """torch_musa-only runs on a MUSA mesh, computed once per case and shape.

    Each entry has ``failure`` (None, or how the first run ended, see ``_failure``) and
    ``timed_out``. After a completed first run it also has ``out`` (that run's output),
    ``flaky`` (None, or how a failing second run ended) and ``deterministic`` (whether the
    second run completed with the same results).
    """
    runs = {}

    def run(case: str, shape: str) -> SimpleNamespace:
        key = (case, shape)
        if key not in runs:
            env = _musa_env(_MUSA_ENVS[case])
            args = ("--arm", "torch-musa", "--mesh", "musa", "--shape", shape)
            outs, results = [], []
            for _ in range(2):
                outs.append(tmp_path_factory.mktemp("torch_musa"))
                results.append(_run(outs[-1], env, *args, timeout=MUSA_MESH_TIMEOUT))
                assert results[-1].returncode != 2, results[-1].stderr[-4000:]  # did not start
                if _failure(results[0]) is not None:
                    break
            entry = SimpleNamespace(
                failure=_failure(results[0]),
                timed_out=results[0].timed_out,
                out=None,
                flaky=None,
                deterministic=False,
            )
            if entry.failure is None:
                entry.out = outs[0]
                entry.flaky = _failure(results[1])
                if entry.flaky is None:
                    entry.deterministic = worker.compare(str(outs[0]), str(outs[1])) == []
            runs[key] = entry
        return runs[key]

    return run


def _check_baseline(
    config: str,
    baseline: SimpleNamespace,
    with_torchada: Optional[Callable[[], SimpleNamespace]] = None,
) -> str:
    """Skip when torch_musa alone cannot complete ``config``; returns a note on its runs.

    When its first run fails with an error, ``with_torchada`` runs the same configuration
    once and must fail with the same exception line. A timeout is not repeated. An OS
    error (``[Errno N]``) fails the test instead of skipping it. When only the second run
    fails, the test goes on against the first run, compared within a tolerance, and that
    is reported as a warning and in the returned note.
    """
    if baseline.failure is None:
        if baseline.flaky is None:
            return ""
        note = f"{config}: torch_musa alone passed once, then failed ({baseline.flaky})"
        warnings.warn(f"{note}; compared with its passing run within a tolerance")
        return note
    if "[Errno " in baseline.failure:  # an OS error comes from the host, not torch_musa
        pytest.fail(f"{config}: torch_musa alone fails with an OS error ({baseline.failure})")
    reason = f"{config}: torch_musa alone fails ({baseline.failure})"
    if with_torchada is not None and not baseline.timed_out:
        failure = _failure(with_torchada())
        assert failure == baseline.failure, f"{reason}; with torchada: {failure or 'passes'}"
        reason += "; with torchada it fails the same way"
    pytest.skip(reason)


@pytest.mark.slow
@pytest.mark.parametrize("mesh,shape", [("musa", "2"), ("musa", "1,2"), ("cuda", "2")])
@pytest.mark.parametrize("case", sorted(_MUSA_ENVS))
def test_musa_mesh_keeps_torch_musa_fsdp2(tmp_path, torch_musa_runs, case, mesh, shape):
    _require_musa(devices=2)
    baseline = torch_musa_runs(case, shape)
    env = _musa_env(_MUSA_ENVS[case])
    args = ("--arm", "torchada", "--mesh", mesh, "--shape", shape)

    def with_torchada() -> SimpleNamespace:
        return _run(tmp_path, env, *args, timeout=MUSA_MESH_TIMEOUT)

    config = f"{case}, {mesh} mesh {shape}"
    if mesh != "musa":  # torch_musa alone cannot shard it; its baseline is a musa mesh
        config += f" (baseline: torch_musa alone on musa mesh {shape})"
    note = _check_baseline(config, baseline, with_torchada)
    _passed(with_torchada(), note)
    # Sharding a "cuda" mesh also passes torchada's device translation.
    trace = "all" if mesh == "musa" else "run"
    exact = baseline.deterministic
    assert worker.compare(str(baseline.out), str(tmp_path), exact=exact, trace=trace) == [], note
    allowed = {t.name for t in _fsdp2.TARGETS} | set(worker.TORCH_MUSA_ADDITIONS)
    for names in _record(tmp_path)["torch_musa_owned"].values():
        assert set(names) <= allowed


@pytest.mark.slow
@pytest.mark.parametrize("order", ["musa,cpu", "cpu,musa"])
def test_musa_and_cpu_meshes_in_one_process(tmp_path, torch_musa_runs, order):
    _require_musa(devices=2)
    musa_baseline = torch_musa_runs("default", "2")
    note = _check_baseline("default, musa mesh 2", musa_baseline)
    upstream = tmp_path / "upstream"
    upstream_env = _musa_env({"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    args = ("--arm", "upstream", "--shape", "2")
    _passed(_run(upstream, upstream_env, *args, timeout=MUSA_MESH_TIMEOUT))
    mixed = tmp_path / "mixed"
    args = ("--arm", "torchada", "--mesh", order, "--shape", "2")
    _passed(_run(mixed, _musa_env({}), *args, timeout=MUSA_MESH_TIMEOUT), note)
    for index, mesh in enumerate(order.split(",")):
        part = tmp_path / mesh
        _extract_mesh(mixed, index, part)
        baseline = upstream if mesh == "cpu" else musa_baseline.out
        exact = True if mesh == "cpu" else musa_baseline.deterministic
        assert worker.compare(str(baseline), str(part), exact=exact) == [], (mesh, note)


def _extract_mesh(out: Path, index: int, dest: Path) -> None:
    """Write mesh ``index`` of a multi-mesh run as a single-mesh run."""
    records, tensors, trace = worker.load(str(out))
    dest.mkdir()
    for rank, (record, tensor) in enumerate(zip(records, tensors)):
        record = dict(record, meshes=[record["meshes"][index]], mesh=[record["mesh"][index]])
        (dest / f"rank{rank}.json").write_text(json.dumps(record))
        torch.save([tensor[index]], dest / f"rank{rank}.pt")
    part = {f"0:{k.split(':')[1]}": v for k, v in trace.items() if k.startswith(f"{index}:")}
    (dest / "trace.json").write_text(json.dumps(part))
