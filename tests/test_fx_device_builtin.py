"""
Tests for FX code generation of torch.device constants under the torch.device patch.

FX prints a device constant as ``device(type=...)`` and binds ``device`` to its
registered builtin. With torch.device replaced by the translating factory, the
builtin must be that factory, or GraphModules holding a device constant fail
with ``NameError: name 'device' is not defined`` (Dynamo's eager and aot_eager
backends run such GraphModules directly).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import torchada

BACKENDS = ("eager", "aot_eager")
DYNAMIC = (False, True)

# Applying the device patch on a non-MUSA platform replaces torch.device for the
# whole process, so the CPU cases run in a child interpreter.
_CPU_SCRIPT = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])

import torch
# torchada loads Dynamo (through Inductor) before it replaces torch.device, so
# Dynamo's constant types hold the original class; keep that order here.
import torch._dynamo
import torch.fx
import torch.fx.graph as fx_graph


class _ZerosOnCpu(torch.nn.Module):
    def forward(self, x):
        cpu = torch.device("cpu")
        return x.to(cpu) + torch.zeros_like(x, device=cpu)


# Source and import block of GraphModules that hold a device constant.
def generated_code():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    empty = graph.call_function(
        torch.ops.aten.empty.memory_format, ([2, 3],), {"device": torch.empty(0).device}
    )
    graph.output((empty, x))
    modules = (
        torch.fx.GraphModule(torch.nn.Module(), graph),
        torch.fx.symbolic_trace(_ZerosOnCpu()),
    )
    # The import block is the part of GraphModule.__reduce__ that FX graph caches hash.
    return [(gm.code, gm.__reduce__()[1][1]) for gm in modules]


# A torchada imported at interpreter start (e.g. by a sitecustomize) would make
# this capture a patched one, so the comparisons below report SKIP instead.
torchada_preloaded = "torchada" in sys.modules
unpatched_code = generated_code()

from torchada import _patch

if _patch._original_torch_device is None:
    _patch._patch_torch_device()
original_device = _patch._original_torch_device
results = {}


class Skip(Exception):
    pass


def record(name, check):
    try:
        check()
        results[name] = "ok"
    except Skip as exc:
        results[name] = "SKIP: {}".format(exc)
    except Exception as exc:
        detail = (str(exc).splitlines() or [""])[0]
        results[name] = "FAILED: {}: {}".format(type(exc).__name__, detail)


def check_patched():
    assert torch.device is _patch.DeviceFactoryWrapper
    assert isinstance(torch.device("cpu"), torch.device)
    assert type(torch.device("cpu")) is original_device


def check_builtin():
    assert fx_graph._custom_builtins["device"].obj is torch.device
    assert fx_graph._illegal_names["device"] is torch.device


def check_graph():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    empty = graph.call_function(
        torch.ops.aten.empty.memory_format, ([2, 3],), {"device": original_device("cpu")}
    )
    from_original = graph.call_function(original_device, ("cpu",))
    from_factory = graph.call_function(torch.device, ("cpu",))
    graph.output((empty, from_original, from_factory, x))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    assert "device(type='cpu')" in gm.code
    out, dev_original, dev_factory, _ = gm(torch.ones(1))
    assert out.device == dev_original == dev_factory == original_device("cpu")
    assert type(dev_original) is original_device
    assert type(dev_factory) is original_device


def check_code_identical():
    if torchada_preloaded:
        raise Skip("torchada was imported before the unpatched capture")
    patched_code = generated_code()
    for (code, _), (patched, _) in zip(unpatched_code, patched_code):
        assert code == patched, (code, patched)


def check_import_block_identical():
    if torchada_preloaded:
        raise Skip("torchada was imported before the unpatched capture")
    patched_code = generated_code()
    for (_, imports), (_, patched) in zip(unpatched_code, patched_code):
        assert imports == patched, (imports, patched)


def check_symbolic_trace():
    gm = torch.fx.symbolic_trace(_ZerosOnCpu())
    assert "device(type='cpu')" in gm.code
    x = torch.ones(3, 2)
    assert torch.equal(gm(x), _ZerosOnCpu()(x))


def fill_rows(x):
    return torch.empty(x.shape[0], 32, device=x.device).fill_(1.0) + x.sum()


def scale_rows(x):
    return (x * torch.ones(x.shape[0], 1, device=x.device)).sum()


def check_compile(backend, dynamic):
    torch._dynamo.reset()
    compiled = torch.compile(fill_rows, backend=backend, dynamic=dynamic, fullgraph=True)
    for rows in (4, 7):
        x = torch.randn(rows, 3)
        assert torch.equal(compiled(x), fill_rows(x))


def zeros_from_factory(x):
    return torch.zeros(2, device=torch.device("cpu")) + x[0, :2]


def check_factory_in_region(backend):
    torch._dynamo.reset()
    compiled = torch.compile(zeros_from_factory, backend=backend, fullgraph=True)
    x = torch.randn(3, 4)
    assert torch.equal(compiled(x), zeros_from_factory(x))


def check_backward(dynamic):
    torch._dynamo.reset()
    compiled = torch.compile(scale_rows, backend="aot_eager", dynamic=dynamic, fullgraph=True)
    x = torch.randn(5, 3, requires_grad=True)
    compiled(x).backward()
    expected = x.detach().clone().requires_grad_(True)
    scale_rows(expected).backward()
    assert torch.equal(x.grad, expected.grad)


record("patched", check_patched)
record("builtin", check_builtin)
record("graph", check_graph)
record("code_identical", check_code_identical)
record("import_block_identical", check_import_block_identical)
record("symbolic_trace", check_symbolic_trace)
for backend in ("eager", "aot_eager"):
    for dynamic in (False, True):
        record("{}-dynamic={}".format(backend, dynamic), lambda: check_compile(backend, dynamic))
    record("factory-{}".format(backend), lambda: check_factory_in_region(backend))
for dynamic in (False, True):
    record("aot_eager-backward-dynamic={}".format(dynamic), lambda: check_backward(dynamic))
print("RESULTS " + json.dumps(results))
"""


@pytest.fixture(scope="module")
def patched_cpu_results(tmp_path_factory):
    src_root = str(Path(torchada.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path_factory.mktemp("inductor-cache"))
    proc = subprocess.run(
        [sys.executable, "-c", _CPU_SCRIPT, src_root],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULTS ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    return json.loads(lines[-1][len("RESULTS ") :])


@pytest.mark.parametrize(
    "case",
    ["patched", "builtin", "graph", "code_identical", "import_block_identical", "symbolic_trace"],
)
def test_fx_codegen_with_patched_device(patched_cpu_results, case):
    result = patched_cpu_results[case]
    if result.startswith("SKIP"):
        pytest.skip(result)
    assert result == "ok"


@pytest.mark.parametrize("dynamic", DYNAMIC)
@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_device_constant_with_patched_device(patched_cpu_results, backend, dynamic):
    assert patched_cpu_results[f"{backend}-dynamic={dynamic}"] == "ok"


@pytest.mark.parametrize("dynamic", DYNAMIC)
def test_aot_eager_backward_with_patched_device(patched_cpu_results, dynamic):
    assert patched_cpu_results[f"aot_eager-backward-dynamic={dynamic}"] == "ok"


@pytest.mark.parametrize("backend", BACKENDS)
def test_device_factory_in_compiled_region_with_patched_device(patched_cpu_results, backend):
    assert patched_cpu_results[f"factory-{backend}"] == "ok"


def _fill_rows(x):
    return torch.empty(x.shape[0], 32, device=x.device).fill_(1.0) + x.sum()


@pytest.mark.musa
def test_fx_device_builtin_is_patched_device():
    import torch.fx.graph as fx_graph

    assert fx_graph._custom_builtins["device"].obj is torch.device
    assert fx_graph._illegal_names["device"] is torch.device


@pytest.mark.musa
def test_dynamo_constant_types_hold_original_device():
    # Dynamo records torch.device as a constant type when it is first imported;
    # torchada loads Dynamo before replacing torch.device so the original class is kept.
    import torch._dynamo.utils as dynamo_utils

    from torchada import _patch

    assert _patch._original_torch_device in dynamo_utils.common_constant_types


@pytest.mark.musa
@pytest.mark.parametrize("dynamic", DYNAMIC)
@pytest.mark.parametrize("backend", BACKENDS)
def test_compile_device_constant_on_musa(backend, dynamic):
    torch._dynamo.reset()
    try:
        compiled = torch.compile(_fill_rows, backend=backend, dynamic=dynamic, fullgraph=True)
        for rows in (4, 7):
            x = torch.randn(rows, 3, device="cuda")
            out = compiled(x)
            assert out.device.type == "musa"
            assert torch.equal(out, _fill_rows(x))
    finally:
        torch._dynamo.reset()


def _zeros_from_factory(x):
    return torch.zeros(2, device=torch.device("cuda")) + x[0, :2]


@pytest.mark.musa
@pytest.mark.parametrize("backend", BACKENDS)
def test_device_factory_in_compiled_region_on_musa(backend):
    torch._dynamo.reset()
    try:
        compiled = torch.compile(_zeros_from_factory, backend=backend, fullgraph=True)
        x = torch.randn(3, 4, device="cuda")
        out = compiled(x)
        assert out.device.type == "musa"
        assert torch.equal(out, _zeros_from_factory(x))
    finally:
        torch._dynamo.reset()
