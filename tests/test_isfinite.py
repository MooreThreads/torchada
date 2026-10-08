import os

import pytest
import torch


def _require_musa():
    import torchada

    if not torchada.is_musa_platform():
        pytest.skip("MUSA platform required")

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA platform required")


_SPECIALS = [0.0, -0.0, 1.5, -2.25, 1e-30, float("nan"), float("inf"), -float("inf")]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_isfinite_matches_cpu(dtype):
    _require_musa()
    host = torch.tensor(_SPECIALS * 16, dtype=dtype).reshape(8, 16)
    host[0, 0] = torch.finfo(dtype).max
    host[0, 1] = torch.finfo(dtype).tiny
    host[0, 2] = torch.finfo(dtype).smallest_normal / 2  # subnormal
    host[0, 3] = -torch.finfo(dtype).max
    assert 0 < host[0, 2] < torch.finfo(dtype).smallest_normal
    device = host.to("cuda")

    expected = torch.isfinite(host)
    for value in (device, device.t(), device[:, ::3]):
        cpu_view = value.cpu()
        assert torch.equal(torch.isfinite(value).cpu(), torch.isfinite(cpu_view))
        assert torch.equal(value.isfinite().cpu(), torch.isfinite(cpu_view))
    assert torch.equal(torch.isfinite(device).cpu(), expected)
    assert torch.isfinite(device).dtype == torch.bool


def test_isfinite_scalar_and_empty():
    _require_musa()
    assert not bool(torch.isfinite(torch.tensor(float("nan"), device="cuda")))
    empty = torch.empty(0, 4, device="cuda")
    assert torch.isfinite(empty).shape == (0, 4)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_isfinite_non_floating_keeps_original(dtype):
    _require_musa()
    value = torch.ones(5, 3, dtype=dtype, device="cuda")
    out = torch.isfinite(value)
    assert out.dtype == torch.bool
    assert bool(out.all())


@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2"])
def test_isfinite_fp8_keeps_original(name):
    _require_musa()
    from torchada import _patch

    dtype = getattr(torch, name, None)
    if dtype is None:
        pytest.skip(f"torch.{name} is not available")
    try:
        value = torch.zeros(4).to(dtype).to("cuda")
    except (RuntimeError, NotImplementedError):
        pytest.skip(f"{name} tensors are not supported on this device")
    assert not _patch._uses_async_isfinite(value)
    try:
        expected = _patch._original_torch_isfinite(value)
    except (RuntimeError, NotImplementedError) as exc:
        with pytest.raises(type(exc)):
            torch.isfinite(value)
    else:
        assert torch.equal(torch.isfinite(value).cpu(), expected.cpu())


def test_isfinite_does_not_track_autograd():
    _require_musa()
    value = torch.randn(32, device="cuda", requires_grad=True)
    out = torch.isfinite(value)
    assert not out.requires_grad
    assert out.grad_fn is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("entry", ["function", "method"])
def test_isfinite_leaves_queued_work_pending(dtype, entry):
    _require_musa()
    if "1" in (os.environ.get("MUSA_LAUNCH_BLOCKING"), os.environ.get("CUDA_LAUNCH_BLOCKING")):
        pytest.skip("launch-blocking mode drains the queue after every launch")
    lhs = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(4096, 8, device="cuda").to(dtype)
    torch.isfinite(value)
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    for _ in range(64):
        lhs = lhs @ lhs.t()
        lhs = lhs / lhs.abs().amax().clamp_min(1)
    assert not stream.query(), "setup did not leave work queued"
    if entry == "function":
        torch.isfinite(value)
    else:
        value.isfinite()
    pending = not stream.query()
    torch.cuda.synchronize()
    assert pending, "isfinite drained the device queue"


@pytest.mark.parametrize(
    "dtype,routed",
    [
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.float32, True),
        (torch.float64, True),
        (torch.int32, False),
        (torch.int64, False),
        (torch.bool, False),
        (torch.complex64, False),
    ],
)
@pytest.mark.parametrize("entry", ["function", "method"])
def test_isfinite_routes_only_musa_real_floats(monkeypatch, dtype, routed, entry):
    _require_musa()
    from torchada import _patch

    calls = []
    original = _patch._finite_mask
    monkeypatch.setattr(
        _patch, "_finite_mask", lambda value: calls.append(value.dtype) or original(value)
    )
    try:
        value = torch.ones(4).to(dtype).to("cuda")
    except (RuntimeError, TypeError, NotImplementedError):
        pytest.skip(f"{dtype} tensors are not supported on this device")
    try:
        torch.isfinite(value) if entry == "function" else value.isfinite()
    except (RuntimeError, NotImplementedError):
        assert not routed
    assert bool(calls) is routed
    calls.clear()
    host = torch.ones(4).to(dtype)
    torch.isfinite(host) if entry == "function" else host.isfinite()
    assert not calls


def test_isfinite_is_scriptable():
    _require_musa()

    @torch.jit.script
    def scripted(value: torch.Tensor) -> torch.Tensor:
        return torch.isfinite(value)

    value = torch.tensor([1.0, float("inf"), float("nan")], device="cuda")
    assert scripted(value).cpu().tolist() == [True, False, False]


def test_compiled_isfinite_graph_is_correct_and_cacheable():
    _require_musa()
    import subprocess
    import sys

    if getattr(getattr(torch, "compiler", None), "save_cache_artifacts", None) is None:
        pytest.skip("torch.compiler.save_cache_artifacts not available")
    probe = (
        "import torchada, torch\n"
        "def f(x):\n"
        "    return torch.where(torch.isfinite(x), x, torch.zeros_like(x))\n"
        "x = torch.tensor([1.0, float('inf'), float('nan'), -2.0], device='cuda')\n"
        "out = torch.compile(f, fullgraph=True)(x)\n"
        "assert out.cpu().tolist() == [1.0, 0.0, 0.0, -2.0], out\n"
        "art = torch.compiler.save_cache_artifacts()\n"
        "assert art is not None, 'no cache artifacts collected'\n"
        "assert len(art[1].aot_autograd_artifacts) == 1, art[1]\n"
    )
    env = dict(os.environ)
    env.pop("TORCHDYNAMO_DISABLE", None)
    result = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
