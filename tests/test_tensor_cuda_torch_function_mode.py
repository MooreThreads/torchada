"""Regression tests for ``.cuda()`` while a TorchFunctionMode is active.

torch_musa implements ``Tensor.musa()`` on top of its own C binding, which
mis-dispatches when another ``__torch_function__`` mode is active and raises
``TypeError: musa(): argument 'temp' (position 1) must be Tensor, not module``.
``with torch.device("cuda"):`` installs exactly such a mode, so decoder
construction wrapped in that context used to break on the first ``.cuda()``
call. The patched ``.cuda()`` must move tensors and modules with ``.to()``.
"""

import pytest
import torch


def _require_musa():
    import torchada

    if not torchada.is_musa_platform():
        pytest.skip("MUSA platform required")

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device required")


class _PassthroughMode(torch.overrides.TorchFunctionMode):
    """Minimal mode: forward every call to the underlying implementation."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


@pytest.mark.gpu
def test_tensor_cuda_within_device_context():
    """``with torch.device(...): t.cuda()`` must move the tensor to MUSA."""
    _require_musa()
    host = torch.randn(8, 8)

    for device in (torch.device("cuda"), torch.device("musa")):
        with device:
            moved = host.cuda()
        assert moved.device.type == "musa"
        assert torch.equal(moved.cpu(), host)


@pytest.mark.gpu
def test_tensor_cuda_within_torch_function_mode():
    """All ``Tensor.cuda()`` device argument forms must survive a mode."""
    _require_musa()
    host = torch.randn(8, 8)

    with _PassthroughMode():
        assert host.cuda().device.type == "musa"
        assert host.cuda(0).device == torch.device("musa:0")
        assert host.cuda(torch.device("cuda:0")).device == torch.device("musa:0")
        assert host.cuda("cuda:0").device == torch.device("musa:0")
        assert host.cuda(non_blocking=True).device.type == "musa"


@pytest.mark.gpu
def test_module_cuda_within_device_context():
    """Module construction inside ``torch.device("cuda")`` must not fail."""
    _require_musa()

    with torch.device("cuda"):
        module = torch.nn.Linear(4, 4).cuda()
    assert all(param.device.type == "musa" for param in module.parameters())


@pytest.mark.gpu
def test_module_cuda_within_torch_function_mode():
    """``Module.cuda()`` must use the same safe move under a mode."""
    _require_musa()
    module = torch.nn.Linear(4, 4)

    with _PassthroughMode():
        module.cuda()
    assert all(param.device.type == "musa" for param in module.parameters())


def test_wrappers_delegate_to_original_cuda_on_non_musa(monkeypatch):
    """The non-MUSA path must call the original method unchanged."""
    from torchada import _patch

    monkeypatch.setattr(_patch, "is_musa_platform", lambda: False)

    calls = []

    def original_cuda(self, device=None, non_blocking=False):
        calls.append((self, device, non_blocking))
        return "moved"

    tensor, module = object(), object()
    assert _patch._wrap_tensor_cuda(original_cuda)(tensor, device=1, non_blocking=True) == "moved"
    assert _patch._wrap_module_cuda(original_cuda)(module, device=1) == "moved"
    assert calls == [(tensor, 1, True), (module, 1, False)]
