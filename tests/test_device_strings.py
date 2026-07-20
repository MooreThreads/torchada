"""
Tests for device string patching (cuda -> musa translation).
"""

import pytest
import torch


def _make_torch_device(device: str):
    return torch.device(device)


class TestTensorDevicePatching:
    """Test tensor device string patching."""

    def test_tensor_cuda_method(self):
        """Test .cuda() method on tensors."""
        import torch

        import torchada

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.cuda()
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"

    def test_tensor_to_cuda_string(self):
        """Test .to('cuda') on tensors."""
        import torch

        import torchada

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to("cuda")
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"

    def test_tensor_to_cuda_device_index(self):
        """Test .to('cuda:0') on tensors."""
        import torch

        import torchada

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to("cuda:0")
            assert y.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert y.device.type == "musa"
                assert y.device.index == 0

    def test_tensor_to_torch_device(self):
        """Test .to(torch.device('cuda')) on tensors."""
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to(torch.device("cuda"))
            assert y.device.type in ("cuda", "musa")


class TestModuleDevicePatching:
    """Test nn.Module device patching."""

    def test_module_cuda_method(self):
        """Test .cuda() method on modules."""
        import torch
        import torch.nn as nn

        import torchada

        if torch.cuda.is_available():
            model = nn.Linear(10, 5)
            model = model.cuda()
            param_device = next(model.parameters()).device
            assert param_device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert param_device.type == "musa"

    def test_module_to_cuda(self):
        """Test .to('cuda') method on modules."""
        import torch
        import torch.nn as nn

        if torch.cuda.is_available():
            model = nn.Linear(10, 5)
            model = model.to("cuda")
            param_device = next(model.parameters()).device
            assert param_device.type in ("cuda", "musa")


class TestFactoryFunctions:
    """Test tensor factory function patching."""

    def test_torch_empty_device_cuda(self):
        """Test torch.empty(device='cuda')."""
        import torch

        import torchada

        if torch.cuda.is_available():
            x = torch.empty(2, 2, device="cuda")
            assert x.device.type in ("cuda", "musa")
            if torchada.is_musa_platform():
                assert x.device.type == "musa"

    def test_torch_zeros_device_cuda(self):
        """Test torch.zeros(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.zeros(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                # MUSA driver issue, not torchada
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_ones_device_cuda(self):
        """Test torch.ones(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.ones(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_randn_device_cuda(self):
        """Test torch.randn(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.randn(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_rand_device_cuda(self):
        """Test torch.rand(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.rand(2, 2, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_full_device_cuda(self):
        """Test torch.full(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.full((2, 2), 3.14, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUDNN" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_arange_device_cuda(self):
        """Test torch.arange(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.arange(10, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_torch_linspace_device_cuda(self):
        """Test torch.linspace(device='cuda')."""
        import torch

        if torch.cuda.is_available():
            try:
                x = torch.linspace(0, 1, 10, device="cuda")
                assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestTranslateDeviceFunction:
    """Unit tests for the _translate_device function."""

    def test_translate_device_none(self):
        """Test that None is passed through unchanged."""
        from torchada._patch import _translate_device

        assert _translate_device(None) is None

    def test_translate_device_cuda_string(self):
        """Test 'cuda' string translation."""
        import torchada
        from torchada._patch import _translate_device

        result = _translate_device("cuda")
        if torchada.is_musa_platform():
            assert result == "musa"
        else:
            assert result == "cuda"

    def test_translate_device_cuda_with_index(self):
        """Test 'cuda:0' and 'cuda:1' string translation."""
        import torchada
        from torchada._patch import _translate_device

        result0 = _translate_device("cuda:0")
        result1 = _translate_device("cuda:1")

        if torchada.is_musa_platform():
            assert result0 == "musa:0"
            assert result1 == "musa:1"
        else:
            assert result0 == "cuda:0"
            assert result1 == "cuda:1"

    def test_translate_device_cpu_unchanged(self):
        """Test that 'cpu' is unchanged."""
        from torchada._patch import _translate_device

        assert _translate_device("cpu") == "cpu"

    def test_translate_device_musa_unchanged(self):
        """Test that 'musa' is unchanged."""
        from torchada._patch import _translate_device

        assert _translate_device("musa") == "musa"
        assert _translate_device("musa:0") == "musa:0"

    def test_translate_device_integer_unchanged(self):
        """Test that integer device IDs are unchanged."""
        from torchada._patch import _translate_device

        assert _translate_device(0) == 0
        assert _translate_device(1) == 1

    def test_translate_device_torch_device_cuda(self):
        """Test torch.device('cuda') translation."""
        import torch

        import torchada

        # Get original torch.device class
        from torchada._patch import _original_torch_device, _translate_device

        if _original_torch_device is None:
            # Not patched yet, use current torch.device
            device_cls = torch.device
        else:
            device_cls = _original_torch_device

        cuda_device = device_cls("cuda")
        result = _translate_device(cuda_device)

        if torchada.is_musa_platform():
            assert result.type == "musa"
        else:
            assert result.type == "cuda"

    def test_translate_device_torch_device_cuda_with_index(self):
        """Test torch.device('cuda', 0) translation."""
        import torch

        import torchada
        from torchada._patch import _original_torch_device, _translate_device

        if _original_torch_device is None:
            device_cls = torch.device
        else:
            device_cls = _original_torch_device

        cuda_device = device_cls("cuda", 0)
        result = _translate_device(cuda_device)

        if torchada.is_musa_platform():
            assert result.type == "musa"
            assert result.index == 0
        else:
            assert result.type == "cuda"
            assert result.index == 0

    def test_translate_device_torch_device_cpu_unchanged(self):
        """Test torch.device('cpu') is unchanged."""
        import torch

        from torchada._patch import _translate_device

        cpu_device = torch.device("cpu")
        result = _translate_device(cpu_device)
        assert result.type == "cpu"


class TestDeviceIndexVariants:
    """Test various cuda:N device index formats."""

    def test_cuda_colon_zero(self):
        """Test cuda:0 works."""
        import torch

        if torch.cuda.is_available():
            x = torch.randn(2, 2)
            y = x.to("cuda:0")
            assert y.device.type in ("cuda", "musa")
            assert y.device.index == 0

    def test_factory_with_cuda_index(self):
        """Test torch.empty(device='cuda:0')."""
        import torch

        if torch.cuda.is_available():
            x = torch.empty(2, 2, device="cuda:0")
            assert x.device.type in ("cuda", "musa")
            assert x.device.index == 0

    def test_torch_device_cuda_zero(self):
        """Test torch.device('cuda:0')."""
        import torch

        import torchada

        device = torch.device("cuda:0")
        if torchada.is_musa_platform():
            assert device.type == "musa"
            assert device.index == 0
        else:
            assert device.type == "cuda"
            assert device.index == 0

    def test_torch_device_with_index_arg(self):
        """Test torch.device('cuda', 0)."""
        import torch

        import torchada

        device = torch.device("cuda", 0)
        if torchada.is_musa_platform():
            assert device.type == "musa"
            assert device.index == 0
        else:
            assert device.type == "cuda"
            assert device.index == 0

    def test_torch_device_from_original_cuda_device(self):
        """Test torch.device(torch.device('cuda:1')) translation on MUSA."""
        import torch

        import torchada
        from torchada._patch import _original_torch_device

        if not torchada.is_musa_platform() or _original_torch_device is None:
            pytest.skip("MUSA platform required with patched torch.device")

        cuda_device = _original_torch_device("cuda:1")
        wrapped = torch.device(cuda_device)

        assert wrapped.type == "musa"
        assert wrapped.index == 1

    def test_torch_device_type_keyword(self):
        """Test torch.device with type= keyword argument."""
        import torch

        import torchada

        # type= keyword with cpu
        device1 = torch.device(type="cpu")
        assert device1.type == "cpu"

        # type= keyword with cuda (should translate to musa on MUSA platform)
        device2 = torch.device(type="cuda", index=0)
        if torchada.is_musa_platform():
            assert device2.type == "musa"
        else:
            assert device2.type == "cuda"
        assert device2.index == 0

    def test_torch_device_remains_a_torchscript_builtin(self):
        """Test importing torchada keeps torch.device scriptable."""
        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("MUSA platform required with patched torch.device")

        scripted = torch.jit.script(_make_torch_device)
        assert scripted("musa") == torch.device("musa")


class TestDeviceContextManager:
    """Test device context manager (with torch.device(...):) works."""

    @pytest.mark.gpu
    def test_device_context_with_musa(self):
        """Test that device context manager works with musa device."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("MUSA platform required")

        device = torch.device("musa:0")
        with device:
            t = torch.empty(10)
            assert t.device.type == "musa"
            assert t.device.index == 0

    @pytest.mark.gpu
    def test_device_context_with_cuda_translated(self):
        """Test that device context manager works with cuda device (translated to musa)."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("MUSA platform required")

        # torch.device("cuda:0") gets translated to musa:0
        device = torch.device("cuda:0")
        assert device.type == "musa"  # Verify translation happened

        with device:
            t = torch.empty(10)
            assert t.device.type == "musa"
            assert t.device.index == 0

    def test_original_functions_in_device_constructors(self):
        """Test that original functions are added to _device_constructors."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("MUSA platform required")

        try:
            from torch.utils._device import _device_constructors

            constructors = _device_constructors()

            # Check that original (unwrapped) functions are in constructors
            if hasattr(torch.empty, "__wrapped__"):
                assert torch.empty.__wrapped__ in constructors
            if hasattr(torch.zeros, "__wrapped__"):
                assert torch.zeros.__wrapped__ in constructors
            if hasattr(torch.ones, "__wrapped__"):
                assert torch.ones.__wrapped__ in constructors
        except ImportError:
            pytest.skip("torch.utils._device not available")
