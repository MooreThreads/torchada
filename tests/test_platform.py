"""
Tests for platform detection and initialization.
"""

import re
from pathlib import Path


class TestPlatformDetection:
    """Test platform detection functionality."""

    def test_import_torchada(self):
        """Test that torchada can be imported."""
        import torchada

        assert torchada is not None
        assert hasattr(torchada, "__version__")

    def test_detect_platform(self):
        """Test platform detection."""
        from torchada import Platform, detect_platform

        platform = detect_platform()
        assert platform in [Platform.CUDA, Platform.MUSA, Platform.CPU]

    def test_platform_detection_functions(self):
        """Test individual platform detection functions."""
        from torchada import (
            Platform,
            detect_platform,
            is_cpu_platform,
            is_cuda_platform,
            is_musa_platform,
        )

        platform = detect_platform()
        if platform == Platform.MUSA:
            assert is_musa_platform()
            assert not is_cuda_platform()
            assert not is_cpu_platform()
        elif platform == Platform.CUDA:
            assert not is_musa_platform()
            assert is_cuda_platform()
            assert not is_cpu_platform()
        else:
            assert not is_musa_platform()
            assert not is_cuda_platform()
            assert is_cpu_platform()

    def test_patches_applied(self):
        """Test that patches are automatically applied on import."""
        import torchada

        assert torchada.is_patched()

    def test_get_version(self):
        """Test version retrieval."""
        import torchada

        version = torchada.get_version()
        assert version == torchada.__version__
        assert version == "0.1.83"
        assert isinstance(version, str)

    def test_project_version_matches_runtime_version(self):
        """The package metadata and runtime version must stay in sync."""
        import torchada

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        match = re.search(
            r'^version = "([^"]+)"$', pyproject.read_text(encoding="utf-8"), re.MULTILINE
        )

        assert match is not None
        assert match.group(1) == torchada.__version__

    def test_get_platform(self):
        """Test get_platform helper."""
        import torchada

        platform = torchada.get_platform()
        assert platform == torchada.detect_platform()

    def test_get_backend(self):
        """Test get_backend returns a module."""
        import torchada

        backend = torchada.get_backend()
        assert backend is not None
        # Should have is_available method
        assert hasattr(backend, "is_available")

    def test_is_gpu_device(self):
        """Test is_gpu_device helper function."""
        import torch

        import torchada

        # Test with torch.device objects
        cpu_device = torch.device("cpu")
        assert torchada.is_gpu_device(cpu_device) is False

        # Test with string device specs
        assert torchada.is_gpu_device("cpu") is False
        assert torchada.is_gpu_device("cuda") is True
        assert torchada.is_gpu_device("cuda:0") is True
        assert torchada.is_gpu_device("musa") is True
        assert torchada.is_gpu_device("musa:0") is True

        # Test with tensor on CPU
        cpu_tensor = torch.randn(10)
        assert torchada.is_gpu_device(cpu_tensor) is False

        # Test with GPU device if available
        if torchada.is_musa_platform():
            musa_device = torch.device("musa:0")
            assert torchada.is_gpu_device(musa_device) is True

            # torch.device("cuda") is translated to musa on MUSA platform
            cuda_device = torch.device("cuda:0")
            assert torchada.is_gpu_device(cuda_device) is True

    def test_is_cuda_like_device_alias(self):
        """Test is_cuda_like_device is alias for is_gpu_device."""
        import torchada

        # Should be the same function
        assert torchada.is_cuda_like_device("cuda") == torchada.is_gpu_device("cuda")
        assert torchada.is_cuda_like_device("cpu") == torchada.is_gpu_device("cpu")
