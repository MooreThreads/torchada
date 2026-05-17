"""
Platform detection utilities.

MUSA is treated as a platform when torch_musa is installed, even when no Moore
Threads GPU is currently available. That keeps build-only environments on the
same compatibility path as runtime GPU environments.
"""

import os
from enum import Enum
from functools import lru_cache


class Platform(Enum):
    """Supported GPU platforms."""

    CUDA = "cuda"
    MUSA = "musa"
    CPU = "cpu"


@lru_cache(maxsize=1)
def detect_platform() -> Platform:
    """
    Detect the current GPU platform.

    Priority:
    1. TORCHADA_PLATFORM environment variable (force specific platform)
    2. MUSA availability (Moore Threads GPU)
    3. CUDA availability (NVIDIA GPU)
    4. CPU fallback

    Returns:
        Platform: The detected or configured platform.
    """
    # Honor explicit platform overrides first.
    forced_platform = os.environ.get("TORCHADA_PLATFORM", "").lower()
    if forced_platform == "cuda":
        return Platform.CUDA
    elif forced_platform == "musa":
        return Platform.MUSA
    elif forced_platform == "cpu":
        return Platform.CPU

    # Prefer MUSA before CUDA so torch_musa installations take the adapter path.
    if _is_musa_available():
        return Platform.MUSA

    # Fall back to native CUDA only when MUSA is not present.
    if _is_cuda_available():
        return Platform.CUDA

    return Platform.CPU


def _is_musa_available() -> bool:
    """
    Return whether the environment should use the MUSA compatibility path.

    Detects the MUSA platform by checking if torch_musa is installed,
    rather than requiring a GPU to be present. This allows torchada to
    work correctly in environments where torch_musa is installed but
    no Moore Threads GPU card is available (e.g., build-only environments,
    CI/CD, CPU-only testing).

    Detection signals (in order):
    1. torch.version.musa is set (torch was built with MUSA support)
    2. torch_musa is importable
    """
    try:
        import torch

        # Primary signal: torch.version.musa is set by torch_musa at build time,
        # regardless of whether a GPU card is present.
        if hasattr(torch.version, "musa") and torch.version.musa is not None:
            return True

        # Secondary signal: torch_musa is importable.
        try:
            import torch_musa  # noqa: F401

            return True
        except ImportError:
            pass

        return False
    except ImportError:
        return False


def _is_cuda_available() -> bool:
    """Return whether native CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, AttributeError):
        return False


def is_musa_platform() -> bool:
    """Return whether the detected platform is MUSA."""
    return detect_platform() == Platform.MUSA


def is_cuda_platform() -> bool:
    """Return whether the detected platform is native CUDA."""
    return detect_platform() == Platform.CUDA


def is_cpu_platform() -> bool:
    """Return whether the detected platform is CPU-only."""
    return detect_platform() == Platform.CPU


def get_device_name() -> str:
    """Return the detected device type string."""
    return detect_platform().value


def get_torch_device_module():
    """
    Get the appropriate torch device module (torch.cuda or torch.musa).

    Returns:
        The torch.cuda or torch.musa module.

    Raises:
        RuntimeError: If no GPU platform is available.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        import torch

        return torch.musa
    elif platform == Platform.CUDA:
        import torch

        return torch.cuda
    else:
        raise RuntimeError("No GPU platform available. Running on CPU only.")


def is_gpu_device(device) -> bool:
    """
    Return whether a device is a CUDA-like GPU device.

    This is a helper function for code that needs to check if a device
    is a GPU device. On MUSA platform, device.type == "cuda" comparisons
    fail because the device type is "musa", not "cuda".

    Use this function instead of `device.type == "cuda"` for portable code.

    Args:
        device: A torch.device object, or an object with a .device attribute

    Returns:
        True if the device is cuda or musa, False otherwise

    Example:
        if torchada.is_gpu_device(tensor.device):
            ...
    """
    import torch

    # Accept tensors, modules, and other objects exposing ``.device``.
    if hasattr(device, "device"):
        device = device.device

    # Accept torch.device objects directly.
    if isinstance(device, torch.device):
        return device.type in ("cuda", "musa")

    # Accept string device specifications.
    if isinstance(device, str):
        return (
            device == "cuda"
            or device.startswith("cuda:")
            or device == "musa"
            or device.startswith("musa:")
        )

    return False


def is_cuda_like_device(device) -> bool:
    """Return whether ``device`` names a CUDA-like GPU device."""
    return is_gpu_device(device)
