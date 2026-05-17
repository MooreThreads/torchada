"""
torchada.cuda - CUDA-compatible API that works on both CUDA and MUSA platforms.

This module provides the same interface as torch.cuda but automatically
routes to torch.musa on Moore Threads hardware.

Note: After importing torchada, you can use standard torch.cuda APIs directly.
This module is provided for internal use and backwards compatibility.

Usage (preferred):
    import torchada  # Apply patches.
    import torch

    # torch.cuda APIs work on MUSA after importing torchada.
    torch.cuda.set_device(0)
    tensor = tensor.cuda()
"""

from typing import Optional, Union

from .._device_compat import _translate_device
from .._platform import Platform, detect_platform


def _get_backend():
    """Get the appropriate backend module (torch.cuda or torch.musa)."""
    platform = detect_platform()

    if platform == Platform.MUSA:
        import torch
        import torch_musa  # noqa: F401 - registers torch.musa

        return torch.musa
    elif platform == Platform.CUDA:
        import torch

        return torch.cuda
    else:
        # Preserve torch.cuda-shaped APIs even when no GPU backend is present.
        import torch

        return torch.cuda


def _backend_attr(name: str):
    """Return an attribute from the active CUDA-compatible backend."""
    return getattr(_get_backend(), name)


def _call_backend(name: str, *args, **kwargs):
    """Call a method on the active CUDA-compatible backend."""
    return _backend_attr(name)(*args, **kwargs)


def _call_backend_with_fallback(primary: str, fallback: str, *args, **kwargs):
    """Call a backend method, falling back to a compatible replacement if absent."""
    backend = _get_backend()
    fn = getattr(backend, primary, None)
    if fn is None:
        fn = getattr(backend, fallback)
    return fn(*args, **kwargs)


# Core device functions.
def is_available() -> bool:
    """Check if CUDA/MUSA is available."""
    return _call_backend("is_available")


def device_count() -> int:
    """Return the number of GPUs available."""
    backend = _get_backend()
    if hasattr(backend, "device_count"):
        return backend.device_count()
    return 0


def current_device() -> int:
    """Return the index of the currently selected device."""
    return _call_backend("current_device")


def set_device(device: Union[int, str, "torch.device"]) -> None:
    """Set the current device."""
    _call_backend("set_device", _translate_device(device))


def get_device_name(device: Optional[Union[int, str]] = None) -> str:
    """Get the name of a device."""
    return _call_backend("get_device_name", _translate_device(device))


def get_device_capability(device: Optional[Union[int, str]] = None) -> tuple:
    """Get the CUDA/MUSA compute capability of a device."""
    return _call_backend("get_device_capability", _translate_device(device))


def get_device_properties(device: Optional[Union[int, str]] = None):
    """Get the properties of a device."""
    return _call_backend("get_device_properties", _translate_device(device))


# Memory management functions.
def memory_allocated(device: Optional[Union[int, str]] = None) -> int:
    """Return the current GPU memory occupied by tensors in bytes."""
    return _call_backend("memory_allocated", _translate_device(device))


def max_memory_allocated(device: Optional[Union[int, str]] = None) -> int:
    """Return the maximum GPU memory occupied by tensors in bytes."""
    return _call_backend("max_memory_allocated", _translate_device(device))


def memory_reserved(device: Optional[Union[int, str]] = None) -> int:
    """Return the current GPU memory managed by the caching allocator in bytes."""
    return _call_backend("memory_reserved", _translate_device(device))


def max_memory_reserved(device: Optional[Union[int, str]] = None) -> int:
    """Return the maximum GPU memory managed by the caching allocator in bytes."""
    return _call_backend("max_memory_reserved", _translate_device(device))


def memory_cached(device: Optional[Union[int, str]] = None) -> int:
    """Deprecated: Use memory_reserved instead."""
    return _call_backend_with_fallback(
        "memory_cached", "memory_reserved", _translate_device(device)
    )


def max_memory_cached(device: Optional[Union[int, str]] = None) -> int:
    """Deprecated: Use max_memory_reserved instead."""
    return _call_backend_with_fallback(
        "max_memory_cached", "max_memory_reserved", _translate_device(device)
    )


def empty_cache() -> None:
    """Release all unoccupied cached memory."""
    _call_backend("empty_cache")


def reset_peak_memory_stats(device: Optional[Union[int, str]] = None) -> None:
    """Reset the peak memory stats."""
    _call_backend("reset_peak_memory_stats", _translate_device(device))


def reset_max_memory_allocated(device: Optional[Union[int, str]] = None) -> None:
    """Reset the starting point in tracking maximum GPU memory occupied."""
    _call_backend("reset_max_memory_allocated", _translate_device(device))


def reset_max_memory_cached(device: Optional[Union[int, str]] = None) -> None:
    """Reset the starting point in tracking maximum GPU memory managed."""
    _call_backend_with_fallback(
        "reset_max_memory_cached", "reset_peak_memory_stats", _translate_device(device)
    )


# Synchronization functions.
def synchronize(device: Optional[Union[int, str]] = None) -> None:
    """Wait for all kernels in all streams on a device to complete."""
    _call_backend("synchronize", _translate_device(device))


# Stream and event aliases are resolved from the active backend at import time.
def _setup_stream_event_classes():
    """Set up Stream and Event classes from the backend."""
    backend = _get_backend()

    # Use backend classes directly so isinstance checks keep backend semantics.
    global Stream, Event, current_stream, default_stream, stream

    Stream = getattr(backend, "Stream", None)
    Event = getattr(backend, "Event", None)
    current_stream = getattr(backend, "current_stream", None)
    default_stream = getattr(backend, "default_stream", None)
    stream = getattr(backend, "stream", None)


# Initialize stream and event aliases.
try:
    _setup_stream_event_classes()
except Exception:
    Stream = None
    Event = None
    current_stream = None
    default_stream = None
    stream = None
