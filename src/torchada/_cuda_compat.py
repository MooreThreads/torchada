"""
CUDA module compatibility helpers for torchada patching.

The patch registry in ``_patch.py`` is responsible for when patching happens.
This module owns the CUDA-shaped shims that get installed onto ``torch.musa``
so the registry stays readable as compatibility coverage grows.
"""

import sys
from collections import OrderedDict
from types import ModuleType
from typing import Any, Callable, Optional

from ._runtime import cuda_to_musa_name

DeviceTranslator = Callable[[Any], Any]

_MUSA_SYNC_DEBUG_MODE = 0
_SYNC_DEBUG_MODE_VALUES = {
    "default": 0,
    "warn": 1,
    "error": 2,
}


class _CudartWrapper:
    """
    Wrapper for CUDA runtime that translates calls to MUSA runtime.

    This allows code like ``torch.cuda.cudart().cudaHostRegister(...)`` to work
    on MUSA by translating to ``torch_musa.musart().musaHostRegister(...)``.
    Resolved attributes are cached on the wrapper.
    """

    def __init__(self, musart_module):
        self._musart = musart_module

    def __getattr__(self, name):
        translated_name = cuda_to_musa_name(name)
        candidates = [translated_name]
        if translated_name != name:
            candidates.append(name)

        for candidate in candidates:
            try:
                value = getattr(self._musart, candidate)
            except AttributeError:
                continue
            object.__setattr__(self, name, value)
            return value

        raise AttributeError(f"CUDA runtime has no attribute '{name}'")


class _CudaModuleWrapper(ModuleType):
    """
    Module wrapper that redirects torch.cuda to torch.musa while preserving
    CUDA-only detection APIs that downstream packages rely on.

    ``torch.cuda.is_available()`` intentionally keeps the original CUDA
    behavior. Everything else resolves through torch.musa, with a few explicit
    remaps for CUDA/MUSA naming differences.
    """

    _NO_REDIRECT = {"is_available"}
    _SPECIAL_ATTRS = {
        "StreamContext": "core.stream.StreamContext",
    }
    _REMAP_ATTRS = {
        "_device_count_nvml": "device_count",
        "nccl": "mccl",
    }
    _NO_CACHE = set()

    def __init__(self, original_cuda, musa_module):
        super().__init__("torch.cuda")
        self._original_cuda = original_cuda
        self._musa_module = musa_module
        self._cudart_wrapper = None

    def cudart(self):
        """Return a CUDA runtime wrapper that delegates to MUSA runtime APIs."""
        if self._cudart_wrapper is None:
            if hasattr(self._musa_module, "musart"):
                self._cudart_wrapper = _CudartWrapper(self._musa_module.musart())
            else:
                return self._original_cuda.cudart()
        return self._cudart_wrapper

    def __getattr__(self, name):
        if name in self._NO_REDIRECT:
            value = getattr(self._original_cuda, name)
        elif name in self._SPECIAL_ATTRS:
            value = self._musa_module
            for part in self._SPECIAL_ATTRS[name].split("."):
                value = getattr(value, part)
        elif name in self._REMAP_ATTRS:
            value = getattr(self._musa_module, self._REMAP_ATTRS[name])
        else:
            value = getattr(self._musa_module, name)

        if name not in self._NO_CACHE:
            object.__setattr__(self, name, value)
        return value

    def __dir__(self):
        attrs = set(dir(self._musa_module))
        attrs.update(self._NO_REDIRECT)
        attrs.update(self._SPECIAL_ATTRS.keys())
        attrs.update(self._REMAP_ATTRS.keys())
        attrs.add("cudart")
        return list(attrs)


def _musa_get_gencode_flags() -> str:
    """
    Return CUDA-style gencode flags for MUSA.

    CUDA's implementation returns NVCC flags. Those flags should not be passed
    to the MUSA toolchain, so compatibility behavior matches a non-CUDA build
    and returns an empty string while preserving the API surface.
    """
    return ""


def _musa_get_sync_debug_mode() -> int:
    """Return the process-local CUDA sync debug mode shim value."""
    return _MUSA_SYNC_DEBUG_MODE


def _musa_set_sync_debug_mode(debug_mode) -> None:
    """
    Set a process-local CUDA sync debug mode shim value.

    torch_musa does not expose CUDA's C++ sync-debug hooks. Keeping the value in
    Python preserves the public setter/getter contract without pretending to
    alter MUSA runtime synchronization behavior.
    """
    global _MUSA_SYNC_DEBUG_MODE

    if isinstance(debug_mode, str):
        if debug_mode not in _SYNC_DEBUG_MODE_VALUES:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )
        _MUSA_SYNC_DEBUG_MODE = _SYNC_DEBUG_MODE_VALUES[debug_mode]
        return None

    if isinstance(debug_mode, int) and debug_mode in _SYNC_DEBUG_MODE_VALUES.values():
        _MUSA_SYNC_DEBUG_MODE = int(debug_mode)
        return None

    raise RuntimeError("invalid value of debug_mode, expected one of `default`, `warn`, `error`")


def _host_memory_stats() -> OrderedDict:
    """Return empty host allocator stats when MUSA exposes no host counters."""
    return OrderedDict()


def _host_memory_stats_as_nested_dict() -> dict:
    """Return empty nested host allocator stats when MUSA exposes no counters."""
    return {}


def _reset_host_memory_stats() -> None:
    """No-op reset for unavailable MUSA host allocator counters."""
    return None


def _make_memory_cached(torch_module, translate_device: DeviceTranslator):
    def memory_cached(device=None) -> int:
        """Deprecated CUDA alias for memory_reserved(), mapped to MUSA."""
        return torch_module.musa.memory_reserved(translate_device(device))

    return memory_cached


def _make_max_memory_cached(torch_module, translate_device: DeviceTranslator):
    def max_memory_cached(device=None) -> int:
        """Deprecated CUDA alias for max_memory_reserved(), mapped to MUSA."""
        return torch_module.musa.max_memory_reserved(translate_device(device))

    return max_memory_cached


def _make_get_stream_from_external(torch_module, translate_device: DeviceTranslator):
    def get_stream_from_external(data_ptr: int, device=None):
        """Wrap an externally allocated MUSA stream using CUDA-compatible API naming."""
        return torch_module.musa.ExternalStream(data_ptr, device=translate_device(device))

    return get_stream_from_external


def _build_musa_sparse_module(musa_module) -> ModuleType:
    """Create a torch.cuda.sparse-compatible module backed by MUSA tensor classes."""
    sparse_module = ModuleType("torch.cuda.sparse")
    for name in [
        "ByteTensor",
        "CharTensor",
        "DoubleTensor",
        "FloatTensor",
        "HalfTensor",
        "IntTensor",
        "LongTensor",
        "ShortTensor",
        "BFloat16Tensor",
    ]:
        if hasattr(musa_module, name):
            setattr(sparse_module, name, getattr(musa_module, name))
    return sparse_module


def install_cuda_memory_compat(
    torch_module,
    cpp_ops_module: Optional[Any],
    translate_device: DeviceTranslator,
) -> None:
    """Install torch.cuda.memory-compatible aliases onto torch.musa.memory."""
    if not hasattr(torch_module.musa, "memory"):
        return

    musa_memory_module = torch_module.musa.memory
    if musa_memory_module is None:
        return

    sys.modules["torch.cuda.memory"] = musa_memory_module

    if hasattr(musa_memory_module, "MUSAPluggableAllocator"):
        musa_memory_module.CUDAPluggableAllocator = musa_memory_module.MUSAPluggableAllocator
        torch_module.musa.CUDAPluggableAllocator = musa_memory_module.MUSAPluggableAllocator

    memory_aliases = {
        "memory_cached": _make_memory_cached(torch_module, translate_device),
        "max_memory_cached": _make_max_memory_cached(torch_module, translate_device),
        "host_memory_stats": _host_memory_stats,
        "host_memory_stats_as_nested_dict": _host_memory_stats_as_nested_dict,
        "reset_accumulated_host_memory_stats": _reset_host_memory_stats,
        "reset_peak_host_memory_stats": _reset_host_memory_stats,
    }
    for name, func in memory_aliases.items():
        if not hasattr(torch_module.musa, name):
            setattr(torch_module.musa, name, func)
        if not hasattr(musa_memory_module, name):
            setattr(musa_memory_module, name, func)

    if cpp_ops_module is None:
        return

    for func_name in [
        "_cuda_beginAllocateCurrentThreadToPool",
        "_cuda_endAllocateToPool",
        "_cuda_releasePool",
    ]:
        func = getattr(cpp_ops_module, func_name, None)
        if func is not None:
            setattr(musa_memory_module, func_name, func)


def install_cuda_module_aliases(torch_module) -> None:
    """Register import aliases for CUDA submodules that map to MUSA modules."""
    musa_module = torch_module.musa

    if hasattr(musa_module, "amp"):
        sys.modules["torch.cuda.amp"] = musa_module.amp

    if hasattr(musa_module, "graphs"):
        sys.modules["torch.cuda.graphs"] = musa_module.graphs

    if hasattr(musa_module, "MUSAGraph") and not hasattr(musa_module, "CUDAGraph"):
        musa_module.CUDAGraph = musa_module.MUSAGraph

    if hasattr(musa_module, "mccl"):
        sys.modules["torch.cuda.nccl"] = musa_module.mccl
        if not hasattr(musa_module, "nccl"):
            musa_module.nccl = musa_module.mccl

    try:
        import torch_musa.core.stream as musa_stream_module

        sys.modules["torch.cuda.streams"] = musa_stream_module
        if not hasattr(musa_module, "streams"):
            musa_module.streams = musa_stream_module
    except ImportError:
        pass

    if not hasattr(musa_module, "sparse"):
        musa_module.sparse = _build_musa_sparse_module(musa_module)
    sys.modules["torch.cuda.sparse"] = musa_module.sparse

    if hasattr(musa_module, "profiler"):
        sys.modules["torch.cuda.profiler"] = musa_module.profiler

    try:
        from .cuda import nvtx as nvtx_stub

        sys.modules["torch.cuda.nvtx"] = nvtx_stub
        musa_module.nvtx = nvtx_stub
    except ImportError:
        pass

    if hasattr(musa_module, "random"):
        sys.modules["torch.cuda.random"] = musa_module.random
    else:
        try:
            from .cuda import random as random_stub

            sys.modules["torch.cuda.random"] = random_stub
            musa_module.random = random_stub
        except ImportError:
            pass


def install_cuda_public_api_shims(torch_module, translate_device: DeviceTranslator) -> None:
    """Install top-level CUDA API shims that torch_musa does not expose."""
    musa_module = torch_module.musa

    try:
        from torch_musa.core._lazy_init import _lazy_call

        if not hasattr(musa_module, "_lazy_call"):
            musa_module._lazy_call = _lazy_call
    except ImportError:
        pass

    if not hasattr(musa_module, "_is_compiled"):
        musa_module._is_compiled = lambda: True

    if not hasattr(musa_module, "has_half"):
        musa_module.has_half = True
    if not hasattr(musa_module, "has_magma"):
        musa_module.has_magma = False
    if hasattr(musa_module, "_lazy_init") and not hasattr(musa_module, "init"):
        musa_module.init = musa_module._lazy_init
    if not hasattr(musa_module, "get_stream_from_external"):
        musa_module.get_stream_from_external = _make_get_stream_from_external(
            torch_module, translate_device
        )

    try:
        from torch_musa.core._lazy_init import default_generators

        if not hasattr(musa_module, "default_generators"):
            musa_module.default_generators = default_generators
    except ImportError:
        pass

    if not hasattr(musa_module, "get_gencode_flags"):
        musa_module.get_gencode_flags = _musa_get_gencode_flags
    if not hasattr(musa_module, "get_sync_debug_mode"):
        musa_module.get_sync_debug_mode = _musa_get_sync_debug_mode
    if not hasattr(musa_module, "set_sync_debug_mode"):
        musa_module.set_sync_debug_mode = _musa_set_sync_debug_mode
