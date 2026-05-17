"""
Compatibility wrapper for the evolving ``torch.accelerator`` API.

This module provides MUSA-backed fallbacks for torch.accelerator APIs that are
missing or incomplete in supported PyTorch versions.
"""

import sys
from types import ModuleType

import torch


class _AcceleratorModuleWrapper(ModuleType):
    """
    Module wrapper that prefers torchada overrides, then torch.accelerator, then
    torch.musa fallbacks for APIs not present in older PyTorch builds.
    """

    _REMAP_ATTRS = {
        "set_device_index": "set_device",
        "set_device_idx": "set_device",
        "current_device_index": "current_device",
        "current_device_idx": "current_device",
    }
    _SPECIAL_ATTRS = {
        "StreamContext": "core.stream.StreamContext",
    }
    _MUSA_OVERRIDES = (
        "empty_cache",
        "empty_host_cache",
        "memory_stats",
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
        "reset_accumulated_memory_stats",
        "reset_peak_memory_stats",
        "get_memory_info",
    )

    def __init__(self, original_accel, musa_module):
        super().__init__("torch.accelerator")
        self._original_accel = original_accel
        self._musa_module = musa_module
        self._overrides = {}

        for name in self._MUSA_OVERRIDES:
            if hasattr(original_accel, name) and hasattr(musa_module, name):
                self._set_override(name, getattr(musa_module, name))

    def _set_override(self, name, value):
        """Install an override that takes precedence over wrapped modules."""
        self._overrides[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name in self._overrides:
            return self._overrides[name]

        try:
            value = getattr(self._original_accel, name)
        except AttributeError:
            if hasattr(self._musa_module, name):
                value = getattr(self._musa_module, name)
            elif name in self._SPECIAL_ATTRS:
                value = self._musa_module
                for part in self._SPECIAL_ATTRS[name].split("."):
                    value = getattr(value, part)
            elif name in self._REMAP_ATTRS:
                value = getattr(self._musa_module, self._REMAP_ATTRS[name])
            else:
                raise AttributeError(f"module 'torch.accelerator' has no attribute '{name}'")

        object.__setattr__(self, name, value)
        return value

    def __dir__(self):
        attrs = set(dir(self._original_accel))
        attrs.update(dir(self._musa_module))
        attrs.update(self._REMAP_ATTRS.keys())
        attrs.update(self._SPECIAL_ATTRS.keys())
        attrs.update(self._overrides.keys())
        return list(attrs)


_original_torch_accelerator = None


def _make_patched_accelerator_synchronize(musa_module):
    """Build a synchronize replacement that delegates to ``torch.musa``."""
    from ._device_compat import _translate_device

    def patched_synchronize(device=None):
        if device is not None and not isinstance(device, (torch.device, str, int)):
            raise TypeError(
                f"synchronize() expected device to be torch.device, str, int, or None, "
                f"but got {type(device).__name__}"
            )
        device = _translate_device(device)
        musa_module.synchronize(device)

    return patched_synchronize


def _make_accelerator_context_managers(accel_module):
    """Build ``device_index`` and ``stream`` context managers bound to wrapper."""

    class device_index:
        """Temporarily set the current accelerator device index."""

        def __init__(self, idx):
            self.idx = idx
            self.prev_idx = None

        def __enter__(self):
            self.prev_idx = accel_module.current_device_index()
            accel_module.set_device_index(self.idx)
            return self

        def __exit__(self, *args):
            if self.prev_idx is not None:
                accel_module.set_device_index(self.prev_idx)

    class stream:
        """Temporarily set the current accelerator stream."""

        def __init__(self, stream_obj):
            self.stream = stream_obj
            self.prev_stream = None

        def __enter__(self):
            self.prev_stream = accel_module.current_stream()
            accel_module.set_stream(self.stream)
            return self

        def __exit__(self, *args):
            if self.prev_stream is not None:
                accel_module.set_stream(self.prev_stream)

    return device_index, stream


def patch_torch_accelerator(torch_module=torch) -> None:
    """Wrap ``torch.accelerator`` with MUSA fallbacks and torchada overrides."""
    global _original_torch_accelerator

    import torch.accelerator as accel

    if _original_torch_accelerator is None:
        _original_torch_accelerator = accel

    wrapper = _AcceleratorModuleWrapper(_original_torch_accelerator, torch_module.musa)
    wrapper._set_override("synchronize", _make_patched_accelerator_synchronize(torch_module.musa))

    device_index_cm, stream_cm = _make_accelerator_context_managers(wrapper)
    if not hasattr(_original_torch_accelerator, "device_index"):
        wrapper._set_override("device_index", device_index_cm)
    if not hasattr(_original_torch_accelerator, "stream"):
        wrapper._set_override("stream", stream_cm)

    sys.modules["torch.accelerator"] = wrapper
    torch_module.accelerator = wrapper


def get_original_torch_accelerator():
    """Return the saved original ``torch.accelerator`` module, if patched."""
    return _original_torch_accelerator
