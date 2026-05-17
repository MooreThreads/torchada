"""
Device and tensor-constructor compatibility helpers.

This module owns the low-level CUDA-device-to-MUSA-device translation used by
the patch registry. Keeping the state here avoids mixing global patch state
with the orchestration code in ``_patch.py``.
"""

import functools
from typing import Any, Callable, Optional

import torch

from ._platform import is_musa_platform

_device_str_cache = {}
_is_musa_platform_cached: Optional[bool] = None

_original_torch_device = None
_original_torch_generator = None
_original_c_generator = None


def _translate_device(device: Any) -> Any:
    """
    Translate ``cuda`` device references to ``musa`` on MUSA platforms.

    Strings are cached because this sits on hot paths such as ``Tensor.to`` and
    tensor factory calls.
    """
    global _is_musa_platform_cached

    if _is_musa_platform_cached is None:
        _is_musa_platform_cached = is_musa_platform()

    if not _is_musa_platform_cached or device is None:
        return device

    if isinstance(device, str):
        if device in _device_str_cache:
            return _device_str_cache[device]
        if device == "cuda" or device.startswith("cuda:"):
            result = device.replace("cuda", "musa")
        else:
            result = device
        _device_str_cache[device] = result
        return result

    if isinstance(device, torch.device):
        if device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    return device


def _wrap_to_method(original_to: Callable) -> Callable:
    """Wrap ``Tensor.to`` to translate CUDA device arguments."""

    @functools.wraps(original_to)
    def wrapped_to(self, *args, **kwargs):
        if args:
            first_arg = args[0]
            if isinstance(first_arg, (str, torch.device)):
                args = (_translate_device(first_arg),) + args[1:]
            elif isinstance(first_arg, torch.dtype) and len(args) >= 2:
                args = (first_arg, _translate_device(args[1])) + args[2:]

        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])

        return original_to(self, *args, **kwargs)

    return wrapped_to


def _musa_device_spec(device: Any) -> Any:
    """Return a device spec suitable for ``.to`` when ``.musa`` is unavailable."""
    if device is None:
        return "musa"
    if isinstance(device, int):
        return f"musa:{device}"
    return device


def _wrap_tensor_cuda(original_cuda: Callable) -> Callable:
    """Wrap ``Tensor.cuda`` to use MUSA on MUSA platforms."""
    _is_musa = is_musa_platform()

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        if _is_musa:
            device = _translate_device(device)
            if hasattr(self, "musa"):
                kwargs = {"device": device, "non_blocking": non_blocking}
                if memory_format is not torch.preserve_format:
                    kwargs["memory_format"] = memory_format
                return self.musa(**kwargs)
            target_device = _musa_device_spec(device)
            return self.to(
                target_device,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )

        kwargs = {"device": device, "non_blocking": non_blocking}
        if memory_format is not torch.preserve_format:
            kwargs["memory_format"] = memory_format
        return original_cuda(self, **kwargs)

    return wrapped_cuda


def _wrap_module_cuda(original_cuda: Callable) -> Callable:
    """Wrap ``nn.Module.cuda`` to use MUSA on MUSA platforms."""
    _is_musa = is_musa_platform()

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None):
        if _is_musa:
            device = _translate_device(device)
            if hasattr(self, "musa"):
                return self.musa(device=device)
            target_device = _musa_device_spec(device)
            return self.to(target_device)
        return original_cuda(self, device=device)

    return wrapped_cuda


class _DeviceFactoryMeta(type):
    """Metaclass that keeps ``isinstance(x, torch.device)`` working."""

    def __instancecheck__(cls, instance):
        if _original_torch_device is not None:
            return isinstance(instance, _original_torch_device)
        return False

    def __subclasscheck__(cls, subclass):
        if _original_torch_device is not None:
            return issubclass(subclass, _original_torch_device)
        return False


class DeviceFactoryWrapper(metaclass=_DeviceFactoryMeta):
    """
    Drop-in ``torch.device`` factory that translates CUDA devices to MUSA.
    """

    _original = None

    def __new__(cls, device=None, index=None, *, type=None):
        original = cls._original
        if original is None:
            raise RuntimeError("DeviceFactoryWrapper not initialized")

        if type is not None:
            device = type

        if isinstance(device, original):
            if device.type == "cuda":
                index = device.index if index is None else index
                device = "musa"
            else:
                return device

        if isinstance(device, str):
            device = _translate_device(device)

        if index is not None:
            return original(device, index)
        if device is not None:
            return original(device)
        return original()


def patch_torch_device(torch_module=torch) -> None:
    """Patch ``torch.device`` with ``DeviceFactoryWrapper``."""
    global _original_torch_device

    if _original_torch_device is not None:
        return

    _original_torch_device = torch_module.device
    DeviceFactoryWrapper._original = _original_torch_device
    torch_module.device = DeviceFactoryWrapper


class _GeneratorMeta(type):
    """Metaclass that preserves ``isinstance(x, torch.Generator)`` behavior."""

    def __instancecheck__(cls, instance):
        if _original_c_generator is not None:
            return isinstance(instance, _original_c_generator)
        return False

    def __subclasscheck__(cls, subclass):
        if _original_c_generator is not None and subclass is _original_c_generator:
            return True
        return super().__subclasscheck__(subclass)


class GeneratorWrapper(metaclass=_GeneratorMeta):
    """Wrapper for ``torch.Generator`` that translates CUDA devices to MUSA."""

    _original = None

    def __new__(cls, device=None):
        original = cls._original
        if original is None:
            raise RuntimeError("GeneratorWrapper not initialized")
        if device is not None:
            device = _translate_device(device)
        return original(device=device)


def patch_torch_generator(torch_module=torch) -> None:
    """Patch ``torch.Generator`` with ``GeneratorWrapper``."""
    global _original_torch_generator, _original_c_generator

    if _original_torch_generator is not None:
        return

    _original_torch_generator = torch_module.Generator
    _original_c_generator = torch_module._C.Generator

    GeneratorWrapper._original = _original_torch_generator
    GeneratorWrapper.__doc__ = _original_torch_generator.__doc__

    torch_module.Generator = GeneratorWrapper


def _wrap_factory_function(original_fn: Callable) -> Callable:
    """Wrap tensor factory functions to translate ``device=`` arguments."""

    @functools.wraps(original_fn)
    def wrapped_fn(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])
        return original_fn(*args, **kwargs)

    return wrapped_fn


_FACTORY_FUNCTIONS = [
    "tensor",
    "as_tensor",
    "asarray",
    "empty",
    "zeros",
    "ones",
    "full",
    "rand",
    "randn",
    "randint",
    "randperm",
    "normal",
    "arange",
    "range",
    "linspace",
    "logspace",
    "eye",
    "empty_strided",
    "empty_permuted",
    "from_file",
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "randint_like",
    "sparse_coo_tensor",
    "sparse_csr_tensor",
    "sparse_csc_tensor",
    "sparse_bsr_tensor",
    "sparse_bsc_tensor",
    "sparse_compressed_tensor",
    "tril_indices",
    "triu_indices",
    "bartlett_window",
    "blackman_window",
    "hamming_window",
    "hann_window",
    "kaiser_window",
]


def get_original_torch_device():
    """Return the saved original ``torch.device`` factory, if patched."""
    return _original_torch_device


def get_original_torch_generator():
    """Return the saved original ``torch.Generator`` factory, if patched."""
    return _original_torch_generator


def get_original_c_generator():
    """Return the saved original C generator type, if patched."""
    return _original_c_generator
