"""
ctypes compatibility for CUDA-named symbols in MUSA runtime libraries.

This module keeps ctypes callers using CUDA-family names while dispatching to
MUSA runtime-family libraries.
"""

from ._runtime import (
    detect_musa_library_type,
    is_musa_runtime_library_path,
    translate_runtime_symbol_name,
)


class _CDLLWrapper:
    """
    Wrapper for ``ctypes.CDLL`` that translates CUDA-family symbol names.

    Loading MUSA libraries such as ``libmusart.so`` or ``libmccl.so`` still lets
    callers access symbols with CUDA/NCCL names.
    """

    def __init__(self, cdll_instance, lib_path: str):
        object.__setattr__(self, "_cdll", cdll_instance)
        object.__setattr__(self, "_lib_path", lib_path)
        object.__setattr__(self, "_lib_type", self._detect_lib_type(lib_path))

    def _detect_lib_type(self, lib_path: str) -> str:
        """Detect the MUSA runtime-family library type from its path."""
        return detect_musa_library_type(lib_path)

    def _translate_name(self, name: str) -> str:
        """Translate CUDA-family symbol names for the wrapped library."""
        lib_type = object.__getattribute__(self, "_lib_type")
        return translate_runtime_symbol_name(name, lib_type)

    def __getattr__(self, name: str):
        cdll = object.__getattribute__(self, "_cdll")
        value = getattr(cdll, self._translate_name(name))
        object.__setattr__(self, name, value)
        return value

    def __setattr__(self, name: str, value):
        cdll = object.__getattribute__(self, "_cdll")
        setattr(cdll, self._translate_name(name), value)

    def __getitem__(self, name: str):
        cdll = object.__getattribute__(self, "_cdll")
        return cdll[self._translate_name(name)]


_original_ctypes_CDLL = None


def patch_ctypes_cdll() -> None:
    """Patch ``ctypes.CDLL`` to wrap MUSA runtime-family libraries."""
    import ctypes

    global _original_ctypes_CDLL

    if _original_ctypes_CDLL is not None:
        return

    _original_ctypes_CDLL = ctypes.CDLL

    class PatchedCDLL:
        """Patched CDLL constructor that wraps MUSA runtime libraries."""

        def __new__(cls, name, *args, **kwargs):
            cdll_instance = _original_ctypes_CDLL(name, *args, **kwargs)
            name_str = str(name) if name else ""
            if is_musa_runtime_library_path(name_str):
                return _CDLLWrapper(cdll_instance, name_str)
            return cdll_instance

    ctypes.CDLL = PatchedCDLL


def get_original_ctypes_cdll():
    """Return the saved original ``ctypes.CDLL`` constructor, if patched."""
    return _original_ctypes_CDLL
