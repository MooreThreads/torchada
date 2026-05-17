"""
Runtime name conversion utilities for CUDA to MUSA.

This module centralizes CUDA-family runtime symbol translation. The public
helpers are exported from ``torchada`` for manual use, and patching code uses
the same table when adapting ``ctypes.CDLL`` and ``torch.cuda.cudart()``.
"""

from typing import Callable, Dict, Tuple

PREFIX_MAPPINGS: Dict[str, str] = {
    "cuda": "musa",
    "nccl": "mccl",
    "cublas": "mublas",
    "curand": "murand",
}

MUSA_LIBRARY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "musart": ("libmusart", "musart.so", "libmusa_runtime"),
    "mccl": ("libmccl", "mccl.so"),
    "mublas": ("libmublas", "mublas.so"),
    "murand": ("libmurand", "murand.so"),
}


def translate_prefix_name(name: str, source_prefix: str, target_prefix: str) -> str:
    """
    Translate ``name`` from one prefix convention to another.

    Names that do not start with ``source_prefix`` are returned unchanged.
    """
    if name.startswith(source_prefix):
        return target_prefix + name[len(source_prefix) :]
    return name


def cuda_to_musa_name(name: str) -> str:
    """
    Convert a CUDA function/symbol name to its MUSA equivalent.

    Examples:
        >>> cuda_to_musa_name("cudaMalloc")
        'musaMalloc'
        >>> cuda_to_musa_name("someOtherFunc")
        'someOtherFunc'
    """
    return translate_prefix_name(name, "cuda", "musa")


def nccl_to_mccl_name(name: str) -> str:
    """
    Convert an NCCL function/symbol name to its MCCL equivalent.

    Examples:
        >>> nccl_to_mccl_name("ncclAllReduce")
        'mcclAllReduce'
        >>> nccl_to_mccl_name("someOtherFunc")
        'someOtherFunc'
    """
    return translate_prefix_name(name, "nccl", "mccl")


def cublas_to_mublas_name(name: str) -> str:
    """
    Convert a cuBLAS function/symbol name to its muBLAS equivalent.

    Examples:
        >>> cublas_to_mublas_name("cublasCreate")
        'mublasCreate'
        >>> cublas_to_mublas_name("someOtherFunc")
        'someOtherFunc'
    """
    return translate_prefix_name(name, "cublas", "mublas")


def curand_to_murand_name(name: str) -> str:
    """
    Convert a cuRAND function/symbol name to its muRAND equivalent.

    Examples:
        >>> curand_to_murand_name("curandCreate")
        'murandCreate'
        >>> curand_to_murand_name("someOtherFunc")
        'someOtherFunc'
    """
    return translate_prefix_name(name, "curand", "murand")


_RUNTIME_TRANSLATORS: Dict[str, Callable[[str], str]] = {
    "musart": cuda_to_musa_name,
    "mccl": nccl_to_mccl_name,
    "mublas": cublas_to_mublas_name,
    "murand": curand_to_murand_name,
}


def detect_musa_library_type(lib_path: str) -> str:
    """Detect which MUSA runtime-family library is referenced by ``lib_path``."""
    lib_path_lower = str(lib_path).lower()
    for library_type, patterns in MUSA_LIBRARY_PATTERNS.items():
        if any(pattern in lib_path_lower for pattern in patterns):
            return library_type
    return "unknown"


def is_musa_runtime_library_path(lib_path: str) -> bool:
    """Return whether ``lib_path`` points at a library needing symbol translation."""
    return detect_musa_library_type(lib_path) != "unknown"


def translate_runtime_symbol_name(name: str, library_type: str) -> str:
    """Translate ``name`` for a detected MUSA runtime-family library type."""
    translator = _RUNTIME_TRANSLATORS.get(library_type)
    if translator is None:
        return name
    return translator(name)
