# Copyright (c) torchada contributors
# SPDX-License-Identifier: MIT
"""
C++ operator overrides infrastructure for torchada.

This module handles building and loading C++ extensions that can override
ATen operator implementations for the PrivateUse1 (MUSA) dispatch key.

C++ extensions are automatically loaded on MUSA platform when torchada is imported.

Usage:
    import torchada  # C++ extensions are loaded automatically on MUSA.

    # Or explicitly load.
    from torchada._cpp_ops import load_cpp_ops
    load_cpp_ops()
"""

import os
import os.path as osp
import subprocess
from dataclasses import dataclass
from typing import List, Optional

_cpp_ops_module: Optional[object] = None
_musa_arch_cached: Optional[str] = None

EXTENSION_NAME = "torchada_cpp_ops"
DEFAULT_MUSA_ARCH = "mp_31"


@dataclass(frozen=True)
class _ExtensionSources:
    """Source files grouped by compiler/toolchain requirements."""

    csrc_dir: str
    cpp_sources: List[str]
    musa_sources: List[str]

    @property
    def all_sources(self) -> List[str]:
        """All source paths in load order."""
        return self.cpp_sources + self.musa_sources

    @property
    def has_sources(self) -> bool:
        """Whether any extension source files were discovered."""
        return bool(self.cpp_sources or self.musa_sources)

    @property
    def needs_musa_loader(self) -> bool:
        """Whether MUSA extension loading is required."""
        return bool(self.musa_sources)


def _detect_musa_arch() -> str:
    """
    Detect MUSA GPU architecture from compute capability.

    Uses musaInfo to get compute capability and maps it to architecture:
    - 2.1 -> mp_21 (MTT S80)
    - 2.2 -> mp_22 (MTT S4000)
    - 3.1 -> mp_31 (MTT S5000)

    Returns:
        Architecture string like "mp_21", "mp_22", or "mp_31".
        Defaults to "mp_31" if detection fails.
    """
    global _musa_arch_cached
    if _musa_arch_cached is not None:
        return _musa_arch_cached

    arch = DEFAULT_MUSA_ARCH
    try:
        result = subprocess.run(
            ["musaInfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "compute capability:" in line.lower():
                # Parse lines like "compute capability:              2.1".
                parts = line.split(":")
                if len(parts) >= 2:
                    version = parts[1].strip()
                    # Convert "2.1" to "mp_21", "3.1" to "mp_31", etc.
                    version_parts = version.split(".")
                    if len(version_parts) >= 2:
                        major = version_parts[0].strip()
                        minor = version_parts[1].strip()
                        arch = f"mp_{major}{minor}"
                        break
    except Exception:
        pass

    _musa_arch_cached = arch
    return arch


def _get_csrc_dir() -> str:
    """Return the packaged C++ source directory."""
    return osp.join(osp.dirname(__file__), "csrc")


def _discover_extension_sources(csrc_dir: Optional[str] = None) -> _ExtensionSources:
    """Discover C++ and MUSA source files for the extension build."""
    csrc_dir = csrc_dir or _get_csrc_dir()
    cpp_sources = []
    musa_sources = []

    for fname in sorted(os.listdir(csrc_dir)):
        fpath = osp.join(csrc_dir, fname)
        if fname.endswith(".cpp"):
            cpp_sources.append(fpath)
        elif fname.endswith((".cu", ".mu")):
            musa_sources.append(fpath)

    return _ExtensionSources(
        csrc_dir=csrc_dir,
        cpp_sources=cpp_sources,
        musa_sources=musa_sources,
    )


def _get_musa_arch_flag() -> str:
    """Return the MUSA offload architecture flag for extension compilation."""
    mtgpu_target = os.environ.get("MTGPU_TARGET", "")
    if not mtgpu_target:
        mtgpu_target = _detect_musa_arch()
    return f"--offload-arch={mtgpu_target}"


def _load_extension(sources: _ExtensionSources, verbose: bool):
    """Load the extension with the toolchain required by the discovered sources."""
    if sources.needs_musa_loader:
        from .utils.cpp_extension import load

        return load(
            name=EXTENSION_NAME,
            sources=sources.all_sources,
            extra_include_paths=[sources.csrc_dir],
            extra_cuda_cflags=[_get_musa_arch_flag()],
            verbose=verbose,
        )

    from torch.utils.cpp_extension import load

    return load(
        name=EXTENSION_NAME,
        sources=sources.all_sources,
        extra_include_paths=[sources.csrc_dir],
        verbose=verbose,
    )


def load_cpp_ops(force_reload: bool = False) -> Optional[object]:
    """
    Load the C++ operator overrides extension.

    The extension is automatically loaded on MUSA platform.

    Args:
        force_reload: If True, reload the extension even if already loaded.

    Returns:
        The loaded extension module, or None if not loaded.
    """
    global _cpp_ops_module

    if _cpp_ops_module is not None and not force_reload:
        return _cpp_ops_module

    from ._platform import is_musa_platform

    if not is_musa_platform():
        return None

    try:
        sources = _discover_extension_sources()
        if not sources.has_sources:
            import warnings

            warnings.warn("torchada C++ ops: no source files found")
            return None

        verbose = os.environ.get("TORCHADA_CPP_OPS_VERBOSE") == "1"
        _cpp_ops_module = _load_extension(sources, verbose)
        _cpp_ops_module._mark_loaded()
        return _cpp_ops_module

    except Exception as e:
        import warnings

        warnings.warn(f"Failed to load torchada C++ ops: {e}")
        return None


def is_loaded() -> bool:
    """Check if C++ ops extension is loaded."""
    return _cpp_ops_module is not None


def get_version() -> Optional[str]:
    """Get C++ ops extension version."""
    if _cpp_ops_module is None:
        return None
    return _cpp_ops_module.get_version()


def get_module() -> Optional[object]:
    """Get the loaded C++ ops module."""
    return _cpp_ops_module
