# Copyright (c) torchada contributors
# SPDX-License-Identifier: MIT
"""
MUSA GPU architectures: the arch table, build-target resolution and build flags.

``ARCHS`` lists the architectures torchada's consumers build for and validate
against, each with its feature row. ``target_archs()`` resolves the archs a build
compiles for from the caller's environment variables, the visible devices, or a
default, in the shape of torch's ``TORCH_CUDA_ARCH_LIST`` resolution.
``arch_flags()``, ``torch_arch_list()`` and ``cache_tag()`` render that result for
mcc, for torch_musa's ``MUSAExtension`` and for a build directory.

This module imports only the standard library; ``torch`` is imported lazily inside the
functions that need it. ``import torchada.arch`` still runs ``import torchada`` first.
Resolving archs never initializes the MUSA runtime: devices are read in process only
when MUSA is already initialized, otherwise through the ``musaInfo`` tool.

Usage:
    from torchada import arch as musa_arch

    archs = musa_arch.target_archs(env=("MTGPU_TARGET", "TORCH_MUSA_ARCH_LIST"))
    flags = musa_arch.arch_flags(archs)  # ["--offload-arch=mp_31"]
"""

import functools
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

__all__ = ["MusaArch", "ARCHS", "target_archs", "arch_flags", "torch_arch_list", "cache_tag"]

_TOKEN_FORMS = "mp_XY, mpXY, X.Y or XY with X in 1-9 and Y in 0-9 (e.g. mp_31, 3.1, 31)"
_TOKEN_RE = re.compile(r"mp_?([0-9])([0-9])|([0-9])\.([0-9])|([0-9])([0-9])")
_SEPARATOR_RE = re.compile(r"[;\s]+")
_MUSAINFO_TIMEOUT = 5


class _Row(NamedTuple):
    # FP8 (E4M3/E5M2) conversion and FP8 MMA; -DENABLE_FP8 and supports_fp8() rely on it.
    fp8: bool = False
    # FP4 (E2M1) conversion and FP4 MMA.
    fp4: bool = False


_ROWS: Dict[Tuple[int, int], _Row] = {
    (3, 1): _Row(fp8=True, fp4=False),
}
_NO_ROW = _Row()


@dataclass(frozen=True, order=True)
class MusaArch:
    """A MUSA GPU architecture, identified by its compute capability only.

    Equality, hashing and ordering use ``(major, minor)``; features are read from
    the arch table, and an arch without a row reports every feature as ``False``.
    """

    major: int
    minor: int

    def __post_init__(self) -> None:
        for field, value, low in (("major", self.major, 1), ("minor", self.minor, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= 9:
                raise ValueError(
                    "MusaArch {} must be an int in [{}, 9], got {!r}".format(field, low, value)
                )

    @property
    def capability(self) -> Tuple[int, int]:
        """The compute capability, e.g. ``(3, 1)``."""
        return (self.major, self.minor)

    @property
    def name(self) -> str:
        """The mcc ``--offload-arch`` token and ``MTGPU_TARGET`` spelling, e.g. ``mp_31``."""
        return "mp_{}{}".format(self.major, self.minor)

    @property
    def fp8(self) -> bool:
        """Whether the arch has FP8 conversion and FP8 MMA (``False`` without a table row)."""
        return _ROWS.get(self.capability, _NO_ROW).fp8

    @property
    def fp4(self) -> bool:
        """Whether the arch has FP4 conversion and FP4 MMA (``False`` without a table row)."""
        return _ROWS.get(self.capability, _NO_ROW).fp4

    def __str__(self) -> str:
        return self.name


ARCHS: Tuple[MusaArch, ...] = tuple(MusaArch(major, minor) for major, minor in sorted(_ROWS))


def _parse(token: str) -> MusaArch:
    """Parse one arch token: ``mp_XY``, ``mpXY``, ``X.Y`` or ``XY``, case-insensitive."""
    match = _TOKEN_RE.fullmatch(token.strip().lower()) if isinstance(token, str) else None
    if match is not None:
        major, minor = (int(digit) for digit in match.groups() if digit is not None)
        try:
            return MusaArch(major, minor)
        except ValueError:
            pass
    raise ValueError("invalid MUSA arch {!r}: expected {}".format(token, _TOKEN_FORMS))


def _unique(archs: Iterable[MusaArch], caller: str) -> List[MusaArch]:
    """De-duplicated, sorted archs; at least one is required."""
    items = list(archs)
    for item in items:
        if not isinstance(item, MusaArch):
            raise TypeError("{}() takes MusaArch values, got {!r}".format(caller, item))
    if not items:
        raise ValueError("{}() needs at least one MUSA arch".format(caller))
    return sorted(set(items))


def _arch_or_none(major, minor) -> Optional[MusaArch]:
    try:
        return MusaArch(int(major), int(minor))
    except (TypeError, ValueError):
        return None


def _torch_musa():
    """``torch.musa`` when torch is already imported and has it, else ``None``."""
    torch = sys.modules.get("torch")
    return getattr(torch, "musa", None) if torch is not None else None


def _musa_initialized(musa) -> bool:
    is_initialized = getattr(musa, "is_initialized", None)
    if is_initialized is None:
        return False
    try:
        return bool(is_initialized())
    except Exception:
        return False


def _in_process_archs(musa) -> Optional[Tuple[MusaArch, ...]]:
    """Archs of every visible device read through an initialized ``torch.musa``."""
    try:
        capabilities = [musa.get_device_capability(i) for i in range(musa.device_count())]
        found = {_arch_or_none(*capability) for capability in capabilities}
    except Exception:
        return None
    return tuple(sorted(arch for arch in found if arch is not None))


def _musainfo_path() -> Optional[str]:
    path = shutil.which("musaInfo")
    if path is not None:
        return path
    from .utils import cpp_extension

    if not cpp_extension.CUDA_HOME:
        return None
    return os.path.join(cpp_extension.CUDA_HOME, "bin", "musaInfo")


@functools.lru_cache(maxsize=None)
def _musainfo_archs() -> Tuple[MusaArch, ...]:
    """Archs of every device ``musaInfo`` reports; ``()`` when it is missing or fails."""
    try:
        path = _musainfo_path()
        if path is None:
            return ()
        result = subprocess.run([path], capture_output=True, text=True, timeout=_MUSAINFO_TIMEOUT)
        found = set()
        for line in result.stdout.splitlines():
            if "compute capability:" not in line.lower():
                continue
            # "compute capability:              3.1"
            parts = line.split(":")[1].strip().split(".")
            if len(parts) >= 2:
                found.add(_arch_or_none(parts[0].strip(), parts[1].strip()))
    except Exception:
        return ()
    return tuple(sorted(arch for arch in found if arch is not None))


def _device_archs() -> Tuple[MusaArch, ...]:
    """Archs of every visible device, without initializing MUSA; ``()`` on any failure."""
    musa = _torch_musa()
    if musa is not None and _musa_initialized(musa):
        return _in_process_archs(musa) or ()
    return _musainfo_archs()


def _initialized_device_archs() -> Optional[Tuple[MusaArch, ...]]:
    """Archs of every visible device when MUSA is already initialized here, else ``None``."""
    musa = _torch_musa()
    if musa is None or not _musa_initialized(musa):
        return None
    return _in_process_archs(musa)


def _names(archs: Iterable[MusaArch]) -> str:
    return "[{}]".format(", ".join(arch.name for arch in archs))


def target_archs(
    env: Sequence[str], *, default: str = "mp_31", check_device: bool = False
) -> Tuple[MusaArch, ...]:
    """The MUSA archs to compile for, de-duplicated and sorted.

    Args:
        env: Environment variable names the caller honours, in precedence order. The
            first one with a non-empty value decides and later ones are not read. Its
            value is split on ``;`` and whitespace.
        default: Arch used when no variable is set and no device arch can be read.
        check_device: When MUSA is already initialized in this process, require every
            visible device's arch to be in the result. Skipped otherwise.

    Returns:
        The archs named by the deciding variable, else those of every visible device,
        else ``default``.

    Raises:
        TypeError: ``env`` is a single string instead of a sequence of names.
        ValueError: The deciding variable holds a malformed token or no token.
        RuntimeError: ``check_device`` is set and a visible device's arch is missing.
    """
    if isinstance(env, str):
        raise TypeError("env must be a sequence of variable names, not a string")
    archs = None
    decided_by = ""
    for variable in env:
        value = os.environ.get(variable, "")
        if not value:
            continue
        tokens = [token for token in _SEPARATOR_RE.split(value) if token]
        if not tokens:
            raise ValueError("{}={!r} names no MUSA arch".format(variable, value))
        try:
            archs = tuple(sorted(set(_parse(token) for token in tokens)))
        except ValueError as e:
            raise ValueError("{}={!r}: {}".format(variable, value, e)) from None
        decided_by = "{}={!r}".format(variable, value)
        break
    if archs is None:
        archs = _device_archs()
        decided_by = "the visible devices"
    if not archs:
        archs = (_parse(default),)
        decided_by = "the default {!r}".format(default)
    if check_device:
        devices = _initialized_device_archs()
        if devices is not None and any(device not in archs for device in devices):
            raise RuntimeError(
                "MUSA archs {} (from {}) do not cover the visible devices {}; kernels built "
                "for them do not launch there (musaErrorInvalidDeviceFunction)".format(
                    _names(archs), decided_by, _names(devices)
                )
            )
    return archs


def arch_flags(archs: Iterable[MusaArch]) -> List[str]:
    """One mcc ``--offload-arch=mp_XY`` flag per arch, de-duplicated and sorted."""
    return ["--offload-arch={}".format(arch.name) for arch in _unique(archs, "arch_flags")]


def torch_arch_list(archs: Iterable[MusaArch]) -> str:
    """The ``TORCH_MUSA_ARCH_LIST`` value torch_musa's ``MUSAExtension`` reads, e.g. ``31;32``."""
    return ";".join(
        "{}{}".format(arch.major, arch.minor) for arch in _unique(archs, "torch_arch_list")
    )


def cache_tag(archs: Iterable[MusaArch]) -> str:
    """Build-directory component for these archs and the MUSA toolkit, e.g. ``mp31_mu50200``.

    The toolkit suffix is ``torch.version.musa`` reduced to lowercase letters and
    digits, and is present only when torch reports a MUSA version.
    """
    tag = "_".join("mp{}{}".format(arch.major, arch.minor) for arch in _unique(archs, "cache_tag"))
    import torch

    musa_version = getattr(torch.version, "musa", None)
    if musa_version:
        tag += "_mu" + re.sub(r"[^0-9a-z]", "", str(musa_version).lower())
    return tag
