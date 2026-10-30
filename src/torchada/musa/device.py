# SPDX-License-Identifier: Apache-2.0
"""Per-architecture hardware facts for MUSA devices.

What the probe answers, and what it does not
-------------------------------------------
The device reports numbers (capability, warp width, shared memory, multiprocessor
count); what a *kernel* may do with them is a per-arch fact and lives in
``torchada.musa.targets``.  Keeping the two apart is what lets a capability answer
stay a set membership rather than a comparison at the call site.

A probe that fails is **not cached**: the gates that read this module run at
import time in some projects, before the device is necessarily usable, and caching
that first ``None`` would pin every consumer to the generic path for the rest of
the process.

The field set mirrors ``torch_musa.get_device_properties()``.  Note the
difference from ``torch_musa.get_arch_list()``: that reports the arches the
**wheel was built for**, not the arch of the device in front of you.  Always take
the device arch from ``get_device_properties().major/minor``; the wheel's answer
belongs to :mod:`torchada.musa.build`.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# Direct submodule imports: `profiles` and `build` import only the standard library,
# so neither can cycle back here, and neither can pull torch in before `__init__`.
from . import profiles
from .build import MusaBuildTargetMismatch
from .targets import (
    UNKNOWN_ARCH,
    MusaArchSpec,
    get_arch_spec,
    native_fp4_arch_names,
    native_fp8_arch_names,
)

logger = logging.getLogger(__name__)

__all__ = [
    "NATIVE_FP4_ARCHS",
    "NATIVE_FP8_ARCHS",
    "MusaDeviceSpec",
    "arch_code",
    "arch_name",
    "clear_device_spec_cache",
    "current_device_spec",
    "get_device_spec",
    "supports_native_fp4",
    "supports_native_fp8",
]


# Capability sets, derived from :mod:`torchada.musa.targets`.  They are exposed
# here as arch *codes* because that is what a probed spec carries; ``targets`` is
# where the facts live and where a new arch is added.
NATIVE_FP8_ARCHS = frozenset(get_arch_spec(name).arch_code for name in native_fp8_arch_names())
NATIVE_FP4_ARCHS = frozenset(get_arch_spec(name).arch_code for name in native_fp4_arch_names())

# Arches whose warp is 32 lanes wide.  Also read from the device at runtime; this
# set only serves as a consistency check against that reading.
_WARP32_ARCHS = frozenset({310})


def arch_code(major: int, minor: int) -> int:
    """``(3, 1)`` -> ``310``.

    This is the number device code compares against ``__MUSA_ARCH__`` and the
    number the TileLang templates compare against ``-D__MUSA_ARCH_LIST__``, so it
    is spelled with the leading major multiplied by 100 -- not with the
    concatenation used in the ``mp_31`` name.
    """
    return int(major) * 100 + int(minor) * 10


def arch_name(major: int, minor: int) -> str:
    """``(3, 1)`` -> ``"mp_31"``, the compiler/CMake target name."""
    return "mp_{}{}".format(int(major), int(minor))


@dataclass(frozen=True)
class MusaDeviceSpec:
    """Immutable hardware facts for one MUSA device.

    Construct with :func:`get_device_spec`; the dataclass is frozen so a cached
    instance cannot be mutated into disagreeing with the device.
    """

    #: ``(major, minor)`` as reported by the driver, e.g. ``(3, 1)``.
    device_capability: Tuple[int, int]
    #: Integer arch code, ``major * 100 + minor * 10`` (``310`` for ``(3, 1)``).
    arch_code: int
    #: Compiler target name, e.g. ``"mp_31"``.
    arch_name: str
    #: Multiprocessor count this device reports at runtime.  It is a reading, not
    #: part of the arch identity, so tuning is never derived from the arch alone.
    mp_count: int
    #: Lanes per warp: 32 on mp_31, 128 on other current MUSA arches.
    warp_size: int
    total_memory: int
    l2_cache_size: int
    shared_memory_per_block: int
    #: Per-block local-memory ceiling a kernel gets when it *asks* for it
    #: (``cudaFuncAttributeMaxDynamicSharedMemorySize`` on the CUDA side).  A kernel
    #: that wants more than the default must request up to this number, and on parts
    #: where it differs from :attr:`shared_memory_per_block` the smaller one is the
    #: one that silently caps occupancy.  ``0`` means this ``torch_musa`` does not
    #: report the field at all ("no opinion", like the other int fields here), so a
    #: caller falls back to :attr:`shared_memory_per_block` rather than concluding
    #: that nothing may be requested.
    shared_memory_per_block_optin: int
    shared_memory_per_multiprocessor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int
    device_name: str
    uuid: Optional[str] = None

    @property
    def arch_spec(self) -> MusaArchSpec:
        """The table row for this arch, or :data:`UNKNOWN_ARCH` for a card we do not know.

        Every capability answer goes through here, so a card the table does not list
        answers ``False`` for everything instead of inheriting a neighbouring row.
        """
        return get_arch_spec(self.arch_name) or UNKNOWN_ARCH

    @property
    def supports_native_fp8(self) -> bool:
        """Whether the arch has FP8 arithmetic in hardware.

        Says nothing about whether the object code in front of you carries the FP8
        kernels; that is the build's answer, in ``torchada.musa.build``.
        """
        return self.arch_spec.native_fp8

    @property
    def supports_native_fp4(self) -> bool:
        """Whether the arch executes FP4 arithmetic, not whether FP4 can be run.

        S5000 answers ``False``, and FP4 weight checkpoints still run through the
        dequantisation path; the two questions must not be conflated.
        """
        return self.arch_spec.native_fp4

    @property
    def has_s4_sqmma(self) -> bool:
        """Whether the arch has the packed ``FP16 x S4`` SQMMA atom (see ``targets``)."""
        return self.arch_spec.s4_sqmma_fp16

    @property
    def bin_profile(self) -> Optional[profiles.MusaBinProfile]:
        """The recorded profile for this arch and bin, or ``None`` if unmodelled.

        The probe above is the authority; this is the same facts on paper, for a
        tuning table that has to be generated, or checked, without the card.
        """
        return profiles.bin_profile(self.arch_name, self.mp_count)

    def __str__(self):  # pragma: no cover - logging aid
        return "{} ({}, {}, mp={}, warp={})".format(
            self.device_name,
            self.arch_name,
            self.device_capability,
            self.mp_count,
            self.warp_size,
        )


def _musa_index(device_id: Any) -> int:
    """Normalize ``None`` / int / ``torch.device`` to a MUSA device index."""
    import torch

    if device_id is None:
        return int(torch.musa.current_device())
    if isinstance(device_id, int):
        return device_id
    index = getattr(device_id, "index", None)
    return int(torch.musa.current_device()) if index is None else int(index)


def _get_properties(device_id: int) -> Any:
    """``torch_musa`` device properties for ``device_id``.

    ``torch`` is imported here rather than at module scope so importing this
    module stays cheap, and so a process that never touches a device never pays
    for the runtime initialization.
    """
    import torch

    return torch.musa.get_device_properties(device_id)


def _int_attr(props: Any, *names: str) -> int:
    """First present attribute among ``names`` as an int, else 0.

    ``torch_musa`` has renamed a couple of property fields between releases
    (``max_threads_per_multi_processor`` vs ``max_threads_per_multiprocessor``),
    so read them tolerantly instead of pinning one spelling.
    """
    for name in names:
        value = getattr(props, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return 0


@functools.lru_cache(maxsize=None)
def _probe_device_spec(device_id: int) -> MusaDeviceSpec:
    """Build the spec for ``device_id``; raise when the device cannot describe it.

    Cached only on success.  Caching a failed probe would pin every gate that
    reads this module to the generic path for the rest of the process, which is
    exactly the decision that must be retried once the device finishes
    initializing.
    """
    props = _get_properties(device_id)
    major = int(getattr(props, "major"))
    minor = int(getattr(props, "minor"))
    return MusaDeviceSpec(
        device_capability=(major, minor),
        arch_code=arch_code(major, minor),
        arch_name=arch_name(major, minor),
        mp_count=_int_attr(props, "multi_processor_count"),
        warp_size=_int_attr(props, "warp_size"),
        total_memory=_int_attr(props, "total_memory"),
        l2_cache_size=_int_attr(props, "L2_cache_size", "l2_cache_size"),
        shared_memory_per_block=_int_attr(props, "shared_memory_per_block"),
        shared_memory_per_block_optin=_int_attr(
            props, "shared_memory_per_block_optin", "shared_memory_per_block_opt_in"
        ),
        shared_memory_per_multiprocessor=_int_attr(
            props,
            "shared_memory_per_multiprocessor",
            "shared_memory_per_multi_processor",
        ),
        regs_per_multiprocessor=_int_attr(
            props, "regs_per_multiprocessor", "regs_per_multi_processor"
        ),
        max_threads_per_block=_int_attr(props, "max_threads_per_block"),
        max_threads_per_multiprocessor=_int_attr(
            props,
            "max_threads_per_multiprocessor",
            "max_threads_per_multi_processor",
        ),
        device_name=str(getattr(props, "name", "unknown")),
        uuid=getattr(props, "uuid", None),
    )


_warned = set()


def _warn_once(key: str, message: str, *args: Any) -> None:
    """Log a diagnostic the first time per process, as these run per gate call."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def clear_device_spec_cache() -> None:
    """Forget every probed device and every once-only warning.

    Used by tests, and after an in-process rebuild (the arch may have changed).
    """
    _probe_device_spec.cache_clear()
    _warned.clear()


def get_device_spec(device_id: Optional[int] = None) -> Optional[MusaDeviceSpec]:
    """Return the spec for ``device_id`` (default: the current device) or ``None``.

    ``None`` means "could not determine", which every caller must treat as "not
    validated": falling back to the generic path is safe, whereas guessing an arch
    is not.  A failed probe is not cached, so a call made before the device is
    usable does not pin the gates for the rest of the process.
    """
    try:
        spec = _probe_device_spec(_musa_index(device_id))
    except MusaBuildTargetMismatch:
        # "Could not determine" and "this binary was built for another card" are
        # different answers: the first is safe to fall back from, the second is the
        # failure this whole check exists to catch, so it must not become ``None``.
        raise
    except Exception as exc:  # noqa: BLE001 - probe path, must not raise
        logger.debug("MUSA device properties unavailable for device %s: %s", device_id, exc)
        return None

    if spec.warp_size and (spec.warp_size == 32) != (spec.arch_code in _WARP32_ARCHS):
        # The device is authoritative; this only catches a wrong assumption in
        # code that hardcodes a warp width for an arch.
        _warn_once(
            "warp:{}".format(spec.arch_name),
            "MUSA arch %s reports warp_size=%d, which disagrees with the recorded "
            "narrow-warp arch set; device properties win.",
            spec.arch_name,
            spec.warp_size,
        )
    mismatch = profiles.profile_mismatch(spec)
    if mismatch:
        # The board is authoritative; a disagreeing row means the recorded table is
        # stale (new silicon stepping, or a typo), and a stale tuning input is worth
        # exactly one warning per bin rather than a silent wrong answer.  Keyed on the
        # bin, not just the arch: a heterogeneous node carries several, and the first
        # one to disagree must not silence the others.
        _warn_once("binprofile:{}:{}".format(spec.arch_name, spec.mp_count), "%s", mismatch)
    return spec


def current_device_spec() -> Optional[MusaDeviceSpec]:
    """Spec for the device this process is currently on."""
    return get_device_spec(None)


def _ask(device_id: Optional[int], question: str) -> bool:
    """Answer ``question`` from the spec of ``device_id``, with one failure policy.

    A probe that fails answers ``False``: a device that is not ready yet must not
    enable a tuned path, and an arch missing from the capability table is not in any
    explicit set either.  A build/device mismatch is re-raised instead, because
    ``False`` would send a wrong-arch binary down the generic path and leave it
    running -- the generic kernels of the same wrong binary.
    """
    try:
        spec = current_device_spec() if device_id is None else get_device_spec(device_id)
    except MusaBuildTargetMismatch:
        raise
    except Exception:  # noqa: BLE001 - a failed probe must not enable anything
        return False
    if spec is None:
        return False
    return bool(getattr(spec, question))


def has_s4_sqmma(device_id: Optional[int] = None) -> bool:
    """Whether this device's arch has the packed ``FP16 x S4`` SQMMA atom."""
    return _ask(device_id, "has_s4_sqmma")


def supports_native_fp8(device_id: Optional[int] = None) -> bool:
    """Whether this device's arch executes native FP8 arithmetic.

    Half the question: the wheel still has to be compiled with ``-DENABLE_FP8`` to
    carry the kernels, which is what :attr:`torchada.musa.build.BuildStamp.fp8`
    records.
    """
    return _ask(device_id, "supports_native_fp8")


def supports_native_fp4(device_id: Optional[int] = None) -> bool:
    """Whether this device's arch executes FP4 arithmetic (no arch does today).

    Deliberately not the same question as "can I serve an FP4 checkpoint": MX FP4
    weights run on any arch here through dequantisation, and answering ``True``
    because a checkpoint runs would advertise kernels that do not exist.
    """
    return _ask(device_id, "supports_native_fp4")
