# SPDX-License-Identifier: Apache-2.0
"""Recorded hardware facts per arch *bin*, for when the card is not in front of you.

:mod:`torchada.musa.device` answers for the device it is running on, by probing.  A tuning
layer needs a third thing: the same answers for a bin it is *not* running on, so a
table can be generated, a CI job can check it, and a build can name the variants
it is about to compile without holding the hardware.

Three rules keep this honest:

* which bins have a row, and which of them have been probed, are two separate
  facts.  A bin enters the model because
  :data:`torchada.musa.targets.TARGET_MP_COUNTS` lists it -- a broker label like
  "30 cores" is not a multiprocessor count, and this module never guesses one;
* one row serves every bin the table lists for its arch.  That is an assumption, so
  it is checked rather than left implicit: :data:`_VERIFIED_BINS` names the bins
  that were actually probed, and each of them agreed field for field.  ``source``
  names the probe and the verified bins;
* when the live board disagrees with the row, :func:`profile_mismatch` says which
  field differs instead of letting the table quietly become wrong.

The value this returns is a plain frozen dataclass: hashable, comparable, and a
good cache key and test fixture, because nothing here holds a device handle.

What "no torch" does and does not buy: this module imports nothing outside the
standard library (as does :mod:`torchada.musa.targets`), so it can never initialise
a MUSA runtime early or pull torch in through a back door.  A normal
``from torchada.musa import profiles`` is still not available without torch --
``torchada/__init__`` imports torch to patch it, so the package entry runs
``import torch`` first.  That is what :mod:`torchada_build` is for: it registers these
same files under a private package name and imports them without touching
``torchada/__init__``, so a ``setup.py`` or a PEP 517 backend can get the arch table on
a host that has no torch at all.  ``tests/test_musa_infra.py`` exercises that entry
point with ``torch``, ``torch_musa`` and ``numpy`` blocked, so the property fails
loudly the day one of these modules grows an import it should not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .targets import get_arch_spec, target_mp_counts

__all__ = [
    "MusaBinProfile",
    "bin_profile",
    "bin_profiles",
    "known_mp_counts",
    "profile_mismatch",
    "verified_mp_counts",
]


@dataclass(frozen=True)
class MusaBinProfile:
    """What one arch bin looks like, without asking the card.

    Field set and meaning match :class:`torchada.musa.device.MusaDeviceSpec`, minus
    the per-card values (``uuid``, ``total_memory``, the product name) which are not
    properties of a bin.
    """

    arch_name: str
    #: Multiprocessor count this row describes.
    mp_count: int
    warp_size: int
    shared_memory_per_block: int
    #: What a kernel may *request* for one block (the opt-in ceiling).  Equal to
    #: :attr:`shared_memory_per_block` on S5000, which is why recording both is
    #: cheap -- the pair is what tells a kernel author whether asking for 192 KiB is
    #: a formality or a hard step up from the default.
    shared_memory_per_block_optin: int
    shared_memory_per_multiprocessor: int
    regs_per_multiprocessor: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int
    #: Where these numbers came from, for the next person who has to trust them.
    source: str


#: Resources measured on one board of the arch, plus the bins the table lists for
#: it.  These are the constants a kernel's ``static_assert`` compiles against, which
#: is why a row is worth recording once rather than per probe.
_RECORDED: Dict[str, Tuple[Dict[str, int], str]] = {
    "mp_31": (
        {
            "warp_size": 32,
            "shared_memory_per_block": 196608,
            "shared_memory_per_block_optin": 196608,
            "shared_memory_per_multiprocessor": 196608,
            "regs_per_multiprocessor": 131072,
            "max_threads_per_block": 1024,
            "max_threads_per_multiprocessor": 3072,
        },
        "torch_musa.get_device_properties probe of an MTT S5000 at mp 48, 56 and 60 "
        "(identical on all three), generated/MUSA-100083/arch-findings.md F1",
    ),
}


#: Bins actually probed, per arch. The row above is reused for the arch's other
#: bins, so this is what that reuse rests on: every bin listed here was compared
#: field for field against the row and agreed.
_VERIFIED_BINS: Dict[str, Tuple[int, ...]] = {
    # MUSA-100083 (mp 56) plus an MP60 board and an MP48 board; all three reported
    # the same resource numbers.  The MP48 reading is the weakest of the three (it
    # was not taken on a lease this project held, see
    # generated/MUSA-60113/torchada-shared-infra-validation.md), which is why it is
    # listed here as a probe rather than as a lease-scoped measurement.
    "mp_31": (48, 56, 60),
}


def verified_mp_counts(arch_name: str) -> Tuple[int, ...]:
    """Bins of ``arch_name`` whose numbers were probed, not inferred from a sibling."""
    return _VERIFIED_BINS.get(arch_name, ())


def _profiles_of(arch_name: str) -> Tuple[MusaBinProfile, ...]:
    recorded = _RECORDED.get(arch_name)
    if recorded is None:
        return ()
    fields, source = recorded
    return tuple(
        MusaBinProfile(arch_name=arch_name, mp_count=mp_count, source=source, **fields)
        for mp_count in target_mp_counts(arch_name)
    )


#: Built once: the table is static, and callers compare by identity or by value.
_BIN_PROFILES: Dict[str, Tuple[MusaBinProfile, ...]] = {
    arch_name: _profiles_of(arch_name) for arch_name in _RECORDED
}


def known_mp_counts(arch_name: str) -> Tuple[int, ...]:
    """Multiprocessor counts recorded for ``arch_name``, empty when none are."""
    return target_mp_counts(arch_name)


def bin_profiles(arch_name: str) -> Tuple[MusaBinProfile, ...]:
    """Every recorded bin of ``arch_name``; empty when the arch has no record."""
    return _BIN_PROFILES.get(arch_name, ())


def bin_profile(arch_name: str, mp_count: Optional[int]) -> Optional[MusaBinProfile]:
    """The modelled profile for this arch and bin, or ``None`` when it is not modelled.

    ``None`` means the arch has no recorded row at all, or ``mp_count`` is not one
    of the bins the table lists for it -- never "a number I did not have".  Code that
    needs to know whether a bin was probed rather than inferred from a sibling asks
    :func:`verified_mp_counts`.
    """
    if mp_count is None:
        return None
    for profile in bin_profiles(arch_name):
        if profile.mp_count == mp_count:
            return profile
    return None


#: Fields compared between a recorded profile and the live board.
_COMPARED = (
    "warp_size",
    "shared_memory_per_block",
    "shared_memory_per_block_optin",
    "shared_memory_per_multiprocessor",
    "regs_per_multiprocessor",
    "max_threads_per_block",
    "max_threads_per_multiprocessor",
)


def profile_mismatch(spec: object) -> Optional[str]:
    """Describe how the live board disagrees with its recorded profile, if it does.

    ``spec`` is duck-typed (``arch_name`` and ``mp_count`` plus the compared
    fields) so this stays torch-free and testable with a stand-in.  Returns
    ``None`` when the bin is not modelled, which is not an error: it means this
    release has no numbers to disagree with.
    """
    arch_name = getattr(spec, "arch_name", "")
    if get_arch_spec(arch_name) is None:
        return None
    profile = bin_profile(arch_name, getattr(spec, "mp_count", None))
    if profile is None:
        return None
    differing = []
    for field in _COMPARED:
        live = getattr(spec, field, None)
        # ``_int_attr`` answers 0 for a field the properties do not carry -- that is
        # "unreported", not "disagrees".  None of these counters is legitimately
        # zero, so a falsy live value is skipped: otherwise a torch_musa release that
        # renamed a key would tell the reader to edit a row that is still correct,
        # which is the opposite of what this warning is for.
        if not live:
            continue
        recorded = getattr(profile, field)
        if live != recorded:
            differing.append("{}={} (recorded {})".format(field, live, recorded))
    if not differing:
        return None
    return "MUSA bin {}/mp{} differs from torchada's recorded profile: {}. The probe wins; update {}.".format(
        arch_name, profile.mp_count, ", ".join(differing), "devices._RECORDED"
    )
