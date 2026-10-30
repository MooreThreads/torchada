# SPDX-License-Identifier: Apache-2.0
"""What each MUSA architecture is, as data: one table, one place to learn it.

Every MUSA project asks the same questions of an arch -- does it have native FP8,
does it have FP4, which compiler was it built with -- and most answer them locally by
comparing a capability tuple at the call site.  A row per arch answers them instead,
so adding an arch is a data change plus the hardware run that justifies it rather than
a new branch in a dozen call sites.

Two things the table deliberately does not answer:

* **whether a build carries the kernels.**  Native FP8 is a per-target build fact
  (``-DENABLE_FP8``), so "the arch can" and "this wheel has it" are different
  questions; :class:`torchada.musa.build.BuildTargets` answers the second one.
* **whether a framework's tuned kernels are validated here.**  That is a property of
  one implementation and its version, not of the silicon: a shared flag would enable
  unrelated code paths in every consumer and cannot say which kernel, version or
  input contract was validated.  Each project keeps that policy for itself.

The table is plain data on purpose.  ``setup.py`` needs the FP8 answer before the
package being built exists, so this module imports nothing at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "MusaArchSpec",
    "TARGETS",
    "UNKNOWN_ARCH",
    "get_arch_spec",
    "native_fp4_arch_names",
    "native_fp8_arch_names",
    "s4_sqmma_arch_names",
    "target_mp_counts",
    "target_names",
]


@dataclass(frozen=True)
class MusaArchSpec:
    """One architecture, and the hardware facts that are true of it.

    :class:`torchada.musa.build.BuildTargets` is the other half of the picture: the
    arches *one build* selected.  This object says nothing about a build.
    """

    #: Compiler target name, e.g. ``"mp_31"``.
    arch_name: str
    #: Integer form device code compares against ``__MUSA_ARCH__`` (``310``).
    arch_code: int
    #: Whether the arch executes FP8 arithmetic in hardware.  Whether a given build
    #: *carries* FP8 kernels is ``BuildTargets.fp8``; the two can disagree, which is
    #: why they are separate questions.
    native_fp8: bool = False
    #: FP4 arithmetic in hardware.  ``False`` everywhere as of MUSA 5.2.0, S5000
    #: included: FP4 checkpoints run through software dequantisation, which is a
    #: different question from what the multiply-add can consume.
    native_fp4: bool = False
    #: Whether the arch has the packed ``FP16 x S4 -> F32F16S4`` SQMMA atom that
    #: 4-bit weight kernels use.  Named after the instruction on purpose: it says
    #: that one packed multiply-add exists, not that int4 arithmetic is available in
    #: general, so a caller cannot read it as "int4 support".
    s4_sqmma_fp16: bool = False
    #: ``mcc`` version this arch has actually been built and validated with, for the
    #: record.  Evidence, not a requirement: no compiler is refused on the strength of
    #: it, because "we tested 5.2" does not establish that 5.1 cannot build the arch.
    #: A project with a support policy passes that floor to
    #: :func:`torchada.musa.build.check_toolchain` itself.
    tested_mcc: Optional[str] = None


#: The arches this project builds and ships for: the two supported MUSA stacks,
#: ``mp_21``/``mp_22``, and ``mp_31`` (S5000).  A card that is not listed is still
#: *described* -- the probe answers through :data:`UNKNOWN_ARCH` -- but a build asked
#: to target it is refused, because an arch nobody has run is not a supported target
#: and guessing its answers is the failure this table exists to prevent.
TARGETS: Dict[str, MusaArchSpec] = {
    spec.arch_name: spec
    for spec in (
        # No compiler recorded for the pre-PH1 arches: nothing has failed to build
        # them, so a version here would be a guess dressed up as a measurement.
        MusaArchSpec("mp_21", 210),
        MusaArchSpec("mp_22", 220),
        # ``s4_sqmma_fp16``: a mute atom named ``MP31_16x64x64_F32F16S4_SS`` compiles
        # and runs on PH1, with packed ``0x11`` (two signed ``+1`` nibbles) producing
        # the expected dot product (sphere-kb probe
        # ``sqmma_marlin_moe_s4_20260429``, 2026-04-29); the atom catalog lists
        # ``FP16 x s4 -> F32F16S4`` with the caveat that the vendor PH PDF does not
        # print it (``docs/kernels/sqmma-atom-catalog.md``).
        MusaArchSpec(
            "mp_31",
            310,
            native_fp8=True,
            s4_sqmma_fp16=True,
            tested_mcc="5.2",
        ),
    )
}

#: What an arch that is not in :data:`TARGETS` answers.  Used by the device probe,
#: which must describe whatever card it finds even when we know nothing about it; a
#: *build* asked to target such an arch is refused instead.
UNKNOWN_ARCH = MusaArchSpec(arch_name="", arch_code=-1)


def target_names() -> Tuple[str, ...]:
    """Every arch name the table knows, in a stable order."""
    return tuple(TARGETS)


def get_arch_spec(arch_name: str) -> Optional[MusaArchSpec]:
    """The table entry for ``arch_name`` (``"mp_31"``), or ``None`` if unknown."""
    return TARGETS.get(arch_name)


def native_fp8_arch_names() -> Tuple[str, ...]:
    """Arches that *can* execute native FP8, build flag permitting."""
    return tuple(name for name, spec in TARGETS.items() if spec.native_fp8)


def native_fp4_arch_names() -> Tuple[str, ...]:
    """Arches with FP4 arithmetic in hardware (empty as of MUSA 5.2.0)."""
    return tuple(name for name, spec in TARGETS.items() if spec.native_fp4)


def s4_sqmma_arch_names() -> Tuple[str, ...]:
    """Arches with the packed ``FP16 x S4`` SQMMA atom (``("mp_31",)``)."""
    return tuple(name for name, spec in TARGETS.items() if spec.s4_sqmma_fp16)


#: Multiprocessor counts recorded per arch, read from device properties rather than
#: from a broker label (MUSA-100006 recorded all four for ``mp_31``).  An arch with
#: no entry has no recorded bin, and :func:`torchada.musa.profiles.bin_profile`
#: answers ``None`` for it rather than inventing one.
TARGET_MP_COUNTS: Dict[str, Tuple[int, ...]] = {
    "mp_31": (48, 52, 56, 60),
}


def target_mp_counts(arch_name: str) -> Tuple[int, ...]:
    """The multiprocessor counts recorded for ``arch_name``, empty when unknown."""
    return TARGET_MP_COUNTS.get(arch_name, ())
