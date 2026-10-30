# SPDX-License-Identifier: Apache-2.0
"""Hardware facts and build identity for MUSA, shared by MUSA projects.

Why this package exists
-----------------------
Every MUSA project needs to answer two questions, and each one answers them
locally and slightly differently:

* **What device is this, and what can it do?**  Which arch, how many
  multiprocessors, is native FP8 part of this build, is the warp 32 or 128?
* **What was this binary built for?**  Which arch did the wheel target, which
  compiler produced it, and is that still the device in front of us?

Answering either one wrong is not a crash on MUSA.  A mubin compiled for another
arch loads, runs, and leaves the output buffer untouched while
``musaDeviceSynchronize`` returns success, so a wrong answer shows up as a wrong
number.  Sharing one implementation is therefore not only about removing
duplication: it is the only way the three consumers can be wrong in the same,
testable place.

``device.py`` answers the first question, by probing.  ``profiles.py`` answers it
for a bin that is not in front of you.  ``build.py`` answers the second, and
deliberately imports nothing but the standard library so a project's ``setup.py``
can use it while the package it is building does not exist yet -- through
:mod:`torchada_build`, which imports these files without importing torch.

What this package does **not** answer: whether a framework's tuned kernels have been
validated on an arch.  That is a property of one implementation and its version, so
each consumer keeps that policy for itself.
"""

from .build import (
    BuildStamp,
    BuildTargets,
    MusaBuildTargetMismatch,
    MusaTargetSelectionError,
    arch_codes,
    arch_list_forms,
    assert_build_matches_device,
    build_cache_tag,
    check_toolchain,
    clear_mcc_version_cache,
    compiler_arch_list,
    compiler_executable,
    mcc_arch_flags,
    mcc_version,
    parse_targets,
    read_build_stamp,
    resolve_targets,
    single_arch_define,
    uniform_native_fp8,
    write_build_stamp,
)
from .device import (
    NATIVE_FP4_ARCHS,
    NATIVE_FP8_ARCHS,
    MusaDeviceSpec,
    arch_code,
    arch_name,
    clear_device_spec_cache,
    current_device_spec,
    get_device_spec,
    has_s4_sqmma,
    supports_native_fp4,
    supports_native_fp8,
)
from .profiles import (
    MusaBinProfile,
    bin_profile,
    bin_profiles,
    known_mp_counts,
    profile_mismatch,
    verified_mp_counts,
)
from .targets import (
    TARGETS,
    UNKNOWN_ARCH,
    MusaArchSpec,
    get_arch_spec,
    native_fp4_arch_names,
    native_fp8_arch_names,
    s4_sqmma_arch_names,
    target_names,
)

__all__ = [
    "BuildStamp",
    "BuildTargets",
    "MusaArchSpec",
    "MusaBinProfile",
    "MusaBuildTargetMismatch",
    "MusaDeviceSpec",
    "MusaTargetSelectionError",
    "NATIVE_FP4_ARCHS",
    "NATIVE_FP8_ARCHS",
    "TARGETS",
    "UNKNOWN_ARCH",
    "arch_code",
    "arch_codes",
    "arch_list_forms",
    "arch_name",
    "assert_build_matches_device",
    "bin_profile",
    "bin_profiles",
    "build_cache_tag",
    "check_toolchain",
    "clear_device_spec_cache",
    "clear_mcc_version_cache",
    "compiler_arch_list",
    "compiler_executable",
    "current_device_spec",
    "get_arch_spec",
    "get_device_spec",
    "has_s4_sqmma",
    "known_mp_counts",
    "mcc_arch_flags",
    "mcc_version",
    "native_fp4_arch_names",
    "native_fp8_arch_names",
    "parse_targets",
    "profile_mismatch",
    "read_build_stamp",
    "resolve_targets",
    "s4_sqmma_arch_names",
    "single_arch_define",
    "supports_native_fp4",
    "supports_native_fp8",
    "target_names",
    "uniform_native_fp8",
    "verified_mp_counts",
    "write_build_stamp",
]
