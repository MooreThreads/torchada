# SPDX-License-Identifier: Apache-2.0
"""Build identity for MUSA: what a binary was compiled for, and its cache key.

Why this module exists
----------------------
A mubin produced for one MUSA arch does not fault when it runs on another:
``musaDeviceSynchronize`` returns success and the output buffer keeps whatever it
had.  Nothing in the runtime will tell you that the binary and the device
disagree, so the check has to live in build metadata:

* ``write_build_stamp()`` records the target arches and the compiler into the
  package that was just built;
* ``assert_build_matches_device()`` compares that record with the device in
  front of it and refuses to run on a mismatch;
* ``build_cache_tag()`` gives JIT caches a directory name that changes when either
  the arch or the compiler changes, so a stale object cannot be reused across them.

This module imports **only the standard library**.  A project's ``setup.py`` needs
to write a stamp before the package it is building can be imported, so importing
torch (or the project itself) here would be a circular dependency.
"""

from __future__ import annotations

import functools
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# The table is stdlib-only and imports nothing from this module, so no cycle.
from .targets import TARGETS

logger = logging.getLogger(__name__)

#: ``(name, module_name)`` pairs already told that the stamp is missing.  The check
#: runs on every device query, and an editable install would otherwise print the
#: notice at each one.
_WARNED_UNSTAMPED = set()


def _reset_unstamped_warnings() -> None:
    """Forget which packages have already been told their stamp is missing."""
    _WARNED_UNSTAMPED.clear()


__all__ = [
    "BuildStamp",
    "BuildTargets",
    "COMPILER_ARCH_LIST_ENV",
    "JIT_ARCH_LIST_ENVS",
    "MusaBuildTargetMismatch",
    "MusaTargetSelectionError",
    "TARGET_ENV",
    "TARGET_ENV_NAMES",
    "arch_codes",
    "arch_list_forms",
    "assert_build_matches_device",
    "build_cache_tag",
    "check_toolchain",
    "clear_mcc_version_cache",
    "compiler_arch_list",
    "compiler_argv",
    "compiler_executable",
    "mcc_arch_flags",
    "mcc_version",
    "musa_home",
    "parse_targets",
    "read_build_stamp",
    "resolve_targets",
    "single_arch_define",
    "uniform_native_fp8",
    "write_build_stamp",
]

#: Module name the stamp is written under, relative to a package directory.
STAMP_MODULE = "_build_stamp"

#: The variable a project's own ``setup.py`` should read: what this wheel targets.
TARGET_ENV = "MTGPU_TARGET"

#: Variables that pin a build target, in priority order, for code that has no
#: selector of its own (a JIT build inside a framework, say).
TARGET_ENV_NAMES = ("MTGPU_TARGET", "VLLM_MUSA_ARCH_LIST", "MATE_MUSA_ARCH_LIST")
_TARGET_ENV = TARGET_ENV_NAMES

#: The subset a JIT build inside a framework reads: it may not claim ``MTGPU_TARGET``,
#: which belongs to whoever builds the wheel.
JIT_ARCH_LIST_ENVS = ("VLLM_MUSA_ARCH_LIST", "MATE_MUSA_ARCH_LIST")

#: The variable ``torch_musa``'s extension builder turns into ``--offload-arch``.
COMPILER_ARCH_LIST_ENV = "TORCH_MUSA_ARCH_LIST"


class MusaBuildTargetMismatch(RuntimeError):
    """The compiled extension was built for a different MUSA arch.

    Deliberately not a plain ``RuntimeError``: consumers catch broad exceptions
    around device probes to fall back to a generic path, which is right for "the
    device is not ready yet" and catastrophic for "these kernels belong to another
    device".  A distinct type lets those handlers re-raise this one case.
    """


@dataclass(frozen=True)
class BuildStamp:
    """What a built artifact was compiled for.

    ``targets`` holds compiler target names (``"mp_31"``).  An empty tuple means
    the build did not record one; that is *unknown*, not a mismatch.
    """

    targets: Tuple[str, ...] = ()
    #: Whether the build carried native FP8 kernels (``-DENABLE_FP8``).
    fp8: bool = False
    #: Whether the build carried native FP4 kernels.  False on every arch so far, and
    #: recorded anyway: it is what the reader of a stamp asks, and a schema that has
    #: to grow when the first FP4 arch lands is a schema that breaks older stamps.
    fp4: bool = False
    mcc_version: str = "unknown"
    #: ``environment`` (pinned by env var), ``device`` (probed), or ``default``.
    source: str = "unknown"


class MusaTargetSelectionError(ValueError):
    """The requested arch selection cannot produce a self-consistent build.

    A subclass of ``ValueError`` because the usual cause is a bad value in an
    environment variable, and callers already report those as configuration errors.
    """


@dataclass(frozen=True)
class BuildTargets:
    """The arch(es) one build targets, and what that build may therefore carry.

    ``names`` holds compiler target names (``"mp_31"``), ``source`` says who chose
    them (``"environment"``, ``"device"`` or ``"default"``) because "nobody asked"
    and "we detected it" are different facts at review time.
    """

    names: Tuple[str, ...] = ()
    #: Whether this build carries native FP8 kernels (``-DENABLE_FP8``).
    fp8: bool = False
    #: Whether this build carries native FP4 kernels (no arch does today).
    fp4: bool = False
    source: str = "default"

    @property
    def display(self) -> str:
        """``"mp_22,mp_31"`` or ``"mp_31"`` -- what the stamp records."""
        return ",".join(self.names)

    @property
    def is_fat(self) -> bool:
        return len(self.names) > 1

    @property
    def codes(self) -> Tuple[str, ...]:
        """Two-digit codes (``("31",)``) as flags and tags spell them."""
        return tuple(name[3:] for name in self.names)

    @property
    def compiler_arch_list(self) -> str:
        """The ``TORCH_MUSA_ARCH_LIST`` value that builds these arches: ``"31;22"``."""
        return ";".join(self.codes)


def parse_targets(raw: str) -> Tuple[str, ...]:
    """``"mp_31,mp_22"`` (also ``"mp_31 mp_22"``, ``"31;22"``, ``"310 220"``) -> names.

    Normalisation goes through :func:`arch_codes`, so every accepted spelling of an
    arch agrees here exactly as it does in the stamp and in the compiler flags.
    """
    parts = [part for part in re.split(r"[;,\s]+", raw.strip()) if part]
    if not parts:
        raise MusaTargetSelectionError("empty MUSA arch selection")
    try:
        return tuple("mp_" + code for code in arch_codes(parts))
    except (RuntimeError, ValueError) as exc:
        raise MusaTargetSelectionError(str(exc)) from None


def resolve_targets(
    raw: Optional[str] = None,
    *,
    device_arch: Optional[str] = None,
    default: Optional[str] = "mp_31",
) -> BuildTargets:
    """Decide which arch(es) a build targets: environment, then device, then default.

    ``raw`` is the value to read first, normally ``os.environ.get(TARGET_ENV)``: this
    function does not look at the environment itself, because the caller knows which
    variables are authoritative for it (``TARGET_ENV_NAMES`` is the whole set a
    framework JIT might consult).

    ``default=None`` makes the last step raise instead.  That is what a JIT cache key
    wants -- there is no wheel to inspect later, so a guess would be cached under the
    wrong arch and reused.  ``setup.py`` keeps the ``mp_31`` default because a wheel
    with an unknown target is still checked against the device at import time.

    ``BuildTargets.fp8`` answers "does this build carry FP8 kernels", which is ``any``
    of the selected arches: ``-DENABLE_FP8`` is one flag for the whole compilation.
    Whether a *mixed* set is acceptable is the consumer's question -- a source tree
    that guards the FP8 kernels per arch can build one artifact for several arches,
    while a layout that compiles them unconditionally cannot.  That check is
    :func:`uniform_native_fp8`, called by the projects whose build layout requires it,
    rather than a parameter here that only one consumer would ever pass.
    """
    if raw and raw.strip():
        names, source = parse_targets(raw), "environment"
    elif device_arch:
        names, source = (device_arch,), "device"
    elif default:
        names, source = (default,), "default"
    else:
        raise MusaTargetSelectionError(
            "cannot tell which MUSA arch to build for: no MTGPU_TARGET/VLLM_MUSA_ARCH_LIST/"
            "MATE_MUSA_ARCH_LIST is set and no device answered"
        )

    _refuse_unknown(names)

    # ``-DENABLE_FP8`` is one flag for the whole compilation, so the build's answer is
    # "does it carry FP8 sources at all" -- ``any``, not the answer of one arch.
    return BuildTargets(
        names=names,
        fp8=any(TARGETS[name].native_fp8 for name in names),
        fp4=any(TARGETS[name].native_fp4 for name in names),
        source=source,
    )


def uniform_native_fp8(targets: BuildTargets) -> BuildTargets:
    """Refuse a target set whose arches disagree on native FP8; return it unchanged.

    A build layout that compiles its FP8 sources unconditionally cannot serve both an
    arch with FP8 hardware and one without: ``-DENABLE_FP8`` is one flag for the whole
    compilation, so those sources would be compiled for an arch that cannot execute
    them.  Projects whose layout does guard them per ``__MUSA_ARCH__`` simply do not
    call this.
    """
    disagree = [name for name in targets.names if not TARGETS[name].native_fp8]
    if targets.is_fat and disagree and len(disagree) != len(targets.names):
        raise MusaTargetSelectionError(
            "-DENABLE_FP8 is one flag for the whole build, but {} disagree on native "
            "FP8 ({} does not have it). Build one arch per wheel, or move the FP8 "
            "sources behind per-arch __MUSA_ARCH__ guards; a global flag would compile "
            "FP8 code for an arch that cannot execute it.".format(
                ", ".join(targets.names), ", ".join(disagree)
            )
        )
    return targets


def _refuse_unknown(names: Iterable[str]) -> None:
    """Reject an arch the table does not list, naming the ones it does."""
    unknown = [name for name in names if name not in TARGETS]
    if unknown:
        raise MusaTargetSelectionError(
            "unknown MUSA arch {}; known arches: {}".format(", ".join(unknown), ", ".join(TARGETS))
        )


def arch_list_forms(names: Iterable[str]) -> Dict[str, str]:
    """The same arch set in the spelling each consumer parses.

    Three tools read a target arch and each wants its own syntax, so a project that
    builds for more than the default arch writes the set three times and has to keep
    the copies in agreement -- the drift this module refuses at build time.  This
    returns all three from one list, sorted so the answer does not depend on order::

        >>> arch_list_forms(["mp_31", "mp_22"])
        {'target': 'mp_22,mp_31', 'torch': '22;31', 'mate': '2.2;3.1'}

    ``target`` is :data:`TARGET_ENV`'s form (what a ``setup.py`` reads), ``torch`` is
    ``TORCH_MUSA_ARCH_LIST``'s (what ``torch_musa`` turns into ``--offload-arch``),
    and ``mate`` is ``MATE_MUSA_ARCH_LIST``'s (dotted, for the MATE JIT).
    """
    parsed = parse_targets(names if isinstance(names, str) else " ".join(names))
    _refuse_unknown(parsed)
    codes = sorted({_normalize(name) for name in parsed})
    return {
        "target": ",".join("mp_" + code for code in codes),
        "torch": ";".join(codes),
        "mate": ";".join("{}.{}".format(code[0], code[1]) for code in codes),
    }


def compiler_arch_list(declared: Optional[str], targets: BuildTargets) -> str:
    """The ``TORCH_MUSA_ARCH_LIST`` value to build with, refusing a disagreement.

    ``torch_musa``'s extension builder turns that variable into ``--offload-arch``
    flags, so a caller keeps control of it -- but only if it names the same arches as
    the stamp.  Otherwise the wheel claims one arch while the object code targets
    another, and on MUSA that combination produces wrong numbers rather than an error.
    """
    wanted = targets.compiler_arch_list
    if not declared or not declared.strip():
        return wanted
    normalised = set(parse_targets(declared))
    normalised = {name[3:] for name in normalised}
    if normalised != set(wanted.split(";")):
        raise MusaTargetSelectionError(
            "TORCH_MUSA_ARCH_LIST={!r} disagrees with the selected arches {!r}: the "
            "wheel would be stamped for one arch while --offload-arch builds another. "
            "Set both, or neither.".format(declared, targets.display)
        )
    return ";".join(sorted(normalised))


#: Where ``torch_musa`` looks when neither ``MUSA_HOME`` nor ``MUSA_PATH`` is set and
#: ``mcc`` is not on ``PATH``.  A module constant rather than a literal in the function
#: so a test can point it at a temporary directory and check both branches.
MUSA_DEFAULT_HOME = "/usr/local/musa"


def musa_home() -> Optional[str]:
    """``MUSA_HOME`` as ``torch_musa`` resolves it, or ``None`` when it cannot.

    The same three guesses, in the same order, as
    ``torch_musa.utils.musa_extension._find_musa_home``: ``$MUSA_HOME`` or
    ``$MUSA_PATH``; then the parent of the ``bin`` directory holding the ``mcc`` on
    ``PATH``; then :data:`MUSA_DEFAULT_HOME` if it exists.  None means the builder
    would raise ``OSError`` when it asks for a MUSA path at all.

    Mirrored rather than re-invented because the whole point of
    :func:`compiler_executable` is to name the compiler that build will run: guessing
    ``/usr/local/musa`` on a host whose ``mcc`` comes from somewhere else would tag a
    cache directory with a version that produced none of its object code.
    """
    home = (os.environ.get("MUSA_HOME") or os.environ.get("MUSA_PATH") or "").strip()
    if home:
        return home
    found = shutil.which("mcc")
    if found:
        # ``<home>/bin/mcc`` is what the builder reconstructs from this, so the home is
        # the parent of ``bin``, not the directory of the executable.
        return os.path.dirname(os.path.dirname(found))
    if os.path.exists(MUSA_DEFAULT_HOME):
        return MUSA_DEFAULT_HOME
    return None


def compiler_executable() -> str:
    """The ``mcc`` command that ``torch_musa``'s extension builder will actually run.

    Not simply ``mcc`` from ``PATH``.  ``torch_musa.utils.musa_extension`` writes its
    ninja file with ``$PYTORCH_MCC`` when that is set, and with ``$MUSA_HOME/bin/mcc``
    (``$MUSA_PATH`` is accepted for the same variable) otherwise, falling back to
    :data:`MUSA_DEFAULT_HOME` when neither is set, so a version read from a different
    executable describes a compiler that is not building anything: with ``PYTORCH_MCC``
    pointing at a ccache wrapper for 5.2 and another ``mcc`` first on ``PATH``
    reporting 5.1, the cache tag and the stamp would say 5.1 while the object code came
    from 5.2.

    The result is a *command*, not a filename: ``$PYTORCH_MCC`` is written into a Ninja
    rule, so it may legitimately be ``env /path/to/mcc`` or ``ccache /path/to/mcc``.
    :func:`mcc_version` splits it into argv (no shell involved) when it runs it, which
    is why a launcher reports the version of the compiler behind it instead of
    ``unknown``.

    ``MCC`` alone is deliberately *not* consulted, for the same reason: the builder
    ignores it, so honouring it here would reintroduce the same disagreement from the
    other side.  A caller that wants a specific compiler passes it to
    :func:`mcc_version` and to the helpers that take ``compiler_version`` explicitly.
    """
    explicit = os.environ.get("PYTORCH_MCC", "").strip()
    if explicit:
        return explicit
    home = musa_home()
    if home:
        return os.path.join(home, "bin", "mcc")
    # No MUSA install found at all: the builder raises ``OSError`` from
    # ``_join_musa_home`` here, so this answers for a host that is about to fail anyway
    # rather than inventing a path.
    return "mcc"


def compiler_argv(executable: Optional[str] = None) -> tuple:
    """``compiler_executable()`` as an argv tuple, without shell interpretation.

    ``shlex.split`` rather than ``/bin/sh -c``: the value comes from a build wrapper
    and is put in a Ninja rule, so it is a command line, but running it through a shell
    would also hand it variable expansion and redirection.  An unbalanced quote yields
    the single element, which then fails to execute and answers ``"unknown"`` instead
    of raising from here.
    """
    command = executable if executable is not None else compiler_executable()
    try:
        argv = tuple(shlex.split(command))
    except ValueError:
        return (command,)
    return argv or (command,)


def mcc_version(mcc: Optional[str] = None, timeout: float = 10.0) -> str:
    """Version of the MUSA compiler, or ``"unknown"``.

    ``mcc --version`` prints the clang version first, so the match is anchored on
    the ``mcc version`` line: an unanchored search tags every cache directory with
    the clang version instead, which is both wrong and constant across compilers.

    With no argument the executable is :func:`compiler_executable` -- the one the
    build will use -- rather than whatever ``mcc`` resolves to on ``PATH``.  The value
    may be a command line rather than a filename (see :func:`compiler_argv`), so it is
    split into argv and executed without a shell.  Cached
    per resolved executable: a JIT build asks for this once per cache key (module
    name, tag, cache directory), which is several times per load, and the answer is a
    subprocess.  Keying on the executable rather than on nothing keeps a caller that
    switches compilers mid-process correct.
    """
    return _mcc_version_cached(mcc or compiler_executable(), timeout)


@functools.lru_cache(maxsize=8)
def _mcc_version_cached(executable: str, timeout: float) -> str:
    command = list(compiler_argv(executable)) + ["--version"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("cannot run %s: %s", command, exc)
        return "unknown"
    text = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"mcc\s+version\s+(\d+\.\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return match.group(1) if match else "unknown"


def clear_mcc_version_cache() -> None:
    """Forget the compiler version, e.g. after the toolchain on PATH changed.

    Only needed by a long-lived process that installs or switches a MUSA toolchain
    while running; a normal build asks once per executable and is done.
    """
    _mcc_version_cached.cache_clear()


def arch_codes(targets: Optional[Iterable[str]] = None) -> Sequence[str]:
    """Arch codes (``"31"``) a build targets, in flag order.

    With no argument the target is read from :data:`_TARGET_ENV`; when none of
    those is set, the caller is expected to have probed the device and pass the
    result in.  This function never guesses ``mp_31``: a wrong arch produces a
    binary that runs and returns wrong numbers.

    The input is materialised once, before anything reads it.  A generator is truthy
    even when it yields nothing, so an emptiness check on the argument itself was
    skipped for ``iter([])`` and the loop below produced an empty code list: the
    caller then emitted no ``--offload-arch`` at all, which is not an error to mcc but
    its own multi-arch default, i.e. a binary for the wrong arch.  An explicitly empty
    selection is refused rather than silently replaced by the environment: "no arches"
    is a different statement from "read the environment".
    """
    names = None if targets is None else tuple(targets)
    if names == ():
        raise MusaTargetSelectionError(
            "empty MUSA arch selection: a build with no --offload-arch flag gets "
            "mcc's own default targets, which is a binary for the wrong arch"
        )
    if not names:
        for name in _TARGET_ENV:
            value = os.environ.get(name, "").strip()
            if value:
                names = parse_targets(value)
                break
    if not names:
        raise RuntimeError(
            "cannot determine the MUSA arch: set MTGPU_TARGET (e.g. mp_31) or probe "
            "the device; guessing an arch produces a binary that runs on the wrong "
            "device and returns wrong results"
        )
    codes = []
    for target in names:
        code = _normalize(target)
        if code not in codes:
            codes.append(code)
    return codes


def _normalize(target: str) -> str:
    """``31`` / ``mp_31`` / ``MP31`` / ``3.1`` / ``310`` -> ``"31"``.

    Both number forms are accepted on purpose.  ``arch_name`` is the two-digit
    spelling used in flags and cache directories, ``arch_code`` is the
    three-digit value the template headers compare against, and callers mix them;
    the mapping between the two is exact for every arch that exists (``mp_31`` ->
    ``310``), so rescaling is total rather than a guess.

    Anything else is rejected instead of scaled.  ``"3100"``-style input would
    otherwise emit ``-D__MUSA_ARCH_LIST__=31000`` and name its cache ``mp_3100``,
    and a wrong arch here compiles one silent wrong branch instead of failing.
    """
    text = str(target).strip().lower()
    if text.startswith("mp_"):
        text = text[3:]
    elif text.startswith("mp"):
        text = text[2:]
    text = text.replace(".", "")
    if len(text) == 3 and text.endswith("0"):
        text = text[:2]  # the ``arch_code`` form: 310 -> 31
    if len(text) != 2 or not text.isdigit():
        raise ValueError(
            "not a MUSA arch target: {!r} (expected a two-digit name such as "
            "'mp_31', or its three-digit code '310')".format(target)
        )
    return text


def single_arch_define(
    name: str = "__MUSA_ARCH_LIST__", targets: Optional[Iterable[str]] = None
) -> str:
    """The single-arch ``-D<name>=<code>`` flag that shipped templates branch on.

    The template headers compare this macro against a single ``__MUSA_ARCH__``
    value (``== 310``, ``>= 220``), so the macro itself cannot express a fat build:
    there is no comparison a comma-separated list can satisfy.  A fat build
    therefore must not set it; the templates branch on ``__MUSA_ARCH__``, which the
    compiler defines per pass, and the arch list reaches the build through
    :func:`arch_flags` and the stamp instead.

    For one target the value is the three-digit form (``mp_31`` -> ``310``) those
    headers compare against, not the two-digit code in flag names; ``single_arch`` in
    the name is the restriction, so a call site cannot read it as a general arch define.
    """
    codes = arch_codes(targets)
    if len(codes) > 1:
        raise RuntimeError(
            "{} cannot describe {} arches: the template macro is compared against a "
            "single __MUSA_ARCH__ and a list has no value there.  For a fat build "
            "use mcc_arch_flags() for --offload-arch and branch on __MUSA_ARCH__ in "
            "the template instead.".format(name, len(codes))
        )
    return "-D{}={}".format(name, int(codes[0]) * 10)


def _version_tuple(text: str) -> Tuple[int, ...]:
    """``"5.2.0"`` -> ``(5, 2, 0)``; an unparsable string yields ``()``."""
    parts = re.findall(r"\d+", text or "")
    return tuple(int(part) for part in parts)


def check_toolchain(
    targets: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
    floor: Optional[str] = None,
) -> Optional[str]:
    """Refuse a selected target the compiler is too old for; returns the version read.

    ``floor`` is a *policy* the caller owns -- "this project supports mcc >= X" -- not
    a property of an arch.  The table records ``tested_mcc``, which is evidence about
    one build and says nothing about whether an older compiler could compile the same
    arch, so nothing is refused without a floor: ``floor=None`` only reports the
    version it read.

    Only the arches being built are checked, after the same normalisation every other
    entry point uses, so ``"21"``, ``"2.1"`` and ``"mp_21"`` all mean the arch that is
    actually selected.  Checking every row would refuse a build for an arch the caller
    never asked for -- an ``mp_21`` target rejected because ``mp_31`` elsewhere in the
    table has a higher recorded floor.

    ``version=None`` asks the compiler; a version of ``"unknown"`` (no ``mcc`` on this
    host, which is normal for a wheel built elsewhere) skips the check instead of
    failing a build that is not happening here.
    """
    resolved = version if version is not None else mcc_version()
    if floor is None or not resolved or resolved == "unknown":
        return resolved
    have = _version_tuple(resolved)
    if not have or have >= _version_tuple(floor):
        return resolved
    names = ", ".join("mp_{}".format(code) for code in arch_codes(targets))
    raise MusaTargetSelectionError(
        "{} needs mcc >= {} but this is {}; upgrade the toolchain or drop the arch "
        "from the build".format(names, floor, resolved)
    )


def mcc_arch_flags(targets: Optional[Iterable[str]] = None) -> List[str]:
    """``--offload-arch`` flags, one per target, for a single-arch or fat build.

    ``mcc`` accepts the flag more than once and bundles the resulting device code
    into one ``__CLANG_OFFLOAD_BUNDLE__``, so passing several targets produces a
    fat object; with no flag at all mcc's own default is a multi-arch bundle, so
    the flags are how a build states what it actually targets.  Named after ``mcc``
    because that is the compiler the spelling belongs to.

    The names are materialised once and normalised by :func:`arch_codes`, so an
    iterator argument cannot be spent before the flags are produced and ``"21"`` means
    ``mp_21``.  An empty selection raises there rather than returning ``[]``: an empty
    flag list hands the build mcc's own multi-arch default, which is the wrong-arch
    binary this module exists to prevent.

    No compiler-version check happens here: a floor is a project's support policy
    (:func:`check_toolchain` takes it explicitly), not an arch property.
    """
    return ["--offload-arch=mp_{}".format(code) for code in arch_codes(targets)]


def build_cache_tag(
    targets: Optional[Iterable[str]] = None, compiler_version: Optional[str] = None
) -> str:
    """Cache-directory tag for the current target.

    A cached object is valid only for the arch and compiler that produced it, so both
    are part of the tag: switching arch or compiler must not resolve to an existing
    directory.  ``cache`` in the name because this is a directory name, not a wheel
    tag; ``compiler_version`` because it takes a version string, not an executable
    (the executable is :func:`compiler_executable`).  The multiprocessor count is
    deliberately *not* part of it -- codegen input is arch plus compiler, and
    bin-dependent tuning belongs in a tuning table rather than in the cache path.
    """
    codes = arch_codes(targets)
    # Sorted so a fat build reaches one cache directory whatever order the arches
    # were listed in; the target set, not the command line, is the cache identity.
    arches = "_".join("mp_{}".format(code) for code in sorted(codes))
    return "{}__mcc{}".format(arches, compiler_version or mcc_version())


def write_build_stamp(
    dest: Path,
    targets: Iterable[str],
    fp8: bool = False,
    *,
    fp4: bool = False,
    compiler_version: Optional[str] = None,
    source: str = "unknown",
) -> Path:
    """Write ``_build_stamp.py`` into the package directory ``dest``.

    Call this from a ``build_ext`` *and* point it at the wheel's staging directory
    (``build/lib/<package>``) as well as the source tree: ``build_py`` runs before
    ``build_ext``, so a stamp written only into the source tree never reaches a
    wheel.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / (STAMP_MODULE + ".py")
    if isinstance(targets, str):
        targets = (targets,)  # read_build_stamp accepts a bare name; writing one must too
    names = repr(tuple(_arch_target(t) for t in targets))
    content = (
        '"""Generated by torchada.musa.build.write_build_stamp. Do not edit."""\n'
        "\n"
        "MTGPU_TARGETS = {}\n"
        "ENABLE_FP8 = {}\n"
        "ENABLE_FP4 = {}\n"
        "MCC_VERSION = {!r}\n"
        "TARGET_SOURCE = {!r}\n"
    ).format(names, bool(fp8), bool(fp4), compiler_version or mcc_version(), source)
    path.write_text(content)
    return path


def _arch_target(target: str) -> str:
    """``"31"`` -> ``"mp_31"``; leaves an already-named target alone."""
    text = str(target).strip()
    return text if text.startswith("mp_") else "mp_{}".format(_normalize(text))


def read_build_stamp(module_name: str) -> Optional[BuildStamp]:
    """Read a stamp written into ``module_name`` (e.g. ``"torchada._build_stamp"``).

    Returns ``None`` when the module is absent, which means "unknown", not "no
    arch": a source checkout that has never been built carries no stamp, and every
    caller must treat that as permission to run with a warning rather than as a
    mismatch.
    """
    import importlib

    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - an unbuilt checkout
        return None
    targets = getattr(module, "MTGPU_TARGETS", ())
    if isinstance(targets, str):
        # A hand-written stamp may name one arch instead of listing it.  ``tuple()``
        # on that string would yield ('m', 'p', '_', '3', '1'), so the tolerance
        # in ``assert_build_matches_device`` would never be reachable from here.
        targets = (targets,)
    return BuildStamp(
        targets=tuple(targets),
        fp8=bool(getattr(module, "ENABLE_FP8", False)),
        fp4=bool(getattr(module, "ENABLE_FP4", False)),
        mcc_version=getattr(module, "MCC_VERSION", "unknown"),
        source=getattr(module, "TARGET_SOURCE", "unknown"),
    )


def assert_build_matches_device(
    spec,
    stamp: Optional[BuildStamp] = None,
    *,
    module_name: Optional[str] = None,
    name: str = "this package",
    warn_when_unstamped: bool = False,
) -> bool:
    """Fail closed when the compiled extension targets a different arch than ``spec``.

    ``spec`` is a :class:`torchada.musa.device.MusaDeviceSpec`, passed untyped
    because this module must not import ``device`` -- ``device`` imports it.  An
    absent stamp is *unknown* and
    only warns: a source checkout has never recorded one, and refusing to run there
    would make every editable install unusable.

    Callers own the policy, so the whole check is one call::

        assert_build_matches_device(
            spec,
            module_name="vllm_musa._build_stamp",
            name="vllm_musa",
            warn_when_unstamped=True,
        )

    ``module_name`` reads the stamp from that module when ``stamp`` is not given;
    ``name`` is what the messages call the package being checked, because "this
    package" is ambiguous in a consumer's log; ``warn_when_unstamped`` says the
    silent-absent default is wrong for that call site (a project whose only
    wrong-arch defence is this check wants to hear that it is off).

    Returns ``True`` when a stamp was compared and matched, ``False`` when there was
    nothing to compare against.
    """
    if stamp is None and module_name is not None:
        stamp = read_build_stamp(module_name)
    if spec is None:
        # No spec means the device could not be probed, so there is nothing to
        # compare.  Answer before the warning, which names the device it lacks.
        return False
    missing = "no build stamp found"
    if stamp is not None and not stamp.targets:
        # A stamp that names no arch recorded nothing, so it is "unknown" rather than
        # a match -- and it is the case ``warn_when_unstamped`` exists for, because a
        # build that forgot to record its target has no other symptom.
        missing = "the build stamp names no arch"
        stamp = None
    if stamp is None:
        key = (name, module_name)
        if warn_when_unstamped and key not in _WARNED_UNSTAMPED:
            _WARNED_UNSTAMPED.add(key)
            logger.warning(
                "%s for %s (%s); its target arch cannot be "
                "checked against device %s. Set MTGPU_TARGET and rebuild to "
                "enable the check.",
                missing,
                name,
                module_name or "no module name given",
                spec.arch_name,
            )
        return False
    targets = stamp.targets
    if isinstance(targets, str):  # a hand-written stamp may carry one name
        targets = (targets,)
    if not targets:
        return False
    names = [_arch_target(target) for target in targets]
    if spec.arch_name not in names:
        raise MusaBuildTargetMismatch(
            "{} was built for MUSA arch {} but the device is {} ({}). "
            "A kernel binary for another arch loads and runs here without "
            "reporting an error, and produces wrong results, so this is refused. "
            "Rebuild with MTGPU_TARGET={} or install a build made for it.".format(
                name, ", ".join(names), spec.arch_name, spec.device_name, spec.arch_name
            )
        )
    if getattr(spec, "supports_native_fp8", False) and not stamp.fp8:
        # Not fatal: the FP8 kernels are simply not in this build, and callers fall
        # back.  Worth saying because it looks like a performance cliff rather than
        # a build choice.
        logger.warning(
            "%s has no FP8 support although %s has native FP8; FP8 paths will "
            "fall back to non-FP8 kernels (mcc %s, target source %s).",
            name,
            spec.arch_name,
            stamp.mcc_version,
            stamp.source,
        )
    return True
