"""
torchada.utils.cpp_extension - C++/CUDA extension utilities.

This module provides CUDAExtension, BuildExtension, and related utilities
that work on both CUDA and MUSA platforms.

Note: After importing torchada, you can use standard torch.utils.cpp_extension
imports - they are automatically patched to use these implementations on MUSA.

Usage (preferred):
    import torchada  # Apply patches first
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

    ext_modules = [
        CUDAExtension(
            name="my_extension",
            sources=["my_extension.cpp", "my_extension_kernel.cu"],
        )
    ]

    setup(
        name="my_package",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
"""

import functools
import importlib
import inspect
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .._mapping import _MAPPING_RULE, EXT_REPLACED_MAPPING
from .._platform import Platform, detect_platform, is_musa_platform

logger = logging.getLogger(__name__)

# Flag to track if torch_musa patches have been applied
_musa_patches_applied = False

# Flag to track if the libtorch-stable header backport has been applied. Kept
# separate from _musa_patches_applied so the (in-memory) import-time patches and
# the (on-disk) header backport are triggered independently — see
# _ensure_stable_headers_patched.
_stable_headers_patched = False

# CUDA/C++ translation units and headers whose CONTENT is CUDA→MUSA ported in
# place. SimplePorting.run() walks every file in a directory; restricting the
# substitution to these extensions keeps in-place porting from rewriting build
# scripts, templates, docs and configs (.py / .jinja / .cmake / .md / .json /
# .gitignore ...) onto themselves. See _patch_simple_porting_modify_file.
_PORTABLE_SOURCE_EXTS = (
    ".cu",
    ".cuh",
    ".cc",
    ".cpp",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".inl",
    ".inc",
    ".ipp",
    ".tpp",
    ".txx",
    ".ixx",
    ".cppm",
)

# MUSA-native sources are already MUSA and must NOT be CUDA→MUSA-substituted:
# the mapping would corrupt valid MUSA constructs (e.g. the asm-volatile
# neutralization rule, which disables un-assemblable CUDA PTX, would disable a
# hand-written MUSA inline-asm block and leave its output uninitialized). They
# compile as-is and are left byte-identical.
_MUSA_NATIVE_EXTS = (
    ".mu",
    ".muh",
)

# A vendor mapping header spells each CUDA name in terms of its MUSA
# counterpart (`#define cudaEventDisableTiming musaEventDisableTiming`,
# `#define CU_MEMORYTYPE_DEVICE MU_MEMORYTYPE_DEVICE`). Substituting the defined
# name as well collapses the line to `#define musaX musaX` -- a self-reference
# that shadows the real definition (`driver_types.h`) and leaves the symbol
# expanding to an undeclared identifier. Such a line is already MUSA-aware, so
# it is kept verbatim. Names never collide across the two sides of a genuine
# alias (`#define cudaCheck cudaCheckImpl` still ports), so only the degenerate
# mapping is skipped.
_SELF_REF_DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+(\w+)(?:\([^()]*\))?\s+\(*\s*(\w+)\s*(?:\([^()]*\))?\s*\)*\s*$"
)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//.*")


def _collapses_to_self_reference(ported: str) -> bool:
    """Whether a ported logical line became a self-referential ``#define X X``.

    Matches the whole logical line, since a mapping may put its replacement past
    a backslash continuation. Trailing comments and redundant parentheses around
    the replacement are ignored first: ``#define X (X)`` and ``#define X X /* n */``
    shadow the real definition exactly as ``#define X X`` does, and vendor headers
    routinely annotate their mappings. Only the decision reads the stripped text;
    the line itself is emitted untouched.
    """
    flat = re.sub(r"\\\s*\n\s*", " ", ported)
    flat = _BLOCK_COMMENT_RE.sub(" ", flat)
    flat = _LINE_COMMENT_RE.sub("", flat)
    match = _SELF_REF_DEFINE_RE.match(flat)
    return match is not None and match.group(1) == match.group(2)


def _get_cuda_home() -> Optional[str]:
    """
    Get the CUDA or MUSA home directory.

    On MUSA platform, this returns MUSA_HOME but is still called CUDA_HOME
    so developers don't need to change their code.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Check MUSA_HOME first, then common paths
        musa_home = os.environ.get("MUSA_HOME")
        if musa_home:
            return musa_home

        # Common MUSA installation paths
        common_paths = [
            "/usr/local/musa",
            "/opt/musa",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None

    elif platform == Platform.CUDA:
        # Use torch's CUDA_HOME
        try:
            from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME

            return TORCH_CUDA_HOME
        except ImportError:
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                return cuda_home
            if os.path.exists("/usr/local/cuda"):
                return "/usr/local/cuda"
            return None

    return None


def _is_cuda_file(path: str) -> bool:
    """Check if a file is a CUDA source file."""
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh"]


def _is_musa_file(path: str) -> bool:
    """
    Check if a file is a MUSA source file.
    Also recognizes .cu/.cuh files as MUSA files for compatibility.

    This function is used to patch torch_musa so it recognizes .cu files.
    """
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh", ".mu", ".muh"]


def _with_explicit_musa_language(flags):
    """Return MUSA compiler flags that force filename-independent parsing."""
    result = list(flags or [])
    for index, flag in enumerate(result):
        if flag == "-x=musa" or flag == "-xmusa":
            return result
        if flag == "-x" and index + 1 < len(result) and result[index + 1] == "musa":
            return result
    # Keep this as the final pre-source language option so it overrides any
    # earlier ``-x`` supplied by generic CUDA-oriented build configuration.
    return [*result, "-x", "musa"]


def _patch_musa_ninja_language(musa_ext):
    """Force ``mcc`` to parse identity-named ``.cu`` files as MUSA.

    torch_musa chooses its ``musa_compile`` Ninja rule via ``_is_musa_file``,
    but that rule historically relied on the ``.mu`` suffix to select the MUSA
    language. In-place porting deliberately keeps ``.cu`` names, so inject the
    explicit language flag into ``musa_cflags`` (which appears before the input
    path) for both setuptools and JIT extension builds.
    """
    original = getattr(musa_ext, "_write_ninja_file", None)
    if original is None or getattr(original, "_torchada_explicit_musa_language", False):
        return

    parameters = list(inspect.signature(original).parameters)
    if "musa_cflags" not in parameters:
        return
    musa_cflags_index = parameters.index("musa_cflags")

    @functools.wraps(original)
    def patched_write_ninja_file(*args, **kwargs):
        args = list(args)
        if "musa_cflags" in kwargs:
            kwargs["musa_cflags"] = _with_explicit_musa_language(kwargs["musa_cflags"])
        elif musa_cflags_index < len(args):
            args[musa_cflags_index] = _with_explicit_musa_language(args[musa_cflags_index])
        else:
            kwargs["musa_cflags"] = _with_explicit_musa_language(None)
        return original(*args, **kwargs)

    patched_write_ninja_file._torchada_explicit_musa_language = True
    musa_ext._write_ninja_file = patched_write_ninja_file


def _path_is_within(path: str, root: str) -> bool:
    """Return whether ``path`` resolves to ``root`` or one of its descendants."""
    try:
        return os.path.commonpath(
            (os.path.realpath(path), os.path.realpath(root))
        ) == os.path.realpath(root)
    except ValueError:
        return False


def _coalesce_port_roots(paths):
    """Canonicalize roots and retain only their shallowest non-overlapping set."""
    roots = sorted(
        {os.path.realpath(os.path.abspath(path)) for path in paths},
        key=lambda path: (len(path.split(os.sep)), path),
    )
    result = []
    for root in roots:
        if not any(_path_is_within(root, ancestor) for ancestor in result):
            result.append(root)
    return result


def _configured_exclusions() -> Tuple[List[str], List[str]]:
    """Resolve extra path roots and directory names excluded from porting.

    ``TORCHADA_EXCLUDE_DIRS`` is a path-list (``os.pathsep`` separated; commas
    are accepted as well). A bare entry such as ``torch_musa`` matches that
    directory name anywhere in an include path. If it is also importable, its
    package root is protected as well. The value adds to the existing system
    directory rules and is read at build time.
    """
    value = os.environ.get("TORCHADA_EXCLUDE_DIRS", "")
    separator_pattern = rf"[{re.escape(os.pathsep)},]"
    roots = []
    names = []
    for item in (part.strip() for part in re.split(separator_pattern, value)):
        if not item:
            continue

        expanded = os.path.expanduser(os.path.expandvars(item))
        if os.path.isabs(expanded) or os.sep in expanded:
            roots.append(os.path.realpath(os.path.abspath(expanded)))
            continue

        names.append(item)
        try:
            module = importlib.import_module(item)
        except (ImportError, AttributeError, OSError):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file:
            roots.append(os.path.realpath(os.path.dirname(module_file)))

    return _coalesce_port_roots(roots), names


def _configured_exclude_dirs() -> List[str]:
    """Return path roots configured through ``TORCHADA_EXCLUDE_DIRS``."""
    return _configured_exclusions()[0]


def _is_configured_exclude_dir(path: str) -> bool:
    """Return whether ``path`` matches a configured root or directory name."""
    roots, names = _configured_exclusions()
    if _path_overlaps_any(path, roots):
        return True
    path_parts = os.path.realpath(path).split(os.sep)
    return any(name in path_parts for name in names)


def _path_overlaps_any(path: str, roots: List[str]) -> bool:
    """Return whether ``path`` contains or is contained by a protected root."""
    return any(
        _path_is_within(path, root) or _path_is_within(root, path) for root in roots
    )


def _validate_portable_symlinks(source_dir: str) -> None:
    """Reject portable symlink files before upstream ``realpath`` can escape.

    SimplePorting resolves each filename before writing. Under in-place porting
    that would modify a symlink target, potentially outside the project tree.
    Failing explicitly preserves both the link and its target.
    """
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            path = os.path.join(root, name)
            if os.path.islink(path) and os.path.splitext(name)[1].lower() in _PORTABLE_SOURCE_EXTS:
                raise RuntimeError(f"Refusing to port symlinked CUDA/C++ source in place: {path}")


def _create_in_place_porter(musa_sp, source_dir: str, mapping_rule):
    """Construct SimplePorting without exposing ``<source>_musa`` to its init.

    Upstream initialization unconditionally removes and recreates the computed
    mirror directory. Seed it with a disposable directory, then redirect both
    input and output to the real source only after construction has completed.
    """
    with tempfile.TemporaryDirectory(prefix="torchada-port-init-") as temp_dir:
        seed_dir = os.path.join(temp_dir, "source")
        os.makedirs(seed_dir)
        porter = musa_sp.SimplePorting(cuda_dir_path=seed_dir, mapping_rule=mapping_rule)
    porter.cuda_dir_path = source_dir
    porter.musa_dir_path = source_dir
    return porter


def _patch_simple_porting_load_replaced_mapping(musa_sp):
    """
    Patch SimplePorting.load_replaced_mapping to suppress unwanted print output.

    Some versions of torch_musa have `print(self.mapping_rule)` in this method.
    This patch wraps the method to redirect stdout during execution.

    This is forward-compatible: if future versions remove the print, this still works.
    """
    import io
    import sys

    original_method = musa_sp.SimplePorting.load_replaced_mapping

    def patched_load_replaced_mapping(self):
        # Temporarily redirect stdout to suppress the print
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = original_method(self)
            # torch_musa's broad ``cuda.h`` rule also rewrites project-local
            # headers such as ``decode_jpegs_cuda.h`` even though their file
            # names are kept unchanged. Narrow it to actual CUDA header
            # includes so relative project includes keep resolving.
            self.mapping_rule = _narrow_cuda_header_mapping(self.mapping_rule)
            return result
        finally:
            sys.stdout = old_stdout

    musa_sp.SimplePorting.load_replaced_mapping = patched_load_replaced_mapping


def _narrow_cuda_header_mapping(mapping_rule):
    """Port CUDA headers without rewriting project-local header names."""
    narrowed = [(key, value) for key, value in mapping_rule if key != "cuda.h"]
    narrowed.extend(
        [
            ('#include <cuda.h>', '#include <musa.h>'),
            ('#include "cuda.h"', '#include "musa.h"'),
            ('#include <torch/cuda.h>', '#include <torch/musa.h>'),
            ('#include "torch/cuda.h"', '#include "torch/musa.h"'),
        ]
    )
    return sorted(narrowed, key=lambda item: len(item[0]), reverse=True)


_INCLUDE_DIRECTIVE_RE = re.compile(r'^\s*#\s*include\s*[<"](?P<header>[^>"]+)[>"]')
_NVJPEG_PREFIX_RULES = frozenset(("nvjpeg", "NVJPEG"))


def _replace_porting_line(line, mapping_rule):
    """Apply mappings without rewriting nvJPEG text in project header paths."""
    for key, value in mapping_rule:
        if key == value:
            continue
        if key in _NVJPEG_PREFIX_RULES:
            include = _INCLUDE_DIRECTIVE_RE.match(line)
            if include:
                start, end = include.span("header")
                header = line[start:end]
                if key == "nvjpeg" and header == "nvjpeg.h":
                    header = f"{value}.h"
                line = (
                    line[:start].replace(key, value)
                    + header
                    + line[end:].replace(key, value)
                )
                continue
        line = line.replace(key, value)
    return line


def _patch_simple_porting_open(musa_sp):
    """
    Patch simple_porting.open to tolerate non-UTF-8 bytes in source files.

    Some source files processed by SimplePorting may contain comments or string
    literals that are not valid UTF-8. The original implementation opens files
    with UTF-8 decoding, which can raise UnicodeDecodeError during porting.

    This patch wraps text-mode UTF-8 file opens with errors="surrogateescape"
    so undecodable bytes can round-trip safely. The original SimplePorting
    modify_file logic is preserved, keeping this patch forward-compatible with
    future torch_musa changes.
    """

    import builtins

    def open_with_surrogateescape(file, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("encoding") == "utf-8" and "errors" not in kwargs:
            kwargs["errors"] = "surrogateescape"
        return builtins.open(file, mode, *args, **kwargs)

    musa_sp.open = open_with_surrogateescape


def _patch_simple_porting_modify_file(musa_sp):
    """Make ``SimplePorting.modify_file`` (a) only rewrite compiled C/C++/CUDA
    files, (b) read the whole source before writing, so porting a file **in
    place** (destination path == source path) is safe, and (c) keep CUDA→MUSA
    mapping lines that substitution would collapse into a self-reference.

    Three problems with the stock method under in-place porting:

    1. ``SimplePorting.run`` walks *every* file in the tree and calls
       ``modify_file`` on each, regardless of extension. In the legacy
       ``<dir>_musa`` mirror mode that only wrote CUDA→MUSA-substituted copies
       into the throwaway mirror. In place (dst == src) it would rewrite build
       scripts, Jinja templates, docs and configs (``.py`` / ``.jinja`` /
       ``.md`` / ``.gitignore`` ...) onto themselves, corrupting committed
       tooling (e.g. a codegen ``generate.py`` gets ``cutlass``→``mutlass``
       applied to its imports). So the content substitution is gated to the
       CUDA/C++ translation-unit / header extensions in
       ``_PORTABLE_SOURCE_EXTS``; any other file is left byte-for-byte
       untouched (or copied verbatim if a distinct mirror destination is still
       in use). MUSA-native ``.mu``/``.muh`` sources (``_MUSA_NATIVE_EXTS``) are
       likewise left untouched: they are already MUSA, so substitution only
       corrupts them (e.g. the asm-volatile neutralization rule disables a
       hand-written MUSA inline-asm block, leaving its result uninitialized).

    2. The stock method opens the destination with ``"w"`` (truncating) while
       the source handle is open, zeroing the file when src == dst. Reading all
       lines up front before opening the destination makes the in-place write
       safe.

    3. A project-local vendor header may already map CUDA onto MUSA itself
       (``#define cudaEventDisableTiming musaEventDisableTiming``). Substituting
       the defined name turns it into ``#define musaEventDisableTiming
       musaEventDisableTiming``, which shadows the runtime's real definition
       with a self-reference, so the symbol expands to an undeclared identifier
       and every translation unit using it fails to compile. In mirror mode the
       original header stayed intact and only the throwaway copy was mangled;
       in place there is no intact copy left. Lines that collapse this way are
       kept verbatim -- see ``_collapses_to_self_reference``.
    """

    def modify_file(self, cuda_filepath, musa_filepath):
        ext = os.path.splitext(cuda_filepath)[1].lower()
        in_place = os.path.realpath(self.cuda_dir_path) == os.path.realpath(self.musa_dir_path)
        if ext in _MUSA_NATIVE_EXTS or ext not in _PORTABLE_SOURCE_EXTS:
            # Leave byte-identical: either MUSA-native (.mu/.muh — already MUSA,
            # substitution would corrupt it) or not a compiled source at all
            # (.py/.jinja/.md/...). In place (dst == src) leave it alone; for a
            # distinct destination (legacy mirror mode) copy it verbatim so the
            # mirror stays complete.
            # SimplePorting.change_filename turns a dotless name such as
            # ``Makefile`` into ``.Makefile``. When the roots are in-place, do
            # not copy to that synthetic path (or overwrite a real hidden file).
            if not in_place and os.path.abspath(cuda_filepath) != os.path.abspath(musa_filepath):
                import shutil

                shutil.copyfile(cuda_filepath, musa_filepath)
            return
        if in_place and not _path_is_within(cuda_filepath, self.cuda_dir_path):
            raise RuntimeError(
                f"Refusing to port CUDA/C++ source outside the in-place root: {cuda_filepath}"
            )
        with open(cuda_filepath, encoding="utf-8", errors="surrogateescape") as f:
            lines = f.readlines()

        def port_line(line):
            if line.startswith("*") or line.startswith("/") or line == "":
                return line
            if "cub/" not in line:
                line = _replace_porting_line(line, self.mapping_rule)
            return line

        out = []
        src_group = []
        ported_group = []

        def flush_group():
            ported_text = "".join(ported_group)
            if _collapses_to_self_reference(ported_text):
                out.append("".join(src_group))
            else:
                out.append(ported_text)
            src_group.clear()
            ported_group.clear()

        # Accumulate each logical line (backslash continuations included) so a
        # mapping that collapses to `#define X X` can be reverted as a whole.
        for line in lines:
            src_group.append(line)
            ported_group.append(port_line(line))
            if not line.rstrip("\n").endswith("\\"):
                flush_group()
        if src_group:
            flush_group()

        with open(musa_filepath, "w", encoding="utf-8", errors="surrogateescape") as f_musa:
            f_musa.writelines(out)

    musa_sp.SimplePorting.modify_file = modify_file


# Anchor for the accessor injection: the ``numel()`` definition that stock
# torch_musa 2.9 already ships. The backported block is spliced in just before
# it. ``mutable_data_ptr`` is the idempotency sentinel — absent in stock
# torch_musa, present after we patch — so a re-run is a no-op.
_STABLE_ACCESSOR_ANCHOR = "  int64_t numel() const {"

# torch::stable::Tensor accessors that torch_musa 2.9.0's older snapshot omits
# but stable-ABI kernels (vLLM, SGLang, ...) call. ``element_size`` /
# ``mutable_data_ptr<T>`` / ``const_data_ptr<T>`` are always injected; the
# sizes/strides/device/storage_offset/is_privateuseone block is gated on
# TORCHADA_STABLE_ACCESSORS (defined by torchada_stable_box.h, which also defines
# torch::stable::Device) so a TU that does not force-include the box header is
# untouched.
_STABLE_ACCESSOR_METHODS = (
    "  void* mutable_data_ptr() const { return data_ptr(); }\n"
    "  const void* const_data_ptr() const { return data_ptr(); }\n"
    "  template <typename T>\n"
    "  T* mutable_data_ptr() const { return reinterpret_cast<T*>(data_ptr()); }\n"
    "  template <typename T>\n"
    "  const T* const_data_ptr() const {\n"
    "    return reinterpret_cast<const T*>(data_ptr());\n"
    "  }\n"
    "  int64_t element_size() const {\n"
    "    return static_cast<int64_t>(\n"
    "        aoti_torch_dtype_element_size(static_cast<int32_t>(scalar_type())));\n"
    "  }\n"
    "#ifdef TORCHADA_STABLE_ACCESSORS\n"
    "  c10::IntArrayRef sizes() const {\n"
    "    int64_t* p;\n"
    "    TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(ath_.get(), &p));\n"
    "    return c10::IntArrayRef(p, dim());\n"
    "  }\n"
    "  c10::IntArrayRef strides() const {\n"
    "    int64_t* p;\n"
    "    TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(ath_.get(), &p));\n"
    "    return c10::IntArrayRef(p, dim());\n"
    "  }\n"
    "  torch::stable::Device device() const {\n"
    "    int32_t dt, di;\n"
    "    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(ath_.get(), &dt));\n"
    "    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_index(ath_.get(), &di));\n"
    "    return torch::stable::Device(dt, di);\n"
    "  }\n"
    "  bool is_privateuseone() const { return device().is_privateuseone(); }\n"
    "  int64_t storage_offset() const {\n"
    "    int64_t o;\n"
    "    TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_offset(ath_.get(), &o));\n"
    "    return o;\n"
    "  }\n"
    "#endif\n\n"
)

# Only column-0 method definitions are matched (``RetType Tensor::method(``); call
# sites inside bodies are indented and so never match, and the negative lookahead
# keeps already-inline / template / comment lines untouched, which makes the
# inline rewrite idempotent.
_TENSOR_INL_DEF_RE = re.compile(
    r"^(?!\s*(?:inline|template|//|\*))" r"([A-Za-z_][\w:<>,\s\*&]*?\bTensor::[A-Za-z_]\w*\s*\()"
)


def _inject_stable_accessors(text: str) -> Tuple[str, str]:
    """Splice the backported accessor block into a ``tensor_struct.h`` body.

    Pure string transform (no IO) so it is unit-testable off-MUSA. Returns
    ``(new_text, status)`` where status is one of ``"injected"`` (block added),
    ``"already"`` (sentinel present — no-op), or ``"anchor-missing"`` (the
    ``numel()`` anchor was not found, so nothing was injected — the caller should
    warn, since stable kernels will then fail to compile on the missing
    accessors).
    """
    if "mutable_data_ptr" in text:
        return text, "already"
    if _STABLE_ACCESSOR_ANCHOR not in text:
        return text, "anchor-missing"
    new_text = text.replace(
        _STABLE_ACCESSOR_ANCHOR,
        _STABLE_ACCESSOR_METHODS + _STABLE_ACCESSOR_ANCHOR,
        1,
    )
    return new_text, "injected"


def _inline_tensor_inl_defs(text: str) -> Tuple[str, bool]:
    """Prefix ``inline`` onto column-0 ``Tensor::`` method definitions.

    Pure string transform (no IO) so it is unit-testable off-MUSA. Returns
    ``(new_text, changed)``. Idempotency comes from the regex itself: its
    ``^(?!\\s*(?:inline|...))`` lookahead rejects a line that already starts with
    ``inline``, so a second pass is a no-op. The match is gated only on the
    regex — a substring check like ``"inline" not in line`` would also skip a
    def whose *trailing comment* happens to contain "inline", leaving it
    non-inline and reintroducing an ODR error.
    """
    lines = text.splitlines(keepends=True)
    changed = False
    for i, line in enumerate(lines):
        if _TENSOR_INL_DEF_RE.match(line):
            lines[i] = "inline " + line
            changed = True
    return "".join(lines), changed


def _patch_torch_musa_stable_headers() -> None:
    """Backport the libtorch-stable ``Tensor`` accessors that stable-ABI kernels
    (as built by vLLM, SGLang, and other torch_musa-dependent projects) use but
    torch_musa 2.9.0's older ``torch::stable`` snapshot omits: ``element_size`` /
    ``mutable_data_ptr<T>`` / ``const_data_ptr<T>`` / sizes / strides / device /
    storage_offset, and mark ``tensor_inl.h`` method definitions ``inline`` (they
    are emitted in every TU that includes them, which is a
    multiple-definition/ODR error otherwise).

    Patches BOTH the ``torch/include`` copy and the torch_musa
    ``generated_cuda_compatible/include`` copy. Idempotent + best-effort: a
    read-only install, or a future torch_musa that already ships these, is a
    no-op.

    This writes into the ``torch`` / ``torch_musa`` site-packages headers, so it
    is invoked lazily by ``_ensure_stable_headers_patched`` from the setuptools
    extension build entry points (``_create_musa_extension`` and the
    BuildExtension) rather than at ``import torchada`` — a bare import, or a
    pure-inference run, never mutates those headers. (The JIT ``load`` /
    ``load_inline`` helpers are intentionally excluded: torchada builds its own
    cpp ops through them at import, which would re-introduce an import-time
    write; libtorch-stable kernels are built via setuptools, not JIT-loaded.)
    """
    try:
        import torch
    except ImportError:
        return

    roots = [os.path.join(os.path.dirname(torch.__file__), "include")]
    try:
        import torch_musa

        roots.append(
            os.path.join(
                os.path.dirname(torch_musa.__file__),
                "share",
                "generated_cuda_compatible",
                "include",
            )
        )
    except ImportError:
        pass

    for root in roots:
        ts = os.path.join(root, "torch", "csrc", "stable", "tensor_struct.h")
        ti = os.path.join(root, "torch", "csrc", "stable", "tensor_inl.h")
        try:
            if os.path.exists(ts):
                with open(ts, encoding="utf-8") as f:
                    s = f.read()
                new_s, status = _inject_stable_accessors(s)
                if status == "injected":
                    with open(ts, "w", encoding="utf-8") as f:
                        f.write(new_s)
                elif status == "anchor-missing":
                    # The accessors were not injected, so stable kernels that call
                    # sizes()/strides()/device()/element_size() will fail to build.
                    # Name it instead of failing later with a cryptic C++ error.
                    logger.warning(
                        "torchada: could not backport torch::stable::Tensor "
                        "accessors into %s (numel() anchor not found); "
                        "libtorch-stable kernels may fail to compile",
                        ts,
                    )
            if os.path.exists(ti):
                with open(ti, encoding="utf-8") as f:
                    contents = f.read()
                new_ti, changed = _inline_tensor_inl_defs(contents)
                if changed:
                    with open(ti, "w", encoding="utf-8") as f:
                        f.write(new_ti)
        except OSError:
            pass  # read-only headers / no write perms — best effort


def _ensure_stable_headers_patched() -> None:
    """Apply the libtorch-stable header backport once, lazily, at build time.

    Separated from ``_apply_musa_patches`` (which runs at ``import torchada``)
    so a bare import — or a pure-inference run that never builds an extension —
    does not write into the ``torch`` / ``torch_musa`` site-packages headers.
    Called from the MUSA setuptools extension build entry points (extension
    construction + ``BuildExtension.build_extensions``); once the backport
    succeeds the flag makes further calls an instant no-op. Best-effort: never
    let it break a build.
    """
    global _stable_headers_patched
    if _stable_headers_patched:
        return
    if not is_musa_platform():
        return
    try:
        _patch_torch_musa_stable_headers()
        # Only mark done on success; an unexpected failure leaves the flag unset
        # so a later build retries (the backport is idempotent).
        _stable_headers_patched = True
    except Exception:  # noqa: BLE001  never let header patching break the build
        pass


def _apply_musa_patches():
    """
    Apply patches to torch_musa modules for CUDA compatibility.

    This function patches:
    1. musa_ext._is_musa_file - to recognize .cu/.cuh files as MUSA sources
    2. musa_ext._write_ninja_file - to pass ``-x musa`` before identity-named
       .cu inputs instead of relying on the legacy .mu suffix
    3. musa_sp.EXT_REPLACED_MAPPING - an identity map so porting keeps the
       original .cu/.cuh names (no .mu/.muh rename), so it can run in place
    4. musa_sp._MAPPING_RULE - to apply CUDA->MUSA symbol mapping
    5. musa_sp.SimplePorting.modify_file - restrict content substitution to
       compiled sources/headers and make in-place (dst == src) writes safe

    These patches are required to compile .cu files in place on MUSA platform.
    """
    global _musa_patches_applied

    if _musa_patches_applied:
        return

    if not is_musa_platform():
        return

    try:
        import torch_musa.utils.musa_extension as musa_ext
        import torch_musa.utils.simple_porting as musa_sp

        # Patch _is_musa_file to recognize .cu/.cuh files
        musa_ext._is_musa_file = _is_musa_file
        _patch_musa_ninja_language(musa_ext)

        # Patch EXT_REPLACED_MAPPING to an identity map: porting keeps the
        # original .cu/.cuh names so it runs in place (no .mu/.muh rename)
        musa_sp.EXT_REPLACED_MAPPING = EXT_REPLACED_MAPPING

        # Patch _MAPPING_RULE with our comprehensive CUDA->MUSA mappings
        # This is the critical patch that enables source code porting
        musa_sp._MAPPING_RULE = _MAPPING_RULE

        # Patch load_replaced_mapping to suppress print(self.mapping_rule)
        # Some versions of torch_musa have an extra print statement that we want to disable
        # This patch is forward-compatible - if the print is removed, this still works
        _patch_simple_porting_load_replaced_mapping(musa_sp)
        # Patch simple_porting.open to tolerate non-UTF-8 source files
        # This preserves SimplePorting's original logic while allowing undecodable bytes to round-trip
        _patch_simple_porting_open(musa_sp)
        # Patch modify_file to read-all-before-write so in-place porting (dst == src) is safe.
        _patch_simple_porting_modify_file(musa_sp)

        # NOTE: the libtorch-stable header backport is intentionally NOT applied
        # here. It writes into the torch / torch_musa site-packages headers, so
        # it is deferred to the build entry points via
        # _ensure_stable_headers_patched() — a bare ``import torchada`` (e.g. for
        # inference) must not mutate those headers.

        _musa_patches_applied = True

    except ImportError:
        # torch_musa not available, patches not needed
        pass


# Apply MUSA patches at module import time
_apply_musa_patches()

# Export CUDA_HOME - always use this name, even on MUSA platform
# This way developers don't need to change their code
CUDA_HOME = _get_cuda_home()


def _port_cuda_source(source_code: str, mapping_rules: Optional[Dict[str, str]] = None) -> str:
    """
    Port CUDA source code to MUSA by applying mapping rules.

    Args:
        source_code: The CUDA source code to port
        mapping_rules: Optional custom mapping rules (defaults to _MAPPING_RULE)

    Returns:
        The ported MUSA source code
    """
    if mapping_rules is None:
        mapping_rules = _MAPPING_RULE

    # Sort rules by length (longest first) to avoid partial replacements
    sorted_rules = sorted(mapping_rules.items(), key=lambda x: len(x[0]), reverse=True)
    return "".join(
        _replace_porting_line(line, sorted_rules)
        for line in source_code.splitlines(keepends=True)
    )


def include_paths(cuda: Optional[bool] = None, device_type: Optional[str] = None) -> List[str]:
    """
    Get include paths for compiling extensions.

    Supports both PyTorch < 2.6 (cuda=True) and PyTorch 2.6+ (device_type="cuda")
    signatures for compatibility.

    Args:
        cuda: (PyTorch < 2.6) Whether to include CUDA/MUSA paths. Deprecated in 2.6+.
        device_type: (PyTorch 2.6+) Device type string, e.g. "cuda", "cpu", "musa".

    Returns:
        List of include paths
    """
    # Handle both old (cuda=bool) and new (device_type=str) signatures
    if device_type is not None:
        # PyTorch 2.6+ style: device_type="cuda" or "cpu"
        # Translate "cuda" to MUSA include paths on MUSA platform
        include_device = device_type.lower() in ("cuda", "musa")
    elif cuda is not None:
        include_device = cuda
    else:
        # Default: include device paths
        include_device = True

    platform = detect_platform()

    if platform == Platform.MUSA:
        paths: List[str] = []
        try:
            import torch_musa.utils.musa_extension as musa_ext

            if hasattr(musa_ext, "include_paths"):
                # musa_ext uses musa=bool parameter, not cuda= or device_type=
                paths = list(musa_ext.include_paths(musa=include_device))
        except ImportError:
            pass

        if not paths:
            # Fallback: construct paths manually
            musa_home = _get_cuda_home()
            if musa_home:
                paths.append(os.path.join(musa_home, "include"))

        # Auto-append torchada's libtorch-stable ABI compat headers so
        # libtorch-stable kernels (vLLM, SGLang, ...) resolve
        # <torch/headeronly/core/Dispatch.h> on MUSA. Appended LAST so a future
        # torch_musa shipping the real header wins.
        if include_device:
            paths.append(stable_compat_include_dir())
        return paths

    else:
        # Check which signature the torch version supports
        import inspect

        from torch.utils.cpp_extension import include_paths as torch_include_paths

        sig = inspect.signature(torch_include_paths)
        if "device_type" in sig.parameters:
            # PyTorch 2.6+
            if device_type is not None:
                return torch_include_paths(device_type=device_type)
            else:
                return torch_include_paths(device_type="cuda" if include_device else "cpu")
        else:
            # PyTorch < 2.6
            return torch_include_paths(cuda=include_device)


def stable_compat_include_dir() -> str:
    """Directory holding torchada's libtorch-stable ABI compat headers.

    Add to an extension's ``include_dirs`` so libtorch-stable
    ``csrc/libtorch_stable`` kernels (vLLM, SGLang, ...) build on torch_musa: it
    shadows the (absent) ``torch/headeronly/core/Dispatch.h`` with a THO_DISPATCH
    shim. Pair with ``stable_compat_box_header`` (force-include) to also get the
    ``TORCH_BOX`` shim + ``CUDA_VERSION`` define.
    """
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "csrc",
        "stable_compat",
    )


def stable_compat_box_header() -> str:
    """Path to the force-include header providing ``TORCH_BOX`` for the stable ABI.

    Pass via ``-include`` in extra_compile_args when building libtorch-stable
    sources (mcc/g++ both accept ``-include <path>``).
    """
    return os.path.join(stable_compat_include_dir(), "torchada_stable_box.h")


def library_paths(cuda: Optional[bool] = None, device_type: Optional[str] = None) -> List[str]:
    """
    Get library paths for compiling extensions.

    Supports both PyTorch < 2.6 (cuda=True) and PyTorch 2.6+ (device_type="cuda")
    signatures for compatibility.

    Args:
        cuda: (PyTorch < 2.6) Whether to include CUDA/MUSA library paths. Deprecated in 2.6+.
        device_type: (PyTorch 2.6+) Device type string, e.g. "cuda", "cpu", "musa".

    Returns:
        List of library paths
    """
    # Handle both old (cuda=bool) and new (device_type=str) signatures
    if device_type is not None:
        # PyTorch 2.6+ style: device_type="cuda" or "cpu"
        # Translate "cuda" to MUSA library paths on MUSA platform
        include_device = device_type.lower() in ("cuda", "musa")
    elif cuda is not None:
        include_device = cuda
    else:
        # Default: include device paths
        include_device = True

    platform = detect_platform()

    if platform == Platform.MUSA:
        if not include_device:
            return []

        try:
            import torch_musa.utils.musa_extension as musa_ext

            if hasattr(musa_ext, "library_paths"):
                # musa_ext uses musa=bool parameter, not cuda= or device_type=
                return musa_ext.library_paths(musa=include_device)
        except ImportError:
            pass

        # Fallback: construct paths manually
        paths = []
        musa_home = _get_cuda_home()
        if musa_home:
            paths.append(os.path.join(musa_home, "lib"))
            paths.append(os.path.join(musa_home, "lib64"))
        return [p for p in paths if os.path.exists(p)]

    else:
        # Check which signature the torch version supports
        import inspect

        from torch.utils.cpp_extension import library_paths as torch_library_paths

        sig = inspect.signature(torch_library_paths)
        if "device_type" in sig.parameters:
            # PyTorch 2.6+
            if device_type is not None:
                return torch_library_paths(device_type=device_type)
            else:
                return torch_library_paths(device_type="cuda" if include_device else "cpu")
        else:
            # PyTorch < 2.6
            return torch_library_paths(cuda=include_device)


def _stable_header_backport_required() -> bool:
    """Return whether this torch version needs the stable-ABI header backport.

    torch 2.11 and newer provide the stable ABI directly. The supported torch
    2.9 line still needs torchada's compatibility backport. Parse only the
    major/minor prefix so vendor and development suffixes do not affect the
    decision.
    """
    import torch

    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    if match is None:
        logger.warning(
            "Unable to determine whether torch %r needs the stable header backport; "
            "applying it for compatibility",
            torch.__version__,
        )
        return True
    return (int(match.group(1)), int(match.group(2))) < (2, 11)


class CUDAExtension:
    """
    A wrapper that creates either a torch CUDAExtension or MUSA MUSAExtension.

    This class provides a unified interface for building CUDA extensions
    that works transparently on both CUDA and MUSA platforms.
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """
        Create a new extension module.

        Args:
            name: The name of the extension
            sources: List of source files
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        platform = detect_platform()
        if platform == Platform.MUSA:
            return _create_musa_extension(name, sources, *args, **kwargs)
        return _create_cuda_extension(name, sources, *args, **kwargs)


class CppExtension:
    """
    A wrapper for creating C++ extensions (no CUDA/MUSA).
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """Create a C++ extension module."""
        from torch.utils.cpp_extension import CppExtension as TorchCppExtension

        return TorchCppExtension(name, sources, *args, **kwargs)


def _create_cuda_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a CUDA extension using torch's CUDAExtension."""
    from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

    return TorchCUDAExtension(name, sources, *args, **kwargs)


def _translate_compile_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate CUDA-style compile args to MUSA-style.

    This function maps:
    - 'nvcc' key to 'mcc' for MUSA compiler flags
    - Keeps 'cxx' key as-is for C++ compiler flags

    This allows developers to use standard 'nvcc' key in extra_compile_args
    and have it work transparently on MUSA platform.
    """
    if "extra_compile_args" not in kwargs:
        return kwargs

    extra_compile_args = kwargs["extra_compile_args"]
    if not isinstance(extra_compile_args, dict):
        return kwargs

    # Create a new dict with translated keys
    new_compile_args = {}
    for key, value in extra_compile_args.items():
        if key == "nvcc":
            # Map 'nvcc' to 'mcc' for MUSA
            new_compile_args["mcc"] = value
        else:
            new_compile_args[key] = value

    # Return a copy of kwargs with updated extra_compile_args
    new_kwargs = kwargs.copy()
    new_kwargs["extra_compile_args"] = new_compile_args
    return new_kwargs


def _translate_link_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Translate canonical CUDA library and feature names used by extensions."""
    new_kwargs = kwargs.copy()

    if kwargs.get("libraries") is not None:
        new_kwargs["libraries"] = [
            "mtjpeg" if library == "nvjpeg" else library
            for library in kwargs["libraries"]
        ]

    if kwargs.get("define_macros") is not None:
        new_kwargs["define_macros"] = [
            ("MTJPEG_FOUND" if name == "NVJPEG_FOUND" else name, value)
            for name, value in kwargs["define_macros"]
        ]

    return new_kwargs


def _create_musa_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a MUSA extension using torch_musa's MUSAExtension.

    The patches applied by _apply_musa_patches() make MUSAExtension accept
    .cu/.cuh files directly by:
    1. Patching musa_ext._is_musa_file to recognize .cu/.cuh as valid MUSA files
    2. Patching musa_sp.EXT_REPLACED_MAPPING to an identity map so .cu/.cuh keep
       their names (no .mu/.muh rename)
    3. Patching musa_sp._MAPPING_RULE to convert CUDA symbols to MUSA in source code
    4. Translating 'nvcc' compile args key to 'mcc' for MUSA compiler
    """
    # Ensure patches are applied
    _apply_musa_patches()
    # torch 2.9 needs the compatibility backport; torch 2.11+ provides the
    # stable ABI directly.
    if _stable_header_backport_required():
        _ensure_stable_headers_patched()

    # Translate CUDA compiler, library, and feature-macro names to MUSA.
    kwargs = _translate_compile_args(kwargs)
    kwargs = _translate_link_args(kwargs)

    try:
        import torch_musa.utils.musa_extension as musa_ext

        # Simply pass sources to MUSAExtension - patches make it accept .cu files
        return musa_ext.MUSAExtension(name, sources, *args, **kwargs)
    except ImportError:
        # Fallback to torch's CUDAExtension if torch_musa is not available
        from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

        return TorchCUDAExtension(name, sources, *args, **kwargs)


def _get_build_extension_class():
    """
    Get the BuildExtension class for the current platform.

    On MUSA platform, returns a custom class that:
    1. Uses SimplePorting to convert CUDA sources to MUSA in place in run()
    2. Registers .cu/.cuh as valid source extensions in build_extensions()
    3. Provides extensible mapping rules via get_mapping_rule() method

    The porting is automatic and transparent: developers list csrc/*.cu source
    paths and the build ports each project-local include root in place (no
    <root>_musa mirror, no .cu->.mu rename), so original #include paths and
    source paths resolve as-is and nothing downstream needs rewriting.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()
        try:
            import torch_musa.utils.musa_extension as musa_ext
            import torch_musa.utils.simple_porting as musa_sp

            # Patch _is_musa_file to also recognize .cu/.cuh files as MUSA sources
            # This allows keeping original CUDA file extensions while still compiling
            # them with the MUSA compiler (mcc with -x musa flag)
            _original_is_musa_file = musa_ext._is_musa_file

            def _patched_is_musa_file(path: str) -> bool:
                """Check if a file is a MUSA source file (including .cu/.cuh)."""
                ext = os.path.splitext(path)[1].lower()
                # Include .cu/.cuh in addition to .mu/.muh
                if ext in [".cu", ".cuh"]:
                    return True
                return _original_is_musa_file(path)

            musa_ext._is_musa_file = _patched_is_musa_file

            class _MUSABuildExtension(musa_ext.BuildExtension):
                """
                Custom BuildExtension that handles CUDA->MUSA source porting.

                - run(): ports each project-local include root's CUDA sources to
                  MUSA in place (no <root>_musa mirror, no .cu->.mu rename), so
                  original includes and source paths resolve as-is without
                  per-file rewriting and no header is reachable through two trees
                - build_extensions(): registers .cu/.cuh as valid extensions
                - get_mapping_rule(): returns mapping rules (override to extend)

                Subclasses can override get_mapping_rule() to add project-specific mappings:

                    class MyBuildExt(_MUSABuildExtension):
                        def get_mapping_rule(self):
                            base_rules = super().get_mapping_rule()
                            return {
                                **base_rules,
                                "my_cuda_func": "my_musa_func",
                            }
                """

                _ported_dirs = set()

                def get_mapping_rule(self):
                    """
                    Get the CUDA->MUSA mapping rules for source porting.

                    Override this method in subclasses to add project-specific mappings.
                    Call super().get_mapping_rule() and merge with additional rules.

                    Returns:
                        dict: Mapping from CUDA symbols to MUSA equivalents
                    """
                    return _MAPPING_RULE.copy()

                def build_extensions(self):
                    # torch 2.9 needs the compatibility backport; torch 2.11+
                    # provides the stable ABI directly.
                    if _stable_header_backport_required():
                        _ensure_stable_headers_patched()
                    # Register .cu, .cuh as valid source extensions
                    self.compiler.src_extensions += [".cu", ".cuh"]
                    super().build_extensions()

                def _port_directory(self, source_dir, mapping_rule=None):
                    """Port a directory's CUDA sources to MUSA **in place** (no
                    ``<dir>_musa`` mirror): SimplePorting rewrites each file's
                    content and, with the identity extension map, keeps its name.
                    Original ``#include`` paths and source paths therefore stay
                    valid, so nothing downstream needs rewriting. Idempotent per
                    process via ``_ported_dirs``.
                    """
                    if mapping_rule is None:
                        mapping_rule = self.get_mapping_rule()

                    source_dir = os.path.realpath(os.path.abspath(source_dir))
                    if source_dir in self._ported_dirs:
                        return source_dir

                    _validate_portable_symlinks(source_dir)
                    musa_sp.LOGGER.setLevel(logging.ERROR)
                    sp = _create_in_place_porter(musa_sp, source_dir, mapping_rule)
                    sp.run()

                    self._ported_dirs.add(source_dir)
                    return source_dir

                @staticmethod
                def _dir_has_portable_sources(path):
                    """True if ``path`` recursively holds any file the porter would
                    rewrite — any extension in ``_PORTABLE_SOURCE_EXTS``. Kept in
                    sync with the porting allowlist so a directory of only
                    ``.cc``/``.cpp`` sources (no ``.h``/``.cu``) is not skipped. A
                    directory of only MUSA-native ``.mu``/``.muh`` sources has
                    nothing to port and is intentionally not a porting target."""
                    try:
                        for _root, _dirs, files in os.walk(path):
                            for f in files:
                                if os.path.splitext(f)[1].lower() in _PORTABLE_SOURCE_EXTS:
                                    return True
                    except OSError:
                        pass
                    return False

                @staticmethod
                def _is_system_include_dir(path):
                    is_system_path = (
                        path.startswith("/usr/")
                        or path.startswith("/opt/")
                        or "site-packages" in path
                        or "dist-packages" in path
                    )
                    if is_system_path:
                        return True

                    # Additional dependency roots can be supplied as paths or
                    # package names, including editable installs such as
                    # TORCHADA_EXCLUDE_DIRS=torch_musa.
                    return _is_configured_exclude_dir(path)

                def run(self):
                    """Port each project-local include root's CUDA sources to MUSA
                    **in place** (no ``<dir>_musa`` mirror, no ``.cu``->``.mu``
                    rename), then compile.

                    Because nothing moves or is renamed, every source's original
                    ``#include`` directives -- relative (``../x``) and root-relative
                    (``dir/x``) alike -- still resolve against the same include roots,
                    and the extension's source paths stay valid. So there is no
                    per-file include rewriting, no source-path remapping, and no
                    second mirror to cause cross-tree ODR. ``.cu``/``.cuh`` compile as
                    MUSA via the patched ``_is_musa_file`` and explicit
                    ``-x musa`` compiler flag.
                    """
                    mapping_rule = self.get_mapping_rule()
                    self._ported_dirs = set()
                    candidate_dirs = []
                    for ext in self.extensions:
                        # System filtering applies only to include roots. An
                        # explicit source parent is always project-owned even
                        # when the checkout lives under /opt, /usr/src, or a
                        # site-packages tree.
                        for include_dir in list(getattr(ext, "include_dirs", None) or []):
                            root = os.path.realpath(os.path.abspath(include_dir))
                            if not self._is_system_include_dir(root):
                                candidate_dirs.append(root)
                        for src in list(getattr(ext, "sources", None) or []):
                            candidate_dirs.append(os.path.dirname(os.path.abspath(src)))

                        # ext.sources and ext.include_dirs are intentionally left
                        # unchanged: the tree is ported in place, so the original
                        # paths and includes resolve as-is.

                    # Coalesce all extensions at once so a child include root is
                    # never ported and then recursively ported again through a
                    # later source-parent ancestor.
                    for root in _coalesce_port_roots(candidate_dirs):
                        if os.path.isdir(root) and self._dir_has_portable_sources(root):
                            self._port_directory(root, mapping_rule)

                    super().run()

            return _MUSABuildExtension
        except ImportError:
            pass

    # Fallback to torch's BuildExtension
    from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension

    return TorchBuildExtension


# Get the actual BuildExtension class at module load time
# This ensures BuildExtension is a proper class that inherits from Command
BuildExtension = _get_build_extension_class()


def load(
    name: str,
    sources: List[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    is_standalone: bool = False,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension at runtime (JIT compilation).

    This function works on both CUDA and MUSA platforms.

    Args:
        name: The name of the extension
        sources: List of source files
        extra_cflags: Extra C++ compiler flags
        extra_cuda_cflags: Extra CUDA/MUSA compiler flags
        extra_ldflags: Extra linker flags
        extra_include_paths: Extra include paths
        build_directory: Directory to build in
        verbose: Whether to print build output
        with_cuda: Whether to include CUDA/MUSA support
        is_python_module: Whether this is a Python module
        is_standalone: Whether this is a standalone executable
        keep_intermediates: Whether to keep intermediate files

    Returns:
        The loaded extension module
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()

        try:
            import torch_musa.utils.musa_extension as musa_ext

            # Use MUSA's load function if available
            # Note: MUSA uses different parameter names:
            #   extra_cuda_cflags -> extra_musa_cflags
            #   with_cuda -> with_musa
            if hasattr(musa_ext, "load"):
                return musa_ext.load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_musa_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory,
                    verbose=verbose,
                    with_musa=with_cuda,
                    is_python_module=is_python_module,
                    is_standalone=is_standalone,
                    keep_intermediates=keep_intermediates,
                )
        except ImportError:
            pass

    # Fallback to torch's load
    from torch.utils.cpp_extension import load as torch_load

    return torch_load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
        keep_intermediates=keep_intermediates,
    )


def load_inline(
    name: str,
    cpp_sources: List[str],
    cuda_sources: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    with_pytorch_error_handling: bool = True,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension from inline source code.

    This function works on both CUDA and MUSA platforms.
    """
    platform = detect_platform()

    # On MUSA platform, apply patches and port CUDA sources to MUSA
    if platform == Platform.MUSA:
        _apply_musa_patches()
        if cuda_sources:
            cuda_sources = [_port_cuda_source(src) for src in cuda_sources]

    from torch.utils.cpp_extension import load_inline as torch_load_inline

    return torch_load_inline(
        name=name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        with_pytorch_error_handling=with_pytorch_error_handling,
        keep_intermediates=keep_intermediates,
    )


# Export all public symbols
# Note: We only export CUDA_HOME, not MUSA_HOME. On MUSA platform, CUDA_HOME
# points to the MUSA installation so developers don't need to change their code.
__all__ = [
    "CUDA_HOME",
    "CUDAExtension",
    "CppExtension",
    "BuildExtension",
    "include_paths",
    "library_paths",
    "load",
    "load_inline",
]
