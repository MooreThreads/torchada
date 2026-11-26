# Copyright (c) torchada contributors
# SPDX-License-Identifier: MIT
"""The CUDA-named headers a compile sees when nothing ported the source first.

torchada translates CUDA names for mcc in two places. The porting tables in
``_mappings/`` rewrite source text at build time, and this layer lets the
preprocessor do it for a build that never reaches the porting path at all: a JIT
that carries its own compiler driver, such as tvm-ffi's ``tvm_ffi.cpp.load_inline``,
which recent SGLang releases use for their kernels. Those call neither ``load()``
nor ``load_inline`` from :mod:`torchada.utils.cpp_extension`, so no source is
rewritten and ``#include <cuda_runtime.h>`` has nothing to resolve against --
``/usr/local/musa/include`` ships ``musa_*`` headers only.

Three things follow, and they are the reason this is a directory of its own rather
than more headers beside the stable-ABI shims:

* **It must precede CUDA toolkit include directories.** A ``CUDA_HOME`` left on a
  build box ships a real ``cuda_runtime.h``; resolving to it compiles against the
  wrong runtime. The CUDA forwarders in ``stable_compat`` use a relative include,
  so they share this ordering requirement.
* **It never uses ``#include_next``.** There is nothing to defer to: the MUSA
  spelling lives in ``<musa_runtime.h>``, which these headers include directly, and a
  CUDA ``<cuda_runtime.h>`` later in the search path is the thing they exist to
  shadow.
* **It is on demand only.** Nothing here is force-included, unlike the stable ABI's
  box header, so a host-only translation unit never sees a device runtime header.

The aliases are not hand-maintained: :data:`DOMAIN_HEADERS` names which porting-table
domain each header mirrors, and ``TestCudaCompatHeaderParity`` in
``tests/test_mappings.py`` fails when a rule and its alias disagree. This module
imports only the standard library (``import torchada.cuda_compat`` still runs
``import torchada`` first), so nothing here needs a device, a compiler or torch.

Usage:
    from torchada.cuda_compat import cuda_compat_include_dir

    flags = ["-I" + cuda_compat_include_dir()] + mcc_flags
"""

import os

__all__ = [
    "COVERED_DOMAINS",
    "DOMAIN_HEADERS",
    "HEADER_CANDIDATE_DOMAINS",
    "SUBWORD_RULES",
    "UNMAPPED_JIT_NAMES",
    "cuda_compat_include_dir",
]

_HERE = os.path.dirname(os.path.abspath(__file__))

#: Porting-table domain -> the headers in ``csrc/cuda_compat`` that mirror it. The
#: header list is exhaustive: every identifier rule in a listed domain must be
#: aliased by one of them, so a rule added to a table without its alias fails a test
#: instead of failing later as an undeclared identifier in someone else's build.
DOMAIN_HEADERS = {
    "cuda_driver": ("cuda.h",),
    "cuda_runtime": (
        "cuda_runtime.h",
        "cuda_runtime_api.h",
        "cudaTypedefs.h",
        "cuda_profiler_api.h",
    ),
    "data_types": ("cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h", "cuda_fp4.h"),
    "cublas": ("cublas_v2.h",),
}

COVERED_DOMAINS = tuple(sorted(DOMAIN_HEADERS))

#: Domains whose rules are identifiers with a MUSA target, so a header could mirror
#: them, but none exists yet. Declared rather than omitted: the next header to add is
#: a data change here plus the file, not a re-survey of ``_mappings/``.
HEADER_CANDIDATE_DOMAINS = (
    "cudnn",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nccl",
)

#: Rules that are sub-word rewrites for the text porter, never aliases. ``cublas``
#: exists so ``CUBLAS_OP_N`` and friends port; a macro of that name would rewrite
#: unrelated identifiers, so the header aliases the constant instead.
SUBWORD_RULES = {"cublas": frozenset({"cublas", "CUBLAS"})}

#: CUDA names SGLang's ``python/sglang/kernels/jit`` tree still uses after the
#: header layer and which the current MUSA toolkit does not provide. Tensor-map
#: declarations are commented out in the MUSA 5.3 headers, so aliasing them would
#: turn an architecture rejection into a misleading compiler error or link error.
#: These are the input to ``scripts/probe_cuda_compat_symbols.py``; a name that
#: gains a real MUSA declaration must leave this list.
UNMAPPED_JIT_NAMES = (
    "CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B",
    "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
    "CU_TENSOR_MAP_DATA_TYPE_INT32",
    "CU_TENSOR_MAP_DATA_TYPE_UINT64",
    "CU_TENSOR_MAP_DATA_TYPE_UINT8",
    "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE",
    "CU_TENSOR_MAP_INTERLEAVE_NONE",
    "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
    "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
    "CU_TENSOR_MAP_L2_PROMOTION_NONE",
    "CU_TENSOR_MAP_SWIZZLE_128B",
    "CU_TENSOR_MAP_SWIZZLE_32B",
    "CU_TENSOR_MAP_SWIZZLE_64B",
    "CU_TENSOR_MAP_SWIZZLE_NONE",
    "CUtensorMap",
    "CUtensorMapDataType",
    "CUtensorMapFloatOOBfill",
    "CUtensorMapInterleave",
    "CUtensorMapL2promotion",
    "CUtensorMapSwizzle",
    "cuTensorMapEncodeTiled",
)


def cuda_compat_include_dir() -> str:
    """Directory of the CUDA-named compatibility headers. Place it **first**.

    Ahead of every other include path, including the toolkit's own: the headers here
    must win over a ``cuda_runtime.h`` that a stray ``CUDA_HOME`` would provide, or
    the compile binds to the CUDA runtime instead of the MUSA one.
    """
    return os.path.join(_HERE, "csrc", "cuda_compat")
