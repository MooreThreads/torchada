// torch_musa 2.9's torch::stable predates torch 2.10's split of the stable
// library macros into torch/csrc/stable/macros.h; they still live in
// library.h. Forward so newer kernels that include macros.h directly compile.
//
// torch_musa 2.11+ ships the real macros.h, and from torch 2.13 its stable
// headers include it from inside library.h (stableivalue_conversions.h,
// tensor_struct.h, ...) for STABLE_TORCH_ERROR_CODE_CHECK, which 2.11 has no
// concept of. When this directory precedes torch_musa's in the search path,
// forwarding to library.h re-enters a library.h that #pragma once skips, so the
// real macros never arrive and 2.13 fails with that macro undeclared.
// Defer to the next macros.h in the search path whenever one exists, so the
// position of this directory does not matter, and fall back to library.h only
// when there is none.
//
// No #pragma once: this file defines nothing itself, and each inclusion must
// keep walking the search path. Both targets carry their own #pragma once.
// With it, GCC would take a second, identical copy of this directory further
// down the path as already included and skip the library.h fallback.
#if defined(__has_include_next)
#if __has_include_next(<torch/csrc/stable/macros.h>)
#include_next <torch/csrc/stable/macros.h>
#else
#include <torch/csrc/stable/library.h>
#endif
#else
#include <torch/csrc/stable/library.h>
#endif
