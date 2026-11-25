#pragma once
// Deprecated forwarder: this header now lives in ../cuda_compat/cuda_fp8.h.
//
// The CUDA-named headers are a layer of their own: a JIT that brings its own
// compiler driver resolves them instead of absent CUDA headers, so that
// directory has to precede CUDA toolkit include directories. The CUDA forwarders
// here use a relative include rather than `#include_next`, so this directory must
// also precede CUDA toolkit include directories for these CUDA names. Torch stable
// ABI shims elsewhere in this tree may still defer through `#include_next`.
//
// Kept so a build that adds only this directory keeps resolving the header. It
// carries no aliases itself, so the two copies cannot drift. Scheduled for
// removal with the next major release.
#include "../cuda_compat/cuda_fp8.h"
