#pragma once
// Deprecated forwarder: this header now lives in ../cuda_compat/cublas_v2.h.
//
// The CUDA-named headers are a layer of their own: a JIT that brings its own
// compiler driver resolves them instead of absent CUDA headers, so that
// directory has to be FIRST on the include path. This directory keeps working
// in either position, because its stand-ins defer to a native header through
// `#include_next`. The two answer to different version drivers and do not share
// that rule, which is why they are not one directory.
//
// Kept so a build that adds only this directory keeps resolving the header. It
// carries no aliases itself, so the two copies cannot drift. Scheduled for
// removal with the next major release.
#include "../cuda_compat/cublas_v2.h"
