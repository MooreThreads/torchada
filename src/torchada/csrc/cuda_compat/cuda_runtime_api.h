#pragma once
// MUSA: the CUDA-named runtime API header, split out from `<cuda_runtime.h>` on
// CUDA but a single header here. See `cuda_runtime.h` in this directory for why
// this layer exists and why it must precede CUDA toolkit include directories.
//
// The porting table maps this name to `<musa_runtime_api.h>`, which is how the
// text porter rewrites an include line. This header includes `<musa_runtime.h>`
// instead: that header is the superset used by the runtime surface below.
#include "cuda_runtime.h"
