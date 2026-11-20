#pragma once
// MUSA: the CUDA-named runtime API header, split out from `<cuda_runtime.h>` on
// CUDA but a single header here. See `cuda_runtime.h` in this directory for why
// this layer exists and why it must be the FIRST include path.
//
// The porting table maps this name to `<musa_runtime_api.h>`, which is how the
// text porter rewrites an include line. This header includes
// `<musa_runtime.h>` instead: it is a superset of the API surface and is known to
// exist, where a separate `musa_runtime_api.h` is not part of the documented MUSA
// header set.
#include "cuda_runtime.h"
