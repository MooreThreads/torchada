#pragma once
// MUSA: the CUDA-named bfloat16 surface (`<cuda_bf16.h>`), for sources compiled
// by mcc without a porting pass. See `cuda_runtime.h` in this directory for why
// this layer exists and why it must precede CUDA toolkit include directories.
//
// MUSA's header is <musa_bf16.h>; it names its types `__mt_bfloat16{,2}`, while
// CUDA-written kernels use `__nv_bfloat16{,2}` and `nv_bfloat16{,2}`. The
// aliases below mirror the identifier entries of
// `torchada/_mappings/data_types.py`, which `tests/test_mappings.py` keeps in
// sync with this file.
//
// Macros rather than `using` declarations, for the same reason as `cuda_fp8.h`:
// the preprocessor replaces the spelling and the compiler still sees one type, so
// overloads and `is_same` are unaffected, while each name stays guarded by
// itself. A translation unit that defines TORCHADA_NV_BFLOAT16_ALIASED before
// including this header opts out entirely (some kernels bring their own aliases
// in through `type_convert.cuh`).
#include <musa_bf16.h>

#ifndef TORCHADA_NV_BFLOAT16_ALIASED
#define TORCHADA_NV_BFLOAT16_ALIASED 1

#ifndef __nv_bfloat16
#define __nv_bfloat16 __mt_bfloat16
#endif
#ifndef __nv_bfloat16_raw
#define __nv_bfloat16_raw __mt_bfloat16_raw
#endif
#ifndef __nv_bfloat162
#define __nv_bfloat162 __mt_bfloat162
#endif
#ifndef __nv_bfloat162_raw
#define __nv_bfloat162_raw __mt_bfloat162_raw
#endif
#ifndef nv_bfloat16
#define nv_bfloat16 __mt_bfloat16
#endif
#ifndef nv_bfloat162
#define nv_bfloat162 __mt_bfloat162
#endif

#endif  // TORCHADA_NV_BFLOAT16_ALIASED
