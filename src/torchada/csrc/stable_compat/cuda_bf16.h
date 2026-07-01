#pragma once
// MUSA: there is no <cuda_bf16.h> on MUSA; libtorch-stable kernels that include
// it (e.g. attention/dtype_bfloat16.cuh) get the MUSA bf16 header instead.
// Resolved ahead of any toolchain header via the stable_compat include dir.
#include <musa_bf16.h>

// MUSA's bf16 header names its types __mt_bfloat16{,2}; kernels written for CUDA
// use __nv_bfloat16{,2}. Alias them so CUDA-named bf16 code compiles. Guarded so
// a translation unit that already defines these (e.g. via type_convert.cuh)
// is not redefined.
#ifndef TORCHADA_NV_BFLOAT16_ALIASED
#define TORCHADA_NV_BFLOAT16_ALIASED 1
using __nv_bfloat16 = __mt_bfloat16;
using __nv_bfloat162 = __mt_bfloat162;
#endif
