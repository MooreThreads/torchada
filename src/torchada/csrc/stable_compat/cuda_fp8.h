#pragma once
// MUSA: there is no <cuda_fp8.h> on MUSA; libtorch-stable kernels that include
// it (e.g. attention/dtype_fp8.cuh under ENABLE_FP8) get the MUSA fp8 header
// instead. Resolved ahead of any toolchain header via the stable_compat include
// dir.
#include <musa_fp8.h>

#ifndef TORCHADA_NV_FP8_ALIASED
#define TORCHADA_NV_FP8_ALIASED 1
using __nv_fp8_e4m3 = __mt_fp8_e4m3;
using __nv_fp8_e5m2 = __mt_fp8_e5m2;
using __nv_fp8x2_e4m3 = __mt_fp8x2_e4m3;
using __nv_fp8x2_e5m2 = __mt_fp8x2_e5m2;
using __nv_fp8x4_e4m3 = __mt_fp8x4_e4m3;
using __nv_fp8x4_e5m2 = __mt_fp8x4_e5m2;
#endif
