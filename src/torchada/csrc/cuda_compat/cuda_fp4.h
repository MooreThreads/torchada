#pragma once
// MUSA: CUDA-named FP4 types for sources compiled by mcc without a porting pass.
#include <musa_fp4.h>

#ifndef __NV_E2M1
#define __NV_E2M1 __MT_E2M1
#endif
#ifndef __nv_fp4_storage_t
#define __nv_fp4_storage_t __mt_fp4_storage_t
#endif
#ifndef __nv_fp4x2_storage_t
#define __nv_fp4x2_storage_t __mt_fp4x2_storage_t
#endif
#ifndef __nv_fp4x4_storage_t
#define __nv_fp4x4_storage_t __mt_fp4x4_storage_t
#endif
#ifndef __nv_fp4_interpretation_t
#define __nv_fp4_interpretation_t __mt_fp4_interpretation_t
#endif
#ifndef __nv_fp4_e2m1
#define __nv_fp4_e2m1 __mt_fp4_e2m1
#endif
#ifndef __nv_fp4x2_e2m1
#define __nv_fp4x2_e2m1 __mt_fp4x2_e2m1
#endif
#ifndef __nv_fp4x4_e2m1
#define __nv_fp4x4_e2m1 __mt_fp4x4_e2m1
#endif
#ifndef __nv_cvt_float2_to_fp4x2
#define __nv_cvt_float2_to_fp4x2 __musa_cvt_float2_to_fp4x2
#endif
#ifndef __nv_cvt_fp4x2_to_halfraw2
#define __nv_cvt_fp4x2_to_halfraw2 __musa_cvt_fp4x2_to_halfraw2
#endif
