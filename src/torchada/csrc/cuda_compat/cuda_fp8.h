#pragma once
// MUSA: the CUDA-named FP8 surface (`<cuda_fp8.h>`), for sources compiled by mcc
// without a porting pass. See `cuda_runtime.h` in this directory for why this
// layer exists and why it must be the FIRST include path.
//
// MUSA's header is <musa_fp8.h>; its storage structs are `__mt_fp8_*`, its
// conversion entry points `__musa_cvt_*`, and its saturate/interpretation enum
// values `__MT_*`. The aliases below mirror the identifier entries of
// `torchada/_mappings/data_types.py`, which `tests/test_mappings.py` keeps in
// sync with this file.
//
// Everything is a macro, types included, so the header says exactly what the
// porting table says and each name is guarded by itself. A macro keeps type
// identity for overloads and template arguments - the preprocessor replaces the
// spelling, the compiler still sees one type.
#include <musa_fp8.h>

// Storage and vector types.
#ifndef __nv_fp8_e4m3
#define __nv_fp8_e4m3 __mt_fp8_e4m3
#endif
#ifndef __nv_fp8_e5m2
#define __nv_fp8_e5m2 __mt_fp8_e5m2
#endif
#ifndef __nv_fp8_interpretation_t
#define __nv_fp8_interpretation_t __mt_fp8_interpretation_t
#endif
#ifndef __nv_fp8_storage_t
#define __nv_fp8_storage_t __mt_fp8_storage_t
#endif
#ifndef __nv_fp8x2_e4m3
#define __nv_fp8x2_e4m3 __mt_fp8x2_e4m3
#endif
#ifndef __nv_fp8x2_e5m2
#define __nv_fp8x2_e5m2 __mt_fp8x2_e5m2
#endif
#ifndef __nv_fp8x2_storage_t
#define __nv_fp8x2_storage_t __mt_fp8x2_storage_t
#endif
#ifndef __nv_fp8x4_e4m3
#define __nv_fp8x4_e4m3 __mt_fp8x4_e4m3
#endif
#ifndef __nv_fp8x4_e5m2
#define __nv_fp8x4_e5m2 __mt_fp8x4_e5m2
#endif
#ifndef __nv_fp8x4_storage_t
#define __nv_fp8x4_storage_t __mt_fp8x4_storage_t
#endif

// Enum values. `__NV_SATFINITE` matches `__MT_SATFINITE` by name only; MUSA gates
// its native conversion on `__MUSA_ARCH__`, so the value is the compiler's to
// interpret, not ours to translate.
#ifndef __NV_E4M3
#define __NV_E4M3 __MT_E4M3
#endif
#ifndef __NV_E5M2
#define __NV_E5M2 __MT_E5M2
#endif
#ifndef __NV_SATFINITE
#define __NV_SATFINITE __MT_SATFINITE
#endif
#ifndef CUDA_R_8F_E4M3
#define CUDA_R_8F_E4M3 MUSA_R_8F_E4M3
#endif
#ifndef CUDA_R_8F_E5M2
#define CUDA_R_8F_E5M2 MUSA_R_8F_E5M2
#endif

// Conversion entry points.
#ifndef __nv_cvt_float_to_fp8
#define __nv_cvt_float_to_fp8 __musa_cvt_float_to_fp8
#endif
#ifndef __nv_cvt_float2_to_fp8x2
#define __nv_cvt_float2_to_fp8x2 __musa_cvt_float2_to_fp8x2
#endif
#ifndef __nv_cvt_fp8_to_halfraw
#define __nv_cvt_fp8_to_halfraw __musa_cvt_fp8_to_halfraw
#endif
#ifndef __nv_cvt_fp8x2_to_halfraw2
#define __nv_cvt_fp8x2_to_halfraw2 __musa_cvt_fp8x2_to_halfraw2
#endif
#ifndef __nv_cvt_bfloat16raw_to_fp8
#define __nv_cvt_bfloat16raw_to_fp8 __musa_cvt_bfloat16raw_to_fp8
#endif
