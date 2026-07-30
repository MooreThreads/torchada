#pragma once
// MUSA: <cublas_v2.h> does not exist; map to the MUSA BLAS header. Resolved via
// the stable_compat include dir for libtorch-stable kernels (torch_utils.h, the
// allspark w8a16 gemm) that include it directly.
#include <mublas.h>

// MUSA: libtorch-stable kernels use the cuBLAS handle spelling; map it to the
// muBLAS handle (defined by <mublas.h> above, and repeated harmlessly by the
// stable-box header) so those kernels compile against the upstream name.
#ifndef cublasHandle_t
#define cublasHandle_t mublasHandle_t
#endif
