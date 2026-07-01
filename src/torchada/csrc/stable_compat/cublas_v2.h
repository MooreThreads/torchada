#pragma once
// MUSA: <cublas_v2.h> does not exist; map to the MUSA BLAS header. Resolved via
// the stable_compat include dir for libtorch-stable kernels (torch_utils.h, the
// allspark w8a16 gemm) that include it directly.
#include <mublas.h>
