#pragma once
// MUSA: there is no <cuda_fp8.h> on MUSA; libtorch-stable kernels that include
// it (e.g. attention/dtype_fp8.cuh under ENABLE_FP8) get the MUSA fp8 header
// instead. Resolved ahead of any toolchain header via the stable_compat include
// dir.
#include <musa_fp8.h>
