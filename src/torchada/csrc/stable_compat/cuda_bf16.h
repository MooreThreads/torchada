#pragma once
// MUSA: there is no <cuda_bf16.h> on MUSA; libtorch-stable kernels that include
// it (e.g. attention/dtype_bfloat16.cuh) get the MUSA bf16 header instead.
// Resolved ahead of any toolchain header via the stable_compat include dir.
#include <musa_bf16.h>
