#pragma once
// MUSA: there is no <cuda_runtime.h> on MUSA; libtorch-stable kernels that
// include it (e.g. torch_utils.h) get the MUSA runtime instead. Resolved ahead
// of any toolchain header via the stable_compat include dir.
#include <musa_runtime.h>
