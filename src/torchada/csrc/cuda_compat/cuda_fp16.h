#pragma once
// MUSA: the CUDA-named half surface (`<cuda_fp16.h>`), for sources compiled by
// mcc without a porting pass. See `cuda_runtime.h` in this directory for why this
// layer exists and why it must precede CUDA toolkit include directories.
//
// MUSA's header is <musa_fp16.h> and already provides `__half`. The aliases
// below are the CUDA spellings currently required by the audited direct-JIT
// sources. They mirror `torchada/_mappings/data_types.py`, which
// `tests/test_mappings.py` keeps in sync with this file.
#include <musa_fp16.h>

#ifndef __nv_half
#define __nv_half __half
#endif
#ifndef nv_half
#define nv_half __half
#endif
