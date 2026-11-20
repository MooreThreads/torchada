#pragma once
// MUSA: the CUDA-named half surface (`<cuda_fp16.h>`), for sources compiled by
// mcc without a porting pass. See `cuda_runtime.h` in this directory for why this
// layer exists and why it must be the FIRST include path.
//
// MUSA's header is <musa_fp16.h> and already provides `__half`; only the two CUDA
// spellings are missing. Mirrors the identifier entries of
// `torchada/_mappings/data_types.py`, which `tests/test_mappings.py` keeps in
// sync with this file.
#include <musa_fp16.h>

#ifndef __nv_half
#define __nv_half __half
#endif
#ifndef nv_half
#define nv_half __half
#endif
