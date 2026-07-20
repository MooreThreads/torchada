"""aten CUDA->MUSA porting rules."""

MAPPING = {
    '#include <ATen/cuda/Atomic.cuh>': '#include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/Atomic.muh"',
    '#include <ATen/cuda/CUDAContext.h>': '#include "torch_musa/csrc/aten/musa/MUSAContext.h"',
    '#include <ATen/cuda/CUDAEvent.h>': '#include "torch_musa/csrc/core/MUSAEvent.h"',
    '#include <ATen/cuda/CUDADataType.h>': '#include "torch_musa/csrc/aten/musa/MUSADtype.muh"',
    '#include <ATen/cuda/CUDAGeneratorImpl.h>': '#include "torch_musa/csrc/aten/musa/CUDAGeneratorImpl.h"',
    '#include <ATen/cuda/CUDAUtils.h>': '#include "torch_musa/share/generated_cuda_compatible/include/ATen/musa/MUSAUtils.h"',
    '#include <ATen/cuda/detail/UnpackRaw.cuh>': '#include "torch_musa/csrc/aten/musa/UnpackRaw.muh"',
    '#include <ATen/cuda/Exceptions.h>': '#include "torch_musa/csrc/aten/musa/Exceptions.h"',
    '#include <ATen/cuda/CUDAGraphsUtils.cuh>': '#include <ATen/musa/MUSAGraphsUtils.muh>',
    'at::cuda': 'at::musa',
}
