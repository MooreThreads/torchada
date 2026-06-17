"""c10 CUDA->MUSA porting rules."""

MAPPING = {
    '#include <c10/cuda/CUDAException.h>': '#include "torch_musa/csrc/core/MUSAException.h"',
    '#include <c10/cuda/CUDAGuard.h>': '#include "torch_musa/csrc/core/MUSAGuard.h"',
    '#include <c10/cuda/CUDAStream.h>': '#include "torch_musa/csrc/core/MUSAStream.h"',
    'c10::cuda': 'c10::musa',
    'C10_CUDA_KERNEL_LAUNCH_CHECK': 'C10_MUSA_KERNEL_LAUNCH_CHECK',
    'C10_CUDA_CHECK': 'C10_MUSA_CHECK',
    'C10_CUDA_ERROR_HANDLED': 'C10_MUSA_ERROR_HANDLED',
    'C10_CUDA_IGNORE_ERROR': 'C10_MUSA_IGNORE_ERROR',
    'c10/cuda/CUDAException.h': 'c10/musa/MUSAException.h',
    'c10/cuda/CUDAStream.h': 'c10/musa/MUSAStream.h',
    'c10/cuda/CUDAGuard.h': 'c10/musa/MUSAGuard.h',
    'c10/cuda/CUDAFunctions.h': 'c10/musa/MUSAFunctions.h',
    'c10/cuda/CUDAMacros.h': 'c10/musa/MUSAMacros.h',
    'c10/cuda/CUDACachingAllocator.h': 'c10/musa/MUSACachingAllocator.h',
    '<c10/cuda/CUDAStream.h>': '"torch_musa/csrc/core/MUSAStream.h"',
    'c10/cuda': 'c10/musa',
}
