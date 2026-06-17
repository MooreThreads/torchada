"""compiler CUDA->MUSA porting rules."""

MAPPING = {
    'CUDA_KERNEL_LOOP': 'MUSA_KERNEL_LOOP',
    'CUDA_1D_KERNEL_LOOP': 'MUSA_1D_KERNEL_LOOP',
    'CUDA_2D_KERNEL_LOOP': 'MUSA_2D_KERNEL_LOOP',
    'CUDA_NUM_THREADS': 'MUSA_NUM_THREADS',
    '#include <THC/THCAtomics.cuh>': '#include <THC/THCAtomics.muh>',
    'asm volatile': 'if(0) asm volatile',
    'const void* __restrict__ ptrs[8]': 'const void* ptrs[8]',
}
