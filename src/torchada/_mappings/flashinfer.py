"""flashinfer CUDA->MUSA porting rules."""

MAPPING = {
    '.FlagHeads<VEC_SIZE>': '.template FlagHeads<VEC_SIZE>',
    '.InclusiveSum<VEC_SIZE>': '.template InclusiveSum<VEC_SIZE>',
    '.Reduce<VEC_SIZE>': '.template Reduce<VEC_SIZE>',
    '.Sum<VEC_SIZE>': '.template Sum<VEC_SIZE>',
    '::cast<vec_size>': '::template cast<vec_size>',
    'SCHEDULER::execute': 'SCHEDULER::template execute',
    '.is_cuda()': '.is_privateuseone()',
    '->philox_cuda_state': '->philox_musa_state',
    'compute_capacity.first >= 8': 'compute_capacity.first >= 3',
    '#include "math.cuh"': '\n// MUSA fast math intrinsics (replacing flashinfer::math functions)\n__device__ __forceinline__ float fast_rsqrtf(float x) { return __frsqrt_rn(x); }\n__device__ __forceinline__ float fast_rcp(float x) { return __frcp_rn(x); }\n',
    'math::shfl_xor_sync(sum_sq, offset);': '__shfl_xor_sync(0xffffffff, sum_sq, offset);',
    'math::rsqrt(smem[0] / float(d) + eps);': 'fast_rsqrtf(smem[0] / float(d) + eps);',
    'math::ptx_rcp(max(sum_low, 1e-8));': 'fast_rcp(max(sum_low, 1e-8));',
    'math::ptx_rcp(denom);': 'fast_rcp(denom);',
    'math::ptx_log2': 'log2f',
    'math::ptx_exp2': 'exp2f',
}
