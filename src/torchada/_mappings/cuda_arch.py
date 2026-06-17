"""cuda_arch CUDA->MUSA porting rules."""

MAPPING = {
    'cuda/std': 'musa/std',
    '<cuda/functional>': '<musa/functional>',
    '<cuda/std/': '<musa/std/',
    '#include <cuda/': '#include <musa/',
    '__CUDA_ARCH__ >= 800': '__MUSA_ARCH__ >= 220',
    '(__CUDA_ARCH__ < 800)': '(__MUSA_ARCH__ < 220)',
    '(__CUDA_ARCH__ >= 900)': '(__MUSA_ARCH__ >= 310)',
    'cuda::std::numeric_limits': 'musa::std::numeric_limits',
    '#include <cuda/std/functional>': '#include <musa/std/functional>',
    '#include <cuda/std/limits>': '#include <musa/std/limits>',
}
