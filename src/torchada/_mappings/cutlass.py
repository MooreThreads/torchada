"""cutlass CUDA->MUSA porting rules."""

MAPPING = {
    '#include "cutlass/array.h"': '#include <mutlass/array.h>',
    '#include <cutlass/array.h>': '#include <mutlass/array.h>',
    '#include <cutlass/cutlass.h>': '#include <mutlass/mutlass.h>',
    '#include <cutlass/numeric_types.h>': '#include <mutlass/numeric_types.h>',
    'cutlass::AlignedArray': 'mutlass::AlignedArray',
    'cutlass::bfloat16_t': 'mutlass::bfloat16_t',
    'cutlass::half_t': 'mutlass::half_t',
    'cutlass': 'mutlass',
    'CUTLASS': 'MUTLASS',
    'cutlass/': 'mutlass/',
    'cutlass::': 'mutlass::',
}
