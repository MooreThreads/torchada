"""cuda_arch CUDA->MUSA porting rules."""

MAPPING = {
    'cuda/std': 'musa/std',
    '<cuda/functional>': '<musa/functional>',
    '<cuda/std/': '<musa/std/',
    '#include <cuda/': '#include <musa/',
    # An arch comparison must be rewritten together with its threshold: torch_musa's
    # general.json renames the bare `__CUDA_ARCH__` to `__MUSA_ARCH__` but leaves the
    # NVIDIA number, and the two scales are unrelated (sm_80 -> mp_22, sm_90 -> mp_31).
    # A surviving `__MUSA_ARCH__ < 800` is true on every current MUSA arch, which
    # silently flips such a guard to its unsupported-hardware branch. Rules are applied
    # longest-key-first, so these win over that bare rename. The comparison is matched
    # without surrounding parentheses so that both `(__CUDA_ARCH__ < 800)` and a bare
    # `__CUDA_ARCH__ < 800` are covered by one rule.
    '__CUDA_ARCH__ >= 800': '__MUSA_ARCH__ >= 220',
    '__CUDA_ARCH__ < 800': '__MUSA_ARCH__ < 220',
    '__CUDA_ARCH__ >= 900': '__MUSA_ARCH__ >= 310',
    '__CUDA_ARCH__ < 900': '__MUSA_ARCH__ < 310',
    'cuda::std::numeric_limits': 'musa::std::numeric_limits',
    '#include <cuda/std/functional>': '#include <musa/std/functional>',
    '#include <cuda/std/limits>': '#include <musa/std/limits>',
}
