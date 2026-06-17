"""curand CUDA->MUSA porting rules."""

MAPPING = {
    'curand': 'murand',
    'CURAND': 'MURAND',
    'curandState': 'murandState',
    'curandStatePhilox4_32_10_t': 'murandStatePhilox4_32_10_t',
    'curand_init': 'murand_init',
    'curand_uniform': 'murand_uniform',
    'curand_uniform4': 'murand_uniform4',
    'curand_normal': 'murand_normal',
    'curand_normal4': 'murand_normal4',
}
