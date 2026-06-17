"""cufft CUDA->MUSA porting rules."""

MAPPING = {
    'cufft': 'mufft',
    'CUFFT': 'MUFFT',
    'cufftHandle': 'mufftHandle',
    'cufftPlan1d': 'mufftPlan1d',
    'cufftPlan2d': 'mufftPlan2d',
    'cufftPlan3d': 'mufftPlan3d',
    'cufftExecC2C': 'mufftExecC2C',
    'cufftExecR2C': 'mufftExecR2C',
    'cufftExecC2R': 'mufftExecC2R',
}
