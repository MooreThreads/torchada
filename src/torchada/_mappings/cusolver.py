"""cusolver CUDA->MUSA porting rules."""

MAPPING = {
    'cusolver': 'musolver',
    'CUSOLVER': 'MUSOLVER',
    'cusolverDnHandle_t': 'musolverDnHandle_t',
    'cusolverDnCreate': 'musolverDnCreate',
    'cusolverDnDestroy': 'musolverDnDestroy',
}
