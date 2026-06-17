"""cusparse CUDA->MUSA porting rules."""

MAPPING = {
    'cusparse': 'musparse',
    'CUSPARSE': 'MUSPARSE',
    'cusparseHandle_t': 'musparseHandle_t',
    'cusparseCreate': 'musparseCreate',
    'cusparseDestroy': 'musparseDestroy',
}
