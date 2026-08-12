"""libtorch_stable CUDA->MUSA porting rules."""

MAPPING = {
    'aoti_torch_get_current_cuda_stream': 'aoti_torch_get_current_musa_stream',
    'torch_get_current_cuda_blas_handle': 'torch_get_current_musa_blas_handle',
    'torch_set_current_cuda_stream': 'torch_set_current_musa_stream',
    'torch_get_cuda_stream_from_pool': 'torch_get_musa_stream_from_pool',
    'torch_cuda_stream_synchronize': 'torch_musa_stream_synchronize',
}
