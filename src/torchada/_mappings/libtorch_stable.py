"""libtorch_stable CUDA->MUSA porting rules."""

MAPPING = {
    'aoti_torch_get_current_cuda_stream': 'aoti_torch_get_current_musa_stream',
    'torch_get_current_cuda_blas_handle': 'torch_get_current_musa_blas_handle',
    'STABLE_TORCH_LIBRARY_IMPL(_C, CUDA': 'STABLE_TORCH_LIBRARY_IMPL(_C, PrivateUse1',
}
