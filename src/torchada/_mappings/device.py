"""device CUDA->MUSA porting rules."""

MAPPING = {
    'torch::cuda': 'torch::musa',
    'torch.cuda': 'torch.musa',
    'at::kCUDA': 'at::kPrivateUse1',
    'at::DeviceType::CUDA': 'at::DeviceType::PrivateUse1',
    'c10::DeviceType::CUDA': 'c10::DeviceType::PrivateUse1',
    'getCurrentCUDAStream': 'getCurrentMUSAStream',
    'getDefaultCUDAStream': 'getDefaultMUSAStream',
    'CUDAStream': 'MUSAStream',
    'CUDAGuard': 'MUSAGuard',
    'OptionalCUDAGuard': 'OptionalMUSAGuard',
    'CUDAStreamGuard': 'MUSAStreamGuard',
    'CUDAEvent': 'MUSAEvent',
    'torch::cuda::getCurrentCUDAStream': 'torch::musa::getCurrentMUSAStream',
    'torch::cuda::getDefaultCUDAStream': 'torch::musa::getDefaultMUSAStream',
    'torch::cuda::getStreamFromPool': 'torch::musa::getStreamFromPool',
    'torch::kCUDA': 'torch::kPrivateUse1',
    'cudaDeviceIndex': 'musaDeviceIndex',
    'CUDADeviceIndex': 'MUSADeviceIndex',
}
