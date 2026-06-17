"""cudnn CUDA->MUSA porting rules."""

MAPPING = {
    'cudnn': 'mudnn',
    'CUDNN': 'MUDNN',
    'cudnnHandle_t': 'mudnnHandle_t',
    'cudnnCreate': 'mudnnCreate',
    'cudnnDestroy': 'mudnnDestroy',
    'cudnnStatus_t': 'mudnnStatus_t',
    'cudnnSetStream': 'mudnnSetStream',
    'cudnnGetStream': 'mudnnGetStream',
    'cudnnTensorDescriptor_t': 'mudnnTensorDescriptor_t',
    'cudnnFilterDescriptor_t': 'mudnnFilterDescriptor_t',
    'cudnnConvolutionDescriptor_t': 'mudnnConvolutionDescriptor_t',
    'cudnnPoolingDescriptor_t': 'mudnnPoolingDescriptor_t',
    'cudnnActivationDescriptor_t': 'mudnnActivationDescriptor_t',
    'cudnnDropoutDescriptor_t': 'mudnnDropoutDescriptor_t',
    'cudnnRNNDescriptor_t': 'mudnnRNNDescriptor_t',
    'cudnnCreateTensorDescriptor': 'mudnnCreateTensorDescriptor',
    'cudnnDestroyTensorDescriptor': 'mudnnDestroyTensorDescriptor',
    'cudnnSetTensor4dDescriptor': 'mudnnSetTensor4dDescriptor',
    'cudnnSetTensorNdDescriptor': 'mudnnSetTensorNdDescriptor',
}
