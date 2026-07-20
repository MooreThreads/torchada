"""Per-domain CUDA->MUSA mapping modules, merged into MAPPING_RULE."""

from . import aten as _aten
from . import c10 as _c10
from . import cublas as _cublas
from . import curand as _curand
from . import cudnn as _cudnn
from . import nccl as _nccl
from . import nvjpeg as _nvjpeg
from . import cusparse as _cusparse
from . import cusolver as _cusolver
from . import cufft as _cufft
from . import cutlass as _cutlass
from . import thrust as _thrust
from . import data_types as _data_types
from . import cuda_driver as _cuda_driver
from . import libtorch_stable as _libtorch_stable
from . import flashinfer as _flashinfer
from . import cuda_arch as _cuda_arch
from . import device as _device
from . import compiler as _compiler
from . import cuda_runtime as _cuda_runtime

# Order is irrelevant: SimplePorting sorts rules by key length; domains are disjoint.
MAPPING_RULE = {
    **_aten.MAPPING,
    **_c10.MAPPING,
    **_cublas.MAPPING,
    **_curand.MAPPING,
    **_cudnn.MAPPING,
    **_nccl.MAPPING,
    **_nvjpeg.MAPPING,
    **_cusparse.MAPPING,
    **_cusolver.MAPPING,
    **_cufft.MAPPING,
    **_cutlass.MAPPING,
    **_thrust.MAPPING,
    **_data_types.MAPPING,
    **_cuda_driver.MAPPING,
    **_libtorch_stable.MAPPING,
    **_flashinfer.MAPPING,
    **_cuda_arch.MAPPING,
    **_device.MAPPING,
    **_compiler.MAPPING,
    **_cuda_runtime.MAPPING,
}
