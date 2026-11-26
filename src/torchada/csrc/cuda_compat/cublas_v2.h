#pragma once
// MUSA: the CUDA-named cuBLAS surface (`<cublas_v2.h>`), for sources compiled by
// mcc without a porting pass. See `cuda_runtime.h` in this directory for why this
// layer exists and why it must precede CUDA toolkit include directories.
//
// MUSA ships muBLAS as <mublas.h> with `mublas*` names; this header adds the
// CUDA spellings. It mirrors the identifier entries of
// `torchada/_mappings/cublas.py`, which `tests/test_mappings.py` keeps in sync
// with this file.
//
// The bare `cublas` / `CUBLAS` sub-word rules in the porting table are
// deliberately not aliased: they exist so the text porter can rewrite tokens such
// as `CUBLAS_OP_N`, and a macro of that name would rewrite unrelated identifiers.
//
// `cublasLt` entry points resolve as mublasLt names: including <mublas.h> only,
// the Lt constants below are usable once a translation unit that needs them also
// includes the Lt header, exactly as on CUDA.
#include <mublas.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#ifndef cublasHandle_t
#define cublasHandle_t mublasHandle_t
#endif
#ifndef cublasStatus_t
#define cublasStatus_t mublasStatus_t
#endif
#ifndef cublasComputeType_t
#define cublasComputeType_t mublasComputeType_t
#endif
#ifndef cublasLtHandle_t
#define cublasLtHandle_t mublasLtHandle_t
#endif
#ifndef cublasLtMatmulDescAttributes_t
#define cublasLtMatmulDescAttributes_t mublasLtMatmulDescAttributes_t
#endif
#ifndef cublasLtMatmulDescOpaque_t
#define cublasLtMatmulDescOpaque_t mublasLtMatmulDescOpaque_t
#endif
#ifndef cublasLtMatmulDesc_t
#define cublasLtMatmulDesc_t mublasLtMatmulDesc_t
#endif
#ifndef cublasLtMatmulHeuristicResult_t
#define cublasLtMatmulHeuristicResult_t mublasLtMatmulHeuristicResult_t
#endif
#ifndef cublasLtMatmulPreferenceAttributes_t
#define cublasLtMatmulPreferenceAttributes_t mublasLtMatmulPreferenceAttributes_t
#endif
#ifndef cublasLtMatmulPreferenceOpaque_t
#define cublasLtMatmulPreferenceOpaque_t mublasLtMatmulPreferenceOpaque_t
#endif
#ifndef cublasLtMatmulPreference_t
#define cublasLtMatmulPreference_t mublasLtMatmulPreference_t
#endif
#ifndef cublasLtMatrixLayoutAttribute_t
#define cublasLtMatrixLayoutAttribute_t mublasLtMatrixLayoutAttribute_t
#endif
#ifndef cublasLtMatrixLayoutOpaque_t
#define cublasLtMatrixLayoutOpaque_t mublasLtMatrixLayoutOpaque_t
#endif
#ifndef cublasLtMatrixLayout_t
#define cublasLtMatrixLayout_t mublasLtMatrixLayout_t
#endif

// ---------------------------------------------------------------------------
// Entry points and constants
// ---------------------------------------------------------------------------
#ifndef cublasCreate
#define cublasCreate mublasCreate
#endif
#ifndef cublasDestroy
#define cublasDestroy mublasDestroy
#endif
#ifndef cublasSetStream
#define cublasSetStream mublasSetStream
#endif
#ifndef cublasGetStream
#define cublasGetStream mublasGetStream
#endif
#ifndef cublasSgemm
#define cublasSgemm mublasSgemm
#endif
#ifndef cublasDgemm
#define cublasDgemm mublasDgemm
#endif
#ifndef cublasHgemm
#define cublasHgemm mublasHgemm
#endif
#ifndef cublasGemmEx
#define cublasGemmEx mublasGemmEx
#endif
#ifndef cublasGemmBatchedEx
#define cublasGemmBatchedEx mublasGemmBatchedEx
#endif
#ifndef cublasGemmStridedBatchedEx
#define cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#endif
#ifndef cublasSgemmBatched
#define cublasSgemmBatched mublasSgemmBatched
#endif
#ifndef cublasDgemmBatched
#define cublasDgemmBatched mublasDgemmBatched
#endif
#ifndef cublasSgemmStridedBatched
#define cublasSgemmStridedBatched mublasSgemmStridedBatched
#endif
#ifndef cublasDgemmStridedBatched
#define cublasDgemmStridedBatched mublasDgemmStridedBatched
#endif
#ifndef CUBLASLT_MATMUL_DESC_A_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_A_SCALE_POINTER MUBLASLT_MATMUL_DESC_A_SCALE_POINTER
#endif
#ifndef CUBLASLT_MATMUL_DESC_B_SCALE_POINTER
#define CUBLASLT_MATMUL_DESC_B_SCALE_POINTER MUBLASLT_MATMUL_DESC_B_SCALE_POINTER
#endif
#ifndef CUBLASLT_MATMUL_DESC_FAST_ACCUM
#define CUBLASLT_MATMUL_DESC_FAST_ACCUM MUBLASLT_MATMUL_DESC_FAST_ACCUM
#endif
#ifndef CUBLASLT_MATMUL_DESC_TRANSA
#define CUBLASLT_MATMUL_DESC_TRANSA MUBLASLT_MATMUL_DESC_TRANSA
#endif
#ifndef CUBLASLT_MATMUL_DESC_TRANSB
#define CUBLASLT_MATMUL_DESC_TRANSB MUBLASLT_MATMUL_DESC_TRANSB
#endif
#ifndef CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#define CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
#endif
#ifndef CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#define CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT MUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
#endif
#ifndef CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#define CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET MUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET
#endif
#ifndef CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F MUBLAS_COMPUTE_32F
#endif
#ifndef CUBLAS_OP_N
#define CUBLAS_OP_N MUBLAS_OP_N
#endif
#ifndef CUBLAS_OP_T
#define CUBLAS_OP_T MUBLAS_OP_T
#endif
#ifndef CUBLAS_STATUS_NOT_SUPPORTED
#define CUBLAS_STATUS_NOT_SUPPORTED MUBLAS_STATUS_NOT_IMPLEMENTED
#endif
#ifndef CUBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#endif
#ifndef cublasGetStatusString
#define cublasGetStatusString mublasGetStatusString
#endif
#ifndef cublasLtMatmul
#define cublasLtMatmul mublasLtMatmul
#endif
#ifndef cublasLtMatmulAlgoGetHeuristic
#define cublasLtMatmulAlgoGetHeuristic mublasLtMatmulAlgoGetHeuristic
#endif
#ifndef cublasLtMatmulDescCreate
#define cublasLtMatmulDescCreate mublasLtMatmulDescCreate
#endif
#ifndef cublasLtMatmulDescDestroy
#define cublasLtMatmulDescDestroy mublasLtMatmulDescDestroy
#endif
#ifndef cublasLtMatmulDescSetAttribute
#define cublasLtMatmulDescSetAttribute mublasLtMatmulDescSetAttribute
#endif
#ifndef cublasLtMatmulPreferenceCreate
#define cublasLtMatmulPreferenceCreate mublasLtMatmulPreferenceCreate
#endif
#ifndef cublasLtMatmulPreferenceDestroy
#define cublasLtMatmulPreferenceDestroy mublasLtMatmulPreferenceDestroy
#endif
#ifndef cublasLtMatmulPreferenceSetAttribute
#define cublasLtMatmulPreferenceSetAttribute mublasLtMatmulPreferenceSetAttribute
#endif
#ifndef cublasLtMatrixLayoutCreate
#define cublasLtMatrixLayoutCreate mublasLtMatrixLayoutCreate
#endif
#ifndef cublasLtMatrixLayoutDestroy
#define cublasLtMatrixLayoutDestroy mublasLtMatrixLayoutDestroy
#endif
#ifndef cublasLtMatrixLayoutSetAttribute
#define cublasLtMatrixLayoutSetAttribute mublasLtMatrixLayoutSetAttribute
#endif
#ifndef cublasSetMathMode
#define cublasSetMathMode mublasSetMathMode
#endif
#ifndef cublasGetMathMode
#define cublasGetMathMode mublasGetMathMode
#endif
#ifndef CUBLAS_DEFAULT_MATH
#define CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#endif
#ifndef CUBLAS_TENSOR_OP_MATH
#define CUBLAS_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#endif
#ifndef cublasLtCreate
#define cublasLtCreate mublasLtCreate
#endif
#ifndef cublasLtDestroy
#define cublasLtDestroy mublasLtDestroy
#endif
#ifndef getCurrentCUDABlasHandle
#define getCurrentCUDABlasHandle getCurrentMUSABlasHandle
#endif
