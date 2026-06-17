#pragma once
// torchada stable-ABI compat: TORCH_BOX + small helpers.
//
// torch_musa 2.9.0's torch::stable runtime predates the TORCH_BOX boxer family
// (it lacks guts::typelist / UnboxType / the boxer templates), so
// STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ...) { ops.impl("x", TORCH_BOX(&x)); }
// in a project's stable torch_bindings.cpp (e.g. vLLM, SGLang) does not compile.
// torch_musa DOES ship the from<>/to<> StableIValue conversions, so we
// synthesize TORCH_BOX with plain
// std traits over those — no dependency on the newer boxer machinery.
//
// Force-include this header (mcc -include) when building libtorch_stable sources.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <c10/util/ArrayRef.h>
#include <tuple>
#include <type_traits>
#include <utility>

// torch_musa's stable headers predate torch::headeronly::HeaderOnlyArrayRef
// (used by some op prototypes in libtorch_stable/ops.h, e.g. the FP8-quant
// group_shape arg). Alias to c10::ArrayRef so those prototypes compile -- no
// impl is needed for the ops that use it.
namespace torch {
namespace headeronly {
template <typename T>
using HeaderOnlyArrayRef = c10::ArrayRef<T>;
using IntHeaderOnlyArrayRef = c10::ArrayRef<int64_t>;
}  // namespace headeronly
}  // namespace torch

// torch_musa's stable ops.h predates torch::stable::contiguous (used by
// libtorch_stable/layernorm_kernels.cu when input.stride(-1) != 1). Provide it
// with the same aoti_torch_call_dispatcher pattern torch_musa already uses for
// empty_like/narrow. Schema:
//   aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous)
//     -> Tensor(a)
// MemoryFormat::Contiguous == 0; box it as int64 -- aoti_torch_call_dispatcher
// reads each StableIValue per the resolved op schema, so the int slot is
// reinterpreted as the MemoryFormat enum. torch_musa ships no from(MemoryFormat)
// overload, which is why the enum can't be passed directly.
#include <torch/csrc/stable/ops.h>
#include <array>
namespace torch {
namespace stable {
inline Tensor contiguous(const Tensor& self) {
  std::array<StableIValue, 2> stack{from(self), from(static_cast<int64_t>(0))};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::contiguous", "", stack.data()));
  return to<Tensor>(stack[0]);
}
}  // namespace stable
}  // namespace torch

// CUDA_VERSION guards in the kernels: define low so the sm100/Blackwell fast
// paths (also gated on cc_major>=10) compile out on MUSA. torchada also passes
// -DCUDA_VERSION=0 via the build, but define here for direct/JIT use.
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

// torch_get_current_cuda_blas_handle has no AOTI stable-ABI shim on torch_musa,
// but torch_musa exposes the stream-bound current handle through its handle pool
// (at::musa::getCurrentMUSABlasHandle). Forward-declare it (resolved at import
// time from the already-loaded libtorch_musa) and return the muBLAS handle so the
// gptq cuBLAS->muBLAS GEMM path works. mublasHandle_t per
// /usr/local/musa/include/internal/mublas_types.h; the typedef is harmless if a
// later <mublas.h> (via the cublas_v2.h->mublas.h mapping) repeats it.
struct _mublasHandle_t;
typedef struct _mublasHandle_t* mublasHandle_t;
namespace at {
namespace musa {
mublasHandle_t getCurrentMUSABlasHandle();
}
}  // namespace at
static inline AOTITorchError torch_get_current_cuda_blas_handle(void** ret) {
  auto handle = at::musa::getCurrentMUSABlasHandle();
  *ret = reinterpret_cast<void*>(handle);
  return handle ? 0 : 1;  // fail fast on a null handle instead of crashing in muBLAS
}

namespace torchada_stable {

template <class T>
using strip_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <class F>
struct fn_traits;
template <class R, class... A>
struct fn_traits<R (*)(A...)> {
  using ret = R;
  using args = std::tuple<strip_t<A>...>;
};

template <auto Fn, class Tup, std::size_t... I>
inline void invoke_boxed(StableIValue* stack, std::index_sequence<I...>) {
  // Materialize args as lvalues so they bind to Tensor& parameters.
  Tup args{to<std::tuple_element_t<I, Tup>>(stack[I])...};
  using R = typename fn_traits<decltype(Fn)>::ret;
  if constexpr (std::is_void_v<R>) {
    std::apply(Fn, args);
  } else {
    stack[0] = from(std::apply(Fn, args));
  }
}

template <auto Fn>
inline void boxed(StableIValue* stack, uint64_t /*nargs*/, uint64_t /*nout*/) {
  // The op's C++ signature is the source of truth for how many stack slots are
  // read/written, so nargs/nout are redundant with it and intentionally unused.
  // A schema/signature mismatch is an author error caught by the stable-ABI
  // unit test, which exercises every signature shape (void / Tensor / int
  // returns, scalar args).
  using Tup = typename fn_traits<decltype(Fn)>::args;
  invoke_boxed<Fn, Tup>(stack,
                        std::make_index_sequence<std::tuple_size_v<Tup>>());
}

}  // namespace torchada_stable

#ifndef TORCH_BOX
#define TORCH_BOX(func) (&::torchada_stable::boxed<func>)
#endif
