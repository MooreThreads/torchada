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

// torch_musa 2.9's torch::stable predates torch 2.10's Tensor::{sizes,strides,
// device}() and torch::stable::{empty,from_blob} that newer vLLM/SGLang stable
// kernels call. Define the stable Device here -- BEFORE <tensor.h> -- so the
// torchada-patched tensor_struct.h accessors (gated on TORCHADA_STABLE_ACCESSORS)
// can reference it; empty/from_blob are defined after <tensor.h> below.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <c10/util/ArrayRef.h>
#include <optional>
#include <vector>
#include <torch/csrc/stable/device.h>
#define TORCHADA_STABLE_ACCESSORS 1
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

// STD_TORCH_CHECK error messages in some libtorch-stable kernels (e.g.
// selective_scan_fwd's dispatch default case) stream a c10::ScalarType into the
// ostringstream, but torch_musa's headeronly snapshot ships no
// operator<<(ostream&, ScalarType). Provide one so those checks compile; it
// prints the underlying enum value (sufficient for an error message).
#include <ostream>
#include <c10/core/ScalarType.h>
namespace c10 {
inline std::ostream& operator<<(std::ostream& os, ScalarType t) {
  return os << "ScalarType(" << static_cast<int>(t) << ")";
}
}  // namespace c10

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

// aten::flatten.using_ints(Tensor self, int start_dim, int end_dim) -> Tensor
inline Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
  std::array<StableIValue, 3> stack{
      from(self), from(start_dim), from(end_dim)};
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::flatten", "using_ints", stack.data()));
  return to<Tensor>(stack[0]);
}

// torch 2.10 factory functions, backported onto torch_musa's AOTI C-shim.
inline Tensor empty(c10::IntArrayRef size, ScalarType dtype,
                    std::optional<int64_t> /*pin_memory, unused*/, Device device) {
  std::vector<int64_t> strides(size.size());
  int64_t acc = 1;
  for (int64_t i = static_cast<int64_t>(size.size()) - 1; i >= 0; --i) {
    strides[i] = acc;
    acc *= size[i];
  }
  AtenTensorHandle h;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(
      static_cast<int64_t>(size.size()), size.data(), strides.data(),
      static_cast<int32_t>(dtype), device.type_, device.index_, &h));
  return Tensor(h);
}
inline Tensor from_blob(void* data, c10::IntArrayRef sizes,
                        c10::IntArrayRef strides, Device device,
                        ScalarType dtype) {
  AtenTensorHandle h;
  TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
      data, static_cast<int64_t>(sizes.size()), sizes.data(), strides.data(),
      /*storage_offset=*/0, static_cast<int32_t>(dtype), device.type_,
      device.index_, &h));
  return Tensor(h);
}
}  // namespace stable
}  // namespace torch

// CUDA_VERSION guards in the kernels: define low so the sm100/Blackwell fast
// paths (also gated on cc_major>=10) compile out on MUSA. torchada also passes
// -DCUDA_VERSION=0 via the build, but define here for direct/JIT use.
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

// Stable-ABI runtime-error check used by some libtorch-stable kernels (e.g.
// minimax_reduce_rms_kernel). The ported kernel calls musa* runtime APIs that
// return a musaError_t (0 == success); wrap them in STD_TORCH_CHECK.
#ifndef STD_CUDA_CHECK
#define STD_CUDA_CHECK(EXPR)                                              \
  do {                                                                   \
    auto _musa_err = (EXPR);                                             \
    STD_TORCH_CHECK(_musa_err == 0, "MUSA runtime error: ",             \
                    static_cast<int64_t>(_musa_err));                    \
  } while (0)
#endif

// Kernel-launch error check used by some libtorch-stable kernels after a <<<>>>
// launch (e.g. selective_scan). Checks the last runtime error via the same
// STD_CUDA_CHECK path.
#ifndef STD_CUDA_KERNEL_LAUNCH_CHECK
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(musaGetLastError())
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

// A multi-value (tuple/pair) return occupies one stack slot per element in the
// stable ABI; torch_musa's from()/to() only convert single <=64-bit values, so
// the boxer spreads a tuple return across consecutive slots itself rather than
// calling from() on the whole aggregate (which would trip the 64-bit
// static_assert).
template <class T>
struct is_tuple_like : std::false_type {};
template <class... Ts>
struct is_tuple_like<std::tuple<Ts...>> : std::true_type {};
template <class A, class B>
struct is_tuple_like<std::pair<A, B>> : std::true_type {};

template <class Ret, std::size_t... J>
inline void store_return(StableIValue* stack, Ret&& ret,
                         std::index_sequence<J...>) {
  ((stack[J] = from(std::get<J>(std::forward<Ret>(ret)))), ...);
}

template <auto Fn, class Tup, std::size_t... I>
inline void invoke_boxed(StableIValue* stack, std::index_sequence<I...>) {
  // Materialize args as lvalues so they bind to Tensor& parameters.
  Tup args{to<std::tuple_element_t<I, Tup>>(stack[I])...};
  using R = typename fn_traits<decltype(Fn)>::ret;
  if constexpr (std::is_void_v<R>) {
    std::apply(Fn, args);
  } else if constexpr (is_tuple_like<strip_t<R>>::value) {
    store_return(stack, std::apply(Fn, args),
                 std::make_index_sequence<std::tuple_size_v<strip_t<R>>>());
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
