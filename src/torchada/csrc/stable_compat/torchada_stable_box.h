#pragma once
#include <torch/version.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/Exception.h>
#include <cstdint>
#include <musa_runtime.h>

// This header can be force-included unconditionally by downstream build files.
// PyTorch 2.11+ already provides the boxer, Device, HeaderOnlyArrayRef and the
// stable free functions below, so compiling the backport there would redefine
// native stable-ABI symbols.

// --- Always-on shims (needed on every torch_musa version) ---
// CUDA_VERSION guards in the kernels: define low so the sm100/Blackwell fast
// paths (also gated on cc_major>=10) compile out on MUSA. torchada also passes
// -DCUDA_VERSION=0 via the build, but define here for direct/JIT use.
#ifndef CUDA_VERSION
#define CUDA_VERSION 0
#endif

// Stable-ABI runtime-error check used by some libtorch-stable kernels (e.g.
// minimax_reduce_rms_kernel). The ported kernel calls musa* runtime APIs that
// return a musaError_t (0 == success); wrap them in STD_TORCH_CHECK.
// mcc translates STD_CUDA_CHECK -> STD_MUSA_CHECK at the source level, so both
// names are guarded here.
#ifndef STD_CUDA_CHECK
#define STD_CUDA_CHECK(EXPR)                                              \
  do {                                                                   \
    auto _musa_err = (EXPR);                                             \
    STD_TORCH_CHECK(_musa_err == 0, "MUSA runtime error: ",              \
                    static_cast<int64_t>(_musa_err));                    \
  } while (0)
#endif
#ifndef STD_MUSA_CHECK
#define STD_MUSA_CHECK(EXPR) STD_CUDA_CHECK(EXPR)
#endif

// Kernel-launch error check used by some libtorch-stable kernels after a <<<>>>
// launch (e.g. selective_scan). Checks the last runtime error via the same
// STD_CUDA_CHECK path.
#ifndef STD_CUDA_KERNEL_LAUNCH_CHECK
#define STD_CUDA_KERNEL_LAUNCH_CHECK() STD_CUDA_CHECK(musaGetLastError())
#endif
#ifndef STD_MUSA_KERNEL_LAUNCH_CHECK
#define STD_MUSA_KERNEL_LAUNCH_CHECK() STD_CUDA_KERNEL_LAUNCH_CHECK()
#endif

// Newer torch_musa exposes the BLAS handle through its stable C shim. Keep a
// CUDA-named forwarding wrapper for downstream sources that bypass source
// porting; normally the mapping rule rewrites the call directly to the MUSA
// name. This avoids depending on torch_musa's private C++ handle-pool ABI.
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 11)
#include <torch/csrc/stable/c/shim.h>
static inline AOTITorchError torch_get_current_cuda_blas_handle(void** ret) {
  return torch_get_current_musa_blas_handle(ret);
}
#else
// torch_musa 2.9 has no stable C shim for the BLAS handle. Provide the MUSA
// name produced by source porting through the legacy C++ handle-pool API, then
// keep the CUDA name as a forwarding wrapper for unported sources.
struct _mublasHandle_t;
typedef struct _mublasHandle_t* mublasHandle_t;
namespace at {
namespace musa {
mublasHandle_t getCurrentMUSABlasHandle();
}
}
static inline AOTITorchError torch_get_current_musa_blas_handle(void** ret) {
  auto handle = at::musa::getCurrentMUSABlasHandle();
  *ret = reinterpret_cast<void*>(handle);
  return handle ? 0 : 1;  // fail fast on a null handle instead of crashing in muBLAS
}
static inline AOTITorchError torch_get_current_cuda_blas_handle(void** ret) {
  return torch_get_current_musa_blas_handle(ret);
}
#endif

// torch_musa's AOTI C-shim exposes aoti_torch_get_current_musa_stream, not the
// upstream _cuda_ spelling that libtorch-stable kernels (torch_utils.h) call.
// Forward so those kernels compile against the upstream name.
#ifndef TORCHADA_HAVE_AOTI_CUDA_STREAM
#define TORCHADA_HAVE_AOTI_CUDA_STREAM 1
static inline AOTITorchError aoti_torch_get_current_cuda_stream(int32_t device_index,
                                                                void** ret) {
  return aoti_torch_get_current_musa_stream(device_index, ret);
}
#endif

// Some libtorch-stable kernels reference TORCH_UTILS_CHECK, which torch_musa's
// stable headers do not define; alias it to the stable check macro.
#ifndef TORCH_UTILS_CHECK
#define TORCH_UTILS_CHECK STD_TORCH_CHECK
#endif

// --- torch < 2.11 backport: boxer, Device, stable free functions ---
// PyTorch 2.11+ already provides the boxer, Device, HeaderOnlyArrayRef and the
// stable free functions below, so compiling the backport there would redefine
// native stable-ABI symbols.
#if TORCH_VERSION_MAJOR < 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 11)
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
// ostringstream. The operator<<(ostream&, ScalarType) lives in
// <c10/core/ScalarType.h>, which those TUs do not transitively include (they
// get ScalarType from a lighter header). Pull it in here -- the box header is
// force-included in every stable TU -- so the streaming operator is in scope.
#include <c10/core/ScalarType.h>

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
// Matches the (size, dtype, layout, device) prefix of torch::stable::empty that
// every MUSA-compiled call uses, e.g.
//   empty({..}, ScalarType::Int, std::nullopt, tensor.device())
// The 3rd slot is layout (a std::nullopt in those calls), NOT pin_memory; it is
// ignored because only a strided, contiguous tensor is produced on MUSA. dtype +
// device carry what aoti_torch_empty_strided needs. The fully defaulted / layout-
// enum forms (empty({1, 1}), empty(size, dtype, some_layout, ...)) are not in the
// MUSA kernel set and intentionally not supported.
inline Tensor empty(c10::IntArrayRef size, ScalarType dtype,
                    std::optional<int64_t> /*layout, ignored*/, Device device) {
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
// Mirrors the no-deleter torch::stable::from_blob (data, sizes, strides, device,
// dtype, storage_offset). The torch 2.11 deleter overload is intentionally NOT
// shimmed: torch_musa 2.9's AOTI C-shim has no from_blob-with-deleter entry, so a
// deleter would have to be silently dropped (leaking / dangling the backing
// storage). Kernels that need the deleter form (e.g. get_cuda_view_from_cpu_tensor)
// must stay out of the MUSA kernel set until torch_musa exposes a deleter-capable
// create_tensor_from_blob.
inline Tensor from_blob(void* data, c10::IntArrayRef sizes,
                        c10::IntArrayRef strides, Device device,
                        ScalarType dtype, int64_t storage_offset = 0) {
  AtenTensorHandle h;
  TORCH_ERROR_CODE_CHECK(aoti_torch_create_tensor_from_blob(
      data, static_cast<int64_t>(sizes.size()), sizes.data(), strides.data(),
      storage_offset, static_cast<int32_t>(dtype), device.type_,
      device.index_, &h));
  return Tensor(h);
}
}  // namespace stable
}  // namespace torch

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

#endif  // torch < 2.11
