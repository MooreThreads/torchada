#pragma once
// torchada stable-ABI compat: torch/headeronly/core/Dispatch.h
//
// Resolves to torch_musa's own Dispatch.h wherever the search path holds one
// (torch 2.11+), so the position of this directory does not matter. The
// fallback supplies the THO_DISPATCH_* macros that libtorch-stable kernels
// (vLLM, SGLang) need on torch_musa 2.9, verbatim from upstream, plus the two
// symbols they reference that it lacks
// (torch::headeronly::impl::ScalarTypeToCPPTypeT, toString), aliased to the c10
// equivalents.
#if defined(__has_include_next)
#if __has_include_next(<torch/headeronly/core/Dispatch.h>)
#include_next <torch/headeronly/core/Dispatch.h>
#define TORCHADA_HAS_NATIVE_THO_DISPATCH 1
#endif
#endif

#ifndef TORCHADA_HAS_NATIVE_THO_DISPATCH
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>  // STD_TORCH_CHECK
#include <torch/version.h>
#include <c10/core/ScalarType.h>

// torch 2.11+ provides these through ScalarType.h; reached only when such a
// tree omits Dispatch.h itself, since one that ships it is taken above.
#if TORCH_VERSION_MAJOR < 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 11)
namespace torch {
namespace headeronly {
namespace impl {
template <torch::headeronly::ScalarType N>
using ScalarTypeToCPPTypeT =
    typename c10::impl::ScalarTypeToCPPType<static_cast<c10::ScalarType>(N)>::type;
}  // namespace impl
inline const char* toString(torch::headeronly::ScalarType t) {
  return c10::toString(static_cast<c10::ScalarType>(t));
}
}  // namespace headeronly
}  // namespace torch
#endif  // torch < 2.11

#define THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...) \
  case enum_type: {                                                          \
    PRELUDE(enum_type);                                                      \
    using HINT [[maybe_unused]] =                                            \
        torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>;            \
    return __VA_ARGS__();                                                    \
  }

#define THO_DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...) \
  CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

// Own namespace: torch_musa's ATen/Dispatch.h defines ::detail::scalar_type for
// the same type (torch::headeronly::ScalarType aliases at::ScalarType), so a
// global one would be a redefinition when both headers land in one TU.
namespace torchada_detail {
inline torch::headeronly::ScalarType scalar_type(torch::headeronly::ScalarType s) {
  return s;
}
}  // namespace torchada_detail

#define THO_DISPATCH_SWITCH_TMPL(                                          \
    PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)                       \
  [&] {                                                                    \
    const auto& the_type = TYPE;                                          \
    constexpr const char* at_dispatch_name = NAME;                        \
    torch::headeronly::ScalarType _st = ::torchada_detail::scalar_type(the_type);  \
    PRELUDE(at_dispatch_name, _st);                                       \
    C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")           \
    switch (_st) {                                                        \
      __VA_ARGS__                                                         \
      default:                                                           \
        CHECK_NOT_IMPLEMENTED(                                           \
            false, '"', at_dispatch_name, "\" not implemented for '",    \
            torch::headeronly::toString(_st), "'");                      \
    }                                                                    \
    C10_DIAGNOSTIC_POP()                                                 \
  }()

#define THO_EMPTY(...)

#define THO_PRIVATE_CASE_TYPE_USING_HINT(enum_type, HINT, ...) \
  THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(THO_EMPTY, enum_type, HINT, __VA_ARGS__)

#define THO_DISPATCH_SWITCH(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH_TMPL(THO_EMPTY, STD_TORCH_CHECK, TYPE, NAME, __VA_ARGS__)

#define THO_DISPATCH_CASE(enum_type, ...) \
  THO_DISPATCH_CASE_TMPL(THO_PRIVATE_CASE_TYPE_USING_HINT, enum_type, __VA_ARGS__)

#endif  // TORCHADA_HAS_NATIVE_THO_DISPATCH
