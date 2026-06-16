#pragma once
// torchada stable-ABI compat: torch/headeronly/core/Dispatch.h
//
// libtorch-stable kernels (e.g. vLLM, SGLang) need the THO_DISPATCH_* macros
// from PyTorch's torch/headeronly/core/Dispatch.h, which postdates torch_musa
// 2.9.0. We ship
// the macros (verbatim from upstream) plus the two symbols they reference that
// torch_musa lacks (torch::headeronly::impl::ScalarTypeToCPPTypeT, toString),
// aliased to the c10 equivalents that DO exist on torch_musa. torchada exposes
// this directory via include_paths() so it shadows the (absent) torch header.
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>  // STD_TORCH_CHECK
#include <c10/core/ScalarType.h>

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

#define THO_PRIVATE_CASE_TYPE_USING_HINT_TMPL(PRELUDE, enum_type, HINT, ...) \
  case enum_type: {                                                          \
    PRELUDE(enum_type);                                                      \
    using HINT [[maybe_unused]] =                                            \
        torch::headeronly::impl::ScalarTypeToCPPTypeT<enum_type>;            \
    return __VA_ARGS__();                                                    \
  }

#define THO_DISPATCH_CASE_TMPL(CASE_TYPE_USING_HINT, enum_type, ...) \
  CASE_TYPE_USING_HINT(enum_type, scalar_t, __VA_ARGS__)

namespace detail {
inline torch::headeronly::ScalarType scalar_type(torch::headeronly::ScalarType s) {
  return s;
}
}  // namespace detail

#define THO_DISPATCH_SWITCH_TMPL(                                          \
    PRELUDE, CHECK_NOT_IMPLEMENTED, TYPE, NAME, ...)                       \
  [&] {                                                                    \
    const auto& the_type = TYPE;                                          \
    constexpr const char* at_dispatch_name = NAME;                        \
    torch::headeronly::ScalarType _st = ::detail::scalar_type(the_type);  \
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
