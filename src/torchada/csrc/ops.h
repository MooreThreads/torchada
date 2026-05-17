// torchada C++ operator overrides
//
// This header provides the infrastructure for registering custom ATen operator
// implementations that override the default PrivateUse1 (MUSA) implementations.
//
// Usage:
//   1. Include this header in your .cpp or .mu file.
//   2. Use TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) to register overrides.
//   3. Use is_override_enabled("op_name") before registering an override.
//   4. Let torchada build and load the extension automatically.
//
// Example:
//   #include "ops.h"
//
//   at::Tensor my_custom_add(const at::Tensor& self, const at::Tensor& other,
//                            const at::Scalar& alpha) {
//       log_op_call("add.Tensor");  // Optional debug logging.
//       // Custom implementation.
//       auto result = at::empty_like(self);
//       result.copy_(self);
//       result.add_(other, alpha);
//       return result;
//   }
//
//   TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
//       // Check the environment at registration time.
//       if (torchada::is_override_enabled("add")) {
//           m.impl("add.Tensor", my_custom_add);
//       }
//   }
//
// Environment variables:
//   TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>=1  - Disable specific operator override
//   TORCHADA_DEBUG_CPP_OPS=1                  - Log operator calls to stdout

#pragma once

#include <torch/extension.h>

namespace torchada {

// Version information.
constexpr const char* VERSION = "0.1.0";

// Return whether an operator override is enabled by environment settings.
inline bool is_override_enabled(const char* op_name) {
    std::string env_var = "TORCHADA_DISABLE_OP_OVERRIDE_";
    env_var += op_name;
    const char* val = std::getenv(env_var.c_str());
    if (val != nullptr && std::string(val) == "1") {
        return false;
    }
    return true;
}

// Log operator calls when C++ operator debugging is enabled.
inline void log_op_call(const char* op_name) {
    const char* debug = std::getenv("TORCHADA_DEBUG_CPP_OPS");
    if (debug != nullptr && std::string(debug) == "1") {
        std::cout << "[torchada] " << op_name << " called" << std::endl;
    }
}

}  // namespace torchada
