#pragma once
// torch_musa 2.9 has no torch/csrc/stable/device.h; newer libtorch-stable kernels
// include it for torch::stable::Device / DeviceType. Provide them on the AOTI
// C-shim. The force-included box header pulls this in before <tensor.h> so the
// patched tensor_struct.h device() accessor can return it.
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cstdint>
namespace torch {
namespace stable {
// Values match c10::DeviceType so comparisons against the AOTI device type hold.
enum class DeviceType : int8_t {
  CPU = 0,
  CUDA = 1,
  HIP = 6,
  XPU = 12,
  MPS = 13,
  Meta = 14,
  PrivateUse1 = 20,
};
struct Device {
  int32_t type_;
  int32_t index_;
  bool is_privateuseone() const {
    return type_ == aoti_torch_device_type_privateuse1();
  }
  bool is_cuda() const { return type_ == aoti_torch_device_type_cuda(); }
  bool is_cpu() const { return type_ == aoti_torch_device_type_cpu(); }
  DeviceType type() const { return static_cast<DeviceType>(type_); }
  int32_t index() const { return index_; }
  bool operator==(const Device& o) const {
    return type_ == o.type_ && index_ == o.index_;
  }
  bool operator!=(const Device& o) const { return !(*this == o); }
};
}  // namespace stable
}  // namespace torch
