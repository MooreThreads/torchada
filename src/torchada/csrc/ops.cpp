// torchada C++ operator overrides - Main source file
//
// This file contains the operator registration infrastructure and example overrides.
// Custom operator implementations can be added here or in separate files.
//
// To add a new operator override:
//   1. Write the implementation function.
//   2. Register it using TORCH_LIBRARY_IMPL(aten, PrivateUse1, m).
//
// Note: Operators registered here will override torch_musa's implementations.
// Use with caution and ensure correctness.

#include "ops.h"

#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAPluggableAllocator.h"
#include <thread>

namespace torchada {

// CUDA-compatible memory pool allocation functions on MUSA.

static void _musa_beginAllocateCurrentThreadToPool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id) {
  auto tid = std::this_thread::get_id();

  c10::musa::MUSACachingAllocator::beginAllocateToPool(
      device, mempool_id, [=](musaStream_t) {
        auto current_tid = std::this_thread::get_id();
        return current_tid == tid;
      });
}

static void _musa_endAllocateToPool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id) {
  c10::musa::MUSACachingAllocator::endAllocateToPool(device, mempool_id);
}

static void _musa_releasePool(
    c10::DeviceIndex device,
    c10::musa::MempoolId_t mempool_id) {
  c10::musa::MUSACachingAllocator::releasePool(device, mempool_id);
}

// Utility functions exposed to Python.

static bool cpp_ops_loaded = false;

bool is_loaded() {
    return cpp_ops_loaded;
}

const char* get_version() {
    return VERSION;
}

void mark_loaded() {
    cpp_ops_loaded = true;
}

}  // namespace torchada


// Python bindings.

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchada C++ operator overrides";

    m.def("is_loaded", &torchada::is_loaded,
          "Check if C++ ops extension is loaded");
    m.def("get_version", &torchada::get_version,
          "Get the C++ ops extension version");
    m.def("_mark_loaded", &torchada::mark_loaded,
          "Mark the extension as loaded (internal use)");

    m.def("_cuda_beginAllocateCurrentThreadToPool",
          &torchada::_musa_beginAllocateCurrentThreadToPool,
          "Begin allocating memory from the current thread to a memory pool");
    m.def("_cuda_endAllocateToPool",
          &torchada::_musa_endAllocateToPool,
          "End allocating memory to a memory pool");
    m.def("_cuda_releasePool",
          &torchada::_musa_releasePool,
          "Release a memory pool");
}
