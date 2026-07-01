#pragma once
// torch_musa 2.9's torch::stable predates torch 2.10's split of the stable
// library macros into torch/csrc/stable/macros.h; they still live in
// library.h. Forward so newer kernels that include macros.h directly compile.
#include <torch/csrc/stable/library.h>
