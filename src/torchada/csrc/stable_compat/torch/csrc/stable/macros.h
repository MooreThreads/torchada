// torchada stable-ABI compat: torch/csrc/stable/macros.h
//
// Resolves to torch_musa's own macros.h wherever the search path holds one
// (torch 2.11+), otherwise to library.h, where torch 2.9 keeps the stable
// library macros. The position of this directory in the search path therefore
// does not matter. Rules for these stand-ins: AGENTS.md, "C++ Operator
// Overrides".
#if defined(__has_include_next)
#if __has_include_next(<torch/csrc/stable/macros.h>)
#include_next <torch/csrc/stable/macros.h>
#else
#include <torch/csrc/stable/library.h>
#endif
#else
#include <torch/csrc/stable/library.h>
#endif
