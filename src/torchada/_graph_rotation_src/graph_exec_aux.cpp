// torchada CUDA-graph executable-rotation aux extension.
//
// Frees a MUSAGraph's executable while KEEPING its template, and re-instantiates
// the executable from the kept template on demand. This lets torchada cap the
// number of LIVE musaGraphExec_t (the MUSA driver throws an illegal-memory-access
// at ~2048 live executables per process) while still being able to replay any
// captured graph by re-instantiating it from its (cheap, uncapped) template.
//
// Compiled against torch_musa's INSTALLED headers at runtime via
// torch.utils.cpp_extension.load -- it does NOT rebuild torch_musa.
//
// graph_/graph_exec_ are `protected` in at::musa::MUSAGraph; we reach them with a
// derived accessor that adds no data members (identical object layout), which is a
// standard, well-defined way to access protected base members.
#include <torch/extension.h>
#include "aten/musa/MUSAGraph.h"
#include <musa_runtime_api.h>

namespace {
struct Accessor : public at::musa::MUSAGraph {
  musaGraphExec_t& exec() { return graph_exec_; }
  musaGraph_t&     tmpl() { return graph_; }
  bool&            has_exec() { return has_graph_exec_; }
  bool&            has_tmpl() { return has_graph_; }
};
}  // namespace

static void free_exec(at::musa::MUSAGraph& g) {
  auto& a = static_cast<Accessor&>(g);
  if (a.has_exec() && a.exec() != nullptr) {
    musaError_t e = musaGraphExecDestroy(a.exec());
    TORCH_CHECK(e == musaSuccess, "musaGraphExecDestroy failed: ", static_cast<int>(e));
    a.exec() = nullptr;
    a.has_exec() = false;
  }
}

static void inst_exec(at::musa::MUSAGraph& g) {
  auto& a = static_cast<Accessor&>(g);
  if (!a.has_exec()) {
    TORCH_CHECK(a.has_tmpl() && a.tmpl() != nullptr,
                "inst_exec: no graph template to instantiate from");
    musaError_t e = musaGraphInstantiate(&a.exec(), a.tmpl(), nullptr, nullptr, 0);
    TORCH_CHECK(e == musaSuccess, "musaGraphInstantiate failed: ", static_cast<int>(e));
    a.has_exec() = true;
  }
}

static bool has_exec(at::musa::MUSAGraph& g) {
  return static_cast<Accessor&>(g).has_exec();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("free_exec", &free_exec, "destroy graph_exec_, keep graph_ template");
  m.def("inst_exec", &inst_exec, "re-instantiate graph_exec_ from the kept template");
  m.def("has_exec", &has_exec, "whether graph_exec_ is currently live");
}
