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
// graph_/graph_exec_/has_graph_exec_ are `protected` in at::musa::MUSAGraph. We
// reach them through pointers-to-members formed inside a derived class, where
// protected access is permitted. Because the members are declared in MUSAGraph,
// the pointer type is pointer-to-member-of-MUSAGraph and applies directly to a
// real MUSAGraph object -- no base->derived downcast, no undefined behavior.
#include <torch/extension.h>
#include "aten/musa/MUSAGraph.h"
#include <musa_runtime_api.h>

namespace {
using Graph = at::musa::MUSAGraph;

// Pointers to torch_musa's protected members, formed where access is allowed.
struct Members : public Graph {
  static musaGraphExec_t Graph::* exec() { return &Members::graph_exec_; }
  static musaGraph_t     Graph::* tmpl() { return &Members::graph_; }
  static bool            Graph::* has_exec_flag() { return &Members::has_graph_exec_; }
};

inline musaGraphExec_t& exec_of(Graph& g) { return g.*Members::exec(); }
inline musaGraph_t&     tmpl_of(Graph& g) { return g.*Members::tmpl(); }
inline bool&            has_exec_flag_of(Graph& g) { return g.*Members::has_exec_flag(); }

// Ground truth for "is the executable live" is the pointer itself, not the flag.
inline bool exec_live(Graph& g) { return exec_of(g) != nullptr; }
}  // namespace

// Destroy graph_exec_, keep the graph_ template. Keyed on the pointer (not just the
// flag) so it stays correct even if torch_musa's flag and pointer ever desync.
static void free_exec(Graph& g) {
  if (exec_live(g)) {
    musaError_t e = musaGraphExecDestroy(exec_of(g));
    TORCH_CHECK(e == musaSuccess, "musaGraphExecDestroy failed: ", static_cast<int>(e));
    exec_of(g) = nullptr;
  }
  has_exec_flag_of(g) = false;  // keep torch_musa's reset()/dtor bookkeeping in sync
}

// Re-instantiate graph_exec_ from the kept template. No-op if already live.
static void inst_exec(Graph& g) {
  if (exec_live(g)) return;
  TORCH_CHECK(tmpl_of(g) != nullptr, "inst_exec: no graph template to instantiate from");
  musaError_t e = musaGraphInstantiate(&exec_of(g), tmpl_of(g), nullptr, nullptr, 0);
  TORCH_CHECK(e == musaSuccess, "musaGraphInstantiate failed: ", static_cast<int>(e));
  has_exec_flag_of(g) = true;  // keep torch_musa's reset()/dtor bookkeeping in sync
}

static bool has_exec(Graph& g) { return exec_live(g); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("free_exec", &free_exec, "destroy graph_exec_, keep graph_ template");
  m.def("inst_exec", &inst_exec, "re-instantiate graph_exec_ from the kept template");
  m.def("has_exec", &has_exec, "whether graph_exec_ is currently live");
}
