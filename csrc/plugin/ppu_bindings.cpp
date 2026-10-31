// Plugin-owned op registration for the core extension (vllm_sail._C).
//
// PPU ops register under the plugin-private namespace _ppu_C.
// vllm_sail/attention/ops/mla_sparse.py calls
// torch.ops._ppu_C.top_k_per_row_prefill_bf16 directly.

#include <torch/csrc/stable/library.h>

void top_k_per_row_prefill_bf16(const torch::stable::Tensor& logits,
                                const torch::stable::Tensor& rowStarts,
                                const torch::stable::Tensor& rowEnds,
                                torch::stable::Tensor& indices,
                                int64_t numRows, int64_t stride0,
                                int64_t stride1, int64_t topK);

void deepseek_dequant_gather_vector(
    torch::stable::Tensor& out, const torch::stable::Tensor& cache,
    const torch::stable::Tensor& seq_lens, const torch::stable::Tensor& gather_lens,
    const torch::stable::Tensor& block_table, int64_t page_size,
    int64_t offset, int64_t max_gather_len);

namespace {

constexpr const char* kBF16TopKSchema =
    "top_k_per_row_prefill_bf16(Tensor logits, Tensor rowStarts, "
    "Tensor rowEnds, Tensor! indices, int numRows, int stride0, "
    "int stride1, int topK) -> ()";

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_ppu_C, m) {
  m.def(kBF16TopKSchema);
  m.def("deepseek_dequant_gather_vector(Tensor! out, Tensor cache, Tensor seq_lens, "
        "Tensor gather_lens, Tensor block_table, int page_size, int offset, int max_gather_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_ppu_C, CUDA, m) {
  m.impl("top_k_per_row_prefill_bf16", TORCH_BOX(&top_k_per_row_prefill_bf16));
  m.impl("deepseek_dequant_gather_vector", TORCH_BOX(&deepseek_dequant_gather_vector));
}

// Stable-ABI extensions are static-initializer registered; the Python module
// only needs to exist so `import vllm_sail._C` loads this library.
// (Mirrors vllm's csrc/core/registration.h REGISTER_EXTENSION macro.)
#include <Python.h>

extern "C" PyObject* PyInit__C() {
  static struct PyModuleDef module = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, 0, nullptr};
  return PyModule_Create(&module);
}
