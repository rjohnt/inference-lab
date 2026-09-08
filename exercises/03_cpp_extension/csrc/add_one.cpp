#include <torch/extension.h>

torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "add_one expects a CUDA tensor");
  return input + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one, "Add one to a CUDA tensor");
}
