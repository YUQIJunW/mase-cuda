#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "./mxint/dequantize.cuh"
#include "./mxfp8/dequantize.cuh"

// refer to https://github.com/pybind/python_example/blob/master/src/main.cpp
// https://pytorch.org/tutorials/advanced/cpp_extension.html

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "This is a CUDA-accelerated PyTorch extension";
    auto m_mxint8 = m.def_submodule("mxint8", "OCP-MXINT8 module");
    auto m_mxfp8 = m.def_submodule("mxfp8", "OCP-MXFP8 module");
    m_mxint8.def("dequantize1d", &mase_cuda::mxint8::dequantize::dequantize1d, py::arg("x"), py::arg("scales"),
                 py::arg("group_size"));
    m_mxfp8.def("dequantize1d", &mase_cuda::mxfp8::dequantize::dequantize1d, py::arg("x"), py::arg("scales"),
                py::arg("group_size"));
}