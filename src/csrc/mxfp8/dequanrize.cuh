#include "c10/core/ScalarType.h"
#include "c10/core/TensorOptions.h"
#include "cute/config.hpp"
#include "cute/pointer.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <cassert>
#include <cstdint>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/tensor_impl.hpp>
#include <cutlass/bfloat16.h>
#include <sys/stat.h>
#include <thrust/host_vector.h>
#include <torch/extension.h>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 750))
#include "cute/arch/copy_sm75.hpp"
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ <= 890))
#include "cute/arch/copy_sm80.hpp"
#else
#include "cute/arch/copy_sm90.hpp"
#endif
#pragma once


namespace mase_cuda {
namespace mxfp8_E4M3 {
namespace dequantize {
template <class TypeX, class TypeScale>
__host__ void dequantize1d_host(TypeX const *x, const int M, TypeScale const *scales, const int group_size,
                                cutlass::bfloat16_t *y) {
    assert(M % group_size == 0);

    uint8_t const *x_raw_uint8 = reinterpret_cast<uint8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    uint8_t const bias = 0x7;
    uint8_t const final_bias = 0x7F-bias;

    const int num_groups = M / group_size;//get the number of groups

    thrust::host_vector<uint8_t> hX(x_raw_uint8, x_raw_uint8 + M);
    thrust::host_vector<uint8_t> hScales(scales_raw_uint8, scales_raw_uint8 + num_groups);

    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
        auto exp = static_cast<uint16_t>(hX[i] & 0x78) >> 3;
        auto frac = static_cast<uint16_t>(hX[i] & 0x07) << 4;

        auto scales = static_cast<uint16_t>(hScales[i / group_size]);
        auto result = exp + scales + final_bias;
        auto exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7;

        auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac);
        y[i] = out;
    }
}
} // namespace dequantize
} // namespace mxfp8_E4M3

namespace mxfp8_E5M2 {
namespace dequantize {
template <class TypeX, class TypeScale>
__host__ void dequantize1d_host(TypeX const *x, const int M, TypeScale const *scales, const int group_size,
                                cutlass::bfloat16_t *y) {
    assert(M % group_size == 0);

    uint8_t const *x_raw_uint8 = reinterpret_cast<uint8_t const *>(x);
    uint8_t const *scales_raw_uint8 = reinterpret_cast<uint8_t const *>(scales);
    uint8_t const bias = 0xF;
    uint8_t const final_bias = 0x7F-bias;

    const int num_groups = M / group_size;//get the number of groups

    thrust::host_vector<uint8_t> hX(x_raw_uint8, x_raw_uint8 + M);
    thrust::host_vector<uint8_t> hScales(scales_raw_uint8, scales_raw_uint8 + num_groups);

    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
        auto exp = static_cast<uint16_t>(hX[i] & 0x7c) >> 2;
        auto frac = static_cast<uint16_t>(hX[i] & 0x03) << 5;

        auto scales = static_cast<uint16_t>(hScales[i / group_size]);
        auto result = exp + scales + final_bias;
        auto exp = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7;

        auto out = cutlass::bfloat16_t::bitcast(sign | exp | frac);
        y[i] = out;
    }
}
} // namespace dequantize
} // namespace mxfp8_E5M2
} // namespace mase_cuda