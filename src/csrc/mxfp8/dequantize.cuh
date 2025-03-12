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
namespace mxfp8 {
namespace dequantize {
template <class TypeX, class TypeScale>
__host__ void dequantize1d_host(TypeX const *x, const int M, TypeScale const *scales, const int group_size,
                                cutlass::bfloat16_t *y) {
    assert(M % group_size == 0);

    uint8_t const *x_raw_uint8 = reinterpret_cast<uint8_t const *>(x);
    int8_t const *scales_raw_int8 = reinterpret_cast<int8_t const *>(scales);
    int8_t const bias = 0x7;
    int8_t const final_bias = -bias;

    const int num_groups = M / group_size;//get the number of groups

    thrust::host_vector<uint8_t> hX(x_raw_uint8, x_raw_uint8 + M);
    thrust::host_vector<uint8_t> hScales(scales_raw_int8, scales_raw_int8 + num_groups);

    for (int i = 0; i < M; ++i) {
        auto sign = static_cast<uint16_t>(hX[i] & 0x80) << 8;
        auto exp = static_cast<uint16_t>(hX[i] & 0x78) >> 3;
        auto frac = static_cast<uint16_t>(hX[i] & 0x07) << 4;

        auto scales = static_cast<uint16_t>(hScales[i / group_size]);
        auto result = exp + scales + final_bias;
        auto exp_out = ((result & 0xFF) | ((result >> 8) * 0xFF)) << 7;

        auto out = cutlass::bfloat16_t::bitcast(sign | exp_out | frac);
        y[i] = out;
    }
}

torch::Tensor dequantize1d(torch::Tensor x, torch::Tensor scales, const int group_size) {
    using namespace cute;

    if (group_size <= 0) {
        throw std::invalid_argument("group_size must be positive");
    }
    const int m = x.numel() * x.itemsize() / sizeof(uint8_t);
    const int num_groups = m / group_size;
    if (m % group_size != 0) {
        throw std::invalid_argument("m must be divisible by group_size");
    }
    if (x.device() != scales.device()) {
        throw std::invalid_argument("x and scales must be on the same device");
    }

    auto y_options = torch::TensorOptions().device(x.device()).dtype(torch::kBFloat16);
    auto y = torch::empty({m}, y_options);

    auto _x = x.contiguous();
    auto _scales = scales.contiguous();

    auto x_ptr = _x.const_data_ptr();
    auto scales_ptr = _scales.const_data_ptr();
    cutlass::bfloat16_t *y_ptr = reinterpret_cast<cutlass::bfloat16_t *>(y.data_ptr());


    if (_x.device().is_cpu()) 
        dequantize1d_host(x_ptr, m, scales_ptr, group_size, y_ptr);
    // } else if (_x.device().is_cuda()) {
    //     auto shape_x = make_shape(m);
    //     auto stride_x = make_stride(Int<1>{});
    //     auto shape_scale = make_shape(num_groups);
    //     auto stride_scale = make_stride(Int<1>{});
    //     auto group_tiler = make_shape(group_size);
    //     auto device = x.get_device();
    //     cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);

    //     if (group_size <= 8) {
    //         auto BLK_M = Int<8>{};
    //         auto BLK_K = Int<128>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     } else if (group_size <= 16) {
    //         auto BLK_M = Int<16>{};
    //         auto BLK_K = Int<64>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     } else if (group_size <= 32) {
    //         auto BLK_M = Int<32>{};
    //         auto BLK_K = Int<32>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     } else if (group_size <= 64) {
    //         auto BLK_M = Int<64>{};
    //         auto BLK_K = Int<16>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     } else if (group_size <= 128) {
    //         auto BLK_M = Int<128>{};
    //         auto BLK_K = Int<8>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     } else {
    //         // larger group sizes
    //         auto BLK_M = Int<128>{};
    //         auto BLK_K = Int<8>{};
    //         auto thd_m = BLK_M;
    //         auto thd_k = BLK_K;
    //         auto cta_tiler = make_shape(BLK_M, BLK_K);
    //         auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    //         auto layout_sScale = make_layout(make_shape(BLK_K));
    //         auto layout_tX = make_layout(make_shape(thd_m, thd_k));
    //         dim3 dimBlock(size(layout_tX));
    //         dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    //         dequantize1d_device<<<dimGrid, dimBlock, 0, stream>>>(x_ptr, shape_x, stride_x, scales_ptr, shape_scale,
    //                                                               stride_scale, group_tiler, cta_tiler, layout_sX,
    //                                                               layout_sScale, layout_tX, y_ptr);
    //     }
    // // } else {
    //     throw std::invalid_argument("x must be on CPU or CUDA");
    // }

    y = y.reshape_as(x);
    return y;
}
} // namespace dequantize
} // namespace mxfp8_E4M3
} // namespace mase_cuda