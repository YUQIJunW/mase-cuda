#include "cute/int_tuple.hpp"
#include "mxfp8/dequantize.cuh"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/bfloat16.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char **argv) {
    using namespace cute;
    using namespace mase_cuda;
    using json = nlohmann::json;

    int m = 4096;
    int group_size = 128;
    bool is_random = false;

    if (argc > 1) {
        m = atoi(argv[1]);
    }

    if (argc > 2) {
        group_size = atoi(argv[2]);
    }
    if (!(m % group_size == 0)) {
        std::cerr << "m is not divisible by group_size" << std::endl;
        return 1;
    }
    const int num_groups = m / group_size;

    if (argc > 3) {
        is_random = bool(atoi(argv[3]));
    }

    std::cout << "Usage: " << argv[0] << " [m] [group_size] [is_random]" << std::endl;
    std::cout << "m=" << m << ", group_size=" << group_size << ", num_groups=" << num_groups
              << ", is_random=" << is_random << std::endl;

    // initialize data
    thrust::host_vector<uint8_t> x_h(m);
    thrust::host_vector<int8_t> scales_h(num_groups);
    thrust::host_vector<cutlass::bfloat16_t> y_h(m);
    thrust::host_vector<cutlass::bfloat16_t> y_ref_h(m);

    for (int i = 0; i < m; ++i) {
        if (is_random) {
            x_h[i] = (129 + i) % 256;
        } else {
            x_h[i] = rand() % 256;
        }
        y_h[i] = cutlass::bfloat16_t(0.0);
        y_ref_h[i] = cutlass::bfloat16_t(0.0);
        if (i % group_size == 0) {
            if (is_random) {
                scales_h[i / group_size] = rand() % 255; // avoid 256 (NaN)
            } else {
                scales_h[i / group_size] = (128 + i) % 256;
            }
        }
    }

    // CPU implementation
    mase_cuda::mxfp8::dequantize::dequantize1d_host(x_h.data(), m, scales_h.data(), group_size, y_ref_h.data());

    // GPU implementation
    auto BLK_M = Int<32>{};
    auto BLK_K = Int<32>{};
    auto thd_m = Int<32>{};
    auto thd_k = Int<32>{};

    thrust::device_vector<uint8_t> x_d = x_h;
    thrust::device_vector<uint8_t> scales_d = scales_h;
    thrust::device_vector<cutlass::bfloat16_t> y_d = y_h;

    auto shape_x = make_shape(m);
    auto stride_x = make_stride(Int<1>{});
    auto shape_scale = make_shape(num_groups);
    auto stride_scale = make_stride(Int<1>{});
    auto group_tiler = make_shape(group_size);

    auto cta_tiler = make_shape(BLK_M, BLK_K);
    auto layout_sX = make_layout(make_shape(BLK_M, BLK_K));
    auto layout_sScale = make_layout(make_shape(BLK_K));

    auto layout_tX = make_layout(make_shape(thd_m, thd_k));

    dim3 dimBlock(size(layout_tX));
    dim3 dimGrid(ceil_div(group_size, BLK_M), ceil_div(num_groups, BLK_K));
    mase_cuda::mxfp8::dequantize::dequantize1d_device<<<dimGrid, dimBlock, 0, 0>>>(
        x_d.data().get(), shape_x, stride_x, scales_d.data().get(), shape_scale, stride_scale, group_tiler, cta_tiler,
        layout_sX, layout_sScale, layout_tX, y_d.data().get());

    // copy y_d back to y_h
    y_h = y_d;

    bool passed = true;
    for (int i = 0; i < m; ++i) {
        if (y_h[i] != y_ref_h[i]) {
            std::cout << "Mismatch at index " << i << ", y_h[i]=" << y_h[i] << ", y_ref_h[i]=" << y_ref_h[i]
                      << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }
}