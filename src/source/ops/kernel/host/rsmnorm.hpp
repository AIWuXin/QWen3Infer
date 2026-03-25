//
// Created by Administrator on 2026/3/23.
//

#ifndef QWEN3INFER_RSMNORM_HPP
#define QWEN3INFER_RSMNORM_HPP

#include <cmath>

#include "../../../../include/tensor/tensorbase.h"

namespace qwi::ops::kernel {
    template<typename T = float>
    void rms_norm_kernel_host(
        const tensor::Tensor& input,
        const tensor::Tensor& gamma,
        tensor::Tensor& output,
        const size_t num_rows,
        const size_t hidden_dim,
        const float eps,
        [[maybe_unused]] void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input.is_empty(), false);
        CHECK_EQ(gamma.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);
        CHECK_EQ(input.size(), num_rows * hidden_dim);
        CHECK_EQ(gamma.size(), hidden_dim);
        CHECK_EQ(output.size(), num_rows * hidden_dim);

        const T* __restrict in_ptr = input.ptr<T>();
        const T* __restrict gamma_ptr = gamma.ptr<T>();
        T* __restrict out_ptr = output.ptr<T>();

        #ifdef USE_OPENMP
        // 根据数据量选择合适的调度策略
        #pragma omp parallel for schedule(dynamic) default(none) \
            shared(num_rows, hidden_dim, in_ptr, gamma_ptr, out_ptr, eps)
        for (size_t row = 0; row < num_rows; ++row) {
            const T* __restrict row_in = in_ptr + row * hidden_dim;
            T* __restrict row_out = out_ptr + row * hidden_dim;

            // 计算平方和（循环展开 4 倍）
            T sum_squares = static_cast<T>(0);
            size_t i = 0;

            // 主循环：每次处理 4 个元素
            #pragma clang loop vectorize(enable) interleave(enable)
            for (; i + 3 < hidden_dim; i += 4) {
                T v0 = row_in[i];
                T v1 = row_in[i + 1];
                T v2 = row_in[i + 2];
                T v3 = row_in[i + 3];
                sum_squares += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }

            // 处理剩余元素
            for (; i < hidden_dim; ++i) {
                T v = row_in[i];
                sum_squares += v * v;
            }

            // 使用 rsqrt (倒数平方根) 替代 sqrt + 除法，更快
            const T inv_rms = static_cast<T>(1) / std::sqrt(
                sum_squares / static_cast<T>(hidden_dim) + static_cast<T>(eps)
            );

            // 归一化并应用 gamma（循环展开）
            i = 0;
            #pragma clang loop vectorize(enable) interleave(enable)
            for (; i + 3 < hidden_dim; i += 4) {
                row_out[i]     = row_in[i]     * inv_rms * gamma_ptr[i];
                row_out[i + 1] = row_in[i + 1] * inv_rms * gamma_ptr[i + 1];
                row_out[i + 2] = row_in[i + 2] * inv_rms * gamma_ptr[i + 2];
                row_out[i + 3] = row_in[i + 3] * inv_rms * gamma_ptr[i + 3];
            }
            for (; i < hidden_dim; ++i) {
                row_out[i] = row_in[i] * inv_rms * gamma_ptr[i];
            }
        }
        return;
        #endif

        // 非 OpenMP 版本
        for (size_t row = 0; row < num_rows; ++row) {
            const T* __restrict row_in = in_ptr + row * hidden_dim;
            T* __restrict row_out = out_ptr + row * hidden_dim;

            // 计算平方和（循环展开）
            T sum_squares = static_cast<T>(0);
            size_t i = 0;
            for (; i + 3 < hidden_dim; i += 4) {
                T v0 = row_in[i];
                T v1 = row_in[i + 1];
                T v2 = row_in[i + 2];
                T v3 = row_in[i + 3];
                sum_squares += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
            }
            for (; i < hidden_dim; ++i) {
                T v = row_in[i];
                sum_squares += v * v;
            }

            const T inv_rms = static_cast<T>(1) / std::sqrt(
                sum_squares / static_cast<T>(hidden_dim) + static_cast<T>(eps)
            );

            // 归一化并应用 gamma
            i = 0;
            for (; i + 3 < hidden_dim; i += 4) {
                row_out[i]     = row_in[i]     * inv_rms * gamma_ptr[i];
                row_out[i + 1] = row_in[i + 1] * inv_rms * gamma_ptr[i + 1];
                row_out[i + 2] = row_in[i + 2] * inv_rms * gamma_ptr[i + 2];
                row_out[i + 3] = row_in[i + 3] * inv_rms * gamma_ptr[i + 3];
            }
            for (; i < hidden_dim; ++i) {
                row_out[i] = row_in[i] * inv_rms * gamma_ptr[i];
            }
        }
    }
}


#endif //QWEN3INFER_RSMNORM_HPP