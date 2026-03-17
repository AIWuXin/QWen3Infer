//
// Created by Administrator on 2026/3/16.
//

#ifndef QWEN3INFER_FILL_HPP
#define QWEN3INFER_FILL_HPP


#include "../../../../include/tensor/tensorbase.h"

namespace qwi::ops::kernel {
    /**
     * @brief 填充 Kernel (Host 版本)
     *
     * @param output0 输出张量
     * @param value 填充值
     * @param dim 指定维度，INT_MIN 表示全局填充（忽略维度限制）
     * @param count 填充的元素个数
     * @param stream CUDA 流指针（Host 版本未使用）
     *
     * 使用示例：
     *   // 全局填充 1000 个元素
     *   fill_kernel_host(output, 3.14f, INT_MIN, 1000, nullptr);
     *
     *   // 在第 0 维填充 100 个元素
     *   fill_kernel_host(output, 3.14f, 0, 100, nullptr);
     */
    template<typename T>
    void fill_kernel_host(
        tensor::Tensor& input0,
        T value,
        int32_t dim,
        const size_t count,
        void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK(dim < input0.ndims());
        CHECK(count < input0.size());

        if (dim == INT_MIN) {
            #ifdef USE_OPENMP
            T* __restrict out_ptr = input0.ptr<T>();
            if (count >= 256*256) {
                #pragma omp parallel for schedule(static, 1024) \
                    default(none) shared(out_ptr, value, count)
                for (int idx = 0; idx < count; ++idx) {
                    out_ptr[idx] = value;
                }
                return;
            }
            #else
            T* out_ptr = output0.ptr<T>();
            #endif

            for (size_t idx = 0; idx < count; ++idx) {
                out_ptr[idx] = value;
            }
            return;
        }

        if (dim < 0) {
            dim += static_cast<int32_t>(input0.ndims());
        }

        if (dim < 0 || dim >= static_cast<int32_t>(input0.ndims())) {
            LOG(ERROR) << "dim " << dim << " is out of bounds";
            throw std::runtime_error("dim out of bounds");
        }

        const auto dims = input0.dims();
        const auto strides = input0.strides();
        const size_t dim_size = dims[dim];

        const size_t outer_size = tensor::reduce_dimension(
            dims.begin(), dims.begin() + dim, size_t{1}
        );
        const size_t inner_size = tensor::reduce_dimension(
            dims.begin() + dim + 1, dims.end(), size_t{1}
        );

        #ifdef USE_OPENMP
        T* __restrict out_ptr = input0.ptr<T>();
        if (outer_size * inner_size >= 256) {
            #pragma omp parallel for collapse(2) schedule(static, 64) \
            default(none) shared(outer_size, inner_size, count, \
            out_ptr, strides, dim, value)
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t j = 0; j < inner_size; ++j) {
                    for (size_t k = 0; k < count; ++k) {
                        const size_t idx = i * strides[0] + k * strides[dim] + j;
                        out_ptr[idx] = value;
                    }
                }
            }
            return;
        }
        #else
        T* out_ptr = input0.ptr<T>();
        #endif

        for (size_t i = 0; i < outer_size; ++i) {
            for (size_t j = 0; j < inner_size; ++j) {
                for (size_t k = 0; k < count; ++k) {
                    const size_t idx = i * strides[0] + k * strides[dim] + j;
                    out_ptr[idx] = value;
                }
            }
        }
    }
}


#endif //QWEN3INFER_FILL_HPP