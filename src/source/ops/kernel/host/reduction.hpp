//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_REDUCTION_HPP
#define QWEN3INFER_REDUCTION_HPP


#ifdef USE_OPENMP
#endif

#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ReductionType Op, typename T>
    T get_reduction_init_value() {
        if constexpr (Op == base::ReductionType::kReduceSum) {
            return static_cast<T>(0);
        } else if constexpr (
            Op == base::ReductionType::kReduceMean
        ) {
            return static_cast<T>(0);
        } else if constexpr (Op == base::ReductionType::kReduceMax) {
            return std::numeric_limits<T>::lowest();
        } else if constexpr (Op == base::ReductionType::kReduceMin) {
            return std::numeric_limits<T>::max();
        } else if constexpr (Op == base::ReductionType::kReduceAll) {
            return static_cast<T>(1);
        } else if constexpr (Op == base::ReductionType::kReduceAny) {
            return static_cast<T>(0);
        } else {
            LOG(FATAL) << "unsupported reduction type";
            throw std::runtime_error("unsupported reduction type");
        }
    }

    template<base::ReductionType Op, typename T>
    T reduction_op(T a, T b) {
        if constexpr (Op == base::ReductionType::kReduceSum) {
            return a + b;
        } else if constexpr (Op == base::ReductionType::kReduceMean) {
            // Mean 的规约阶段先累加，最终除法在 kernel 出口或 Host 端执行 (sum / N)
            return a + b;
        } else if constexpr (Op == base::ReductionType::kReduceMax) {
            return (a > b) ? a : b;
        } else if constexpr (Op == base::ReductionType::kReduceMin) {
            return (a < b) ? a : b;
        } else if constexpr (Op == base::ReductionType::kReduceAll) {
            // 逻辑与：非零视为 true
            return static_cast<bool>(a) && static_cast<bool>(b);
        } else if constexpr (Op == base::ReductionType::kReduceAny) {
            // 逻辑或：非零视为 true
            return static_cast<bool>(a) || static_cast<bool>(b);
        } else {
            LOG(FATAL) << "unsupported reduction type";
            throw std::runtime_error("unsupported reduction type");
        }
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_kernel_host(
        const tensor::Tensor& input0,
        tensor::Tensor& output0,
        [[maybe_unused]] int32_t dim,
        void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);
        CHECK_EQ(output0.ndims() == 1, true);
        CHECK_EQ(output0.dim(0) == 1, true);

        #ifdef USE_OPENMP
        const size_t n = input0.size();
        const T* __restrict in_ptr = input0.ptr<T>();
        T* __restrict out_ptr = output0.ptr<T>();

        T result = get_reduction_init_value<Op, T>();

        if (n >= 256*256) {
        #define OMP_REDUCTION_LOOP(op, out, in)              \
            for (size_t idx = 0; idx < n; ++idx) {           \
                result = reduction_op<op, T>(out, in[idx]);  \
            }                                                \

            if constexpr (
                Op == base::ReductionType::kReduceSum ||
                Op == base::ReductionType::kReduceMean
            ) {
                #pragma omp parallel for reduction(+:result)  \
                    schedule(static, 1024) default(none)      \
                    shared(n, in_ptr, out_ptr)
                OMP_REDUCTION_LOOP(Op, result, in_ptr);
            } else if constexpr (
                Op == base::ReductionType::kReduceMax
            ) {
                #pragma omp parallel for reduction(max:result)  \
                    schedule(static, 1024) default(none)        \
                    shared(n, in_ptr, out_ptr)
                OMP_REDUCTION_LOOP(Op, result, in_ptr);
            } else if constexpr (
                Op == base::ReductionType::kReduceMin
            ) {
                #pragma omp parallel for reduction(min:result)  \
                    schedule(static, 1024) default(none)        \
                    shared(n, in_ptr, out_ptr)
                OMP_REDUCTION_LOOP(Op, result, in_ptr);
            } else if constexpr (
                Op == base::ReductionType::kReduceAll
            ) {
                bool local_res = static_cast<bool>(result);
                #pragma omp parallel for reduction(&&:local_res)  \
                    schedule(static, 1024) default(none)          \
                    shared(n, in_ptr, out_ptr)
                for (size_t idx = 0; idx < n; ++idx) {
                    local_res = reduction_op<Op, T>(local_res, in_ptr[idx]);
                }
                result = static_cast<T>(local_res);
            } else if constexpr (
                Op == base::ReductionType::kReduceAny
            ) {
                bool local_res = static_cast<bool>(result);
                #pragma omp parallel for reduction(||:local_res)  \
                    schedule(static, 1024) default(none)          \
                    shared(n, in_ptr, out_ptr)
                for (size_t idx = 0; idx < n; ++idx) {
                    local_res = reduction_op<Op, T>(local_res, in_ptr[idx]);
                }
                result = static_cast<T>(local_res);
            }

            if constexpr (Op == base::ReductionType::kReduceMean) {
                result = result / static_cast<T>(n);
            }

            out_ptr[0] = result;
        #undef OMP_REDUCTION_LOOP
            return;
        }
        #else
        const size_t n = input0.size();
        const T* in_ptr = input0.ptr<T>();
        T* out_ptr = output0.ptr<T>();

        T result = get_reduction_init_value<Op, T>();
        #endif

        for (size_t idx = 0; idx < n; ++idx) {
            result = reduction_op<Op, T>(result, in_ptr[idx]);
        }
        if constexpr (Op == base::ReductionType::kReduceMean) {
            result = result / static_cast<T>(n);
        }
        out_ptr[0] = result;
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_dim_kernel_host(
        const tensor::Tensor& input0,
        tensor::Tensor& output0,
        int32_t dim,
        void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);

        if (dim < 0) {
            dim += static_cast<int32_t>(input0.ndims());
        }
        if (dim < 0 || dim >= input0.ndims()) {
            LOG(ERROR) << "dim " << dim << " is out of bounds";
            throw std::runtime_error("dim out of bounds");
        }

        const auto input_dims = input0.dims();
        const auto input_strides = input0.strides();
        const auto reduce_dim_size = static_cast<int32_t>(input_dims[dim]);

        // 计算外层和内层大小
        const size_t outer_size = tensor::reduce_dimension(
            input_dims.begin(), input_dims.begin() + dim, size_t{1}
        );
        const size_t inner_size = tensor::reduce_dimension(
            input_dims.begin() + dim + 1, input_dims.end(), size_t{1}
        );

        const T* in_ptr = input0.ptr<T>();
        T* out_ptr = output0.ptr<T>();

        #ifdef USE_OPENMP
        if (outer_size * inner_size >= 256) {
            #pragma omp parallel for collapse(2) schedule(static, 64) \
                default(none) shared(outer_size, inner_size, reduce_dim_size, \
                in_ptr, out_ptr, input_strides, dim)
            for (size_t i = 0; i < outer_size; ++i) {
                for (size_t j = 0; j < inner_size; ++j) {
                    T result = get_reduction_init_value<Op, T>();
                    for (int32_t k = 0; k < reduce_dim_size; ++k) {
                        const size_t in_idx = i * input_strides[0]
                            + k * input_strides[dim] + j;
                        result = reduction_op<Op, T>(result, in_ptr[in_idx]);
                    }
                    if constexpr (Op == base::ReductionType::kReduceMean) {
                        result = result / static_cast<T>(reduce_dim_size);
                    }
                    // 输出索引：i * inner_size + j
                    const size_t out_idx = i * inner_size + j;
                    out_ptr[out_idx] = result;
                }
            }
            return;
        }
        #endif

        for (size_t i = 0; i < outer_size; ++i) {
            for (size_t j = 0; j < inner_size; ++j) {
                T result = get_reduction_init_value<Op, T>();
                for (int32_t k = 0; k < reduce_dim_size; ++k) {
                    const size_t in_idx = i * input_strides[0]
                        + k * input_strides[dim] + j;
                    result = reduction_op<Op, T>(result, in_ptr[in_idx]);
                }
                if constexpr (Op == base::ReductionType::kReduceMean) {
                    result = result / static_cast<T>(reduce_dim_size);
                }
                const size_t out_idx = i * inner_size + j;
                out_ptr[out_idx] = result;
            }
        }
    }

}


#endif //QWEN3INFER_REDUCTION_HPP