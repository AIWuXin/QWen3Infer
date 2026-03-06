//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_REDUCTION_HPP
#define QWEN3INFER_REDUCTION_HPP


#ifdef USE_OPENMP
    #include <omp.h>
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
        } else if constexpr (Op == base::ReductionType::kReductionAll) {
            return static_cast<T>(1);
        } else if constexpr (Op == base::ReductionType::kReductionAny) {
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
            return std::max(a, b);
        } else if constexpr (Op == base::ReductionType::kReduceMin) {
            return std::min(a, b);
        } else if constexpr (Op == base::ReductionType::kReductionAll) {
            // 逻辑与：非零视为 true
            return static_cast<bool>(a) && static_cast<bool>(b);
        } else if constexpr (Op == base::ReductionType::kReductionAny) {
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
        CHECK_EQ(output0.dims() == input0.dims(), true);

        #ifdef USE_OPENMP
            const T* __restrict in_ptr = input0.ptr<T>();
            T* __restrict out_ptr = output0.ptr<T>();
        #else
        #endif
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_dim_kernel_host(
        const tensor::Tensor& input0,
        tensor::Tensor& output0,
        [[maybe_unused]] int32_t dim,
        void* stream
    ) {

    }
}


#endif //QWEN3INFER_REDUCTION_HPP