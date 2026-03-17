//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_REDUCTION_CUH
#define QWEN3INFER_REDUCTION_CUH


#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_dim_kernel_device(
        const tensor::Tensor& input0,
        tensor::Tensor& output0,
        [[maybe_unused]] int32_t dim,
        void* stream
    );


    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_kernel_device(
        const tensor::Tensor& input0,
        tensor::Tensor& output0,
        [[maybe_unused]] int32_t dim,
        void* stream
    );
}


#endif //QWEN3INFER_REDUCTION_CUH