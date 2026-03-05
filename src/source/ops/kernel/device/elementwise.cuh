//
// Created by Administrator on 2026/3/5.
//

#ifndef QWEN3INFER_ELEMENTWISE_CUH
#define QWEN3INFER_ELEMENTWISE_CUH


#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op = base::ElementWiseType::kElementAdd, typename T = float>
    void element_wise_kernel_device(
        const tensor::Tensor& input0,
        const tensor::Tensor& input1,
        tensor::Tensor& output0,
        void* stream
    ) {
        LOG(FATAL) << "CUDA kernel not implemented yet!";
        throw std::runtime_error("CUDA kernel not implemented yet!");
    }
}


#endif //QWEN3INFER_ELEMENTWISE_CUH