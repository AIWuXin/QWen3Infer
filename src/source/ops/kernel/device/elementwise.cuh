//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_ELEMENTWISE_CUH
#define QWEN3INFER_ELEMENTWISE_CUH


#include <device_launch_parameters.h>

#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op = base::ElementWiseType::kElementAdd, typename T = float>
    void element_wise_kernel_device(
        const tensor::Tensor& input0,
        const tensor::Tensor& input1,
        tensor::Tensor& output0,
        void* stream
    );
}


#endif //QWEN3INFER_ELEMENTWISE_CUH