//
// Created by Administrator on 2026/3/16.
//

#ifndef QWEN3INFER_FILL_CUH
#define QWEN3INFER_FILL_CUH


#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<typename T>
    void fill_kernel_device(
        tensor::Tensor& input0,
        T value,
        int32_t dim,
        size_t count,
        void* stream
    );
}


#endif //QWEN3INFER_FILL_CUH