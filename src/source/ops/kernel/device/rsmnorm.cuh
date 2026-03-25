//
// Created by Administrator on 2026/3/23.
//

#ifndef QWEN3INFER_RSMNORM_CUH
#define QWEN3INFER_RSMNORM_CUH


#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<typename T = float>
    void rms_norm_kernel_device(
        const tensor::Tensor& input,
        const tensor::Tensor& gamma,
        tensor::Tensor& output,
        const size_t num_rows,
        const size_t hidden_dim,
        const float eps,
        void* stream
    );
}


#endif //QWEN3INFER_RSMNORM_CUH