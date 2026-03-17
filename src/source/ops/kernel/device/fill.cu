//
// Created by Administrator on 2026/3/16.
//


#include "fill.cuh"
#include "math.cuh"


namespace qwi::ops::kernel {
    template<typename T>
    void fill_kernel_device(
        tensor::Tensor& input0,
        T value,
        int32_t dim,
        size_t count,
        void* stream
    ) {

    }

    template void fill_kernel_device<float>(
        tensor::Tensor& input0,
        float value,
        int32_t dim,
        size_t count,
        void* stream
    );
}
