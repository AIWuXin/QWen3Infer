//
// Created by Administrator on 2026/3/5.
//

#ifndef QWEN3INFER_ELEMENTWISE_HPP
#define QWEN3INFER_ELEMENTWISE_HPP

#ifdef USE_OPENMP
    #include <omp.h>
#endif

#include <armadillo>
#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op = base::ElementWiseType::kElementAdd, typename T = float>
    void element_wise_kernel_host(
        const tensor::Tensor& input0,
        const tensor::Tensor& input1,
        tensor::Tensor& output0,
        void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);

        CHECK_EQ(input0.dims() == input1.dims(), true);
        CHECK_EQ(output0.dims() == input0.dims(), true);

        const T* in0_ptr = input0.ptr<T>();
        const T* in1_ptr = input1.ptr<T>();
        T* out0_ptr = output0.ptr<T>();
        const size_t n = input0.size();

        #ifdef USE_OPENMP
        if (n >= 4096) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                out0_ptr[i] = base::element_wise_op<Op>(in0_ptr[i], in1_ptr[i]);
            }
            return;
        }
        #endif

        for (size_t i = 0; i < n; ++i) {
            out0_ptr[i] = base::element_wise_op<Op>(in0_ptr[i], in1_ptr[i]);
        }
    }
}


#endif //QWEN3INFER_ELEMENTWISE_HPP