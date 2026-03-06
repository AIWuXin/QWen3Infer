//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_MATH_CUH
#define QWEN3INFER_MATH_CUH


#include <device_launch_parameters.h>

#include "../../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op>
    __device__ __forceinline__ void elementwise_kernel_cu(
        const size_t size, const float* __restrict in0,
        const float* __restrict in1, float* __restrict out0
    ) {
        auto in0_vec = reinterpret_cast<const float4*>(in0);
        auto in1_vec = reinterpret_cast<const float4*>(in1);
        auto out0_vec = reinterpret_cast<float4*>(out0);

        const size_t vec_size = size / 4;  // 处理 4 个一组

        for (
            size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
            tid < vec_size; tid += blockDim.x * gridDim.x
        ) {
            const float4 in_val0 = in0_vec[tid];
            const float4 in_val1 = in1_vec[tid];
            float4 out_val0 = out0_vec[tid];

            if constexpr (Op == base::ElementWiseType::kElementAdd) {
                out_val0.x = in_val0.x + in_val1.x;
                out_val0.y = in_val0.y + in_val1.y;
                out_val0.z = in_val0.z + in_val1.z;
                out_val0.w = in_val0.w + in_val1.w;
            } else if constexpr (
                Op == base::ElementWiseType::kElementSubtract
            ) {
                out_val0.x = in_val0.x - in_val1.x;
                out_val0.y = in_val0.y - in_val1.y;
                out_val0.z = in_val0.z - in_val1.z;
                out_val0.w = in_val0.w - in_val1.w;
            } else if constexpr (
                Op == base::ElementWiseType::kElementMultiply
            ) {
                out_val0.x = in_val0.x * in_val1.x;
                out_val0.y = in_val0.y * in_val1.y;
                out_val0.z = in_val0.z * in_val1.z;
                out_val0.w = in_val0.w * in_val1.w;
            } else if constexpr (
                Op == base::ElementWiseType::kElementDivide
            ) {
                out_val0.x = __fdividef(in_val0.x, in_val1.x);
                out_val0.y = __fdividef(in_val0.y, in_val1.y);
                out_val0.z = __fdividef(in_val0.z, in_val1.z);
                out_val0.w = __fdividef(in_val0.w, in_val1.w);
            } else {
            #ifndef NDEBUG
                printf("CUDA kernel: unknown op %d, defaulting to nothing to do!\n", static_cast<int>(Op));
            #endif
            }
            out0_vec[tid] = out_val0;
        }

        for (size_t idx = vec_size * 4 + threadIdx.x; idx < size; idx += blockDim.x) {
            if constexpr (Op == base::ElementWiseType::kElementAdd) {
                out0[idx] = in0[idx] + in1[idx];
            } else if constexpr (Op == base::ElementWiseType::kElementSubtract) {
                out0[idx] = in0[idx] - in1[idx];
            } else if constexpr (Op == base::ElementWiseType::kElementMultiply) {
                out0[idx] = in0[idx] * in1[idx];
            } else if constexpr (Op == base::ElementWiseType::kElementDivide) {
                out0[idx] = in0[idx] / in1[idx];
            } else {
                #ifndef NDEBUG
                    printf("CUDA kernel: unknown op %d, defaulting to nothing to do!\n", static_cast<int>(Op));
                #endif
            }
        }
    }
}


#endif //QWEN3INFER_MATH_CUH