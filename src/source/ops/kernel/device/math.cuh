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

    template<base::ReductionType Op, typename T>
    __device__ __forceinline__ T get_reduction_init_value_cu() {
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
    __device__ __forceinline__ T reduction_op(T a, T b) {
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
            return static_cast<T>(0);
        }
    }

    template<base::ReductionType OP, const size_t ThreadNum>
    __device__ __forceinline__ void reduction_kernel_cu(
        const size_t size,
        const float* __restrict in0,
        float* __restrict out0
    ) {
        const size_t tid = threadIdx.x;
        const size_t global_idx = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t total_size_per_block = blockDim.x * gridDim.x;
        float local_sum = get_reduction_init_value_cu<OP, float>();
        __shared__ float block_local_sum[ThreadNum];

        for (int idx = 0; idx < size; idx += total_size_per_block) {
            local_sum = reduction_op<OP, float>(local_sum, in0[idx]);
        }
        block_local_sum[tid] = local_sum;
        __syncthreads();

        for (int reduce_idx = blockDim.x >> 2; reduce_idx > 0; reduce_idx >>= 1) {
            if (tid < reduce_idx) {
                block_local_sum[tid] = reduction_op<OP, float>(
                    block_local_sum[tid], block_local_sum[tid + reduce_idx]
                );
            }
            __syncthreads();
        }
    }
}


#endif //QWEN3INFER_MATH_CUH