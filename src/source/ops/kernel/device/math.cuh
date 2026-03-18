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
            return static_cast<T>(0);
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

    template<base::ReductionType Op, typename T>
    __device__ __forceinline__ T warp_reduce_op(T val) {
    #pragma unroll
        for (int offset = 32 >> 1; offset > 0; offset >>= 1) {
            T other = __shfl_down_sync(0xFFFFFFFF, val, offset);
            val = reduction_op<Op, T>(other, val);
        }

        return val;  // 只有一个warp中0号线程是正确答案
    }

    template<base::ReductionType OP>
    __device__ __forceinline__ void reduction_kernel_cu(
        const size_t size,
        const float* __restrict in0,
        float* __restrict out0,
        const float div_number
    ) {
        const size_t tid = threadIdx.x;
        const size_t global_idx = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t total_size_per_block = blockDim.x * gridDim.x;
        float local_sum = get_reduction_init_value_cu<OP, float>();
        constexpr  size_t warp_size = 32;
        const size_t idx_in_warp = threadIdx.x % warp_size;
        const size_t warp_idx = threadIdx.x / warp_size;
        __shared__ float warp_sum[warp_size];

        const size_t vec_size = size / 4;
        const auto vec_in0 = reinterpret_cast<const float4*>(in0);
        for (size_t idx = global_idx; idx < vec_size; idx += total_size_per_block) {
            float4 vec_data = vec_in0[idx];
            local_sum = reduction_op<OP, float>(local_sum, vec_data.x);
            local_sum = reduction_op<OP, float>(local_sum, vec_data.y);
            local_sum = reduction_op<OP, float>(local_sum, vec_data.z);
            local_sum = reduction_op<OP, float>(local_sum, vec_data.w);
        }
        const int remainder_start = vec_size * 4;
        for (int idx = remainder_start + global_idx; idx < size; idx += total_size_per_block) {
            local_sum = reduction_op<OP, float>(local_sum, in0[idx]);
        }  // step1: 全局求和

        local_sum = warp_reduce_op<OP, float>(local_sum);
        if (idx_in_warp == 0) {
            warp_sum[warp_idx] = local_sum;
        }
        __syncthreads();  // step2: warp间规约

        if (warp_idx == 0) {
            local_sum = idx_in_warp < blockDim.x / warp_size ?
                warp_sum[idx_in_warp] : get_reduction_init_value_cu<OP, float>();
            local_sum = warp_reduce_op<OP, float>(local_sum);

            if (tid == 0) {
                if constexpr (OP == base::ReductionType::kReduceMean) {
                    local_sum = local_sum / div_number;
                }
                out0[blockIdx.x] = local_sum;  // 每个 block 写一个值
            }
        }
    }


    template<base::ReductionType Op>
    __device__ __forceinline__ void reduction_dim_kernel_cu(
        const float* __restrict in0,
        float* __restrict out0,
        const size_t outer_size,
        const size_t reduce_dim_size,
        const size_t inner_size,
        const size_t stride_outer,  // input_strides[0]
        const size_t stride_reduce, // input_strides[dim]
        const float div_number      // for Mean
    ) {
        const size_t out_idx = blockIdx.x;
        const size_t i = out_idx / inner_size;
        const size_t j = out_idx % inner_size;
        const size_t tid = threadIdx.x;

        float local_val = get_reduction_init_value_cu<Op, float>();

        // 计算当前 (i, j) 在 reduce_dim=0 位置的起始地址
        const float* row_ptr = in0 + i * stride_outer + j;

        // 只有当 stride_reduce == 1 时（连续内存），才能用 float4 向量化
        if (stride_reduce == 1) {
            // ===== 向量化路径 =====
            const size_t vec_size = reduce_dim_size / 4;
            const auto vec_row_ptr = reinterpret_cast<const float4*>(row_ptr);

            // 处理 4 个一组的数据
            for (size_t k = tid; k < vec_size; k += blockDim.x) {
                float4 vec_data = vec_row_ptr[k];
                local_val = reduction_op<Op, float>(local_val, vec_data.x);
                local_val = reduction_op<Op, float>(local_val, vec_data.y);
                local_val = reduction_op<Op, float>(local_val, vec_data.z);
                local_val = reduction_op<Op, float>(local_val, vec_data.w);
            }

            // 处理剩余部分（非4对齐的尾部）
            const size_t remainder_start = vec_size * 4;
            for (size_t k = remainder_start + tid; k < reduce_dim_size; k += blockDim.x) {
                local_val = reduction_op<Op, float>(local_val, row_ptr[k]);
            }
        } else {
            // ===== 非连续内存路径（无法向量化） =====
            for (size_t k = tid; k < reduce_dim_size; k += blockDim.x) {
                const size_t in_idx = i * stride_outer + k * stride_reduce + j;
                local_val = reduction_op<Op, float>(local_val, in0[in_idx]);
            }
        }

        local_val = warp_reduce_op<Op, float>(local_val);

        __shared__ float warp_sums[32];
        if (tid % 32 == 0) {
            warp_sums[tid / 32] = local_val;
        }
        __syncthreads();

        if (tid < 32) {
            local_val = (tid < blockDim.x / 32) ? warp_sums[tid]
                                                : get_reduction_init_value_cu<Op, float>();
            local_val = warp_reduce_op<Op, float>(local_val);

            if (tid == 0) {
                if constexpr (Op == base::ReductionType::kReduceMean) {
                    local_val /= div_number;
                }
                out0[out_idx] = local_val;
            }
        }
    }

    __device__ __forceinline__ void fill_kernel_cu(
        float* __restrict in0,
        const size_t count,
        const float value
    ) {
        const size_t global_idx = threadIdx.x + blockDim.x * blockIdx.x;
        const size_t total_threads = blockDim.x * gridDim.x;

        // ===== 向量化版本：每次处理 4 个 float =====
        const size_t vec_count = count / 4;
        const float4 val4 = make_float4(value, value, value, value);
        auto vec_ptr = reinterpret_cast<float4*>(in0);

        // 向量化填充主体
        for (size_t i = global_idx; i < vec_count; i += total_threads) {
            vec_ptr[i] = val4;
        }

        // 处理尾部（非4对齐的部分）
        const size_t scalar_start = vec_count * 4;
        for (size_t i = scalar_start + global_idx; i < count; i += total_threads) {
            in0[i] = value;
        }
    }

    __device__ __forceinline__ void fill_dim_kernel_cu(
        float* __restrict in0,
        const size_t count,
        const float value,
        const size_t outer_size,
        const size_t reduce_dim_size,
        const size_t inner_size,
        const size_t stride_outer,  // input_strides[0]
        const size_t stride_reduce // input_strides[dim]
    ) {
        // 每个 block 处理一个 (i, j) 位置，填充 k=0 到 k=count-1
        const size_t out_idx = blockIdx.x;
        const size_t i = out_idx / inner_size;      // 外层索引
        const size_t j = out_idx % inner_size;      // 内层索引
        const size_t tid = threadIdx.x;

        // 计算实际填充数量（不超过 reduce_dim_size）
        const size_t fill_count = (count < reduce_dim_size) ? count : reduce_dim_size;

        if (stride_reduce == 1) {
            // ===== 连续内存路径：可向量化 =====
            // 计算当前 (i, j) 在 k=0 位置的起始地址
            float* row_ptr = in0 + i * stride_outer + j;

            // 使用 float4 向量化，一次写 4 个元素
            const size_t vec_size = fill_count / 4;
            float4 val4 = make_float4(value, value, value, value);
            auto vec_row_ptr = reinterpret_cast<float4*>(row_ptr);

            // 向量化填充
            for (size_t k = tid; k < vec_size; k += blockDim.x) {
                vec_row_ptr[k] = val4;
            }

            // 处理剩余部分（非4对齐的尾部）
            const size_t remainder_start = vec_size * 4;
            for (size_t k = remainder_start + tid; k < fill_count; k += blockDim.x) {
                row_ptr[k] = value;
            }
        } else {
            // ===== 非连续内存路径：按索引计算 =====
            // 计算基地址：i * stride_outer + j
            const size_t base_idx = i * stride_outer + j;

            for (size_t k = tid; k < fill_count; k += blockDim.x) {
                // in_idx = i * stride_outer + k * stride_reduce + j
                const size_t in_idx = base_idx + k * stride_reduce;
                in0[in_idx] = value;
            }
        }
    }
}


#endif //QWEN3INFER_MATH_CUH