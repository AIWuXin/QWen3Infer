//
// Created by Administrator on 2026/3/23.
//


#include "math.cuh"
#include "rsmnorm.cuh"


namespace qwi::ops::kernel {
    __global__ void rms_norm_kernel_cu_launch(
        const float* __restrict input,
        const float* __restrict gamma,
        float* __restrict output,
        const size_t hidden_dim,
        const float eps
    ) {
        const size_t row = blockIdx.x;
        const size_t tid = threadIdx.x;
        const size_t lane_id = tid % 32;
        const size_t warp_id = tid / 32;

        // 输入/输出指针偏移到当前行
        const float* row_input = input + row * hidden_dim;
        float* row_output = output + row * hidden_dim;

        // 向量化指针
        const auto* input4 = reinterpret_cast<const float4*>(row_input);
        const auto* gamma4 = reinterpret_cast<const float4*>(gamma);
        auto* output4 = reinterpret_cast<float4*>(row_output);

        const size_t vec_hidden = hidden_dim / 4;

        __shared__ float warp_sum[32];  // 最多 1024 线程 = 32 warp
        __shared__ float inv_rms_shared;

        // ==================== Stage 1: 计算局部平方和 ====================
        float local_sum = 0.0f;

        // 向量化处理
        #pragma unroll 4
        for (size_t i = tid; i < vec_hidden; i += blockDim.x) {
            const auto [x, y, z, w] = input4[i];
            local_sum += x * x + y * y + z * z + w * w;
        }

        // 处理尾部（非4对齐的部分）
        const size_t remainder_start = vec_hidden * 4;
        for (size_t i = remainder_start + tid; i < hidden_dim; i += blockDim.x) {
            float val = row_input[i];
            local_sum += val * val;
        }

        // ==================== Stage 2: Block 级规约 ====================
        // Warp 内规约
        local_sum = warp_reduce_op<base::ReductionType::kReduceSum, float>(local_sum);

        // 每个 warp 的 leader 写入 shared memory
        if (lane_id == 0) {
            warp_sum[warp_id] = local_sum;
        }
        __syncthreads();

        // 第一个 warp 再次规约所有 warp 的结果
        if (warp_id == 0) {
            local_sum = (lane_id < (blockDim.x + 31) / 32) ? warp_sum[lane_id] : 0.0f;
            local_sum = warp_reduce_op<base::ReductionType::kReduceSum, float>(local_sum);

            // 计算 inv_rms 并广播
            if (lane_id == 0) {
                inv_rms_shared = rsqrtf(local_sum / static_cast<float>(hidden_dim) + eps);
            }
        }
        __syncthreads();

        float inv_rms = inv_rms_shared;

        // ==================== Stage 3: 归一化与 gamma 缩放 ====================
        // 使用 __ldg 读取 gamma（只读，可缓存）
        #pragma unroll 4
        for (size_t i = tid; i < vec_hidden; i += blockDim.x) {
            float4 val = input4[i];
            float4 g = __ldg(&gamma4[i]);

            val.x = val.x * inv_rms * g.x;
            val.y = val.y * inv_rms * g.y;
            val.z = val.z * inv_rms * g.z;
            val.w = val.w * inv_rms * g.w;

            output4[i] = val;
        }

        // 处理尾部
        for (size_t i = remainder_start + tid; i < hidden_dim; i += blockDim.x) {
            float val = row_input[i];
            float g = __ldg(&gamma[i]);
            row_output[i] = val * inv_rms * g;
        }
    }

    template<typename T>
    void rms_norm_kernel_device(
        const tensor::Tensor& input,
        const tensor::Tensor& gamma,
        tensor::Tensor& output,
        const size_t num_rows,
        const size_t hidden_dim,
        const float eps,
        void* stream
    ) {
        CHECK_EQ(input.is_empty(), false);
        CHECK_EQ(gamma.is_empty(), false);
        CHECK_EQ(output.is_empty(), false);
        CHECK_EQ(input.size(), num_rows * hidden_dim);
        CHECK_EQ(gamma.size(), hidden_dim);
        CHECK_EQ(output.size(), num_rows * hidden_dim);

        if constexpr (std::is_same_v<T, float>) {
            // 根据 hidden_dim 选择 block 大小
            constexpr size_t threads_256 = 256;
            constexpr size_t threads_512 = 512;
            constexpr size_t threads_1024 = 1024;

            size_t block_size;
            if (hidden_dim <= 2048) {
                block_size = threads_256;
            } else if (hidden_dim <= 8192) {
                block_size = threads_512;
            } else {
                block_size = threads_1024;
            }

            if (stream) {
                auto cuda_stream = static_cast<cudaStream_t>(stream);
                if constexpr (std::is_same_v<T, float>) {
                    rms_norm_kernel_cu_launch<<<num_rows, block_size, 0, cuda_stream>>>(
                        input.ptr<float>(),
                        gamma.ptr<float>(),
                        output.ptr<float>(),
                        hidden_dim,
                        eps
                    );
                } else {
                    LOG(ERROR) << "Unsupported CUDA kernel data types for RMSNorm!";
                }
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    rms_norm_kernel_cu_launch<<<num_rows, block_size>>>(
                        input.ptr<float>(),
                        gamma.ptr<float>(),
                        output.ptr<float>(),
                        hidden_dim,
                        eps
                    );
                } else {
                    LOG(ERROR) << "Unsupported CUDA kernel data types for RMSNorm!";
                }
            }

            #ifndef NDEBUG
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                LOG(ERROR) << "RMSNorm kernel launch failed: " << cudaGetErrorString(err);
            }
            #endif
        } else {
            LOG(FATAL) << "Unsupported CUDA kernel device types for RMSNorm!";
            throw std::runtime_error("Unsupported CUDA kernel device types for RMSNorm!");
        }
    }

    template void rms_norm_kernel_device<float>(
        const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&,
        size_t, size_t, float, void*
    );
}
