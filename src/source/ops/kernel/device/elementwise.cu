//
// Created by Administrator on 2026/3/5.
//

#include "elementwise.cuh"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op>
    __launch_bounds__(256, 4)
    __global__ void elementwise_kernel_cu(
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

    template<base::ElementWiseType Op, typename T>
    void element_wise_kernel_device(
        const tensor::Tensor& input0,
        const tensor::Tensor& input1,
        tensor::Tensor& output0,
        void* stream
    ) {
        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(input1.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);
        int32_t size = static_cast<int32_t>(input0.size());

        CHECK_EQ(size, input1.size());
        CHECK_EQ(size, output0.size());

        size_t thread_num = 256;
        size_t block_num = std::min<size_t>(
            (size + thread_num - 1) / thread_num,
            256 * 20  // 限制最大 block 数，避免过小的工作量
        );
        if (stream) {
            auto cuda_stream = static_cast<cudaStream_t>(stream);
            if constexpr (std::is_same_v<T, float>) {
                elementwise_kernel_cu<Op><<<block_num, thread_num, 0, cuda_stream>>>(
                    size, input0.ptr<float>(),
                    input1.ptr<float>(),
                    output0.ptr<float>()
                );
            } else {
                LOG(FATAL) << "Unsupported CUDA kernel device types!";
                throw std::runtime_error("Unsupported CUDA kernel device types!");
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                elementwise_kernel_cu<Op><<<block_num, thread_num>>>(
                    size, input0.ptr<float>(),
                    input1.ptr<float>(),
                    output0.ptr<float>()
                );
            } else {
                LOG(FATAL) << "Unsupported CUDA kernel device types!";
                throw std::runtime_error("Unsupported CUDA kernel device types!");
            }
        }

        #ifndef NDEBUG
            cudaDeviceSynchronize();
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                LOG(FATAL) << "CUDA kernel failed: " << cudaGetErrorString(err);
            }
        #endif
    }

#define INSTANTIATE_ELEMENT_WISE_DEVICE(OP, T)                                \
    template void element_wise_kernel_device<OP, T>(                          \
        const tensor::Tensor&, const tensor::Tensor&, tensor::Tensor&, void*  \
    )

#define INSTANTIATE_ELEMENT_WISE_DEVICE_ALL_OPS(T)                                \
    INSTANTIATE_ELEMENT_WISE_DEVICE(base::ElementWiseType::kElementAdd, T);       \
    INSTANTIATE_ELEMENT_WISE_DEVICE(base::ElementWiseType::kElementSubtract, T);  \
    INSTANTIATE_ELEMENT_WISE_DEVICE(base::ElementWiseType::kElementMultiply, T);  \
    INSTANTIATE_ELEMENT_WISE_DEVICE(base::ElementWiseType::kElementDivide, T)     \

    INSTANTIATE_ELEMENT_WISE_DEVICE_ALL_OPS(float);

#undef INSTANTIATE_ELEMENT_WISE_DEVICE
#undef INSTANTIATE_ELEMENT_WISE_DEVICE_ALL_OPS
}
