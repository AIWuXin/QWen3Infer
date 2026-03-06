//
// Created by Administrator on 2026/3/5.
//

#include "elementwise.cuh"
#include "math.cuh"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op>
    __launch_bounds__(256, 4)
    __global__ void elementwise_kernel_cu_launch(
        const size_t size, const float* __restrict in0,
        const float* __restrict in1, float* __restrict out0
    ) {
        elementwise_kernel_cu<Op>(size, in0, in1, out0);
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
        const int32_t size = static_cast<int32_t>(input0.size());

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
                elementwise_kernel_cu_launch<Op><<<block_num, thread_num, 0, cuda_stream>>>(
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
                elementwise_kernel_cu_launch<Op><<<block_num, thread_num>>>(
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
