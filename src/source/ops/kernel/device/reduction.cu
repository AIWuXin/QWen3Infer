//
// Created by Administrator on 2026/3/6.
//


#include "reduction.cuh"
#include "math.cuh"


namespace qwi::ops::kernel {
    template<base::ReductionType Op, const size_t ThreadNum>
    __launch_bounds__(256, 1)
    __global__ void reduction_kernel_cu_launch(
        const size_t size,
        const float* __restrict in0,
        float* __restrict out0
    ) {
        reduction_kernel_cu<Op, ThreadNum>(size, in0, out0);
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_dim_kernel_device(
        const tensor::Tensor &input0,
        tensor::Tensor &output0,
        [[maybe_unused]] int32_t dim,
        void *stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);
        CHECK_EQ(output0.ndims() == 1, true);
        CHECK_EQ(output0.dim(0) == 1, true);

        const size_t size = input0.size();
        constexpr size_t thread_num0 = 256;
        const size_t block_num0 = std::min<size_t>(
            (size + thread_num0 - 1) / thread_num0,
            256 * 20  // 限制最大 block 数，避免过小的工作量
        );
        const size_t thread_num1 = block_num0 > thread_num0 ? thread_num0 : block_num0;
        const size_t block_num1 = block_num0 > thread_num0 ?
            (block_num0 + block_num0 - 1) / block_num0 : 1;
        if (stream) {
            auto cuda_stream = static_cast<cudaStream_t>(stream);
            if constexpr (std::is_same_v<T, float>) {
                float* device_local_ptr = nullptr;
                cudaMalloc(&device_local_ptr, block_num0 * sizeof(float));
                reduction_kernel_cu_launch<Op, thread_num0><<<block_num0, thread_num0, cuda_stream>>>(
                    size, input0.ptr<float>(),
                    device_local_ptr
                );  // 第一阶段规约, block内部规约
                reduction_kernel_cu_launch<Op, thread_num1><<<block_num1, thread_num1, cuda_stream>>>(
                    block_num0, device_local_ptr,
                    output0.ptr<float>()
                );  // 第二阶段规约, grid内部规约
                cudaFree(device_local_ptr);
            } else {
                LOG(FATAL) << "Unsupported CUDA kernel device types!";
                throw std::runtime_error("Unsupported CUDA kernel device types!");
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                float* device_local_ptr = nullptr;
                cudaMalloc(&device_local_ptr, block_num0 * sizeof(float));
                reduction_kernel_cu_launch<Op, thread_num0><<<block_num0, thread_num0>>>(
                    size, input0.ptr<float>(),
                    device_local_ptr
                );  // 第一阶段规约, block内部规约
                reduction_kernel_cu_launch<Op, thread_num1><<<block_num1, thread_num1>>>(
                    block_num0, device_local_ptr,
                    output0.ptr<float>()
                );  // 第二阶段规约, grid内部规约
                cudaFree(device_local_ptr);
            } else {
                LOG(FATAL) << "Unsupported CUDA kernel device types!";
                throw std::runtime_error("Unsupported CUDA kernel device types!");
            }
        }
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    void reduction_kernel_device(
        const tensor::Tensor &input0,
        tensor::Tensor &output0,
        [[maybe_unused]] int32_t dim,
        void *stream
    ) {
    }

#define INSTANTIATE_REDUCTION_DEVICE(OP, T)                      \
    template void reduction_kernel_device<OP, T>(                \
        const tensor::Tensor&, tensor::Tensor&, int32_t, void *  \
    )

#define INSTANTIATE_REDUCTION_DEVICE_ALL_OPS(T)                         \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceSum, T);   \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceMean, T);  \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceMax, T);   \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceMin, T);   \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceAll, T);   \
    INSTANTIATE_REDUCTION_DEVICE(base::ReductionType::kReduceAny, T);

    INSTANTIATE_REDUCTION_DEVICE_ALL_OPS(float);

#undef INSTANTIATE_REDUCTION_DEVICE
#undef INSTANTIATE_REDUCTION_DEVICE_ALL_OPS

#define INSTANTIATE_REDUCTION_DIM_DEVICE(OP, T)              \
    template void reduction_dim_kernel_device<OP, T>(        \
    const tensor::Tensor&, tensor::Tensor&, int32_t, void *  \
)

#define INSTANTIATE_REDUCTION_DIM_DEVICE_ALL_OPS(T)                         \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceSum, T);   \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceMean, T);  \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceMax, T);   \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceMin, T);   \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceAll, T);   \
    INSTANTIATE_REDUCTION_DIM_DEVICE(base::ReductionType::kReduceAny, T);

    INSTANTIATE_REDUCTION_DIM_DEVICE_ALL_OPS(float);

#undef INSTANTIATE_REDUCTION_DIM_DEVICE
#undef INSTANTIATE_REDUCTION_DIM_DEVICE_ALL_OPS
}
