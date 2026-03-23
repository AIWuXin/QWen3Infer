//
// Created by Administrator on 2026/3/6.
//


#include "reduction.cuh"
#include "math.cuh"


namespace qwi::ops::kernel {
    template<base::ReductionType Op>
    __launch_bounds__(256, 4)
    __global__ void reduction_kernel_cu_launch(
        const size_t size,
        const float* __restrict in0,
        float* __restrict out0,
        const float div_number
    ) {
        reduction_kernel_cu<Op>(size, in0, out0, div_number);
    }

    template<base::ReductionType Op>
    __launch_bounds__(256, 4)
    __global__ void reduction_dim_kernel_cu_launch(
        const float* __restrict in0,
        float* __restrict out0,
        const size_t outer_size,
        const size_t reduce_dim_size,
        const size_t inner_size,
        const size_t stride_outer,  // input_strides[0]
        const size_t stride_reduce, // input_strides[dim]
        const float div_number      // for Mean
    ) {
        reduction_dim_kernel_cu<Op>(
            in0, out0, outer_size,
            reduce_dim_size, inner_size,
            stride_outer, stride_reduce,
            div_number
        );
    }

    template<base::ReductionType Op, typename T>
    void reduction_kernel_device(
        const tensor::Tensor &input0,
        tensor::Tensor &output0,
        [[maybe_unused]] int32_t dim,
        void *stream
    ) {
        CHECK_EQ(input0.is_empty(), false);
        CHECK_EQ(output0.is_empty(), false);
        CHECK_EQ(output0.ndims() == 1, true);
        CHECK_EQ(output0.dim(0) == 1, true);

        const size_t size = input0.size();
        constexpr size_t thread_num0 = 256;
        const size_t block_num0 = std::min<size_t>(
            (size + thread_num0 - 1) / thread_num0,
            1024 * 4  // 限制最大 block 数，避免过小的工作量
        );
        constexpr size_t thread_num1 = 256;
        const size_t block_num1 = block_num0 > thread_num0 ?
            (block_num0 + block_num0 - 1) / block_num0 : 1;
        if (stream) {
            auto cuda_stream = static_cast<cudaStream_t>(stream);
            if constexpr (std::is_same_v<T, float>) {
                float* device_local_ptr = nullptr;
                cudaMalloc(&device_local_ptr, block_num0 * sizeof(float));
                reduction_kernel_cu_launch<Op><<<block_num0, thread_num0, 0, cuda_stream>>>(
                    size, input0.ptr<float>(),
                    device_local_ptr, 1.f
                );  // 第一阶段规约, block内部规约
                reduction_kernel_cu_launch<Op><<<block_num1, thread_num1, 0, cuda_stream>>>(
                    block_num0, device_local_ptr,
                    output0.ptr<float>(), static_cast<float>(size)
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
                reduction_kernel_cu_launch<Op><<<block_num0, thread_num0>>>(
                    size, input0.ptr<float>(),
                    device_local_ptr, 1.f
                );  // 第一阶段规约, block内部规约
                reduction_kernel_cu_launch<Op><<<block_num1, thread_num1>>>(
                    block_num0, device_local_ptr,
                    output0.ptr<float>(), static_cast<float>(size)
                );  // 第二阶段规约, grid内部规约
                cudaFree(device_local_ptr);
            } else {
                LOG(FATAL) << "Unsupported CUDA kernel device types!";
                throw std::runtime_error("Unsupported CUDA kernel device types!");
            }
        }
    }

    template<base::ReductionType Op, typename T>
    void reduction_dim_kernel_device(
        const tensor::Tensor &input0,
        tensor::Tensor &output0,
        [[maybe_unused]] int32_t dim,
        void *stream
    ) {
        if (dim < 0) {
            dim += input0.ndims();
        }

        const auto input_dims = input0.dims();
        const auto input_strides = input0.strides();
        const auto reduce_dim_size = static_cast<int32_t>(input_dims[dim]);

        // 计算外层和内层大小
        const size_t outer_size = tensor::reduce_dimension(
            input_dims.begin(), input_dims.begin() + dim, size_t{1}
        );
        const size_t inner_size = tensor::reduce_dimension(
            input_dims.begin() + dim + 1, input_dims.end(), size_t{1}
        );

        const size_t total_outputs = outer_size * inner_size;
        constexpr size_t thread_num = 256;

        if (stream) {
            auto cuda_stream = static_cast<cudaStream_t>(stream);
            reduction_dim_kernel_cu_launch<Op>
            <<<total_outputs, thread_num, 0, cuda_stream>>>(
                    input0.ptr<float>(),
                    output0.ptr<float>(),
                    outer_size,
                    reduce_dim_size,
                    inner_size,
                    (dim > 0) ? input_strides[static_cast<size_t>(dim) - 1] : 0,
                    input_strides[dim],
                    static_cast<float>(reduce_dim_size)  // for Mean
            );
        } else {
            reduction_dim_kernel_cu_launch<Op>
                <<<total_outputs, thread_num>>>(
                    input0.ptr<float>(),
                    output0.ptr<float>(),
                    outer_size,
                    reduce_dim_size,
                    inner_size,
                    (dim > 0) ? input_strides[static_cast<size_t>(dim) - 1] : 0,
                    input_strides[dim],
                    static_cast<float>(reduce_dim_size)
            );
        }

        #ifndef NDEBUG
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                LOG(ERROR) << "Kernel launch failed: " << cudaGetErrorString(err);
            }
        #endif
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
