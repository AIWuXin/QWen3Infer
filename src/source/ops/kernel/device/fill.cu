//
// Created by Administrator on 2026/3/16.
//


#include "fill.cuh"
#include "math.cuh"


namespace qwi::ops::kernel {
    __launch_bounds__(256, 4)
    __global__ void fill_kernel_cu_launch(
        float* __restrict in0,
        const size_t count,
        const float value
    ) {
        fill_kernel_cu(
            in0, count, value
        );
    }

    __launch_bounds__(256, 4)
    __global__ void fill_dim_kernel_cu_launch(
        float* __restrict in0,
        const size_t count,
        const float value,
        const size_t outer_size,
        const size_t reduce_dim_size,
        const size_t inner_size,
        const size_t stride_outer,  // input_strides[0]
        const size_t stride_reduce // input_strides[dim]
    ) {
        fill_dim_kernel_cu(
            in0, count, value, outer_size,
            reduce_dim_size, inner_size,
            stride_outer, stride_reduce
        );
    }

    template<typename T>
    void fill_kernel_device(
        tensor::Tensor& input0,
        T value,
        int32_t dim,
        size_t count,
        void* stream
    ) {
        UNUSED(stream);

        CHECK_EQ(input0.is_empty(), false);
        CHECK(count > 0);
        CHECK(count <= input0.size());

        if (dim == INT_MIN) {
            constexpr size_t thread_num = 256;
            const size_t block_num = std::min<size_t>(
                (count + thread_num - 1) / thread_num,
                1024 * 4  // 限制最大 block 数，避免过小的工作量
            );
            if (stream) {
                auto cuda_stream = static_cast<cudaStream_t>(stream);

                if constexpr (std::is_same_v<T, float>) {
                    fill_kernel_cu_launch<<<block_num, thread_num, 0, cuda_stream>>>(
                        input0.ptr<float>(), count, value
                    );
                }
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    fill_kernel_cu_launch<<<block_num, thread_num>>>(
                        input0.ptr<float>(), count, value
                    );
                }
            }
        } else {
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

            constexpr size_t thread_num = 256;
            const size_t block_num = outer_size * inner_size;

            if (stream) {
                auto cuda_stream = static_cast<cudaStream_t>(stream);

                if constexpr (std::is_same_v<T, float>) {
                    fill_dim_kernel_cu_launch<<<block_num, thread_num, 0, cuda_stream>>>(
                        input0.ptr<float>(), count, value,
                        outer_size, reduce_dim_size, inner_size,
                        input_strides[std::min<size_t>(dim-1, 0)],
                        input_strides[dim]
                    );
                }
            } else {
                if constexpr (std::is_same_v<T, float>) {
                    fill_dim_kernel_cu_launch<<<block_num, thread_num>>>(
                        input0.ptr<float>(), count, value,
                        outer_size, reduce_dim_size, inner_size,
                        input_strides[std::min<size_t>(dim-1, 0)],
                        input_strides[dim]
                    );
                }
            }
        }

        #ifndef NDEBUG
            if (const cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
                LOG(ERROR) << "Kernel launch failed: " << cudaGetErrorString(err);
            }
        #endif

        if (stream) {
            cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }
    }

    template void fill_kernel_device<float>(
        tensor::Tensor& input0,
        float value,
        int32_t dim,
        size_t count,
        void* stream
    );
}
