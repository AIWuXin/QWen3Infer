//
// Created by Administrator on 2026/3/4.
//

#ifndef QWEN3INFER_KERNELS_INTERFACE_HPP
#define QWEN3INFER_KERNELS_INTERFACE_HPP


#include <functional>

#include "../../include/tensor/tensorbase.h"
#include "kernel/host/elementwise.hpp"
#include "kernel/device/elementwise.cuh"
#include "kernel/host/reduction.hpp"
#include "kernel/device/reduction.cuh"


namespace qwi::ops::kernel {
    template<base::ElementWiseType Op = base::ElementWiseType::kElementAdd, typename T = float>
    using ElementWiseKernel = std::function<void(
        const tensor::Tensor&,
        const tensor::Tensor&,
        tensor::Tensor&,
        void*
    )>;

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    using ReductionKernel = std::function<void(
        const tensor::Tensor&,
        tensor::Tensor&,
        int32_t dim,
        void*
    )>;

    template<base::ElementWiseType Op = base::ElementWiseType::kElementAdd, typename T = float>
    ElementWiseKernel<Op, T> get_element_wise_kernel(
        const base::DeviceType device_type
    ) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return element_wise_kernel_host<Op, T>;
        }

        if (device_type == base::DeviceType::kDeviceCUDA) {
            return element_wise_kernel_device<Op, T>;
        }

        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }

    template<base::ReductionType Op = base::ReductionType::kReduceSum, typename T = float>
    ReductionKernel<Op, T> get_reduction_wise_kernel(
        const base::DeviceType device_type,
        int32_t dim
    ) {
        if (dim < -1) {
            LOG(FATAL) << "UnSupported dimensiton: dim < -1";
            throw std::invalid_argument("dim < -1");
        }

        if (device_type == base::DeviceType::kDeviceCPU) {
            if (dim == -1) {
                return reduction_kernel_host<Op, T>;
            }

            return reduction_dim_kernel_host<Op, T>;
        }

        if (device_type == base::DeviceType::kDeviceCUDA) {
            if (dim == -1) {
                return reduction_kernel_device<Op, T>;
            }

            return reduction_dim_kernel_device<Op, T>;
        }

        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}


#endif //QWEN3INFER_KERNELS_INTERFACE_HPP