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
#include "kernel/host/fill.hpp"
#include "kernel/device/fill.cuh"
#include "kernel/host/rsmnorm.hpp"
#include "kernel/device/rsmnorm.cuh"


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

    template<typename T = float>
    using FillKernel = std::function<void(
        tensor::Tensor&,
        T value,
        int32_t dim,
        size_t count,
        void* stream
    )>;

    template<typename T = float>
    using RmsNormKernel = std::function<void(
        const tensor::Tensor&,
        const tensor::Tensor&,
        tensor::Tensor&,
        size_t num_rows,
        size_t hidden_dim,
        float eps,
        void* stream
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
        const int32_t dim
    ) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            if (dim == INT_MIN) {
                return reduction_kernel_host<Op, T>;
            }

            return reduction_dim_kernel_host<Op, T>;
        }

        if (device_type == base::DeviceType::kDeviceCUDA) {
            if (dim == INT_MIN) {
                return reduction_kernel_device<Op, T>;
            }

            return reduction_dim_kernel_device<Op, T>;
        }

        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }

    template<typename T = float>
    FillKernel<T> get_fill_kernel(
        const base::DeviceType device_type
    ) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return fill_kernel_host<T>;
        }
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return fill_kernel_device<T>;
        }
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }

    template<typename T = float>
    RmsNormKernel<T> get_rms_norm_kernel(
        const base::DeviceType device_type
    ) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return rms_norm_kernel_host<T>;
        }
        if (device_type == base::DeviceType::kDeviceCUDA) {
            return rms_norm_kernel_device<T>;
        }
        LOG(FATAL) << "Unknown device type for get rms norm kernel.";
        return nullptr;
    }
}


#endif //QWEN3INFER_KERNELS_INTERFACE_HPP