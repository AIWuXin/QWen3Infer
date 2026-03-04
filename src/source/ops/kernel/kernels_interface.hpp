//
// Created by Administrator on 2026/3/4.
//

#ifndef QWEN3INFER_KERNELS_INTERFACE_HPP
#define QWEN3INFER_KERNELS_INTERFACE_HPP


#include "../../../include/tensor/tensorbase.h"


namespace qwi::ops::kernel {
    typedef void (*ElementWiseKernel) (
        const tensor::Tensor& input1,
        const tensor::Tensor& input2,
        const tensor::Tensor& output,
        void* stream
    );

    inline ElementWiseKernel get_element_wise_kernel(const base::DeviceType device_type) {
        if (device_type == base::DeviceType::kDeviceCPU) {
            return element_wise_kernel_host;
        }

        if (device_type == base::DeviceType::kDeviceCUDA) {
            return element_wise_kernel_device;
        }

        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}


#endif //QWEN3INFER_KERNELS_INTERFACE_HPP