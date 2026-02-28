//
// Created by Administrator on 2026/2/27.
//


#include "../../include/tensor/tensorbase.h"


namespace qwi::tensor {
    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const bool need_alloc,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0},
        need_alloc, device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const bool need_alloc,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1},
        need_alloc, device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const size_t dim2, const bool need_alloc,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1, dim2},
        need_alloc, device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const size_t dim2, const size_t dim3,
        const bool need_alloc,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1, dim2, dim3},
        need_alloc, device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        std::vector<size_t> dims,
        bool need_alloc,
        base::DeviceType device_type,
        std::shared_ptr<base::DeviceAllocator> allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : dims_(std::move(dims)), data_type_(data_type) {
        auto byte_size = reduce_dimension(
            this->dims_.begin(),
            this->dims_.end(),
            size_t{1}
        );
    }
}
