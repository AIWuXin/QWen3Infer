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
        const std::shared_ptr<base::DeviceAllocator>& allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : dims_(std::move(dims)), data_type_(data_type) {
        const auto byte_size = reduce_dimension(
            this->dims_.begin(),
            this->dims_.end(),
            size_t{1}
        );
        this->size_ = byte_size;

        if (need_alloc && allocator && !buffer) {
            this->allocate(
                allocator, this->byte_size(), device_type
            );
        } else {
            this->init_buffer(
                allocator,
                device_type,
                need_alloc,
                buffer
            );
        }
    }

    base::ReturnStatus Tensor::allocate(
        const std::shared_ptr<base::DeviceAllocator>& allocator,
        const size_t byte_size,
        const base::DeviceType device_type
    ) {
        if (!allocator) {
            LOG(ERROR) << "Tensor::allocate() called with nullptr" << std::endl;
            this->buffer_ = nullptr;
            return base::ReturnStatus::NoAllocator;
        }

        if (byte_size == 0) {
            LOG(WARNING) << "Tensor::allocate() called with zero byte size" << std::endl;
            this->buffer_ = std::make_shared<base::Buffer>();
            return base::ReturnStatus::ZeroByteSize;
        }

        auto memory_buffer = base::MemoryBuffer(
            nullptr, byte_size, true,
            0, device_type
        );
        this->buffer_ = std::make_shared<base::Buffer>(
            memory_buffer, allocator, false
        );

        if (!this->buffer_->get_ptr()) {
            LOG(ERROR) << "Tensor::allocate() called with nullptr" << std::endl;
            return base::ReturnStatus::ErrorAllocating;
        }

        return base::ReturnStatus::Success;
    }

    size_t Tensor::byte_size() const {
        return base::data_type_to_size(
            this->data_type_
        ) * this->size_;
    }

    base::ReturnStatus Tensor::init_buffer(
        const std::shared_ptr<base::DeviceAllocator>& allocator,
        base::DeviceType device_type,
        bool need_alloc,
        const std::shared_ptr<base::Buffer> &buffer
    ) {
        if (!allocator && !need_alloc) {
            // 无分配器并且不需要分配说明且buffer非空
            // 说明是外部数据

            this->buffer_ = buffer;
            return base::ReturnStatus::Success;
        }

        const auto state = this->allocate(
            allocator, this->byte_size(),
            device_type
        );

        return state;
    }
}
