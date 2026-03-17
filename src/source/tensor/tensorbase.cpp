//
// Created by Administrator on 2026/2/27.
//


#include <utility>
#include <cstring>

#include "../../include/tensor/tensorbase.h"
#include "../../include/ops/ops.h"


namespace qwi::tensor {
    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0},
        device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1},
        device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const size_t dim2,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1, dim2},
        device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        const size_t dim0, const size_t dim1,
        const size_t dim2, const size_t dim3,
        const base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator> &allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : Tensor(
        data_type, std::vector{dim0, dim1, dim2, dim3},
        device_type, allocator, buffer
    ) {}

    Tensor::Tensor(
        const base::DataType data_type,
        std::vector<size_t> dims,
        base::DeviceType device_type,
        const std::shared_ptr<base::DeviceAllocator>& allocator,
        const std::shared_ptr<base::Buffer> &buffer
    ) : dims_(std::move(dims)), data_type_(data_type) {
        const auto element_count = reduce_dimension(
            this->dims_.begin(),
            this->dims_.end(),
            size_t{1}
        );
        this->size_ = element_count;

        if (allocator && !buffer) {
            if (device_type != base::DeviceType::kDeviceUnknown) {
                this->allocate(
                    allocator, this->byte_size(), device_type
                );
            }
        }

        this->init_buffer(
            allocator,
            device_type,
            buffer
        );
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

    base::Status Tensor::fill(
        const double value,
        const int32_t dim,
        const size_t count
    ) const {
        auto fill_function = ops::Fill(
            this->get_data_type(),
            value, dim, count,
            this->get_device_type(),
            "tensor self fill"
        );
        fill_function.set_input(0, *this);
        return fill_function.forward();
    }

    size_t Tensor::byte_size() const {
        return base::data_type_to_size(
            this->data_type_
        ) * this->size_;
    }

    base::ReturnStatus Tensor::to_float() {
        this->data_type_ = base::DataType::kDataFloat32;
        return base::ReturnStatus::Success;
    }

    // base::ReturnStatus Tensor::to_int8() {
    //     this->data_type_ = base::DataType::kDataInt8;
    //     return base::ReturnStatus::Success;
    // }

    base::ReturnStatus Tensor::set_dims(std::vector<size_t> dims) {
        if (dims_.empty() || this->size_ == 0) {
            this->dims_ = std::move(dims);
            const auto element_count = reduce_dimension(
                this->dims_.begin(),
                this->dims_.end(),
                size_t{1}
            );
            this->size_ = element_count;
            return base::ReturnStatus::Success;
        }

        LOG(ERROR) << "Tensor::set_dims() must be zero size!" << std::endl;
        return base::ReturnStatus::Error;
    }

    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = reduce_dimension(
                    dims_.begin() + i + 1, dims_.end(), size_t{1}
                );
                strides.push_back(stride);
            }
            strides.push_back(size_t{1});
        }
        return strides;
    }

    size_t Tensor::ndims() const {
        return this->dims_.size();
    }

    bool Tensor::is_empty() const {
        return this->size_ == 0
        || this->buffer_ == nullptr
        || this->buffer_->get_ptr() == nullptr;
    }

    base::DataType Tensor::get_data_type() const {
        return this->data_type_;
    }

    base::DeviceType Tensor::get_device_type() const {
        return this->buffer_->get_device_type();
    }

    size_t Tensor::dim(size_t idx) const {
        if (idx >= this->ndims()) {
            LOG(ERROR) << "Tensor::dim() index out of bounds!" << std::endl;
            throw std::out_of_range("Tensor::dim() index out of bounds!");
        }

        return this->dims_[idx];
    }

    std::vector<size_t> Tensor::dims() const {
        return this->dims_;
    }

    size_t Tensor::size() const {
        return this->size_;
    }

    base::ReturnStatus Tensor::init_buffer(
        const std::shared_ptr<base::DeviceAllocator>& allocator,
        base::DeviceType device_type,
        const std::shared_ptr<base::Buffer> &buffer
    ) {
        if (!allocator && buffer) {
            // 无分配器说明且buffer非空
            // 说明是外部数据

            this->buffer_ = buffer;
            return base::ReturnStatus::Success;
        }

        if (device_type == base::DeviceType::kDeviceUnknown) {
            // 说明暂时不需要分配
            auto memory_buffer = base::MemoryBuffer(
                nullptr, this->byte_size(), true, 0,
                base::DeviceType::kDeviceUnknown
            );
            this->buffer_ = std::make_shared<base::Buffer>(
                memory_buffer,
                nullptr, false
            );
            return base::ReturnStatus::NoAllocator;
        }

        const auto state = this->allocate(
            allocator, this->byte_size(),
            device_type
        );

        return state;
    }
}
