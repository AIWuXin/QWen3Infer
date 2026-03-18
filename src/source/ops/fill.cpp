//
// Created by Administrator on 2026/3/16.
//


#include "../../include/ops/fill.h"
#include "kernels_interface.hpp"


namespace qwi::ops {
    base::Status Fill::check() const {
        const auto input0 = this->get_input(0);

        base::Status status = check_tensor_with_dim(
            input0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!status) {
            LOG(ERROR) << "The input tensor 0 error in the reduction layer.";
            return status;
        }

        if (this->count() < 0) {
            LOG(ERROR) << "The parameter `count` cannot smaller than zero!";
            return base::Status(
                base::ReturnStatus::InvalidArgument,
                "The parameter `count` cannot smaller than zero!"
            );
        }

        if (this->dim() >= static_cast<int32_t>(input0.ndims())) {
            LOG(ERROR) << "The dimension `dim` must be less than `ndims` in input0!";
            return base::Status(
                base::ReturnStatus::InvalidArgument,
                "The dimension `dim` must be less than `ndims` in input0!"
            );
        }

        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status Fill::forward() {
        auto status = this->check();
        if (!status) {
            return status;
        }

        if (this->count() == 0) {
            LOG(WARNING) << "The parameter `count` is zero, nothing to do!";
            return base::Status(
                base::ReturnStatus::Success
            );
        }

        auto input0 = this->get_input(0);
        void* stream = nullptr;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
            stream = this->cuda_config_->stream;
        }

        switch (this->data_type()) {
            case base::DataType::kDataFloat32: {
                kernel::get_fill_kernel<float>(this->device_type_)(
                    input0,
                    static_cast<float>(this->value_),
                    this->dim_,
                    this->count_,
                    stream
                );
                break;
            } default: {
                LOG(FATAL) << "Unsupported data type.";
                throw std::runtime_error("Unsupported data type.");
            }
        }

        return base::Status(
            base::ReturnStatus::Success
        );
    }

    inline double Fill::get_value() const {
        return this->value_;
    }

    inline int32_t Fill::dim() const {
        return this->dim_;
    }

    inline size_t Fill::count() const {
        return this->count_;
    }

    inline void Fill::set_value(double value) {
        this->value_ = value;
    }

    inline void Fill::set_dim(size_t dim) {
        this->dim_ = dim;
    }

    inline void Fill::set_count(size_t count) {
        this->count_ = count;
    }
}
