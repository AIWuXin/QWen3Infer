//
// Created by Administrator on 2026/3/6.
//


#include "../../include/ops/reduction.h"
#include "kernels_interface.hpp"


namespace qwi::ops::kernel {
    Reduction::Reduction(
        base::DataType data_type,
        std::string layer_name,
        base::DeviceType device_type,
        base::ReductionType op_type,
        const int32_t dim
    ) : CommonLayer(
        device_type,
        base::LayerType::kLayerReduction,
        data_type,
        std::move(layer_name)
    ) {
        this->inputs_.resize(size_t{1});
        this->outputs_.resize(size_t{1});
        this->op_type_ = op_type;
        this->dim_ = dim;
    }

    base::Status Reduction::check() const {
        const auto input0 = this->get_input(0);
        const auto output0 = this->get_output(0);

        base::Status status = check_tensor_with_dim(
            input0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!status) {
            LOG(ERROR) << "The input tensor 0 error in the reduction layer.";
            return status;
        }

        status = check_tensor_with_dim(
            output0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!status) {
            LOG(ERROR) << "The output tensor 0 error in the reduction layer.";
            return status;
        }

        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status Reduction::forward() {
        const auto input0 = this->get_input(0);
        auto output0 = this->get_output(0);
        void* stream = nullptr;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
            stream = cuda_config_->stream;
        }



        return base::Status{base::ReturnStatus::Success};
    }

    base::ReductionType Reduction::get_op_type() const {
        return this->op_type_;
    }

    int32_t Reduction::get_dim() const {
        return this->dim_;
    }
}
