//
// Created by Administrator on 2026/3/4.
//


#include <utility>

#include "../../include/ops/elementwise.h"


namespace qwi::ops {
    Elementwise::Elementwise(
        const base::DataType data_type,
        std::string layer_name,
        const base::DeviceType device_type
    ) : CommonLayer(
        device_type,
        base::LayerType::kLayerElementWise,
        data_type,
        std::move(layer_name)
    ) {}

    base::Status Elementwise::check() const {
        const auto input0 = this->get_input(0);
        const auto input1 = this->get_input(1);
        auto output0 = this->get_output(0);

        base::Status status = check_tensor_with_dim(
            input0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!status) {
            LOG(ERROR) << "The input tensor 0 error in the add layer.";
            return status;
        }

        status = check_tensor_with_dim(
            input1, this->data_type(),
            this->device_type(), input1.dims()
        );
        if (!status) {
            LOG(ERROR) << "The input tensor 1 error in the add layer.";
            return status;
        }

        status = check_tensor_with_dim(
            output0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!status) {
            LOG(ERROR) << "The output tensor 0 error in the add layer.";
            return status;
        }

        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status Elementwise::forward() {
        return CommonLayer::forward();

        auto input1 = this->get_input(0);
        auto input2 = this->get_input(1);
        auto output = this->get_output(0);
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
        }


    }
}
