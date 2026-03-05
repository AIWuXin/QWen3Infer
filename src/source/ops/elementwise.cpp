//
// Created by Administrator on 2026/3/4.
//


#include <utility>

#include "../../include/ops/elementwise.h"

#include "kernels_interface.hpp"


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
    ) {
        this->inputs_.resize(size_t{2});
        this->outputs_.resize(size_t{1});
    }

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
        const auto input0 = this->get_input(0);
        const auto input1 = this->get_input(1);
        auto output0 = this->get_output(0);
        void* stream = nullptr;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
            stream = cuda_config_->stream;
        }

        switch (this->data_type()) {
            case base::DataType::kDataFloat32: {
                kernel::get_element_wise_kernel<
                    base::ElementWiseType::kElementAdd, float
                >(this->device_type())(
                    input0, input1,
                    output0, stream
                );
                break;
            }
            case base::DataType::kDataFloat16: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType fp16 not supported yet!"
                };
            }
            case base::DataType::kDataFloat8: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType fp16 not supported yet!"
                };
            }
            case base::DataType::kDataInt32: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int32 not supported yet!"
                };
            }
            case base::DataType::kDataInt16: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int16 not supported yet!"
                };
            }
            case base::DataType::kDataInt8: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int8 not supported yet!"
                };
            }
            default: {
                return base::Status{
                    base::ReturnStatus::InvalidArgument,
                    "Unsupported data type!"
                };
            }
        }

        return base::Status{base::ReturnStatus::Success};
    }
}
