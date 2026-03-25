//
// Created by Administrator on 2026/3/16.
//

#include "../../include/ops/rsmnorm.h"
#include "kernels_interface.hpp"

namespace qwi::ops {
    RmsNorm::RmsNorm(
        const base::DataType data_type,
        std::string layer_name,
        const base::DeviceType device_type,
        const float eps
    ) : WeightedLayer(
        device_type,
        base::LayerType::kLayerRMSNorm,
        data_type,
        std::move(layer_name)
    ) {
        this->inputs_.resize(size_t{1});
        this->outputs_.resize(size_t{1});
        this->eps_ = eps;
    }

    base::Status RmsNorm::check() const {
        // 检查权重（gamma）是否已设置
        const auto status = this->check_weights();
        if (!status) {
            LOG(ERROR) << "Weight check failed in RmsNorm layer: " << status.get_message();
            return status;
        }

        const auto input0 = this->get_input(0);
        const auto output0 = this->get_output(0);
        const auto& gamma = this->weight(0);

        // 检查输入
        base::Status check_status = check_tensor_with_dim(
            input0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!check_status) {
            LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
            return check_status;
        }

        // 检查输出形状与输入一致
        check_status = check_tensor_with_dim(
            output0, this->data_type(),
            this->device_type(), input0.dims()
        );
        if (!check_status) {
            LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
            return check_status;
        }

        // 检查 gamma 权重形状：最后一维应该是 hidden_dim
        const size_t hidden_dim = input0.dim(input0.ndims() - 1);
        std::vector<size_t> gamma_dims = {hidden_dim};
        check_status = check_tensor_with_dim(
            gamma, this->data_type(),
            this->device_type(), gamma_dims
        );
        if (!check_status) {
            LOG(ERROR) << "The gamma weight shape mismatch in the rmsnorm layer. "
                       << "Expected [" << hidden_dim << "], but got inconsistent shape.";
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "Gamma weight shape mismatch in RmsNorm"
            };
        }

        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status RmsNorm::forward() {
        auto status = this->check();
        if (!status) {
            return status;
        }

        const auto input0 = this->get_input(0);
        auto output0 = this->get_output(0);
        const auto& gamma = this->weight(0);
        
        void* stream = nullptr;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
            stream = cuda_config_->stream;
        }

        const size_t hidden_dim = input0.dim(input0.ndims() - 1);
        const size_t num_rows = input0.size() / hidden_dim;

        switch (this->data_type()) {
            case base::DataType::kDataFloat32: {
                kernel::get_rms_norm_kernel<float>(
                    this->device_type()
                )(
                    input0, gamma, output0,
                    num_rows, hidden_dim,
                    this->eps_, stream
                );
                break;
            }
            case base::DataType::kDataFloat16: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType fp16 not supported yet in RmsNorm!"
                };
            }
            case base::DataType::kDataFloat8: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType fp8 not supported yet in RmsNorm!"
                };
            }
            case base::DataType::kDataInt32: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int32 not supported yet in RmsNorm!"
                };
            }
            case base::DataType::kDataInt16: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int16 not supported yet in RmsNorm!"
                };
            }
            case base::DataType::kDataInt8: {
                return base::Status{
                    base::ReturnStatus::NotImplement,
                    "DataType int8 not supported yet in RmsNorm!"
                };
            }
            default: {
                return base::Status{
                    base::ReturnStatus::InvalidArgument,
                    "Unsupported data type in RmsNorm!"
                };
            }
        }

        return base::Status{base::ReturnStatus::Success};
    }

    float RmsNorm::get_eps() const {
        return this->eps_;
    }
}
