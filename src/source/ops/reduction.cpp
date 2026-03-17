//
// Created by Administrator on 2026/3/6.
//


#include "../../include/ops/reduction.h"
#include "kernels_interface.hpp"


namespace qwi::ops {
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
            this->device_type(), output0.dims()
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
        auto status = this->check();
        if (!status) {
            return status;
        }

        const auto input0 = this->get_input(0);
        auto output0 = this->get_output(0);
        void* stream = nullptr;
        if (device_type_ == base::DeviceType::kDeviceCUDA) {
            CHECK(cuda_config_ != nullptr);
            stream = cuda_config_->stream;
        }

        switch (this->data_type()) {
            case base::DataType::kDataFloat32: {
                switch (this->op_type_) {
                    case base::ReductionType::kReduceSum: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceSum, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    case base::ReductionType::kReduceMean: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceMean, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    case base::ReductionType::kReduceMax: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceMax, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    case base::ReductionType::kReduceMin: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceMin, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    case base::ReductionType::kReduceAll: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceAll, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    case base::ReductionType::kReduceAny: {
                        kernel::get_reduction_wise_kernel<
                            base::ReductionType::kReduceAny, float
                        >(this->device_type(), this->dim_)(
                            input0, output0,
                            this->dim_, stream
                        );
                        break;
                    }
                    default: {
                        return base::Status{
                            base::ReturnStatus::InvalidArgument,
                            "Unsupported reduction type!"
                        };
                    }
                }
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
                    "DataType fp8 not supported yet!"
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

    base::ReductionType Reduction::get_op_type() const {
        return this->op_type_;
    }

    int32_t Reduction::get_dim() const {
        return this->dim_;
    }
}
