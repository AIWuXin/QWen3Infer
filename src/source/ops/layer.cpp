//
// Created by Administrator on 2026/3/3.
//


#include "../../include/ops/layer.h"

#include <cstdarg>


namespace qwi::ops {
    base::Status CommonLayer::init() {
        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status CommonLayer::check() const {
        return base::Status{
            base::ReturnStatus::NotImplement,
            "The check function is not implement yet!"
        };
    }

    base::Status CommonLayer::forward() {
        return base::Status{
            base::ReturnStatus::NotImplement,
            "The forward function is not implement yet!"
        };
    }

    base::Status CommonLayer::check_tensor_with_dim(
        const tensor::Tensor &tensor,
        const base::DataType data_type,
        const base::DeviceType device_type,
        ...
    ) {
        std::va_list args;

        if (tensor.is_empty()) {
            LOG(ERROR) << "Tensor is empty!" << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "Tensor is empty!"
            };
        }

        if (tensor.get_data_type() != data_type) {
            LOG(ERROR) << "The tensor has a wrong device type." << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "The tensor has a wrong device type."
            };
        }

        if (tensor.get_device_type() != device_type) {
            LOG(ERROR) << "The tensor has a wrong device type." << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "The tensor has a wrong device type."
            };
        }

        va_start(args, device_type);
        const size_t dims = tensor.ndims();
        for (size_t i = 0; i < dims; ++i) {
            size_t dim = va_arg(args, size_t);
            if (dim != tensor.dim(i)) {
                return base::Status{
                    base::ReturnStatus::Error,
                    "The tensor has a wrong dim in dim" + std::to_string(i)
                };
            }
        }
        va_end(args);
        return base::Status{
            base::ReturnStatus::Success
        };
    }

    base::Status CommonLayer::check_tensor_with_dim(
        const tensor::Tensor &tensor,
        const base::DataType data_type,
        const base::DeviceType device_type,
        const std::vector<size_t>& act_dims
    ) {
        if (tensor.is_empty()) {
            LOG(ERROR) << "Tensor is empty!" << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "Tensor is empty!"
            };
        }

        if (tensor.get_data_type() != data_type) {
            LOG(ERROR) << "The tensor has a wrong device type." << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "The tensor has a wrong device type."
            };
        }

        if (tensor.get_device_type() != device_type) {
            LOG(ERROR) << "The tensor has a wrong device type." << std::endl;
            return base::Status{
                base::ReturnStatus::InvalidArgument,
                "The tensor has a wrong device type."
            };
        }

        const size_t dims = tensor.ndims();
        for (size_t i = 0; i < dims; ++i) {
            size_t dim = act_dims[i];
            if (dim != tensor.dim(i)) {
                return base::Status{
                    base::ReturnStatus::Error,
                    "The tensor has a wrong dim in dim" + std::to_string(i)
                };
            }
        }
        return base::Status{
            base::ReturnStatus::Success
        };
    }

    const tensor::Tensor & CommonLayer::get_input(const size_t idx) const {
        if (idx >= inputs_.size()) {
            LOG(FATAL) << "get_input index out of bounds!" << std::endl;
            throw std::out_of_range("get_input index out of bounds!");
        }

        return this->inputs_.at(idx);
    }

    tensor::Tensor & CommonLayer::get_input(size_t idx) {
        if (idx >= inputs_.size()) {
            LOG(FATAL) << "get_input index out of bounds!" << std::endl;
            throw std::out_of_range("get_input index out of bounds!");
        }

        return this->inputs_.at(idx);
    }

    const tensor::Tensor & CommonLayer::get_output(size_t idx) const {
        if (idx >= outputs_.size()) {
            LOG(FATAL) << "get_output index out of bounds!" << std::endl;
            throw std::out_of_range("get_output index out of bounds!");
        }

        return this->outputs_.at(idx);
    }

    tensor::Tensor & CommonLayer::get_output(size_t idx) {
        if (idx >= outputs_.size()) {
            LOG(FATAL) << "get_output index out of bounds!" << std::endl;
            throw std::out_of_range("get_output index out of bounds!");
        }

        return this->outputs_.at(idx);
    }

    void CommonLayer::set_input(
        size_t idx, const tensor::Tensor &input
    ) {
        if (idx >= inputs_.size()) {
            LOG(FATAL) << "set_input index out of bounds!" << std::endl;
            throw std::out_of_range("set_input index out of bounds!");
        }

        this->inputs_.at(idx) = input;
    }

    void CommonLayer::set_output(
        size_t idx, const tensor::Tensor &output
    ) {
        if (idx >= outputs_.size()) {
            LOG(FATAL) << "set_output index out of bounds!" << std::endl;
            throw std::out_of_range("set_output index out of bounds!");
        }

        this->outputs_.at(idx) = output;
    }

    size_t CommonLayer::input_size() const {
        return this->inputs_.size();
    }

    size_t CommonLayer::output_size() const {
        return this->outputs_.size();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &input1,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_input(1, input1);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &input1,
        const tensor::Tensor &input2,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_input(1, input1);
        this->set_input(2, input2);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &input1,
        const tensor::Tensor &input2,
        const tensor::Tensor &input3,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_input(1, input1);
        this->set_input(2, input2);
        this->set_input(3, input3);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &input1,
        const tensor::Tensor &input2,
        const tensor::Tensor &input3,
        const tensor::Tensor &input4,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_input(1, input1);
        this->set_input(2, input2);
        this->set_input(3, input3);
        this->set_input(4, input4);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(
        const tensor::Tensor &input0,
        const tensor::Tensor &input1,
        const tensor::Tensor &input2,
        const tensor::Tensor &input3,
        const tensor::Tensor &input4,
        const tensor::Tensor &input5,
        const tensor::Tensor &output0
    ) {
        this->set_input(0, input0);
        this->set_input(1, input1);
        this->set_input(2, input2);
        this->set_input(3, input3);
        this->set_input(4, input4);
        this->set_input(5, input5);
        this->set_output(0, output0);

        return this->forward();
    }

    base::Status CommonLayer::forward(const std::vector<tensor::Tensor> &inputs, const std::vector<tensor::Tensor> &outputs) {
        for (size_t i = 0; i < inputs.size(); i++) {
            this->set_input(i, inputs[i]);
        }
        for (size_t i = 0; i < outputs.size(); i++) {
            this->set_output(i, outputs[i]);
        }

        return this->forward();
    }
}
