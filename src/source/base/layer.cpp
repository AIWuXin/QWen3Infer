//
// Created by Administrator on 2026/3/3.
//


#include "../../include/ops/layer.h"


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
}
