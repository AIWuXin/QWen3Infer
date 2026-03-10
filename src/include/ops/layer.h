//
// Created by Administrator on 2026/3/2.
//

#ifndef QWEN3INFER_LAYER_H
#define QWEN3INFER_LAYER_H


#include <utility>

#include "../tensor/tensorbase.h"


namespace qwi::ops {
    class BaseLayer {
    public:
        [[nodiscard]] std::string layer_name() const {
            return layer_name_;
        }

        [[nodiscard]] base::LayerType layer_type() const {
            return layer_type_;
        }

        [[nodiscard]] base::DataType data_type() const {
            return data_type_;
        }

        [[nodiscard]] base::DeviceType device_type() const {
            return device_type_;
        }
        BaseLayer(
            std::string layer_name,
            const base::LayerType layer_type,
            const base::DataType data_type,
            const base::DeviceType device_type
        )
            : layer_name_(std::move(layer_name)),
              layer_type_(layer_type),
              data_type_(data_type),
              device_type_(device_type) {
        }
        virtual ~BaseLayer() = default;
        virtual base::Status init() = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& input1,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& input1,
            const tensor::Tensor& input2,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& input1,
            const tensor::Tensor& input2,
            const tensor::Tensor& input3,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& input1,
            const tensor::Tensor& input2,
            const tensor::Tensor& input3,
            const tensor::Tensor& input4,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const tensor::Tensor& input0,
            const tensor::Tensor& input1,
            const tensor::Tensor& input2,
            const tensor::Tensor& input3,
            const tensor::Tensor& input4,
            const tensor::Tensor& input5,
            const tensor::Tensor& output0
        ) = 0;
        virtual base::Status forward(
            const std::vector<tensor::Tensor>& inputs,
            const std::vector<tensor::Tensor>& outputs
        ) = 0;
        [[nodiscard]] virtual base::Status check() const = 0;
        virtual void set_input(
            size_t idx, const tensor::Tensor& input
        ) = 0;
        virtual void set_output(
            size_t idx, const tensor::Tensor& output
        ) = 0;
        [[nodiscard]] virtual size_t input_size() const = 0;
        [[nodiscard]] virtual size_t output_size() const = 0;
        virtual tensor::Tensor& get_input(size_t idx) = 0;
        virtual tensor::Tensor& get_output(size_t idx) = 0;
        [[nodiscard]] virtual const tensor::Tensor& get_input(size_t idx) const = 0;
        [[nodiscard]] virtual const tensor::Tensor& get_output(size_t idx) const = 0;
    protected:
        std::string layer_name_;
        base::LayerType layer_type_ = base::LayerType::kLayerUnknown;
        base::DataType data_type_ = base::DataType::kDataUnknown;
        base::DeviceType device_type_ = base::DeviceType::kDeviceUnknown;
        virtual base::Status forward() = 0;
    };

    class CommonLayer : public BaseLayer {
    public:
        explicit CommonLayer(
            base::DeviceType device_type,
            base::LayerType layer_type,
            base::DataType data_type,
            std::string layer_name = ""
        ) : BaseLayer(
            std::move(layer_name),
            layer_type,
            data_type,
            device_type
        ) {
            if (device_type == base::DeviceType::kDeviceCUDA) {
                this->cuda_config_ = std::make_shared<base::CudaConfig>();
            }
        }
        base::Status init() override;
        [[nodiscard]] base::Status check() const override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &input1,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &input1,
            const tensor::Tensor &input2,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &input1,
            const tensor::Tensor &input2,
            const tensor::Tensor &input3,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &input1,
            const tensor::Tensor &input2,
            const tensor::Tensor &input3,
            const tensor::Tensor &input4,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const tensor::Tensor &input0,
            const tensor::Tensor &input1,
            const tensor::Tensor &input2,
            const tensor::Tensor &input3,
            const tensor::Tensor &input4,
            const tensor::Tensor &input5,
            const tensor::Tensor &output0
        ) override;
        base::Status forward(
            const std::vector<tensor::Tensor> &inputs,
            const std::vector<tensor::Tensor> &outputs
        ) override;
        [[nodiscard]] const tensor::Tensor &get_input(size_t idx) const override;
        tensor::Tensor &get_input(size_t idx) override;
        [[nodiscard]] const tensor::Tensor &get_output(size_t idx) const override;
        tensor::Tensor &get_output(size_t idx) override;
        void set_input(size_t idx, const tensor::Tensor &input) override;
        void set_output(size_t idx, const tensor::Tensor &output) override;
        size_t input_size() const override;
        size_t output_size() const override;
        void set_cuda_config(const std::shared_ptr<base::CudaConfig> &config);
    protected:
        std::vector<tensor::Tensor> inputs_;
        std::vector<tensor::Tensor> outputs_;
        std::shared_ptr<base::CudaConfig> cuda_config_;
        base::Status forward() override;
        static base::Status check_tensor_with_dim(
            const tensor::Tensor &tensor,
            base::DataType data_type,
            base::DeviceType device_type,
            ...
        );
        static base::Status check_tensor_with_dim(
            const tensor::Tensor &tensor,
            base::DataType data_type,
            base::DeviceType device_type,
            const std::vector<size_t>& act_dims
        );
    };

class WeightedLayer : public CommonLayer {
    public:
        explicit WeightedLayer(
            base::DeviceType device_type,
            base::LayerType layer_type,
            base::DataType data_type,
            std::string layer_name = ""
        ) : CommonLayer(
            device_type,
            layer_type,
            data_type,
            std::move(layer_name)
        ) {}

        // 权重管理
        [[nodiscard]] const std::vector<tensor::Tensor>& weights() const {
            return weights_;
        }

        [[nodiscard]] std::vector<tensor::Tensor>& weights() {
            return weights_;
        }

        void set_weights(const std::vector<tensor::Tensor>& weights) {
            weights_ = weights;
        }

        void add_weight(const tensor::Tensor& weight) {
            weights_.push_back(weight);
        }

        [[nodiscard]] const tensor::Tensor& weight(size_t idx) const {
            CHECK_LT(idx, weights_.size());
            return weights_[idx];
        }

        tensor::Tensor& weight(size_t idx) {
            CHECK_LT(idx, weights_.size());
            return weights_[idx];
        }

        [[nodiscard]] size_t weight_count() const {
            return weights_.size();
        }

        // Scales 管理（用于量化）
        [[nodiscard]] const tensor::Tensor& scales() const {
            return scales_;
        }

        tensor::Tensor& scales() {
            return scales_;
        }

        void set_scales(const tensor::Tensor& scales) {
            scales_ = scales;
        }

        [[nodiscard]] bool has_scales() const {
            return !scales_.is_empty();
        }

        // Group size 管理（用于分组量化）
        [[nodiscard]] size_t group_size() const {
            return group_size_;
        }

        void set_group_size(size_t group_size) {
            group_size_ = group_size;
        }

        [[nodiscard]] bool is_quantized() const {
            return group_size_ > 0;
        }

        // 初始化检查
        [[nodiscard]] base::Status check_weights() const {
            for (size_t i = 0; i < weights_.size(); ++i) {
                if (weights_[i].is_empty()) {
                    return base::Status{
                        base::ReturnStatus::InvalidArgument,
                        "Weight " + std::to_string(i) + " is empty"
                    };
                }
                if (weights_[i].get_data_type() != data_type_) {
                    return base::Status{
                        base::ReturnStatus::InvalidArgument,
                        "Weight " + std::to_string(i) + " data type mismatch"
                    };
                }
            }
            return base::Status{base::ReturnStatus::Success};
        }

    protected:
        size_t group_size_ = 0;                    // 量化分组大小，0表示不量化
        tensor::Tensor scales_;                    // 量化缩放因子
        std::vector<tensor::Tensor> weights_;      // 权重张量列表
    };
}


#endif //QWEN3INFER_LAYER_H