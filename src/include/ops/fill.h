//
// Created by Administrator on 2026/3/16.
//

#ifndef QWEN3INFER_FILL_H
#define QWEN3INFER_FILL_H


#include "layer.h"


namespace qwi::ops {
    class Fill: public CommonLayer {
    public:
        explicit Fill(
            const base::DataType data_type,
            const double value,
            const int32_t dim,
            const size_t count,
            const base::DeviceType device_type,
            std::string layer_name
        ) : CommonLayer(
            device_type,
            base::LayerType::kLayerFill,
            data_type,
            std::move(layer_name)
        ), dim_(dim), count_(count), value_(value) {
            this->inputs_.resize(size_t{1});
        }
        [[nodiscard]] base::Status check() const override;
        base::Status forward() override;
        [[nodiscard]] double get_value() const;
        [[nodiscard]] size_t dim() const;
        [[nodiscard]] size_t count() const;
        void set_value(double value);
        void set_dim(size_t dim);
        void set_count(size_t count);
    private:
        int32_t dim_;
        size_t count_;
        double value_;
    };
}


#endif //QWEN3INFER_FILL_H