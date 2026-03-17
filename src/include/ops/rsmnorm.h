//
// Created by Administrator on 2026/3/10.
//

#ifndef QWEN3INFER_RSMNORM_H
#define QWEN3INFER_RSMNORM_H


#include "layer.h"


namespace qwi::ops {
    class RmsNorm: public WeightedLayer {
    public:
        explicit RmsNorm(
            const base::DeviceType device_type,
            const base::DataType data_type,
            const size_t dim,
            std::string layer_name = ""
        ) : WeightedLayer(
            device_type,
            base::LayerType::kLayerRMSNorm,
            data_type,
            std::move(layer_name)
        ), dim_(dim) {}
        [[nodiscard]] base::Status check() const override;
        base::Status forward() override;
    private:
        size_t dim_;
    };
}


#endif //QWEN3INFER_RSMNORM_H