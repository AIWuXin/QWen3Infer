//
// Created by Administrator on 2026/3/16.
//

#ifndef QWEN3INFER_RSMNORM_H
#define QWEN3INFER_RSMNORM_H

#include "layer.h"

namespace qwi::ops {
    class RmsNorm : public WeightedLayer {
    public:
        explicit RmsNorm(
            base::DataType data_type,
            std::string layer_name,
            base::DeviceType device_type,
            float eps = 1e-6f
        );
        [[nodiscard]] base::Status check() const override;
        base::Status forward() override;
        [[nodiscard]] float get_eps() const;
    private:
        float eps_;
    };
}

#endif //QWEN3INFER_RSMNORM_H
