//
// Created by Administrator on 2026/3/4.
//

#ifndef QWEN3INFER_ELEMENTWISE_H
#define QWEN3INFER_ELEMENTWISE_H


#include "layer.h"


namespace qwi::ops {
    class Elementwise : public CommonLayer {
    public:
        explicit Elementwise(
            base::DataType data_type,
            std::string layer_name,
            base::DeviceType device_type
        );
        [[nodiscard]] base::Status check() const override;
        base::Status forward() override;
    };
}


#endif //QWEN3INFER_ELEMENTWISE_H