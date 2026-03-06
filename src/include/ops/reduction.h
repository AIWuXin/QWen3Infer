//
// Created by Administrator on 2026/3/6.
//

#ifndef QWEN3INFER_REDUCTION_H
#define QWEN3INFER_REDUCTION_H


#include "layer.h"


namespace qwi::ops::kernel {
    class Reduction : public CommonLayer {
    public:
        explicit Reduction(
            base::DataType data_type,
            std::string layer_name,
            base::DeviceType device_type,
            base::ReductionType op_type = base::ReductionType::kReduceSum,
            int32_t dim = -1
        );
        [[nodiscard]] base::Status check() const override;
        base::Status forward() override;
        base::ReductionType get_op_type() const;
        int32_t get_dim() const;
    private:
        base::ReductionType op_type_;
        int32_t dim_;
    };
}


#endif //QWEN3INFER_REDUCTION_H