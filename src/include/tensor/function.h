//
// Created by Administrator on 2026/3/16.
//

#ifndef QWEN3INFER_FUNCTION_H
#define QWEN3INFER_FUNCTION_H


#include "tensorbase.h"


namespace qwi::tensor {
    Tensor empty(
        std::vector<size_t> dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    Tensor zeros(
        const std::vector<size_t> &dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    Tensor ones(
        const std::vector<size_t> &dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    Tensor add(const Tensor &a, const Tensor &b);
    Tensor sub(const Tensor &a, const Tensor &b);
    Tensor mul(const Tensor &a, const Tensor &b);
    Tensor div(const Tensor &a, const Tensor &b);

    Tensor& add_(Tensor &a, const Tensor &b);  // 带 _ 后缀表示 in-place
    Tensor& sub_(Tensor &a, const Tensor &b);
    Tensor& mul_(Tensor &a, const Tensor &b);
    Tensor& div_(Tensor &a, const Tensor &b);

    Tensor add(const Tensor &a, const Tensor &b, Tensor &out);
    Tensor sub(const Tensor &a, const Tensor &b, Tensor &out);
    Tensor mul(const Tensor &a, const Tensor &b, Tensor &out);
    Tensor div(const Tensor &a, const Tensor &b, Tensor &out);

    Tensor add(const Tensor &a, double scalar);  // a + 3.14
    Tensor sub(const Tensor &a, double scalar);
    Tensor mul(const Tensor &a, double scalar);  // a * 2.0
    Tensor div(const Tensor &a, double scalar);

    Tensor& add_(Tensor &a, double scalar);
    Tensor& sub_(Tensor &a, double scalar);
    Tensor& mul_(Tensor &a, double scalar);
    Tensor& div_(Tensor &a, double scalar);

    Tensor sum(const Tensor &a);
    Tensor mean(const Tensor &a);
    Tensor max(const Tensor &a);
    Tensor min(const Tensor &a);
    Tensor all(const Tensor &a);
    Tensor any(const Tensor &a);

    Tensor sum(const Tensor &a, int32_t dim);
    Tensor mean(const Tensor &a, int32_t dim);
    Tensor max(const Tensor &a, int32_t dim);
    Tensor min(const Tensor &a, int32_t dim);
    Tensor all(const Tensor &a, int32_t dim);
    Tensor any(const Tensor &a, int32_t dim);

}


#endif //QWEN3INFER_FUNCTION_H