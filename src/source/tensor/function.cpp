//
// Created by Administrator on 2026/3/17.
//


#include <utility>

#include "../../include/tensor/function.h"
#include "../../include/ops/ops.h"


namespace qwi::tensor {
    Tensor empty(
        std::vector<size_t> dims,
        base::DataType data_type,
        base::DeviceType device_type
    ) {
        std::shared_ptr<base::DeviceAllocator> allocator = nullptr;

        if (device_type == base::DeviceType::kDeviceCPU) {
            allocator = base::CpuDeviceAllocatorFactory::get_instance();
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            allocator = base::CudaDeviceAllocatorFactory::get_instance();
        } else {
            LOG(ERROR) << "Unknown device type";
            throw std::runtime_error("Unknown device type");
        }

        auto tensor = Tensor(
            data_type, std::move(dims), device_type,
            allocator, nullptr
        );

        return tensor;
    }

    Tensor zeros(
        const std::vector<size_t> &dims,
        const base::DataType data_type,
        const base::DeviceType device_type
    ) {
        const auto tensor = empty(dims, data_type, device_type);
        auto status = tensor.fill(0.0, INT_MIN, tensor.size());
        if (!status) {
            LOG(ERROR) << "Tensor::fill() failed with" << status.get_message();
        }

        return tensor;
    }

    Tensor ones(
        const std::vector<size_t> &dims,
        const base::DataType data_type,
        const base::DeviceType device_type
    ) {
        const auto tensor = empty(dims, data_type, device_type);
        auto status = tensor.fill(1.0, INT_MIN, tensor.size());
        if (!status) {
            LOG(ERROR) << "Tensor::fill() failed with" << status.get_message();
        }

        return tensor;
    }

    namespace detail {
        // TODO: 后续可扩展广播
        inline bool check_same_shape(
            const Tensor& a, const Tensor& b
        ) {
            return a.dims() == b.dims();
        }

        inline Tensor create_like(const Tensor& a) {
            return empty(a.dims(), a.get_data_type(), a.get_device_type());
        }

        inline base::Status run_elementwise(
            const Tensor& a,
            const Tensor& b,
            Tensor& out,
            base::ElementWiseType op_type
        ) {
            if (!check_same_shape(a, b)) {
                return base::Status{
                    base::ReturnStatus::InvalidArgument,
                    "Shape mismatch for elementwise operation"
                };
            }

            ops::Elementwise layer(
                a.get_data_type(),
                "elementwise_op",
                a.get_device_type(),
                op_type
            );

            layer.set_input(0, a);
            layer.set_input(1, b);
            layer.set_output(0, out);

            return layer.forward();
        }

        inline base::Status run_elementwise_scalar(
            const Tensor& a,
            double scalar,
            Tensor& out,
            base::ElementWiseType op_type
        ) {
            // 创建标量 tensor
            Tensor b = empty({1}, a.get_data_type(), a.get_device_type());
            auto status = b.fill(scalar, INT_MIN, 1);
            if (!status) return status;

            // 广播：需要扩展 b 到 a 的形状
            // 简化处理：暂时用 fill 填充整个 out
            status = out.fill(scalar, INT_MIN, out.size());
            if (!status) return status;

            // 然后执行 elementwise
            ops::Elementwise layer(
                a.get_data_type(),
                "elementwise_scalar",
                a.get_device_type(),
                op_type
            );
            layer.set_input(0, a);
            layer.set_input(1, out);  // out 已经被填充为标量值
            layer.set_output(0, out);

            return layer.forward();
        }

        inline base::Status run_reduction(
                    const Tensor& a,
                    Tensor& out,
                    int32_t dim,
                    base::ReductionType op_type
                ) {
            ops::Reduction layer(
                a.get_data_type(),
                "reduction",
                a.get_device_type(),
                op_type,
                dim
            );
            layer.set_input(0, a);
            layer.set_output(0, out);
            return layer.forward();
        }

        // 计算归约后的输出形状
        inline std::vector<size_t> calc_reduce_shape(
            const Tensor& a,
            int32_t dim,
            bool keepdim = true
        ) {
            auto dims = a.dims();
            if (dim < 0) dim += dims.size();

            std::vector<size_t> out_dims;
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i == static_cast<size_t>(dim)) {
                    if (keepdim) out_dims.push_back(1);
                } else {
                    out_dims.push_back(dims[i]);
                }
            }
            return out_dims;
        }
    }

// ========== 逐元素运算（返回新 Tensor）==========
    Tensor add(const Tensor &a, const Tensor &b) {
        Tensor out = detail::create_like(a);
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementAdd));
        return out;
    }

    Tensor sub(const Tensor &a, const Tensor &b) {
        Tensor out = detail::create_like(a);
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementSubtract));
        return out;
    }

    Tensor mul(const Tensor &a, const Tensor &b) {
        Tensor out = detail::create_like(a);
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementMultiply));
        return out;
    }

    Tensor div(const Tensor &a, const Tensor &b) {
        Tensor out = detail::create_like(a);
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementDivide));
        return out;
    }

    // ========== 逐元素运算（in-place）==========
    Tensor& add_(Tensor &a, const Tensor &b) {
        STATUS_CHECK(detail::run_elementwise(a, b, a, base::ElementWiseType::kElementAdd));
        return a;
    }

    Tensor& sub_(Tensor &a, const Tensor &b) {
        STATUS_CHECK(detail::run_elementwise(a, b, a, base::ElementWiseType::kElementSubtract));
        return a;
    }

    Tensor& mul_(Tensor &a, const Tensor &b) {
        STATUS_CHECK(detail::run_elementwise(a, b, a, base::ElementWiseType::kElementMultiply));
        return a;
    }

    Tensor& div_(Tensor &a, const Tensor &b) {
        STATUS_CHECK(detail::run_elementwise(a, b, a, base::ElementWiseType::kElementDivide));
        return a;
    }

    // ========== 逐元素运算（指定输出）==========
    Tensor add(const Tensor &a, const Tensor &b, Tensor &out) {
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementAdd));
        return out;
    }

    Tensor sub(const Tensor &a, const Tensor &b, Tensor &out) {
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementSubtract));
        return out;
    }

    Tensor mul(const Tensor &a, const Tensor &b, Tensor &out) {
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementMultiply));
        return out;
    }

    Tensor div(const Tensor &a, const Tensor &b, Tensor &out) {
        STATUS_CHECK(detail::run_elementwise(a, b, out, base::ElementWiseType::kElementDivide));
        return out;
    }

    // ========== 标量运算（返回新 Tensor）==========
    Tensor add(const Tensor &a, double scalar) {
        Tensor out = detail::create_like(a);
        // 先填充标量值，然后做 elementwise add
        auto status = out.fill(scalar, INT_MIN, out.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, out, out, base::ElementWiseType::kElementAdd));
        return out;
    }

    Tensor sub(const Tensor &a, double scalar) {
        Tensor out = detail::create_like(a);
        auto status = out.fill(scalar, INT_MIN, out.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, out, out, base::ElementWiseType::kElementSubtract));
        return out;
    }

    Tensor mul(const Tensor &a, double scalar) {
        Tensor out = detail::create_like(a);
        auto status = out.fill(scalar, INT_MIN, out.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, out, out, base::ElementWiseType::kElementMultiply));
        return out;
    }

    Tensor div(const Tensor &a, double scalar) {
        Tensor out = detail::create_like(a);
        auto status = out.fill(scalar, INT_MIN, out.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, out, out, base::ElementWiseType::kElementDivide));
        return out;
    }

    // ========== 标量运算（in-place）==========
    Tensor& add_(Tensor &a, double scalar) {
        Tensor tmp = detail::create_like(a);
        auto status = tmp.fill(scalar, INT_MIN, tmp.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, tmp, a, base::ElementWiseType::kElementAdd));
        return a;
    }

    Tensor& sub_(Tensor &a, double scalar) {
        Tensor tmp = detail::create_like(a);
        auto status = tmp.fill(scalar, INT_MIN, tmp.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, tmp, a, base::ElementWiseType::kElementSubtract));
        return a;
    }

    Tensor& mul_(Tensor &a, double scalar) {
        Tensor tmp = detail::create_like(a);
        auto status = tmp.fill(scalar, INT_MIN, tmp.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, tmp, a, base::ElementWiseType::kElementMultiply));
        return a;
    }

    Tensor& div_(Tensor &a, double scalar) {
        Tensor tmp = detail::create_like(a);
        auto status = tmp.fill(scalar, INT_MIN, tmp.size());
        STATUS_CHECK(status);
        STATUS_CHECK(detail::run_elementwise(a, tmp, a, base::ElementWiseType::kElementDivide));
        return a;
    }

    // ========== 全局归约运算（返回标量 Tensor）==========
    Tensor sum(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceSum));
        return out;
    }

    Tensor mean(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceMean));
        return out;
    }

    Tensor max(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceMax));
        return out;
    }

    Tensor min(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceMin));
        return out;
    }

    Tensor all(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceAll));
        return out;
    }

    Tensor any(const Tensor &a) {
        Tensor out = empty({1}, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, INT_MIN, base::ReductionType::kReduceAny));
        return out;
    }

    // ========== 按维度归约 ==========
    Tensor sum(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceSum));
        return out;
    }

    Tensor mean(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceMean));
        return out;
    }

    Tensor max(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceMax));
        return out;
    }

    Tensor min(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceMin));
        return out;
    }

    Tensor all(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceAll));
        return out;
    }

    Tensor any(const Tensor &a, int32_t dim) {
        auto out_dims = detail::calc_reduce_shape(a, dim, true);
        Tensor out = empty(out_dims, a.get_data_type(), a.get_device_type());
        STATUS_CHECK(detail::run_reduction(a, out, dim, base::ReductionType::kReduceAny));
        return out;
    }

    Tensor operator+(const Tensor &self, const Tensor &other) {
        return add(self, other);
    }

    Tensor operator-(const Tensor &self, const Tensor &other) {
        return sub(self, other);
    }

    Tensor operator*(const Tensor &self, const Tensor &other) {
        return mul(self, other);
    }

    Tensor operator/(const Tensor &self, const Tensor &other) {
        return div(self, other);
    }

    Tensor operator+(const Tensor &self, double scalar) {
        return add(self, scalar);
    }

    Tensor operator-(const Tensor &self, double scalar) {
        return sub(self, scalar);
    }

    Tensor operator*(const Tensor &self, double scalar) {
        return mul(self, scalar);
    }

    Tensor operator/(const Tensor &self, double scalar) {
        return div(self, scalar);
    }

    Tensor operator+(double scalar, const Tensor &self) {
        return add(self, scalar);  // 加法交换律
    }

    Tensor operator-(double scalar, const Tensor &self) {
        // scalar - tensor = -(tensor - scalar)
        Tensor neg_self = empty(self.dims(), self.get_data_type(), self.get_device_type());
        auto status = neg_self.fill(0, INT_MIN, 0);  // 先填充 0
        // neg_self = 0 - self
        sub_(neg_self, self);  // 简化实现
        // 然后 neg_self + scalar
        return add(neg_self, scalar);
    }

    Tensor operator*(double scalar, const Tensor &self) {
        return mul(self, scalar);  // 乘法交换律
    }

    Tensor operator/(double scalar, const Tensor &self) {
        // scalar / tensor，需要逐元素除法
        Tensor tmp = empty(self.dims(), self.get_data_type(), self.get_device_type());
        auto status = tmp.fill(scalar, INT_MIN, tmp.size());
        return div(tmp, self);
    }

    // Tensor 成员运算符
    Tensor Tensor::operator+=(const Tensor &other) {
        add_(*this, other);
        return *this;
    }

    Tensor Tensor::operator-=(const Tensor &other) {
        sub_(*this, other);
        return *this;
    }

    Tensor Tensor::operator*=(const Tensor &other) {
        mul_(*this, other);
        return *this;
    }

    Tensor Tensor::operator/=(const Tensor &other) {
        div_(*this, other);
        return *this;
    }

    Tensor Tensor::operator+=(double scalar) {
        add_(*this, scalar);
        return *this;
    }

    Tensor Tensor::operator-=(double scalar) {
        sub_(*this, scalar);
        return *this;
    }

    Tensor Tensor::operator*=(double scalar) {
        mul_(*this, scalar);
        return *this;
    }

    Tensor Tensor::operator/=(double scalar) {
        div_(*this, scalar);
        return *this;
    }
}
