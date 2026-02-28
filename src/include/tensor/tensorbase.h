//
// Created by Administrator on 2026/2/27.
//

#ifndef QWEN3INFER_TENSORBASE_H
#define QWEN3INFER_TENSORBASE_H


#include <numeric>

#include "../base/buffer.h"


namespace qwi::tensor {
    class Tensor {
        std::shared_ptr<base::Buffer> buffer_;
        std::vector<size_t> dims_;
        size_t ndims_ = 0;
        size_t size_ = 0;
        base::DataType data_type_ = base::DataType::kDataUnknown;
    public:
        explicit Tensor() = default;
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, bool need_alloc = false,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            bool need_alloc = false,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            size_t dim2, bool need_alloc = false,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            size_t dim2, size_t dim3,
            bool need_alloc = false,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            std::vector<size_t> dims,
            bool need_alloc = false,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator>& allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        base::ReturnStatus init_buffer(
            const std::shared_ptr<base::DeviceAllocator>& allocator,
            base::DeviceType device_type,
            bool need_alloc,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        base::ReturnStatus allocate(
            const std::shared_ptr<base::DeviceAllocator>& allocator,
            size_t byte_size,
            base::DeviceType device_type
        );
        [[nodiscard]] size_t byte_size() const;
    };

    Tensor empty(
        std::vector<size_t> dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    Tensor zeros(
        std::vector<size_t> dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    Tensor ones(
        std::vector<size_t> dims,
        base::DataType data_type = base::DataType::kDataFloat32,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    );

    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return size_t{0};
        }

        auto size = std::accumulate(
            begin, end, init,
            std::multiplies<>()
        );

        return size_t{size};
    }
}


#endif //QWEN3INFER_TENSORBASE_H