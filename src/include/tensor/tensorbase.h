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
        base::ReturnStatus init_buffer(
            const std::shared_ptr<base::DeviceAllocator>& allocator,
            base::DeviceType device_type,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
    public:
        explicit Tensor() = default;
        explicit Tensor(
            base::DataType data_type,
            size_t dim0,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            size_t dim2,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            size_t dim0, size_t dim1,
            size_t dim2, size_t dim3,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator> &allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        explicit Tensor(
            base::DataType data_type,
            std::vector<size_t> dims,
            base::DeviceType device_type = base::DeviceType::kDeviceCPU,
            const std::shared_ptr<base::DeviceAllocator>& allocator = nullptr,
            const std::shared_ptr<base::Buffer> &buffer = nullptr
        );
        base::ReturnStatus allocate(
            const std::shared_ptr<base::DeviceAllocator>& allocator,
            size_t byte_size,
            base::DeviceType device_type
        );
        [[nodiscard]] size_t byte_size() const;
        // TODO: 待实现类型转换
        base::ReturnStatus to_float();  // 不是真正的类型转换，待实现
        // base::ReturnStatus to_float16();
        // base::ReturnStatus to_float8();
        // base::ReturnStatus to_int32();
        // base::ReturnStatus to_int16();
        // base::ReturnStatus to_int8();
        template <typename T>
        T* ptr();
        template <typename T>
        const T* ptr() const;
        base::ReturnStatus set_dims(
            std::vector<size_t> dims
        );
        [[nodiscard]] std::vector<size_t> strides() const;
        template <typename T>
        T& index(size_t offset);
        template <typename T>
        const T& index(size_t offset) const;
        template <typename T>
        T& index(const std::vector<size_t>& indices);
        template <typename T>
        const T& index(const std::vector<size_t>& indices) const;
        template <typename T>
        T& operator[](std::vector<size_t> indices);
        template <typename T>
        const T& operator[](std::vector<size_t> indices) const;
        [[nodiscard]] size_t ndims() const;
        [[nodiscard]] bool is_empty() const;
        [[nodiscard]] base::DataType get_data_type() const;
        [[nodiscard]] base::DeviceType get_device_type() const;
        [[nodiscard]] size_t dim(size_t idx) const;
        [[nodiscard]] std::vector<size_t> dims() const;
        [[nodiscard]] size_t size() const;
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

    template<typename T>
    T *Tensor::ptr() {
        if (!this->buffer_ || !this->buffer_->get_ptr()) {
            return nullptr;
        }
        return static_cast<T *>(this->buffer_->get_ptr());
    }

    template<typename T>
    const T *Tensor::ptr() const {
        if (!this->buffer_ || !this->buffer_->get_ptr()) {
            return nullptr;
        }
        return const_cast<const T *>(
            static_cast<T *>(this->buffer_->get_ptr())
        );
    }

    template<typename T>
    T & Tensor::index(const size_t offset) {
        if (offset >= this->size_) {
            LOG(FATAL) << "index out of bounds" << std::endl;
            throw std::out_of_range("index out of bounds");
        }
        T& val = *(
            static_cast<T *>(this->buffer_->get_ptr()) + offset
        );
        return val;
    }

    template<typename T>
    const T & Tensor::index(const size_t offset) const {
        if (offset >= this->size_) {
            LOG(FATAL) << "index out of bounds" << std::endl;
            throw std::out_of_range("index out of bounds");
        }

        const T& val = *(
            static_cast<T *>(this->buffer_->get_ptr()) + offset
        );
        return val;
    }

    template<typename T>
    T & Tensor::index(const std::vector<size_t>& indices) {
        if (indices.size() != this->ndims()) {
            LOG(FATAL) << "indices size must equal ndims!" << std::endl;
            throw std::out_of_range("indices size must equal ndims!");
        }

        for (int idx = 0; idx < this->ndims(); ++idx) {
            if (indices[idx] >= this->dims_[idx]) {
                LOG(FATAL) << "index out of bounds at dimension "
                << idx << "!" << std::endl;
                throw std::out_of_range("index out of bounds");
            }
        }

        auto strides = this->strides();
        size_t offset = strides[0] * indices[0];
        for (size_t idx = 1; idx < this->ndims(); ++idx) {
            offset += strides[idx] * indices[idx];
        }

        T& val = *(
            static_cast<T *>(this->buffer_->get_ptr()) + offset
        );
        return val;
    }

    template<typename T>
    const T & Tensor::index(const std::vector<size_t>& indices) const {
        if (indices.size() != this->ndims()) {
            LOG(FATAL) << "indices size must equal ndims!" << std::endl;
            throw std::out_of_range("indices size must equal ndims!");
        }

        for (int idx = 0; idx < this->ndims(); ++idx) {
            if (indices[idx] >= this->dims_[idx]) {
                LOG(FATAL) << "index out of bounds at dimension "
                << idx << "!" << std::endl;
                throw std::out_of_range("index out of bounds");
            }
        }

        auto strides = this->strides();
        size_t offset = strides[0] * indices[0];
        for (size_t idx = 1; idx < this->ndims(); ++idx) {
            offset += strides[idx] * indices[idx];
        }

        const T& val = *(
            static_cast<T *>(this->buffer_->get_ptr()) + offset
        );
        return val;
    }

    template<typename T>
    T & Tensor::operator[](const std::vector<size_t> indices) {
        return this->index<T>(indices);
    }

    template<typename T>
    const T & Tensor::operator[](
        const std::vector<size_t> indices
    ) const {
        return this->index<T>(indices);
    }
}


#endif //QWEN3INFER_TENSORBASE_H