//
// Created by Administrator on 2026/2/26.
//


#include "../../include/base/buffer.h"

#include <algorithm>
#include <cuda_runtime_api.h>


namespace qwi::base {
    Buffer::Buffer(
        const MemoryBuffer &memory_buffer,
        const std::shared_ptr<DeviceAllocator>& allocator,
        bool use_external
    ) : use_external_(use_external),
    allocator_(allocator), memory_buffer_(memory_buffer) {
        if (!memory_buffer.data && allocator) {
            this->memory_buffer_ = allocator->allocate(
                memory_buffer.byte_size
            );
            this->use_external_ = false;
        }
    }

    Buffer::~Buffer() {
        if (!this->use_external_) {
            if (this->memory_buffer_.data && this->allocator_) {
                this->allocator_->release(
                    this->memory_buffer_
                );
            }
        }
    }

    void *Buffer::get_ptr() {
        return this->memory_buffer_.data;
    }

    const void * Buffer::get_ptr() const {
        return this->memory_buffer_.data;
    }

    size_t Buffer::get_byte_size() const {
        return this->memory_buffer_.byte_size;
    }

    DeviceType Buffer::get_device_type() const {
        return this->memory_buffer_.device_type;
    }

    /***
     * @brief 将 Buffer 数据迁移到指定的 CUDA 设备
     *
     * 根据当前数据所在位置，执行相应的内存迁移：
     * - CPU -> CUDA: 分配设备内存，执行 Host-to-Device 拷贝
     * - CUDA -> CUDA (同设备): 直接返回，不做任何操作
     * - CUDA -> CUDA (跨设备): 通过 Host 中转执行设备间拷贝
     *
     * @param device_id 目标 CUDA 设备 ID (从 0 开始)
     *
     * @note 迁移完成后，原内存会被自动释放
     * @note 如果 byte_size 为 0 或 data 为空，仅更新设备类型标记
     * @note 跨设备拷贝使用 Host 内存中转，不依赖 P2P 支持
     *
     * @warning 如果目标设备内存分配失败，会记录 FATAL 日志并返回
     *
     * @code
     *   Buffer buf(...);  // 假设当前在 CPU
     *   buf.cuda(0);      // 迁移到 GPU 0
     *   buf.cuda(1);      // 从 GPU 0 迁移到 GPU 1
     * @endcode
     ***/
    void Buffer::cuda(const size_t device_id) {
        if (this->get_device_type() == DeviceType::kDeviceCUDA) {
            if (this->memory_buffer_.device_id == device_id) {
                return;
            }
        }

        if (this->memory_buffer_.byte_size == 0 || !this->memory_buffer_.data) {
            return;
        }

        auto cuda_allocator = CudaDeviceAllocatorFactory::get_instance();

        cudaSetDevice(device_id);
        auto new_buffer = cuda_allocator->allocate(
            this->memory_buffer_.byte_size
        );
        if (!new_buffer.data) {
            LOG(ERROR) << "Failed to allocate memory buffer on device " << device_id;
            return;
        }

        if (this->get_device_type() == DeviceType::kDeviceCPU) {
            cuda_allocator->memcpy(
                this->memory_buffer_.data,
                new_buffer.data,
                this->memory_buffer_.byte_size,
                MemcpyKind::kMemcpyHost2Device
            );
        } else if (this->get_device_type() == DeviceType::kDeviceCUDA) {
            if (this->memory_buffer_.device_id != device_id) {
                cudaSetDevice(this->memory_buffer_.device_id);

                auto cpu_allocator = CpuDeviceAllocatorFactory::get_instance();
                auto temp_buffer = cpu_allocator->allocate(
                    this->memory_buffer_.byte_size
                );
                cuda_allocator->memcpy(
                    this->memory_buffer_.data,
                    temp_buffer.data,
                    this->memory_buffer_.byte_size,
                    MemcpyKind::kMemcpyDevice2Host
                );
                cpu_allocator->release(temp_buffer);

                cudaSetDevice(device_id);

                cuda_allocator->memcpy(
                    temp_buffer.data,
                    new_buffer.data,
                    this->memory_buffer_.byte_size,
                    MemcpyKind::kMemcpyHost2Device
                );
            } else {
                cuda_allocator->memcpy(
                    this->memory_buffer_.data,
                    new_buffer.data,
                    this->memory_buffer_.byte_size,
                    MemcpyKind::kMemcpyDevice2Device
                );
            }
        } else {
            LOG(WARNING) << "Unknown Device Type!";
        }

        // 释放原内存
        if (this->allocator_) {
            this->allocator_->release(this->memory_buffer_);
        }

        // 更新状态
        this->memory_buffer_ = new_buffer;
        this->allocator_ = cuda_allocator;
        this->use_external_ = false;
    }

    void Buffer::cpu() {
        if (this->get_device_type() == DeviceType::kDeviceCPU) {
            return;
        }

        if (this->memory_buffer_.byte_size == 0 || !this->memory_buffer_.data) {
            return;
        }

        auto cpu_allocator = CpuDeviceAllocatorFactory::get_instance();
        auto new_buffer = cpu_allocator->allocate(
            this->memory_buffer_.byte_size
        );
        if (!new_buffer.data) {
            LOG(ERROR) << "Failed to allocate memory buffer on cpu " << std::endl;
            return;
        }

        cpu_allocator->memcpy(
            this->memory_buffer_.data,
            new_buffer.data,
            this->memory_buffer_.byte_size,
            MemcpyKind::kMemcpyDevice2Host
        );

        // 释放原内存
        if (this->allocator_) {
            this->allocator_->release(this->memory_buffer_);
        }

        // 更新状态
        this->memory_buffer_ = new_buffer;
        this->allocator_ = cpu_allocator;
        this->use_external_ = false;
    }

    ReturnStatus Buffer::allocate() {
        if (this->get_ptr()) {
            return ReturnStatus::AlreadyAllocated;
        }

        auto allocator = this->allocator_;
        const auto byte_size = this->memory_buffer_.byte_size;

        if (allocator && byte_size != 0) {
            this->use_external_ = false;
            auto buffer = allocator->allocate(
                this->memory_buffer_.byte_size
            );
            if (!buffer.data) {
                LOG(ERROR) << "Failed to allocate memory buffer on " <<
                    device_type_to_str(this->get_device_type()) << std::endl;
                return ReturnStatus::ErrorAllocating;
            }
            return ReturnStatus::Success;
        }

        if (byte_size == 0) {
            return ReturnStatus::ZeroByteSize;
        }

        if (!this->allocator_) {
            return ReturnStatus::NoAllocator;
        }

        return ReturnStatus::Error;
    }

    std::shared_ptr<Buffer> Buffer::get_shared_from_this() {
        return this->shared_from_this();
    }

    bool Buffer::is_external() const {
        return this->use_external_;
    }

    ReturnStatus Buffer::copy_from(const Buffer &other) {
        // 1. 检查源 buffer
        if (!other.get_ptr() || other.get_byte_size() == 0) {
            LOG(WARNING) << "Source buffer is empty";
            return ReturnStatus::ZeroByteSize;
        }

        // 2. 目标未分配时自动分配
        if (!this->get_ptr()) {
            auto state = this->allocate();
            if (state != ReturnStatus::Success) {
                return state;
            }
        }

        // 3. 计算拷贝大小（取较小值）
        size_t copy_size = std::min(this->get_byte_size(), other.get_byte_size());
        if (copy_size == 0) {
            return ReturnStatus::Success;
        }

        // 4. 根据设备类型执行拷贝
        auto src_type = other.get_device_type();
        auto dst_type = this->get_device_type();

        if (src_type == DeviceType::kDeviceCPU && dst_type == DeviceType::kDeviceCPU) {
            // CPU -> CPU
            auto allocator = CpuDeviceAllocatorFactory::get_instance();
            allocator->memcpy(
                other.get_ptr(), this->get_ptr(), copy_size,
                MemcpyKind::kMemcpyHost2Host
            );
        }
        else if (src_type == DeviceType::kDeviceCPU && dst_type == DeviceType::kDeviceCUDA) {
            // CPU -> CUDA: 设置目标设备
            cudaSetDevice(this->memory_buffer_.device_id);
            auto allocator = CudaDeviceAllocatorFactory::get_instance();
            allocator->memcpy(
                other.get_ptr(), this->get_ptr(), copy_size,
                MemcpyKind::kMemcpyHost2Device
            );
        }
        else if (src_type == DeviceType::kDeviceCUDA && dst_type == DeviceType::kDeviceCPU) {
            // CUDA -> CPU: 设置源设备
            cudaSetDevice(other.memory_buffer_.device_id);
            auto allocator = CudaDeviceAllocatorFactory::get_instance();
            allocator->memcpy(
                other.get_ptr(), this->get_ptr(), copy_size,
                MemcpyKind::kMemcpyDevice2Host
            );
        }
        else if (src_type == DeviceType::kDeviceCUDA && dst_type == DeviceType::kDeviceCUDA) {
            // CUDA -> CUDA
            if (other.memory_buffer_.device_id == this->memory_buffer_.device_id) {
                // 同设备
                cudaSetDevice(this->memory_buffer_.device_id);
                auto allocator = CudaDeviceAllocatorFactory::get_instance();
                allocator->memcpy(
                    other.get_ptr(), this->get_ptr(), copy_size,
                    MemcpyKind::kMemcpyDevice2Device
                );
            } else {
                // 跨设备: 通过 Host 内存中转
                cudaSetDevice(other.memory_buffer_.device_id);
                auto cpu_allocator = CpuDeviceAllocatorFactory::get_instance();
                auto temp_buffer = cpu_allocator->allocate(copy_size);
                if (!temp_buffer.data) {
                    LOG(ERROR) << "Failed to allocate temp buffer for cross-device copy";
                    return ReturnStatus::ErrorAllocating;
                }

                auto cuda_allocator = CudaDeviceAllocatorFactory::get_instance();
                // 源设备 -> Host
                cuda_allocator->memcpy(
                    other.get_ptr(), temp_buffer.data, copy_size,
                    MemcpyKind::kMemcpyDevice2Host
                );

                // 切换到目标设备
                cudaSetDevice(this->memory_buffer_.device_id);
                // Host -> 目标设备
                cuda_allocator->memcpy(
                    temp_buffer.data, this->get_ptr(), copy_size,
                    MemcpyKind::kMemcpyHost2Device
                );

                cpu_allocator->release(temp_buffer);
            }
        }
        else {
            LOG(WARNING) << "Unknown device type combination";
            return ReturnStatus::Error;
        }

        return ReturnStatus::Success;
    }

    ReturnStatus Buffer::copy_from(const Buffer *other) {
        if (!other) {
            LOG(WARNING) << "Source buffer pointer is null";
            return ReturnStatus::Error;
        }
        return this->copy_from(*other);
    }
}
