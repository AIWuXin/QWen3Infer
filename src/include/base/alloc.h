//
// Created by Administrator on 2026/2/24.
//

#ifndef QWEN3INFER_ALLOC_H
#define QWEN3INFER_ALLOC_H


#include <map>

#include "type_extension.hpp"


namespace qwi::base {
    struct MemoryBuffer {
        void* data = nullptr;
        size_t byte_size = 0;
        bool busy = false;
        DeviceType device_type = DeviceType::kDeviceUnknown;
        size_t device_id = 0;

        MemoryBuffer() = default;
        MemoryBuffer(
            void* data, size_t byte_size,
            bool busy, size_t device_id = 0,
            DeviceType device_type = DeviceType::kDeviceCPU
        ) : data(data), byte_size(byte_size),
        busy(busy), device_id(device_id), device_type(device_type) {}
    };

    class DeviceAllocator {
        DeviceType device_type_ = DeviceType::kDeviceUnknown;
    public:
        explicit DeviceAllocator(
            const DeviceType device_type
        ) : device_type_(device_type) {}
        virtual ~DeviceAllocator() = default;
        [[nodiscard]] virtual DeviceType device_type() const {
            return device_type_;
        }
        [[nodiscard]] virtual MemoryBuffer allocate(size_t byte_size) const = 0;
        virtual void release(MemoryBuffer &ptr) const = 0;
        virtual void memcpy(
            const void* src,
            void* dst,
            size_t byte_size,
            MemcpyKind memcpy_kind = MemcpyKind::kMemcpyHost2Host,
            void* stream = nullptr,
            bool need_sync = false
        ) const;
        virtual void memset_zero(
            void* ptr,
            size_t byte_size,
            void* stream = nullptr,
            bool need_sync = false
        );
    };

    class CpuDeviceAllocator: public DeviceAllocator {
    public:
        explicit CpuDeviceAllocator();
        [[nodiscard]] MemoryBuffer allocate(size_t byte_size) const override;
        void release(MemoryBuffer &ptr) const override;
    };

    class CudaDeviceAllocator: public DeviceAllocator {
        mutable std::map<int, size_t> no_busy_size_;
        mutable std::map<int, std::vector<MemoryBuffer>> small_buffer_map_;
        // 每秒净增长（cudaMalloc - cudaFree）
        mutable std::map<int, int64_t> net_growth_;
        // 每秒缓存命中（用于辅助统计）
        mutable std::map<int, size_t> cache_hit_bytes_;
        mutable std::map<int, size_t> current_threshold_;
        mutable std::map<int, std::chrono::steady_clock::time_point> last_update_;
        size_t check_and_get_threshold(int device_id) const;
    public:
        explicit CudaDeviceAllocator();
        ~CudaDeviceAllocator() override;
        [[nodiscard]] MemoryBuffer allocate(size_t byte_size) const override;
        void release(MemoryBuffer &ptr) const override;
        static size_t get_device_total_memory(int device_id);
    };

    class CpuDeviceAllocatorFactory {
        static std::shared_ptr<CpuDeviceAllocator> instance_;
    public:
        static std::shared_ptr<CpuDeviceAllocator> get_instance() {
            if (CpuDeviceAllocatorFactory::instance_ == nullptr) {
                CpuDeviceAllocatorFactory::instance_ = std::make_shared<CpuDeviceAllocator>();
            }

            return CpuDeviceAllocatorFactory::instance_;
        }
    };

    class CudaDeviceAllocatorFactory {
        static std::shared_ptr<CudaDeviceAllocator> instance_;
    public:
        static std::shared_ptr<CudaDeviceAllocator> get_instance() {
            if (CudaDeviceAllocatorFactory::instance_ == nullptr) {
                CudaDeviceAllocatorFactory::instance_ = std::make_shared<CudaDeviceAllocator>();
            }

            return CudaDeviceAllocatorFactory::instance_;
        }
    };
}


#endif //QWEN3INFER_ALLOC_H