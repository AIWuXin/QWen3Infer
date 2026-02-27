//
// Created by Administrator on 2026/2/24.
//

#ifndef QWEN3INFER_BUFFER_H
#define QWEN3INFER_BUFFER_H


#include "alloc.h"


namespace qwi::base {
    class Buffer: NoCopyable, std::enable_shared_from_this<Buffer> {
        bool use_external_ = false;
        std::shared_ptr<DeviceAllocator> allocator_;
        MemoryBuffer memory_buffer_;

    public:
        explicit Buffer() = default;
        explicit Buffer(
            const MemoryBuffer &memory_buffer,
            const std::shared_ptr<DeviceAllocator>& allocator,
            bool use_external = false
        );
        virtual ~Buffer();
        void* get_ptr();
        const void* get_ptr() const;
        size_t get_byte_size() const;
        DeviceType get_device_type() const;
        void cuda(size_t device_id = 0);
        void cpu();
        ReturnStatus allocate();
        std::shared_ptr<Buffer> get_shared_from_this();
        bool is_external() const;
        ReturnStatus copy_from(const Buffer& other);
        ReturnStatus copy_from(const Buffer* other);
    };
}


#endif //QWEN3INFER_BUFFER_H