//
// Created by Administrator on 2026/2/24.
//


#include <cstdlib>

#include "../../include/base/alloc.h"


namespace qwi::base {
    CpuDeviceAllocator::CpuDeviceAllocator(
        ) : DeviceAllocator(DeviceType::kDeviceCPU) {}

    MemoryBuffer CpuDeviceAllocator::allocate(size_t byte_size) const {
        if (!byte_size) {
            return MemoryBuffer();
        }

#ifdef HAVE_POSIX_MEMALIGN
        void* data = nullptr;
        const size_t alignment = byte_size >= static_cast<size_t>(1024) ?
        static_cast<size_t>(32) : static_cast<size_t>(16);
        const int status = posix_memalign(
            &data,
            alignment,
            byte_size
        );
        if (status != 0) {
            return MemoryBuffer();
        }
        return MemoryBuffer(
            data, byte_size, true
        );
#else
        void* data = malloc(byte_size);
        return data;
#endif
    }

    void CpuDeviceAllocator::release(MemoryBuffer &ptr) const {
        if (ptr.data) {
            free(ptr.data);
            ptr.byte_size = 0;
            ptr.busy = false;
        }
    }

    std::shared_ptr<CpuDeviceAllocator> CpuDeviceAllocatorFactory::instance_ = nullptr;
}
