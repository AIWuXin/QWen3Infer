//
// Created by Administrator on 2026/2/24.
//

#ifndef QWEN3INFER_TYPE_EXTENSION_HPP
#define QWEN3INFER_TYPE_EXTENSION_HPP


#include <glog/logging.h>


namespace qwi::base {
    constexpr size_t KB = 1 << 10;   // 1024
    constexpr size_t MB = 1 << 20;   // 1048576
    constexpr size_t GB = 1 << 30;   // 1073741824
    constexpr size_t BIG_MEMORY_SIZE = size_t{1} * MB;

    enum class MemcpyKind {
        kMemcpyHost2Host = 0,
        kMemcpyHost2Device,
        kMemcpyDevice2Host,
        kMemcpyDevice2Device,
    };

    enum class DeviceType {
        kDeviceUnknown = 0,
        kDeviceCPU,
        kDeviceCUDA,
    };

    enum class BufferType {
        kBufferUnknown = 0,
        kBufferBig,
        kBufferSmall,
    };

    enum class ReturnStatus {
        Success = 0,  // 成功
        Error = -1,  // 失败
        AlreadyAllocated = 1,  // 已经分配过内存
        ErrorAllocating = -2,  // 分配内存时发生错误
        ZeroByteSize = 2,  // 零字节长度
        NoAllocator = -3,  // 无分配器
    };

    enum class DataType {
        kDataUnknown = 0,
        kDataFloat32,
        kDataFloat16,
        kDataFloat8,
        kDataInt32,
        kDataInt16,
        kDataInt8,
    };

    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
        NoCopyable(const NoCopyable&) = delete;
        NoCopyable& operator=(const NoCopyable&) = delete;
    };

    inline std::string device_type_to_str(DeviceType type) {
        switch (type) {
            case DeviceType::kDeviceUnknown:
                return "kDeviceUnknown";
            case DeviceType::kDeviceCUDA:
                return "kDeviceCUDA";
            case DeviceType::kDeviceCPU:
                return "kDeviceCPU";
            default:
                LOG(ERROR) << "Unknown device type" << std::endl;
                return "";
        }
    }
}


#endif //QWEN3INFER_TYPE_EXTENSION_HPP