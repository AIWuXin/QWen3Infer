//
// Created by Administrator on 2026/2/24.
//

#ifndef QWEN3INFER_TYPE_EXTENSION_HPP
#define QWEN3INFER_TYPE_EXTENSION_HPP


#include <cuda_runtime_api.h>
#include <driver_types.h>
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
        NotImplement = -4,  // 未实现的函数
        InvalidArgument = -5  // 非法的参数
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

    enum class LayerType {
        kLayerUnknown = 0,
        kLayerLinear,
        kLayerConvolution,
        kLayerElementWise,
        kLayerMatMul,
        kLayerRMSNorm,
        kLayerSoftmax,
        kLayerRope,
        kLayerSwiGelu,
        kLayerMultiHeadAttention,
    };

    enum class ElementWiseType {
        kElementWiseUnknown = 0,
        kElementAdd,
        kElementSubtract,
        kElementMultiply,
        kElementDivide,
    };

    class NoCopyable {
    protected:
        NoCopyable() = default;
        ~NoCopyable() = default;
    public:
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

    inline size_t data_type_to_size(DataType type) {
        switch (type) {
            case DataType::kDataFloat32:
                return sizeof(float);
            case DataType::kDataFloat16:
                return sizeof(uint16_t);
            case DataType::kDataFloat8:
                return sizeof(uint8_t);
            case DataType::kDataInt32:
                return sizeof(int32_t);
            case DataType::kDataInt16:
                return sizeof(int16_t);
            case DataType::kDataInt8:
                return sizeof(int8_t);
            default:
                return 0;
        }
    }

    class Status {
    public:
        explicit Status(
            ReturnStatus code = ReturnStatus::Success,
            std::string err_message = ""
        ) : code_(code), message_(std::move(err_message)) {}
        Status(const Status& other) = default;
        Status& operator=(const Status& other) = default;
        Status& operator=(const ReturnStatus code){
            this->code_ = code;
            return *this;
        }

        bool operator==(const ReturnStatus code) const {
            if (code_ == code) {
                return true;
            }
            return false;
        }

        bool operator!=(const ReturnStatus code) const {
            return !(*this == code);
        }

        explicit operator int() const {
            return static_cast<int>(this->code_);
        }

        explicit operator bool() const {
            return static_cast<int>(this->code_) >= 0;
        }

        [[nodiscard]] ReturnStatus get_code() const {
            return this->code_;
        }

        [[nodiscard]] const std::string& get_message() const {
            return this->message_;
        }

        void set_err_msg(
            const std::string& err_msg
        ) {
            this->message_ = err_msg;
        }
    private:
        ReturnStatus code_ = ReturnStatus::Success;
        std::string message_;
    };

#define STATUS_CHECK(call)                                                               \
    do {                                                                                 \
        const base::Status& status = call;                                               \
        if (!status) {                                                                   \
            const size_t buf_size = 512;                                                 \
            char buf[buf_size];                                                          \
            snprintf(buf, buf_size - 1,                                                  \
            "Infer error\n File:%s Line:%d\n Error code:%d\n Error msg:%s\n", __FILE__,  \
            __LINE__, int(status), status.get_message().c_str());                        \
            LOG(FATAL) << buf;                                                           \
        }                                                                                \
    } while (0)

    struct CudaConfig {
        cudaStream_t stream = nullptr;
        ~CudaConfig() {
            if (stream) {
                cudaStreamDestroy(stream);
            }
        }
    };

#define UNUSED(expr)   \
    do {               \
        (void)(expr);  \
    } while (0)

    // 模板化的操作函数
    template<ElementWiseType Op, typename T>
    T element_wise_op(T a, T b) {
        if constexpr (Op == ElementWiseType::kElementAdd) return a + b;
        else if constexpr (Op == ElementWiseType::kElementSubtract) return a - b;
        else if constexpr (Op == ElementWiseType::kElementMultiply) return a * b;
        else if constexpr (Op == ElementWiseType::kElementDivide) return a / b;
        else throw std::runtime_error("Unknown element wise type");
    }
}


#endif //QWEN3INFER_TYPE_EXTENSION_HPP