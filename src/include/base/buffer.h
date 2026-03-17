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

        /**
         * @brief 将 Buffer 内存清零 (同步操作)
         *
         * 根据 Buffer 所在设备类型，使用最优方式将内存全部置零：
         * - CPU: 使用 std::memset
         * - CUDA: 使用 cudaMemset，自动设置目标设备上下文
         *
         * @return ReturnStatus::Success 成功清零
         * @return ReturnStatus::NotInitialized Buffer 未分配内存或数据指针为空
         *
         * @note 这是同步操作，会等待 CUDA 操作完成
         * @note 对于外部内存 (use_external_=true)，同样适用
         *
         * @code
         *   Buffer buf(DataType::kDataFloat32, 1024, allocator);
         *   buf.allocate();
         *   buf.memset_zero();  // 清零
         * @endcode
         */
        ReturnStatus memset_zero_sync(
            void* stream = nullptr
        );

        ReturnStatus memset_zero_async(
            void* stream = nullptr
        );

        /**
         * @brief 将 Buffer 内存清零 (支持异步流)
         *
         * 功能同 memset_zero()，但允许指定 CUDA 流进行异步操作。
         * 对于 CPU Buffer，stream 参数被忽略，仍同步执行。
         *
         * @param stream CUDA 流指针 (cudaStream_t)，nullptr 表示默认流同步执行
         * @param need_sync 是否需要在函数返回前同步流 (仅 CUDA 有效)
         *
         * @return ReturnStatus::Success 成功发起清零操作
         * @return ReturnStatus::NotInitialized Buffer 未分配内存或数据指针为空
         *
         * @note 当 need_sync=false 且 stream!=nullptr 时，函数立即返回，操作在后台执行
         * @note 异步操作后需确保 Buffer 生命周期覆盖操作完成，避免内存被提前释放
         *
         * @code
         *   cudaStream_t stream;
         *   cudaStreamCreate(&stream);
         *   buf.memset(stream, false);  // 异步清零
         *   // ... 其他不依赖 buf 数据的并发操作 ...
         *   cudaStreamSynchronize(stream);  // 确保清零完成
         *   cudaStreamDestroy(stream);
         * @endcode
         */
        ReturnStatus memset_zero(void* stream = nullptr, bool need_sync = true);
    };
}


#endif //QWEN3INFER_BUFFER_H