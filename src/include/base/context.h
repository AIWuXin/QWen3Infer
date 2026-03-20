//
// Created by Administrator on 2026/3/18.
//

#ifndef QWEN3INFER_CONTEXT_H
#define QWEN3INFER_CONTEXT_H


#include <cuda_runtime.h>

namespace qwi::base {
    class CudaContext {
    public:
        // 禁止拷贝
        CudaContext(const CudaContext&) = delete;
        CudaContext& operator=(const CudaContext&) = delete;

        // 获取当前线程的默认 Stream
        static cudaStream_t current_stream() {
            return instance().stream_;
        }

        // Stream 切换 guard
        class StreamGuard {
            cudaStream_t prev_stream_;
        public:
            explicit StreamGuard(cudaStream_t new_stream)
                : prev_stream_(CudaContext::current_stream()) {
                instance().stream_ = new_stream;
            }
            ~StreamGuard() {
                instance().stream_ = prev_stream_;
            }
        };

        // 同步当前 Stream
        static void synchronize() {
            if (const auto err = cudaStreamSynchronize(current_stream()); err != cudaSuccess) {
                LOG(ERROR) << "CUDA kernel execution failed: " << cudaGetErrorString(err);
            }
        }

    private:
        CudaContext() {
            int device = 0;
            cudaError_t err = cudaGetDevice(&device);  // 检查是否有激活的设备
            if (err != cudaSuccess) {
                // 没有激活的设备，设置默认设备
                cudaSetDevice(0);
            }
            // cudaStreamCreate(&stream_);
            stream_ = nullptr;
        }
        ~CudaContext() {
            // cudaStreamDestroy(stream_);
        }

        static CudaContext& instance() {
            thread_local CudaContext ctx;
            return ctx;
        }

        cudaStream_t stream_;
    };

} // namespace qwi::base


#endif //QWEN3INFER_CONTEXT_H