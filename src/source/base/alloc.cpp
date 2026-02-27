//
// Created by Administrator on 2026/2/26.
//


#include "../../include/base/alloc.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>


namespace qwi::base {
    void DeviceAllocator::memcpy(
        const void *src, void *dst,
        size_t byte_size, MemcpyKind memcpy_kind,
        void *stream, bool need_sync
    ) const {
        CHECK_NE(src, nullptr);
        CHECK_NE(dst, nullptr);
        if (byte_size <= 0) {
            return;
        }

        cudaStream_t cur_stream = nullptr;
        if (stream) {
            cur_stream = static_cast<CUstream_st*>(stream);
        }
        switch (memcpy_kind) {
            case MemcpyKind::kMemcpyHost2Host: {
                std::memcpy(dst, src, byte_size);
                break;
            }
            case MemcpyKind::kMemcpyHost2Device: {
                if (!cur_stream) {
                    cudaMemcpy(
                        dst, src, byte_size,
                        cudaMemcpyHostToDevice
                    );
                } else {
                    cudaMemcpyAsync(
                        dst, src, byte_size,
                        cudaMemcpyHostToDevice,
                        cur_stream
                    );
                }
                break;
            }
            case MemcpyKind::kMemcpyDevice2Host: {
                if (!cur_stream) {
                    cudaMemcpy(
                        dst, src, byte_size,
                        cudaMemcpyDeviceToHost
                    );
                } else {
                    cudaMemcpyAsync(
                        dst, src, byte_size,
                        cudaMemcpyDeviceToHost,
                        cur_stream
                    );
                }
                break;
            }
            case MemcpyKind::kMemcpyDevice2Device: {
                if (!cur_stream) {
                    cudaMemcpy(
                        dst, src, byte_size,
                        cudaMemcpyDeviceToDevice
                    );
                } else {
                    cudaMemcpyAsync(
                        dst, src, byte_size,
                        cudaMemcpyDeviceToDevice,
                        cur_stream
                    );
                }
                break;
            }
            default:
                LOG(FATAL) << "Unknown MemcpyKind! " << static_cast<int>(memcpy_kind);
        }

        if (need_sync) {
            cudaStreamSynchronize(cur_stream);
        }
    }

    void DeviceAllocator::memset_zero(
        void *ptr, size_t byte_size,
        void *stream, bool need_sync
    ) {
        CHECK(this->device_type_ != DeviceType::kDeviceUnknown);
        if (this->device_type_ == DeviceType::kDeviceCPU) {
            std::memset(ptr, 0, byte_size);
        } else {
            if (stream) {
                const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
                cudaMemsetAsync(ptr, 0, byte_size, cuda_stream);
                if (need_sync) {
                    cudaStreamSynchronize(cuda_stream);
                }
            } else {
                cudaMemset(ptr, 0, byte_size);
            }
        }
    }
}
