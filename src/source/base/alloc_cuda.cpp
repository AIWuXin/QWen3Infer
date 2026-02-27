//
// Created by Administrator on 2026/2/24.
//


#include <algorithm>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../../include/base/alloc.h"


namespace qwi::base {
    CudaDeviceAllocator::CudaDeviceAllocator(
    ) : DeviceAllocator(DeviceType::kDeviceCUDA) {
        // 获取显存
        int device_id = -1;
        cudaGetDeviceCount(&device_id);

        for (int idx = 0; idx < device_id; ++idx) {
            size_t total_bytes, free_bytes = 0;
            cudaSetDevice(idx);
            cudaMemGetInfo(&free_bytes, &total_bytes);

            // 计算初始阈值：10% 或 128MB 取大
            // size_t init_bytes = std::max(
            //     total_bytes / 10,
            //     size_t{128} * MB
            // );
            this->current_threshold_[idx] = size_t{0};
            this->net_growth_[idx] = 0;
            this->cache_hit_bytes_[idx] = 0;
            this->no_busy_size_[idx] = 0;

            // 初始化时间戳
            this->last_update_[idx] = std::chrono::steady_clock::now();
        }
    }

    MemoryBuffer CudaDeviceAllocator::allocate(size_t byte_size) const {
        int device_id = -1;
        cudaError_t state = cudaGetDevice(&device_id);
        CHECK(state == cudaSuccess);

        if (byte_size == 0) {
            return {};
        }

        if (byte_size > BIG_MEMORY_SIZE) {
            void* ptr = nullptr;
            state = cudaMalloc(&ptr, byte_size);
            if (state != cudaSuccess) {
                char info_buf[1<<10];
                snprintf(
                    info_buf, 1024,
                    "Error: CUDA error when allocating %lu MB memory! "
                    "maybe there's no enough memory left on device.",
                    byte_size >> 20
                );
                LOG(ERROR) << info_buf;
                return {};
            }

            const MemoryBuffer buffer(
                ptr, byte_size, true,
                device_id, DeviceType::kDeviceCUDA
            );
            return buffer;
        }

        auto &small_buffer = this->small_buffer_map_[device_id];
        for (size_t idx = 0; idx < small_buffer.size(); ++idx) {
            if (
                !small_buffer[idx].busy &&
                small_buffer[idx].byte_size >= byte_size
            ) {
                small_buffer[idx].busy = true;
                this->no_busy_size_[device_id] -= small_buffer[idx].byte_size;
                this->cache_hit_bytes_[device_id] += small_buffer[idx].byte_size;
                return small_buffer[idx];
            }
        }

        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (state != cudaSuccess) {
            char info_buf[1<<10];
            snprintf(
                info_buf, 1024,
                "Error: CUDA error when allocating %lu MB memory! "
                "maybe there's no enough memory left on device.",
                byte_size >> 20
            );
            LOG(ERROR) << info_buf;
            return {};
        }

        this->net_growth_[device_id] += byte_size;
        MemoryBuffer buffer(
            ptr, byte_size, true,
            device_id, DeviceType::kDeviceCUDA
        );
        small_buffer.emplace_back(
            buffer
        );
        return buffer;
    }

    size_t CudaDeviceAllocator::get_device_total_memory(int device_id) {
        size_t total_bytes, free_bytes = 0;
        cudaSetDevice(device_id);
        cudaMemGetInfo(&free_bytes, &total_bytes);

        return total_bytes;
    }

    size_t CudaDeviceAllocator::check_and_get_threshold(int device_id) const {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(
            now - this->last_update_[device_id]
        ).count();
        auto threshold = this->current_threshold_[device_id];
        if (elapsed < 1.0) return threshold;
        int growth = this->net_growth_[device_id];

        threshold = static_cast<size_t>(threshold + growth * 0.5);
        size_t total_mem = get_device_total_memory(device_id);
        threshold = std::clamp(
            threshold, size_t{64} * MB,
            static_cast<size_t>(total_mem * 0.9)
        );

        // 清零，开始下一秒
        net_growth_[device_id] = 0;
        cache_hit_bytes_[device_id] = 0;
        last_update_[device_id] = now;

        return threshold;
    }

    void CudaDeviceAllocator::release(MemoryBuffer &ptr) const {
        if (!ptr.data || ptr.device_type != DeviceType::kDeviceCUDA) {
            return;
        }

        cudaError_t state = cudaSuccess;

        if (ptr.byte_size <= BIG_MEMORY_SIZE) {
            const auto device_id = static_cast<int>(ptr.device_id);
            auto &small_buffers = this->small_buffer_map_[device_id];
            const auto threshold = this->check_and_get_threshold(device_id);
            const size_t last_threshold = this->current_threshold_[device_id];

            if (threshold < last_threshold) {
                std::vector<MemoryBuffer> temp_buffer;
                for (int idx = 0; idx < small_buffers.size(); ++idx) {
                    if (!small_buffers[idx].busy) {
                        cudaSetDevice(device_id);
                        state = cudaFree(small_buffers[idx].data);
                        if (state != cudaSuccess) {
                            char info_buf[1<<10];
                            snprintf(
                                info_buf, 1024,
                                "Error: CUDA error when release memory on device %lu",
                                static_cast<size_t>(device_id)
                            );
                            LOG(ERROR) << info_buf;
                        }
                    } else {
                        temp_buffer.push_back(small_buffers[idx]);
                    }
                }
                small_buffers.clear();
                this->small_buffer_map_[device_id] = std::move(temp_buffer);
                this->no_busy_size_[device_id] = 0;
                this->current_threshold_[device_id] = threshold;
                return;
            } else {
                for (int idx = 0; idx < small_buffers.size(); ++idx) {
                    if (ptr.data == small_buffers[idx].data) {
                        this->no_busy_size_[device_id] += small_buffers[idx].byte_size;
                        this->current_threshold_[device_id] = threshold;
                        small_buffers[idx].busy = false;

                        // 外部 ptr 置空（防止外部继续使用）
                        ptr = {};
                        return;
                    }
                }
                LOG(WARNING) << "Buffer not found in small_buffer_map, potential memory leak";
            }
        }

        state = cudaSetDevice(ptr.device_id);
        if (state == cudaSuccess) {
            state = cudaFree(ptr.data);
        }
        if (state != cudaSuccess) {
            char info_buf[1<<10];
            snprintf(
                info_buf, 1024,
                "Error: CUDA error when set device %lu",
                ptr.device_id
            );
            LOG(ERROR) << info_buf;
        }

        ptr = MemoryBuffer();
    }

    CudaDeviceAllocator::~CudaDeviceAllocator() {
        for (auto& [device_id, buffers] : small_buffer_map_) {
            cudaSetDevice(device_id);
            for (auto& buf : buffers) {
                if (buf.data) {
                    cudaFree(buf.data);
                }
            }
        }
        small_buffer_map_.clear();
    }

    std::shared_ptr<CudaDeviceAllocator> CudaDeviceAllocatorFactory::instance_ = nullptr;
}
