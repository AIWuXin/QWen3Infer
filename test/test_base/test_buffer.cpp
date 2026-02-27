//
// Created by Administrator on 2026/2/24.
//

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

#include "../../src/include/base/buffer.h"

// 简单的示例测试
TEST(BufferTest, BasicAssertion) {
    auto allocator = qwi::base::CpuDeviceAllocatorFactory::get_instance();
    auto ptr = allocator->allocate(28);
    EXPECT_NE(nullptr, ptr.data);
    allocator->release(ptr);
}

TEST(BufferTest, CpuAllocatorReadWrite) {
    auto allocator = qwi::base::CpuDeviceAllocatorFactory::get_instance();
    auto buf = allocator->allocate(1024);
    EXPECT_NE(nullptr, buf.data);

    // 写入数据
    char* ptr = static_cast<char*>(buf.data);
    for (int i = 0; i < 1024; ++i) {
        ptr[i] = static_cast<char>(i % 256);
    }

    // 验证数据
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(ptr[i], static_cast<char>(i % 256));
    }

    allocator->release(buf);
}

TEST(BufferTest, CpuMemsetZero) {
    auto allocator = qwi::base::CpuDeviceAllocatorFactory::get_instance();
    auto buf = allocator->allocate(1024);
    EXPECT_NE(nullptr, buf.data);

    // 先填充非零值
    std::fill_n(static_cast<char*>(buf.data), 1024, 0xFF);

    // 使用 memset_zero 清零
    allocator->memset_zero(buf.data, 1024);

    // 验证全部为零
    auto* ptr = static_cast<char*>(buf.data);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(ptr[i], 0);
    }

    allocator->release(buf);
}

TEST(BufferTest, CpuMemcpyHost2Host) {
    auto allocator = qwi::base::CpuDeviceAllocatorFactory::get_instance();
    auto src = allocator->allocate(1024);
    auto dst = allocator->allocate(1024);

    // 填充源数据
    for (int i = 0; i < 1024; ++i) {
        static_cast<char*>(src.data)[i] = static_cast<char>(i % 256);
    }

    // 执行 Host2Host 拷贝
    allocator->memcpy(src.data, dst.data, 1024,
        qwi::base::MemcpyKind::kMemcpyHost2Host);

    // 验证数据一致
    EXPECT_EQ(0, std::memcmp(src.data, dst.data, 1024));

    allocator->release(src);
    allocator->release(dst);
}

TEST(BufferTest, CudaAllocatorBasic) {
    qwi::base::CudaDeviceAllocator allocator;
    EXPECT_EQ(allocator.device_type(), qwi::base::DeviceType::kDeviceCUDA);

    // 测试小内存分配 (< 1MB)
    auto buf1 = allocator.allocate(size_t{1} * qwi::base::KB);  // 1KB
    EXPECT_NE(nullptr, buf1.data);
    EXPECT_EQ(buf1.byte_size, size_t{1} * qwi::base::KB);
    EXPECT_TRUE(buf1.busy);

    // 释放小内存（应该缓存，不真正释放）
    allocator.release(buf1);
    EXPECT_EQ(nullptr, buf1.data);

    // 再次分配相同大小（应该命中缓存）
    auto buf2 = allocator.allocate(size_t{1} * qwi::base::KB);
    EXPECT_NE(nullptr, buf2.data);

    allocator.release(buf2);
    EXPECT_EQ(nullptr, buf2.data);
}

TEST(BufferTest, CudaAllocatorReadWrite) {
    qwi::base::CudaDeviceAllocator allocator;
    auto buf = allocator.allocate(1024);
    EXPECT_NE(nullptr, buf.data);

    // 准备主机数据
    std::vector<char> host_data(1024);
    for (int i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<char>(i % 256);
    }

    // 拷贝到设备
    cudaError_t err = cudaMemcpy(buf.data, host_data.data(), 1024, cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    // 拷贝回主机
    std::vector<char> result(1024);
    err = cudaMemcpy(result.data(), buf.data, 1024, cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);

    // 验证数据
    EXPECT_EQ(host_data, result);

    allocator.release(buf);
}

TEST(BufferTest, CudaMemsetZero) {
    qwi::base::CudaDeviceAllocator allocator;
    auto buf = allocator.allocate(1024);
    EXPECT_NE(nullptr, buf.data);

    // 先写入数据
    std::vector<char> host_data(1024, 0xAB);
    cudaMemcpy(buf.data, host_data.data(), 1024, cudaMemcpyHostToDevice);

    // 使用 memset_zero 清零
    allocator.memset_zero(buf.data, 1024);

    // 拷贝回主机验证
    std::vector<char> result(1024);
    cudaMemcpy(result.data(), buf.data, 1024, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(result[i], 0);
    }

    allocator.release(buf);
}

TEST(BufferTest, CudaMemcpyAllDirections) {
    qwi::base::CudaDeviceAllocator allocator;

    // 准备主机数据
    std::vector<char> host_data(1024);
    for (int i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<char>(i % 256);
    }

    auto dev_buf1 = allocator.allocate(1024);
    auto dev_buf2 = allocator.allocate(1024);

    // Test 1: Host2Device
    allocator.memcpy(host_data.data(), dev_buf1.data, 1024,
        qwi::base::MemcpyKind::kMemcpyHost2Device);

    // Test 2: Device2Device
    allocator.memcpy(dev_buf1.data, dev_buf2.data, 1024,
        qwi::base::MemcpyKind::kMemcpyDevice2Device);

    // Test 3: Device2Host
    std::vector<char> result(1024);
    allocator.memcpy(dev_buf2.data, result.data(), 1024,
        qwi::base::MemcpyKind::kMemcpyDevice2Host);

    // 验证数据一致
    EXPECT_EQ(host_data, result);

    allocator.release(dev_buf1);
    allocator.release(dev_buf2);
}

TEST(BufferTest, CudaMemcpyAsync) {
    qwi::base::CudaDeviceAllocator allocator;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<char> host_data(1024, 0xCD);
    auto dev_buf = allocator.allocate(1024);

    // 异步拷贝 H2D
    allocator.memcpy(host_data.data(), dev_buf.data, 1024,
        qwi::base::MemcpyKind::kMemcpyHost2Device,
        stream, true);  // need_sync = true

    // 异步拷贝 D2H
    std::vector<char> result(1024);
    allocator.memcpy(dev_buf.data, result.data(), 1024,
        qwi::base::MemcpyKind::kMemcpyDevice2Host,
        stream, true);

    EXPECT_EQ(host_data, result);

    allocator.release(dev_buf);
    cudaStreamDestroy(stream);
}

TEST(BufferTest, MemcpyZeroSize) {
    qwi::base::CpuDeviceAllocator allocator;
    auto buf = allocator.allocate(1024);

    // 拷贝 0 字节应该安全返回
    allocator.memcpy(buf.data, buf.data, 0);
    SUCCEED();

    allocator.release(buf);
}

TEST(BufferTest, MemsetZeroZeroSize) {
    qwi::base::CpuDeviceAllocator allocator;
    auto buf = allocator.allocate(1024);

    // 清零 0 字节应该安全
    allocator.memset_zero(buf.data, 0);
    SUCCEED();

    allocator.release(buf);
}

TEST(BufferTest, CudaAllocatorBigMemory) {
    qwi::base::CudaDeviceAllocator allocator;

    // 测试大内存分配 (>= 1MB)
    auto buf1 = allocator.allocate(size_t{2} * qwi::base::MB);  // 2MB
    EXPECT_NE(nullptr, buf1.data);
    EXPECT_EQ(buf1.byte_size, size_t{2} * qwi::base::MB);

    // 释放大内存（应该直接 cudaFree，不缓存）
    allocator.release(buf1);
    EXPECT_EQ(nullptr, buf1.data);

    // 再次分配（应该重新 cudaMalloc）
    auto buf2 = allocator.allocate(size_t{2} * qwi::base::MB);
    EXPECT_NE(nullptr, buf2.data);
    allocator.release(buf2);
}

TEST(BufferTest, CudaAllocatorMultipleSmall) {
    qwi::base::CudaDeviceAllocator allocator;

    // 多次分配小内存
    std::vector<qwi::base::MemoryBuffer> buffers;
    for (int i = 0; i < 10; ++i) {
        buffers.push_back(allocator.allocate(size_t{4} * qwi::base::KB));  // 4KB each
    }

    // 验证都分配成功
    for (const auto& buf : buffers) {
        EXPECT_NE(nullptr, buf.data);
    }

    // 逐个释放
    for (auto& buf : buffers) {
        allocator.release(buf);
        EXPECT_EQ(nullptr, buf.data);
    }
}

TEST(BufferTest, CudaAllocatorZeroSize) {
    qwi::base::CudaDeviceAllocator allocator;

    // 分配 0 字节
    auto buf = allocator.allocate(0);
    EXPECT_EQ(nullptr, buf.data);
    EXPECT_EQ(buf.byte_size, 0);
    EXPECT_FALSE(buf.busy);
}

// ==================== 新增测试 ====================

// 阈值动态变化测试
TEST(BufferTest, CudaAllocatorThresholdGrowth) {
    qwi::base::CudaDeviceAllocator allocator;

    // 分配较大内存使net_growth增加
    auto buf1 = allocator.allocate(100 * qwi::base::MB);
    EXPECT_NE(nullptr, buf1.data);
    allocator.release(buf1);

    // 等待1秒让阈值更新
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 触发阈值更新
    auto buf2 = allocator.allocate(1 * qwi::base::KB);
    allocator.release(buf2);

    // 阈值应该在64MB到90%显存之间
    SUCCEED();  // 如果能执行到这里说明没有崩溃
}

// 大小内存交替测试
TEST(BufferTest, CudaAllocatorAlternatingSizes) {
    qwi::base::CudaDeviceAllocator allocator;

    for (int i = 0; i < 10; ++i) {
        // 小内存（缓存）
        auto small = allocator.allocate(100 * qwi::base::KB);
        EXPECT_NE(nullptr, small.data);

        // 大内存（直接）
        auto big = allocator.allocate(2 * qwi::base::MB);
        EXPECT_NE(nullptr, big.data);

        // 先释放大内存
        allocator.release(big);
        EXPECT_EQ(nullptr, big.data);

        // 再释放小内存
        allocator.release(small);
        EXPECT_EQ(nullptr, small.data);
    }
}

// 正好1MB边界测试
TEST(BufferTest, CudaAllocatorExact1MB) {
    qwi::base::CudaDeviceAllocator allocator;

    // 正好1MB应该作为小内存缓存
    auto buf1 = allocator.allocate(1 * qwi::base::MB);
    EXPECT_NE(nullptr, buf1.data);
    allocator.release(buf1);
    EXPECT_EQ(nullptr, buf1.data);

    // 1MB + 1字节应该作为大内存
    auto buf2 = allocator.allocate(1 * qwi::base::MB + 1);
    EXPECT_NE(nullptr, buf2.data);
    allocator.release(buf2);
    EXPECT_EQ(nullptr, buf2.data);
}

// 缓存复用测试
TEST(BufferTest, CudaAllocatorCacheReuse) {
    qwi::base::CudaDeviceAllocator allocator;

    // 分配并释放，创建缓存
    auto buf1 = allocator.allocate(256 * qwi::base::KB);
    void* ptr1 = buf1.data;
    allocator.release(buf1);

    // 立即分配相同大小，应该复用缓存
    auto buf2 = allocator.allocate(256 * qwi::base::KB);
    void* ptr2 = buf2.data;

    // 如果缓存命中，指针应该相同
    EXPECT_EQ(ptr1, ptr2);
    allocator.release(buf2);
}

// 不同大小缓存测试
TEST(BufferTest, CudaAllocatorDifferentSizes) {
    qwi::base::CudaDeviceAllocator allocator;

    // 分配不同大小
    auto buf1 = allocator.allocate(100 * qwi::base::KB);
    auto buf2 = allocator.allocate(200 * qwi::base::KB);
    auto buf3 = allocator.allocate(300 * qwi::base::KB);

    allocator.release(buf1);
    allocator.release(buf2);
    allocator.release(buf3);

    // 申请一个中间大小
    auto buf4 = allocator.allocate(150 * qwi::base::KB);
    EXPECT_NE(nullptr, buf4.data);
    allocator.release(buf4);
}

// 重复释放测试
TEST(BufferTest, CudaAllocatorDoubleRelease) {
    qwi::base::CudaDeviceAllocator allocator;

    auto buf = allocator.allocate(1 * qwi::base::MB);
    EXPECT_NE(nullptr, buf.data);

    allocator.release(buf);
    EXPECT_EQ(nullptr, buf.data);

    // 重复释放应该安全
    allocator.release(buf);
    SUCCEED();
}

// 压力测试
TEST(BufferTest, CudaAllocatorStressTest) {
    qwi::base::CudaDeviceAllocator allocator;

    constexpr int iterations = 100;
    std::vector<qwi::base::MemoryBuffer> buffers;

    // 快速分配
    for (int i = 0; i < iterations; ++i) {
        buffers.push_back(allocator.allocate(10 * qwi::base::KB));
    }

    // 释放一半
    for (int i = 0; i < iterations / 2; ++i) {
        allocator.release(buffers[i]);
    }

    // 再分配一半
    for (int i = 0; i < iterations / 2; ++i) {
        buffers[i] = allocator.allocate(10 * qwi::base::KB);
    }

    // 全部释放
    for (auto& buf : buffers) {
        allocator.release(buf);
    }

    SUCCEED();
}

// 混合压力测试
TEST(BufferTest, CudaAllocatorMixedPressure) {
    qwi::base::CudaDeviceAllocator allocator;

    std::vector<qwi::base::MemoryBuffer> small_buffers;
    std::vector<qwi::base::MemoryBuffer> big_buffers;

    // 同时持有小内存和大内存
    for (int i = 0; i < 5; ++i) {
        small_buffers.push_back(allocator.allocate(500 * qwi::base::KB));
        big_buffers.push_back(allocator.allocate(5 * qwi::base::MB));
    }

    // 交错释放
    for (int i = 0; i < 5; ++i) {
        allocator.release(small_buffers[i]);
        allocator.release(big_buffers[i]);
    }

    SUCCEED();
}

// 1. Buffer 基础构造和析构测试
TEST(BufferTest, BufferBasicConstruction) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto mem_buffer = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);

    {
        qwi::base::Buffer buffer(mem_buffer, allocator, false);
        EXPECT_NE(buffer.get_ptr(), nullptr);
        EXPECT_EQ(buffer.get_byte_size(), 1024);
        EXPECT_EQ(buffer.get_device_type(), qwi::base::DeviceType::kDeviceCPU);
    }  // 析构时自动释放内存
    SUCCEED();
}

// 2. Buffer 外部内存管理测试
TEST(BufferTest, BufferExternalMemory) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto mem = allocator->allocate(1024);
    EXPECT_NE(mem.data, nullptr);

    {
        // 使用外部内存，Buffer 不负责释放
        qwi::base::Buffer buffer(mem, allocator, true);
        EXPECT_EQ(buffer.get_ptr(), mem.data);
    }

    // Buffer 析构后外部内存仍然存在
    EXPECT_NE(mem.data, nullptr);
    allocator->release(mem);
}

// 3. Buffer CUDA 迁移测试 (cuda() 方法目前是空实现，可以后续补充)
TEST(BufferTest, BufferCudaMigration) {
    auto cpu_allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto mem_buffer = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);

    qwi::base::Buffer buffer(mem_buffer, cpu_allocator, false);
    EXPECT_EQ(buffer.get_device_type(), qwi::base::DeviceType::kDeviceCPU);

    // 迁移到 CUDA 设备 0 (目前是空实现)
    buffer.cuda(0);

    // 如果 cuda() 实现完成，这里应该验证设备类型变化
    // EXPECT_EQ(buffer.get_device_type(), qwi::base::DeviceType::kDeviceCUDA);
    SUCCEED();
}

// 4. Buffer 零大小测试
TEST(BufferTest, BufferZeroSize) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto mem_buffer = qwi::base::MemoryBuffer(nullptr, 0, false, 0, qwi::base::DeviceType::kDeviceCPU);

    qwi::base::Buffer buffer(mem_buffer, allocator, false);
    EXPECT_EQ(buffer.get_byte_size(), 0);
}

// 5. Buffer 数据读写测试
TEST(BufferTest, BufferReadWrite) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto mem_buffer = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);

    qwi::base::Buffer buffer(mem_buffer, allocator, false);
    char* ptr = static_cast<char*>(buffer.get_ptr());

    // 写入数据
    for (int i = 0; i < 1024; ++i) {
        ptr[i] = static_cast<char>(i % 256);
    }

    // 验证数据
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(ptr[i], static_cast<char>(i % 256));
    }
}

// ==================== Buffer copy_from 测试 ====================

// 1. CPU -> CPU 拷贝测试
TEST(BufferCopyFromTest, CpuToCpu) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    // 创建源 buffer 并填充数据
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 1024; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    // 创建目标 buffer
    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 验证数据
    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
}

// 2. 源 buffer 大于目标 buffer 测试
TEST(BufferCopyFromTest, SrcLargerThanDst) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    // 源 1024 字节，目标 512 字节
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 1024; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 验证只拷贝了 512 字节
    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    for (int i = 0; i < 512; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
}

// 3. 源 buffer 小于目标 buffer 测试
TEST(BufferCopyFromTest, SrcSmallerThanDst) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    // 源 512 字节，目标 1024 字节
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 512; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 先填充目标 buffer 为特定值
    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    std::fill_n(dst_ptr, 1024, 0xAB);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 验证前 512 字节被拷贝，后 512 字节保持不变
    for (int i = 0; i < 512; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
    for (int i = 512; i < 1024; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(0xAB));
    }
}

// 4. 目标 buffer 拷贝测试 (Buffer 构造时会自动分配，测试正常拷贝即可)
TEST(BufferCopyFromTest, DstNormalCopy) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    // 创建源 buffer
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 512; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    // 创建目标 buffer（构造时会自动分配）
    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 验证已分配
    EXPECT_NE(dst.get_ptr(), nullptr);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 验证数据正确
    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    for (int i = 0; i < 512; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
}

// 5. 空源 buffer 测试
TEST(BufferCopyFromTest, EmptySrc) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    // 创建空源 buffer (大小为 0)
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 0, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::ZeroByteSize);
}

// 6. nullptr 指针测试
TEST(BufferCopyFromTest, NullPtrSrc) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 传入 nullptr
    auto status = dst.copy_from(nullptr);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Error);
}

// 7. CPU -> CUDA 拷贝测试
TEST(BufferCopyFromTest, CpuToCuda) {
    auto cpu_allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto cuda_allocator = std::make_shared<qwi::base::CudaDeviceAllocator>();

    // 创建 CPU 源 buffer
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, cpu_allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 1024; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    // 创建 CUDA 目标 buffer
    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCUDA);
    qwi::base::Buffer dst(dst_mem, cuda_allocator, false);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 拷贝回 CPU 验证
    std::vector<char> result(1024);
    cudaMemcpy(result.data(), dst.get_ptr(), 1024, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(result[i], static_cast<char>(i % 256));
    }
}

// 8. CUDA -> CPU 拷贝测试
TEST(BufferCopyFromTest, CudaToCpu) {
    auto cpu_allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();
    auto cuda_allocator = std::make_shared<qwi::base::CudaDeviceAllocator>();

    // 准备主机数据并拷到 GPU
    std::vector<char> host_data(1024);
    for (int i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<char>(i % 256);
    }

    // 创建 CUDA 源 buffer
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCUDA);
    qwi::base::Buffer src(src_mem, cuda_allocator, false);
    cudaMemcpy(src.get_ptr(), host_data.data(), 1024, cudaMemcpyHostToDevice);

    // 创建 CPU 目标 buffer
    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, cpu_allocator, false);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 验证数据
    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
}

// 9. CUDA -> CUDA 同设备拷贝测试
TEST(BufferCopyFromTest, CudaToCudaSameDevice) {
    auto cuda_allocator = std::make_shared<qwi::base::CudaDeviceAllocator>();

    // 准备主机数据
    std::vector<char> host_data(1024);
    for (int i = 0; i < 1024; ++i) {
        host_data[i] = static_cast<char>(i % 256);
    }

    // 创建两个 CUDA buffer（同设备）
    auto src_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCUDA);
    qwi::base::Buffer src(src_mem, cuda_allocator, false);
    cudaMemcpy(src.get_ptr(), host_data.data(), 1024, cudaMemcpyHostToDevice);

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 1024, false, 0, qwi::base::DeviceType::kDeviceCUDA);
    qwi::base::Buffer dst(dst_mem, cuda_allocator, false);

    // 执行拷贝
    auto status = dst.copy_from(src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    // 拷贝回 CPU 验证
    std::vector<char> result(1024);
    cudaMemcpy(result.data(), dst.get_ptr(), 1024, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_EQ(result[i], static_cast<char>(i % 256));
    }
}

// 10. 使用指针版本测试
TEST(BufferCopyFromTest, PointerVersion) {
    auto allocator = std::make_shared<qwi::base::CpuDeviceAllocator>();

    auto src_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer src(src_mem, allocator, false);
    char* src_ptr = static_cast<char*>(src.get_ptr());
    for (int i = 0; i < 512; ++i) {
        src_ptr[i] = static_cast<char>(i % 256);
    }

    auto dst_mem = qwi::base::MemoryBuffer(nullptr, 512, false, 0, qwi::base::DeviceType::kDeviceCPU);
    qwi::base::Buffer dst(dst_mem, allocator, false);

    // 使用指针版本
    auto status = dst.copy_from(&src);
    EXPECT_EQ(status, qwi::base::ReturnStatus::Success);

    char* dst_ptr = static_cast<char*>(dst.get_ptr());
    for (int i = 0; i < 512; ++i) {
        EXPECT_EQ(dst_ptr[i], static_cast<char>(i % 256));
    }
}

// 主函数
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
