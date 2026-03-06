//
// Created by Administrator on 2026/3/5.
//


#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <gtest/gtest.h>

#include "../../src/include/ops/elementwise.h"
#include "../../src/include/tensor/tensorbase.h"
#include "../../src/include/base/alloc.h"

using namespace qwi;


class ElementwiseTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;

    // 计时辅助函数
    template<typename Func>
    double benchmark(Func&& func, int warmup_iters = 5, int test_iters = 20) {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            func();
        }

        // 正式测试
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < test_iters; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> ms = (end - start) / test_iters;
        return ms.count();  // 返回单次执行时间(ms)
    }
};


// ==================== 基本功能测试 ====================

TEST_F(ElementwiseTest, Add1D) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_1d",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(
        base::DataType::kDataFloat32, 100,
        base::DeviceType::kDeviceCPU, cpu_allocator_
    );
    tensor::Tensor input1(
        base::DataType::kDataFloat32, 100,
        base::DeviceType::kDeviceCPU, cpu_allocator_
    );
    tensor::Tensor output(
        base::DataType::kDataFloat32, 100,
        base::DeviceType::kDeviceCPU, cpu_allocator_
    );

    float* ptr0 = input0.ptr<float>();
    float* ptr1 = input1.ptr<float>();
    for (size_t i = 0; i < 100; ++i) {
        ptr0[i] = static_cast<float>(i);
        ptr1[i] = static_cast<float>(i * 2);
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    const auto status = elementwise.forward();
    EXPECT_TRUE(status);

    const float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(i + i * 2));
    }
}

TEST_F(ElementwiseTest, Add2D) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_2d",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr0 = input0.ptr<float>();
    float* ptr1 = input1.ptr<float>();
    for (size_t i = 0; i < 12; ++i) {
        ptr0[i] = 1.0f;
        ptr1[i] = 2.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], 3.0f);
    }
}

TEST_F(ElementwiseTest, Add3D) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_3d",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 2, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 2, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 2, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr0 = input0.ptr<float>();
    float* ptr1 = input1.ptr<float>();
    for (size_t i = 0; i < 24; ++i) {
        ptr0[i] = static_cast<float>(i);
        ptr1[i] = 1.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < 24; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(i) + 1.0f);
    }
}

// ==================== 大向量测试（触发 OpenMP） ====================

TEST_F(ElementwiseTest, AddLargeVector) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_large",
        base::DeviceType::kDeviceCPU
    );

    const size_t size = 10000;
    tensor::Tensor input0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr0 = input0.ptr<float>();
    float* ptr1 = input1.ptr<float>();
    for (size_t i = 0; i < size; ++i) {
        ptr0[i] = static_cast<float>(i);
        ptr1[i] = static_cast<float>(size - i);
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], static_cast<float>(size));
    }
}

// ==================== 边界值测试 ====================

TEST_F(ElementwiseTest, AddNegativeNumbers) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_negative",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr0 = input0.ptr<float>();
    float* ptr1 = input1.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        ptr0[i] = -5.0f;
        ptr1[i] = 3.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], -2.0f);
    }
}

// ==================== 错误处理测试 ====================

TEST_F(ElementwiseTest, CheckEmptyInput) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_check_empty",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0;
    tensor::Tensor input1(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.check();
    EXPECT_FALSE(status);
}

TEST_F(ElementwiseTest, CheckWrongDataType) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_check_type",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat16, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.check();
    EXPECT_FALSE(status);
}

// ==================== 单元素和零值测试 ====================

TEST_F(ElementwiseTest, AddSingleElement) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_single",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    input0.ptr<float>()[0] = 42.0f;
    input1.ptr<float>()[0] = 58.0f;

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 100.0f);
}

TEST_F(ElementwiseTest, AddZeros) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "test_add_zeros",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 100, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 100, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 100, base::DeviceType::kDeviceCPU, cpu_allocator_);

    cpu_allocator_->memset_zero(input0.ptr<float>(), 100 * sizeof(float));
    cpu_allocator_->memset_zero(input1.ptr<float>(), 100 * sizeof(float));

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    const float* out_ptr = output.ptr<float>();
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(out_ptr[i], 0.0f);
    }
}

// 50x50 = 2500 元素（小量，串行）
TEST_F(ElementwiseTest, Add_50x50) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_50x50",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 50, 50, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 50, 50, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 50, 50, base::DeviceType::kDeviceCPU, cpu_allocator_);

    // 初始化数据
    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
    for (size_t i = 0; i < 2500; ++i) {
        p0[i] = static_cast<float>(i);
        p1[i] = 1.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 100);

    std::cout << "[Perf] 50x50 (2,500 elements): " << avg_ms << " ms/op" << std::endl;
    std::cout << "[Perf] Throughput: " << (2500 * sizeof(float) * 3 / avg_ms / 1e6) << " GB/s" << std::endl;
}

// 200x200 = 40,000 元素（中等，可能触发 OpenMP）
TEST_F(ElementwiseTest, Add_200x200) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_200x200",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 200, 200, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 200, 200, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 200, 200, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
    for (size_t i = 0; i < 40000; ++i) {
        p0[i] = static_cast<float>(i);
        p1[i] = 1.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 100);

    std::cout << "[Perf] 200x200 (40,000 elements): " << avg_ms << " ms/op" << std::endl;
    std::cout << "[Perf] Throughput: " << (40000 * sizeof(float) * 3 / avg_ms / 1e6) << " GB/s" << std::endl;
}

// 512x512 = 262,144 元素（大矩阵，OpenMP 并行）
TEST_F(ElementwiseTest, Add_512x512) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_512x512",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
    for (size_t i = 0; i < 512*512; ++i) {
        p0[i] = static_cast<float>(i);
        p1[i] = 1.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 100);

    std::cout << "[Perf] 512x512 (262,144 elements): " << avg_ms << " ms/op" << std::endl;
    std::cout << "[Perf] Throughput: " << (262144 * sizeof(float) * 3 / avg_ms / 1e6) << " GB/s" << std::endl;
}

// 1024x1024 = 1,048,576 元素（1M，典型 LLM 激活值大小）
TEST_F(ElementwiseTest, Add_1024x1024) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_1024x1024",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 1024, 1024, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 1024, 1024, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1024, 1024, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
    for (size_t i = 0; i < 1024*1024; ++i) {
        p0[i] = static_cast<float>(i % 1000);
        p1[i] = 1.0f;
    }

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    const double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 100);

    std::cout << "[Perf] 1024x1024 (1,048,576 elements): " << avg_ms << " ms/op" << std::endl;
    std::cout << "[Perf] Throughput: " << (1048576.0 * sizeof(float) * 3 / avg_ms / 1e6) << " GB/s" << std::endl;
}

// 4096x4096 = 16,777,216 元素（超大矩阵，测试内存带宽）
TEST_F(ElementwiseTest, Add_4096x4096) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_4096x4096",
        base::DeviceType::kDeviceCPU
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 4096, 4096, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 4096, 4096, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 4096, 4096, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < 4096*4096; ++i) {
        p0[i] = static_cast<float>(i % 100);
        p1[i] = 2.0f;
    }
#else
    for (size_t i = 0; i < 4096*4096; ++i) {
        p0[i] = static_cast<float>(i % 100);
        p1[i] = 2.0f;
    }
#endif
    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 100);

    std::cout << "[Perf] 4096x4096 (16,777,216 elements): " << avg_ms << " ms/op" << std::endl;
    std::cout << "[Perf] Throughput: " << (16777216.0 * sizeof(float) * 3 / avg_ms / 1e6) << " GB/s" << std::endl;

    // 验证结果
    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 2.0f);
    EXPECT_FLOAT_EQ(out[4096*4096-1], 17.0f);  // 99 + 2
}

// 对比测试：串行 vs OpenMP（通过阈值控制）
TEST_F(ElementwiseTest, ThresholdComparison) {
    constexpr size_t size = 256*256;  // 刚好超过 256*256 阈值

    tensor::Tensor input0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* p0 = input0.ptr<float>();
    float* p1 = input1.ptr<float>();
    for (size_t i = 0; i < size; ++i) {
        p0[i] = static_cast<float>(i);
        p1[i] = 1.0f;
    }

    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "perf_threshold",
        base::DeviceType::kDeviceCPU
    );
    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    double avg_ms = benchmark([&]() {
        elementwise.forward();
    }, 10, 50);

    std::cout << "[Perf] 256*256 elements (OpenMP threshold test): " << avg_ms << " ms/op" << std::endl;

    #ifdef USE_OPENMP
    std::cout << "[Perf] OpenMP enabled, max threads: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "[Perf] OpenMP disabled (sequential)" << std::endl;
    #endif
}


// ==================== CUDA 测试（需要 GPU） ====================
#include <cuda_runtime.h>

class ElementwiseCudaTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 检查 CUDA 可用性
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available, skipping CUDA tests";
        }
        cuda_allocator_ = base::CudaDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;
};

// CUDA 基础加法测试
TEST_F(ElementwiseCudaTest, Add1D) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "cuda_add_1d",
        base::DeviceType::kDeviceCUDA,
        base::ElementWiseType::kElementAdd
    );

    constexpr size_t size = 10000;
    tensor::Tensor input0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);

    // 在 CPU 上准备数据，然后拷贝到 GPU
    std::vector<float> host_data0(size), host_data1(size);
    for (size_t i = 0; i < size; ++i) {
        host_data0[i] = static_cast<float>(i);
        host_data1[i] = 1.0f;
    }

    // 拷贝到 GPU
    cudaMemcpy(input0.ptr<float>(), host_data0.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input1.ptr<float>(), host_data1.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    // 拷贝结果回 CPU 验证
    std::vector<float> result(size);
    cudaMemcpy(result.data(), output.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], static_cast<float>(i) + 1.0f);
    }
}

// CUDA 减法测试
TEST_F(ElementwiseCudaTest, Subtract2D) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "cuda_sub_2d",
        base::DeviceType::kDeviceCUDA,
        base::ElementWiseType::kElementSubtract
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 512, 512, base::DeviceType::kDeviceCUDA, cuda_allocator_);

    const size_t size = 512 * 512;
    std::vector<float> host0(size, 10.0f), host1(size, 3.0f);

    cudaMemcpy(input0.ptr<float>(), host0.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input1.ptr<float>(), host1.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    std::vector<float> result(size);
    cudaMemcpy(result.data(), output.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(result[i], 7.0f);  // 10 - 3 = 7
    }
}

// CUDA 乘法测试
TEST_F(ElementwiseCudaTest, MultiplyLarge) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "cuda_mul_large",
        base::DeviceType::kDeviceCUDA,
        base::ElementWiseType::kElementMultiply
    );

    constexpr size_t size = 1024 * 1024;  // 1M 元素
    tensor::Tensor input0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);

    std::vector<float> host0(size, 2.0f), host1(size, 3.0f);

    cudaMemcpy(input0.ptr<float>(), host0.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input1.ptr<float>(), host1.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    std::vector<float> result(size);
    cudaMemcpy(result.data(), output.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

    // 只采样验证几个点（加快测试）
    EXPECT_FLOAT_EQ(result[0], 6.0f);
    EXPECT_FLOAT_EQ(result[size/2], 6.0f);
    EXPECT_FLOAT_EQ(result[size-1], 6.0f);
}

// CUDA 除法测试
TEST_F(ElementwiseCudaTest, DivideBasic) {
    ops::Elementwise elementwise(
        base::DataType::kDataFloat32,
        "cuda_div_basic",
        base::DeviceType::kDeviceCUDA,
        base::ElementWiseType::kElementDivide
    );

    tensor::Tensor input0(base::DataType::kDataFloat32, 1000, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor input1(base::DataType::kDataFloat32, 1000, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1000, base::DeviceType::kDeviceCUDA, cuda_allocator_);

    std::vector<float> host0(1000, 100.0f), host1(1000, 4.0f);

    cudaMemcpy(input0.ptr<float>(), host0.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input1.ptr<float>(), host1.data(), 1000 * sizeof(float), cudaMemcpyHostToDevice);

    elementwise.set_input(0, input0);
    elementwise.set_input(1, input1);
    elementwise.set_output(0, output);

    auto status = elementwise.forward();
    EXPECT_TRUE(status);

    std::vector<float> result(1000);
    cudaMemcpy(result.data(), output.ptr<float>(), 1000 * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_FLOAT_EQ(result[i], 25.0f);  // 100 / 4 = 25
    }
}

// CUDA 性能测试
TEST_F(ElementwiseCudaTest, PerfCudaVsCpu) {
    ops::Elementwise cuda_elementwise(
        base::DataType::kDataFloat32,
        "cuda_perf",
        base::DeviceType::kDeviceCUDA,
        base::ElementWiseType::kElementAdd
    );

    ops::Elementwise cpu_elementwise(
        base::DataType::kDataFloat32,
        "cpu_perf",
        base::DeviceType::kDeviceCPU,
        base::ElementWiseType::kElementAdd
    );

    constexpr size_t size = 10 * 1024 * 1024;  // 10M 元素

    // CUDA 版本
    tensor::Tensor cuda_in0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor cuda_in1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);
    tensor::Tensor cuda_out(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCUDA, cuda_allocator_);

    cudaMemset(cuda_in0.ptr<float>(), 0, size * sizeof(float));
    cudaMemset(cuda_in1.ptr<float>(), 0, size * sizeof(float));

    cuda_elementwise.set_input(0, cuda_in0);
    cuda_elementwise.set_input(1, cuda_in1);
    cuda_elementwise.set_output(0, cuda_out);

    // 预热
    for (int i = 0; i < 5; ++i) cuda_elementwise.forward();
    cudaDeviceSynchronize();

    // 计时 CUDA
    auto start = std::chrono::high_resolution_clock::now();
    cuda_elementwise.forward();
    cudaDeviceSynchronize();  // 确保完成
    auto end = std::chrono::high_resolution_clock::now();
    auto cuda_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // CPU 版本
    auto cpu_allocator = base::CpuDeviceAllocatorFactory::get_instance();
    tensor::Tensor cpu_in0(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator);
    tensor::Tensor cpu_in1(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator);
    tensor::Tensor cpu_out(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator);

    cpu_elementwise.set_input(0, cpu_in0);
    cpu_elementwise.set_input(1, cpu_in1);
    cpu_elementwise.set_output(0, cpu_out);

    start = std::chrono::high_resolution_clock::now();
    cpu_elementwise.forward();
    end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "[Perf] 10M elements - CPU: " << cpu_ms << " ms, CUDA: " << cuda_ms << " ms" << std::endl;
    std::cout << "[Perf] Speedup: " << (cpu_ms / cuda_ms) << "x" << std::endl;

    // 验证 CUDA 结果正确
    std::vector<float> result(100);
    cudaMemcpy(result.data(), cuda_out.ptr<float>(), 100 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(result[i], 0.0f);  // 0 + 0 = 0
    }
}
