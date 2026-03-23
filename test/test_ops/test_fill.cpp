//
// Created by Administrator on 2026/3/17.
//

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>

#include "../../src/include/ops/fill.h"
#include "../../src/include/tensor/function.h"
#include "../../src/include/base/alloc.h"

using namespace qwi;

class FillTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CudaDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;

    // 辅助函数：创建测试 tensor
    tensor::Tensor create_test_tensor(
        std::vector<size_t> dims,
        base::DeviceType device_type = base::DeviceType::kDeviceCPU
    ) {
        auto allocator = (device_type == base::DeviceType::kDeviceCPU)
            ? cpu_allocator_ : cuda_allocator_;
        return tensor::Tensor(
            base::DataType::kDataFloat32,
            dims,
            device_type,
            allocator,
            nullptr
        );
    }
};

// ========== 基本功能测试 ==========

TEST_F(FillTest, FillGlobalCPU) {
    auto tensor = create_test_tensor({10, 10}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        3.14f,
        INT_MIN,  // 全局填充
        100,      // 填充全部 100 个元素
        base::DeviceType::kDeviceCPU,
        "fill_test"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 验证所有元素都被填充
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            EXPECT_FLOAT_EQ(tensor.index<float>({i, j}), 3.14f);
        }
    }
}

TEST_F(FillTest, FillPartialCPU) {
    auto tensor = create_test_tensor({5, 5}, base::DeviceType::kDeviceCPU);
    // 初始化为 0
    auto status = tensor.fill(0.0f, INT_MIN, 25);
    EXPECT_TRUE(status);

    // 只填充前 10 个元素
    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        5.0f,
        INT_MIN,
        10,  // 只填充 10 个
        base::DeviceType::kDeviceCPU,
        "fill_partial"
    );
    fill_layer.set_input(0, tensor);
    status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 验证前 10 个是 5.0，后面是 0.0
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            float expected = (i * 5 + j < 10) ? 5.0f : 0.0f;
            EXPECT_FLOAT_EQ(tensor.index<float>({i, j}), expected);
        }
    }
}

TEST_F(FillTest, FillDim0CPU) {
    // 3x4 tensor，在 dim=0 上填充前 2 行
    auto tensor = create_test_tensor({3, 4}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        7.0f,
        0,        // dim=0
        2,        // 填充前 2 行
        base::DeviceType::kDeviceCPU,
        "fill_dim0"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 验证前 2 行是 7.0，第 3 行未填充（应该是随机值或未初始化）
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(tensor.index<float>({0, j}), 7.0f);
        EXPECT_FLOAT_EQ(tensor.index<float>({1, j}), 7.0f);
    }
}

TEST_F(FillTest, FillDim1CPU) {
    // 3x4 tensor，在 dim=1 上填充前 2 列
    auto tensor = create_test_tensor({3, 4}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        9.0f,
        1,        // dim=1
        2,        // 填充前 2 列
        base::DeviceType::kDeviceCPU,
        "fill_dim1"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 验证前 2 列是 9.0
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(tensor.index<float>({i, 0}), 9.0f);
        EXPECT_FLOAT_EQ(tensor.index<float>({i, 1}), 9.0f);
    }
}

// ========== 高维 Tensor 测试 ==========

TEST_F(FillTest, Fill3DCPU) {
    // 2x3x4 tensor，在 dim=1 上填充
    auto tensor = create_test_tensor({2, 3, 4}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        2.0f,
        1,        // dim=1
        2,        // 填充前 2 个
        base::DeviceType::kDeviceCPU,
        "fill_3d"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 简单验证：检查一些位置的值
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_FLOAT_EQ(tensor.index<float>({i, 0, k}), 2.0f);
            EXPECT_FLOAT_EQ(tensor.index<float>({i, 1, k}), 2.0f);
        }
    }
}

TEST_F(FillTest, Fill4DCPU) {
    auto tensor = create_test_tensor({2, 3, 4, 5}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        1.0f,
        INT_MIN,  // 全局填充
        120,      // 2*3*4*5 = 120
        base::DeviceType::kDeviceCPU,
        "fill_4d"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);

    // 抽样验证
    EXPECT_FLOAT_EQ(tensor.index<float>({0, 0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor.index<float>({1, 2, 3, 4}), 1.0f);
}

// ========== 边界条件测试 ==========

TEST_F(FillTest, FillZeroCount) {
    auto tensor = create_test_tensor({5, 5}, base::DeviceType::kDeviceCPU);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        5.0f,
        INT_MIN,
        0,  // 填充 0 个
        base::DeviceType::kDeviceCPU,
        "fill_zero"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);  // 应该成功，但什么都不做
}

// TEST_F(FillTest, FillCountExceedsSize) {
//     auto tensor = create_test_tensor({3, 3}, base::DeviceType::kDeviceCPU);
//
//     // count 超过实际大小，应该被截断
//     ops::Fill fill_layer(
//         base::DataType::kDataFloat32,
//         8.0f,
//         INT_MIN,
//         100,  // 超过 9
//         base::DeviceType::kDeviceCPU,
//         "fill_exceed"
//     );
//     fill_layer.set_input(0, tensor);
//
//     auto status = fill_layer.forward();
//     EXPECT_TRUE(status);
//
//     // 应该只填充 9 个元素
//     for (size_t i = 0; i < 3; ++i) {
//         for (size_t j = 0; j < 3; ++j) {
//             EXPECT_FLOAT_EQ(tensor.index<float>({i, j}), 8.0f);
//         }
//     }
// }

// TEST_F(FillTest, NegativeDim) {
//     // 测试负维度索引（如果支持）
//     auto tensor = create_test_tensor({3, 4, 5}, base::DeviceType::kDeviceCPU);
//
//     // dim=-1 应该等价于 dim=2（最后一维）
//     ops::Fill fill_layer(
//         base::DataType::kDataFloat32,
//         3.0f,
//         -1,       // 最后一维
//         3,
//         base::DeviceType::kDeviceCPU,
//         "fill_neg_dim"
//     );
//     fill_layer.set_input(0, tensor);
//
//     auto status = fill_layer.forward();
//     EXPECT_TRUE(status);
// }

// ========== 不同数据类型测试 ==========

// TEST_F(FillTest, FillInt32CPU) {
//     tensor::Tensor tensor(
//         base::DataType::kDataInt32,
//         {5, 5},
//         base::DeviceType::kDeviceCPU,
//         cpu_allocator_,
//         nullptr
//     );
//
//     ops::Fill fill_layer(
//         base::DataType::kDataInt32,
//         42.0,     // 会被转换为 int
//         INT_MIN,
//         25,
//         base::DeviceType::kDeviceCPU,
//         "fill_int32"
//     );
//     fill_layer.set_input(0, tensor);
//
//     auto status = fill_layer.forward();
//     EXPECT_TRUE(status);
//
//     for (size_t i = 0; i < 5; ++i) {
//         for (size_t j = 0; j < 5; ++j) {
//             EXPECT_EQ(tensor.index<int32_t>({i, j}), 42);
//         }
//     }
// }

// ========== Tensor::fill 成员函数测试 ==========

TEST_F(FillTest, TensorFillMemberFunction) {
    auto tensor = tensor::zeros({10, 10}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    auto status = tensor.fill(5.5f, INT_MIN, 50);  // 填充前 50 个
    EXPECT_TRUE(status);

    // 验证前 50 个是 5.5，后 50 个是 0
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            float expected = (i * 10 + j < 50) ? 5.5f : 0.0f;
            EXPECT_FLOAT_EQ(tensor.index<float>({i, j}), expected);
        }
    }
}

// ========== CUDA 测试 ==========

TEST_F(FillTest, FillGlobalCUDA) {
    std::cout << "[DEBUG] Creating tensor..." << std::endl;
    auto tensor = create_test_tensor({100, 100}, base::DeviceType::kDeviceCUDA);
    std::cout << "[DEBUG] Tensor created, size=" << tensor.size() << std::endl;

    std::cout << "[DEBUG] Calling fill..." << std::endl;
    const auto status = tensor.fill(3.14, INT_MIN, 10000);
    std::cout << "[DEBUG] Fill returned" << std::endl;

    EXPECT_TRUE(status);

    std::cout << "[DEBUG] Calling cpu()..." << std::endl;
    tensor.cpu();
    std::cout << "[DEBUG] cpu() returned" << std::endl;

    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            EXPECT_FLOAT_EQ(tensor.index<float>({i, j}), 3.14f);
        }
    }
}

TEST_F(FillTest, FillDimCUDASmall) {
    auto tensor = create_test_tensor({10, 20}, base::DeviceType::kDeviceCUDA);

    ops::Fill fill_layer(
        base::DataType::kDataFloat32,
        2.71f,
        1,        // dim=1
        10,
        base::DeviceType::kDeviceCUDA,
        "fill_cuda_dim"
    );
    fill_layer.set_input(0, tensor);

    auto status = fill_layer.forward();
    EXPECT_TRUE(status);
}

// ========== 性能测试 ==========

class FillPerfTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CudaDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;

    // 计时辅助函数
    template<typename Func>
    double measure_time(Func&& func, int warmup_runs = 3, int timing_runs = 10) {
        // Warmup
        for (int i = 0; i < warmup_runs; ++i) {
            func();
        }

        // Timing
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timing_runs; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count() / timing_runs;  // 平均每次的毫秒数
    }
};

// CPU 性能测试
TEST_F(FillPerfTest, FillGlobalCPUPerf) {
    const std::vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    std::cout << "\n=== CPU Fill Global Performance ===" << std::endl;
    std::cout << "Size\t\tTime(ms)\tBandwidth(GB/s)" << std::endl;

    for (size_t n : sizes) {
        auto tensor = tensor::empty({n}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

        auto fill_func = [&]() {
            tensor.fill(1.0f, INT_MIN, n);
        };

        double avg_time = measure_time(fill_func);
        double bytes = n * sizeof(float);
        double bandwidth = (bytes / (avg_time / 1000.0)) / (1024 * 1024 * 1024);  // GB/s

        std::cout << n << "\t\t" << std::fixed << std::setprecision(3)
                  << avg_time << "\t\t" << bandwidth << std::endl;
    }
}

TEST_F(FillPerfTest, FillDimCPUPerf) {
    // 测试不同维度填充的性能
    std::cout << "\n=== CPU Fill Dim Performance ===" << std::endl;
    std::cout << "Shape\t\tDim\tTime(ms)" << std::endl;

    struct TestCase {
        std::vector<size_t> dims;
        int32_t dim;
        std::string name;
    };

    std::vector<TestCase> cases = {
        {{1000, 1000}, 0, "1000x1000 dim0"},
        {{1000, 1000}, 1, "1000x1000 dim1"},
        {{100, 100, 100}, 0, "100x100x100 dim0"},
        {{100, 100, 100}, 1, "100x100x100 dim1"},
        {{100, 100, 100}, 2, "100x100x100 dim2"},
    };

    for (const auto& tc : cases) {
        auto tensor = tensor::empty(tc.dims, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

        auto fill_func = [&]() {
            tensor.fill(1.0f, tc.dim, tc.dims[tc.dim]);
        };

        double avg_time = measure_time(fill_func);
        std::cout << tc.name << "\t" << avg_time << std::endl;
    }
}

TEST_F(FillPerfTest, FillGlobalCUDAPerf) {
    const std::vector<size_t> sizes = {
        1000, 10000, 100000,
        1000000, 10000000, 100000000  // 大尺寸测试 GPU 优势
    };

    std::cout << "\n=== CUDA Fill Global Performance ===" << std::endl;
    std::cout << "Size\t\tTime(ms)\tBandwidth(GB/s)" << std::endl;

    for (size_t n : sizes) {
        auto tensor = tensor::empty({n}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);

        // 创建 CUDA stream（如果需要）
        // auto cuda_config = std::make_shared<base::CudaConfig>();
        // ... 设置 stream ...

        auto fill_func = [&]() {
            tensor.fill(1.0f, INT_MIN, n);
            // 如果需要同步：cudaDeviceSynchronize();
        };

        double avg_time = measure_time(fill_func, 5, 20);  // GPU 需要更多 warmup
        double bytes = n * sizeof(float);
        double bandwidth = (bytes / (avg_time / 1000.0)) / (1024 * 1024 * 1024);

        std::cout << n << "\t\t" << std::fixed << std::setprecision(3)
                  << avg_time << "\t\t" << bandwidth << std::endl;
    }
}

TEST_F(FillPerfTest, FillDimCUDAPerf) {
    std::cout << "\n=== CUDA Fill Dim Performance ===" << std::endl;

    struct TestCase {
        std::vector<size_t> dims;
        int32_t dim;
        size_t count;
        std::string name;
    };

    std::vector<TestCase> cases = {
        {{1024, 1024}, 0, 1024, "1024x1024 dim0"},
        {{1024, 1024}, 1, 1024, "1024x1024 dim1"},
        {{256, 256, 256}, 0, 256, "256x256x256 dim0"},
        {{256, 256, 256}, 1, 256, "256x256x256 dim1"},
        {{256, 256, 256}, 2, 256, "256x256x256 dim2"},
    };

    for (const auto& tc : cases) {
        auto tensor = tensor::empty(tc.dims, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);

        auto fill_func = [&]() {
            tensor.fill(1.0f, tc.dim, tc.count);
        };

        double avg_time = measure_time(fill_func, 5, 20);
        std::cout << tc.name << "\t" << avg_time << " ms" << std::endl;
    }
}

TEST_F(FillPerfTest, FillCPUsCUDASpeedup) {
    // 对比 CPU 和 CUDA 性能
    const size_t n = 1 * 1024 * 1024;  // 10M 元素

    std::cout << "\n=== CPU vs CUDA Speedup ===" << std::endl;
    std::cout << "Size: " << n << " elements (" << (n * sizeof(float) / (1024*1024)) << " MB)" << std::endl;

    // CPU
    auto cpu_tensor = tensor::empty({n}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto cpu_fill = [&]() { cpu_tensor.fill(1.0f, INT_MIN, n); };
    double cpu_time = measure_time(cpu_fill, 3, 10);

    // CUDA
    auto cuda_tensor = tensor::empty({n}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);
    auto cuda_fill = [&]() { cuda_tensor.fill(1.0f, INT_MIN, n); };
    double cuda_time = measure_time(cuda_fill, 5, 20);

    std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "CUDA time: " << cuda_time << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time / cuda_time) << "x" << std::endl;
}

// 主函数由 gtest_main 提供
