//
// Created by Administrator on 2026/3/25.
//


#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <gtest/gtest.h>
#include <cmath>
#include <cstdlib>

#include "../../src/include/ops/rsmnorm.h"
#include "../../src/include/tensor/tensorbase.h"
#include "../../src/include/tensor/function.h"
#include "../../src/include/base/alloc.h"

#include <cuda_runtime.h>

using namespace qwi;


class RmsNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;

    // 计算参考 RMSNorm 结果
    void compute_ref_rmsnorm(
        const float* input,
        const float* gamma,
        float* output,
        size_t num_rows,
        size_t hidden_dim,
        float eps
    ) {
        for (size_t row = 0; row < num_rows; ++row) {
            const float* row_in = input + row * hidden_dim;
            float* row_out = output + row * hidden_dim;

            float sum_squares = 0.0f;
            for (size_t i = 0; i < hidden_dim; ++i) {
                sum_squares += row_in[i] * row_in[i];
            }

            float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_dim) + eps);

            for (size_t i = 0; i < hidden_dim; ++i) {
                row_out[i] = row_in[i] * inv_rms * gamma[i];
            }
        }
    }

    template<typename Func>
    double benchmark(Func&& func, int warmup_iters = 5, int test_iters = 20) {
        for (int i = 0; i < warmup_iters; ++i) {
            func();
        }
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < test_iters; ++i) {
            func();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = (end - start) / test_iters;
        return ms.count();
    }
};


// ==================== 基本功能测试 ====================

TEST_F(RmsNormTest, Basic1D) {
    const size_t hidden_dim = 64;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_1d",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    // 使用 tensor::zeros 创建并初始化
    auto input = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto output = tensor::empty({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    // 使用 fill 填充数据
    for (size_t i = 0; i < hidden_dim; ++i) {
        input.index<float>({i}) = static_cast<float>(i + 1);
        gamma.index<float>({i}) = 1.0f;
    }

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    auto status = rms_norm.forward();
    EXPECT_TRUE(status);

    // 验证
    std::vector<float> ref_output(hidden_dim);
    compute_ref_rmsnorm(input.ptr<float>(), gamma.ptr<float>(), ref_output.data(), 1, hidden_dim, 1e-6f);

    for (size_t i = 0; i < hidden_dim; ++i) {
        EXPECT_NEAR(output.index<float>({i}), ref_output[i], 1e-5f);
    }
}

TEST_F(RmsNormTest, Basic2D) {
    const size_t num_rows = 4;
    const size_t hidden_dim = 32;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_2d",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    auto input = tensor::zeros({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    // 填充数据
    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        input.ptr<float>()[i] = static_cast<float>((i % 10) + 1);
    }
    for (size_t i = 0; i < hidden_dim; ++i) {
        gamma.index<float>({i}) = static_cast<float>(i + 1);
    }

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    auto status = rms_norm.forward();
    EXPECT_TRUE(status);

    // 验证
    std::vector<float> ref_output(num_rows * hidden_dim);
    compute_ref_rmsnorm(input.ptr<float>(), gamma.ptr<float>(), ref_output.data(), num_rows, hidden_dim, 1e-6f);

    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        EXPECT_NEAR(output.ptr<float>()[i], ref_output[i], 1e-5f);
    }
}

TEST_F(RmsNormTest, AllOnesInput) {
    const size_t hidden_dim = 128;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_ones",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    // 使用 ones 创建全 1 张量
    auto input = tensor::ones({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::ones({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto output = tensor::empty({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    auto status = rms_norm.forward();
    EXPECT_TRUE(status);

    // 输入全为 1，gamma 全为 1，RMS = 1，输出应该全为 1
    for (size_t i = 0; i < hidden_dim; ++i) {
        EXPECT_NEAR(output.index<float>({i}), 1.0f, 3e-4f);
    }
}

TEST_F(RmsNormTest, ZeroInput) {
    const size_t hidden_dim = 64;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_zero",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    // 使用 zeros 创建全 0 张量
    auto input = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto output = tensor::empty({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    auto status = rms_norm.forward();
    EXPECT_TRUE(status);

    // 输入全为 0，输出应该全为 0
    for (size_t i = 0; i < hidden_dim; ++i) {
        EXPECT_FLOAT_EQ(output.index<float>({i}), 0.0f);
    }
}

// ==================== 大向量测试 ====================

TEST_F(RmsNormTest, LargeHiddenDim) {
    const size_t num_rows = 128;
    const size_t hidden_dim = 4096;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_large",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    auto input = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::ones({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    // 使用 fill 填充随机数据
    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        input.ptr<float>()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    double avg_ms = benchmark([&]() {
        rms_norm.forward();
    }, 3, 10);

    std::cout << "[Perf] RMSNorm CPU " << num_rows << "x" << hidden_dim
              << ": " << avg_ms << " ms/op" << std::endl;

    // 抽样验证
    std::vector<float> ref_output(hidden_dim);
    compute_ref_rmsnorm(input.ptr<float>(), gamma.ptr<float>(), ref_output.data(), 1, hidden_dim, 1e-6f);

    for (size_t i = 0; i < hidden_dim; i += 100) {
        EXPECT_NEAR(output.ptr<float>()[i], ref_output[i], 1e-4f);
    }
}

// ==================== 边界条件测试 ====================

TEST_F(RmsNormTest, CheckWrongGammaShape) {
    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_check_wrong_gamma",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );

    auto input = tensor::zeros({64}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto gamma = tensor::zeros({32}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);  // 错误的 shape
    auto output = tensor::empty({64}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, input);
    rms_norm.set_output(0, output);

    EXPECT_FALSE(rms_norm.check());
}

TEST_F(RmsNormTest, DifferentEpsValues) {
    const size_t hidden_dim = 64;
    float eps_values[] = {1e-5f, 1e-6f, 1e-7f};

    for (float eps : eps_values) {
        ops::RmsNorm rms_norm(
            base::DataType::kDataFloat32,
            "test_rmsnorm_eps",
            base::DeviceType::kDeviceCPU,
            eps
        );

        auto input = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
        auto gamma = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
        auto output = tensor::empty({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

        // 填充小值，eps 影响更明显
        for (size_t i = 0; i < hidden_dim; ++i) {
            input.index<float>({i}) = 0.01f;
        }
        gamma.index<float>({0}) = 1.0f;

        rms_norm.add_weight(gamma);
        rms_norm.set_input(0, input);
        rms_norm.set_output(0, output);

        EXPECT_TRUE(rms_norm.forward());
        EXPECT_FLOAT_EQ(rms_norm.get_eps(), eps);
    }
}


// ==================== CUDA 测试 ====================

class CudaRmsNormTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
        cuda_allocator_ = base::CudaDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
    std::shared_ptr<base::DeviceAllocator> cuda_allocator_;
};

TEST_F(CudaRmsNormTest, BasicCUDA) {
    const size_t num_rows = 8;
    const size_t hidden_dim = 64;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_cuda",
        base::DeviceType::kDeviceCUDA,
        1e-6f
    );

    // 在 CPU 上创建并初始化数据
    auto cpu_input = tensor::zeros({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto cpu_gamma = tensor::zeros({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        cpu_input.ptr<float>()[i] = static_cast<float>((i % 10) + 1);
    }
    for (size_t i = 0; i < hidden_dim; ++i) {
        cpu_gamma.index<float>({i}) = static_cast<float>(i + 1);
    }

    // 使用 cuda() 转移到 GPU
    cpu_input.cuda(0);
    cpu_gamma.cuda(0);
    auto output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);

    rms_norm.add_weight(cpu_gamma);
    rms_norm.set_input(0, cpu_input);
    rms_norm.set_output(0, output);

    EXPECT_TRUE(rms_norm.forward());

    // 使用 cpu() 转移回 CPU 验证
    output.cpu();
    cpu_input.cpu();
    cpu_gamma.cpu();

    // 计算参考结果
    std::vector<float> ref_output(num_rows * hidden_dim);
    for (size_t row = 0; row < num_rows; ++row) {
        float sum_squares = 0.0f;
        for (size_t i = 0; i < hidden_dim; ++i) {
            float v = cpu_input.index<float>({row, i});
            sum_squares += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_dim) + 1e-6f);
        for (size_t i = 0; i < hidden_dim; ++i) {
            ref_output[row * hidden_dim + i] = cpu_input.index<float>({row, i}) * inv_rms * cpu_gamma.index<float>({i});
        }
    }

    // 验证
    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        EXPECT_NEAR(output.ptr<float>()[i], ref_output[i], 3e-4f);
    }
}

TEST_F(CudaRmsNormTest, TypicalLLMSize) {
    const size_t num_rows = 128;
    const size_t hidden_dim = 4096;

    ops::RmsNorm rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_llm",
        base::DeviceType::kDeviceCUDA,
        1e-6f
    );

    // CPU 初始化
    auto cpu_input = tensor::zeros({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    cpu_input.fill(0.01f, INT_MIN, num_rows * hidden_dim);

    // 转移到 GPU
    cpu_input.cuda(0);
    auto gamma = tensor::ones({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);
    auto output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);

    rms_norm.add_weight(gamma);
    rms_norm.set_input(0, cpu_input);
    rms_norm.set_output(0, output);

    // 预热
    for (int i = 0; i < 5; ++i) {
        rms_norm.forward();
    }
    cudaDeviceSynchronize();

    // 性能测试
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 20; ++i) {
        rms_norm.forward();
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> avg_ms = (end - start) / 20;
    double bytes = num_rows * hidden_dim * sizeof(float) * 3;
    double bandwidth = (bytes / (avg_ms.count() / 1000.0)) / (1024 * 1024 * 1024);

    std::cout << "[Perf] RMSNorm CUDA " << num_rows << "x" << hidden_dim
              << ": " << avg_ms.count() << " ms/op, "
              << bandwidth << " GB/s" << std::endl;

    // 验证
    output.cpu();
    cpu_input.cpu();  // 复制一份用于参考计算
    for (size_t row = 0; row < num_rows; ++row) {
        float sum_squares = 0.0f;
        for (size_t i = 0; i < hidden_dim; ++i) {
            float v = cpu_input.index<float>({row, i});
            sum_squares += v * v;
        }
        float inv_rms = 1.0f / std::sqrt(sum_squares / static_cast<float>(hidden_dim) + 1e-6f);
        for (size_t i = 0; i < hidden_dim; ++i) {
            float expected = cpu_input.index<float>({row, i}) * inv_rms * 1.0f;
            EXPECT_NEAR(output.index<float>({row, i}), expected, 1e-4f);
        }
    }
}

TEST_F(CudaRmsNormTest, CompareWithCPU) {
    const size_t num_rows = 16;
    const size_t hidden_dim = 256;

    // CPU 初始化数据
    auto cpu_input = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        cpu_input.ptr<float>()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    }
    auto cpu_gamma = tensor::ones({hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    // CPU 版本
    ops::RmsNorm cpu_rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_cpu",
        base::DeviceType::kDeviceCPU,
        1e-6f
    );
    auto cpu_output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    cpu_rms_norm.add_weight(cpu_gamma);
    cpu_rms_norm.set_input(0, cpu_input);
    cpu_rms_norm.set_output(0, cpu_output);
    cpu_rms_norm.forward();

    // CUDA 版本
    ops::RmsNorm cuda_rms_norm(
        base::DataType::kDataFloat32,
        "test_rmsnorm_cuda",
        base::DeviceType::kDeviceCUDA,
        1e-6f
    );
    cpu_input.cuda(0);
    cpu_gamma.cuda(0);
    auto cuda_output = tensor::empty({num_rows, hidden_dim}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);

    cuda_rms_norm.add_weight(cpu_gamma);
    cuda_rms_norm.set_input(0, cpu_input);
    cuda_rms_norm.set_output(0, cuda_output);
    cuda_rms_norm.forward();

    // 转移回 CPU 比较
    cuda_output.cpu();
    auto* cpu_result = cpu_output.ptr<float>();

    for (size_t i = 0; i < num_rows * hidden_dim; ++i) {
        EXPECT_NEAR(cuda_output.ptr<float>()[i], cpu_result[i], 1e-4f);
    }
}
