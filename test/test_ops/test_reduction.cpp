//
// Created by Administrator on 2026/3/9.
//


#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <gtest/gtest.h>

#include "../../src/include/ops/reduction.h"
#include "../../src/include/tensor/tensorbase.h"
#include "../../src/include/base/alloc.h"

using namespace qwi;


class ReductionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;

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


// ==================== 全局规约测试 (dim = -1) ====================

TEST_F(ReductionTest, GlobalSum1D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_sum_1d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 100, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 100; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    const auto status = reduction.forward();
    EXPECT_TRUE(status);

    // Sum of 0..99 = 4950
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 4950.0f);
}

TEST_F(ReductionTest, GlobalMean2D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_mean_2d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMean,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 10, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 100; ++i) {
        ptr[i] = 5.0f;
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 5.0f);
}

TEST_F(ReductionTest, GlobalMax3D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_max_3d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMax,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 3, 4, 5, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 60; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 59.0f);
}

TEST_F(ReductionTest, GlobalMin) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_min",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMin,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 50, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 50; ++i) {
        ptr[i] = static_cast<float>(100 - i);  // 100, 99, 98, ...
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 51.0f);
}

TEST_F(ReductionTest, GlobalAllTrue) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_all_true",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceAll,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        ptr[i] = 1.0f;  // All non-zero = true
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 1.0f);
}

TEST_F(ReductionTest, GlobalAllFalse) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_all_false",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceAll,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        ptr[i] = (i < 9) ? 1.0f : 0.0f;  // Last element is 0
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 0.0f);
}

TEST_F(ReductionTest, GlobalAnyTrue) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_any_true",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceAny,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        ptr[i] = (i == 5) ? 1.0f : 0.0f;  // Only one true
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 1.0f);
}

TEST_F(ReductionTest, GlobalAnyFalse) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_global_any_false",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceAny,
        INT_MIN
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 10; ++i) {
        ptr[i] = 0.0f;  // All false
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 0.0f);
}


// ==================== 按维度规约测试 ====================

TEST_F(ReductionTest, ReduceSumDim0_2D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_sum_dim0_2d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        0
    );

    // 3x4 matrix: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    tensor::Tensor input(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    // Output after reducing dim 0: 1x4 = [15, 18, 21, 24]
    tensor::Tensor output(base::DataType::kDataFloat32, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float data[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    std::memcpy(input.ptr<float>(), data, sizeof(data));

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 15.0f);   // 1+5+9
    EXPECT_FLOAT_EQ(out[1], 18.0f);   // 2+6+10
    EXPECT_FLOAT_EQ(out[2], 21.0f);   // 3+7+11
    EXPECT_FLOAT_EQ(out[3], 24.0f);   // 4+8+12
}

TEST_F(ReductionTest, ReduceSumDim1_2D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_sum_dim1_2d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        1
    );

    // 3x4 matrix: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    tensor::Tensor input(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    // Output after reducing dim 1: 3x1 = [10, 26, 42]
    tensor::Tensor output(base::DataType::kDataFloat32, 3, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float data[] = {1,2,3,4, 5,6,7,8, 9,10,11,12};
    std::memcpy(input.ptr<float>(), data, sizeof(data));

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 10.0f);   // 1+2+3+4
    EXPECT_FLOAT_EQ(out[1], 26.0f);   // 5+6+7+8
    EXPECT_FLOAT_EQ(out[2], 42.0f);   // 9+10+11+12
}

TEST_F(ReductionTest, ReduceMeanDim0_3D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_mean_dim0_3d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMean,
        0
    );

    // 2x3x4 tensor
    tensor::Tensor input(base::DataType::kDataFloat32, 2, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 24; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    // Mean of [i, i+12] = (i + i+12) / 2 = i + 6
    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 6.0f);    // (0 + 12) / 2
    EXPECT_FLOAT_EQ(out[1], 7.0f);    // (1 + 13) / 2
    EXPECT_FLOAT_EQ(out[11], 17.0f);  // (11 + 23) / 2
}

TEST_F(ReductionTest, ReduceMaxDim1_3D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_max_dim1_3d",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMax,
        1
    );

    // 2x3x4 tensor
    tensor::Tensor input(base::DataType::kDataFloat32, 2, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 2, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < 24; ++i) {
        ptr[i] = static_cast<float>(i);
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    // Max along dim 1 (size 3): each output[i,k] = max over j of input[i,j,k]
    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 8.0f);   // max(0, 4, 8)
    EXPECT_FLOAT_EQ(out[1], 9.0f);   // max(1, 5, 9)
    EXPECT_FLOAT_EQ(out[4], 20.0f);  // max(12, 16, 20)
}

TEST_F(ReductionTest, ReduceMinDimNegative) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_min_dim_negative",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceMin,
        -1  // Should be treated as last dim (dim 1 for 2D)
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 3, 4, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 3, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float data[] = {3,1,4,2, 6,5,7,8, 9,10,2,11};
    std::memcpy(input.ptr<float>(), data, sizeof(data));

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    float* out = output.ptr<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);   // min of first row
    EXPECT_FLOAT_EQ(out[1], 5.0f);   // min of second row
    EXPECT_FLOAT_EQ(out[2], 2.0f);   // min of third row
}


// ==================== 大向量测试（触发 OpenMP） ====================

TEST_F(ReductionTest, GlobalSumLarge) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_sum_large",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    const size_t size = 100000;
    tensor::Tensor input(base::DataType::kDataFloat32, size, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = 1.0f;
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    double avg_ms = benchmark([&]() {
        reduction.forward();
    }, 10, 50);

    std::cout << "[Perf] GlobalSum 100K elements: " << avg_ms << " ms/op" << std::endl;

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], static_cast<float>(size));
}

TEST_F(ReductionTest, ReduceDimLarge2D) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_sum_dim_large",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        0
    );

    const size_t rows = 1000, cols = 1000;
    tensor::Tensor input(base::DataType::kDataFloat32, rows, cols, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, cols, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float* ptr = input.ptr<float>();
    for (size_t i = 0; i < rows * cols; ++i) {
        ptr[i] = 1.0f;
    }

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    double avg_ms = benchmark([&]() {
        reduction.forward();
    }, 5, 20);

    std::cout << "[Perf] ReduceSum dim0 1000x1000: " << avg_ms << " ms/op" << std::endl;

    float* out = output.ptr<float>();
    for (size_t j = 0; j < cols; ++j) {
        EXPECT_FLOAT_EQ(out[j], static_cast<float>(rows));
    }
}


// ==================== 边界值测试 ====================

TEST_F(ReductionTest, SingleElement) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_single_element",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    input.ptr<float>()[0] = 42.0f;

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 42.0f);
}

TEST_F(ReductionTest, EmptyInput) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_empty_input",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    tensor::Tensor input;  // Empty
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.check();
    EXPECT_FALSE(status);
}

TEST_F(ReductionTest, WrongDataType) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_wrong_dtype",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    tensor::Tensor input(base::DataType::kDataFloat16, 10, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.check();
    EXPECT_FALSE(status);
}

TEST_F(ReductionTest, ZeroInput) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_zero_input",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 100, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    cpu_allocator_->memset_zero(input.ptr<float>(), 100 * sizeof(float));

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 0.0f);
}

TEST_F(ReductionTest, NegativeNumbers) {
    ops::kernel::Reduction reduction(
        base::DataType::kDataFloat32,
        "test_negative",
        base::DeviceType::kDeviceCPU,
        base::ReductionType::kReduceSum,
        -1
    );

    tensor::Tensor input(base::DataType::kDataFloat32, 5, base::DeviceType::kDeviceCPU, cpu_allocator_);
    tensor::Tensor output(base::DataType::kDataFloat32, 1, base::DeviceType::kDeviceCPU, cpu_allocator_);

    float data[] = {-5, -3, -1, 2, 7};  // Sum = 0
    std::memcpy(input.ptr<float>(), data, sizeof(data));

    reduction.set_input(0, input);
    reduction.set_output(0, output);

    auto status = reduction.forward();
    EXPECT_TRUE(status);

    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 0.0f);
}
