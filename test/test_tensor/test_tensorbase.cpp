//
// Created by 12591 on 2026/2/28.
//

#include <gtest/gtest.h>
#include "../../src/include/tensor/tensorbase.h"
#include "../../src/include/base/alloc.h"

using namespace qwi;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;
};

// 测试默认构造函数
TEST_F(TensorTest, DefaultConstructor) {
    tensor::Tensor t;
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试 1D Tensor 构造函数（不分配内存）
TEST_F(TensorTest, Constructor1D_NoAlloc) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        10,  // dim0
        false,  // need_alloc
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    // 不分配内存，byte_size 应该为 0
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试 1D Tensor 构造函数（分配内存）
TEST_F(TensorTest, Constructor1D_WithAlloc) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        10,  // dim0 = 10
        true,  // need_alloc
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    // 10 * sizeof(float) = 40 bytes
    EXPECT_EQ(t.byte_size(), 10 * sizeof(float));
}

// 测试 2D Tensor 构造函数
TEST_F(TensorTest, Constructor2D) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        3, 4,  // 3x4 matrix
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 3 * 4 * sizeof(float));
}

// 测试 3D Tensor 构造函数
TEST_F(TensorTest, Constructor3D) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        2, 3, 4,  // 2x3x4
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 2 * 3 * 4 * sizeof(float));
}

// 测试 4D Tensor 构造函数
TEST_F(TensorTest, Constructor4D) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        2, 3, 4, 5,  // 2x3x4x5
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 2 * 3 * 4 * 5 * sizeof(float));
}

// 测试 vector 构造函数
TEST_F(TensorTest, ConstructorVector) {
    std::vector<size_t> dims = {2, 3, 4, 5, 6};
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        dims,
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    size_t expected_size = 2 * 3 * 4 * 5 * 6 * sizeof(float);
    EXPECT_EQ(t.byte_size(), expected_size);
}

// 测试不同数据类型
TEST_F(TensorTest, DifferentDataTypes) {
    // Float32
    {
        tensor::Tensor t(base::DataType::kDataFloat32, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 4);
    }
    // Float16
    {
        tensor::Tensor t(base::DataType::kDataFloat16, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 2);
    }
    // Float8
    {
        tensor::Tensor t(base::DataType::kDataFloat8, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 1);
    }
    // Int32
    {
        tensor::Tensor t(base::DataType::kDataInt32, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 4);
    }
    // Int16
    {
        tensor::Tensor t(base::DataType::kDataInt16, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 2);
    }
    // Int8
    {
        tensor::Tensor t(base::DataType::kDataInt8, 10, true, base::DeviceType::kDeviceCPU, cpu_allocator_);
        EXPECT_EQ(t.byte_size(), 10 * 1);
    }
}

// 测试空 allocator 情况
TEST_F(TensorTest, NullAllocator) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        10,
        true,
        base::DeviceType::kDeviceCPU,
        nullptr  // null allocator
    );
    // 应该无法分配内存
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试外部 buffer 传入
TEST_F(TensorTest, ExternalBuffer) {
    // 创建一个外部 buffer
    size_t byte_size = 100 * sizeof(float);
    auto memory_buffer = base::MemoryBuffer(
        nullptr, byte_size, true, 0, base::DeviceType::kDeviceCPU
    );
    auto external_buffer = std::make_shared<base::Buffer>(
        memory_buffer, cpu_allocator_, false
    );

    tensor::Tensor t(
        base::DataType::kDataFloat32,
        100,
        false,  // 不需要分配，使用外部 buffer
        base::DeviceType::kDeviceCPU,
        nullptr,
        external_buffer
    );

    EXPECT_EQ(t.byte_size(), byte_size);
}

// 测试 need_alloc=false 但提供 allocator 的情况
TEST_F(TensorTest, NoAllocWithAllocator) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        10,
        false,  // 不需要分配
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    // 虽然提供了 allocator，但 need_alloc=false，所以不分配
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试大 Tensor
TEST_F(TensorTest, LargeTensor) {
    std::vector<size_t> dims = {1024, 1024};  // 1M 元素
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        dims,
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 1024 * 1024 * sizeof(float));
}

// 测试 0 维度 Tensor
TEST_F(TensorTest, ZeroDimension) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        0,  // 0 元素
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试多维度的空 Tensor
TEST_F(TensorTest, MultiDimWithZero) {
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        3, 0, 4,  // 3x0x4 = 0 元素
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试空 vector 构造函数
TEST_F(TensorTest, EmptyVectorDims) {
    std::vector<size_t> dims;
    tensor::Tensor t(
        base::DataType::kDataFloat32,
        dims,
        true,
        base::DeviceType::kDeviceCPU,
        cpu_allocator_
    );
    EXPECT_EQ(t.byte_size(), 0);
}

// 测试不同数据类型的组合
TEST_F(TensorTest, MixedDataTypesAndDims) {
    struct TestCase {
        base::DataType dtype;
        size_t elem_size;
        std::vector<size_t> dims;
    };

    std::vector<TestCase> test_cases = {
        {base::DataType::kDataFloat32, 4, {10, 20}},
        {base::DataType::kDataFloat16, 2, {10, 20}},
        {base::DataType::kDataInt32, 4, {5, 6, 7}},
        {base::DataType::kDataInt8, 1, {100}},
    };

    for (const auto& tc : test_cases) {
        tensor::Tensor t(tc.dtype, tc.dims, true, base::DeviceType::kDeviceCPU, cpu_allocator_);

        size_t expected_elems = 1;
        for (auto d : tc.dims) {
            expected_elems *= d;
        }
        EXPECT_EQ(t.byte_size(), expected_elems * tc.elem_size);
    }
}

// 测试 allocate 方法直接调用
TEST_F(TensorTest, DirectAllocate) {
    tensor::Tensor t;
    size_t byte_size = 100 * sizeof(float);

    auto status = t.allocate(cpu_allocator_, byte_size, base::DeviceType::kDeviceCPU);

    EXPECT_EQ(status, base::ReturnStatus::Success);
    EXPECT_EQ(t.byte_size(), byte_size);
}

// 测试 allocate 失败情况（null allocator）
TEST_F(TensorTest, AllocateNullAllocator) {
    tensor::Tensor t;
    auto status = t.allocate(nullptr, 100, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(status, base::ReturnStatus::NoAllocator);
}

// 测试 allocate 0 字节
TEST_F(TensorTest, AllocateZeroBytes) {
    tensor::Tensor t;
    auto status = t.allocate(cpu_allocator_, 0, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(status, base::ReturnStatus::ZeroByteSize);
}

// 测试 init_buffer 方法 - 外部数据路径
TEST_F(TensorTest, InitBufferExternal) {
    size_t byte_size = 100 * sizeof(float);
    auto memory_buffer = base::MemoryBuffer(
        nullptr, byte_size, true, 0, base::DeviceType::kDeviceCPU
    );
    auto external_buffer = std::make_shared<base::Buffer>(
        memory_buffer, cpu_allocator_, false
    );

    tensor::Tensor t;
    auto status = t.init_buffer(
        nullptr,           // 无 allocator
        base::DeviceType::kDeviceCPU,
        false,             // 不需要分配
        external_buffer    // 外部 buffer
    );

    EXPECT_EQ(status, base::ReturnStatus::Success);
    EXPECT_EQ(t.byte_size(), byte_size);
}

// 测试 init_buffer 方法 - 需要分配路径
TEST_F(TensorTest, InitBufferNeedAlloc) {
    // 先创建一个带维度的 Tensor
    tensor::Tensor t(base::DataType::kDataFloat32, std::vector<size_t>{10, 10}, false);

    auto status = t.init_buffer(
        cpu_allocator_,
        base::DeviceType::kDeviceCPU,
        true  // 需要分配
    );

    EXPECT_EQ(status, base::ReturnStatus::Success);
    EXPECT_EQ(t.byte_size(), 100 * sizeof(float));
}

// 主函数由 gtest_main 提供
