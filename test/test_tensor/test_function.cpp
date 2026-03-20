//
// Created by Administrator on 2026/3/17.
//

#include <gtest/gtest.h>
#include "../../src/include/tensor/function.h"
#include "../../src/include/tensor/tensorbase.h"

using namespace qwi;

class TensorFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_allocator_ = base::CpuDeviceAllocatorFactory::get_instance();
    }

    std::shared_ptr<base::DeviceAllocator> cpu_allocator_;

    // 辅助函数：创建测试用的 2D tensor
    tensor::Tensor create_test_tensor_2d() {
        auto t = tensor::empty({3, 4}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                t.index<float>({i, j}) = static_cast<float>(i * 4 + j);
            }
        }
        return t;
    }

    // 创建填充了指定值的 tensor
    tensor::Tensor create_filled_tensor(std::vector<size_t> dims, float value) {
        auto t = tensor::empty(dims, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
        auto status = t.fill(value, INT_MIN, t.size());
        EXPECT_TRUE(status);
        return t;
    }
};

// ========== 创建函数测试 ==========

TEST_F(TensorFunctionTest, EmptyCreatesUninitializedTensor) {
    auto t = tensor::empty({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(t.dims(), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.byte_size(), 6 * sizeof(float));
    EXPECT_EQ(t.get_device_type(), base::DeviceType::kDeviceCPU);
    EXPECT_EQ(t.get_data_type(), base::DataType::kDataFloat32);
}

TEST_F(TensorFunctionTest, ZerosCreatesZeroFilledTensor) {
    auto t = tensor::zeros({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto t_cuda = tensor::zeros({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCUDA);
    t_cuda.cpu();
    EXPECT_EQ(t.dims(), (std::vector<size_t>{2, 3}));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.index<float>({i, j}), 0.0f);
            EXPECT_FLOAT_EQ(t_cuda.index<float>({i, j}), 0.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OnesCreatesOneFilledTensor) {
    auto t = tensor::ones({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(t.dims(), (std::vector<size_t>{2, 3}));

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(t.index<float>({i, j}), 1.0f);
        }
    }
}

TEST_F(TensorFunctionTest, EmptyWithDifferentDataTypes) {
    auto t_int32 = tensor::empty({10}, base::DataType::kDataInt32, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(t_int32.byte_size(), 10 * sizeof(int32_t));

    auto t_float16 = tensor::empty({10}, base::DataType::kDataFloat16, base::DeviceType::kDeviceCPU);
    EXPECT_EQ(t_float16.byte_size(), 10 * sizeof(uint16_t));
}

// ========== 逐元素运算测试（Tensor-Tensor）==========

TEST_F(TensorFunctionTest, AddTwoTensors) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);

    auto c = tensor::add(a, b);

    EXPECT_EQ(c.dims(), (std::vector<size_t>{2, 3}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 3.0f);
        }
    }
}

TEST_F(TensorFunctionTest, SubTwoTensors) {
    auto a = create_filled_tensor({2, 3}, 5.0f);
    auto b = create_filled_tensor({2, 3}, 3.0f);

    auto c = tensor::sub(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 2.0f);
        }
    }
}

TEST_F(TensorFunctionTest, MulTwoTensors) {
    auto a = create_filled_tensor({2, 3}, 3.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    auto c = tensor::mul(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 12.0f);
        }
    }
}

TEST_F(TensorFunctionTest, DivTwoTensors) {
    auto a = create_filled_tensor({2, 3}, 12.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    auto c = tensor::div(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 3.0f);
        }
    }
}

// ========== In-place 运算测试 ==========

TEST_F(TensorFunctionTest, AddInPlace) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);

    auto& ref = tensor::add_(a, b);

    // 检查返回值引用
    EXPECT_EQ(&ref, &a);

    // 检查 a 被修改
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 3.0f);
        }
    }
}

TEST_F(TensorFunctionTest, SubInPlace) {
    auto a = create_filled_tensor({2, 3}, 5.0f);
    auto b = create_filled_tensor({2, 3}, 3.0f);

    tensor::sub_(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 2.0f);
        }
    }
}

TEST_F(TensorFunctionTest, MulInPlace) {
    auto a = create_filled_tensor({2, 3}, 3.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    tensor::mul_(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 12.0f);
        }
    }
}

TEST_F(TensorFunctionTest, DivInPlace) {
    auto a = create_filled_tensor({2, 3}, 12.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    tensor::div_(a, b);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 3.0f);
        }
    }
}

// ========== Out 参数测试 ==========

TEST_F(TensorFunctionTest, AddWithOutParameter) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);
    auto out = tensor::empty({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    auto result = tensor::add(a, b, out);

    // 应该返回 out 的引用
    EXPECT_EQ(result.ptr<float>(), out.ptr<float>());

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(out.index<float>({i, j}), 3.0f);
        }
    }
}

// ========== 标量运算测试 ==========

TEST_F(TensorFunctionTest, AddScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = tensor::add(a, 3.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 8.0f);
        }
    }
}

TEST_F(TensorFunctionTest, SubScalar) {
    auto a = create_filled_tensor({2, 3}, 10.0f);

    auto c = tensor::sub(a, 3.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 7.0f);
        }
    }
}

TEST_F(TensorFunctionTest, MulScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = tensor::mul(a, 3.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 15.0f);
        }
    }
}

TEST_F(TensorFunctionTest, DivScalar) {
    auto a = create_filled_tensor({2, 3}, 12.0f);

    auto c = tensor::div(a, 3.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 4.0f);
        }
    }
}

// ========== 标量 In-place 测试 ==========

TEST_F(TensorFunctionTest, AddScalarInPlace) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    tensor::add_(a, 3.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 8.0f);
        }
    }
}

TEST_F(TensorFunctionTest, MulScalarInPlace) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    tensor::mul_(a, 2.0);

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 10.0f);
        }
    }
}

// ========== 运算符重载测试 ==========

TEST_F(TensorFunctionTest, OperatorAdd) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);

    auto c = a + b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 3.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorSub) {
    auto a = create_filled_tensor({2, 3}, 5.0f);
    auto b = create_filled_tensor({2, 3}, 3.0f);

    auto c = a - b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 2.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorMul) {
    auto a = create_filled_tensor({2, 3}, 3.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    auto c = a * b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 12.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorDiv) {
    auto a = create_filled_tensor({2, 3}, 12.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    auto c = a / b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 3.0f);
        }
    }
}

// ========== 标量运算符测试 ==========

TEST_F(TensorFunctionTest, OperatorAddScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = a + 3.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 8.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorMulScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = a * 2.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 10.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorScalarAddTensor) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = 3.0 + a;  // 标量在前

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 8.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorScalarMulTensor) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    auto c = 2.0 * a;  // 标量在前

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(c.index<float>({i, j}), 10.0f);
        }
    }
}

// ========== 复合赋值运算符测试 ==========

TEST_F(TensorFunctionTest, OperatorAddAssign) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);

    a += b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 3.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorSubAssign) {
    auto a = create_filled_tensor({2, 3}, 5.0f);
    auto b = create_filled_tensor({2, 3}, 3.0f);

    a -= b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 2.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorMulAssign) {
    auto a = create_filled_tensor({2, 3}, 3.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    a *= b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 12.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorDivAssign) {
    auto a = create_filled_tensor({2, 3}, 12.0f);
    auto b = create_filled_tensor({2, 3}, 4.0f);

    a /= b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 3.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorAddAssignScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    a += 3.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 8.0f);
        }
    }
}

TEST_F(TensorFunctionTest, OperatorMulAssignScalar) {
    auto a = create_filled_tensor({2, 3}, 5.0f);

    a *= 2.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(a.index<float>({i, j}), 10.0f);
        }
    }
}

// ========== 全局归约测试 ==========

TEST_F(TensorFunctionTest, SumAllElements) {
    // 1+2+3+4+5+6 = 21
    auto t = create_test_tensor_2d();  // 0,1,2,3,4,5,6,7,8,9,10,11

    auto result = tensor::sum(t);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1}));
    EXPECT_FLOAT_EQ(result.index<float>(0), 66.0f);  // 0+1+...+11 = 66
}

TEST_F(TensorFunctionTest, MeanAllElements) {
    auto t = create_filled_tensor({2, 3}, 10.0f);  // 全是 10

    auto result = tensor::mean(t);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1}));
    EXPECT_FLOAT_EQ(result.index<float>(0), 10.0f);
}

TEST_F(TensorFunctionTest, MaxAllElements) {
    auto t = create_test_tensor_2d();  // 最大值是 11

    auto result = tensor::max(t);

    EXPECT_FLOAT_EQ(result.index<float>(0), 11.0f);
}

TEST_F(TensorFunctionTest, MinAllElements) {
    auto t = create_test_tensor_2d();  // 最小值是 0

    auto result = tensor::min(t);

    EXPECT_FLOAT_EQ(result.index<float>(0), 0.0f);
}

TEST_F(TensorFunctionTest, AllAllElements) {
    // 全是非零值
    auto t1 = create_filled_tensor({2, 3}, 1.0f);
    auto result1 = tensor::all(t1);
    EXPECT_FLOAT_EQ(result1.index<float>(0), 1.0f);  // true

    // 有零值
    auto t2 = tensor::zeros({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    t2.index<float>({0, 0}) = 1.0f;  // 只有一个是非零
    auto result2 = tensor::all(t2);
    EXPECT_FLOAT_EQ(result2.index<float>(0), 0.0f);  // false
}

TEST_F(TensorFunctionTest, AnyAllElements) {
    // 有非零值
    auto t1 = create_filled_tensor({2, 3}, 1.0f);
    auto result1 = tensor::any(t1);
    EXPECT_FLOAT_EQ(result1.index<float>(0), 1.0f);  // true

    // 全是零值
    auto t2 = tensor::zeros({2, 3}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto result2 = tensor::any(t2);
    EXPECT_FLOAT_EQ(result2.index<float>(0), 0.0f);  // false
}

// ========== 维度归约测试 ==========

TEST_F(TensorFunctionTest, SumDim0) {
    // 3x4 tensor
    // [[0,1,2,3],
    //  [4,5,6,7],
    //  [8,9,10,11]]
    // sum on dim 0 -> [12, 15, 18, 21]
    auto t = create_test_tensor_2d();

    auto result = tensor::sum(t, 0);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1, 4}));
    EXPECT_FLOAT_EQ(result.index<float>({0, 0}), 12.0f);
    EXPECT_FLOAT_EQ(result.index<float>({0, 1}), 15.0f);
    EXPECT_FLOAT_EQ(result.index<float>({0, 2}), 18.0f);
    EXPECT_FLOAT_EQ(result.index<float>({0, 3}), 21.0f);
}

TEST_F(TensorFunctionTest, SumDim1) {
    // 3x4 tensor
    // sum on dim 1 -> [6, 22, 38]
    auto t = create_test_tensor_2d();

    auto result = tensor::sum(t, 1);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{3, 1}));
    EXPECT_FLOAT_EQ(result.index<float>({0, 0}), 6.0f);   // 0+1+2+3
    EXPECT_FLOAT_EQ(result.index<float>({1, 0}), 22.0f);  // 4+5+6+7
    EXPECT_FLOAT_EQ(result.index<float>({2, 0}), 38.0f);  // 8+9+10+11
}

TEST_F(TensorFunctionTest, MeanDim0) {
    auto t = create_filled_tensor({2, 4}, 10.0f);

    auto result = tensor::mean(t, 0);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1, 4}));
    for (size_t j = 0; j < 4; ++j) {
        EXPECT_FLOAT_EQ(result.index<float>({0, j}), 10.0f);
    }
}

TEST_F(TensorFunctionTest, MaxDim0) {
    // 3x4 tensor, max on dim 0
    auto t = create_test_tensor_2d();

    auto result = tensor::max(t, 0);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1, 4}));
    // [8,9,10,11]
    EXPECT_FLOAT_EQ(result.index<float>({0, 0}), 8.0f);
    EXPECT_FLOAT_EQ(result.index<float>({0, 3}), 11.0f);
}

TEST_F(TensorFunctionTest, MinDim0) {
    // 3x4 tensor, min on dim 0
    auto t = create_test_tensor_2d();

    auto result = tensor::min(t, 0);

    EXPECT_EQ(result.dims(), (std::vector<size_t>{1, 4}));
    // [0,1,2,3]
    EXPECT_FLOAT_EQ(result.index<float>({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(result.index<float>({0, 3}), 3.0f);
}

// ========== 负维度索引测试 ==========

TEST_F(TensorFunctionTest, SumNegativeDim) {
    auto t = create_test_tensor_2d();  // 3x4

    // dim=-1 应该等价于 dim=1
    auto result1 = tensor::sum(t, -1);
    auto result2 = tensor::sum(t, 1);

    EXPECT_EQ(result1.dims(), result2.dims());
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(result1.index<float>({i, 0}), result2.index<float>({i, 0}));
    }
}

// ========== 链式操作测试 ==========

TEST_F(TensorFunctionTest, ChainOperations) {
    auto a = create_filled_tensor({2, 3}, 1.0f);
    auto b = create_filled_tensor({2, 3}, 2.0f);
    auto c = create_filled_tensor({2, 3}, 3.0f);

    // (a + b) * c = (1+2)*3 = 9
    auto result = (a + b) * c;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(result.index<float>({i, j}), 9.0f);
        }
    }
}

TEST_F(TensorFunctionTest, ChainWithScalar) {
    auto a = create_filled_tensor({2, 3}, 2.0f);

    // a * 3 + 4 = 2*3+4 = 10
    auto result = a * 3.0 + 4.0;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(result.index<float>({i, j}), 10.0f);
        }
    }
}

// ========== 高维 Tensor 测试 ==========

TEST_F(TensorFunctionTest, HighDimOperations) {
    // 3D tensor: 2x3x4
    auto a = tensor::ones({2, 3, 4}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);
    auto b = tensor::ones({2, 3, 4}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    auto c = a + b;

    EXPECT_EQ(c.dims(), (std::vector<size_t>{2, 3, 4}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                EXPECT_FLOAT_EQ(c.index<float>({i, j, k}), 2.0f);
            }
        }
    }
}

TEST_F(TensorFunctionTest, HighDimSum) {
    auto t = tensor::ones({2, 3, 4}, base::DataType::kDataFloat32, base::DeviceType::kDeviceCPU);

    auto result = tensor::sum(t, 1);  // sum on dim 1

    EXPECT_EQ(result.dims(), (std::vector<size_t>{2, 1, 4}));
    for (size_t i = 0; i < 2; ++i) {
        for (size_t k = 0; k < 4; ++k) {
            EXPECT_FLOAT_EQ(result.index<float>({i, 0, k}), 3.0f);  // 1+1+1
        }
    }
}

// ========== 性能/大 Tensor 测试 ==========

TEST_F(TensorFunctionTest, LargeTensorAdd) {
    const size_t n = 2000;
    auto a = create_filled_tensor({n, n}, 1.0f);
    auto b = create_filled_tensor({n, n}, 2.0f);

    auto c = a + b;

    // 抽样检查
    EXPECT_FLOAT_EQ(c.index<float>({0, 0}), 3.0f);
    EXPECT_FLOAT_EQ(c.index<float>({n/2, n/2}), 3.0f);
    EXPECT_FLOAT_EQ(c.index<float>({n-1, n-1}), 3.0f);
}

// 主函数由 gtest_main 提供

