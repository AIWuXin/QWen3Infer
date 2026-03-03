//
// Created by Administrator on 2026/3/3.
//


#include <gtest/gtest.h>
#include "../../src/include/base/type_extension.hpp"

using namespace qwi::base;

class StatusTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

// 测试默认构造函数
TEST_F(StatusTest, DefaultConstructor) {
    Status s;
    EXPECT_EQ(s.get_code(), ReturnStatus::Success);
    EXPECT_EQ(s.get_message(), "");
    EXPECT_TRUE(static_cast<bool>(s));  // Success 应该转换为 true
}

// 测试带参数的构造函数
TEST_F(StatusTest, ParameterizedConstructor) {
    Status s1(ReturnStatus::Success, "success message");
    EXPECT_EQ(s1.get_code(), ReturnStatus::Success);
    EXPECT_EQ(s1.get_message(), "success message");

    Status s2(ReturnStatus::Error, "error occurred");
    EXPECT_EQ(s2.get_code(), ReturnStatus::Error);
    EXPECT_EQ(s2.get_message(), "error occurred");
    EXPECT_FALSE(static_cast<bool>(s2));  // Error 应该转换为 false
}

// 测试拷贝构造函数
TEST_F(StatusTest, CopyConstructor) {
    Status original(ReturnStatus::AlreadyAllocated, "already allocated");
    Status copy(original);

    EXPECT_EQ(copy.get_code(), ReturnStatus::AlreadyAllocated);
    EXPECT_EQ(copy.get_message(), "already allocated");
}

// 测试拷贝赋值操作符
TEST_F(StatusTest, CopyAssignment) {
    Status s1(ReturnStatus::Success, "original");
    Status s2(ReturnStatus::Error, "error");

    s2 = s1;
    EXPECT_EQ(s2.get_code(), ReturnStatus::Success);
    EXPECT_EQ(s2.get_message(), "original");
}

// 测试与 ReturnStatus 的赋值操作
TEST_F(StatusTest, ReturnStatusAssignment) {
    Status s;
    s = ReturnStatus::ErrorAllocating;
    EXPECT_EQ(s.get_code(), ReturnStatus::ErrorAllocating);

    s = ReturnStatus::ZeroByteSize;
    EXPECT_EQ(s.get_code(), ReturnStatus::ZeroByteSize);
}

// 测试 == 操作符
TEST_F(StatusTest, EqualityOperator) {
    Status s(ReturnStatus::Success);
    EXPECT_TRUE(s == ReturnStatus::Success);
    EXPECT_FALSE(s == ReturnStatus::Error);

    Status s2(ReturnStatus::NoAllocator);
    EXPECT_TRUE(s2 == ReturnStatus::NoAllocator);
    EXPECT_FALSE(s2 == ReturnStatus::Success);
}

// 测试 != 操作符
TEST_F(StatusTest, InequalityOperator) {
    Status s(ReturnStatus::Success);
    EXPECT_FALSE(s != ReturnStatus::Success);
    EXPECT_TRUE(s != ReturnStatus::Error);
}

// 测试转换为 int
TEST_F(StatusTest, IntConversion) {
    Status s1(ReturnStatus::Success);
    EXPECT_EQ(static_cast<int>(s1), 0);

    Status s2(ReturnStatus::Error);
    EXPECT_EQ(static_cast<int>(s2), -1);

    Status s3(ReturnStatus::AlreadyAllocated);
    EXPECT_EQ(static_cast<int>(s3), 1);

    Status s4(ReturnStatus::ErrorAllocating);
    EXPECT_EQ(static_cast<int>(s4), -2);
}

// 测试转换为 bool（成功状态返回 true）
TEST_F(StatusTest, BoolConversionSuccess) {
    Status s1(ReturnStatus::Success);
    EXPECT_TRUE(static_cast<bool>(s1));

    Status s2(ReturnStatus::AlreadyAllocated);
    EXPECT_TRUE(static_cast<bool>(s2));

    Status s3(ReturnStatus::ZeroByteSize);
    EXPECT_TRUE(static_cast<bool>(s3));
}

// 测试转换为 bool（失败状态返回 false）
TEST_F(StatusTest, BoolConversionFailure) {
    Status s1(ReturnStatus::Error);
    EXPECT_FALSE(static_cast<bool>(s1));

    Status s2(ReturnStatus::ErrorAllocating);
    EXPECT_FALSE(static_cast<bool>(s2));

    Status s3(ReturnStatus::NoAllocator);
    EXPECT_FALSE(static_cast<bool>(s3));
}

// 测试 get_code() 方法
TEST_F(StatusTest, GetCode) {
    Status s1(ReturnStatus::Success);
    EXPECT_EQ(s1.get_code(), ReturnStatus::Success);

    Status s2(ReturnStatus::Error);
    EXPECT_EQ(s2.get_code(), ReturnStatus::Error);
}

// 测试 get_message() 方法
TEST_F(StatusTest, GetMessage) {
    Status s(ReturnStatus::Success, "test message");
    EXPECT_EQ(s.get_message(), "test message");

    Status s2;  // 默认构造，空消息
    EXPECT_EQ(s2.get_message(), "");
}

// 测试 set_err_msg() 方法
TEST_F(StatusTest, SetErrMsg) {
    Status s(ReturnStatus::Error);
    EXPECT_EQ(s.get_message(), "");

    s.set_err_msg("new error message");
    EXPECT_EQ(s.get_message(), "new error message");

    s.set_err_msg("updated message");
    EXPECT_EQ(s.get_message(), "updated message");
}

// 测试链式设置错误消息
TEST_F(StatusTest, ChainedSetErrMsg) {
    Status s;
    s = ReturnStatus::Error;
    s.set_err_msg("step 1");
    EXPECT_EQ(s.get_message(), "step 1");

    s.set_err_msg("step 2");
    EXPECT_EQ(s.get_message(), "step 2");
}

// 测试所有 ReturnStatus 枚举值
TEST_F(StatusTest, AllReturnStatusValues) {
    // 成功状态 (code >= 0)
    Status s1(ReturnStatus::Success);
    EXPECT_TRUE(s1 == ReturnStatus::Success);
    EXPECT_TRUE(static_cast<bool>(s1));

    Status s2(ReturnStatus::AlreadyAllocated);
    EXPECT_TRUE(s2 == ReturnStatus::AlreadyAllocated);
    EXPECT_TRUE(static_cast<bool>(s2));

    Status s3(ReturnStatus::ZeroByteSize);
    EXPECT_TRUE(s3 == ReturnStatus::ZeroByteSize);
    EXPECT_TRUE(static_cast<bool>(s3));

    // 失败状态 (code < 0)
    Status s4(ReturnStatus::Error);
    EXPECT_TRUE(s4 == ReturnStatus::Error);
    EXPECT_FALSE(static_cast<bool>(s4));

    Status s5(ReturnStatus::ErrorAllocating);
    EXPECT_TRUE(s5 == ReturnStatus::ErrorAllocating);
    EXPECT_FALSE(static_cast<bool>(s5));

    Status s6(ReturnStatus::NoAllocator);
    EXPECT_TRUE(s6 == ReturnStatus::NoAllocator);
    EXPECT_FALSE(static_cast<bool>(s6));
}

// 测试在 if 条件中使用 Status
TEST_F(StatusTest, UseInIfCondition) {
    Status success(ReturnStatus::Success);
    Status error(ReturnStatus::Error);

    if (success) {
        // 应该进入这里
        EXPECT_TRUE(true);
    } else {
        EXPECT_TRUE(false);  // 不应该到这里
    }

    if (error) {
        EXPECT_TRUE(false);  // 不应该到这里
    } else {
        // 应该进入这里
        EXPECT_TRUE(true);
    }
}

// 测试 Status 的布尔逻辑运算
TEST_F(StatusTest, BooleanLogic) {
    Status success(ReturnStatus::Success);
    Status error(ReturnStatus::Error);

    EXPECT_TRUE(success && true);
    EXPECT_FALSE(error && true);
    EXPECT_TRUE(success || false);
    EXPECT_TRUE(error || success);
}

// 主函数由 gtest_main 提供
