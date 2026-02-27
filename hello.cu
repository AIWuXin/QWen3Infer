//
// Created by Administrator on 2026/2/24.
//


#include <iostream>
#include "src/include/base/library.h"


int main() {
    std::cout << "=== CUDA Debug Test Started ===" << std::endl;
    std::cout << "Current device: ";

    // 显示当前 GPU 信息
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::cout << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // 调用测试函数（会触发核函数）
    run_test_kernel();

    std::cout << "================================" << std::endl;
    std::cout << "Test finished successfully!" << std::endl;

    return 0;
}
