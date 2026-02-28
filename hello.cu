//
// Created by Administrator on 2026/2/24.
//


#include <iostream>
#include "src/include/base/library.h"
#include "src/include/base/buffer.h"


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

    qwi::base::Buffer buffer2;
    {
        auto cpu_allocator = qwi::base::CpuDeviceAllocatorFactory::get_instance();
        auto memory_buffer = cpu_allocator->allocate(qwi::base::KB);
        cpu_allocator->memset_zero(
            memory_buffer.data,
            memory_buffer.byte_size
        );
        auto buffer = qwi::base::Buffer(
            memory_buffer, cpu_allocator,
            false
        );
        buffer.allocate();
    }

    return 0;
}
