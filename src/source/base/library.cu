//
// Created by Administrator on 2026/2/24.
//


#include "../../include/base/library.h"
#include <cstdio>
#include <cuda_runtime.h>


// 简单的测试核函数 - 在这里打断点！
__global__ void test_debug_kernel(float* data, int n) {
    // 获取线程信息
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // 断点位置 1：观察线程索引（建议在这里打断点）
    if (idx < n) {
        float value = data[idx];  // 读取数据

        // 断点位置 2：观察计算过程（或在这里打断点）
        value = value * 2.0f + 1.0f;

        // 断点位置 3：观察写入前（或在这里打断点）
        data[idx] = value;

        // 打印线程信息（调试用，会输出到控制台）
        printf("Block[%d], Thread[%d] (global idx: %d): processed value = %f\n",
               bid, tid, idx, value);
    }
}

// Host 端包装函数
void run_test_kernel() {
    const int N = 64;  // 较小的数据量，方便调试观察
    const int bytes = N * sizeof(float);

    // 主机内存
    float h_data[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // 设备内存
    float* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // 启动配置：2个block，每个block 32个线程（共64线程，每个处理一个元素）
    int blockSize = 32;
    int gridSize = (N + blockSize - 1) / blockSize;  // = 2

    printf("Launching kernel with gridSize=%d, blockSize=%d\n", gridSize, blockSize);

    // 调用核函数
    test_debug_kernel<<<gridSize, blockSize>>>(d_data, N);

    // 同步等待完成（重要！调试时必须有这一行）
    cudaDeviceSynchronize();

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 拷贝结果回主机
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // 打印结果（验证）
    printf("Results:\n");
    for (int i = 0; i < 10; i++) {  // 只打印前10个
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }
    printf("...\n");

    // 清理
    cudaFree(d_data);
    printf("Kernel execution completed!\n");
}
