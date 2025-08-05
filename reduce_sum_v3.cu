#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define WARP_SIZE 32

template <typename T>
__global__ void reduce_sum(const T *d_in, T *d_out, size_t n) {
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // warp-level reduction
    // 一个warp内，首线程负责累加这个warp内的所有元素
    if (tid == 0) {
        T block_sum = 0;
        for (size_t i = 0; i < blockDim.x; i ++) {
            for (size_t j = idx + i; j < n; j += blockDim.x * gridDim.x) {
                block_sum += d_in[j];
            }
        }
        // warp间使用原子操作将结果写入输出，原子操作次数为 n/WARP_SIZE
        atomicAdd(d_out, block_sum);
    }
}

template <typename T>
T reduce_sum_cpu(const T *h_in, size_t n) {
    T sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += h_in[i];
    }
    return sum;
}

int main() {
    size_t n = 1000000; // Size of the array
    float *d_in, *d_out;
    float *h_in = (float *)malloc(n * sizeof(float));
    float h_out;

    // Initialize input data
    for (size_t i = 0; i < n; i ++) {
        h_in[i] = static_cast<float>(i);
    }

    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMalloc((void**)&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute sum
    size_t blockSize = 256;
    size_t numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    reduce_sum<float><<<numBlocks, blockSize>>>(d_in, d_out, n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    printf("Kernel execution time: %f ms\n", ker_time);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum: %f\n", h_out);

    float cpu_time = 0;
    cudaEventRecord(start,0);
    float sum_ref = reduce_sum_cpu(h_in, n);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_time, start, stop);
    printf("CPU execution time: %f ms\n", cpu_time);
    printf("CPU Sum: %f\n", sum_ref);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}

/*
Kernel execution time: 1.376064 ms
Sum: 499999932416.000000
CPU execution time: 2.428352 ms
CPU Sum: 499940360192.000000
*/