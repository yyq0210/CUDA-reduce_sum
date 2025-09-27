#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

template <typename T>
__device__ T warp_reduce(T var) {
#pragma unroll
    for (int stride = WARP_SIZE / 2; stride > 0; stride >>= 1) {
        var += __shfl_down_sync(0xFFFFFFFF, var, stride);
    }
    return var;
}

template <typename T>
__global__ void reduce_sum(const T *d_in, T *d_out, size_t n) {
    extern __shared__ T shared_data[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // warp-level reduction
    // 一个warp内，首线程负责累加这个warp内的所有元素
    T sum = 0;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum += d_in[i];
    }

    T warp_sum = warp_reduce(sum);

    if (tid % WARP_SIZE == 0) {
        shared_data[tid / WARP_SIZE] = warp_sum;
    }
    __syncthreads();

    if (tid < WARP_SIZE) {
        T block_sum = (tid < (blockDim.x + 31) / WARP_SIZE) ? shared_data[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        printf("%f\n", block_sum); 
        if (tid == 0) {
            atomicAdd(d_out, block_sum);
        }
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
    size_t n = 10000; // Size of the array
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

    size_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    size_t shared_memory_size = (BLOCK_SIZE / 32 + 1) * sizeof(float);
    // Launch the kernel with dynamic shared memory
    reduce_sum<float><<<numBlocks, BLOCK_SIZE, shared_memory_size>>>(d_in, d_out, n);

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
Kernel execution time: 1.049696 ms
Sum: 499999965184.000000
CPU execution time: 2.285120 ms
CPU Sum: 499940360192.000000
*/