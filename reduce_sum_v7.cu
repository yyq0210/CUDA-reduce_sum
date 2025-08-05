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

// First pass: Reduce within blocks
template <typename T>
__global__ void reduce_warp_shuffle_first_pass_kernel(T *intermediate, const T *input, size_t n) {
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    T sum = 0;
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x){
        sum += input[i];
    }
    T warp_sum = warp_reduce(sum);
    if (tid % 32 == 0) {
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();
    if (tid < 32) {
        T block_sum = (tid < (blockDim.x + 31) / 32) ? smem[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        if (tid == 0) {
            intermediate[blockIdx.x] = block_sum;
        }
    }
}

// Second pass: Reduce block results
template <typename T>
__global__ void reduce_warp_shuffle_second_pass_kernel(T *output, const T *intermediate, size_t n){
    extern __shared__ T smem[];
    size_t tid = threadIdx.x;
    T sum = 0;
    for (size_t i = tid; i < n; i += blockDim.x) {
        sum += intermediate[i];
    }
    T warp_sum = warp_reduce(sum);
    if (tid % 32 == 0) {
        smem[tid / 32] = warp_sum;
    }
    __syncthreads();
    if (tid < 32) {
        T block_sum = (tid < (blockDim.x + 31) / 32) ? smem[tid] : T(0);
        block_sum = warp_reduce(block_sum);
        if (tid == 0) {
            *output += block_sum;
        }
    }
}

template <typename T>
void reduce_sum(const T *d_in, T *d_out, size_t n, const dim3 &grid, const dim3 &block) {
    // Allocate intermediate buffer for block results
    T *d_intermediate;
    cudaMalloc(&d_intermediate, grid.x * sizeof(T));
    // First pass: reduce within blocks
    size_t smem_size1 = ((block.x + 31) / 32) * sizeof(T);
    reduce_warp_shuffle_first_pass_kernel<<<grid, block, smem_size1>>>(d_intermediate, d_in,n);

    // CUDA_CHECK(cudaGetLastError());
    // Second pass: reduce block results
    dim3 grid2(1);
    dim3 block2(min(grid.x, block.x));
    size_t smem_size2 = ((block2.x + 31) / 32) * sizeof(T);
    reduce_warp_shuffle_second_pass_kernel<<<grid2, block2, smem_size2>>>(d_out, d_intermediate, grid.x);
    // CUDA_CHECK(cudaGetLastError());
    cudaFree(d_intermediate);
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

    size_t numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    dim3 grid(numBlocks);
    dim3 block(BLOCK_SIZE);
    // Launch the kernel with dynamic shared memory
    reduce_sum(d_in, d_out, n, grid, block);

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
Kernel execution time: 1.342176 ms
Sum: 499999473664.000000
CPU execution time: 2.300352 ms
CPU Sum: 499940360192.000000
*/