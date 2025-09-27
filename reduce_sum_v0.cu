#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cub/block/block_reduce.cuh>
#include <device_launch_parameters.h>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

#define WARP_SIZE 32
#define BLOCK_SIZE 256

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}


template <typename T>
__global__ void d_reduce_sum(const T* d_in, uint N, T* d_out) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ T reduce_smem[NUM_WARPS];
    // keep the data in register is enough for warp operaion.
    T sum = (idx < N) ? d_in[idx] : T(0);
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0) {
        reduce_smem[warp] = sum;
    }
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0) {
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    }
    if (tid == 0) {
        atomicAdd(d_out, sum);
    }
}

// Block All Reduce Sum + float4
// grid(N/256), block(256/4)
// a: Nx1, y=sum(a)
template <const int NUM_THREADS = 256 / 4>
__global__ void d_reduce_sum_vec4(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg_a = FLOAT4(a[idx]);
    // keep the data in register is enough for warp operaion.
    float sum = (idx < N) ? (reg_a.x + reg_a.y + reg_a.z + reg_a.w) : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    // perform warp sync reduce.
    sum = warp_reduce_sum<WARP_SIZE>(sum);
    // warp leaders store the data to shared memory.
    if (lane == 0)
        reduce_smem[warp] = sum;
    __syncthreads(); // make sure the data is in shared memory.
    // the first warp compute the final sum.
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if (warp == 0)
        sum = warp_reduce_sum<NUM_WARPS>(sum);
    if (tid == 0)
        atomicAdd(y, sum);
}

template <typename T>
T reduce_sum_cpu(const T *h_in, size_t n) {
    T sum = T(0);
    for (size_t i = 0; i < n; i++) {
        sum += h_in[i];
    }
    return sum;
}
int main() {
    size_t n = 10000; // Size of the array
    float *d_in, *d_out, *d2_out;
    float *h_in = (float *)malloc(n * sizeof(float));
    float h_out;

    // Initialize input data
    for (size_t i = 0; i < n; i++) {
        h_in[i] = static_cast<float>(i);
    }

    cudaMalloc((void**)&d_in, n * sizeof(float));
    cudaMalloc((void**)&d_out, n * sizeof(float));
    cudaMalloc((void**)&d2_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute sum
    size_t blockSize = 256;
    size_t numBlocks = (n + blockSize - 1) / blockSize;

    cudaEvent_t start,stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    

    float gpu_time = 0;
    cudaEventRecord(start,0);
    d_reduce_sum_vec4<<<numBlocks, BLOCK_SIZE>>>(d_in, d2_out, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("Kernel 2 execution time: %f ms\n", gpu_time);
    cudaMemcpy(&h_out, d2_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU_Vec Sum: %f\n", h_out);

    d_reduce_sum<float><<<numBlocks, BLOCK_SIZE>>>(d_in, n, d_out);
    cudaDeviceSynchronize();
    // gridsum(d_out, numBlocks);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop);// must float ker_time
    printf("Kernel execution time: %f ms\n", ker_time);
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("GPU Sum: %f\n", h_out);

    float cpu_time = 0;
    cudaEventRecord(start,0);
    float sum_ref = reduce_sum_cpu<float>(h_in, n);
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
Kernel execution time: 4.140032 ms
Sum: 499947667456.000000
CPU execution time: 2.529216 ms
CPU Sum: 499940360192.000000
*/