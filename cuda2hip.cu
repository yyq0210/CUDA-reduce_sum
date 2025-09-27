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