#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP       = 32;
constexpr int WARPS_PER_BLK  = 8;
constexpr int BLOCK_SIZE     = WARP * WARPS_PER_BLK;
constexpr unsigned FULL  = 0xffffffff;

torch::Tensor gevm_fp32(torch::Tensor x, torch::Tensor A);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gevm_fp32", &gevm_fp32, "fast GEVM (row-major)");
}

__global__ void gevm_fp32_kernel(const float* __restrict__ x,
                                 const float* __restrict__ A,
                                 float*       __restrict__ y,
                                 int M, int N)
{
    int col = blockIdx.x;
    if (col >= N) return;

    float sum = 0.f;
    for (int row = threadIdx.x; row < M; row += blockDim.x)
        sum += x[row] * A[row * N + col];

    // ── intra-warp reduce ──
    for (int off = WARP / 2; off > 0; off >>= 1)
        sum += __shfl_down_sync(FULL, sum, off);

    // ── block-level reduce ──
    __shared__ float buf[BLOCK_SIZE / WARP];
    if ((threadIdx.x & (WARP - 1)) == 0)
        buf[threadIdx.x / WARP] = sum;
    __syncthreads();

    if (threadIdx.x < WARP) {
        float v = (threadIdx.x < blockDim.x / WARP) ? buf[threadIdx.x] : 0.f;
        for (int off = WARP / 2; off > 0; off >>= 1)
            v += __shfl_down_sync(FULL, v, off);
        if (threadIdx.x == 0)
            y[col] = v;
    }
}

__global__ void gevm_tiled_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ A,
                                       float*       __restrict__ y,
                                       int M, int N)
{
    const int lane   = threadIdx.x;               // 0…31
    const int warpId = threadIdx.y;               // 0…7
    const int colBase= blockIdx.x * WARP;
    const int col    = colBase + lane;

    float sum = 0.f;

    for (int row = warpId; row < M; row += WARPS_PER_BLK)
    {
        float xv = 0.f;
        if (lane == 0)
            xv = x[row];
        xv = __shfl_sync(FULL, xv, 0);

        float a = 0.f;
        if (col < N)
            a = A[row * N + col];

        sum += xv * a;
    }

    __shared__ float sm[WARPS_PER_BLK][WARP];     // 1 KB
    sm[warpId][lane] = sum;
    __syncthreads();

    if (warpId == 0 && col < N)
    {
        float colSum = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLK; ++w)
            colSum += sm[w][lane];
        y[col] = colSum;
    }
}

torch::Tensor gevm_fp32(torch::Tensor x,
                        torch::Tensor A)
{
    TORCH_CHECK(A.is_cuda() && x.is_cuda(), "need CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32,
                "fp32 only");

    int M = A.size(0), N = A.size(1);
    TORCH_CHECK(x.numel() == M, "shape mismatch: x should have M elements");

    auto y = torch::empty({N}, x.options());

    // dim3 grid(N);
    // dim3 block(BLOCK_SIZE);
    // size_t shmem = (BLOCK_SIZE / WARP) * sizeof(float);
    // gevm_fp32_kernel<<<grid, block, shmem>>>(
    //     x.data_ptr<float>(),
    //     A.data_ptr<float>(),
    //     y.data_ptr<float>(),
    //     M, N);

    dim3 grid((N + WARP - 1) / WARP);
    dim3 block(WARP, WARPS_PER_BLK);
    gevm_tiled_fp32_kernel<<<grid, block>>>(
        x.data_ptr<float>(), A.data_ptr<float>(), y.data_ptr<float>(), M, N);

    return y;
}