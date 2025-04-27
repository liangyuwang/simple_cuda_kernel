#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP       = 32;
constexpr int BLOCK_SIZE = 256;
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

torch::Tensor gevm_fp32(torch::Tensor x,
                        torch::Tensor A)
{
    TORCH_CHECK(A.is_cuda() && x.is_cuda(), "need CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32,
                "fp32 only");

    int M = A.size(0), N = A.size(1);
    TORCH_CHECK(x.numel() == M, "shape mismatch: x should have M elements");

    auto y = torch::empty({N}, x.options());

    dim3 grid(N);
    dim3 block(BLOCK_SIZE);
    size_t shmem = (BLOCK_SIZE / WARP) * sizeof(float);

    gevm_fp32_kernel<<<grid, block, shmem>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N);

    return y;
}
