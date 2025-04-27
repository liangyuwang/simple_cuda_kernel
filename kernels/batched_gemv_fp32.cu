#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP       = 32;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL  = 0xffffffff;

torch::Tensor batched_gemv_fp32(torch::Tensor A, torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gemv_fp32", &batched_gemv_fp32, "fast GEMV (row-major)");
}

/*  A: [B, M, N]  (row-major, contiguous)
 *  x: [B, N]
 *  y: [B, M]     ← output
 */
__global__ void batched_gemv_fp32_kernel(
        const float* __restrict__ A,
        const float* __restrict__ x,
        float*       __restrict__ y,
        int M, int N)
{
    const int row = blockIdx.x;   // 0 … M-1
    const int b   = blockIdx.y;   // batch index

    if (row >= M) return;

    const size_t batch_stride_A = (size_t)M * N;
    const float* A_b = A + (size_t)b * batch_stride_A;
    const float* x_b = x + (size_t)b * N;
          float* y_b = y + (size_t)b * M;

    const float* a_row = A_b + (size_t)row * N;

    float sum = 0.f;
    for (int col = threadIdx.x; col < N; col += BLOCK_SIZE)
        sum += a_row[col] * x_b[col];

    for (int off = WARP >> 1; off; off >>= 1)
        sum += __shfl_down_sync(FULL, sum, off);

    __shared__ float buf[BLOCK_SIZE / WARP];   // 8 × 4B = 32 B
    if ((threadIdx.x & (WARP - 1)) == 0)
        buf[threadIdx.x >> 5] = sum;
    __syncthreads();

    if (threadIdx.x < WARP) {
        float v = (threadIdx.x < BLOCK_SIZE / WARP) ? buf[threadIdx.x] : 0.f;
        for (int off = WARP >> 1; off; off >>= 1)
            v += __shfl_down_sync(FULL, v, off);
        if (threadIdx.x == 0)
            y_b[row] = v;
    }
}

torch::Tensor batched_gemv_fp32(torch::Tensor A,
                        torch::Tensor x)
{
    TORCH_CHECK(A.is_cuda() && x.is_cuda(),        "need CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 &&
                x.dtype() == torch::kFloat32,      "fp32 only");
    TORCH_CHECK(A.dim() == 3 && x.dim() == 2,
                "expect A:[B,M,N], x:[B,N]");

    const int64_t B = A.size(0);
    const int64_t M = A.size(1);
    const int64_t N = A.size(2);
    TORCH_CHECK(x.size(0) == B && x.size(1) == N,
                "shape mismatch between A and x");

    auto y = torch::empty({B, M}, A.options());

    dim3 grid(M, B);
    dim3 block(BLOCK_SIZE);

    batched_gemv_fp32_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(N));

    return y;
}