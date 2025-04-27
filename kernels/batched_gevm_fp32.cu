#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP          = 32;
constexpr int WARPS_PER_BLK = 8;
constexpr unsigned FULL     = 0xffffffff;

torch::Tensor batched_gevm_fp32(torch::Tensor x, torch::Tensor A);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_gevm_fp32", &batched_gevm_fp32,
          "Batched GEVM (x^T·A) in FP32 (row-major, tiled)");
}

/*  x:  [B, M]      (row-major, contiguous)
 *  A:  [B, M, N]   (row-major, contiguous)
 *  y:  [B, N]      (row-major, contiguous)
 */
__global__ void batched_gevm_tiled_fp32_kernel(
        const float* __restrict__ x,
        const float* __restrict__ A,
        float*       __restrict__ y,
        int M, int N)
{
    const int lane    = threadIdx.x;
    const int warpId  = threadIdx.y;
    const int colBase = blockIdx.x * WARP;
    const int col     = colBase + lane;
    const int b       = blockIdx.y;

    const float* x_b = x + (size_t)b * M;
    const float* A_b = A + (size_t)b * M * N;
          float* y_b = y + (size_t)b * N;

    float sum = 0.f;

    for (int row = warpId; row < M; row += WARPS_PER_BLK) {
        float xv = __shfl_sync(FULL, (lane == 0 ? x_b[row] : 0.f), 0);

        const float* A_row = A_b + (size_t)row * N;

        float a = (col < N) ? A_row[col] : 0.f;
        sum += xv * a;
    }

    __shared__ float sm[WARPS_PER_BLK][WARP];
    sm[warpId][lane] = sum;
    __syncthreads();

    if (warpId == 0 && col < N) {
        float colSum = 0.f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_BLK; ++w)
            colSum += sm[w][lane];
        y_b[col] = colSum;
    }
}

torch::Tensor batched_gevm_fp32(torch::Tensor x,
                                torch::Tensor A)
{
    TORCH_CHECK(x.is_cuda() && A.is_cuda(),       "tensors must be on CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 &&
                A.dtype() == torch::kFloat32,     "FP32 only");
    TORCH_CHECK(x.dim() == 2 && A.dim() == 3,
                "expected x:[B,M]  A:[B,M,N]");
    TORCH_CHECK(x.size(0) == A.size(0) &&
                x.size(1) == A.size(1),
                "shape mismatch between x and A");

    const int B = A.size(0);
    const int M = A.size(1);
    const int N = A.size(2);

    auto y = torch::empty({B, N}, x.options());

    dim3 grid((N + WARP - 1) / WARP, B);
    dim3 block(WARP, WARPS_PER_BLK);

    batched_gevm_tiled_fp32_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        A.data_ptr<float>(),
        y.data_ptr<float>(),
        M, N);
    return y;
}