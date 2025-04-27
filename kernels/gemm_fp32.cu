#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLK  = 16;           // 16×16 tile  ⇒ 256 threads / CTA
constexpr int WARP = 32;
constexpr unsigned FULL = 0xffffffff;

torch::Tensor gemm_fp32(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_fp32", &gemm_fp32, 
          "Tiny GEMM 16x16 (row-major, fp32)");
}

/* ─── Tiny GEMM ────────────────────────────────────────────────
 *  C = A · B
 *  A: [M, K]   row-major
 *  B: [K, N]   row-major
 *  C: [M, N]   row-major
 */
__global__ void gemm_tiled_fp32_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float*       __restrict__ C,
                                       int  M, int  N, int  K)
{
    const int row = blockIdx.y * BLK + threadIdx.y;
    const int col = blockIdx.x * BLK + threadIdx.x;

    __shared__ float As[BLK][BLK];
    __shared__ float Bs[BLK][BLK];

    float sum = 0.f;

    for (int k0 = 0; k0 < K; k0 += BLK) {
        if (row < M && k0 + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[(size_t)row * K + (k0 + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.f;

        if (col < N && k0 + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(size_t)(k0 + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLK; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[(size_t)row * N + col] = sum;
}


torch::Tensor gemm_fp32(torch::Tensor A,
                        torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(A.dtype()==torch::kFloat32 && B.dtype()==torch::kFloat32,
                "fp32 only");
    TORCH_CHECK(A.dim()==2 && B.dim()==2, "expect A[M,K], B[K,N]");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    TORCH_CHECK(B.size(0)==K, "shape mismatch: A[M,K]·B[K,N]");

    auto C = torch::empty({M, N}, A.options());

    dim3 block(BLK, BLK);  // 16×16 = 256 threads
    dim3 grid((N+BLK-1)/BLK,
              (M+BLK-1)/BLK);

    gemm_tiled_fp32_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(N),
        static_cast<int>(K));

    return C;
}