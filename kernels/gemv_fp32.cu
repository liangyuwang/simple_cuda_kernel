#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP       = 32;
constexpr int BLOCK_SIZE = 256;
constexpr unsigned FULL  = 0xffffffff;

torch::Tensor gemv_fp32(torch::Tensor A, torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemv_fp32", &gemv_fp32, "fast GEMV (row-major)");
}

__global__ void gemv_fp32_kernel(const float* __restrict__ A,
                                 const float* __restrict__ x,
                                 float*       __restrict__ y,
                                 int M, int N)
{
    int row = blockIdx.x;
    if (row >= M) return;
    const float* a_row = A + row * N;

    float sum = 0.f;
    for (int col = threadIdx.x; col < N; col += blockDim.x)
        sum += a_row[col] * x[col];

    // warp reduce
    for (int off = WARP/2; off > 0; off >>= 1)
        sum += __shfl_down_sync(FULL, sum, off);

    __shared__ float buf[BLOCK_SIZE/WARP];
    if ((threadIdx.x & (WARP-1)) == 0)
        buf[threadIdx.x / WARP] = sum;
    __syncthreads();

    if (threadIdx.x < WARP) {
        float v = (threadIdx.x < blockDim.x/WARP) ? buf[threadIdx.x] : 0.f;
        for (int off = WARP/2; off > 0; off >>= 1)
            v += __shfl_down_sync(FULL, v, off);
        if (threadIdx.x == 0)
            y[row] = v;
    }
}

torch::Tensor gemv_fp32(torch::Tensor A,
                        torch::Tensor x)
{
    TORCH_CHECK(A.is_cuda() && x.is_cuda(), "need CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && x.dtype() == torch::kFloat32,"fp32 only");

    int M = A.size(0), N = A.size(1);
    TORCH_CHECK(x.numel()==N, "shape mismatch");

    // Create output tensor y
    torch::Tensor y = torch::empty({M}, A.options());

    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    size_t shmem = (BLOCK_SIZE / WARP) * sizeof(float);
    gemv_fp32_kernel<<<grid,block,shmem>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, N);
    
    return y;
}