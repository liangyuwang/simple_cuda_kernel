# Simple CUDA Kernels

A collection of ultra-simple yet high-performance CUDA kernels.

This repository provides minimal (~20 lines) implementations of essential CUDA kernels, achieving reasonable performance while maintaining maximum simplicity.
Perfect for learning, experimenting, or building lightweight custom CUDA extensions.

## Currently Implemented

| Kernel              | Description                           | PyTorch Equivalent | Example API Usage | Performance vs PyTorch | 
|:--------------------|:--------------------------------------|:--------------------|:--------------|:------------------------|
| 🔹 [**GEMV**](kernels/gemv_fp32.cu)          | General Matrix-Vector Multiplication  | `torch.mv(A, x)`   | [`gemv(A, x)`](api/pytorch/gemv.py) | ⚡ ~72–141% |
| 🔹 [**GEVM**](kernels/gevm_fp32.cu)          | General Vector-Matrix Multiplication  | `torch.matmul(x, A)` | [`gevm(x, A)`](api/pytorch/gevm.py) | ⚡ ~46–80% |
| 🔹 [**GEMM**](kernels/gemm_fp32.cu)          | General Matrix-Matrix Multiplication  | `torch.mm(A, B)`   | [`gemm(A, B)`](api/pytorch/gemm.py) | ⚡ ~14–25% |
| 🔹 [**Batched GEMV**](kernels/batched_gemv_fp32.cu)  | Batched Matrix-Vector Multiplication  | `torch.matmul(A, x)` | [`batched_gemv(A, x)`](api/pytorch/batched_gemv.py) | ⚡ ~77–188% |
| 🔹 [**Batched GEVM**](kernels/batched_gevm_fp32.cu)  | Batched Vector-Matrix Multiplication  | `torch.matmul(x, A)` | [`batched_gevm(x, A)`](api/pytorch/batched_gevm.py) | ⚡ ~80–98% |
| 🔹 [**Batched GEMM**](kernels/batched_gemm_fp32.cu)  | Batched Matrix-Matrix Multiplication  | `torch.bmm(A, B)`   | [`batched_gemm(A, B)`](api/pytorch/batched_gemm.py) | ⚡ ~14–18% |
