import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.cpp_extension import load


# Compiling and loading the C++ module
module = load(
    name='gemm_fp32',
    sources=['kernels/gemm_fp32.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=["-O2"],
    verbose=True
)

gemm_fp32 = module.gemm_fp32