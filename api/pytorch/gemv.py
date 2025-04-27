import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.cpp_extension import load


# Compiling and loading the C++ module
module = load(
    name='gemv_fp32',
    sources=['kernels/gemv_fp32.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=["-O2"],
    verbose=True
)

gemv_fp32 = module.gemv_fp32