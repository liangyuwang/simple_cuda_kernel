import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.cpp_extension import load


# Compiling and loading the C++ module
module = load(
    name='batched_gevm_fp32',
    sources=['kernels/batched_gevm_fp32.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=["-O2"],
    verbose=True
)

batched_gevm_fp32 = module.batched_gevm_fp32