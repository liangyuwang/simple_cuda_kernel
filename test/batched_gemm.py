import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import torch, math, time
from api.pytorch.batched_gemm import batched_gemm_fp32 as batched_gemm

# ───────────────────────────────── accuracy ──────────────────────────────────
def check_accuracy(BS: int = 12,
                   M: int = 4096,
                   K: int = 4096,
                   N: int = 4096,
                   device: str = "cuda",
                   dtype  = torch.float32,
                   tol: float = 1e-4):
    torch.manual_seed(0)
    A = torch.randn(BS, M, K, device=device, dtype=dtype)
    B = torch.randn(BS, K, N, device=device, dtype=dtype)

    # ------- batched gemm -------
    y_ref  = torch.matmul(A, B)
    y_impl = batched_gemm(A, B)
    abs_err  = (y_ref - y_impl).abs().max().item()
    rel_err  = abs_err / y_ref.abs().max().item()

    print(f"max error |Δ|={abs_err:.3e},   rel error={rel_err:.3e}")

    if not (abs_err < tol and rel_err < tol):
        warnings.warn(f"accuracy check failed: "
                    f"abs={abs_err:.3e}, rel={rel_err:.3e}", RuntimeWarning)

    return {
        "batched_gemm_abs_err":  abs_err,
        "batched_gemm_rel_err":  rel_err,
    }

# ─────────────────────────────────  speed  ──────────────────────────────────
def _timeitCUDA(fn, warmup=10, repeat=100):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat     # ms

def benchmark_speed(BS: int = 12,
                    M: int = 8192,
                    K: int = 8192,
                    N: int = 8192,
                    device: str = "cuda",
                    dtype  = torch.float32,
                    warmup: int = 10,
                    repeat: int = 100):
    torch.manual_seed(0)
    A = torch.randn(BS, M, K, device=device, dtype=dtype)
    B = torch.randn(BS, K, N, device=device, dtype=dtype)

    # batched gemm ----------------------------------------------------------
    t_custom = _timeitCUDA(lambda: batched_gemm(A, B),
                           warmup, repeat)
    t_torch  = _timeitCUDA(lambda: torch.matmul(A, B),
                           warmup, repeat)

    # FLOPs = 2*M*N
    flops = 2.0 * M * K * N
    gflops_batched_gemm = flops / (t_custom  * 1e-3) / 1e9

    print(f"{t_custom:6.3f} ms  vs torch {t_torch:6.3f} ms"
          f"  → speedup {t_torch/t_custom:5.2f}x,  {gflops_batched_gemm:6.1f} GFLOPS")

    return {
        "batched_gemm_ms":  t_custom,
        "batched_gemm_ms_torch": t_torch,
        "batched_gemm_speedup": t_torch / t_custom,
        "batched_gemm_gflops":  gflops_batched_gemm,
    }

# ────────────────────────────── quick self-test ─────────────────────────────
if __name__ == "__main__":
    shape_list = [
        # ── square / power-of-two ─────────────────────────────
        (32,  256,   256,   256),
        (64,  512,   512,   512),
        (32,  1024,  1024,  1024),
        (16,  2048,  2048,  2048),
        (8,   4096,  4096,  4096),
        (4,   8192,  8192,  8192),

        # ── tall-skinny (M ≫ N) ─────────────────────────────
        (64,  4096,  512,   256),
        (32,  8192,  1024,  512),
        (16,  16384, 2048,  1024),

        # ── short-wide (M ≪ N) ──────────────────────────────
        (64,  256,   512,   4096),
        (32,  512,   1024,  8192),
        (16,  1024,  2048,  16384),

        # ── fat matrices (K very large) ─────────────────────
        (64,  512,   4096,  512),
        (32,  1024,  8192,  1024),
        (16,  2048,  16384, 2048),

        # ── rectangular / "real-world" multiples of 128/256 ─
        (64,  4096, 6144, 7680),     # ViT scale
        (64,  4096, 7680, 6144),
        (32,  7168, 4096, 10240),    # BERT-style block dims
        (32,  12288, 4096, 4096),    # GPT-like width
        (64,  4096, 4096, 5120),
        (32,  5120, 4096, 4096),
        (16,  8192, 7168, 8192),

        # ── non-power-of-two / prime-ish ─────────────────────
        (64,  3000, 3000, 3000),
        (32,  4093, 4099, 4093),    # primes close together
        (32,  5000, 4096, 8191),
        (16,  4096, 5000, 8191),

        # ── very small matrices (micro batched GEMM) ─────────
        (128, 32,   64,   32),
        (128, 64,   32,   64),
        (64,  128,  64,  128),
        (64,  64,  128,   64),
        (32,  128, 128,  256),
        (32,  256, 128,  128),
        (256, 64,   64,   64),

        # ── very large batches ───────────────────────────────
        (512, 128,  128,  128),
        (512, 64,   128,   64),
        (1024, 64,  64,    64),
        (2048, 32,  64,    32),
    ]

    for i, shape in enumerate(shape_list):
        print(f"[{i}], test shape: {shape}")
        check_accuracy(BS=shape[0], M=shape[1], K=shape[2], N=shape[3])
        benchmark_speed(BS=shape[0], M=shape[1], K=shape[2], N=shape[3])
