import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import torch, math, time
from api.pytorch.gemv import gemv_fp32 as gemv

# ───────────────────────────────── accuracy ──────────────────────────────────
def check_accuracy(M: int = 4096,
                   N: int = 4096,
                   device: str = "cuda",
                   dtype  = torch.float32,
                   tol: float = 1e-4):
    torch.manual_seed(0)
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N,     device=device, dtype=dtype)

    # ------- gemv -------
    y_ref  = torch.mv(A, x)
    y_impl = gemv(A, x)
    abs_err  = (y_ref - y_impl).abs().max().item()
    rel_err  = abs_err / y_ref.abs().max().item()

    print(f"max error |Δ|={abs_err:.3e},   rel error={rel_err:.3e}")

    if not (abs_err < tol and rel_err < tol):
        warnings.warn(f"accuracy check failed: "
                    f"abs={abs_err:.3e}, rel={rel_err:.3e}", RuntimeWarning)

    return {
        "gemv_abs_err":  abs_err,
        "gemv_rel_err":  rel_err,
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

def benchmark_speed(M: int = 8192,
                    N: int = 8192,
                    device: str = "cuda",
                    dtype  = torch.float32,
                    warmup: int = 10,
                    repeat: int = 100):
    torch.manual_seed(0)
    A = torch.randn(M, N, device=device, dtype=dtype)
    x = torch.randn(N,     device=device, dtype=dtype)

    # GEMV ----------------------------------------------------------
    t_custom = _timeitCUDA(lambda: gemv(A, x),
                           warmup, repeat)
    t_torch  = _timeitCUDA(lambda: torch.mv(A, x),
                           warmup, repeat)

    # FLOPs = 2*M*N
    flops = 2.0 * M * N
    gflops_gemv = flops / (t_custom  * 1e-3) / 1e9

    print(f"{t_custom:6.3f} ms  vs torch {t_torch:6.3f} ms"
          f"  → speedup {t_torch/t_custom:5.2f}x,  {gflops_gemv:6.1f} GFLOPS")

    return {
        "gemv_ms":  t_custom,
        "gemv_ms_torch": t_torch,
        "gemv_speedup": t_torch / t_custom,
        "gemv_gflops":  gflops_gemv,
    }

# ────────────────────────────── quick self-test ─────────────────────────────
if __name__ == "__main__":
    shape_list = [
        # ── square / power-of-two ─────────────────────────────
        (256,   256),
        (512,   512),
        (1024,  1024),
        (2048,  2048),
        (4096,  4096),
        (8192,  8192),
        (16384, 16384),
        # ── tall-skinny  (M ≫ N) ─────────────────────────────
        (4096,   256),
        (8192,   512),
        (16384, 1024),
        (16384, 4096),
        # ── short-wide  (M ≪ N) ──────────────────────────────
        (256,   4096),
        (512,   8192),
        (1024, 16384),
        (4096, 16384),
        # ── rectangular / “real-world” multiples of 128/256 ─
        (6144,  7680),     # typical ViT scale
        (7680,  6144),
        (7168, 10240),     # BERT-style block dims
        (12288, 4096),     # GPT-like width
        # ── non-power-of-two / prime-ish ─────────────────────
        (3000,  3000),
        (5000,  8191),     # 8191 is prime
    ]

    for i, shape in enumerate(shape_list):
        print(f"[{i}], test shape: {shape}")
        check_accuracy(M=shape[0], N=shape[1])
        benchmark_speed(M=shape[0], N=shape[1])
