import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warnings
import torch, math, time
from api.pytorch.batched_gevm import batched_gevm_fp32 as batched_gevm

# ───────────────────────────────── accuracy ──────────────────────────────────
def check_accuracy(B: int = 12,
                   M: int = 4096,
                   N: int = 4096,
                   device: str = "cuda",
                   dtype  = torch.float32,
                   tol: float = 1e-4):
    torch.manual_seed(0)
    x = torch.randn(B, M,    device=device, dtype=dtype)
    A = torch.randn(B, M, N, device=device, dtype=dtype)

    # ------- gevm -------
    y_ref  = torch.matmul(x.unsqueeze(1), A).squeeze(1)
    y_impl = batched_gevm(x, A)
    abs_err  = (y_ref - y_impl).abs().max().item()
    rel_err  = abs_err / y_ref.abs().max().item()

    print(f"max error |Δ|={abs_err:.3e},   rel error={rel_err:.3e}")

    if not (abs_err < tol and rel_err < tol):
        warnings.warn(f"accuracy check failed: "
                    f"abs={abs_err:.3e}, rel={rel_err:.3e}", RuntimeWarning)

    return {
        "gevm_abs_err":  abs_err,
        "gevm_rel_err":  rel_err,
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

def benchmark_speed(B: int = 12,
                    M: int = 8192,
                    N: int = 8192,
                    device: str = "cuda",
                    dtype  = torch.float32,
                    warmup: int = 10,
                    repeat: int = 100):
    torch.manual_seed(0)
    x = torch.randn(B, M,    device=device, dtype=dtype)
    A = torch.randn(B, M, N, device=device, dtype=dtype)

    # gevm ----------------------------------------------------------
    t_custom = _timeitCUDA(lambda: batched_gevm(x, A),
                           warmup, repeat)
    t_torch  = _timeitCUDA(lambda: torch.matmul(x.unsqueeze(1), A).squeeze(1),
                           warmup, repeat)

    # FLOPs = 2*M*N
    flops = 2.0 * M * N
    gflops_gevm = flops / (t_custom  * 1e-3) / 1e9

    print(f"{t_custom:6.3f} ms  vs torch {t_torch:6.3f} ms"
          f"  → speedup {t_torch/t_custom:5.2f}x,  {gflops_gevm:6.1f} GFLOPS")

    return {
        "gevm_ms":  t_custom,
        "gevm_ms_torch": t_torch,
        "gevm_speedup": t_torch / t_custom,
        "gevm_gflops":  gflops_gevm,
    }

# ────────────────────────────── quick self-test ─────────────────────────────
if __name__ == "__main__":
    shape_list = [
        # ── square / power-of-two ─────────────────────────────
        (32,  256,    256),
        (64,  512,    512),
        (32,  1024,   1024),
        (16,  2048,   2048),
        (8,   4096,   4096),
        (4,   8192,   8192),
        (2,   16384,  16384),

        # ── tall-skinny (M ≫ N) ─────────────────────────────
        (64,  4096,   256),
        (32,  8192,   512),
        (16,  16384,  1024),
        (8,   16384,  4096),
        (64,  2048,   64),
        (32,  4096,   128),
        (16,  8192,   256),

        # ── short-wide (M ≪ N) ──────────────────────────────
        (64,  256,    4096),
        (32,  512,    8192),
        (16,  1024,   16384),
        (8,   4096,   16384),
        (64,  128,    2048),
        (32,  128,    4096),
        (16,  256,    8192),

        # ── rectangular / "real-world" multiples of 128/256 ─
        (64,  6144,   7680),    # typical ViT scale
        (64,  7680,   6144),
        (32,  7168,   10240),   # BERT-style block dims
        (32,  12288,  4096),    # GPT-like width
        (64,  4096,   5120),
        (32,  5120,   4096),
        (16,  8192,   7168),

        # ── non-power-of-two / prime-ish ─────────────────────
        (64,  3000,   3000),
        (32,  5000,   8191),    # 8191 is prime
        (16,  1234,   5678),
        (64,  4093,   4099),    # both prime-ish

        # ── very small matrices (micro-batches) ──────────────
        (128, 32,    32),
        (128, 64,    64),
        (64,  128,   64),
        (64,  64,    128),
        (32,  128,   256),
        (32,  256,   128),

        # ── very large batches ───────────────────────────────
        (256, 128,   128),
        (512, 64,    128),
        (512, 64,    64),
        (1024, 32,   32),
    ]

    for i, shape in enumerate(shape_list):
        print(f"[{i}], test shape: {shape}")
        check_accuracy(B=shape[0], M=shape[1], N=shape[2])
        benchmark_speed(B=shape[0], M=shape[1], N=shape[2])
