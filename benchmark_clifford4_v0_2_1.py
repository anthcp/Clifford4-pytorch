#!/usr/bin/env python3
"""
benchmark_clifford4_v0_2_1.py
===============================================================================
Benchmark script for clifford4_core_v0_2_1.py
===============================================================================
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from clifford4_core_v0_2_1 import Clifford4, python_clifford_blade_coeff


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_callable(fn: Callable[[], object], device: torch.device, warmup: int = 5, repeat: int = 20) -> Dict[str, float]:
    for _ in range(warmup):
        _ = fn()
    _sync_if_cuda(device)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn()
        _sync_if_cuda(device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times, dtype=np.float64)
    return {
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "std_s": float(arr.std()),
    }


def run_benchmarks(batch_sizes: List[int] = [1, 16, 128, 1024, 8192], repeat: int = 20, warmup: int = 5, use_gpu_if_available: bool = True) -> None:
    print("\n=== Running performance benchmarks ===")
    devices = [torch.device("cpu")]
    if use_gpu_if_available and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    dtype = torch.float64

    for device in devices:
        print(f"\n--- Device: {device} | dtype={dtype} ---")
        for batch_size in batch_sizes:
            print(f"\nBatch size = {batch_size}")
            a = Clifford4(torch.randn(batch_size, 16, device=device, dtype=dtype), device=device, dtype=dtype)
            b = Clifford4(torch.randn(batch_size, 16, device=device, dtype=dtype), device=device, dtype=dtype)
            v = Clifford4.vector(torch.randn(batch_size, 4, device=device, dtype=dtype), device=device, dtype=dtype)
            B_simple = -Clifford4.basis_blade("e12", device=device, dtype=dtype)
            R_simple = B_simple.exp_simple_bivector(t=1.2345)
            R_batch = Clifford4(R_simple.data.unsqueeze(0).repeat(batch_size, 1), device=device, dtype=dtype)
            B_general = Clifford4.random_bivector(batch_shape=(batch_size,), device=device, dtype=dtype, seed=123)

            benchmarks = {
                "gp": lambda: a.gp(b),
                "wedge": lambda: a.wedge(b),
                "inner": lambda: a.inner(b),
                "left_contraction": lambda: a.left_contraction(b),
                "sandwich_simple": lambda: R_batch.sandwich(v),
                "normalize_rotor": lambda: R_batch.normalize_rotor(),
            }
            if batch_size <= 1024:
                benchmarks["exp_simple_bivector"] = lambda: B_simple.exp_simple_bivector(t=0.7)
                benchmarks["exp_bivector_general_first"] = lambda: Clifford4(B_general.data[0], device=device, dtype=dtype).exp_bivector_general(t=0.7)
                benchmarks["inverse_general_first"] = lambda: (Clifford4(B_general.data[0], device=device, dtype=dtype) + Clifford4.scalar(1.0, device=device, dtype=dtype)).inverse_general()

            for name, fn in benchmarks.items():
                stats = _time_callable(fn, device=device, warmup=warmup, repeat=repeat)
                print(f"{name:24s} median={stats['median_s']*1e3:9.3f} ms  mean={stats['mean_s']*1e3:9.3f} ms  min={stats['min_s']*1e3:9.3f} ms")

    print("\n=== Benchmarks complete ===")


def run_gp_throughput_benchmark(batch_sizes: List[int] = [16, 128, 1024, 8192, 32768], repeat: int = 20, warmup: int = 5, use_gpu_if_available: bool = True) -> None:
    print("\n=== Running GP throughput benchmark ===")
    devices = [torch.device("cpu")]
    if use_gpu_if_available and torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    dtype = torch.float64

    for device in devices:
        print(f"\n--- Device: {device} ---")
        for batch_size in batch_sizes:
            a = Clifford4(torch.randn(batch_size, 16, device=device, dtype=dtype), device=device, dtype=dtype)
            b = Clifford4(torch.randn(batch_size, 16, device=device, dtype=dtype), device=device, dtype=dtype)
            stats = _time_callable(lambda: a.gp(b), device=device, warmup=warmup, repeat=repeat)
            median = stats["median_s"]
            gps_per_sec = batch_size / median
            print(f"batch={batch_size:6d}  median={median*1e3:9.3f} ms  throughput={gps_per_sec:12.1f} products/s")

    print("\n=== GP throughput benchmark complete ===")


def run_python_clifford_benchmarks(repeat: int = 50, warmup: int = 10, batch_sizes: List[int] = [1, 16, 128, 1024]) -> None:
    print("\n=== Running Clifford4 vs python-clifford benchmarks ===")
    try:
        from clifford import Cl
    except ImportError:
        print("python-clifford not installed — skipping comparison benchmarks")
        return

    layout, blades = Cl(4, 0)
    _ = layout
    device = torch.device("cpu")
    dtype = torch.float64
    rng = np.random.default_rng(123)

    def rand_mv_torch() -> Clifford4:
        return Clifford4(torch.tensor(rng.standard_normal(16), dtype=dtype, device=device), device=device, dtype=dtype)

    def rand_vec_torch() -> Clifford4:
        return Clifford4.vector(torch.tensor(rng.standard_normal(4), dtype=dtype, device=device), device=device, dtype=dtype)

    def rand_biv_torch() -> Clifford4:
        data = torch.zeros(16, dtype=dtype, device=device)
        data[5:11] = torch.tensor(rng.standard_normal(6), dtype=dtype, device=device)
        return Clifford4(data, device=device, dtype=dtype)

    def to_cl(mv_t: Clifford4):
        return mv_t.to_python_clifford(blades)

    a_t = rand_mv_torch()
    b_t = rand_mv_torch()
    v_t = rand_vec_torch()
    B_simple_t = -Clifford4.basis_blade("e12", device=device, dtype=dtype)
    R_t = B_simple_t.exp_simple_bivector(t=0.7)

    a_c = to_cl(a_t)
    b_c = to_cl(b_t)
    v_c = to_cl(v_t)
    B_simple_c = to_cl(B_simple_t)

    B2 = float((B_simple_c * B_simple_c).value[0])
    mag = math.sqrt(max(0.0, -B2))
    if mag > 1e-12:
        R_c = math.cos(0.5 * 0.7 * mag) + (math.sin(0.5 * 0.7 * mag) / mag) * B_simple_c
    else:
        R_c = 1.0 + 0.5 * 0.7 * B_simple_c

    single_cases = {
        "gp_clifford4": lambda: a_t.gp(b_t),
        "gp_python_clifford": lambda: a_c * b_c,
        "wedge_clifford4": lambda: a_t.wedge(b_t),
        "wedge_python_clifford": lambda: a_c ^ b_c,
        "reverse_clifford4": lambda: a_t.reverse(),
        "reverse_python_clifford": lambda: ~a_c,
        "sandwich_clifford4": lambda: R_t.sandwich(v_t),
        "sandwich_python_clifford": lambda: R_c * v_c * (~R_c),
        "exp_simple_clifford4": lambda: B_simple_t.exp_simple_bivector(t=0.7),
        "exp_simple_python_clifford": lambda: math.cos(0.5 * 0.7 * mag) + (math.sin(0.5 * 0.7 * mag) / mag) * B_simple_c if mag > 1e-12 else 1.0 + 0.5 * 0.7 * B_simple_c,
    }

    print("\n--- Single multivector latency (CPU) ---")
    for name, fn in single_cases.items():
        stats = _time_callable(fn, device=device, warmup=warmup, repeat=repeat)
        print(f"{name:28s} median={stats['median_s']*1e6:10.3f} us  mean={stats['mean_s']*1e6:10.3f} us")

    print("\n--- Batched throughput: Clifford4 batch vs python-clifford loop (CPU) ---")
    for batch_size in batch_sizes:
        a_batch = Clifford4(torch.tensor(rng.standard_normal((batch_size, 16)), dtype=dtype, device=device), device=device, dtype=dtype)
        b_batch = Clifford4(torch.tensor(rng.standard_normal((batch_size, 16)), dtype=dtype, device=device), device=device, dtype=dtype)
        a_list_cl = [to_cl(Clifford4(a_batch.data[i], device=device, dtype=dtype)) for i in range(batch_size)]
        b_list_cl = [to_cl(Clifford4(b_batch.data[i], device=device, dtype=dtype)) for i in range(batch_size)]

        stats_torch = _time_callable(lambda: a_batch.gp(b_batch), device=device, warmup=warmup, repeat=repeat)
        stats_cl = _time_callable(lambda: [x * y for x, y in zip(a_list_cl, b_list_cl)], device=device, warmup=max(1, warmup // 2), repeat=repeat)

        throughput_torch = batch_size / stats_torch["median_s"]
        throughput_cl = batch_size / stats_cl["median_s"]
        print(f"batch={batch_size:5d}  Clifford4 median={stats_torch['median_s']*1e3:9.3f} ms  python-clifford median={stats_cl['median_s']*1e3:9.3f} ms  speedup={throughput_torch/throughput_cl:9.2f}x")

    print("\n--- General bivector exponential: Clifford4 vs python-clifford matrix-exp reference ---")

    def coeff_array_cl(mv_cl) -> np.ndarray:
        arr = np.zeros(16, dtype=np.float64)
        if np.isscalar(mv_cl):
            arr[0] = float(mv_cl)
            return arr
        arr[0] = float(mv_cl.value[0])
        for i, blade in enumerate(Clifford4.BASIS[1:], start=1):
            arr[i] = python_clifford_blade_coeff(mv_cl, blade.name, blades)
        return arr

    def left_mul_matrix_cl(mv_cl) -> np.ndarray:
        cols = []
        for name in Clifford4.BASIS_NAMES:
            basis = 1.0 if name == "1" else blades[name]
            cols.append(coeff_array_cl(mv_cl * basis))
        return np.stack(cols, axis=1)

    def matrix_exp_np(A: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eig(A)
        Vinv = np.linalg.inv(vecs)
        expD = np.diag(np.exp(vals))
        out = vecs @ expD @ Vinv
        out = np.real_if_close(out, tol=1000)
        return np.asarray(out, dtype=np.float64)

    Bgen_t = rand_biv_torch()
    Bgen_c = to_cl(Bgen_t)

    stats_torch = _time_callable(lambda: Bgen_t.exp_bivector_general(t=0.7), device=device, warmup=warmup, repeat=repeat)
    stats_cl = _time_callable(lambda: matrix_exp_np(0.5 * 0.7 * left_mul_matrix_cl(Bgen_c))[:, 0], device=device, warmup=max(1, warmup // 2), repeat=repeat)

    print(f"exp_bivector_general Clifford4 median={stats_torch['median_s']*1e3:9.3f} ms  python-clifford-ref median={stats_cl['median_s']*1e3:9.3f} ms")
    print("\n=== Clifford4 vs python-clifford benchmarks complete ===")


if __name__ == "__main__":
    run_benchmarks(batch_sizes=[1, 16, 128, 1024, 8192], repeat=20, warmup=5, use_gpu_if_available=True)
    run_gp_throughput_benchmark(batch_sizes=[16, 128, 1024, 8192, 32768], repeat=20, warmup=5, use_gpu_if_available=True)
    run_python_clifford_benchmarks(repeat=50, warmup=10, batch_sizes=[1, 16, 128, 1024])
