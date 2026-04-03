#!/usr/bin/env python3
"""
test_clifford4_v0_2_1.py
===============================================================================
Test and cross-validation script for clifford4_core_v0_2_1.py

Includes:
- core unit tests
- batched rotor tests
- python-clifford spot cross-validation
- fuller python-clifford randomized cross-validation

Run:
    python test_clifford4_v0_2_1.py
===============================================================================
"""

from __future__ import annotations

import math
import numpy as np
import torch

from clifford4_core_v0_2_1 import Clifford4, python_clifford_blade_coeff


def run_cl4_unit_tests() -> None:
    print("=== Running Cl(4,0) v0.2.1 unit tests ===")
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10

    one = Clifford4.scalar(1.0, device=device, dtype=dtype)
    e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
    e2 = Clifford4.basis_blade("e2", device=device, dtype=dtype)
    e3 = Clifford4.basis_blade("e3", device=device, dtype=dtype)
    e4 = Clifford4.basis_blade("e4", device=device, dtype=dtype)
    e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)
    e23 = Clifford4.basis_blade("e23", device=device, dtype=dtype)
    e34 = Clifford4.basis_blade("e34", device=device, dtype=dtype)
    e123 = Clifford4.basis_blade("e123", device=device, dtype=dtype)
    I4 = Clifford4.basis_blade("e1234", device=device, dtype=dtype)

    assert Clifford4.BASIS_INDEX["e12"] == 5
    assert Clifford4.BITMASK_TO_INDEX[0b1111] == 15
    print("✓ Explicit basis maps pass")

    full = one + e1 + 2.0 * e12 + 3.0 * I4
    assert abs(full.grade(0).coeff("1").item() - 1.0) < tol
    assert abs(full.grade(1).coeff("e1").item() - 1.0) < tol
    assert abs(full.grade(2).coeff("e12").item() - 2.0) < tol
    assert abs(full.grade(4).coeff("e1234").item() - 3.0) < tol
    print("✓ Grade projections pass")

    assert abs(e1.gp(e1).coeff("1").item() - 1.0) < tol
    ab = e1.gp(e2)
    ba = e2.gp(e1)
    assert abs(ab.coeff("e12").item() - 1.0) < tol
    assert abs(ba.coeff("e12").item() + 1.0) < tol
    assert ab.almost_equal(-ba, atol=tol)
    print("✓ Basis convention passes")

    assert e12.reverse().almost_equal(-e12, atol=tol)
    assert e123.reverse().almost_equal(-e123, atol=tol)
    assert I4.reverse().almost_equal(I4, atol=tol)
    print("✓ Reverse signs pass")

    assert e1.wedge(e2).almost_equal(e1.gp(e2), atol=tol)
    assert abs(e1.inner(e1).coeff("1").item() - 1.0) < tol
    assert abs(e1.left_contraction(e12).coeff("e2").item() - 1.0) < tol
    print("✓ Tensorized wedge/inner/left contraction pass")

    lhs = e1.gp(e2.gp(e3))
    rhs = e1.gp(e2).gp(e3)
    assert lhs.almost_equal(rhs, atol=tol)
    print("✓ Associativity passes")

    R = (-e12).exp_simple_bivector(t=math.pi / 2.0, atol=tol)
    v_rot = R.sandwich(e1)
    assert abs(v_rot.coeff("e2").item() - 1.0) < 1e-10
    assert abs(v_rot.coeff("e1").item()) < 1e-10
    print("✓ Simple bivector rotor passes")

    Bgen = e12 + e34
    Rg = Bgen.exp_bivector_general(t=0.7, atol=tol)
    RRr = Rg.gp(Rg.reverse())
    assert abs(RRr.coeff("1").item() - 1.0) < 1e-10
    print("✓ General bivector rotor normalization passes")

    v = Clifford4.vector(torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=dtype), device=device, dtype=dtype)
    vinv = v.inverse_vector(atol=tol)
    assert abs(v.gp(vinv).coeff("1").item() - 1.0) < 1e-10
    print("✓ Vector inverse passes")

    Rinv = R.inverse_rotor(atol=tol)
    Id = R.gp(Rinv)
    assert abs(Id.coeff("1").item() - 1.0) < 1e-10
    assert torch.max(torch.abs(Id.data[1:])).item() < 1e-10
    print("✓ Rotor inverse passes")

    M = Clifford4.from_coeff_dict({"1": 1.2, "e1": 0.3, "e23": -0.4, "e1234": 0.1}, device=device, dtype=dtype)
    Minv = M.inverse_general(atol=1e-12)
    P = M.gp(Minv)
    assert abs(P.coeff("1").item() - 1.0) < 1e-8
    assert torch.max(torch.abs(P.data[1:])).item() < 1e-8
    print("✓ General inverse passes")

    R2 = (2.7 * R).normalize_rotor()
    chk = R2.gp(R2.reverse())
    assert abs(chk.coeff("1").item() - 1.0) < 1e-10
    print("✓ Rotor normalization helper passes")

    print("=== All v0.2.1 unit tests passed ===")


def run_batched_random_rotor_tests(batch_size: int = 128, seed: int = 7, use_gpu_if_available: bool = True) -> None:
    device = torch.device("cuda") if (use_gpu_if_available and torch.cuda.is_available()) else torch.device("cpu")
    dtype = torch.float64
    tol = 1e-10
    print(f"\n=== Running batched random rotor tests on {device} (batch_size={batch_size}) ===")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    vecs = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=gen)
    angles = torch.randn(batch_size, device=device, dtype=dtype, generator=gen)

    V = Clifford4.vector(vecs, device=device, dtype=dtype)
    in_sq = V.gp(V).coeff("1")

    B = -Clifford4.basis_blade("e12", device=device, dtype=dtype)
    Rfixed = B.exp_simple_bivector(t=1.2345)
    Rfixed_batch = Clifford4(Rfixed.data.unsqueeze(0).repeat(batch_size, 1), device=device, dtype=dtype)
    Vrot = Rfixed_batch.sandwich(V)
    out_sq = Vrot.gp(Vrot).coeff("1")
    assert torch.allclose(in_sq, out_sq, atol=tol, rtol=0.0)
    print("✓ Batched fixed-plane norm preservation passes")

    Bbatch = Clifford4.random_bivector(batch_shape=(batch_size,), device=device, dtype=dtype, seed=seed + 1)
    Rlist = []
    for n in range(batch_size):
        Rn = Clifford4(Bbatch.data[n], device=device, dtype=dtype).exp_bivector_general(t=float(angles[n].item()))
        Rlist.append(Rn.data)
    Rstack = Clifford4(torch.stack(Rlist, dim=0), device=device, dtype=dtype)

    unit_check = Rstack.gp(Rstack.reverse()).coeff("1")
    assert torch.allclose(unit_check, torch.ones_like(unit_check), atol=1e-10, rtol=0.0)
    print("✓ Batched general-bivector rotor normalization passes")

    Vrot2 = Rstack.sandwich(V)
    out_sq2 = Vrot2.gp(Vrot2).coeff("1")
    assert torch.allclose(in_sq, out_sq2, atol=1e-8, rtol=0.0)
    print("✓ Batched general-bivector norm preservation passes")

    print("=== All batched rotor tests passed ===")


def run_python_clifford_cross_validation() -> None:
    print("\n=== Running python-clifford cross-validation ===")
    try:
        from clifford import Cl
    except ImportError:
        print("python-clifford not installed — skipping cross-validation")
        return

    layout, blades = Cl(4, 0)
    _ = layout
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10

    e1_t = Clifford4.basis_blade("e1", device=device, dtype=dtype)
    e2_t = Clifford4.basis_blade("e2", device=device, dtype=dtype)
    e12_t = Clifford4.basis_blade("e12", device=device, dtype=dtype)
    e34_t = Clifford4.basis_blade("e34", device=device, dtype=dtype)

    e1_c = blades["e1"]
    e2_c = blades["e2"]
    e12_c = blades["e12"]
    e34_c = blades["e34"]

    assert abs(python_clifford_blade_coeff(e12_c, "e12", blades) - 1.0) < tol
    print("✓ Coefficient extraction passes")

    gp_t = e1_t.gp(e2_t)
    gp_c = e1_c * e2_c
    assert abs(gp_t.coeff("e12").item() - python_clifford_blade_coeff(gp_c, "e12", blades)) < tol
    print("✓ GP matches python-clifford")

    rev_t = e12_t.reverse().coeff("e12").item()
    rev_c = python_clifford_blade_coeff(~e12_c, "e12", blades)
    assert abs(rev_t - rev_c) < tol
    print("✓ Reverse matches python-clifford")

    ns_t = e12_t + e34_t
    ns_c = e12_c + e34_c
    wedge_t = ns_t.wedge(ns_t).coeff("e1234").item()
    wedge_c = python_clifford_blade_coeff(ns_c ^ ns_c, "e1234", blades)
    assert abs(wedge_t - wedge_c) < tol
    print("✓ Wedge matches python-clifford")

    mv_t = Clifford4.from_coeff_dict({"1": 1.5, "e1": -2.0, "e23": 0.75, "e1234": -0.25}, device=device, dtype=dtype)
    mv_c = mv_t.to_python_clifford(blades)
    mv_back = Clifford4.from_python_clifford(mv_c, blades, device=device, dtype=dtype)
    assert mv_t.almost_equal(mv_back, atol=tol)
    print("✓ Bridge conversion passes")

    print("=== All python-clifford cross-validation tests passed ===")


def run_python_clifford_full_cross_validation(
    n_random_mv: int = 24,
    n_random_vec: int = 24,
    n_random_biv: int = 24,
    seed: int = 11,
    atol: float = 1e-9,
) -> None:
    print("\n=== Running FULL python-clifford cross-validation suite ===")

    try:
        from clifford import Cl
    except ImportError:
        print("python-clifford not installed — skipping full cross-validation")
        return

    layout, blades = Cl(4, 0)
    _ = layout

    dtype = torch.float64
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)

    basis_names = Clifford4.BASIS_NAMES

    def coeff_array_torch(mv: Clifford4) -> np.ndarray:
        arr = mv.data.detach().cpu().numpy()
        if arr.ndim != 1:
            raise ValueError("Expected unbatched Clifford4 multivector.")
        return arr.copy()

    def coeff_array_cl(mv_cl) -> np.ndarray:
        arr = np.zeros(16, dtype=np.float64)
        if np.isscalar(mv_cl):
            arr[0] = float(mv_cl)
            return arr
        arr[0] = float(mv_cl.value[0])
        for i, blade in enumerate(Clifford4.BASIS[1:], start=1):
            arr[i] = python_clifford_blade_coeff(mv_cl, blade.name, blades)
        return arr

    def assert_mv_close(label: str, mv_t: Clifford4, mv_c, tol: float = atol) -> None:
        a = coeff_array_torch(mv_t)
        b = coeff_array_cl(mv_c)
        if not np.allclose(a, b, atol=tol, rtol=0.0):
            diff = np.abs(a - b)
            worst = int(np.argmax(diff))
            raise AssertionError(
                f"{label} mismatch at blade {basis_names[worst]}: "
                f"torch={a[worst]:.16e}, clifford={b[worst]:.16e}, diff={diff[worst]:.3e}"
            )

    def random_full_mv(scale: float = 1.0) -> Clifford4:
        data = torch.tensor(scale * rng.standard_normal(16), dtype=dtype, device=device)
        return Clifford4(data, device=device, dtype=dtype)

    def random_scalar(nonzero: bool = True) -> Clifford4:
        x = rng.standard_normal()
        if nonzero and abs(x) < 0.2:
            x = 1.0 if x >= 0 else -1.0
        return Clifford4.scalar(float(x), device=device, dtype=dtype)

    def random_vector(nonzero: bool = True) -> Clifford4:
        v = rng.standard_normal(4)
        if nonzero and np.linalg.norm(v) < 0.2:
            v[0] += 1.0
        return Clifford4.vector(torch.tensor(v, dtype=dtype), device=device, dtype=dtype)

    def random_bivector() -> Clifford4:
        coeffs = rng.standard_normal(6)
        data = torch.zeros(16, dtype=dtype, device=device)
        data[5:11] = torch.tensor(coeffs, dtype=dtype)
        return Clifford4(data, device=device, dtype=dtype)

    def random_simple_bivector(nonzero: bool = True) -> Clifford4:
        while True:
            u = random_vector(nonzero=True)
            v = random_vector(nonzero=True)
            B = u.wedge(v).grade(2)
            if (not nonzero) or (float(torch.max(torch.abs(B.data)).item()) > 1e-8):
                return B

    def to_cl(mv_t: Clifford4):
        return mv_t.to_python_clifford(blades)

    def grade_project_cl(mv_cl, g_target: int):
        if np.isscalar(mv_cl):
            return float(mv_cl) if g_target == 0 else 0.0
        arr = coeff_array_cl(mv_cl)
        out = 0.0
        for i, name in enumerate(basis_names):
            if Clifford4.BASIS[i].grade != g_target:
                continue
            coeff = arr[i]
            if coeff == 0.0:
                continue
            if name == "1":
                out = out + coeff
            else:
                out = out + coeff * blades[name]
        return out

    def inner_cl_by_definition(a_t: Clifford4, b_t: Clifford4):
        out = 0.0
        a_arr = coeff_array_torch(a_t)
        b_arr = coeff_array_torch(b_t)
        for i, ai in enumerate(a_arr):
            if abs(ai) == 0.0:
                continue
            ri = Clifford4.BASIS[i].grade
            ai_mv = ai if basis_names[i] == "1" else ai * blades[basis_names[i]]
            for j, bj in enumerate(b_arr):
                if abs(bj) == 0.0:
                    continue
                rj = Clifford4.BASIS[j].grade
                bj_mv = bj if basis_names[j] == "1" else bj * blades[basis_names[j]]
                prod = ai_mv * bj_mv
                out = out + grade_project_cl(prod, abs(ri - rj))
        return out

    def left_contraction_cl_by_definition(a_t: Clifford4, b_t: Clifford4):
        out = 0.0
        a_arr = coeff_array_torch(a_t)
        b_arr = coeff_array_torch(b_t)
        for i, ai in enumerate(a_arr):
            if abs(ai) == 0.0:
                continue
            ri = Clifford4.BASIS[i].grade
            ai_mv = ai if basis_names[i] == "1" else ai * blades[basis_names[i]]
            for j, bj in enumerate(b_arr):
                if abs(bj) == 0.0:
                    continue
                rj = Clifford4.BASIS[j].grade
                if ri > rj:
                    continue
                bj_mv = bj if basis_names[j] == "1" else bj * blades[basis_names[j]]
                prod = ai_mv * bj_mv
                out = out + grade_project_cl(prod, rj - ri)
        return out

    def left_mul_matrix_cl(mv_cl) -> np.ndarray:
        cols = []
        for name in basis_names:
            b = 1.0 if name == "1" else blades[name]
            col = coeff_array_cl(mv_cl * b)
            cols.append(col)
        return np.stack(cols, axis=1)

    def matrix_exp_np(A: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eig(A)
        Vinv = np.linalg.inv(vecs)
        expD = np.diag(np.exp(vals))
        out = vecs @ expD @ Vinv
        out = np.real_if_close(out, tol=1000)
        return np.asarray(out, dtype=np.float64)

    def exp_bivector_general_cl_by_matrix(B_t: Clifford4, t: float) -> Clifford4:
        B_cl = to_cl(B_t)
        L = left_mul_matrix_cl(B_cl)
        E = matrix_exp_np(0.5 * t * L)
        coeffs = E[:, 0]
        return Clifford4(torch.tensor(coeffs, dtype=dtype, device=device), device=device, dtype=dtype)

    def assert_identity_close(label: str, mv: Clifford4, tol: float = 1e-8):
        arr = coeff_array_torch(mv)
        target = np.zeros(16, dtype=np.float64)
        target[0] = 1.0
        if not np.allclose(arr, target, atol=tol, rtol=0.0):
            diff = np.abs(arr - target)
            worst = int(np.argmax(diff))
            raise AssertionError(
                f"{label} identity check failed at blade {basis_names[worst]}: "
                f"value={arr[worst]:.16e}, target={target[worst]:.16e}, diff={diff[worst]:.3e}"
            )

    e1_t = Clifford4.basis_blade("e1", device=device, dtype=dtype)
    e2_t = Clifford4.basis_blade("e2", device=device, dtype=dtype)
    e12_t = Clifford4.basis_blade("e12", device=device, dtype=dtype)
    assert abs(e1_t.gp(e2_t).coeff("e12").item() - 1.0) < atol
    assert_mv_close("basis bridge e12", e12_t, blades["e12"])
    print("✓ Basis convention and bridge sanity pass")

    for n in range(n_random_mv):
        a = random_full_mv()
        b = random_full_mv()
        assert_mv_close(f"gp[{n}]", a.gp(b), to_cl(a) * to_cl(b))
        assert_mv_close(f"wedge[{n}]", a.wedge(b), to_cl(a) ^ to_cl(b))
        assert_mv_close(f"reverse[{n}]", a.reverse(), ~to_cl(a))

        arr = coeff_array_torch(a)
        gi_cl = 0.0
        cc_cl = 0.0
        for i, name in enumerate(basis_names):
            coeff = arr[i]
            if coeff == 0.0:
                continue
            g = Clifford4.BASIS[i].grade
            coeff_gi = ((-1) ** g) * coeff
            coeff_cc = ((-1) ** (g * (g + 1) // 2)) * coeff
            gi_cl = gi_cl + (coeff_gi if name == "1" else coeff_gi * blades[name])
            cc_cl = cc_cl + (coeff_cc if name == "1" else coeff_cc * blades[name])

        assert_mv_close(f"grade_involution[{n}]", a.grade_involution(), gi_cl)
        assert_mv_close(f"clifford_conjugate[{n}]", a.clifford_conjugate(), cc_cl)

    print("✓ gp(), wedge(), reverse(), grade_involution(), clifford_conjugate() pass")

    for n in range(n_random_mv):
        a = random_full_mv()
        b = random_full_mv()
        assert_mv_close(f"inner[{n}]", a.inner(b), inner_cl_by_definition(a, b))
        assert_mv_close(f"left_contraction[{n}]", a.left_contraction(b), left_contraction_cl_by_definition(a, b))

    print("✓ inner() and left_contraction() pass against matched definitions")

    for n in range(n_random_vec):
        v = random_vector(nonzero=True)
        v_unit = v.normalize_vector()
        vv = v_unit.gp(v_unit).coeff("1").item()
        if abs(vv - 1.0) >= atol:
            raise AssertionError(f"normalize_vector[{n}] failed: v^2={vv}")

        vinv = v.inverse_vector()
        assert_identity_close(f"inverse_vector left[{n}]", v.gp(vinv), tol=1e-8)
        assert_identity_close(f"inverse_vector right[{n}]", vinv.gp(v), tol=1e-8)

        v_cl = to_cl(v)
        vinv_cl = v_cl / float((v_cl * v_cl).value[0])
        assert_mv_close(f"inverse_vector vs python-clifford[{n}]", vinv, vinv_cl)

    print("✓ normalize_vector() and inverse_vector() pass")

    for n in range(n_random_vec):
        s = random_scalar(nonzero=True)
        sinv = s.inverse_scalar()
        assert_identity_close(f"inverse_scalar left[{n}]", s.gp(sinv), tol=1e-12)
        assert_identity_close(f"inverse_scalar right[{n}]", sinv.gp(s), tol=1e-12)
        s_cl = to_cl(s)
        sinv_cl = 1.0 / float(s_cl)
        assert_mv_close(f"inverse_scalar vs python-clifford[{n}]", sinv, sinv_cl)

    print("✓ inverse_scalar() pass")

    for n in range(n_random_biv):
        B = random_simple_bivector(nonzero=True)
        t = float(rng.uniform(-1.5, 1.5))

        R = B.exp_simple_bivector(t=t)
        Rn = R.normalize_rotor()
        assert_identity_close(f"normalize_rotor[{n}]", Rn.gp(Rn.reverse()), tol=1e-9)

        Rinv = R.inverse_rotor()
        assert_identity_close(f"inverse_rotor left[{n}]", R.gp(Rinv), tol=1e-8)
        assert_identity_close(f"inverse_rotor right[{n}]", Rinv.gp(R), tol=1e-8)

        B_cl = to_cl(B)
        B2 = coeff_array_cl(B_cl * B_cl)[0]
        mag = math.sqrt(max(0.0, -B2))
        if mag > 1e-12:
            R_cl = math.cos(0.5 * t * mag) + (math.sin(0.5 * t * mag) / mag) * B_cl
        else:
            R_cl = 1.0 + 0.5 * t * B_cl

        assert_mv_close(f"exp_simple_bivector[{n}]", R, R_cl, tol=1e-8)
        assert_mv_close(f"inverse_rotor vs python-clifford[{n}]", Rinv, ~R_cl, tol=1e-8)

        v = random_vector(nonzero=True)
        assert_mv_close(f"sandwich[{n}]", R.sandwich(v), R_cl * to_cl(v) * (~R_cl), tol=1e-8)

    print("✓ exp_simple_bivector(), normalize_rotor(), inverse_rotor(), sandwich() pass")

    for n in range(n_random_biv):
        B = random_bivector()
        t = float(rng.uniform(-1.2, 1.2))
        R = B.exp_bivector_general(t=t)
        R_ref = exp_bivector_general_cl_by_matrix(B, t)
        assert_identity_close(f"exp_bivector_general norm[{n}]", R.gp(R.reverse()), tol=1e-7)

        v = random_vector(nonzero=True)
        v_t = R.sandwich(v)
        v_ref = R_ref.sandwich(v)
        if not v_t.almost_equal(v_ref, atol=1e-7):
            R_ref_cl = to_cl(R_ref)
            assert_mv_close(f"exp_bivector_general action[{n}]", v_t, R_ref_cl * to_cl(v) * (~R_ref_cl), tol=1e-7)

    print("✓ exp_bivector_general() pass")

    for n in range(n_random_mv):
        M = random_full_mv(scale=0.4) + Clifford4.scalar(1.0, device=device, dtype=dtype)
        Minv = M.inverse_general()
        assert_identity_close(f"inverse_general left[{n}]", M.gp(Minv), tol=1e-7)
        assert_identity_close(f"inverse_general right[{n}]", Minv.gp(M), tol=1e-7)

        M_cl = to_cl(M)
        L = left_mul_matrix_cl(M_cl)
        rhs = np.zeros(16, dtype=np.float64)
        rhs[0] = 1.0
        x = np.linalg.solve(L, rhs)
        Minv_ref = Clifford4(torch.tensor(x, dtype=dtype, device=device), device=device, dtype=dtype)

        if not Minv.almost_equal(Minv_ref, atol=1e-7):
            a = coeff_array_torch(Minv)
            b = coeff_array_torch(Minv_ref)
            if not np.allclose(a, b, atol=1e-6, rtol=0.0):
                diff = np.abs(a - b)
                worst = int(np.argmax(diff))
                raise AssertionError(
                    f"inverse_general vs ref[{n}] mismatch at {basis_names[worst]}: "
                    f"torch={a[worst]:.16e}, ref={b[worst]:.16e}, diff={diff[worst]:.3e}"
                )

    print("✓ inverse_general() pass")

    s = random_scalar(nonzero=True)
    v = random_vector(nonzero=True)
    B = random_simple_bivector(nonzero=True)
    R = B.exp_simple_bivector(t=0.7)
    M = random_full_mv(scale=0.3) + Clifford4.scalar(1.0, device=device, dtype=dtype)

    assert_identity_close("inverse(auto)-scalar", s.gp(s.inverse(mode='auto')), tol=1e-10)
    assert_identity_close("inverse(auto)-vector", v.gp(v.inverse(mode='auto')), tol=1e-8)
    assert_identity_close("inverse(auto)-rotor", R.gp(R.inverse(mode='auto')), tol=1e-8)
    assert_identity_close("inverse(auto)-general", M.gp(M.inverse(mode='auto')), tol=1e-7)
    print("✓ inverse(mode='auto') pass")

    print("=== FULL python-clifford cross-validation suite passed ===")


if __name__ == "__main__":
    run_cl4_unit_tests()
    run_batched_random_rotor_tests(batch_size=128, seed=7, use_gpu_if_available=True)
    run_python_clifford_cross_validation()
    run_python_clifford_full_cross_validation(
        n_random_mv=24,
        n_random_vec=24,
        n_random_biv=24,
        seed=11,
        atol=1e-9,
    )
