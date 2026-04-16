"""
test_clifford41_grammar_molecule_v0.1.0.py

Comprehensive Cl(4,1) / CGA test harness aimed at the Grammar Molecule stack.

Goals
-----
1. Verify the algebraic core against the reference `clifford` package.
2. Verify the conformal geometry layer used by grammar-molecule style state carriers.
3. Detect which higher operators needed by the grammar molecule are present or absent.
4. Support partial implementations by SKIPPING tests for missing methods instead of crashing.

Run:
    python test_clifford41_grammar_molecule_v0.1.0.py
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from clifford import Cl
    _HAS_CLIFFORD = True
except Exception:
    Cl = None
    _HAS_CLIFFORD = False

from clifford41_conformal import Clifford41

# -----------------------------------------------------------------------------
# Reference clifford setup
# -----------------------------------------------------------------------------
if _HAS_CLIFFORD:
    layout, blades = Cl(4, 1)
else:
    layout, blades = None, {}

REF_BASIS_NAMES = [
    "1",
    "e1", "e2", "e3", "e4", "e5",
    "e12", "e13", "e14", "e15", "e23", "e24", "e25", "e34", "e35", "e45",
    "e123", "e124", "e125", "e134", "e135", "e145", "e234", "e235", "e245", "e345",
    "e1234", "e1235", "e1245", "e1345", "e2345",
    "e12345",
]

if _HAS_CLIFFORD:
    REF_BLADE_MAP = {
        "1": 1,
        "e1": blades["e1"],
    "e2": blades["e2"],
    "e3": blades["e3"],
    "e4": blades["e4"],
    "e5": blades["e5"],
    "e12": blades["e12"],
    "e13": blades["e13"],
    "e14": blades["e14"],
    "e15": blades["e15"],
    "e23": blades["e23"],
    "e24": blades["e24"],
    "e25": blades["e25"],
    "e34": blades["e34"],
    "e35": blades["e35"],
    "e45": blades["e45"],
    "e123": blades["e123"],
    "e124": blades["e124"],
    "e125": blades["e125"],
    "e134": blades["e134"],
    "e135": blades["e135"],
    "e145": blades["e145"],
    "e234": blades["e234"],
    "e235": blades["e235"],
    "e245": blades["e245"],
    "e345": blades["e345"],
    "e1234": blades["e1234"],
    "e1235": blades["e1235"],
    "e1245": blades["e1245"],
    "e1345": blades["e1345"],
    "e2345": blades["e2345"],
        "e12345": blades["e12345"],
    }
else:
    REF_BLADE_MAP = {}

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
tol = 1e-9


def ref_e_o():
    if not _HAS_CLIFFORD:
        raise RuntimeError("reference clifford package not installed")
    return 0.5 * (blades["e5"] - blades["e4"])


def ref_e_inf():
    if not _HAS_CLIFFORD:
        raise RuntimeError("reference clifford package not installed")
    return blades["e4"] + blades["e5"]


def ref_conformal_lift(x: Sequence[float]):
    if not _HAS_CLIFFORD:
        raise RuntimeError("reference clifford package not installed")
    x = np.asarray(x, dtype=float)
    r2 = float(np.dot(x, x))
    return ref_e_o() + x[0] * blades["e1"] + x[1] * blades["e2"] + x[2] * blades["e3"] + 0.5 * r2 * ref_e_inf()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@dataclass
class Counters:
    passed: int = 0
    failed: int = 0
    skipped: int = 0


COUNTS = Counters()


def print_header() -> None:
    print("=== Grammar Molecule Cl(4,1) Test Harness v0.1.0 ===")
    print(f"Device   : {device}")
    print(f"DType    : {dtype}")
    print(f"Tolerance: {tol}")
    print()



def has_method(obj, name: str) -> bool:
    return callable(getattr(obj, name, None))



def pt_basis(name: str) -> Clifford41:
    mapped = "I" if name == "e12345" and "I" in Clifford41.BASIS_INDEX else name
    return Clifford41.basis(mapped, device=device, dtype=dtype)



def ref_basis(name: str):
    if not _HAS_CLIFFORD:
        raise RuntimeError("reference clifford package not installed")
    return REF_BLADE_MAP[name]



def ref_scalar(mv) -> float:
    return float(mv.value[0])



def max_abs_diff(a, b) -> float:
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a_np - b_np))) if a_np.shape != () or b_np.shape != () else float(abs(a_np - b_np))



def assert_close(name: str, got, expected, atol: float = tol) -> None:
    diff = max_abs_diff(got, expected)
    if diff <= atol:
        print(f"✓ {name} — PASS")
        COUNTS.passed += 1
    else:
        print(f"✗ {name} — FAIL (max diff = {diff:.2e})")
        print(f"   Expected: {expected}")
        print(f"   Got     : {got}")
        COUNTS.failed += 1



def skip_test(name: str, reason: str) -> None:
    print(f"• {name} — SKIP ({reason})")
    COUNTS.skipped += 1



def fail_test(name: str, exc: BaseException) -> None:
    print(f"✗ {name} — ERROR")
    print(f"   {type(exc).__name__}: {exc}")
    COUNTS.failed += 1



def run_test(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
    except AssertionError as exc:
        fail_test(name, exc)
    except Exception as exc:  # pragma: no cover - debugging path
        fail_test(name, exc)



def pt_scalar(mv: Clifford41) -> float:
    return float(mv.data[..., 0].item())



def pt_coeff(mv: Clifford41, basis_name: str) -> float:
    idx = Clifford41.BASIS_INDEX["I" if basis_name == "e12345" and "I" in Clifford41.BASIS_INDEX else basis_name]
    return float(mv.data[..., idx].item())



def mv_to_numpy(mv: Clifford41) -> np.ndarray:
    return mv.data.detach().cpu().numpy()



def ref_to_numpy(mv) -> np.ndarray:
    if not _HAS_CLIFFORD:
        raise RuntimeError("reference clifford package not installed")
    # Reference clifford stores coefficients in the same bitmask order for this basis list.
    coeffs = np.asarray(mv.value, dtype=float)
    if coeffs.shape[-1] != 32:
        raise ValueError(f"Reference multivector length {coeffs.shape[-1]} != 32")
    return coeffs



def compare_full_mv(name: str, pt_mv: Clifford41, ref_mv, atol: float = tol) -> None:
    pt_arr = mv_to_numpy(pt_mv)
    ref_arr = ref_to_numpy(ref_mv)
    assert_close(name, pt_arr, ref_arr, atol=atol)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_basis_and_metric() -> None:
    for basis_name, expected in [("e1", 1.0), ("e2", 1.0), ("e3", 1.0), ("e4", 1.0), ("e5", -1.0)]:
        A = pt_basis(basis_name)
        got = pt_scalar(A * A)
        assert_close(f"Metric square {basis_name}^2", got, expected)



def test_basis_anticommutation() -> None:
    A = pt_basis("e1")
    B = pt_basis("e2")
    AB = A * B
    BA = B * A
    assert_close("Anticommutation e1e2 + e2e1", mv_to_numpy(AB + BA), np.zeros(32))
    assert_close("Basis bivector e1*e2 -> e12", pt_coeff(AB, "e12"), 1.0)



def test_reverse_signs() -> None:
    expectations = {
        "e1": 1.0,
        "e12": -1.0,
        "e123": -1.0,
        "e1234": 1.0,
        "e12345": 1.0,
    }
    for name, expected in expectations.items():
        got = pt_coeff(pt_basis(name).reverse(), name)
        assert_close(f"Reverse sign for {name}", got, expected)



def test_grade_projection() -> None:
    mv = pt_basis("e1") + 2.0 * pt_basis("e12") + 3.0 * pt_basis("e123")
    g1 = mv.grade(1)
    g2 = mv.grade(2)
    g3 = mv.grade(3)
    assert_close("grade(1) keeps e1", pt_coeff(g1, "e1"), 1.0)
    assert_close("grade(1) removes e12", pt_coeff(g1, "e12"), 0.0)
    assert_close("grade(2) keeps e12", pt_coeff(g2, "e12"), 2.0)
    assert_close("grade(3) keeps e123", pt_coeff(g3, "e123"), 3.0)



def test_reference_basis_products() -> None:
    if not _HAS_CLIFFORD:
        skip_test("Reference basis products", "reference clifford package not installed")
        return
    pairs = [("e1", "e2"), ("e2", "e3"), ("e1", "e5"), ("e12", "e23"), ("e45", "e45")]
    for a_name, b_name in pairs:
        pt_mv = pt_basis(a_name) * pt_basis(b_name)
        ref_mv = ref_basis(a_name) * ref_basis(b_name)
        compare_full_mv(f"Reference GP {a_name}*{b_name}", pt_mv, ref_mv)



def test_null_basis() -> None:
    if not hasattr(Clifford41, "e_o") or not hasattr(Clifford41, "e_inf"):
        skip_test("Null basis tests", "e_o/e_inf helpers not present")
        return
    eo = Clifford41.e_o(device=device, dtype=dtype)
    einf = Clifford41.e_inf(device=device, dtype=dtype)
    assert_close("e_o is null", pt_scalar(eo * eo), 0.0)
    assert_close("e_inf is null", pt_scalar(einf * einf), 0.0)
    assert_close("e_o · e_inf scalar", pt_scalar(eo * einf), -1.0)



def test_conformal_lift_nullity() -> None:
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-0.5, 1.2, -2.8],
        [10.0, -5.0, 0.0],
    ]
    for p in points:
        P = Clifford41.conformal_lift(torch.tensor(p, device=device, dtype=dtype))
        assert_close(f"Conformal null norm {p}", float(P.norm2().item()), 0.0)



def test_conformal_lift_against_reference() -> None:
    if not _HAS_CLIFFORD:
        skip_test("Conformal lift against reference", "reference clifford package not installed")
        return
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [-0.5, 1.2, -2.8],
    ]
    for p in points:
        pt_mv = Clifford41.conformal_lift(torch.tensor(p, device=device, dtype=dtype))
        ref_mv = ref_conformal_lift(p)
        compare_full_mv(f"Conformal lift coefficients {p}", pt_mv, ref_mv)



def test_distance_identity() -> None:
    p = [1.0, 2.0, -1.0]
    q = [-2.0, 0.5, 3.0]
    P = Clifford41.conformal_lift(torch.tensor(p, device=device, dtype=dtype))
    Q = Clifford41.conformal_lift(torch.tensor(q, device=device, dtype=dtype))
    lhs = -2.0 * pt_scalar(P * Q)
    rhs = float(np.sum((np.asarray(p) - np.asarray(q)) ** 2))
    assert_close("CGA distance identity -2 P·Q = |p-q|^2", lhs, rhs)



def test_rotor_null_preservation_and_reference() -> None:
    B = pt_basis("e12")
    angle = math.pi / 3.0
    R = Clifford41.rotor(B, angle=angle, device=device, dtype=dtype)
    P = Clifford41.conformal_lift(torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype))
    P_rot = R.sandwich(P)
    assert_close("Rotor preserves nullness", float(P_rot.norm2().item()), 0.0)

    if _HAS_CLIFFORD:
        ref_R = math.cos(angle / 2.0) - math.sin(angle / 2.0) * ref_basis("e12")
        ref_P = ref_conformal_lift([1.0, 0.0, 0.0])
        ref_rot = ref_R * ref_P * ~ref_R
        compare_full_mv("Rotor sandwich matches reference", P_rot, ref_rot, atol=1e-8)
    else:
        skip_test("Rotor sandwich matches reference", "reference clifford package not installed")



def test_batch_conformal_lift() -> None:
    batch = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    pts = Clifford41.conformal_lift(batch)
    norms = pts.norm2().detach().cpu().numpy()
    assert_close("Batch conformal lift norms", norms, np.zeros(3))



def test_wedge_if_present() -> None:
    if not has_method(Clifford41, "wedge"):
        skip_test("Wedge operator", "wedge() not implemented")
        return
    A = pt_basis("e1")
    B = pt_basis("e2")
    W = A.wedge(B)
    assert_close("Wedge e1^e2 -> e12", pt_coeff(W, "e12"), 1.0)
    if _HAS_CLIFFORD:
        ref_W = ref_basis("e1") ^ ref_basis("e2")
        compare_full_mv("Wedge matches reference", W, ref_W)
    else:
        skip_test("Wedge matches reference", "reference clifford package not installed")



def test_inner_if_present() -> None:
    candidate_names = ["inner", "dot"]
    inner_name = next((name for name in candidate_names if has_method(Clifford41, name)), None)
    if inner_name is None:
        skip_test("Inner product", "inner()/dot() not implemented")
        return
    fn = getattr(pt_basis("e1"), inner_name)
    got_mv = fn(pt_basis("e1"))
    assert_close(f"{inner_name}(e1,e1) scalar", pt_scalar(got_mv), 1.0)



def test_dual_if_present() -> None:
    if not has_method(Clifford41, "dual"):
        skip_test("Dual/undual", "dual() not implemented")
        return
    A = pt_basis("e1")
    D = A.dual()
    if has_method(D, "undual"):
        U = D.undual()
        compare_full_mv("undual(dual(e1)) recovers e1", U, ref_basis("e1"))
    else:
        skip_test("undual(dual(e1))", "undual() not implemented")



def test_commutator_if_present() -> None:
    if not has_method(Clifford41, "commutator"):
        skip_test("Commutator product", "commutator() not implemented")
        return
    A = pt_basis("e12")
    B = pt_basis("e23")
    C = A.commutator(B)
    if _HAS_CLIFFORD:
        ref_C = 0.5 * (ref_basis("e12") * ref_basis("e23") - ref_basis("e23") * ref_basis("e12"))
        compare_full_mv("Commutator matches reference", C, ref_C)
    else:
        skip_test("Commutator matches reference", "reference clifford package not installed")



def test_inverse_if_present() -> None:
    if not has_method(Clifford41, "inverse"):
        skip_test("Inverse", "inverse() not implemented")
        return
    A = 2.0 * pt_basis("e1")
    Ainv = A.inverse()
    prod = A * Ainv
    assert_close("Inverse A*A^-1 scalar", pt_scalar(prod), 1.0, atol=1e-8)



def test_normalize_if_present() -> None:
    for name in ["normalize", "normalised", "normalized"]:
        if has_method(Clifford41, name):
            fn = getattr(2.0 * pt_basis("e12"), name)
            B = fn()
            norm2 = pt_scalar(B * B.reverse())
            assert_close(f"{name} produces unit versor norm", abs(norm2), 1.0, atol=1e-8)
            return
    skip_test("Normalization", "normalize()/normalised()/normalized() not implemented")



def test_translator_if_present() -> None:
    if not has_method(Clifford41, "translator"):
        skip_test("Translator", "translator() not implemented")
        return
    t = torch.tensor([0.25, -0.5, 0.75], device=device, dtype=dtype)
    T = Clifford41.translator(t, device=device, dtype=dtype)
    P = Clifford41.conformal_lift(torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype))
    moved = T.sandwich(P)
    if _HAS_CLIFFORD:
        expected = ref_conformal_lift([1.25, 1.5, 3.75])
        compare_full_mv("Translator moves conformal point", moved, expected, atol=1e-8)
    else:
        skip_test("Translator moves conformal point", "reference clifford package not installed")



def test_exp_log_if_present() -> None:
    has_exp = has_method(Clifford41, "exp") or has_method(Clifford41, "exp_bivector")
    has_log = has_method(Clifford41, "log") or has_method(Clifford41, "log_rotor")
    if not has_exp:
        skip_test("Bivector exponential", "exp()/exp_bivector() not implemented")
    if not has_log:
        skip_test("Rotor logarithm", "log()/log_rotor() not implemented")
    # No hard fail here without a stable API signature.



def test_grammar_molecule_capability_report() -> None:
    required = [
        "gp",
        "reverse",
        "grade",
        "conformal_lift",
        "sandwich",
        "rotor",
        "wedge",
        "inner",
        "dual",
        "commutator",
        "translator",
        "inverse",
        "normalize",
    ]
    present = []
    missing = []
    for name in required:
        if has_method(Clifford41, name) or hasattr(Clifford41, name):
            present.append(name)
        else:
            missing.append(name)
    print("\nGrammar Molecule capability snapshot")
    print("-----------------------------------")
    print("Present:", ", ".join(present) if present else "None")
    print("Missing:", ", ".join(missing) if missing else "None")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print_header()
    if not _HAS_CLIFFORD:
        print("Note: python-clifford is not installed in this environment; reference-comparison tests will be skipped.\n")

    tests: List[Tuple[str, Callable[[], None]]] = [
        ("basis_and_metric", test_basis_and_metric),
        ("basis_anticommutation", test_basis_anticommutation),
        ("reverse_signs", test_reverse_signs),
        ("grade_projection", test_grade_projection),
        ("reference_basis_products", test_reference_basis_products),
        ("null_basis", test_null_basis),
        ("conformal_lift_nullity", test_conformal_lift_nullity),
        ("conformal_lift_against_reference", test_conformal_lift_against_reference),
        ("distance_identity", test_distance_identity),
        ("rotor_null_preservation_and_reference", test_rotor_null_preservation_and_reference),
        ("batch_conformal_lift", test_batch_conformal_lift),
        ("wedge_if_present", test_wedge_if_present),
        ("inner_if_present", test_inner_if_present),
        ("dual_if_present", test_dual_if_present),
        ("commutator_if_present", test_commutator_if_present),
        ("inverse_if_present", test_inverse_if_present),
        ("normalize_if_present", test_normalize_if_present),
        ("translator_if_present", test_translator_if_present),
        ("exp_log_if_present", test_exp_log_if_present),
        ("grammar_molecule_capability_report", test_grammar_molecule_capability_report),
    ]

    for test_name, fn in tests:
        run_test(test_name, fn)

    print("\n" + "=" * 72)
    print(f"PASS   : {COUNTS.passed}")
    print(f"FAIL   : {COUNTS.failed}")
    print(f"SKIP   : {COUNTS.skipped}")
    print(f"TOTAL  : {COUNTS.passed + COUNTS.failed + COUNTS.skipped}")
    if COUNTS.failed == 0:
        print("✅ No failing tests.")
    else:
        print("⚠️ Some tests failed.")
