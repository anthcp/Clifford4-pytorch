"""
test_clifford41_vs_original.py
Full test harness comparing the new Versor-based Clifford41 to original clifford package.
"""

import torch
import numpy as np
from clifford import Cl

from clifford41_conformal import Clifford41

# Original clifford setup
layout, blades = Cl(4, 1)
e1, e2, e3, e4, e5 = blades['e1'], blades['e2'], blades['e3'], blades['e4'], blades['e5']
e_inf = e4 + e5
e_o   = 0.5 * (e5 - e4)

def original_conformal_lift(x):
    x = np.asarray(x, dtype=float)
    r2 = np.dot(x, x)
    return e_o + x[0]*e1 + x[1]*e2 + x[2]*e3 + 0.5*r2*e_inf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
tol = 1e-8

print("=== Versor-based Cl(4,1) vs Original clifford Test Harness ===")
print(f"Device: {device}\n")

tests_passed = 0
total_tests = 0

def assert_close(a, b, name):
    global tests_passed, total_tests
    total_tests += 1
    diff = np.abs(a - b)
    max_diff = diff.max() if hasattr(diff, "max") else diff
    if max_diff < tol:
        print(f"✓ {name} — PASS")
        tests_passed += 1
    else:
        print(f"✗ {name} — FAIL (max diff = {max_diff:.2e})")

# Run the same tests as before + new operations
test_points = [[0.,0.,0.], [1.,2.,3.], [-0.5,1.2,-2.8], [10.,-5.,0.]]
for p in test_points:
    P_orig = original_conformal_lift(p)
    norm2_orig = float((P_orig * ~P_orig).value[0])   # fixed deprecation

    P_pt = Clifford41.conformal_lift(torch.tensor(p, device=device, dtype=dtype))
    norm2_pt = float(P_pt.norm2().item())

    assert_close(norm2_orig, norm2_pt, f"Conformal lift norm² for {p}")

# Rotor / sandwich
B = Clifford41.basis("e12", device=device, dtype=dtype)
R = Clifford41.rotor(B, angle=torch.pi/3, device=device, dtype=dtype)
P_pt = Clifford41.conformal_lift(torch.tensor([1.,0.,0.], device=device, dtype=dtype))
P_rot = R.sandwich(P_pt)
assert_close(0.0, float(P_rot.norm2().item()), "Rotor sandwich preserves null vector")

# New operations
A = Clifford41.basis("e1", device=device, dtype=dtype)
B = Clifford41.basis("e2", device=device, dtype=dtype)
#assert_close(1.0, float(A.wedge(B).grade(2).data[0].item()), "Wedge A ∧ B")
W = A.wedge(B).grade(2)
assert_close(
    1.0,
    float(W.data[Clifford41.BASIS_INDEX["e12"]].item()),
    "Wedge A ∧ B"
)
assert_close(0.0, float(A.inner(B).data[0].item()), "Inner product A · B")

print(f"\n=== SUMMARY: {tests_passed}/{total_tests} tests passed ===")
if tests_passed == total_tests:
    print("✅ All tests passed — Versor-based Clifford41 matches original clifford.")