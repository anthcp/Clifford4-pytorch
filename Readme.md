# Clifford Algebra PyTorch Cl(4,0) v0.1.1

A compact PyTorch-based multivector library for **Cl(4,0)** with:

- explicit basis-index maps
- geometric, exterior, inner, and left-contraction products
- grade projections and involutions
- simple-bivector rotor exponentials
- batched tensor support
- optional cross-validation with `python-clifford`

This library is designed for **numerical Clifford algebra work in PyTorch**, especially when you want a transparent coefficient-level implementation that can be checked directly against `python-clifford`.

---

## Features

- **Cl(4,0)** Euclidean signature: \( e_i^2 = +1 \)
- Dense 16-component multivector representation
- Explicit fixed basis ordering
- Geometric product using a precomputed multiplication table
- Exterior product
- Hestenes-style inner product
- Left contraction
- Reverse, grade involution, Clifford conjugation
- Rotor sandwich action
- Closed-form exponential for **simple bivectors only**
- Batched multivector support via leading tensor dimensions
- Bridges to and from `python-clifford`

---

## Installation

### Required

```bash

pip install torch numpy

```

Optional

For plotting the demo:

pip install matplotlib

For cross-validation against python-clifford:

pip install clifford


⸻

Basis and conventions

This library stores coefficients in the following fixed order:
```
Index	Blade	Grade
0	1	0
1	e1	1
2	e2	1
3	e3	1
4	e4	1
5	e12	2
6	e13	2
7	e14	2
8	e23	2
9	e24	2
10	e34	2
11	e123	3
12	e124	3
13	e134	3
14	e234	3
15	e1234	4

The multiplication convention is aligned with python-clifford, so for example:

e1 * e2 = +e12
e2 * e1 = -e12
e12 * e12 = -1

```
⸻

Import

Assuming your file is named:

clifford_algebra_pytorch_cl4_v0.1.1.py

```
import it like this:

from clifford_algebra_pytorch_cl4_v0.1.1 import Clifford4
```
If Python complains about dots in the filename, rename the file to something import-safe like:

clifford_algebra_pytorch_cl4_v0_1_1.py

and import:
```
from clifford_algebra_pytorch_cl4_v0_1_1 import Clifford4
```

⸻

Quick start

Create scalars, vectors, and basis blades
```
import torch
from clifford_algebra_pytorch_cl4_v0_1_1 import Clifford4

dtype = torch.float64
device = torch.device("cpu")

one = Clifford4.scalar(1.0, device=device, dtype=dtype)
e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
e2 = Clifford4.basis_blade("e2", device=device, dtype=dtype)
e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)

v = Clifford4.vector(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype), device=device, dtype=dtype)
```
Build a multivector from a coefficient dictionary
```
mv = Clifford4.from_coeff_dict({
    "1": 1.0,
    "e1": 2.0,
    "e23": -0.5,
    "e1234": 0.25,
}, device=device, dtype=dtype)

print(mv)
print(mv.coeff_dict())

```
⸻

Core operations

Geometric product
```
ab = e1.gp(e2)
print(ab.coeff_dict())   # should show {'e12': 1.0}
```
Exterior product
```
w = e1.wedge(e2)
print(w.coeff_dict())    # {'e12': 1.0}
```
Inner product

This implementation uses a Hestenes-style inner product:

[
\langle A_r B_s \rangle_{|r-s|}
]

for homogeneous grade components, extended by bilinearity.
```
inner_val = e1.inner(e1)
print(inner_val.coeff_dict())   # {'1': 1.0}
```
Left contraction

This implementation uses:

[
A_r ,\lrcorner, B_s = \langle A_r B_s \rangle_{s-r}, \quad r \le s
]

and zero otherwise.
```
res = e1.left_contraction(e12)
print(res.coeff_dict())   # {'e2': 1.0}
```

⸻

Grade projections and involutions

Grade projection
```
mv = Clifford4.from_coeff_dict({
    "1": 1.0,
    "e1": 2.0,
    "e12": 3.0,
    "e123": 4.0,
}, device=device, dtype=dtype)

print(mv.grade(0).coeff_dict())   # scalar part
print(mv.grade(1).coeff_dict())   # vector part
print(mv.grade(2).coeff_dict())   # bivector part
print(mv.grade(3).coeff_dict())   # trivector part
```
Reverse
```
print(e12.reverse().coeff_dict())   # {'e12': -1.0}
```
Grade involution
```
print(mv.grade_involution().coeff_dict())
```
Clifford conjugation
```
print(mv.clifford_conjugate().coeff_dict())
```

⸻

Norm-like scalar via reverse

A common scalar quantity is the scalar part of ( M \widetilde{M} ):
```
n2 = mv.norm_sq_via_reverse()
print(n2.item())
```

⸻

Rotor usage

Important restriction

exp_simple_bivector() is only valid for simple bivectors.

That means it works for a single Euclidean rotation plane such as:
	•	e12
	•	e13
	•	e24

but not for a general nonsimple 4D bivector such as:

e12 + e34

because that generally represents two independent planes.

Build a rotor from a simple bivector
```
import math

B = -Clifford4.basis_blade("e12", device=device, dtype=dtype)
R = B.exp_simple_bivector(t=math.pi / 2.0)
print(R.coeff_dict())
```
This computes:

[
R = \exp\left(\frac{t}{2} B\right)
]

for a simple bivector (B).

Rotate a vector with the sandwich action
```
v_rot = R.sandwich(e1)
print(v_rot.coeff_dict())
```
With the standard convention used here, a (90^\circ) rotation in the e12 plane sends:

e1 -> e2

when using:
```
B = -e12
R = exp((pi/2)/2 * B)
```
Check rotor normalization
```
RRr = R.gp(R.reverse())
print(RRr.coeff_dict())   # should be close to {'1': 1.0}
```

⸻

Batched usage

One of the main advantages of this implementation is that it supports leading batch dimensions.

Batched vectors
```
vecs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 4.0],
], dtype=dtype)

V = Clifford4.vector(vecs, device=device, dtype=dtype)
print(V.data.shape)   # (3, 16)
```
Apply the same rotor to a batch
```
B = Clifford4.basis_blade("e12", coeff=-1.0, device=device, dtype=dtype)
R = B.exp_simple_bivector(t=math.pi / 3.0)

R_batch = Clifford4(R.data.unsqueeze(0).repeat(3, 1), device=device, dtype=dtype)
V_rot = R_batch.sandwich(V)

print(V_rot.data.shape)   # (3, 16)
```
Check norm preservation across the batch
```
in_sq = V.gp(V).coeff("1")
out_sq = V_rot.gp(V_rot).coeff("1")

print(torch.allclose(in_sq, out_sq, atol=1e-10, rtol=0.0))

```
⸻

Working with coefficients directly

Read one coefficient
```
c = mv.coeff("e23")
print(c.item())
```
Export to a readable dictionary
```
print(mv.coeff_dict(atol=1e-12))
```
Test whether a multivector is pure grade
```
print(e12.is_pure_grade(2))
print((e1 + e12).is_pure_grade(2))
```

⸻

Interoperability with python-clifford

If you have python-clifford installed, you can convert back and forth.

Convert to python-clifford
```
from clifford import Cl
layout, blades = Cl(4, 0)

mv = Clifford4.from_coeff_dict({
    "1": 1.0,
    "e1": 2.0,
    "e23": -0.5,
}, device=device, dtype=dtype)

mv_cl = mv.to_python_clifford(blades)
print(mv_cl)
```

Convert from python-clifford
```
mv_back = Clifford4.from_python_clifford(mv_cl, blades, device=device, dtype=dtype)
print(mv_back.coeff_dict())
```

Extract a coefficient safely from python-clifford

Do not use the inner product | as a generic coefficient extractor.

Use the helper instead:
```
from clifford_algebra_pytorch_cl4_v0_1_1 import python_clifford_blade_coeff

coeff = python_clifford_blade_coeff(mv_cl, "e23", blades)
print(coeff)

```
⸻

Recommended workflow for debugging

A good debugging workflow is:
	1.	build a multivector in Clifford4
	2.	compute the result in PyTorch
	3.	convert to python-clifford
	4.	recompute symbolically there
	5.	compare coefficients blade by blade

Example:
```
from clifford import Cl
from clifford_algebra_pytorch_cl4_v0_1_1 import Clifford4, python_clifford_blade_coeff

layout, blades = Cl(4, 0)

a = Clifford4.basis_blade("e1", device=device, dtype=dtype)
b = Clifford4.basis_blade("e2", device=device, dtype=dtype)

ab_torch = a.gp(b)
ab_cliff = a.to_python_clifford(blades) * b.to_python_clifford(blades)

print(ab_torch.coeff("e12").item())
print(python_clifford_blade_coeff(ab_cliff, "e12", blades))

```
⸻

Running the built-in tests

The file contains three main test routines:
	•	run_cl4_unit_tests()
	•	run_batched_random_rotor_tests()
	•	run_python_clifford_cross_validation()

Run the script directly:
```
python clifford_algebra_pytorch_cl4_v0_1_1.py
```
If python-clifford is not installed, the cross-validation section will be skipped.

⸻

Example: simple 2-plane rotation
```
import math
import torch
from clifford_algebra_pytorch_cl4_v0_1_1 import Clifford4

dtype = torch.float64
device = torch.device("cpu")

e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)

B = -e12
R = B.exp_simple_bivector(t=math.pi / 2.0)

v_rot = R.sandwich(e1)

print("Rotor:", R.coeff_dict(atol=1e-12))
print("Rotated vector:", v_rot.coeff_dict(atol=1e-12))
```
Expected result: the rotated vector is approximately e2.

⸻

Example: reject a nonsimple bivector exponential
```
e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)
e34 = Clifford4.basis_blade("e34", device=device, dtype=dtype)

B = e12 + e34

try:
    R = B.exp_simple_bivector(t=1.0)
except ValueError as exc:
    print("Expected failure:", exc)
```
This should fail, because e12 + e34 is not a simple bivector in 4D.

⸻

Limitations

Current version intentionally keeps the implementation simple and explicit.

Important limitations
	•	Only Cl(4,0) is implemented
	•	exp_simple_bivector() only supports simple bivectors
	•	to_python_clifford() and from_python_clifford() support unbatched multivectors only
	•	Products are implemented with explicit loops over the 16 basis blades, which is clear and robust but not yet optimized for large-scale GPU workloads

⸻

When to use this library

This implementation is a good fit when you want:
	•	a readable reference implementation
	•	transparent coefficient-level debugging
	•	PyTorch compatibility
	•	batch support for many multivectors
	•	direct comparison with python-clifford

It is especially useful for:
	•	rotor-based experiments
	•	PLVS-style multivector simulations
	•	verifying Clifford algebra identities numerically
	•	building custom geometric-algebra research code in PyTorch

⸻

Future improvements

Possible future extensions:
	•	faster tensorized geometric product
	•	support for general bivector exponentials in 4D
	•	more signatures such as Cl(3,0), Cl(1,3), Cl(3,1)
	•	multivector inverses and normalization helpers
	•	outermorphisms and linear maps
	•	GPU-optimized batched kernels

⸻

Minimal API summary

Constructors
```
Clifford4.zeros(...)
Clifford4.scalar(...)
Clifford4.vector(...)
Clifford4.basis_blade(...)
Clifford4.from_coeff_dict(...)
Clifford4.random_vector(...)
```
Basic methods
```
mv.coeff(...)
mv.coeff_dict(...)
mv.grade(...)
mv.reverse()
mv.grade_involution()
mv.clifford_conjugate()
mv.scalar_part()
mv.vector_part()
mv.bivector_part()
mv.norm_sq_via_reverse()
mv.is_pure_grade(...)
mv.almost_equal(...)
```
Products
```
a.gp(b)
a.wedge(b)
a.inner(b)
a.left_contraction(b)
```
Rotor methods
```
B.exp_simple_bivector(...)
R.sandwich(x)
```
python-clifford bridge
```
mv.to_python_clifford(blades)
Clifford4.from_python_clifford(mv, blades)
python_clifford_blade_coeff(mv, blade_name, blades)

```
⸻

License / usage note

Add your preferred license here if you plan to publish or share the file.

If you want, I can also turn this into a shorter “practical README” version with fewer theory notes and more copy-paste examples.