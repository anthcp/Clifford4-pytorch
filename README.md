# clifford4-core

A PyTorch-based Clifford algebra library for **Cl(4,0)** with:

- tensorized geometric, exterior, inner, and left-contraction products
- simple and general bivector exponentials
- scalar, vector, rotor, and general inverse helpers
- batched multivector support
- optional comparison and cross-validation against `python-clifford`

This package is organized for editable installation with `pip install -e .`.

---

## Installation

### Editable install

```bash
pip install -e .
```

### Optional development dependencies

```bash
pip install -e ".[dev]"
```

The `dev` extras include:

- `clifford`
- `matplotlib`
- `pytest`

---

## Package layout

```text
clifford4_package/
├── pyproject.toml
├── README.md
├── benchmark_clifford4.py
├── src/
│   └── clifford4_core/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_clifford4.py
```

---

## Import

After installation:

```python
from clifford4_core import Clifford4, python_clifford_blade_coeff
```

---

## Basis and conventions

This library stores coefficients in the following fixed order:

| Index | Blade | Grade |
|---:|---|---:|
| 0 | 1 | 0 |
| 1 | e1 | 1 |
| 2 | e2 | 1 |
| 3 | e3 | 1 |
| 4 | e4 | 1 |
| 5 | e12 | 2 |
| 6 | e13 | 2 |
| 7 | e14 | 2 |
| 8 | e23 | 2 |
| 9 | e24 | 2 |
| 10 | e34 | 2 |
| 11 | e123 | 3 |
| 12 | e124 | 3 |
| 13 | e134 | 3 |
| 14 | e234 | 3 |
| 15 | e1234 | 4 |

The multiplication convention is aligned with `python-clifford`:

```text
e1 * e2 = +e12
e2 * e1 = -e12
e12 * e12 = -1
```

---

## Quick start

```python
import math
import torch
from clifford4_core import Clifford4

dtype = torch.float64
device = torch.device("cpu")

e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
e2 = Clifford4.basis_blade("e2", device=device, dtype=dtype)
e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)

ab = e1.gp(e2)
print(ab.coeff_dict())   # {'e12': 1.0}

R = (-e12).exp_simple_bivector(t=math.pi / 2.0)
v_rot = R.sandwich(e1)
print(v_rot.coeff_dict())  # approximately {'e2': 1.0}
```

---

## Main features

### Construct multivectors

```python
one = Clifford4.scalar(1.0, device=device, dtype=dtype)

v = Clifford4.vector(
    torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=dtype),
    device=device,
    dtype=dtype,
)

mv = Clifford4.from_coeff_dict({
    "1": 1.0,
    "e1": 2.0,
    "e23": -0.5,
    "e1234": 0.25,
}, device=device, dtype=dtype)
```

### Products

```python
gp_val = e1.gp(e2)
wedge_val = e1.wedge(e2)
inner_val = e1.inner(e1)
left_val = e1.left_contraction(e12)
```

### Involutions and projections

```python
scalar_part = mv.grade(0)
vector_part = mv.grade(1)
bivector_part = mv.grade(2)

rev = mv.reverse()
gi = mv.grade_involution()
cc = mv.clifford_conjugate()
```

### Norms and normalization

```python
n_rev = mv.norm_sq_via_reverse()
n_coeff = mv.scalar_norm_sq()

v_unit = v.normalize_vector()
R_unit = R.normalize_rotor()
```

### Inverses

```python
s_inv = Clifford4.scalar(2.0).inverse_scalar()
v_inv = v.inverse_vector()
R_inv = R.inverse_rotor()
m_inv = mv.inverse_general()

auto_inv = mv.inverse(mode="auto")
```

### Rotor exponentials

`exp_simple_bivector()` is for **simple** bivectors only.

```python
R_simple = (-e12).exp_simple_bivector(t=0.7)
```

`exp_bivector_general()` supports arbitrary 4D bivectors using the left-multiplication matrix exponential:

```python
B_general = Clifford4.from_coeff_dict({
    "e12": 1.0,
    "e34": 0.5,
}, device=device, dtype=dtype)

R_general = B_general.exp_bivector_general(t=0.7)
```

---

## Batched usage

Leading dimensions are treated as batch dimensions.

```python
vecs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 4.0],
], dtype=dtype)

V = Clifford4.vector(vecs, device=device, dtype=dtype)

B = Clifford4.basis_blade("e12", coeff=-1.0, device=device, dtype=dtype)
R = B.exp_simple_bivector(t=math.pi / 3.0)
R_batch = Clifford4(R.data.unsqueeze(0).repeat(3, 1), device=device, dtype=dtype)

V_rot = R_batch.sandwich(V)
print(V_rot.data.shape)   # (3, 16)
```

---

## Python-clifford interoperability

### Convert to python-clifford

```python
from clifford import Cl
layout, blades = Cl(4, 0)

mv_cl = mv.to_python_clifford(blades)
```

### Convert from python-clifford

```python
mv_back = Clifford4.from_python_clifford(mv_cl, blades, device=device, dtype=dtype)
```

### Safe coefficient extraction

```python
coeff = python_clifford_blade_coeff(mv_cl, "e23", blades)
```

---

## Testing

Run the test script from the package root:

```bash
python tests/test_clifford4.py
```

Or with pytest:

```bash
pytest
```

The test suite includes:

- unit tests
- batched rotor tests
- spot checks against `python-clifford`
- broader randomized cross-validation

---

## Benchmarks

Run:

```bash
python benchmark_clifford4.py
```

The benchmark script includes:

- Clifford4 kernel timings
- geometric-product throughput
- CPU comparison against `python-clifford`

---

## Minimal API summary

### Constructors

```python
Clifford4.zeros(...)
Clifford4.scalar(...)
Clifford4.vector(...)
Clifford4.basis_blade(...)
Clifford4.from_coeff_dict(...)
Clifford4.random_vector(...)
Clifford4.random_bivector(...)
```

### Core methods

```python
mv.coeff(...)
mv.coeff_dict(...)
mv.grade(...)
mv.reverse()
mv.grade_involution()
mv.clifford_conjugate()

a.gp(b)
a.wedge(b)
a.inner(b)
a.left_contraction(b)

mv.norm_sq_via_reverse()
mv.scalar_norm_sq()
mv.reverse_norm_sq()

mv.normalize_coefficients()
mv.normalize_vector()
mv.normalize_rotor()

mv.inverse_scalar()
mv.inverse_vector()
mv.inverse_rotor()
mv.inverse_general()
mv.inverse(mode="auto")

B.exp_simple_bivector(...)
B.exp_bivector_general(...)
R.sandwich(x)
```

### Python-clifford helpers

```python
mv.to_python_clifford(blades)
Clifford4.from_python_clifford(mv, blades)
python_clifford_blade_coeff(mv, blade_name, blades)
```

---

## Notes

- `exp_simple_bivector()` requires a simple bivector.
- `exp_bivector_general()` is the robust path for arbitrary 4D bivectors.
- `inverse_general()` uses a left-multiplication matrix solve, so it is more general but heavier than the specialized scalar/vector/rotor inverses.
- Batched products are tensorized with `torch.einsum`.

---

## License

The package metadata currently marks this as MIT. Adjust as needed for your project.
