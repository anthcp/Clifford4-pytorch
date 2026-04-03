#!/usr/bin/env python3
"""
clifford_algebra_pytorch_cl4_v0.1.1.py
===============================================================================
PLVS Clifford-torch library for Cl(4,0) — cleaned v0.1.1

Changes from v0.1.0
-------------------
1. Fixed the geometric-product sign convention to match python-clifford.
   In particular:
       e1*e2 = +e12
       e2*e1 = -e12
2. Added named inner() and left_contraction() products.
3. Added batched random rotor tests.
4. Added to_python_clifford() / from_python_clifford() bridges for debugging.
5. Kept explicit basis-index maps, convention-safe tests, and documented
   simple-bivector rotor exponential.

Conventions
-----------
Basis order for coefficient storage:

    index  blade   bitmask  grade
    -----  ------  -------  -----
      0      1      0000      0
      1      e1     0001      1
      2      e2     0010      1
      3      e3     0100      1
      4      e4     1000      1
      5      e12    0011      2
      6      e13    0101      2
      7      e14    1001      2
      8      e23    0110      2
      9      e24    1010      2
     10      e34    1100      2
     11      e123   0111      3
     12      e124   1011      3
     13      e134   1101      3
     14      e234   1110      3
     15      e1234  1111      4

Metric: Cl(4,0), so e_i^2 = +1 for i = 1..4.

Rotor exponential
-----------------
The method exp_simple_bivector() is intentionally restricted to SIMPLE bivectors.
For a simple Euclidean bivector B with B^2 = -|B|^2, we use

    exp((t/2) B) = cos((t/2)|B|) + sin((t/2)|B|)/|B| * B

This is correct for a single rotation plane only. In 4D, a general bivector is
not necessarily simple, so this closed form is not used for arbitrary bivectors.

Versioning note
---------------
Filename and docstring include version number as requested.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class BasisBlade:
    name: str
    bitmask: int
    grade: int


class Clifford4:
    """
    Dense multivector representation for Cl(4,0) using a 16-component tensor.

    The last tensor axis stores the 16 basis-blade coefficients in the fixed
    order given by BASIS. Any leading axes are treated as batch dimensions.
    """

    BASIS: List[BasisBlade] = [
        BasisBlade("1",     0b0000, 0),
        BasisBlade("e1",    0b0001, 1),
        BasisBlade("e2",    0b0010, 1),
        BasisBlade("e3",    0b0100, 1),
        BasisBlade("e4",    0b1000, 1),
        BasisBlade("e12",   0b0011, 2),
        BasisBlade("e13",   0b0101, 2),
        BasisBlade("e14",   0b1001, 2),
        BasisBlade("e23",   0b0110, 2),
        BasisBlade("e24",   0b1010, 2),
        BasisBlade("e34",   0b1100, 2),
        BasisBlade("e123",  0b0111, 3),
        BasisBlade("e124",  0b1011, 3),
        BasisBlade("e134",  0b1101, 3),
        BasisBlade("e234",  0b1110, 3),
        BasisBlade("e1234", 0b1111, 4),
    ]

    BASIS_NAMES: List[str] = [b.name for b in BASIS]
    BASIS_INDEX: Dict[str, int] = {b.name: i for i, b in enumerate(BASIS)}
    BITMASK_TO_INDEX: Dict[int, int] = {b.bitmask: i for i, b in enumerate(BASIS)}

    _gp_index = None
    _gp_sign = None
    _wedge_keep = None
    _grade_masks = None
    _reverse_sign = None
    _grade_involution_sign = None
    _clifford_conjugation_sign = None
    _basis_bitmasks = None
    _basis_grades = None

    @classmethod
    def _build_tables(cls, device: torch.device, dtype: torch.dtype) -> None:
        if cls._gp_index is not None:
            return

        n = len(cls.BASIS)
        gp_index = torch.zeros((n, n), dtype=torch.long, device=device)
        gp_sign = torch.ones((n, n), dtype=dtype, device=device)
        wedge_keep = torch.zeros((n, n), dtype=torch.bool, device=device)

        for i, blade_a in enumerate(cls.BASIS):
            a = blade_a.bitmask
            for j, blade_b in enumerate(cls.BASIS):
                b = blade_b.bitmask

                result_mask = a ^ b
                result_index = cls.BITMASK_TO_INDEX[result_mask]

                # Correct convention matching python-clifford:
                # count how many basis vectors in a must move past lower-index
                # basis vectors in b.
                sign = 1
                for bit in range(4):
                    if (a >> bit) & 1:
                        lower_bits_in_b = b & ((1 << bit) - 1)
                        swaps = lower_bits_in_b.bit_count()
                        if swaps % 2 == 1:
                            sign *= -1

                gp_index[i, j] = result_index
                gp_sign[i, j] = float(sign)
                wedge_keep[i, j] = ((a & b) == 0)

        grade_masks = []
        for g in range(5):
            mask = torch.tensor(
                [1.0 if blade.grade == g else 0.0 for blade in cls.BASIS],
                dtype=dtype,
                device=device,
            )
            grade_masks.append(mask)

        reverse_sign = torch.tensor(
            [(-1) ** (b.grade * (b.grade - 1) // 2) for b in cls.BASIS],
            dtype=dtype,
            device=device,
        )
        grade_involution_sign = torch.tensor(
            [(-1) ** b.grade for b in cls.BASIS],
            dtype=dtype,
            device=device,
        )
        clifford_conjugation_sign = torch.tensor(
            [(-1) ** (b.grade * (b.grade + 1) // 2) for b in cls.BASIS],
            dtype=dtype,
            device=device,
        )

        cls._gp_index = gp_index
        cls._gp_sign = gp_sign
        cls._wedge_keep = wedge_keep
        cls._grade_masks = grade_masks
        cls._reverse_sign = reverse_sign
        cls._grade_involution_sign = grade_involution_sign
        cls._clifford_conjugation_sign = clifford_conjugation_sign
        cls._basis_bitmasks = torch.tensor([b.bitmask for b in cls.BASIS], dtype=torch.long, device=device)
        cls._basis_grades = torch.tensor([b.grade for b in cls.BASIS], dtype=torch.long, device=device)

    def __init__(self, data: torch.Tensor, device=None, dtype=torch.float64):
        if device is None:
            device = data.device
        self.device = torch.device(device)
        self.dtype = dtype
        Clifford4._build_tables(self.device, self.dtype)

        data = data.to(device=self.device, dtype=self.dtype)
        if data.shape[-1] != 16:
            raise ValueError(f"Expected last dimension 16, got {data.shape[-1]}")
        self.data = data

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------
    @classmethod
    def zeros(cls, batch_shape=(), device=None, dtype=torch.float64) -> "Clifford4":
        return cls(torch.zeros(*batch_shape, 16, device=device, dtype=dtype), device, dtype)

    @classmethod
    def scalar(cls, s: float, device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        data[0] = s
        return cls(data, device, dtype)

    @classmethod
    def vector(cls, coeffs: torch.Tensor, device=None, dtype=torch.float64) -> "Clifford4":
        if coeffs.shape[-1] != 4:
            raise ValueError("Vector constructor expects last dimension 4.")
        coeffs = coeffs.to(device=device, dtype=dtype)
        data = torch.zeros(*coeffs.shape[:-1], 16, device=coeffs.device, dtype=coeffs.dtype)
        data[..., 1:5] = coeffs
        return cls(data, coeffs.device, coeffs.dtype)

    @classmethod
    def basis_blade(cls, name: str, coeff: float = 1.0, device=None, dtype=torch.float64) -> "Clifford4":
        if name not in cls.BASIS_INDEX:
            raise KeyError(f"Unknown basis blade '{name}'")
        data = torch.zeros(16, device=device, dtype=dtype)
        data[cls.BASIS_INDEX[name]] = coeff
        return cls(data, device, dtype)

    @classmethod
    def from_coeff_dict(cls, coeffs: Dict[str, float], device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        for name, value in coeffs.items():
            if name not in cls.BASIS_INDEX:
                raise KeyError(f"Unknown basis blade '{name}'")
            data[cls.BASIS_INDEX[name]] = float(value)
        return cls(data, device, dtype)

    @classmethod
    def random_vector(cls, batch_shape=(), device=None, dtype=torch.float64, seed: Optional[int] = None) -> "Clifford4":
        if seed is not None:
            gen = torch.Generator(device=device or "cpu")
            gen.manual_seed(seed)
            coeffs = torch.randn(*batch_shape, 4, device=device, dtype=dtype, generator=gen)
        else:
            coeffs = torch.randn(*batch_shape, 4, device=device, dtype=dtype)
        return cls.vector(coeffs, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def clone(self) -> "Clifford4":
        return Clifford4(self.data.clone(), self.device, self.dtype)

    def coeff(self, blade_name: str) -> torch.Tensor:
        return self.data[..., self.BASIS_INDEX[blade_name]]

    def coeff_dict(self, atol: float = 0.0) -> Dict[str, float]:
        arr = self.data.detach().cpu().numpy()
        if arr.ndim != 1:
            raise ValueError("coeff_dict() only supports unbatched multivectors.")
        out = {}
        for i, name in enumerate(self.BASIS_NAMES):
            val = float(arr[i])
            if abs(val) > atol:
                out[name] = val
        return out

    def __repr__(self) -> str:
        try:
            pieces = [f"{v:+.6g}*{k}" for k, v in self.coeff_dict(atol=1e-14).items()]
            return "Clifford4(" + (" ".join(pieces) if pieces else "0") + ")"
        except Exception:
            return f"Clifford4(shape={tuple(self.data.shape)}, dtype={self.dtype}, device={self.device})"

    def scalar_part(self) -> torch.Tensor:
        return self.data[..., 0]

    def vector_part(self) -> "Clifford4":
        return self.grade(1)

    def bivector_part(self) -> "Clifford4":
        return self.grade(2)

    def is_pure_grade(self, grade: int, atol: float = 1e-12) -> bool:
        for g in range(5):
            if g == grade:
                continue
            if torch.max(torch.abs(self.grade(g).data)).item() > atol:
                return False
        return True

    def almost_equal(self, other: "Clifford4", atol: float = 1e-10) -> bool:
        return torch.allclose(self.data, other.data, atol=atol, rtol=0.0)

    # -------------------------------------------------------------------------
    # Arithmetic
    # -------------------------------------------------------------------------
    def _coerce_scalar(self, other):
        if isinstance(other, (int, float)):
            return Clifford4.scalar(float(other), device=self.device, dtype=self.dtype)
        return other

    def __add__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        if not isinstance(other, Clifford4):
            return NotImplemented
        return Clifford4(self.data + other.data, self.device, self.dtype)

    def __radd__(self, other) -> "Clifford4":
        return self.__add__(other)

    def __sub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        if not isinstance(other, Clifford4):
            return NotImplemented
        return Clifford4(self.data - other.data, self.device, self.dtype)

    def __rsub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        if not isinstance(other, Clifford4):
            return NotImplemented
        return Clifford4(other.data - self.data, self.device, self.dtype)

    def __neg__(self) -> "Clifford4":
        return Clifford4(-self.data, self.device, self.dtype)

    def __mul__(self, other) -> "Clifford4":
        if isinstance(other, (int, float)):
            return Clifford4(self.data * float(other), self.device, self.dtype)
        return NotImplemented

    def __rmul__(self, other) -> "Clifford4":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Clifford4":
        if isinstance(other, (int, float)):
            return Clifford4(self.data / float(other), self.device, self.dtype)
        return NotImplemented

    # -------------------------------------------------------------------------
    # Algebraic involutions and projections
    # -------------------------------------------------------------------------
    def grade(self, k: int) -> "Clifford4":
        if k < 0 or k > 4:
            raise ValueError("Grade must be in {0,1,2,3,4}.")
        return Clifford4(self.data * self._grade_masks[k], self.device, self.dtype)

    def reverse(self) -> "Clifford4":
        return Clifford4(self.data * self._reverse_sign, self.device, self.dtype)

    def grade_involution(self) -> "Clifford4":
        return Clifford4(self.data * self._grade_involution_sign, self.device, self.dtype)

    def clifford_conjugate(self) -> "Clifford4":
        return Clifford4(self.data * self._clifford_conjugation_sign, self.device, self.dtype)

    # -------------------------------------------------------------------------
    # Products
    # -------------------------------------------------------------------------
    def gp(self, other: "Clifford4") -> "Clifford4":
        """
        Geometric product.
        """
        if not isinstance(other, Clifford4):
            raise TypeError("gp expects a Clifford4 operand.")

        a = self.data
        b = other.data
        out = torch.zeros_like(a)

        for i in range(16):
            ai = a[..., i]
            if torch.all(ai == 0):
                continue
            for j in range(16):
                k = int(self._gp_index[i, j].item())
                s = self._gp_sign[i, j]
                out[..., k] += s * ai * b[..., j]

        return Clifford4(out, self.device, self.dtype)

    def wedge(self, other: "Clifford4") -> "Clifford4":
        """
        Exterior product.
        """
        if not isinstance(other, Clifford4):
            raise TypeError("wedge expects a Clifford4 operand.")

        a = self.data
        b = other.data
        out = torch.zeros_like(a)

        for i in range(16):
            ai = a[..., i]
            if torch.all(ai == 0):
                continue
            for j in range(16):
                if not bool(self._wedge_keep[i, j].item()):
                    continue
                k = int(self._gp_index[i, j].item())
                s = self._gp_sign[i, j]
                out[..., k] += s * ai * b[..., j]

        return Clifford4(out, self.device, self.dtype)

    def inner(self, other: "Clifford4") -> "Clifford4":
        """
        Hestenes inner product for homogeneous blade components:
            <A_r B_s>_{|r-s|}
        extended by bilinearity over all blade components.

        This is a named product for convenience and debugging.
        """
        if not isinstance(other, Clifford4):
            raise TypeError("inner expects a Clifford4 operand.")

        a = self.data
        b = other.data
        out = torch.zeros_like(a)

        for i in range(16):
            ai = a[..., i]
            if torch.all(ai == 0):
                continue
            ri = int(self._basis_grades[i].item())
            for j in range(16):
                rj = int(self._basis_grades[j].item())
                target_grade = abs(ri - rj)
                k = int(self._gp_index[i, j].item())
                rk = int(self._basis_grades[k].item())
                if rk != target_grade:
                    continue
                s = self._gp_sign[i, j]
                out[..., k] += s * ai * b[..., j]

        return Clifford4(out, self.device, self.dtype)

    def left_contraction(self, other: "Clifford4") -> "Clifford4":
        """
        Left contraction for homogeneous blade components:
            A_r ⨼ B_s = <A_r B_s>_{s-r}   if r <= s
                        0                 otherwise
        extended by bilinearity over all blade components.
        """
        if not isinstance(other, Clifford4):
            raise TypeError("left_contraction expects a Clifford4 operand.")

        a = self.data
        b = other.data
        out = torch.zeros_like(a)

        for i in range(16):
            ai = a[..., i]
            if torch.all(ai == 0):
                continue
            ri = int(self._basis_grades[i].item())
            for j in range(16):
                rj = int(self._basis_grades[j].item())
                if ri > rj:
                    continue
                target_grade = rj - ri
                k = int(self._gp_index[i, j].item())
                rk = int(self._basis_grades[k].item())
                if rk != target_grade:
                    continue
                s = self._gp_sign[i, j]
                out[..., k] += s * ai * b[..., j]

        return Clifford4(out, self.device, self.dtype)

    # -------------------------------------------------------------------------
    # Norms and rotor actions
    # -------------------------------------------------------------------------
    def norm_sq_via_reverse(self) -> torch.Tensor:
        return self.gp(self.reverse()).scalar_part()

    def exp_simple_bivector(self, t: float = 1.0, atol: float = 1e-10) -> "Clifford4":
        """
        Exponential for SIMPLE bivectors only.

        For a simple Euclidean bivector B with B^2 = -|B|^2, we use

            exp((t/2) B) = cos((t/2)|B|) + sin((t/2)|B|)/|B| * B

        Restrictions:
        - self must be pure grade-2
        - self must satisfy B ^ B = 0
        - B^2 must be scalar up to tolerance
        """
        B = self.grade(2)

        if not self.is_pure_grade(2, atol=atol):
            raise ValueError("exp_simple_bivector() requires a pure grade-2 multivector.")

        B_wedge_B = B.wedge(B)
        if torch.max(torch.abs(B_wedge_B.data)).item() > atol:
            raise ValueError("exp_simple_bivector() requires a simple bivector: detected nonzero B ^ B.")

        B2 = B.gp(B)
        nonscalar = B2.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() > atol:
            raise ValueError("exp_simple_bivector() requires B^2 to be scalar.")

        b2_scalar = B2.scalar_part()
        if torch.any(b2_scalar > atol):
            raise ValueError("For Euclidean simple bivectors, B^2 should be non-positive scalar.")

        mag = torch.sqrt(torch.clamp(-b2_scalar, min=0.0))
        half_arg = 0.5 * t * mag
        cos_term = torch.cos(half_arg)
        sin_over_mag = torch.where(
            mag > atol,
            torch.sin(half_arg) / mag,
            0.5 * t * torch.ones_like(mag),
        )

        out = torch.zeros_like(self.data)
        out[..., 0] = cos_term
        out += sin_over_mag.unsqueeze(-1) * B.data
        return Clifford4(out, self.device, self.dtype)

    def sandwich(self, x: "Clifford4") -> "Clifford4":
        return self.gp(x).gp(self.reverse())

    # -------------------------------------------------------------------------
    # python-clifford bridge
    # -------------------------------------------------------------------------
    def to_python_clifford(self, blades: Dict[str, object]):
        """
        Convert an unbatched Clifford4 multivector to a python-clifford multivector.

        Parameters
        ----------
        blades : dict
            The blades dictionary returned by:
                layout, blades = Cl(4, 0)

        Returns
        -------
        mv : python-clifford multivector
        """
        arr = self.data.detach().cpu().numpy()
        if arr.ndim != 1:
            raise ValueError("to_python_clifford() only supports unbatched multivectors.")

        mv = 0.0
        for i, blade in enumerate(self.BASIS):
            coeff = float(arr[i])
            if abs(coeff) == 0.0:
                continue
            if blade.name == "1":
                mv = mv + coeff
            else:
                mv = mv + coeff * blades[blade.name]
        return mv

    @classmethod
    def from_python_clifford(cls, mv, blades: Dict[str, object], device=None, dtype=torch.float64) -> "Clifford4":
        """
        Convert a python-clifford multivector to Clifford4 by direct coefficient extraction.
        """
        data = torch.zeros(16, device=device, dtype=dtype)

        # scalar
        data[0] = float(mv.value[0])

        for i, blade in enumerate(cls.BASIS[1:], start=1):
            basis_mv = blades[blade.name]
            idx = int(np.argmax(np.abs(basis_mv.value)))
            data[i] = float(mv.value[idx])

        return cls(data, device=device, dtype=dtype)


# =============================================================================
# python-clifford helpers
# =============================================================================

def python_clifford_blade_coeff(mv, blade_name: str, blades: Dict[str, object]) -> float:
    """
    Correct coefficient extraction for python-clifford multivectors.

    Do not use the inner product | for coefficient extraction.
    """
    blade = blades[blade_name]
    idx = int(np.argmax(np.abs(blade.value)))
    return float(mv.value[idx])


# =============================================================================
# Tests
# =============================================================================

def run_cl4_unit_tests() -> None:
    print("=== Running Cl(4,0) cleaned v0.1.1 unit tests ===")
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10

    one = Clifford4.scalar(1.0, device=device, dtype=dtype)
    e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
    e2 = Clifford4.basis_blade("e2", device=device, dtype=dtype)
    e3 = Clifford4.basis_blade("e3", device=device, dtype=dtype)
    e4 = Clifford4.basis_blade("e4", device=device, dtype=dtype)
    e12 = Clifford4.basis_blade("e12", device=device, dtype=dtype)
    e13 = Clifford4.basis_blade("e13", device=device, dtype=dtype)
    e23 = Clifford4.basis_blade("e23", device=device, dtype=dtype)
    e34 = Clifford4.basis_blade("e34", device=device, dtype=dtype)
    e123 = Clifford4.basis_blade("e123", device=device, dtype=dtype)
    I4 = Clifford4.basis_blade("e1234", device=device, dtype=dtype)

    assert Clifford4.BASIS_INDEX["1"] == 0
    assert Clifford4.BASIS_INDEX["e12"] == 5
    assert Clifford4.BITMASK_TO_INDEX[0b1111] == 15
    print("✓ Explicit basis-index maps pass")

    full = one + e1 + 2.0 * e12 + 3.0 * I4
    assert abs(full.grade(0).coeff("1").item() - 1.0) < tol
    assert abs(full.grade(1).coeff("e1").item() - 1.0) < tol
    assert abs(full.grade(2).coeff("e12").item() - 2.0) < tol
    assert abs(full.grade(4).coeff("e1234").item() - 3.0) < tol
    print("✓ Grade projectors pass")

    for ei in [e1, e2, e3, e4]:
        sq = ei.gp(ei)
        assert abs(sq.coeff("1").item() - 1.0) < tol
    print("✓ Metric signature tests pass")

    ab = e1.gp(e2)
    ba = e2.gp(e1)
    assert ab.almost_equal(-ba, atol=tol)
    assert abs(ab.coeff("e12").item() - 1.0) < tol
    assert abs(ba.coeff("e12").item() + 1.0) < tol
    print("✓ Anticommutation and sign convention tests pass")

    # startup sanity test for convention
    assert abs(Clifford4.basis_blade("e1", device=device, dtype=dtype)
               .gp(Clifford4.basis_blade("e2", device=device, dtype=dtype))
               .coeff("e12").item() - 1.0) < 1e-12
    print("✓ python-clifford-compatible basis convention sanity test passes")

    w12 = e1.wedge(e2)
    assert w12.almost_equal(ab, atol=tol)
    print("✓ Exterior product tests pass")

    assert e1.reverse().almost_equal(e1, atol=tol)
    assert e12.reverse().almost_equal(-e12, atol=tol)
    assert e123.reverse().almost_equal(-e123, atol=tol)
    assert I4.reverse().almost_equal(I4, atol=tol)
    print("✓ Reverse signs pass")

    lhs = e1.gp(e2.gp(e3))
    rhs = e1.gp(e2).gp(e3)
    assert lhs.almost_equal(rhs, atol=tol)
    print("✓ Associativity passes")

    lhs = (e1 + e2).gp(e3)
    rhs = e1.gp(e3) + e2.gp(e3)
    assert lhs.almost_equal(rhs, atol=tol)
    print("✓ Distributivity passes")

    # Named products
    inner_val = e1.inner(e1)
    assert abs(inner_val.coeff("1").item() - 1.0) < tol

    left_val = e1.left_contraction(e12)
    assert abs(left_val.coeff("e2").item() - 1.0) < tol
    print("✓ inner() and left_contraction() tests pass")

    # Simple vs nonsimple bivectors
    assert torch.max(torch.abs(e12.wedge(e12).data)).item() < tol
    nonsimple = e12 + e34
    assert torch.max(torch.abs(nonsimple.wedge(nonsimple).data)).item() > tol
    print("✓ Simple-vs-nonsimple bivector checks pass")

    # Rotor normalization
    R = e12.exp_simple_bivector(t=math.pi)
    RRr = R.gp(R.reverse())
    assert abs(RRr.coeff("1").item() - 1.0) < 1e-10
    print("✓ Rotor normalization passes")

    # Rotation in e12 plane: standard convention now matches python-clifford
    Rrot = (-e12).exp_simple_bivector(t=math.pi / 2.0, atol=tol)
    v_rot = Rrot.sandwich(e1)
    assert abs(v_rot.coeff("e2").item() - 1.0) < 1e-10
    assert abs(v_rot.coeff("e1").item()) < 1e-10
    print("✓ Simple bivector rotor action passes")

    rejected = False
    try:
        _ = nonsimple.exp_simple_bivector(t=1.0, atol=tol)
    except ValueError:
        rejected = True
    assert rejected
    print("✓ Nonsimple bivector rejection passes")

    print("=== All cleaned v0.1.1 unit tests passed ===")


def run_batched_random_rotor_tests(batch_size: int = 64, seed: int = 7) -> None:
    print(f"\n=== Running batched random rotor tests (batch_size={batch_size}) ===")
    dtype = torch.float64
    device = torch.device("cpu")
    tol = 1e-10

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    angles = torch.randn(batch_size, device=device, dtype=dtype, generator=gen)
    vecs = torch.randn(batch_size, 4, device=device, dtype=dtype, generator=gen)

    # Use a fixed simple bivector plane e12 for a clean batch test.
    B = Clifford4.basis_blade("e12", device=device, dtype=dtype)
    R = B.exp_simple_bivector(t=1.2345)  # same rotor for all; fine for norm preservation
    R_batch = Clifford4(R.data.unsqueeze(0).repeat(batch_size, 1), device=device, dtype=dtype)

    V = Clifford4.vector(vecs, device=device, dtype=dtype)
    V_rot = R_batch.sandwich(V)

    # Orthogonal rotor action should preserve v^2
    in_sq = V.gp(V).coeff("1")
    out_sq = V_rot.gp(V_rot).coeff("1")
    assert torch.allclose(in_sq, out_sq, atol=tol, rtol=0.0)
    print("✓ Batched rotor norm-preservation test passes")

    # Also test batch of angle-dependent rotors in the same plane
    out_data = torch.zeros(batch_size, 16, device=device, dtype=dtype)
    for n in range(batch_size):
        out_data[n] = B.exp_simple_bivector(t=float(angles[n].item())).data
    Rn = Clifford4(out_data, device=device, dtype=dtype)
    unit_check = Rn.gp(Rn.reverse()).coeff("1")
    assert torch.allclose(unit_check, torch.ones_like(unit_check), atol=1e-10, rtol=0.0)
    print("✓ Batched random rotor normalization test passes")

    print("=== All batched random rotor tests passed ===")


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
    print("✓ Correct python-clifford coefficient extraction passes")

    gp_t = e1_t.gp(e2_t)
    gp_c = e1_c * e2_c
    assert abs(gp_t.coeff("e12").item() - python_clifford_blade_coeff(gp_c, "e12", blades)) < tol
    print("✓ Geometric product matches python-clifford")

    rev_t = e12_t.reverse().coeff("e12").item()
    rev_c = python_clifford_blade_coeff(~e12_c, "e12", blades)
    assert abs(rev_t - rev_c) < tol
    print("✓ Reverse matches python-clifford")

    ns_t = e12_t + e34_t
    ns_c = e12_c + e34_c
    wedge_t = ns_t.wedge(ns_t).coeff("e1234").item()
    wedge_c = python_clifford_blade_coeff(ns_c ^ ns_c, "e1234", blades)
    assert abs(wedge_t - wedge_c) < tol
    print("✓ Wedge/simple-bivector detection matches python-clifford")

    # Bridge tests
    mv_t = Clifford4.from_coeff_dict({
        "1": 1.5, "e1": -2.0, "e23": 0.75, "e1234": -0.25
    }, device=device, dtype=dtype)
    mv_c = mv_t.to_python_clifford(blades)
    mv_back = Clifford4.from_python_clifford(mv_c, blades, device=device, dtype=dtype)
    assert mv_t.almost_equal(mv_back, atol=tol)
    print("✓ to_python_clifford()/from_python_clifford() bridge passes")

    print("=== All python-clifford cross-validation tests passed ===")


# =============================================================================
# Demo
# =============================================================================

def demo_rotor_triad() -> None:
    import matplotlib.pyplot as plt

    dtype = torch.float64
    device = torch.device("cpu")

    e1 = Clifford4.basis_blade("e1", device=device, dtype=dtype)
    B = -Clifford4.basis_blade("e12", device=device, dtype=dtype)

    angle = 2.0 * math.pi / 3.0
    R1 = B.exp_simple_bivector(t=angle)
    R2 = B.exp_simple_bivector(t=2.0 * angle)
    R3 = B.exp_simple_bivector(t=3.0 * angle)

    v1 = R1.sandwich(e1)
    v2 = R2.sandwich(e1)
    v3 = R3.sandwich(e1)

    def vec3(mv: Clifford4) -> np.ndarray:
        return np.array([
            mv.coeff("e1").item(),
            mv.coeff("e2").item(),
            mv.coeff("e3").item(),
        ])

    p1 = vec3(v1)
    p2 = vec3(v2)
    p3 = vec3(v3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(0, 0, 0, p1[0], p1[1], p1[2], label="R(120°) e1")
    ax.quiver(0, 0, 0, p2[0], p2[1], p2[2], label="R(240°) e1")
    ax.quiver(0, 0, 0, p3[0], p3[1], p3[2], label="R(360°) e1")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("e1")
    ax.set_ylabel("e2")
    ax.set_zlabel("e3")
    ax.set_title("Cl(4,0) simple-bivector rotor triad v0.1.1")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_cl4_unit_tests()
    run_batched_random_rotor_tests(batch_size=64, seed=7)
    run_python_clifford_cross_validation()
    demo_rotor_triad()
    print("\nCl(4,0) cleaned library v0.1.1 ready.")