#!/usr/bin/env python3
"""
clifford_algebra_pytorch_cl4_v0.2.1.py
===============================================================================
PLVS Clifford-torch library for Cl(4,0) — v0.2.1

New in v0.2.1
-------------
1. Keeps tensorized geometric/algebra products from v0.2.0.
2. Fixes general 4D bivector exponentials:
     - exp_simple_bivector() remains the strict fast path for simple bivectors
     - exp_bivector_general() now uses a robust matrix-exponential method
3. Keeps multivector inverses and normalization helpers.
4. Keeps GPU-friendly batched kernels based on torch.einsum.
5. Includes full cross-validation helpers against python-clifford.

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

Metric: Cl(4,0), so e_i^2 = +1.

Notes on general bivector exponentials
--------------------------------------
A general 4D bivector is not necessarily simple. The reliable implementation here
computes

    R = exp((t/2) B)

by building the left-multiplication matrix L_B and evaluating:

    exp((t/2) L_B) acting on the scalar basis vector 1.

This supports both simple and nonsimple bivectors, including cases like:
    e12 + e34

Versioning note
---------------
Filename and docstring include version number as requested.
===============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

    The last tensor axis stores basis coefficients. Any leading axes are treated
    as batch dimensions.
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

    # cache keyed by (device.type, str(device), str(dtype))
    _CACHE: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}

    @classmethod
    def _cache_key(cls, device: torch.device, dtype: torch.dtype) -> Tuple[str, str, str]:
        return (device.type, str(device), str(dtype))

    @classmethod
    def _build_tables(cls, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        key = cls._cache_key(device, dtype)
        if key in cls._CACHE:
            return cls._CACHE[key]

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

                # Convention matched to python-clifford:
                # e1*e2 = +e12, e2*e1 = -e12
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
        basis_grades = torch.tensor([b.grade for b in cls.BASIS], dtype=torch.long, device=device)

        gp_tensor = torch.zeros((n, n, n), dtype=dtype, device=device)
        wedge_tensor = torch.zeros((n, n, n), dtype=dtype, device=device)
        inner_tensor = torch.zeros((n, n, n), dtype=dtype, device=device)
        left_contr_tensor = torch.zeros((n, n, n), dtype=dtype, device=device)

        for i in range(n):
            ri = int(basis_grades[i].item())
            for j in range(n):
                rj = int(basis_grades[j].item())
                k = int(gp_index[i, j].item())
                rk = int(basis_grades[k].item())
                s = gp_sign[i, j]

                gp_tensor[k, i, j] = s
                if wedge_keep[i, j]:
                    wedge_tensor[k, i, j] = s
                if rk == abs(ri - rj):
                    inner_tensor[k, i, j] = s
                if ri <= rj and rk == (rj - ri):
                    left_contr_tensor[k, i, j] = s

        cache = {
            "gp_index": gp_index,
            "gp_sign": gp_sign,
            "gp_tensor": gp_tensor,
            "wedge_tensor": wedge_tensor,
            "inner_tensor": inner_tensor,
            "left_contr_tensor": left_contr_tensor,
            "grade_masks": torch.stack(grade_masks, dim=0),
            "reverse_sign": reverse_sign,
            "grade_involution_sign": grade_involution_sign,
            "clifford_conjugation_sign": clifford_conjugation_sign,
            "basis_grades": basis_grades,
        }
        cls._CACHE[key] = cache
        return cache

    def __init__(self, data: torch.Tensor, device=None, dtype=torch.float64):
        if device is None:
            device = data.device
        self.device = torch.device(device)
        self.dtype = dtype

        data = data.to(device=self.device, dtype=self.dtype)
        if data.shape[-1] != 16:
            raise ValueError(f"Expected last dimension 16, got {data.shape[-1]}")
        self.data = data
        self._tables = self._build_tables(self.device, self.dtype)

    # -------------------------------------------------------------------------
    # Constructors
    # -------------------------------------------------------------------------
    @classmethod
    def zeros(cls, batch_shape=(), device=None, dtype=torch.float64) -> "Clifford4":
        return cls(torch.zeros(*batch_shape, 16, device=device, dtype=dtype), device=device, dtype=dtype)

    @classmethod
    def scalar(cls, s: float, device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        data[0] = s
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def vector(cls, coeffs: torch.Tensor, device=None, dtype=torch.float64) -> "Clifford4":
        if coeffs.shape[-1] != 4:
            raise ValueError("Vector constructor expects last dimension 4.")
        coeffs = coeffs.to(device=device, dtype=dtype)
        data = torch.zeros(*coeffs.shape[:-1], 16, device=coeffs.device, dtype=coeffs.dtype)
        data[..., 1:5] = coeffs
        return cls(data, device=coeffs.device, dtype=coeffs.dtype)

    @classmethod
    def basis_blade(cls, name: str, coeff: float = 1.0, device=None, dtype=torch.float64) -> "Clifford4":
        if name not in cls.BASIS_INDEX:
            raise KeyError(f"Unknown basis blade '{name}'")
        data = torch.zeros(16, device=device, dtype=dtype)
        data[cls.BASIS_INDEX[name]] = coeff
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def from_coeff_dict(cls, coeffs: Dict[str, float], device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        for name, value in coeffs.items():
            if name not in cls.BASIS_INDEX:
                raise KeyError(f"Unknown basis blade '{name}'")
            data[cls.BASIS_INDEX[name]] = float(value)
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def random_vector(cls, batch_shape=(), device=None, dtype=torch.float64, seed: Optional[int] = None) -> "Clifford4":
        if seed is not None:
            gen = torch.Generator(device=device or "cpu")
            gen.manual_seed(seed)
            coeffs = torch.randn(*batch_shape, 4, device=device, dtype=dtype, generator=gen)
        else:
            coeffs = torch.randn(*batch_shape, 4, device=device, dtype=dtype)
        return cls.vector(coeffs, device=device, dtype=dtype)

    @classmethod
    def random_bivector(cls, batch_shape=(), device=None, dtype=torch.float64, seed: Optional[int] = None) -> "Clifford4":
        if seed is not None:
            gen = torch.Generator(device=device or "cpu")
            gen.manual_seed(seed)
            coeffs = torch.randn(*batch_shape, 6, device=device, dtype=dtype, generator=gen)
        else:
            coeffs = torch.randn(*batch_shape, 6, device=device, dtype=dtype)
        data = torch.zeros(*coeffs.shape[:-1], 16, device=coeffs.device, dtype=coeffs.dtype)
        data[..., 5:11] = coeffs
        return cls(data, device=coeffs.device, dtype=coeffs.dtype)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def clone(self) -> "Clifford4":
        return Clifford4(self.data.clone(), device=self.device, dtype=self.dtype)

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
            parts = [f"{v:+.6g}*{k}" for k, v in self.coeff_dict(atol=1e-14).items()]
            return "Clifford4(" + (" ".join(parts) if parts else "0") + ")"
        except Exception:
            return f"Clifford4(shape={tuple(self.data.shape)}, dtype={self.dtype}, device={self.device})"

    def scalar_part(self) -> torch.Tensor:
        return self.data[..., 0]

    def vector_part(self) -> "Clifford4":
        return self.grade(1)

    def bivector_part(self) -> "Clifford4":
        return self.grade(2)

    def trivector_part(self) -> "Clifford4":
        return self.grade(3)

    def pseudoscalar_part(self) -> "Clifford4":
        return self.grade(4)

    def is_pure_grade(self, grade: int, atol: float = 1e-12) -> bool:
        if not (0 <= grade <= 4):
            raise ValueError("Grade must be in {0,1,2,3,4}.")
        others = self.data.clone()
        grade_mask = self._tables["grade_masks"][grade].bool()
        others[..., grade_mask] = 0.0
        return torch.max(torch.abs(others)).item() <= atol

    def almost_equal(self, other: "Clifford4", atol: float = 1e-10) -> bool:
        return torch.allclose(self.data, other.data, atol=atol, rtol=0.0)

    def to(self, device=None, dtype=None) -> "Clifford4":
        device = self.device if device is None else torch.device(device)
        dtype = self.dtype if dtype is None else dtype
        return Clifford4(self.data.to(device=device, dtype=dtype), device=device, dtype=dtype)

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
        return Clifford4(self.data + other.data, device=self.device, dtype=self.dtype)

    def __radd__(self, other) -> "Clifford4":
        return self.__add__(other)

    def __sub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        if not isinstance(other, Clifford4):
            return NotImplemented
        return Clifford4(self.data - other.data, device=self.device, dtype=self.dtype)

    def __rsub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        if not isinstance(other, Clifford4):
            return NotImplemented
        return Clifford4(other.data - self.data, device=self.device, dtype=self.dtype)

    def __neg__(self) -> "Clifford4":
        return Clifford4(-self.data, device=self.device, dtype=self.dtype)

    def __mul__(self, other) -> "Clifford4":
        if isinstance(other, (int, float)):
            return Clifford4(self.data * float(other), device=self.device, dtype=self.dtype)
        return NotImplemented

    def __rmul__(self, other) -> "Clifford4":
        return self.__mul__(other)

    def __truediv__(self, other) -> "Clifford4":
        if isinstance(other, (int, float)):
            return Clifford4(self.data / float(other), device=self.device, dtype=self.dtype)
        return NotImplemented

    # -------------------------------------------------------------------------
    # Involutions / projections
    # -------------------------------------------------------------------------
    def grade(self, k: int) -> "Clifford4":
        if not (0 <= k <= 4):
            raise ValueError("Grade must be in {0,1,2,3,4}.")
        mask = self._tables["grade_masks"][k]
        return Clifford4(self.data * mask, device=self.device, dtype=self.dtype)

    def reverse(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["reverse_sign"], device=self.device, dtype=self.dtype)

    def grade_involution(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["grade_involution_sign"], device=self.device, dtype=self.dtype)

    def clifford_conjugate(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["clifford_conjugation_sign"], device=self.device, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Tensorized products
    # -------------------------------------------------------------------------
    def _tensor_product(self, other: "Clifford4", tensor_name: str) -> "Clifford4":
        if not isinstance(other, Clifford4):
            raise TypeError(f"{tensor_name} expects a Clifford4 operand.")
        T = self._tables[tensor_name]
        out = torch.einsum("...i,...j,kij->...k", self.data, other.data, T)
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def gp(self, other: "Clifford4") -> "Clifford4":
        return self._tensor_product(other, "gp_tensor")

    def wedge(self, other: "Clifford4") -> "Clifford4":
        return self._tensor_product(other, "wedge_tensor")

    def inner(self, other: "Clifford4") -> "Clifford4":
        return self._tensor_product(other, "inner_tensor")

    def left_contraction(self, other: "Clifford4") -> "Clifford4":
        return self._tensor_product(other, "left_contr_tensor")

    # -------------------------------------------------------------------------
    # Norms / helpers
    # -------------------------------------------------------------------------
    def norm_sq_via_reverse(self) -> torch.Tensor:
        return self.gp(self.reverse()).scalar_part()

    def scalar_norm_sq(self) -> torch.Tensor:
        return torch.sum(self.data * self.data, dim=-1)

    def reverse_norm_sq(self) -> torch.Tensor:
        return self.norm_sq_via_reverse()

    def normalize_coefficients(self, atol: float = 1e-12) -> "Clifford4":
        n = torch.sqrt(torch.clamp(self.scalar_norm_sq(), min=0.0))
        safe = torch.where(n > atol, n, torch.ones_like(n))
        out = self.data / safe.unsqueeze(-1)
        out = torch.where((n > atol).unsqueeze(-1), out, self.data)
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def normalize_vector(self, atol: float = 1e-12) -> "Clifford4":
        v = self.grade(1)
        if not self.is_pure_grade(1, atol=atol):
            raise ValueError("normalize_vector() requires a pure vector.")
        vv = v.gp(v).scalar_part()
        if torch.any(vv <= atol):
            raise ValueError("Cannot normalize a zero vector.")
        n = torch.sqrt(vv)
        return Clifford4(v.data / n.unsqueeze(-1), device=self.device, dtype=self.dtype)

    def normalize_rotor(self, atol: float = 1e-12) -> "Clifford4":
        rr = self.gp(self.reverse())
        nonscalar = rr.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() > atol:
            raise ValueError("normalize_rotor() expects an element with scalar R~R.")
        s = rr.scalar_part()
        if torch.any(s <= atol):
            raise ValueError("Cannot normalize rotor with non-positive or zero R~R.")
        return Clifford4(self.data / torch.sqrt(s).unsqueeze(-1), device=self.device, dtype=self.dtype)

    # -------------------------------------------------------------------------
    # Inverses
    # -------------------------------------------------------------------------
    def inverse_scalar(self, atol: float = 1e-12) -> "Clifford4":
        if not self.is_pure_grade(0, atol=atol):
            raise ValueError("inverse_scalar() requires a pure scalar.")
        s = self.scalar_part()
        if torch.any(torch.abs(s) <= atol):
            raise ValueError("Scalar is not invertible.")
        out = torch.zeros_like(self.data)
        out[..., 0] = 1.0 / s
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def inverse_vector(self, atol: float = 1e-12) -> "Clifford4":
        if not self.is_pure_grade(1, atol=atol):
            raise ValueError("inverse_vector() requires a pure vector.")
        vv = self.gp(self).scalar_part()
        if torch.any(torch.abs(vv) <= atol):
            raise ValueError("Vector is not invertible.")
        return Clifford4(self.data / vv.unsqueeze(-1), device=self.device, dtype=self.dtype)

    def inverse_rotor(self, atol: float = 1e-12) -> "Clifford4":
        rr = self.gp(self.reverse())
        nonscalar = rr.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() > atol:
            raise ValueError("inverse_rotor() expects scalar R~R.")
        s = rr.scalar_part()
        if torch.any(torch.abs(s) <= atol):
            raise ValueError("Rotor is not invertible.")
        return Clifford4(self.reverse().data / s.unsqueeze(-1), device=self.device, dtype=self.dtype)

    def left_mul_matrix(self) -> torch.Tensor:
        """
        Return matrix L(M) such that:
            vec(M * X) = L(M) @ vec(X)

        Returns
        -------
        torch.Tensor
            Shape:
                (16, 16)      for an unbatched multivector
                (..., 16, 16) for batched multivectors
        """
        basis_cols = torch.eye(16, device=self.device, dtype=self.dtype)

        if self.data.ndim == 1:
            basis_mv = Clifford4(basis_cols, device=self.device, dtype=self.dtype)
            prod = self.gp(basis_mv)          # (16,16), row j = M*e_j coefficients
            return prod.data.transpose(0, 1)  # columns = M*e_j

        expand_shape = self.data.shape[:-1] + (16, 16)
        basis_data = basis_cols.expand(expand_shape)
        basis_mv = Clifford4(basis_data, device=self.device, dtype=self.dtype)
        prod = self.gp(basis_mv)
        return prod.data.transpose(-2, -1)

    def inverse_general(self, atol: float = 1e-12) -> "Clifford4":
        """
        General inverse via left-multiplication matrix solve:
            M X = 1
        """
        L = self.left_mul_matrix()
        rhs = torch.zeros(*L.shape[:-2], 16, device=self.device, dtype=self.dtype)
        rhs[..., 0] = 1.0

        det = torch.linalg.det(L)
        if torch.any(torch.abs(det) <= atol):
            raise ValueError("General multivector is not invertible or is numerically singular.")

        x = torch.linalg.solve(L, rhs.unsqueeze(-1)).squeeze(-1)
        return Clifford4(x, device=self.device, dtype=self.dtype)

    def inverse(self, mode: str = "auto", atol: float = 1e-12) -> "Clifford4":
        if mode == "scalar":
            return self.inverse_scalar(atol=atol)
        if mode == "vector":
            return self.inverse_vector(atol=atol)
        if mode == "rotor":
            return self.inverse_rotor(atol=atol)
        if mode == "general":
            return self.inverse_general(atol=atol)
        if mode != "auto":
            raise ValueError("mode must be one of: auto, scalar, vector, rotor, general")

        try:
            if self.is_pure_grade(0, atol=atol):
                return self.inverse_scalar(atol=atol)
            if self.is_pure_grade(1, atol=atol):
                return self.inverse_vector(atol=atol)
            rr = self.gp(self.reverse())
            nonscalar = rr.data.clone()
            nonscalar[..., 0] = 0.0
            if torch.max(torch.abs(nonscalar)).item() <= atol:
                return self.inverse_rotor(atol=atol)
        except Exception:
            pass

        return self.inverse_general(atol=atol)

    # -------------------------------------------------------------------------
    # Rotor exponentials
    # -------------------------------------------------------------------------
    def exp_simple_bivector(self, t: float = 1.0, atol: float = 1e-10) -> "Clifford4":
        """
        Exponential for SIMPLE bivectors only:
            exp((t/2) B) = cos((t/2)|B|) + sin((t/2)|B|)/|B| * B
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
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def exp_bivector_general(self, t: float = 1.0, atol: float = 1e-10) -> "Clifford4":
        """
        Exponential for an arbitrary 4D bivector using the left-multiplication matrix.

        Computes:
            R = exp((t/2) B)

        by forming the 16x16 left-multiplication matrix L_B and evaluating

            exp((t/2) L_B) acting on the scalar basis vector 1.

        This is robust for both simple and nonsimple bivectors, including cases like:
            e12 + e34
        """
        B = self.grade(2)
        if not self.is_pure_grade(2, atol=atol):
            raise ValueError("exp_bivector_general() requires a pure grade-2 multivector.")

        L = B.left_mul_matrix()
        E = torch.matrix_exp(0.5 * t * L)

        e0 = torch.zeros(16, device=self.device, dtype=self.dtype)
        e0[0] = 1.0

        if E.ndim == 2:
            coeffs = E @ e0
        else:
            coeffs = torch.einsum("...ij,j->...i", E, e0)

        R = Clifford4(coeffs, device=self.device, dtype=self.dtype)
        return R.normalize_rotor(atol=atol)

    def sandwich(self, x: "Clifford4") -> "Clifford4":
        return self.gp(x).gp(self.reverse())

    # -------------------------------------------------------------------------
    # python-clifford bridge
    # -------------------------------------------------------------------------
    def to_python_clifford(self, blades: Dict[str, object]):
        arr = self.data.detach().cpu().numpy()
        if arr.ndim != 1:
            raise ValueError("to_python_clifford() only supports unbatched multivectors.")

        mv = 0.0
        for i, blade in enumerate(self.BASIS):
            coeff = float(arr[i])
            if coeff == 0.0:
                continue
            if blade.name == "1":
                mv = mv + coeff
            else:
                mv = mv + coeff * blades[blade.name]
        return mv

    @classmethod
    def from_python_clifford(cls, mv, blades: Dict[str, object], device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        data[0] = float(mv.value[0])

        for i, blade in enumerate(cls.BASIS[1:], start=1):
            basis_mv = blades[blade.name]
            idx = int(np.argmax(np.abs(basis_mv.value)))
            data[i] = float(mv.value[idx])

        return cls(data, device=device, dtype=dtype)


# =============================================================================
# python-clifford helper
# =============================================================================

def python_clifford_blade_coeff(mv, blade_name: str, blades: Dict[str, object]) -> float:
    blade = blades[blade_name]
    idx = int(np.argmax(np.abs(blade.value)))
    return float(mv.value[idx])


# =============================================================================
# Tests
# =============================================================================

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
    """
    Full cross-validation suite against python-clifford.
    """
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
        # Plain scalar case
        if np.isscalar(mv_cl):
            arr[0] = float(mv_cl)
            return arr
        # python-clifford multivector case
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
        # scalar shortcut
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

        # involutions via explicit coefficient signs
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
            term_gi = coeff_gi if name == "1" else coeff_gi * blades[name]
            term_cc = coeff_cc if name == "1" else coeff_cc * blades[name]
            gi_cl = gi_cl + term_gi
            cc_cl = cc_cl + term_cc

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
    ax.set_title("Cl(4,0) simple-bivector rotor triad v0.2.1")
    ax.legend()
    plt.tight_layout()
    plt.show()


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
    demo_rotor_triad()
    print("\nCl(4,0) v0.2.1 ready.")