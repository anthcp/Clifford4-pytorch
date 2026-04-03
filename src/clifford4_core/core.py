#!/usr/bin/env python3
"""
clifford4_core_v0_2_1.py
===============================================================================
Core Clifford-torch module for Cl(4,0) — v0.2.1
===============================================================================
"""

from __future__ import annotations

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
                sign = 1
                for bit in range(4):
                    if (a >> bit) & 1:
                        lower_bits_in_b = b & ((1 << bit) - 1)
                        if lower_bits_in_b.bit_count() % 2 == 1:
                            sign *= -1
                gp_index[i, j] = result_index
                gp_sign[i, j] = float(sign)
                wedge_keep[i, j] = ((a & b) == 0)

        grade_masks = []
        for g in range(5):
            grade_masks.append(torch.tensor(
                [1.0 if blade.grade == g else 0.0 for blade in cls.BASIS],
                dtype=dtype, device=device
            ))

        reverse_sign = torch.tensor(
            [(-1) ** (b.grade * (b.grade - 1) // 2) for b in cls.BASIS],
            dtype=dtype, device=device
        )
        grade_involution_sign = torch.tensor(
            [(-1) ** b.grade for b in cls.BASIS],
            dtype=dtype, device=device
        )
        clifford_conjugation_sign = torch.tensor(
            [(-1) ** (b.grade * (b.grade + 1) // 2) for b in cls.BASIS],
            dtype=dtype, device=device
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
            "gp_tensor": gp_tensor,
            "wedge_tensor": wedge_tensor,
            "inner_tensor": inner_tensor,
            "left_contr_tensor": left_contr_tensor,
            "grade_masks": torch.stack(grade_masks, dim=0),
            "reverse_sign": reverse_sign,
            "grade_involution_sign": grade_involution_sign,
            "clifford_conjugation_sign": clifford_conjugation_sign,
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
        data = torch.zeros(16, device=device, dtype=dtype)
        data[cls.BASIS_INDEX[name]] = coeff
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def from_coeff_dict(cls, coeffs: Dict[str, float], device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        for name, value in coeffs.items():
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

    def is_pure_grade(self, grade: int, atol: float = 1e-12) -> bool:
        others = self.data.clone()
        grade_mask = self._tables["grade_masks"][grade].bool()
        others[..., grade_mask] = 0.0
        return torch.max(torch.abs(others)).item() <= atol

    def almost_equal(self, other: "Clifford4", atol: float = 1e-10) -> bool:
        return torch.allclose(self.data, other.data, atol=atol, rtol=0.0)

    def _coerce_scalar(self, other):
        if isinstance(other, (int, float)):
            return Clifford4.scalar(float(other), device=self.device, dtype=self.dtype)
        return other

    def __add__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        return Clifford4(self.data + other.data, device=self.device, dtype=self.dtype)

    def __radd__(self, other) -> "Clifford4":
        return self.__add__(other)

    def __sub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
        return Clifford4(self.data - other.data, device=self.device, dtype=self.dtype)

    def __rsub__(self, other) -> "Clifford4":
        other = self._coerce_scalar(other)
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

    def grade(self, k: int) -> "Clifford4":
        return Clifford4(self.data * self._tables["grade_masks"][k], device=self.device, dtype=self.dtype)

    def reverse(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["reverse_sign"], device=self.device, dtype=self.dtype)

    def grade_involution(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["grade_involution_sign"], device=self.device, dtype=self.dtype)

    def clifford_conjugate(self) -> "Clifford4":
        return Clifford4(self.data * self._tables["clifford_conjugation_sign"], device=self.device, dtype=self.dtype)

    def _tensor_product(self, other: "Clifford4", tensor_name: str) -> "Clifford4":
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
        vv = v.gp(v).scalar_part()
        if torch.any(vv <= atol):
            raise ValueError("Cannot normalize a zero vector.")
        return Clifford4(v.data / torch.sqrt(vv).unsqueeze(-1), device=self.device, dtype=self.dtype)

    def normalize_rotor(self, atol: float = 1e-12) -> "Clifford4":
        rr = self.gp(self.reverse())
        nonscalar = rr.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() > atol:
            raise ValueError("normalize_rotor() expects scalar R~R.")
        s = rr.scalar_part()
        if torch.any(s <= atol):
            raise ValueError("Cannot normalize rotor with non-positive or zero R~R.")
        return Clifford4(self.data / torch.sqrt(s).unsqueeze(-1), device=self.device, dtype=self.dtype)

    def inverse_scalar(self, atol: float = 1e-12) -> "Clifford4":
        s = self.scalar_part()
        if torch.any(torch.abs(s) <= atol):
            raise ValueError("Scalar is not invertible.")
        out = torch.zeros_like(self.data)
        out[..., 0] = 1.0 / s
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def inverse_vector(self, atol: float = 1e-12) -> "Clifford4":
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
        basis_cols = torch.eye(16, device=self.device, dtype=self.dtype)
        if self.data.ndim == 1:
            basis_mv = Clifford4(basis_cols, device=self.device, dtype=self.dtype)
            prod = self.gp(basis_mv)
            return prod.data.transpose(0, 1)
        expand_shape = self.data.shape[:-1] + (16, 16)
        basis_data = basis_cols.expand(expand_shape)
        basis_mv = Clifford4(basis_data, device=self.device, dtype=self.dtype)
        prod = self.gp(basis_mv)
        return prod.data.transpose(-2, -1)

    def inverse_general(self, atol: float = 1e-12) -> "Clifford4":
        L = self.left_mul_matrix()
        rhs = torch.zeros(*L.shape[:-2], 16, device=self.device, dtype=self.dtype)
        rhs[..., 0] = 1.0
        det = torch.linalg.det(L)
        if torch.any(torch.abs(det) <= atol):
            raise ValueError("General multivector is not invertible or numerically singular.")
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
        rr = self.gp(self.reverse())
        nonscalar = rr.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() <= atol:
            return self.inverse_rotor(atol=atol)
        return self.inverse_general(atol=atol)

    def exp_simple_bivector(self, t: float = 1.0, atol: float = 1e-10) -> "Clifford4":
        B = self.grade(2)
        B_wedge_B = B.wedge(B)
        if torch.max(torch.abs(B_wedge_B.data)).item() > atol:
            raise ValueError("exp_simple_bivector() requires a simple bivector.")
        B2 = B.gp(B)
        nonscalar = B2.data.clone()
        nonscalar[..., 0] = 0.0
        if torch.max(torch.abs(nonscalar)).item() > atol:
            raise ValueError("exp_simple_bivector() requires scalar B^2.")
        b2_scalar = B2.scalar_part()
        if torch.any(b2_scalar > atol):
            raise ValueError("For Euclidean simple bivectors, B^2 should be non-positive scalar.")
        mag = torch.sqrt(torch.clamp(-b2_scalar, min=0.0))
        half_arg = 0.5 * t * mag
        cos_term = torch.cos(half_arg)
        sin_over_mag = torch.where(mag > atol, torch.sin(half_arg) / mag, 0.5 * t * torch.ones_like(mag))
        out = torch.zeros_like(self.data)
        out[..., 0] = cos_term
        out += sin_over_mag.unsqueeze(-1) * B.data
        return Clifford4(out, device=self.device, dtype=self.dtype)

    def exp_bivector_general(self, t: float = 1.0, atol: float = 1e-10) -> "Clifford4":
        B = self.grade(2)
        L = B.left_mul_matrix()
        E = torch.matrix_exp(0.5 * t * L)
        e0 = torch.zeros(16, device=self.device, dtype=self.dtype)
        e0[0] = 1.0
        coeffs = E @ e0 if E.ndim == 2 else torch.einsum("...ij,j->...i", E, e0)
        return Clifford4(coeffs, device=self.device, dtype=self.dtype).normalize_rotor(atol=atol)

    def sandwich(self, x: "Clifford4") -> "Clifford4":
        return self.gp(x).gp(self.reverse())

    def to_python_clifford(self, blades: Dict[str, object]):
        arr = self.data.detach().cpu().numpy()
        if arr.ndim != 1:
            raise ValueError("to_python_clifford() only supports unbatched multivectors.")
        mv = 0.0
        for i, blade in enumerate(self.BASIS):
            coeff = float(arr[i])
            if coeff == 0.0:
                continue
            mv = mv + coeff if blade.name == "1" else mv + coeff * blades[blade.name]
        return mv

    @classmethod
    def from_python_clifford(cls, mv, blades: Dict[str, object], device=None, dtype=torch.float64) -> "Clifford4":
        data = torch.zeros(16, device=device, dtype=dtype)
        if np.isscalar(mv):
            data[0] = float(mv)
            return cls(data, device=device, dtype=dtype)
        data[0] = float(mv.value[0])
        for i, blade in enumerate(cls.BASIS[1:], start=1):
            basis_mv = blades[blade.name]
            idx = int(np.argmax(np.abs(basis_mv.value)))
            data[i] = float(mv.value[idx])
        return cls(data, device=device, dtype=dtype)


def python_clifford_blade_coeff(mv, blade_name: str, blades: Dict[str, object]) -> float:
    if np.isscalar(mv):
        return float(mv) if blade_name == "1" else 0.0
    blade = blades[blade_name]
    idx = int(np.argmax(np.abs(blade.value)))
    return float(mv.value[idx])
