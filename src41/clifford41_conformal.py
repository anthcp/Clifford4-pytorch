"""
clifford41_conformal.py
Versor-inspired Cl(4,1) PyTorch implementation.

Upgraded for grammar-molecule work:
- robust scalar/basis constructors with batch tensor support
- inverse + normalization helpers
- bivector exponential with closed forms for simple bivectors
- generic exponential fallback via scaling/squaring + power series
- rotor logarithm for simple rotors plus small-angle series fallback
- exp_bivector() / log_rotor() aliases
"""

from __future__ import annotations
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BasisBlade:
    name: str
    bitmask: int
    grade: int


class Clifford41:
    """Conformal Geometric Algebra Cl(4,1) — 32-dimensional multivectors."""

    BASIS: List[BasisBlade] = [
        BasisBlade("1", 0b00000, 0),
        BasisBlade("e1", 0b00001, 1),
        BasisBlade("e2", 0b00010, 1),
        BasisBlade("e3", 0b00100, 1),
        BasisBlade("e4", 0b01000, 1),
        BasisBlade("e5", 0b10000, 1),
        BasisBlade("e12", 0b00011, 2), BasisBlade("e13", 0b00101, 2),
        BasisBlade("e14", 0b01001, 2), BasisBlade("e15", 0b10001, 2),
        BasisBlade("e23", 0b00110, 2), BasisBlade("e24", 0b01010, 2),
        BasisBlade("e25", 0b10010, 2), BasisBlade("e34", 0b01100, 2),
        BasisBlade("e35", 0b10100, 2), BasisBlade("e45", 0b11000, 2),
        BasisBlade("e123", 0b00111, 3), BasisBlade("e124", 0b01011, 3),
        BasisBlade("e125", 0b10011, 3), BasisBlade("e134", 0b01101, 3),
        BasisBlade("e135", 0b10101, 3), BasisBlade("e145", 0b11001, 3),
        BasisBlade("e234", 0b01110, 3), BasisBlade("e235", 0b10110, 3),
        BasisBlade("e245", 0b11010, 3), BasisBlade("e345", 0b11100, 3),
        BasisBlade("e1234", 0b01111, 4), BasisBlade("e1235", 0b10111, 4),
        BasisBlade("e1245", 0b11011, 4), BasisBlade("e1345", 0b11101, 4),
        BasisBlade("e2345", 0b11110, 4),
        BasisBlade("I", 0b11111, 5),
    ]

    BASIS_NAMES = [b.name for b in BASIS]
    BASIS_INDEX: Dict[str, int] = {b.name: i for i, b in enumerate(BASIS)}
    BITMASK_TO_INDEX: Dict[int, int] = {b.bitmask: i for i, b in enumerate(BASIS)}
    _CACHE: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = {}

    @classmethod
    def _build_tables(cls, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        key = (device.type, str(device), str(dtype))
        if key in cls._CACHE:
            return cls._CACHE[key]

        n = len(cls.BASIS)
        gp_index = torch.zeros((n, n), dtype=torch.long, device=device)
        gp_sign = torch.ones((n, n), dtype=dtype, device=device)
        metric = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0], dtype=dtype, device=device)

        for i, a in enumerate(cls.BASIS):
            mask_a = a.bitmask
            for j, b in enumerate(cls.BASIS):
                mask_b = b.bitmask
                result_mask = mask_a ^ mask_b
                result_idx = cls.BITMASK_TO_INDEX[result_mask]

                sign = 1
                for bit in range(5):
                    if (mask_a >> bit) & 1:
                        lower = mask_b & ((1 << bit) - 1)
                        if bin(lower).count("1") % 2 == 1:
                            sign = -sign
                overlap = mask_a & mask_b
                for bit in range(5):
                    if (overlap >> bit) & 1:
                        sign *= int(metric[bit].item())
                gp_index[i, j] = result_idx
                gp_sign[i, j] = float(sign)

        grade_masks = []
        for g in range(6):
            mask = torch.tensor([1.0 if b.grade == g else 0.0 for b in cls.BASIS], dtype=dtype, device=device)
            grade_masks.append(mask)
        reverse_sign = torch.tensor([(-1) ** (b.grade * (b.grade - 1) // 2) for b in cls.BASIS], dtype=dtype, device=device)

        cache = {
            "gp_index": gp_index,
            "gp_sign": gp_sign,
            "grade_masks": torch.stack(grade_masks),
            "reverse_sign": reverse_sign,
        }
        cls._CACHE[key] = cache
        return cache

    def __init__(self, data: torch.Tensor, device: Optional[torch.device] = None, dtype=torch.float64):
        if device is None:
            device = data.device
        self.device = torch.device(device)
        self.dtype = dtype
        data = data.to(device=self.device, dtype=self.dtype)
        if data.shape[-1] != 32:
            raise ValueError(f"Cl(4,1) expects last dimension 32, got {data.shape[-1]}")
        self.data = data
        self._tables = self._build_tables(self.device, self.dtype)

    @classmethod
    def zeros(cls, batch_shape=(), device=None, dtype=torch.float64):
        return cls(torch.zeros(*batch_shape, 32, device=device, dtype=dtype), device=device, dtype=dtype)

    @classmethod
    def scalar(cls, value, device=None, dtype=torch.float64):
        if torch.is_tensor(value):
            value = value.to(device=device if device is not None else value.device,
                             dtype=dtype if dtype is not None else value.dtype)
            data = torch.zeros(*value.shape, 32, device=value.device, dtype=value.dtype)
            data[..., 0] = value
            return cls(data, device=value.device, dtype=value.dtype)
        data = torch.zeros(32, device=device, dtype=dtype)
        data[0] = value
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def vector(cls, coeffs: torch.Tensor, device=None, dtype=torch.float64):
        if coeffs.shape[-1] != 5:
            raise ValueError("vector expects 5 coefficients")
        coeffs = coeffs.to(device=device if device is not None else coeffs.device,
                           dtype=dtype if dtype is not None else coeffs.dtype)
        data = torch.zeros(*coeffs.shape[:-1], 32, device=coeffs.device, dtype=coeffs.dtype)
        data[..., 1:6] = coeffs
        return cls(data, device=coeffs.device, dtype=coeffs.dtype)

    @classmethod
    def basis(cls, name: str, coeff=1.0, device=None, dtype=torch.float64):
        idx = cls.BASIS_INDEX[name]
        if torch.is_tensor(coeff):
            coeff = coeff.to(device=device if device is not None else coeff.device,
                             dtype=dtype if dtype is not None else coeff.dtype)
            data = torch.zeros(*coeff.shape, 32, device=coeff.device, dtype=coeff.dtype)
            data[..., idx] = coeff
            return cls(data, device=coeff.device, dtype=coeff.dtype)
        data = torch.zeros(32, device=device, dtype=dtype)
        data[idx] = coeff
        return cls(data, device=device, dtype=dtype)

    @classmethod
    def e_o(cls, device=None, dtype=torch.float64):
        return cls.basis("e5", 0.5, device, dtype) - cls.basis("e4", 0.5, device, dtype)

    @classmethod
    def e_inf(cls, device=None, dtype=torch.float64):
        return cls.basis("e4", 1.0, device, dtype) + cls.basis("e5", 1.0, device, dtype)

    @classmethod
    def conformal_lift(cls, x: torch.Tensor, device=None, dtype=torch.float64):
        if x.shape[-1] != 3:
            raise ValueError("conformal_lift expects (..., 3)")
        x = x.to(device=device, dtype=dtype)
        r2 = torch.sum(x ** 2, dim=-1, keepdim=True)
        data = torch.zeros(*x.shape[:-1], 32, device=x.device, dtype=x.dtype)
        data[..., 1:4] = x
        data[..., 4] = -0.5 + 0.5 * r2.squeeze(-1)
        data[..., 5] = +0.5 + 0.5 * r2.squeeze(-1)
        return cls(data, device=x.device, dtype=x.dtype)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = self.scalar(other, device=self.device, dtype=self.dtype)
        if isinstance(other, Clifford41):
            return Clifford41(self.data + other.data, device=self.device, dtype=self.dtype)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = self.scalar(other, device=self.device, dtype=self.dtype)
        if isinstance(other, Clifford41):
            return Clifford41(self.data - other.data, device=self.device, dtype=self.dtype)
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = self.scalar(other, device=self.device, dtype=self.dtype)
        if isinstance(other, Clifford41):
            return Clifford41(other.data - self.data, device=self.device, dtype=self.dtype)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Clifford41(self.data * other, device=self.device, dtype=self.dtype)
        if torch.is_tensor(other):
            return Clifford41(self.data * other.unsqueeze(-1), device=self.device, dtype=self.dtype)
        if isinstance(other, Clifford41):
            return self.gp(other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Clifford41(self.data / other, device=self.device, dtype=self.dtype)
        if torch.is_tensor(other):
            return Clifford41(self.data / other.unsqueeze(-1), device=self.device, dtype=self.dtype)
        return NotImplemented

    def gp(self, other: "Clifford41") -> "Clifford41":
        tables = self._tables
        idx = tables["gp_index"]
        sgn = tables["gp_sign"]
        batch_shape = torch.broadcast_shapes(self.data.shape[:-1], other.data.shape[:-1])
        a = self.data.expand(*batch_shape, -1)
        b = other.data.expand(*batch_shape, -1)
        result = torch.zeros(*batch_shape, 32, device=self.device, dtype=self.dtype)
        for i in range(32):
            for j in range(32):
                k = idx[i, j].item()
                result[..., k] += sgn[i, j] * a[..., i] * b[..., j]
        return Clifford41(result, device=self.device, dtype=self.dtype)

    def wedge(self, other: "Clifford41") -> "Clifford41":
        return (self.gp(other) - other.gp(self)) * 0.5

    def inner(self, other: "Clifford41") -> "Clifford41":
        return (self.gp(other) + other.gp(self)) * 0.5

    def left_contract(self, other: "Clifford41") -> "Clifford41":
        return self.inner(other)

    def right_contract(self, other: "Clifford41") -> "Clifford41":
        return other.inner(self)

    def conjugate(self) -> "Clifford41":
        signs = torch.tensor([(-1) ** g for g in range(6)], dtype=self.dtype, device=self.device)
        grade_index = self._tables["grade_masks"].argmax(dim=0)
        return Clifford41(self.data * signs[grade_index], device=self.device, dtype=self.dtype)

    def dual(self) -> "Clifford41":
        I = Clifford41.basis("I", device=self.device, dtype=self.dtype)
        return self.gp(-1.0 * I)

    def undual(self) -> "Clifford41":
        I = Clifford41.basis("I", device=self.device, dtype=self.dtype)
        return self.gp(I)

    def commutator(self, other: "Clifford41") -> "Clifford41":
        return (self.gp(other) - other.gp(self)) * 0.5

    def reverse(self):
        return Clifford41(self.data * self._tables["reverse_sign"], device=self.device, dtype=self.dtype)

    def grade(self, g: int) -> "Clifford41":
        return Clifford41(self.data * self._tables["grade_masks"][g], device=self.device, dtype=self.dtype)

    def norm2(self):
        return self.gp(self.reverse()).data[..., 0]

    def norm(self):
        return torch.sqrt(self.norm2().abs())

    def scalar_part(self) -> torch.Tensor:
        return self.data[..., 0]

    def even(self) -> "Clifford41":
        return self.grade(0) + self.grade(2) + self.grade(4)

    def bivector_part(self) -> "Clifford41":
        return self.grade(2)

    def sandwich(self, x: "Clifford41"):
        return self.gp(x).gp(self.reverse())

    def inverse(self) -> "Clifford41":
        n2 = self.norm2()
        n2 = torch.where(n2.abs() > 1e-12, n2, torch.ones_like(n2))
        return self.reverse() / n2

    def normalize(self) -> "Clifford41":
        n = self.norm()
        n = torch.where(n > 1e-12, n, torch.ones_like(n))
        return Clifford41(self.data / n.unsqueeze(-1), device=self.device, dtype=self.dtype)

    @classmethod
    def _identity_like(cls, A: "Clifford41") -> "Clifford41":
        return cls.scalar(torch.ones(A.data.shape[:-1], device=A.device, dtype=A.dtype), device=A.device, dtype=A.dtype)

    def _is_pure_grade(self, g: int, tol: float = 1e-12) -> bool:
        remainder = (self - self.grade(g)).data.abs().amax(dim=-1)
        return bool(torch.all(remainder < tol))

    def _square_is_scalar(self, tol: float = 1e-10) -> bool:
        sq = self.gp(self)
        remainder = (sq - sq.grade(0)).data.abs().amax(dim=-1)
        return bool(torch.all(remainder < tol))

    @classmethod
    def _exp_series(cls, A: "Clifford41", terms: int = 24) -> "Clifford41":
        batch_norm = torch.linalg.vector_norm(A.data, dim=-1)
        max_norm = float(batch_norm.max().detach().cpu().item()) if batch_norm.numel() else 0.0
        scale_steps = 0 if max_norm <= 1.0 else max(0, int(torch.ceil(torch.log2(torch.tensor(max_norm))).item()))
        scaled = A / float(2 ** scale_steps)
        result = cls._identity_like(A)
        term = cls._identity_like(A)
        for k in range(1, terms + 1):
            term = term.gp(scaled) / float(k)
            result = result + term
        for _ in range(scale_steps):
            result = result.gp(result)
        return result

    @classmethod
    def exp(cls, A: "Clifford41", device=None, dtype=torch.float64):
        """General multivector exponential.

        Fast path: simple bivector B with scalar B².
        Fallback: scaling/squaring power series.
        """
        if not isinstance(A, Clifford41):
            raise TypeError("exp() expects a Clifford41 multivector")

        if A._is_pure_grade(2) and A._square_is_scalar():
            sigma = A.gp(A).scalar_part()
            pos = sigma > 1e-12
            neg = sigma < -1e-12
            zero = ~(pos | neg)

            alpha = torch.sqrt(torch.clamp(sigma, min=0.0))
            beta = torch.sqrt(torch.clamp(-sigma, min=0.0))

            coeff_scalar = torch.zeros_like(sigma)
            coeff_b = torch.zeros_like(sigma)

            coeff_scalar = torch.where(pos, torch.cosh(alpha), coeff_scalar)
            coeff_b = torch.where(pos, torch.sinh(alpha) / torch.where(alpha > 1e-12, alpha, torch.ones_like(alpha)), coeff_b)

            coeff_scalar = torch.where(neg, torch.cos(beta), coeff_scalar)
            coeff_b = torch.where(neg, torch.sin(beta) / torch.where(beta > 1e-12, beta, torch.ones_like(beta)), coeff_b)

            coeff_scalar = torch.where(zero, torch.ones_like(sigma), coeff_scalar)
            coeff_b = torch.where(zero, torch.ones_like(sigma), coeff_b)

            return cls.scalar(coeff_scalar, device=A.device, dtype=A.dtype) + A * coeff_b

        return cls._exp_series(A)

    @classmethod
    def exp_bivector(cls, B: "Clifford41", device=None, dtype=torch.float64):
        return cls.exp(B, device=device, dtype=dtype)

    @classmethod
    def log(cls, R: "Clifford41", device=None, dtype=torch.float64):
        """Rotor logarithm.

        Fast path: normalized simple rotor with only grades 0 and 2.
        Fallback: Mercator series around the identity.
        """
        if not isinstance(R, Clifford41):
            raise TypeError("log() expects a Clifford41 multivector")

        Rn = R.normalize()
        biv = Rn.grade(2)
        scalar = Rn.scalar_part()
        remainder = (Rn - Rn.grade(0) - biv).data.abs().amax(dim=-1)
        bb = biv.gp(biv)
        bb_non_scalar = (bb - bb.grade(0)).data.abs().amax(dim=-1)

        if bool(torch.all(remainder < 1e-10)) and bool(torch.all(bb_non_scalar < 1e-8)):
            sigma = bb.scalar_part()
            neg = sigma < -1e-12
            pos = sigma > 1e-12
            zero = ~(neg | pos)

            mag = torch.sqrt(torch.clamp(sigma.abs(), min=0.0))
            unit = biv / torch.where(mag > 1e-12, mag, torch.ones_like(mag))

            s_clamped = torch.clamp(scalar, -1.0, 1.0)
            ang_elliptic = torch.atan2(mag, s_clamped)

            ratio = mag / torch.where(scalar.abs() > 1e-12, scalar.abs(), torch.ones_like(scalar))
            ratio = torch.clamp(ratio, max=1.0 - 1e-12)
            ang_hyper = torch.atanh(ratio) * torch.sign(torch.where(scalar.abs() > 1e-12, scalar, torch.ones_like(scalar)))

            angle = torch.zeros_like(scalar)
            angle = torch.where(neg, ang_elliptic, angle)
            angle = torch.where(pos, ang_hyper, angle)
            angle = torch.where(zero, mag, angle)
            return unit * angle

        X = Rn - cls._identity_like(Rn)
        result = X
        term = X
        for k in range(2, 21):
            term = term.gp(X)
            result = result + term * (((-1) ** (k + 1)) / float(k))
        return result.even().grade(2)

    @classmethod
    def log_rotor(cls, R: "Clifford41", device=None, dtype=torch.float64):
        return cls.log(R, device=device, dtype=dtype)

    @classmethod
    def translator(cls, t: torch.Tensor, device=None, dtype=torch.float64):
        if t.shape[-1] != 3:
            raise ValueError("translator expects (..., 3)")
        t = t.to(device=device, dtype=dtype)
        coeffs = torch.zeros(*t.shape[:-1], 5, device=t.device, dtype=t.dtype)
        coeffs[..., :3] = t
        euclid_t = cls.vector(coeffs, device=t.device, dtype=t.dtype)
        e_inf = cls.e_inf(device=t.device, dtype=t.dtype)
        return cls.scalar(torch.ones(t.shape[:-1], device=t.device, dtype=t.dtype), t.device, t.dtype) - (euclid_t.gp(e_inf) * 0.5)

    @classmethod
    def motor(cls, rotor: "Clifford41", translator: "Clifford41"):
        return rotor.gp(translator)

    def apply_motor(self, M: "Clifford41"):
        return M.sandwich(self)

    @classmethod
    def rotor(cls, plane: "Clifford41", angle: float, device=None, dtype=torch.float64):
        c = torch.cos(torch.tensor(angle / 2, device=device, dtype=dtype))
        s = torch.sin(torch.tensor(angle / 2, device=device, dtype=dtype))
        return cls.scalar(c, device, dtype) - (plane * cls.scalar(s, device, dtype))

    def __repr__(self):
        return f"Clifford41(shape={self.data.shape}, device={self.device})"
