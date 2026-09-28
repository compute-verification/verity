"""Hint generation for the FP8 unit: ``hints_fp8`` is a vectorised (torch int64) ``gen_step`` on the adder-side params of
:mod:`relation` -- every line mirrors the gadget's ``gen`` -- and ``oracle_words`` is the model itself
(``silicon.tc_dot(ADA_E4M3_M16N8K32, ...)``) for the tests.  A wrong hint cannot pass the constraints (the prover's
self-check ``witness.check_constraints`` and the tests catch it), so a bug here is completeness, not soundness."""
from __future__ import annotations

import numpy as np
import torch
from verity.ml.circuits.silicon.silicon import ADA_E4M3_M16N8K32, tc_dot

from ..compile import System
from .relation import ACC_FRAC, ADA, FP8Params

__all__ = ["hints_fp8", "oracle_words", "random_finite_operands", "random_finite_fp32"]


def _bit_length(v: torch.Tensor, max_bits: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    for k in range(max_bits + 1):
        out = out + (v >= (1 << k)).to(torch.int64)
    return out


def _decode_e4m3(p: FP8Params, w: torch.Tensor):
    t = (w >> p.op_frac) & (2**p.op_exp - 1)
    f = w & (2**p.op_frac - 1)
    nz = (t != 0).to(torch.int64)
    return w >> 7, f + nz * 2**p.op_frac, t - nz - (p.op_bias - 1)


def _decode_fp32(p: FP8Params, c: torch.Tensor):
    t = (c >> ACC_FRAC) & 0xFF
    f = c & (2**ACC_FRAC - 1)
    nz = (t != 0).to(torch.int64)
    z = ((t == 0) & (f == 0)).to(torch.int64)
    M = f + nz * 2**ACC_FRAC
    e = t - nz - (p.bias - 1)
    return {"s": c >> 31, "t": t, "f": f, "nz": nz, "z": z}, (c >> 31, e + z * (p.floor - e), M >> p.rescale)


def hints_fp8(p: FP8Params, sys: System, c, a, b, device="cpu") -> torch.Tensor:
    """``c`` (l,) FP32 words, ``a``/``b`` (l, 32) E4M3 bytes -> (n_hints, l) int64 in the order of ``sys.hints``.
    On CUDA the ~280 small kernels are captured once per (system, l, k, stream) as a CUDA graph and replayed (as
    ``hints_torch``; lane hp2-host) -- the same ops on the same values, so the hints are identical."""
    # numpy / list / torch (host or already on ``device``: relchain.marshal keeps the instance arrays device-resident)
    as_i64 = lambda x: (x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))).to(dtype=torch.int64, device=device)  # noqa: E731
    c, a, b = as_i64(c), as_i64(a), as_i64(b)
    if USE_CUDA_GRAPH and str(device).startswith("cuda"):
        key = (id(sys), c.shape[0], a.shape[1], str(device), torch.cuda.current_stream(device).cuda_stream)
        g = _GRAPHS.get(key)
        if g is None:
            try:
                g = _Graph(p, sys, c, a, b, device)
            except Exception:  # noqa: BLE001 -- capture unsupported here: eager
                g = False
            _GRAPHS[key] = g
        if g:
            return g.run(c, a, b)
    return _hints_fp8_eager(p, sys, c, a, b, device)


class _Graph:
    def __init__(self, p, sys, c, a, b, device):
        self.c, self.a, self.b = torch.zeros_like(c), torch.zeros_like(a), torch.zeros_like(b)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _hints_fp8_eager(p, sys, self.c, self.a, self.b, device)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = _hints_fp8_eager(p, sys, self.c, self.a, self.b, device)

    def run(self, c, a, b):
        self.c.copy_(c)
        self.a.copy_(a)
        self.b.copy_(b)
        self.graph.replay()
        return self.out.clone()


_GRAPHS: dict = {}
USE_CUDA_GRAPH = True


def _hints_fp8_eager(p: FP8Params, sys: System, c, a, b, device) -> torch.Tensor:
    l, k = c.shape[0], a.shape[1]
    H: dict[str, torch.Tensor] = {}
    acc, state = _decode_fp32(p, c)
    if sys.chain is not None:
        for n in ("s", "t", "f", "nz", "z"):
            H[f"acc.{n}"] = acc[n]
    s_a, m_a, e_a = _decode_e4m3(p, a)
    s_b, m_b, e_b = _decode_e4m3(p, b)
    u = m_a * m_b
    zp = (u == 0).to(torch.int64)
    e_p = e_a + e_b
    s_p = (s_a + s_b) % 2
    e_hat_p = e_p + zp * (p.floor - e_p)
    for i in range(k):
        H[f"prod[{i}].z"] = zp[:, i]
    start = 0
    for g, size in enumerate(p.groups):
        tag = f"g{g}"
        sl_ = slice(start, start + size)
        start += size
        s_st, e_st, M_st = state
        cands = torch.cat([e_st[:, None], e_hat_p[:, sl_]], dim=1)
        options = torch.cat([cands, torch.full((l, 1), p.floor, dtype=torch.int64, device=device)], dim=1)
        E = options.max(dim=1).values
        first = (options == E[:, None]).to(torch.int64).argmax(dim=1)
        H[f"{tag}.max.E"] = E
        for j in range(size + 2):
            H[f"{tag}.max.sel[{j}]"] = (first == j).to(torch.int64)
        ds = E[:, None] - cands
        Pm = M_st << p.acc_shift                                         # acc_shift = 0 on the adder-side params
        dd = torch.clamp(ds[:, 0], max=p.acc_bits)
        q_acc = Pm >> dd
        H[f"{tag}.acc_align.q"] = q_acc
        H[f"{tag}.acc_align.r"] = Pm - (q_acc << dd)
        ug = u[:, sl_]
        d = ds[:, 1:]
        sh = torch.clamp(d - p.prod_shift, min=0)
        q_mid = ug >> sh
        r_mid = ug - (q_mid << sh)
        low = d <= p.prod_shift
        high = d >= p.prod_bits
        q_pr = torch.where(low, ug, torch.where(high, torch.zeros_like(ug), q_mid))
        r_pr = torch.where(low, torch.zeros_like(ug), torch.where(high, ug, r_mid))
        for i in range(size):
            H[f"{tag}.prod_align[{i}].q"] = q_pr[:, i]
            H[f"{tag}.prod_align[{i}].r"] = r_pr[:, i]
        sl_pow = torch.where(low, torch.ones_like(d) << torch.clamp(p.prod_shift - d, min=0), torch.ones_like(d))
        aligned = q_pr * sl_pow
        S = (1 - 2 * s_st) * q_acc + ((1 - 2 * s_p[:, sl_]) * aligned).sum(dim=1)
        sgn = (S < 0).to(torch.int64)
        zs = (S == 0).to(torch.int64)
        H[f"{tag}.sum.sgn"] = sgn
        H[f"{tag}.sum.z"] = zs
        v = S - 2 * sgn * S
        lb = _bit_length(v, p.s_bits)
        H[f"{tag}.lead.l"] = lb
        e_raw = E + lb - p.width
        clamp = (e_raw < p.e_min).to(torch.int64)
        e1 = torch.maximum(e_raw, torch.full_like(e_raw, p.e_min))
        kk = E - e1 + p.acc_sig - p.width
        kpos = torch.clamp(kk, min=0)
        kneg = torch.clamp(-kk, min=0)
        Mn = torch.where(kk >= 0, v << kpos, v >> kneg)
        rn = torch.where(kk >= 0, torch.zeros_like(v), v - (Mn << kneg))
        H[f"{tag}.norm.clamp"] = clamp
        H[f"{tag}.norm.M"] = Mn
        H[f"{tag}.norm.r"] = rn
        zo = (Mn == 0).to(torch.int64)
        ovf = (e1 > p.e_max).to(torch.int64)
        H[f"{tag}.out.z"] = zo
        H[f"{tag}.out.ovf"] = ovf
        state = (sgn - sgn * zo, e1 + zo * (p.floor - e1), Mn)
    M_fin = state[2]
    H["pack.hidden"] = M_fin >> (p.acc_sig - 1)
    H["pack.low"] = M_fin & (2 ** (p.acc_sig - 1) - 1)
    names = [n for _, n in sys.hints]
    return torch.stack([H[n] for n in names])


def oracle_words(c, a, b) -> np.ndarray:
    """``silicon.tc_dot(ADA_E4M3_M16N8K32, c[i], a[i], b[i])`` per unit (host Python; the reference)."""
    c, a, b = np.asarray(c), np.asarray(a), np.asarray(b)
    return np.asarray([tc_dot(ADA_E4M3_M16N8K32, int(c[i]), [int(x) for x in a[i]], [int(x) for x in b[i]]) for i in range(len(c))],
                      dtype=np.int64)


def random_finite_operands(rng: np.random.Generator, shape) -> np.ndarray:
    """Uniform E4M3 bytes without the NaN codes (0x7f / 0xff)."""
    w = rng.integers(0, 256, size=shape, dtype=np.int64)
    nan = (w & 0x7F) == 0x7F
    w[nan] = w[nan] & ~0x08                    # 0x7f -> 0x77 (the largest finite), keeps the sign
    return w


def random_finite_fp32(rng: np.random.Generator, n: int, exp_lo: int = 90, exp_hi: int = 160) -> np.ndarray:
    """Finite FP32 words with exponent fields in [exp_lo, exp_hi) (plus a few zeros and subnormals), as int64."""
    t = rng.integers(exp_lo, exp_hi, size=n, dtype=np.int64)
    f = rng.integers(0, 2**ACC_FRAC, size=n, dtype=np.int64)
    s = rng.integers(0, 2, size=n, dtype=np.int64)
    kind = rng.integers(0, 16, size=n)
    t = np.where(kind == 0, 0, t)                      # zero / subnormal
    f = np.where(kind == 1, 0, f)
    return (s << 31) | (t << ACC_FRAC) | f
