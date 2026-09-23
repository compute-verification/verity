"""A vectorised hint generator: ``gen_step`` for a batch of units in torch (int64), producing the same
``(n_hints, l)`` tensor as ``witness.hints_tensor`` (which runs the checker's Python ``gen_step`` per unit).

Every line mirrors a gadget's ``gen`` (or the ``check``-side value it feeds on) in ``verity_numerical.checker``;
the prover's self-check (``check_constraints``) and the gate's cross-check against ``hints_tensor`` are what make it
trustworthy -- a wrong hint cannot pass the constraints, so a bug here is completeness, not soundness.
"""
from __future__ import annotations

import torch
from verity_numerical.checker.params import Params

from . import wide as wide_mod
from .compile import System

__all__ = ["hints_torch"]


def _bit_length(v: torch.Tensor, max_bits: int) -> torch.Tensor:
    out = torch.zeros_like(v)
    for k in range(max_bits + 1):
        out = out + (v >= (1 << k)).to(torch.int64)
    return out


def _decode_bf16(p: Params, w: torch.Tensor):
    t = (w >> p.frac) & (2**p.exp - 1)
    f = w & (2**p.frac - 1)
    nz = (t != 0).to(torch.int64)
    m = f + nz * 2**p.frac
    e = t - nz - (p.bias - 1)
    s = w >> (p.word_bits - 1)
    return s, m, e


def _decode_fp32(p: Params, c: torch.Tensor):
    t = (c >> p.acc_frac) & (2**p.exp - 1)
    f = c & (2**p.acc_frac - 1)
    nz = (t != 0).to(torch.int64)
    z = ((t == 0) & (f == 0)).to(torch.int64)
    M = f + nz * 2**p.acc_frac
    e = t - nz - (p.bias - 1)
    e_hat = e + z * (p.floor - e)
    s = c >> (p.acc_word_bits - 1)
    return {"s": s, "t": t, "f": f, "nz": nz, "z": z}, (s, e_hat, M)


def hints_torch(p: Params, sys: System, c, a, b, y=None, device="cpu") -> torch.Tensor:
    """``c`` (l,), ``a``/``b`` (l, k), ``y`` (l,) or None (unit mode: no acc/epi rows) -> (n_hints, l) int64.
    On CUDA the ~150 small kernels are captured once per (system, l, k) as a CUDA graph and replayed."""
    c = torch.as_tensor(c, dtype=torch.int64).to(device, non_blocking=True)
    a = torch.as_tensor(a, dtype=torch.int64).to(device, non_blocking=True)
    b = torch.as_tensor(b, dtype=torch.int64).to(device, non_blocking=True)
    y = None if y is None else torch.as_tensor(y, dtype=torch.int64).to(device, non_blocking=True)
    if USE_CUDA_GRAPH and str(device).startswith("cuda"):
        key = (id(sys), c.shape[0], a.shape[1], y is not None, str(device))
        g = _GRAPHS.get(key)
        if g is None:
            try:
                g = _Graph(p, sys, c, a, b, y, device)
            except Exception:  # noqa: BLE001 -- capture unsupported here: eager
                g = False
            _GRAPHS[key] = g
        if g:
            return g.run(c, a, b, y)
    return _hints_eager(p, sys, c, a, b, y, device)


class _Graph:
    def __init__(self, p, sys, c, a, b, y, device):
        self.c, self.a, self.b = torch.zeros_like(c), torch.zeros_like(a), torch.zeros_like(b)
        self.y = None if y is None else torch.zeros_like(y)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                _hints_eager(p, sys, self.c, self.a, self.b, self.y, device)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = _hints_eager(p, sys, self.c, self.a, self.b, self.y, device)

    def run(self, c, a, b, y):
        self.c.copy_(c)
        self.a.copy_(a)
        self.b.copy_(b)
        if y is not None:
            self.y.copy_(y)
        self.graph.replay()
        return self.out.clone()


_GRAPHS: dict = {}
USE_CUDA_GRAPH = True
_WIDE_IDX: dict = {}


def _wide_index(sys, tag: str, kidx: int, idx: list[int], device) -> torch.Tensor:
    key = (id(sys), tag, kidx, str(device))
    hit = _WIDE_IDX.get(key)
    if hit is None or hit[0] is not sys:
        hit = (sys, torch.tensor(list(idx), dtype=torch.int64, device=device))
        _WIDE_IDX[key] = hit
    return hit[1]


def _hints_eager(p: Params, sys: System, c, a, b, y, device) -> torch.Tensor:
    l = c.shape[0]
    k = a.shape[1]
    H: dict[str, torch.Tensor] = {}

    acc, state = _decode_fp32(p, c)
    if sys.chain is not None:
        for n in ("s", "t", "f", "nz", "z"):
            H[f"acc.{n}"] = acc[n]
    s_a, m_a, e_a = _decode_bf16(p, a)
    s_b, m_b, e_b = _decode_bf16(p, b)
    # products
    u = m_a * m_b                                                  # (l, k)
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
        cands = torch.cat([e_st[:, None], e_hat_p[:, sl_]], dim=1)          # (l, size + 1)
        options = torch.cat([cands, torch.full((l, 1), p.floor, dtype=torch.int64, device=device)], dim=1)
        E = options.max(dim=1).values
        first = (options == E[:, None]).to(torch.int64).argmax(dim=1)        # first attaining index
        H[f"{tag}.max.E"] = E
        for j in range(size + 2):
            H[f"{tag}.max.sel[{j}]"] = (first == j).to(torch.int64)
        ds = E[:, None] - cands                                             # (l, size + 1)
        # AlignState on (M_st, ds[:, 0])
        Pm = M_st << p.acc_shift
        dd = torch.clamp(ds[:, 0], max=p.acc_bits)
        q_acc = Pm >> dd
        H[f"{tag}.acc_align.q"] = q_acc
        H[f"{tag}.acc_align.r"] = Pm - (q_acc << dd)
        # AlignProduct on (u_i, ds[:, 1 + i])
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
        # aligned product terms: q * sl with sl = 2^(prod_shift - d) for d <= prod_shift, else 1 (0 rows: q = 0)
        sl_pow = torch.where(low, torch.ones_like(d) << torch.clamp(p.prod_shift - d, min=0), torch.ones_like(d))
        aligned = q_pr * sl_pow
        # SignedSum
        S = (1 - 2 * s_st) * q_acc + ((1 - 2 * s_p[:, sl_]) * aligned).sum(dim=1)
        sgn = (S < 0).to(torch.int64)
        zs = (S == 0).to(torch.int64)
        H[f"{tag}.sum.sgn"] = sgn
        H[f"{tag}.sum.z"] = zs
        v = S - 2 * sgn * S
        # LeadingBit
        lb = _bit_length(v, p.s_bits)
        H[f"{tag}.lead.l"] = lb
        # Normalise
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
        if tag in sys.wide:                                            # wide.py: partial signs, limbs, carries
            HM = 2**wide_mod.H - 1
            signed_terms = torch.cat([((1 - 2 * s_st) * q_acc)[:, None], (1 - 2 * s_p[:, sl_]) * aligned], dim=1)
            SL = torch.zeros_like(S)
            SH = torch.zeros_like(S)
            for kidx, idx in enumerate(sys.wide[tag]):
                # the partition's term indices as a cached device tensor: a Python list here is a host->device copy
                # inside the CUDA-graph capture, which aborts it (and leaves the allocator's capture state dirty, so
                # the witness program's capture failed after it) -- lane hostphase
                idx_t = _wide_index(sys, tag, kidx, idx, device)
                A = signed_terms.index_select(1, idx_t).sum(dim=1)
                sk = (A < 0).to(torch.int64)
                H[f"{tag}.sum.part[{kidx}].sgn"] = sk
                vk = A.abs()
                SL = SL + (1 - 2 * sk) * (vk & HM)
                SH = SH + (1 - 2 * sk) * (vk >> wide_mod.H)
            vl, vh = v & HM, v >> wide_mod.H
            H[f"{tag}.sum.vl"], H[f"{tag}.sum.vh"] = vl, vh
            H[f"{tag}.sum.carry"] = (SL - (1 - 2 * sgn) * vl) >> wide_mod.H
            mul, div = torch.ones_like(kk) << kpos, torch.ones_like(kk) << kneg
            H[f"{tag}.norm.carry"] = (div * (Mn & HM) + rn - mul * vl) >> wide_mod.H
        # StateOut
        zo = (Mn == 0).to(torch.int64)
        ovf = (e1 > p.e_max).to(torch.int64)
        H[f"{tag}.out.z"] = zo
        H[f"{tag}.out.ovf"] = ovf
        s_out = sgn - sgn * zo
        e_out = e1 + zo * (p.floor - e1)
        state = (s_out, e_out, Mn)
    # Pack
    M_fin = state[2]
    H["pack.hidden"] = M_fin >> p.acc_frac
    H["pack.low"] = M_fin & (2**p.acc_frac - 1)
    if sys.chain is not None:
        L = p.acc_frac - p.frac
        hi, lo = y >> L, y & (2**L - 1)
        lsb = hi & 1
        t = lo + lsb + (2 ** (L - 1) - 1)
        H["epi.lo"], H["epi.hi2"], H["epi.lsb"] = lo, hi >> 1, lsb
        H["epi.carry"], H["epi.rem"] = t >> L, t & (2**L - 1)
        H["epi.fhi"] = (y & (2**p.acc_frac - 1)) >> L
    names = [n for _, n in sys.hints]
    return torch.stack([H[n] for n in names])
