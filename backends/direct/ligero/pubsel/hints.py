"""Hint generators of the v2 (public-selection) units: ``hints_v2`` vectorised in torch (the prover's), ``reference_hints_v2``
on Python integers per unit (the gate's cross-check).  Both follow ``relation.compile_v2_unit`` row for row; a wrong hint
cannot pass the constraints (the prover self-checks them), so a bug here is completeness, not soundness."""
from __future__ import annotations

from verity_numerical.checker.params import Params

from ..compile import System
from .relation import ACC_FRAC, LIMB, Plan, aligned_product, operand_decode_int, plan

__all__ = ["hints_v2", "reference_hints_v2", "step_int"]


# ---------------------------------------------------------------------------------------------------------------------
# Python integers: one unit
# ---------------------------------------------------------------------------------------------------------------------

def step_int(p: Params, pl: Plan, c: int, a: list[int], b: list[int], y: int | None, chain: bool) -> dict[str, int]:
    """The hint rows of one unit (``c``: the incoming FP32 word; ``y``: the unit's output word, chain mode's epilogue)."""
    H: dict[str, int] = {}
    t = (c >> ACC_FRAC) & (2**p.acc_exp - 1)
    f = c & (2**ACC_FRAC - 1)
    nz = int(t != 0)
    z = int(t == 0 and f == 0)
    if chain:
        H.update({"acc.s": c >> 31, "acc.t": t, "acc.f": f, "acc.nz": nz, "acc.z": z})
    M24 = f + nz * 2**ACC_FRAC
    e = t - nz - (p.acc_bias - 1)
    s_st, e_st, Mt = c >> 31, e + z * (p.floor - e), M24 >> pl.drop
    start = 0
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        terms = []
        for i in range(start, start + size):
            sa, ma, ea = operand_decode_int(p, pl, a[i])
            sb, mb, eb = operand_decode_int(p, pl, b[i])
            u = ma * mb
            terms.append((sa ^ sb, u, ea + eb if u else p.floor))
        start += size
        G = max([p.floor] + [ee for _, _, ee in terms])
        E = max(G, e_st)
        delta, dacc = E - G, E - e_st
        H[f"{tag}.delta"], H[f"{tag}.dacc"] = delta, dacc
        Q = sum((1 - 2 * s) * aligned_product(pl, u, E - ee) for s, u, ee in terms)
        q_acc = (Mt << pl.ss) >> min(dacc, pl.W)
        sign = 1 - 2 * s_st
        S = sign * q_acc + Q
        sgn = int(S < 0)
        v = -S if sgn else S
        H[f"{tag}.sgn"] = sgn
        if pl.wide:
            vL, vH = v & (2**LIMB - 1), v >> LIMB
            H[f"{tag}.vL"], H[f"{tag}.vH"] = vL, vH
            qL, QL = q_acc & (2**LIMB - 1), Q & (2**LIMB - 1)
            H[f"{tag}.qL"], H[f"{tag}.qH"] = qL, q_acc >> LIMB
            H[f"{tag}.QL"], H[f"{tag}.QH"] = QL, (Q - QL) >> LIMB
            H[f"{tag}.carry"] = (sign * qL + QL - (1 - 2 * sgn) * vL) >> LIMB
        else:
            H[f"{tag}.v"] = v
            H[f"{tag}.q"], H[f"{tag}.Q"] = q_acc, Q
        l = v.bit_length()
        e1 = max(E + l - p.width, p.acc_e_min)
        tt = e1 - E + p.width - pl.nb if v else pl.t_lo      # a zero sum: any shift in range (the relation masks e1 by z)
        Mn = (v >> tt) if tt >= 0 else (v << -tt)
        for i in range(pl.n_t):
            H[f"{tag}.nsel[{i}]"] = int(tt == pl.t_lo + i)
        hid, fo = Mn >> (pl.nb - 1), Mn & (2 ** (pl.nb - 1) - 1)
        H[f"{tag}.hid"], H[f"{tag}.fo"] = hid, fo
        zo = int(Mn == 0)
        H[f"{tag}.z"], H[f"{tag}.ovf"] = zo, int(e1 > p.acc_e_max and not zo)
        s_st = sgn * (1 - zo)
        e_st = e1 + zo * (p.floor - e1)
        Mt = Mn
    if chain and not pl.fp8:
        L = ACC_FRAC - p.frac
        H["epi.zl"] = int((fo & (2 ** (L - 1) - 1)) + ((fo >> L) & 1) == 0)   # the last group's (unmasked) output fraction
    return H


def reference_hints_v2(p: Params, sys: System, units: list[tuple], device):
    """(n_hints, l) from ``step_int`` on every unit ``(c, a, b[, y])`` (the gate's independent generator)."""
    import torch

    pl = plan(p)
    names = [n for _, n in sys.hints]
    pos = {n: i for i, n in enumerate(names)}
    out = torch.zeros((len(names), len(units)), dtype=torch.int64)
    chain = sys.chain is not None
    for col, u in enumerate(units):
        c, a, b = u[:3]
        y = u[3] if len(u) > 3 else None
        for n, val in step_int(p, pl, int(c), list(a), list(b), None if y is None else int(y), chain).items():
            out[pos[n], col] = val
    return out.to(device)


# ---------------------------------------------------------------------------------------------------------------------
# torch: a batch of units
# ---------------------------------------------------------------------------------------------------------------------

def _bit_length(v, max_bits: int):
    import torch

    out = torch.zeros_like(v)
    for k in range(max_bits + 1):
        out = out + (v >= (1 << k)).to(torch.int64)
    return out


def _decode(pl: Plan, w):
    import torch

    t = (w >> pl.op_frac) & (2**pl.op_exp - 1)
    f = w & (2**pl.op_frac - 1)
    nz = (t != 0).to(torch.int64)
    return w >> (pl.op_word_bits - 1), f + nz * 2**pl.op_frac, t - nz - (pl.op_bias - 1)


def hints_v2(p: Params, sys: System, c, a, b, y=None, device="cpu"):
    """``c`` (l,), ``a``/``b`` (l, k), ``y`` (l,) or None -> (n_hints, l) int64 in ``sys.hints`` order.  On CUDA the kernels
    are captured once per (system, l) as a CUDA graph and replayed."""
    import torch

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
        import torch

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


def _hints_eager(p: Params, sys: System, c, a, b, y, device):
    import torch

    pl = plan(p)
    chain = sys.chain is not None
    H: dict = {}
    t = (c >> ACC_FRAC) & (2**p.acc_exp - 1)
    f = c & (2**ACC_FRAC - 1)
    nz = (t != 0).to(torch.int64)
    z = ((t == 0) & (f == 0)).to(torch.int64)
    s_c = c >> 31
    if chain:
        H.update({"acc.s": s_c, "acc.t": t, "acc.f": f, "acc.nz": nz, "acc.z": z})
    M24 = f + nz * 2**ACC_FRAC
    e = t - nz - (p.acc_bias - 1)
    s_st, e_st, Mt = s_c, e + z * (p.floor - e), M24 >> pl.drop
    s_a, m_a, e_a = _decode(pl, a)
    s_b, m_b, e_b = _decode(pl, b)
    u = m_a * m_b
    sp = (s_a + s_b) % 2
    ep = torch.where(u == 0, torch.full_like(u, p.floor), e_a + e_b)
    start = 0
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        sl = slice(start, start + size)
        start += size
        ug, sg, eg = u[:, sl], sp[:, sl], ep[:, sl]
        G = torch.clamp(eg.max(dim=1).values, min=p.floor)
        E = torch.maximum(G, e_st)
        delta, dacc = E - G, E - e_st
        H[f"{tag}.delta"], H[f"{tag}.dacc"] = delta, dacc
        d = E[:, None] - eg
        al = torch.where(d <= pl.PS, ug << torch.clamp(pl.PS - d, min=0),
                         torch.where(d < pl.PB, ug >> torch.clamp(d - pl.PS, min=0), torch.zeros_like(ug)))
        Q = ((1 - 2 * sg) * al).sum(dim=1)
        q_acc = (Mt << pl.ss) >> torch.clamp(dacc, max=pl.W)
        sign = 1 - 2 * s_st
        S = sign * q_acc + Q
        sgn = (S < 0).to(torch.int64)
        v = S.abs()
        H[f"{tag}.sgn"] = sgn
        if pl.wide:
            vL, vH = v & (2**LIMB - 1), v >> LIMB
            H[f"{tag}.vL"], H[f"{tag}.vH"] = vL, vH
            qL, QL = q_acc & (2**LIMB - 1), Q & (2**LIMB - 1)
            H[f"{tag}.qL"], H[f"{tag}.qH"] = qL, q_acc >> LIMB
            H[f"{tag}.QL"], H[f"{tag}.QH"] = QL, (Q - QL) >> LIMB
            H[f"{tag}.carry"] = (sign * qL + QL - (1 - 2 * sgn) * vL) >> LIMB
        else:
            H[f"{tag}.v"] = v
            H[f"{tag}.q"], H[f"{tag}.Q"] = q_acc, Q
        l = _bit_length(v, pl.SB)
        e1 = torch.clamp(E + l - p.width, min=p.acc_e_min)
        tt = torch.where(v != 0, e1 - E + p.width - pl.nb, torch.full_like(v, pl.t_lo))
        Mn = torch.where(tt >= 0, v >> torch.clamp(tt, min=0), v << torch.clamp(-tt, min=0))
        for i in range(pl.n_t):
            H[f"{tag}.nsel[{i}]"] = (tt == pl.t_lo + i).to(torch.int64)
        hid, fo = Mn >> (pl.nb - 1), Mn & (2 ** (pl.nb - 1) - 1)
        H[f"{tag}.hid"], H[f"{tag}.fo"] = hid, fo
        zo = (Mn == 0).to(torch.int64)
        H[f"{tag}.z"], H[f"{tag}.ovf"] = zo, ((e1 > p.acc_e_max) & (zo == 0)).to(torch.int64)
        s_st = sgn * (1 - zo)
        e_st = e1 + zo * (p.floor - e1)
        Mt = Mn
    if chain and not pl.fp8:
        L = ACC_FRAC - p.frac
        H["epi.zl"] = ((fo & (2 ** (L - 1) - 1)) + ((fo >> L) & 1) == 0).to(torch.int64)
    names = [n for _, n in sys.hints]
    return torch.stack([H[n] for n in names])
