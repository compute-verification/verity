"""Hint generators of the v3 (private-operand) units: ``hints_v3`` vectorised in torch (the prover's), ``reference_hints_v3``
on Python integers per unit (the gate's cross-check).  Both follow ``relation.compile_v3_unit`` row for row; a wrong hint
cannot pass the constraints (the prover self-checks them), so a bug here is completeness, not soundness."""
from __future__ import annotations

from verity_numerical.checker.params import Params

from ..compile import System
from ..pubsel.relation import ACC_FRAC, LIMB, Plan, aligned_product, operand_decode_int, plan
from .relation import ShiftPlan, shift_plan

__all__ = ["hints_v3", "reference_hints_v3", "step_int"]


# ---------------------------------------------------------------------------------------------------------------------
# Python integers: one unit
# ---------------------------------------------------------------------------------------------------------------------

def _word_hints(H: dict, pl: Plan, name: str, w: int) -> None:
    t = (w >> pl.op_frac) & (2**pl.op_exp - 1)
    f = w & (2**pl.op_frac - 1)
    nz = int(t != 0)
    x = t - nz
    H[f"{name}.s"] = w >> (pl.op_word_bits - 1)
    H[f"{name}.nz"] = nz
    for j in range(pl.op_exp):
        H[f"{name}.x[{j}]"] = (x >> j) & 1
    for j in range(pl.op_frac):
        H[f"{name}.f[{j}]"] = (f >> j) & 1


def _align_choice(pl: Plan, sp: ShiftPlan, u: int, d: int) -> tuple[int, int, int, int]:
    """(R, coarse index, K, X) of one product: coarse index 0..3 = c in {-1, 0, 1, 2}, 4 = dead, 5 = zero."""
    RB = sp.RB
    if u == 0:
        return 0, 5, 0, 0
    if d >= pl.PB:
        R = RB - 1 - ((d - pl.PB) % RB)
        return R, 4, (d - pl.PB) // RB, 0
    L = pl.PS - d
    c = -(L // RB)                    # floor division
    R = L + RB * c
    return R, c + 1, 0, aligned_product(pl, u, d)


def step_int(p: Params, pl: Plan, c: int, a: list[int], b: list[int], y: int | None, chain: bool) -> dict[str, int]:
    """The hint rows of one unit (``c``: the incoming FP32 word; ``y``: the unit's output word, chain mode's epilogue)."""
    sp = shift_plan(p, pl)
    RB, UB = sp.RB, sp.UB
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
    k = sum(pl.groups)
    for i in range(k):
        _word_hints(H, pl, f"a_words[{i}]", a[i])
        _word_hints(H, pl, f"b_words[{i}]", b[i])
    start = 0
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        terms = []
        for i in range(start, start + size):
            sa, ma, ea = operand_decode_int(p, pl, a[i])
            sb, mb, eb = operand_decode_int(p, pl, b[i])
            terms.append((i, sa ^ sb, ma * mb, ea + eb))
        start += size
        E = max([e_st] + [ee for _, _, u, ee in terms if u])
        H[f"{tag}.E"] = E
        which = 0 if e_st == E else 1 + next(j for j, (_, _, u, ee) in enumerate(terms) if u and ee == E)
        for j in range(size + 1):
            H[f"{tag}.which[{j}]"] = int(j == which)
        Q = SL = 0
        for j, (i, s, u, ee) in enumerate(terms):
            scope = f"{tag}.p[{i}]"
            for jj in range(UB):
                H[f"{scope}.u[{jj}]"] = (u >> jj) & 1
            R, ci, K, X = _align_choice(pl, sp, u, E - ee)
            for r in range(RB):
                H[f"{scope}.r[{r}]"] = int(r == R)
            Y = u << R
            for kk in range(sp.ND):
                H[f"{scope}.D{kk}"] = (Y >> (kk * RB)) & (2**RB - 1)
            for cc in range(sp.ncoarse):
                H[f"{scope}.c[{cc}]"] = int(cc == ci)
            if pl.wide:
                H[f"{scope}.XL"], H[f"{scope}.XH"] = X & (2**LIMB - 1), X >> LIMB
            else:
                H[f"{scope}.X"] = X
            H[f"{scope}.K"] = K
            H[f"{scope}.s"] = s
            if which == j + 1:
                H[f"{tag}.C"] = X
            Q += (1 - 2 * s) * X
            SL += (1 - 2 * s) * (X & (2**LIMB - 1))
        if which == 0:
            H[f"{tag}.C"] = 1
        dacc = E - e_st
        q_acc = (Mt << pl.ss) >> min(dacc, pl.W)
        sign = 1 - 2 * s_st
        S = sign * q_acc + Q
        sgn = int(S < 0)
        v = -S if sgn else S
        H[f"{tag}.sgn"] = sgn
        if pl.wide:
            vL, vH = v & (2**LIMB - 1), v >> LIMB
            H[f"{tag}.vL"], H[f"{tag}.vH"] = vL, vH
            qL = q_acc & (2**LIMB - 1)
            H[f"{tag}.qL"], H[f"{tag}.qH"] = qL, q_acc >> LIMB
            # the low-limb sum is of each product's signed low limb, sum_i (1 - 2 s_i) XL_i (not Q's low limb)
            H[f"{tag}.carry"] = (sign * qL + SL - (1 - 2 * sgn) * vL) >> LIMB
        else:
            H[f"{tag}.v"] = v
            H[f"{tag}.q"] = q_acc
        l = v.bit_length()
        e1 = max(E + l - p.width, p.acc_e_min)
        tt = e1 - E + p.width - pl.nb
        Mn = (v >> tt) if tt >= 0 else (v << -tt)
        for i in range(pl.n_t):
            H[f"{tag}.nsel[{i}]"] = int(tt == pl.t_lo + i)
        hid, fo = Mn >> (pl.nb - 1), Mn & (2 ** (pl.nb - 1) - 1)
        H[f"{tag}.hid"], H[f"{tag}.fo"] = hid, fo
        zo = int(Mn == 0)
        H[f"{tag}.z"], H[f"{tag}.ovf"] = zo, int(e1 > p.acc_e_max)
        s_st = sgn * (1 - zo)
        e_st = e1 + zo * (p.floor - e1)
        Mt = Mn
    if chain and not pl.fp8:
        L = ACC_FRAC - p.frac
        H["epi.zl"] = int((fo & (2 ** (L - 1) - 1)) + ((fo >> L) & 1) == 0)
    return H


def reference_hints_v3(p: Params, sys: System, units: list[tuple], device):
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


def hints_v3(p: Params, sys: System, c, a, b, y=None, device="cpu"):
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


def _decode(pl: Plan, w):
    import torch

    t = (w >> pl.op_frac) & (2**pl.op_exp - 1)
    f = w & (2**pl.op_frac - 1)
    nz = (t != 0).to(torch.int64)
    return w >> (pl.op_word_bits - 1), f + nz * 2**pl.op_frac, t - nz - (pl.op_bias - 1), t - nz, f, nz


def _hints_eager(p: Params, sys: System, c, a, b, y, device):
    import torch

    pl = plan(p)
    sp = shift_plan(p, pl)
    RB, UB = sp.RB, sp.UB
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
    k = a.shape[1]
    for name, w in (("a_words", a), ("b_words", b)):
        s_, m_, e_, x_, f_, nz_ = _decode(pl, w)
        for i in range(k):
            H[f"{name}[{i}].s"] = s_[:, i]
            H[f"{name}[{i}].nz"] = nz_[:, i]
            for j in range(pl.op_exp):
                H[f"{name}[{i}].x[{j}]"] = (x_[:, i] >> j) & 1
            for j in range(pl.op_frac):
                H[f"{name}[{i}].f[{j}]"] = (f_[:, i] >> j) & 1
    s_a, m_a, e_a = _decode(pl, a)[:3]
    s_b, m_b, e_b = _decode(pl, b)[:3]
    u = m_a * m_b
    sp_ = (s_a + s_b) % 2
    ep = e_a + e_b
    NEG = -(2**20)
    start = 0
    for g, size in enumerate(pl.groups):
        tag = f"g{g}"
        sl = slice(start, start + size)
        idx = list(range(start, start + size))
        start += size
        ug, sg, eg = u[:, sl], sp_[:, sl], ep[:, sl]
        cands = torch.cat([e_st[:, None], torch.where(ug != 0, eg, torch.full_like(eg, NEG))], dim=1)   # (l, size + 1)
        E = cands.max(dim=1).values
        H[f"{tag}.E"] = E
        first = (cands == E[:, None]).to(torch.int64)
        first = first * (first.cumsum(dim=1) == 1).to(torch.int64)      # the first candidate equal to E
        for j in range(size + 1):
            H[f"{tag}.which[{j}]"] = first[:, j]
        d = E[:, None] - eg
        dead = (d >= pl.PB) & (ug != 0)
        zero = ug == 0
        live = ~dead & ~zero
        L = pl.PS - d
        cfl = torch.div(L, RB, rounding_mode="floor")
        R_live = L - RB * cfl
        R_dead = RB - 1 - torch.remainder(d - pl.PB, RB)
        R = torch.where(live, R_live, torch.where(dead, R_dead, torch.zeros_like(d)))
        K = torch.where(dead, torch.div(d - pl.PB, RB, rounding_mode="floor"), torch.zeros_like(d))
        ci = torch.where(live, 1 - cfl, torch.where(dead, torch.full_like(d, 4), torch.full_like(d, 5)))
        Y = ug << R
        X = torch.where(d <= pl.PS, ug << torch.clamp(pl.PS - d, min=0),
                        torch.where(d < pl.PB, ug >> torch.clamp(d - pl.PS, min=0), torch.zeros_like(ug)))
        X = torch.where(zero, torch.zeros_like(X), X)
        for j, i in enumerate(idx):
            scope = f"{tag}.p[{i}]"
            for jj in range(UB):
                H[f"{scope}.u[{jj}]"] = (ug[:, j] >> jj) & 1
            for r in range(RB):
                H[f"{scope}.r[{r}]"] = (R[:, j] == r).to(torch.int64)
            for kk in range(sp.ND):
                H[f"{scope}.D{kk}"] = (Y[:, j] >> (kk * RB)) & (2**RB - 1)
            for cc in range(sp.ncoarse):
                H[f"{scope}.c[{cc}]"] = (ci[:, j] == cc).to(torch.int64)
            if pl.wide:
                H[f"{scope}.XL"], H[f"{scope}.XH"] = X[:, j] & (2**LIMB - 1), X[:, j] >> LIMB
            else:
                H[f"{scope}.X"] = X[:, j]
            H[f"{scope}.K"] = K[:, j]
            H[f"{scope}.s"] = sg[:, j]
        H[f"{tag}.C"] = first[:, 0] + (first[:, 1:] * X).sum(dim=1)
        sX = (1 - 2 * sg) * X
        Q = sX.sum(dim=1)
        dacc = E - e_st
        q_acc = (Mt << pl.ss) >> torch.clamp(dacc, max=pl.W)
        sign = 1 - 2 * s_st
        S = sign * q_acc + Q
        sgn = (S < 0).to(torch.int64)
        v = S.abs()
        H[f"{tag}.sgn"] = sgn
        if pl.wide:
            vL, vH = v & (2**LIMB - 1), v >> LIMB
            H[f"{tag}.vL"], H[f"{tag}.vH"] = vL, vH
            qL = q_acc & (2**LIMB - 1)
            H[f"{tag}.qL"], H[f"{tag}.qH"] = qL, q_acc >> LIMB
            SL = ((1 - 2 * sg) * (X & (2**LIMB - 1))).sum(dim=1)
            H[f"{tag}.carry"] = (sign * qL + SL - (1 - 2 * sgn) * vL) >> LIMB
        else:
            H[f"{tag}.v"] = v
            H[f"{tag}.q"] = q_acc
        l = _bit_length(v, pl.SB)
        e1 = torch.clamp(E + l - p.width, min=p.acc_e_min)
        tt = e1 - E + p.width - pl.nb
        Mn = torch.where(tt >= 0, v >> torch.clamp(tt, min=0), v << torch.clamp(-tt, min=0))
        for i in range(pl.n_t):
            H[f"{tag}.nsel[{i}]"] = (tt == pl.t_lo + i).to(torch.int64)
        hid, fo = Mn >> (pl.nb - 1), Mn & (2 ** (pl.nb - 1) - 1)
        H[f"{tag}.hid"], H[f"{tag}.fo"] = hid, fo
        zo = (Mn == 0).to(torch.int64)
        H[f"{tag}.z"], H[f"{tag}.ovf"] = zo, (e1 > p.acc_e_max).to(torch.int64)
        s_st = sgn * (1 - zo)
        e_st = e1 + zo * (p.floor - e1)
        Mt = Mn
    if chain and not pl.fp8:
        L = ACC_FRAC - p.frac
        H["epi.zl"] = ((fo & (2 ** (L - 1) - 1)) + ((fo >> L) & 1) == 0).to(torch.int64)
    names = [n for _, n in sys.hints]
    return torch.stack([H[n] for n in names])
