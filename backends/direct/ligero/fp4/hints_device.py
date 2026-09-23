"""The NVFP4 unit's public rows and hints on the device (lane fp4-fast): ``witness.public_vectors_fp4`` +
``witness.unit_hints`` (honest rule only) as vectorised torch int64 arithmetic over the columns, captured once per
(system, l, device) as a CUDA graph and replayed -- the same ``(n_hints, l)`` tensor, bit for bit, as the numpy hint
generator ``witness.hints_fp4`` (asserted by ``hints_device_test.py`` on >= 1e5 random units of every edge family and
on the bench instances; the A/B switch is ``FP4ChainRunner(hints=...)`` / ``run.py --fp4-hints numpy|device``).

Why bit-exact: every hint here is the same integer the numpy line computes -- int64 with the same wraparound, arithmetic
right shifts of the same signed values, the same shift amounts (all < 64: ``_bit_length`` is a binary search whose
largest shift is 63), ``floor`` division only of non-negative operands, one-hot placement as ``scatter_add_``.  The
witness builder reduces the hints mod p, so a hint is a witness value, not a residue: nothing here is reduced.

Consistency checks the numpy path makes with ``assert`` (the sum stays in binary32, the digit identity closes, exact
shifts are exact, coarse shifts fit) and the domain check of the decode are folded into one ``bad`` flag computed on
the device; the caller reads it once (``HintsDevice.__call__`` raises ``ValueError`` like the numpy path raises).  The
wrong-rule variants (``witness.VARIANTS``, the negatives) stay on the numpy path.
"""
from __future__ import annotations

import numpy as np
import torch

from ..compile import System
from .relation import (
    ACC_BITS,
    ACC_CAND_OFF,
    ACC_E_OFF,
    E_FIELD_MAX,
    FRAC_BITS,
    GRID_FLOOR,
    GROUP,
    GROUPS,
    K,
    MAG_BITS,
    NUMERATOR,
    POSITIONS,
    RADIX,
    WORDS,
    shifter_plan,
)

__all__ = ["HintsDevice", "PIN_NAMES", "hints_device_for", "public_rows_torch", "unit_hints_torch"]

MASK = 2**RADIX - 1
#: The pin names in the order ``compile_fp4_unit`` declares them (== ``[n for _, n in sys.pins]``, asserted by
#: ``HintsDevice``); ``public_rows_torch`` returns its rows in this order.
PIN_NAMES = tuple([f"a[{i}].n" for i in range(K)] + [f"b[{i}].n" for i in range(K)]
                  + [f"g[{g}].sm" for g in range(GROUPS)] + [f"g[{g}].X" for g in range(GROUPS)]
                  + [f"g[{g}].part" for g in range(GROUPS)] + ["anc.G"])
N_PINS = len(PIN_NAMES)
_SM, _X, _PART, _G = 2 * K, 2 * K + GROUPS, 2 * K + 2 * GROUPS, 2 * K + 3 * GROUPS


def _numerators(codes: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
    mag = num[codes & 7]
    return torch.where((codes & 8) != 0, -mag, mag)


def _scales(sc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e, m = (sc >> 3) & 0xF, sc & 7
    return torch.where(e == 0, m, 8 + m), torch.where(e == 0, torch.full_like(e, -9), e - 10)


def _bit_length(v: torch.Tensor) -> torch.Tensor:
    """``v.bit_length()`` for 0 <= v < 2^63 (numpy's ``_bit_length(v, 50)`` on the same range): 6 steps, shifts <= 63."""
    n = torch.zeros_like(v)
    for s in (32, 16, 8, 4, 2, 1):
        n = n + s * ((v >> (n + s)) != 0).to(torch.int64)
    return torch.where(v != 0, n + 1, torch.zeros_like(v))


def _domain_bad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(l,) bool: an operand byte off the domain (``witness._check_domain``: codes < 16, scales < 128 and != 0x7F)."""
    bad = torch.zeros(a.shape[0], dtype=torch.bool, device=a.device)
    for w in (a, b):
        codes, sc = w[:, :K], w[:, K:]
        bad |= ((codes < 0) | (codes >= 16)).any(1) | ((sc < 0) | (sc >= 128)).any(1) | (sc == 0x7F).any(1)
    return bad


def public_rows_torch(a: torch.Tensor, b: torch.Tensor, num: torch.Tensor | None = None) -> torch.Tensor:
    """``a``/``b`` (l, 68) int64 on the device -> (N_PINS, l) int64 in ``PIN_NAMES`` order: the verifier's decode
    (``witness.public_vectors_fp4`` without ``c``/``y``).  Signed values (numerators, X, G) are left signed."""
    if num is None:
        num = torch.tensor(NUMERATOR, dtype=torch.int64, device=a.device)
    l = a.shape[0]
    n_a, n_b = _numerators(a[:, :K], num), _numerators(b[:, :K], num)
    ma, ea = _scales(a[:, K:])
    mb, eb = _scales(b[:, K:])
    prod = (n_a * n_b).reshape(l, GROUPS, GROUP)
    part = (prod != 0).any(2) & (ma != 0) & (mb != 0)
    X = -2 + ea + eb
    G = torch.where(part, X - FRAC_BITS, torch.full_like(X, GRID_FLOOR)).max(1).values
    return torch.cat([n_a.t(), n_b.t(), (ma * mb).t(), X.t(), part.to(torch.int64).t(), G[None, :]], dim=0)


def _shift_hints(H: dict, path: str, d: torch.Tensor, mag: torch.Tensor, *, bits: int, n_j: int, exact: bool,
                 dead: bool, bad: torch.Tensor) -> torch.Tensor:
    """``witness._shift_hints`` (honest rule): the hints of one shifter under ``path`` and the placed digit vector
    (l, POSITIONS).  The numpy asserts become ``bad |= ...``."""
    l = d.shape[0]
    is_dead = d < -bits
    if not dead:
        bad |= is_dead
    u = torch.where(is_dead, torch.full_like(d, bits), torch.clamp(-d, min=0))
    L = torch.where(is_dead, torch.zeros_like(d), torch.clamp(d, min=0))
    q = mag >> u
    H[f"{path}.u"] = u
    if bits <= MAG_BITS:
        H[f"{path}.q"] = q
        parts = [(q, 0)]
    else:
        H[f"{path}.qh"], H[f"{path}.ql"] = q >> RADIX, q & MASK
        parts = [(q & MASK, 0), (q >> RADIX, 1)]
    if not exact:
        H[f"{path}.rem"] = mag - (q << u)
    else:
        bad |= (q << u) != mag
    j, r = L // RADIX, L % RADIX                                              # L >= 0: floor == trunc
    bad |= j >= n_j
    H[f"{path}.rlo"], H[f"{path}.rhi"] = r % 4, r // 4
    for jj in range(n_j):
        H[f"{path}.sel{jj}"] = (j == jj).to(torch.int64)
    if dead:
        H[f"{path}.dead"] = is_dead.to(torch.int64)
        H[f"{path}.k"] = torch.where(is_dead, -d - bits, torch.ones_like(d))
    D = torch.zeros((l, POSITIONS), dtype=torch.int64, device=d.device)
    jc = torch.clamp(j, max=n_j - 1)                                          # (a bad j never indexes out of D)
    for qp, off in parts:
        x2 = qp << r
        nbits = (bits if len(parts) == 1 else RADIX) + 11
        for c in range(-(-nbits // RADIX)):
            dig = (x2 >> (RADIX * c)) & MASK
            D.scatter_add_(1, (jc + (c + off))[:, None], dig[:, None])        # one-hot placement: position c + off + j
    return D


def unit_hints_torch(sys: System, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor, pub: torch.Tensor | None = None,
                     num: torch.Tensor | None = None, pos_w: torch.Tensor | None = None):
    """``witness.unit_hints`` (honest rule) on the device: ``c`` (l,) FP32 words, ``a``/``b`` (l, 68) int64 ->
    (``(n_hints, l)`` int64 in ``sys.hints`` order, ``bad`` (l,) bool: the columns on which the numpy path would have
    raised, ``y`` (l,) the output words).  ``pub``: the rows ``public_rows_torch`` computed (else recomputed).  Every op
    is graph-capturable (no host syncs, no host-side tensor construction: ``num`` / ``pos_w`` are passed in)."""
    l = c.shape[0]
    dev = c.device
    if num is None:
        num = torch.tensor(NUMERATOR, dtype=torch.int64, device=dev)
    if pos_w is None:
        pos_w = torch.ones(POSITIONS, dtype=torch.int64, device=dev) << (RADIX * torch.arange(POSITIONS, dtype=torch.int64, device=dev))
    bad = _domain_bad(a, b)
    H: dict[str, torch.Tensor] = {}
    # the accumulator
    s, t, f = c >> 31, (c >> 23) & 0xFF, c & (2**23 - 1)
    nz = (t != 0).to(torch.int64)
    z = ((t == 0) & (f == 0)).to(torch.int64)
    H.update({"acc.s": s, "acc.t": t, "acc.f": f, "acc.nz": nz, "acc.z": z, "acc.fh": f >> RADIX, "acc.fl": f & MASK})
    M = f + nz * 2 ** (ACC_BITS - 1)
    tp = t + 1 - nz
    acc_cand = torch.where(z == 1, torch.full_like(tp, GRID_FLOOR), tp - ACC_CAND_OFF)
    acc_e = tp - ACC_E_OFF
    # the operands (the decode, as the pins)
    if pub is None:
        pub = public_rows_torch(a, b, num)
    n_a, n_b = pub[:K].t(), pub[K:2 * K].t()                                  # (l, K)
    sm, X, part, G = pub[_SM:_X].t(), pub[_X:_PART].t(), pub[_PART:_G].t() == 1, pub[_G]
    prod = (n_a * n_b).reshape(l, GROUPS, GROUP)
    gd = prod.sum(2)
    mant = gd * sm
    lsb = torch.maximum(G, acc_cand)
    H["anc.lsb"] = lsb
    plan = shifter_plan()
    Psum = torch.zeros((l, POSITIONS), dtype=torch.int64, device=dev)
    for g in range(GROUPS):
        sgn = (mant[:, g] < 0).to(torch.int64)
        H[f"g{g}.sgn"] = sgn
        d = torch.where(part[:, g] | (mant[:, g] != 0), X[:, g] - lsb, torch.zeros_like(lsb))
        bits, n_j, exact, dead = plan[f"g{g}.shift"]
        D = _shift_hints(H, f"g{g}.shift", d, mant[:, g].abs(), bits=bits, n_j=n_j, exact=exact, dead=dead, bad=bad)
        Psum += (1 - 2 * sgn)[:, None] * D
    d_acc = torch.where(z == 1, torch.zeros_like(lsb), acc_e - lsb)
    bits, n_j, exact, dead = plan["accterm.shift"]
    D = _shift_hints(H, "accterm.shift", d_acc, M, bits=bits, n_j=n_j, exact=exact, dead=dead, bad=bad)
    Psum += (1 - 2 * s)[:, None] * D
    # the exact sum and its RZ normalisation
    T = (Psum * pos_w[None, :]).sum(1)
    sgn_t = (T < 0).to(torch.int64)
    magT = T.abs()
    nzM = (magT != 0).to(torch.int64)
    bl = _bit_length(magT)
    e_val = bl - 1 + lsb
    normal = e_val >= -126
    grid = torch.where(normal, e_val - 23, torch.full_like(e_val, -149))
    sigma = torch.where(nzM == 1, grid - lsb, torch.zeros_like(grid))
    sig_pos, sig_neg = torch.clamp(sigma, min=0), torch.clamp(-sigma, min=0)
    Mo = torch.where(sigma >= 0, magT >> sig_pos, magT << sig_neg)
    Mo = torch.where(nzM == 1, Mo, torch.zeros_like(Mo))
    R = torch.where(sigma >= 0, magT - (Mo << sig_pos), torch.zeros_like(magT))
    t_o = torch.where((nzM == 1) & normal, e_val + 127, torch.zeros_like(e_val))
    hid = ((nzM == 1) & normal).to(torch.int64)
    bad |= t_o > E_FIELD_MAX                                                  # the sum leaves binary32 (D20)
    bad |= (Mo >> (ACC_BITS - 1)) != hid
    hid = (Mo >> (ACC_BITS - 1)) & 1
    f_o = Mo & (2 ** (ACC_BITS - 1) - 1)
    sign_out = sgn_t * nzM
    H.update({"out.sgn": sgn_t, "out.t": t_o, "out.hid": hid, "out.fh": f_o >> RADIX, "out.fl": f_o & MASK, "out.nzM": nzM})
    bits, n_j, exact, dead = plan["out.shift"]
    DM = _shift_hints(H, "out.shift", sigma, Mo, bits=bits, n_j=n_j, exact=exact, dead=dead, bad=bad)
    for pn in range(n_j):
        H[f"out.r{pn}"] = (R >> (RADIX * pn)) & MASK
    bad |= (R >> (RADIX * n_j)) != 0
    carry = torch.zeros_like(T)
    sT = 1 - 2 * sgn_t
    for pn in range(POSITIONS):
        S = sT * Psum[:, pn] - DM[:, pn] + carry
        if pn < n_j:
            S = S - H[f"out.r{pn}"]
        bad |= (S & MASK) != 0                                                # the carry chain closes
        carry = S >> RADIX
        if pn < POSITIONS - 1:
            H[f"out.c{pn + 1}"] = carry
    bad |= carry != 0                                                         # the digit identity closes
    y = (sign_out << 31) | (t_o << 23) | f_o
    return torch.stack([H[n] for _, n in sys.hints]), bad, y


class HintsDevice:
    """The prover's per-sub-batch device work for one (system, l, device): copy ``c`` (l,) and ``a``/``b`` (l, 68)
    into static buffers (the operands as bytes: 8x less H2D than int64), replay the captured graph (eager on CPU / when
    capture is unavailable), and return ``(pub_rows (N_PINS, l) int64, hints (n_hints, l) int64, y (l,) int64)``.
    Raises ``ValueError`` when a column is off the domain or its hints do not close (the numpy path's asserts)."""

    def __init__(self, sys: System, l: int, device, graph: bool = True):
        if tuple(n for _, n in sys.pins) != PIN_NAMES:
            raise ValueError("the system's pins are not the fp4-nvf4 pins in compile order")
        self.sys, self.l, self.device = sys, l, torch.device(device)
        dev = self.device
        self.num = torch.tensor(NUMERATOR, dtype=torch.int64, device=dev)
        self.pos_w = torch.ones(POSITIONS, dtype=torch.int64, device=dev) << (RADIX * torch.arange(POSITIONS, dtype=torch.int64, device=dev))
        self.c = torch.zeros(l, dtype=torch.int64, device=dev)
        self.a8 = torch.zeros((l, WORDS), dtype=torch.uint8, device=dev)
        self.b8 = torch.zeros((l, WORDS), dtype=torch.uint8, device=dev)
        self.graph = None
        self.captured = False
        if graph and dev.type == "cuda":
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._compute()
            torch.cuda.current_stream(dev).wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.out = self._compute()
            self.graph = g
            self.captured = True

    def _compute(self):
        a, b = self.a8.to(torch.int64), self.b8.to(torch.int64)
        pub = public_rows_torch(a, b, self.num)
        hints, bad, y = unit_hints_torch(self.sys, self.c, a, b, pub, self.num, self.pos_w)
        return pub, hints, bad.any(), y

    @staticmethod
    def _bytes(w) -> torch.Tensor:
        w = np.asarray(w)
        if w.dtype != np.uint8:
            if w.size and (w.min() < 0 or w.max() > 255):
                raise ValueError("public inputs rejected: an operand word is not a byte")
            w = w.astype(np.uint8)
        return torch.from_numpy(np.ascontiguousarray(w))

    def __call__(self, c, a, b, check: bool = True):
        """``c`` (l,) FP32 words, ``a``/``b`` (l, 68) operand words (numpy / torch, integer dtypes)."""
        if isinstance(c, torch.Tensor):
            self.c.copy_(c.to(torch.int64))
        else:
            self.c.copy_(torch.from_numpy(np.ascontiguousarray(np.asarray(c, dtype=np.int64))))
        self.a8.copy_(a.to(torch.uint8) if isinstance(a, torch.Tensor) else self._bytes(a))
        self.b8.copy_(b.to(torch.uint8) if isinstance(b, torch.Tensor) else self._bytes(b))
        if self.graph is not None:
            self.graph.replay()
            pub, hints, bad, y = self.out
            pub, hints, y = pub.clone(), hints.clone(), y.clone()
        else:
            pub, hints, bad, y = self._compute()
        if check and bool(bad):
            raise ValueError("fp4 device hints: a unit is off the domain or its hints do not close (the numpy path would "
                             "have raised); run with hints='numpy' for the message")
        return pub, hints, y


_CACHE: dict = {}


def hints_device_for(sys: System, l: int, device, graph: bool = True, slot=None) -> HintsDevice:
    """The cached generator for (system, l, device[, slot]).  ``slot``: a pipeline slot index -- sub-batches in flight
    on different streams need their own instance (the graph's static input / output buffers are not shareable); the
    graph is captured on the caller's current stream context and replays on the current stream."""
    key = (id(sys), l, str(device), graph, slot)
    hit = _CACHE.get(key)
    if hit is None or hit[0] is not sys:
        hit = (sys, HintsDevice(sys, l, device, graph=graph))
        _CACHE[key] = hit
    return hit[1]
