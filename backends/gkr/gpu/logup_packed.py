"""LogUp fractional GKR on the a-packed2 kernels (``../packed/logup_gpu.py``) with the Fiat-Shamir transcript.

Same protocol as ``logup.prove_table`` (the torch reference of the low-bit-first schedule): the packed prover draws
its "coins" from a :class:`FSCoins` whose ``next()`` first absorbs every round message produced since the previous
challenge (the ``rounds`` list the packed level loop appends to) and then squeezes the transcript -- so the message
order and the challenge derivation are exactly those of the torch path, one device-to-host sync per round.

* ``R16`` (single base column, 65 536 rows): the whole ``LogUpProver`` machinery -- ``leaf_level`` straight from the
  int32 queries, ``tree_level``, the big level with ``k`` *packed base rounds* on the de-interleaved queries
  (``k = 3`` when the region alignment allows, else the largest that does, else 0: the messages do not depend on ``k``).
* the vector-valued tables (POW, ALIGN, LEAD, NORM: ``q = z - Σ β^j v_j`` is an extension element, so the packed
  base rounds do not apply): leaves from ``logup.build_leaves``, level ``n-1`` by the a-gpu ``_combine_level`` kernel,
  then ``tree_level`` and the generic four-table rounds of the packed prover.
"""
from __future__ import annotations

import torch

from . import logup
from .field import DEG, P, W, Ext, py_ext, py_exts
from .logup import LeafClaim
from .transcript import Transcript

try:
    from packed import logup_gpu as _lg
    from packed import sumcheck_packed as _sp
    from packed.logup_reference import Coins as _Coins
    AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        from backends.gkr.packed import logup_gpu as _lg
        from backends.gkr.packed import sumcheck_packed as _sp
        from backends.gkr.packed.logup_reference import Coins as _Coins
        AVAILABLE = True
    except ImportError:
        AVAILABLE = False

_PROVER = None

if AVAILABLE and (_lg.ref.W != W or _lg.ref.P != P):
    raise ImportError(f"packed kernels are generated for X^6 = {_lg.ref.W}, gpu.field uses {W}")


def get_prover(device) -> "_lg.LogUpProver":
    global _PROVER
    if _PROVER is None:
        _PROVER = _lg.LogUpProver(str(device), k=3)
    return _PROVER


class FSCoins(_Coins if AVAILABLE else object):
    """``coins.next()`` = absorb the pending round messages, then one transcript challenge."""

    def __init__(self, tr: Transcript, out: list[Ext]) -> None:
        self.tr, self.out = tr, out
        self.rounds: list | None = None
        self.n_abs = 0
        self.syncs = 0

    def bind(self, rounds: list) -> None:
        self.rounds, self.n_abs = rounds, 0

    def flush(self) -> None:
        if self.rounds is not None and self.n_abs < len(self.rounds):
            pend = self.rounds[self.n_abs:]
            hs = py_exts(torch.stack(pend).to(torch.int64)) if len(pend) > 1 else py_exts(pend[0].to(torch.int64))
            self.syncs += 1
            self.out.extend(hs)
            self.tr.absorb_exts(hs)
            self.n_abs = len(self.rounds)

    def next(self) -> Ext:
        self.flush()
        return self.tr.challenge()


def _pick_k(Nw: int, Nt: int, n: int, kmax: int = 3) -> int:
    """Largest ``k <= kmax`` for which every region of the big level is aligned to 128 eq chunks through round ``k``."""
    for k in range(kmax, 0, -1):
        if Nw % (1 << (k + 1)) or Nt % (1 << (k + 1)):
            continue
        ok = True
        for i in range(1, k + 1):
            Rw, Rt = Nw >> (i + 1), Nt >> (i + 1)
            Rp = (1 << (n - 1 - i)) - Rw - Rt
            for R in (Rw, Rt, Rp):
                if R and (R & -R) < 128:
                    ok = False
        if ok:
            return k
    return 0


def _deinterleave(v: torch.Tensor) -> torch.Tensor:
    return torch.cat([v[0::2], v[1::2]], dim=0)


def _walk(lp, levels: dict, n: int, tr: Transcript, out: list[Ext], big) -> LeafClaim:
    """The GKR walk root -> leaves; ``big(rho, lam, coins, rounds)`` runs level ``n-1`` and returns ``(T4, rs)``."""
    root_p = levels[0][0][0].to(torch.int64)
    root_q = py_ext(levels[0][1][0].to(torch.int64))
    if bool((root_p % P).any()):
        raise ValueError("LogUp: fractional sum is not zero (a query is not in its table)")
    if not any(root_q):
        raise ValueError("LogUp: q_root is zero")
    out.append(root_q)
    tr.absorb_ext(root_q)
    coins = FSCoins(tr, out)
    lam = coins.next()
    rho: list[Ext] = []
    for lv in range(n):
        rounds: list = []
        coins.bind(rounds)
        if lv == n - 1:
            T4, rs = big(rho, lam, coins, rounds)
        else:
            rs = []
            T4 = levels[lv + 1].view(4, 1 << lv, 6)
            T4 = lp.generic_level(T4, rho, lam, coins, rs, rounds)
        levels[lv + 1] = None
        coins.flush()
        vals = py_exts(T4[:, 0].to(torch.int64))
        out.extend(vals)
        tr.absorb_exts(vals)
        mu = coins.next()
        lam = coins.next()
        rho = [mu] + rs
    p_next = logup.e_add(vals[0], logup.e_mul(mu, logup.e_sub(vals[1], vals[0])))
    q_next = logup.e_add(vals[2], logup.e_mul(mu, logup.e_sub(vals[3], vals[2])))
    tr.logup_syncs = getattr(tr, "logup_syncs", 0) + coins.syncs
    return logup.leaf_claim_top(rho, lam, logup.e_add(p_next, logup.e_mul(lam, q_next)))


def prove_range_table(w: torch.Tensor, table: torch.Tensor, mult: torch.Tensor, z: Ext, n: int, tr: Transcript,
                      out: list[Ext]) -> LeafClaim:
    """Single-column base table (R16): ``w (Nw,)`` int32 queries in leaf order, ``table (Nt,)`` int32, ``mult`` (Nt,)."""
    lp = get_prover(w.device)
    dev = lp.device
    Nw, Nt = w.numel(), table.numel()
    if n != _lg.lref.log2ceil(Nw + Nt):
        raise ValueError("hypercube dimension mismatch")
    k = _pick_k(Nw, Nt, n, lp.k)
    z6 = _sp.to_dev([z], dev)
    pt = torch.zeros((Nt, 6), dtype=torch.int32, device=dev)
    pt[:, 0] = torch.where(mult == 0, torch.zeros_like(mult), P - mult)
    qt = z6.expand(Nt, 6).clone()
    qt[:, 0] = _sp.sub_mod(z6[0, 0].expand(Nt), table)
    T4_tab = torch.stack([pt[0::2], pt[1::2], qt[0::2], qt[1::2]], 0).contiguous()
    half_total = 1 << (n - 2)
    pq = torch.empty((2, 1 << (n - 1), 6), dtype=torch.int32, device=dev)
    pq[0] = 0
    pq[1] = lp.one6
    zR = _sp.pre_scale(z6)[0].contiguous()
    z2 = _sp.to_dev([_lg.ref.mul(z, z)], dev)[0].contiguous()
    lp.tk.leaf_level(w, z6[0].contiguous(), zR, z2, pq[0], pq[1], half_total)
    x = torch.arange(Nw // 2, (Nw + Nt) // 2, device=dev)
    dst = torch.where(x % 2 == 0, x // 2, half_total + x // 2)
    mul = lp.tk.ext_mul_rows
    pq[0][dst] = _sp.add_mod(mul(T4_tab[0], T4_tab[3]), mul(T4_tab[1], T4_tab[2]))
    pq[1][dst] = mul(T4_tab[2], T4_tab[3])
    levels = {n - 1: pq}
    for lv in range(n - 1, 0, -1):
        levels[lv - 1] = lp.tk.tree_level(levels[lv])
    w_e = w.view(-1, 2)[:, 0].contiguous()
    w_o = w.view(-1, 2)[:, 1].contiguous()
    lp.k, k_saved = k, lp.k

    def big(rho, lam, coins, rounds):
        T4 = lp._big_level(w_e, w_o, T4_tab, n, rho, z, lam, coins, rounds, _lg.Timing())
        return T4, lp._big_rs

    try:
        return _walk(lp, levels, n, tr, out, big)
    finally:
        lp.k = k_saved


def prove_ext_table(p_leaf: torch.Tensor, q_leaf: torch.Tensor, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
    """Vector-valued table: ``p_leaf (2^n,)`` int32 base, ``q_leaf (2^n, 6)`` int32 (``logup.build_leaves``)."""
    lp = get_prover(q_leaf.device)
    np_, nq = logup._combine_level(p_leaf, q_leaf)                    # level n-1, natural order
    pq = torch.stack([_deinterleave(np_), _deinterleave(nq)], 0).contiguous()
    levels = {n - 1: pq}
    for lv in range(n - 1, 0, -1):
        levels[lv - 1] = lp.tk.tree_level(levels[lv])
    half = 1 << (n - 1)
    T4 = torch.zeros((4, half, 6), dtype=torch.int32, device=q_leaf.device)
    T4[0, :, 0] = p_leaf[0::2]
    T4[1, :, 0] = p_leaf[1::2]
    T4[2] = q_leaf[0::2]
    T4[3] = q_leaf[1::2]

    def big(rho, lam, coins, rounds):
        rs: list[Ext] = []
        return lp.generic_level(T4, rho, lam, coins, rs, rounds), rs

    return _walk(lp, levels, n, tr, out, big)


__all__ = ["AVAILABLE", "prove_range_table", "prove_ext_table", "FSCoins", "DEG"]
