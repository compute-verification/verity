"""The logUp fractional-sum GKR prover on the GPU (Triton kernels from :mod:`kernels_triton`), bit-exact against
:mod:`logup_reference` (same coins -> same transcript).

One table ``T`` (``Nt`` rows, base values ``t_v``) and the witness queries ``w_j`` (``Nw`` base values).  Leaves
``(1, z - w_j) | (-m_v, z - t_v) | (0, 1) pad``, ``2^n`` of them; level ``k`` has ``2^k`` fractions; the sumcheck at level
``k`` (``k`` rounds, low bit first) is over the pairs of level ``k + 1``.

What runs where (DESIGN.md 3 applied to logUp):

* **multiplicities**: ``torch.bincount`` (``t_v = v`` range tables) or sort + ``searchsorted``.
* **tree values**: :func:`kernels_triton.leaf_level` forms level ``n - 1`` straight from the base queries (the leaves are
  never materialised: ``p = 2 z - (w0 + w1)``, ``q = z^2 - z (w0 + w1) + w0 w1``); :func:`kernels_triton.tree_level`
  the levels above, each stored *de-interleaved* (``[evens | odds]``) so a level doubles as the ``p0, p1, q0, q1``
  operands of the sumcheck below it.
* **the big sumcheck (level ``n - 1``, over the leaf pairs)**: the witness region's leaves are ``(1, z - w)`` with ``w``
  base, so ``A_x = 2 z - s_x`` and ``B_x = z^2 - z s_x + pi_x`` with ``s = w_e + w_o`` and ``pi = w_e w_o`` *base*
  tables: the first ``k`` rounds are packed rounds on ``w_e, w_o`` (:func:`kernels_triton.logup_packed_inner_product`:
  ``half^2`` base Hadamards + ``half`` linear rows per ``x``), the table region (65,536 leaves) runs the generic
  extension kernel on its own eq-weight slice, the padding region is a constant (``A = 0, B = 1``) whose eq-sum is
  added analytically.  Regions are aligned to the eq split (``K1`` divides every region's ``b``-range), so the
  multilinear extensions restrict cleanly.  After round ``k`` the whole level is assembled as four extension tables
  (:func:`kernels_triton.fold_k` on ``w_e, w_o``; ``p0 = p1 = 1`` in the witness region) and the remaining rounds are
  generic.
* **generic rounds** (every other level): :func:`kernels_triton.logup_ext_inner_product` (the two ``F_{p^6}``
  Hadamards ``A = p0 q1 + p1 q0``, ``B = q0 q1`` per pair and ``x``, limbs in registers, ``tl.dot`` against the eq
  limbs), :func:`kernels_triton.reduce_contract`, :func:`kernels_triton.finish_round` with ``coeff = [1, lambda]``,
  :func:`kernels_triton.fold_ext4`.  Levels too small for the fused kernel (``< 128`` pairs per eq chunk) run in torch.

Coins are drawn from a :class:`logup_reference.Coins` in protocol order (``lambda_0``; per level ``r_1..r_k, mu,
lambda``): the coins do not depend on the transcript, so the device is never synchronised mid-proof here (Fiat-Shamir
would add one host round trip per round, ``n (n - 1) / 2`` of them).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from . import logup_reference as lref
from . import reference as ref
from . import sumcheck_packed as sp

P = ref.P
Ext = ref.Ext


def to_dev(v, device) -> torch.Tensor:
    return sp.to_dev(v, device)


def multiplicities(w: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``m_v`` = number of queries equal to ``table[v]`` (``int32 (Nt,)``); queries outside the table count nowhere."""
    Nt = table.numel()
    if bool((table == torch.arange(Nt, device=table.device, dtype=table.dtype)).all()):
        return torch.bincount(w[w < Nt], minlength=Nt)[:Nt].to(torch.int32)
    order = torch.argsort(table)
    st = table[order]
    pos = torch.searchsorted(st, w)
    pos = pos.clamp(max=Nt - 1)
    hit = st[pos] == w
    m_sorted = torch.bincount(pos[hit], minlength=Nt)
    m = torch.zeros(Nt, dtype=torch.int64, device=table.device)
    m[order] = m_sorted
    return m.to(torch.int32)


@dataclass
class Timing:
    """CUDA-event phase times (ms); ``levels[k]`` the whole of level ``k``'s sumcheck."""

    histogram: float = 0.0
    tree: float = 0.0
    big_packed: float = 0.0        # level n-1: the k packed rounds (witness + table + pad)
    big_fold_k: float = 0.0        # level n-1: fold_k + assembling the four extension tables
    big_ext: float = 0.0           # level n-1: the remaining generic rounds
    levels: dict = field(default_factory=dict)
    total: float = 0.0


class LogUpProver:
    def __init__(self, device: str = "cuda", k: int = 3, k1_log2: int = sp.K1_LOG2_DEFAULT):
        self.device = torch.device(device)
        self.k = k
        self.kernel = sp.Kernel(device, k1_log2=k1_log2, mode="triton")
        self.tk = self.kernel._tk
        self.pw16 = self.kernel.pw16
        self.one6 = self.kernel.one6

    # -- small helpers ------------------------------------------------------------------------------------------------

    def _ev(self):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _inputs(self, rho: list[Ext], rs: list[Ext], i: int, K1: int) -> sp.RoundInputs:
        """Round ``i`` eq inputs at an explicit inner depth ``K1`` (``e_hi`` covers the whole level)."""
        kappa = int(math.log2(K1))
        e_lo = sp.pre_scale(sp.eq_table_dev(rho[i:i + kappa], self.device))
        e_hi = sp.eq_table_dev(rho[i + kappa:], self.device)
        gm = rho[i - 1]
        eq_x = to_dev([ref.eval_at(ref.sub(ref.ONE, gm), gm, x) for x in range(4)], self.device)
        prefix = to_dev(ref.eq_points(rho[:i - 1], rs), self.device)
        inp = sp.RoundInputs(sp.decompose(e_lo), e_hi, eq_x, prefix, None, None)
        inp.e_lo32 = torch.nn.functional.pad(inp.e_lo_limbs, (0, 32 - sp.DL))
        return inp

    def _fold4(self, T4: torch.Tensor, r: Ext) -> torch.Tensor:
        m = T4.shape[1] // 2
        out = torch.empty((4, m, 6), dtype=torch.int32, device=self.device)
        r_R = sp.pre_scale(to_dev([r], self.device))[0].contiguous()
        return self.tk.fold_ext4(T4, m, r_R, out)

    def _ext_ip(self, T4: torch.Tensor, e_lo32: torch.Tensor, e_hi: torch.Tensor, K1: int, K2: int) -> torch.Tensor:
        """``(3, 2, 6)``: ``sum_b e(b) A_x(b)`` and ``sum_b e(b) B_x(b)`` (plain) over the ``K2`` chunks of ``T4``'s pairs."""
        split = sp.Kernel.split_b1(K1, K2)
        Zs = self.tk.logup_ext_inner_product(T4, e_lo32, K1, K2, split)
        Z = Zs.to(torch.int64).sum(0) if split > 1 else Zs[0]
        return self.tk.reduce_contract(Z, e_hi.contiguous(), self.pw16, K2, 6, 6).view(3, 2, 6)

    # -- one generic round --------------------------------------------------------------------------------------------

    def generic_round(self, T4: torch.Tensor, rho: list[Ext], rs: list[Ext], i: int, lam6: torch.Tensor) -> torch.Tensor:
        """Level with ``len(rho)`` rounds, round ``i`` on ``T4 (4, t_i, 6)`` = ``p0, p1, q0, q1``: ``S_i(0..3)`` as ``(4, 6)``."""
        n = len(rho)
        m = 1 << (n - i)
        K1, K2 = self.kernel.split(m)
        if K1 % 128 == 0:
            inp = self._inputs(rho, rs, i, K1)
            Q = self._ext_ip(T4, inp.e_lo32, inp.e_hi, K1, K2)
            coeff = torch.stack([self.one6, lam6], 0)
            return self.tk.finish_round(Q, coeff, inp.prefix, inp.eq_x)
        inp = self._inputs(rho, rs, i, 1)
        V = sp.Kernel._three_points(T4.view(4, m, 2, 6), 2)                     # (4, m, 3, 6)
        A = sp.add_mod(sp.ext_mul(V[0], V[3]), sp.ext_mul(V[1], V[2]))
        B = sp.ext_mul(V[2], V[3])
        U = sp.add_mod(A, sp.ext_mul(B, lam6[None, None]))                       # (m, 3, 6)
        e = inp.e_hi                                                              # (m, 6): eq over all of b
        Q = (sp.ext_mul(U, e[:, None, :]).to(torch.int64).sum(0) % P).to(torch.int32)
        return self.kernel._q_to_s(Q, inp)

    def generic_level(self, T4: torch.Tensor, rho: list[Ext], lam: Ext, coins: lref.Coins, rs: list[Ext], rounds: list,
                      i0: int = 1) -> torch.Tensor:
        """Rounds ``i0..len(rho)`` with folds; returns the ``(4, 1, 6)`` final tables."""
        lam6 = to_dev([lam], self.device)
        for i in range(i0, len(rho) + 1):
            rounds.append(self.generic_round(T4, rho, rs, i, lam6))
            r = coins.next()
            rs.append(r)
            T4 = self._fold4(T4, r)
        return T4

    # -- the big level (n - 1) ----------------------------------------------------------------------------------------

    def _k1_regions(self, i: int, ranges: list[int]) -> int:
        K1 = 1 << self.kernel.k1_log2
        for R in ranges:
            if R:
                K1 = min(K1, R & -R)
        if K1 % 128:
            raise ValueError(f"round {i}: region ranges {ranges} are not aligned to 128 eq-chunks")
        return K1

    def _big_level(self, w_e: torch.Tensor, w_o: torch.Tensor, T4_tab: torch.Tensor, n: int, rho: list[Ext], z: Ext,
                   lam: Ext, coins: lref.Coins, rounds: list, tm: Timing) -> torch.Tensor:
        """Level ``n - 1``'s sumcheck over the leaf pairs: ``k`` packed rounds on the de-interleaved base queries
        ``w_e, w_o`` (``(Nw / 2,)`` each) + the table leaves ``T4_tab (4, Nt / 2, 6)`` + the analytic pad, then
        ``fold_k`` and the generic rounds.  Returns the ``(4, 1, 6)`` final tables; ``self._big_rs`` the challenges."""
        Nw, Nt = 2 * w_e.numel(), 2 * T4_tab.shape[1]
        k_lvl = n - 1
        k = min(self.k, k_lvl)
        dev = self.device
        lam6 = to_dev([lam], dev)
        z6 = to_dev([z], dev)
        c1 = to_dev([ref.add(ref.add(z, z), ref.mul(lam, ref.mul(z, z)))], dev)[0]   # 2 z + lambda z^2
        c2 = to_dev([ref.add(ref.ONE, ref.mul(lam, z))], dev)[0]                    # 1 + lambda z
        rs: list[Ext] = []
        t0 = self._ev()
        for i in range(1, k + 1):
            half = 1 << (i - 1)
            rows = 3 * (half * half + half)
            Rw, Rt = Nw >> (i + 1), Nt >> (i + 1)          # b-ranges of the regions: pairs = leaves / 2, then >> i
            Rp = (1 << (k_lvl - i)) - Rw - Rt
            K1 = self._k1_regions(i, [Rw, Rt, Rp])
            inp = self._inputs(rho, rs, i, K1)
            Bw, Bt = Rw // K1, Rt // K1
            e_hi = inp.e_hi
            # witness region: packed
            split = sp.Kernel.split_b1(K1, Bw)
            Zs = self.tk.logup_packed_inner_product(w_e, w_o, inp.e_lo32, i, K1, Bw, split)
            Z = Zs.to(torch.int64).sum(0) if split > 1 else Zs[0]
            Tw = self.tk.reduce_contract(Z, e_hi[:Bw].contiguous(), self.pw16, Bw, rows, 1).view(3, half * half + half, 6)
            eq_r = ref.eq_table(rs)
            c_prod = to_dev([ref.mul(a, b) for a in eq_r for b in eq_r], dev)
            c_lin = to_dev(eq_r, dev)
            pi = (sp.ext_mul(Tw[:, :half * half], c_prod[None]).to(torch.int64).sum(1) % P).to(torch.int32)   # (3, 6)
            s = (sp.ext_mul(Tw[:, half * half:], c_lin[None]).to(torch.int64).sum(1) % P).to(torch.int32)
            Ew = (e_hi[:Bw].to(torch.int64).sum(0) % P).to(torch.int32)
            Q = sp.add_mod(sp.sub_mod(sp.ext_mul(c1[None], Ew[None]).expand(3, 6), sp.ext_mul(c2[None], s)),
                           sp.ext_mul(lam6, pi))
            # table region: generic on its own eq-weight slice
            if Bt:
                Qt = self._ext_ip(T4_tab, inp.e_lo32, e_hi[Bw:Bw + Bt], K1, Bt)
                Q = sp.add_mod(Q, sp.add_mod(Qt[:, 0], sp.ext_mul(lam6, Qt[:, 1])))
            # padding: A = 0, B = 1
            if Rp:
                Ep = (e_hi[Bw + Bt:].to(torch.int64).sum(0) % P).to(torch.int32)
                Q = sp.add_mod(Q, sp.ext_mul(lam6, Ep[None]).expand(3, 6))
            rounds.append(self.kernel._q_to_s(Q, inp))
            r = coins.next()
            rs.append(r)
            if Bt:
                T4_tab = self._fold4(T4_tab, r)
        t1 = self._ev()
        # fold_k on the base queries and the assembly of the four extension tables
        M = 1 << (k_lvl - k)
        T4 = torch.empty((4, M, 6), dtype=torch.int32, device=dev)
        Mw, Mt = Nw >> (k + 1), Nt >> (k + 1)
        if k:
            coeffs_R = sp.pre_scale(to_dev(ref.eq_table(rs), dev))
            We = self.tk.fold_k(w_e, k, coeffs_R)
            Wo = self.tk.fold_k(w_o, k, coeffs_R)
        else:
            We = torch.zeros((Mw, 6), dtype=torch.int32, device=dev)
            We[:, 0] = w_e
            Wo = torch.zeros((Mw, 6), dtype=torch.int32, device=dev)
            Wo[:, 0] = w_o
        T4[0, :Mw] = self.one6
        T4[1, :Mw] = self.one6
        T4[2, :Mw] = sp.sub_mod(z6, We)
        T4[3, :Mw] = sp.sub_mod(z6, Wo)
        T4[:, Mw:Mw + Mt] = T4_tab
        T4[0:2, Mw + Mt:] = 0
        T4[2:4, Mw + Mt:] = self.one6
        t2 = self._ev()
        T4 = self.generic_level(T4, rho, lam, coins, rs, rounds, i0=k + 1)
        t3 = self._ev()
        torch.cuda.synchronize()
        tm.big_packed = t0.elapsed_time(t1)
        tm.big_fold_k = t1.elapsed_time(t2)
        tm.big_ext = t2.elapsed_time(t3)
        self._big_rs = rs
        return T4

    # -- the whole proof ----------------------------------------------------------------------------------------------

    def prove(self, w: torch.Tensor, table: torch.Tensor, z: Ext, coins: lref.Coins, mult: torch.Tensor | None = None
              ) -> tuple[lref.Transcript, lref.LeafClaim, Timing]:
        """``w (Nw,) int32`` queries (natural order), ``table (Nt,) int32``.  Same messages as
        ``logup_reference.prove_table(*leaves(w, table, mult, z), coins)``."""
        dev = self.device
        tm = Timing()
        T_start = self._ev()
        Nw, Nt = w.numel(), table.numel()
        n = lref.log2ceil(Nw + Nt)
        k_pack = min(self.k, n - 1)
        if Nw % (1 << (k_pack + 1)) or Nt % (1 << (k_pack + 1)):
            raise ValueError("Nw and Nt must be multiples of 2^(k+1)")
        if mult is None:
            mult = multiplicities(w, table)
        t_hist = self._ev()
        # leaves of the table region as four extension tables (p0, p1, q0, q1 over the table's pairs)
        z6 = to_dev([z], dev)
        pt = torch.zeros((Nt, 6), dtype=torch.int32, device=dev)
        pt[:, 0] = torch.where(mult == 0, torch.zeros_like(mult), P - mult)
        qt = z6.expand(Nt, 6).clone()
        qt[:, 0] = sp.sub_mod(z6[0].expand(Nt), table)
        T4_tab = torch.stack([pt[0::2], pt[1::2], qt[0::2], qt[1::2]], 0).contiguous()
        # level n-1 values, de-interleaved: witness pairs from the kernel, table pairs in torch, pad (0, 1)
        half_total = 1 << (n - 2)
        pq = torch.empty((2, 1 << (n - 1), 6), dtype=torch.int32, device=dev)
        pq[0] = 0
        pq[1] = self.one6
        zR = sp.pre_scale(z6)[0].contiguous()
        z2 = to_dev([ref.mul(z, z)], dev)[0].contiguous()
        self.tk.leaf_level(w, z6[0].contiguous(), zR, z2, pq[0], pq[1], half_total)
        x = torch.arange(Nw // 2, (Nw + Nt) // 2, device=dev)
        dst = torch.where(x % 2 == 0, x // 2, half_total + x // 2)
        pq[0][dst] = sp.add_mod(sp.ext_mul(T4_tab[0], T4_tab[3]), sp.ext_mul(T4_tab[1], T4_tab[2]))
        pq[1][dst] = sp.ext_mul(T4_tab[2], T4_tab[3])
        levels = {n - 1: pq}
        for lv in range(n - 1, 0, -1):
            levels[lv - 1] = self.tk.tree_level(levels[lv])
        t_tree = self._ev()
        w_e = w.view(-1, 2)[:, 0].contiguous()
        w_o = w.view(-1, 2)[:, 1].contiguous()
        # the GKR walk, root first
        q_root = tuple(int(v) for v in levels[0][1][0].tolist())
        tr = lref.Transcript(q_root)
        lam = coins.next()
        rho: list[Ext] = []
        vals = None
        mu = None
        for lv in range(n):
            e0 = self._ev()
            rounds: list = []
            if lv == n - 1:
                T4 = self._big_level(w_e, w_o, T4_tab, n, rho, z, lam, coins, rounds, tm)
                rs = self._big_rs
            else:
                rs = []
                T4 = levels[lv + 1].view(4, 1 << lv, 6)
                T4 = self.generic_level(T4, rho, lam, coins, rs, rounds)
            vals = T4[:, 0]
            tr.levels.append({"rounds": [sp.S_to_ext(S) for S in rounds], "vals": sp.S_to_ext(vals)})
            mu = coins.next()
            lam = coins.next()
            rho = [mu] + rs
            e1 = self._ev()
            torch.cuda.synchronize()
            tm.levels[lv] = e0.elapsed_time(e1)
        T_end = self._ev()
        torch.cuda.synchronize()
        tm.histogram = T_start.elapsed_time(t_hist)
        tm.tree = t_hist.elapsed_time(t_tree)
        tm.total = T_start.elapsed_time(T_end)
        v = tr.levels[-1]["vals"] if n else None
        if n:
            p_next = ref.add(v[0], ref.mul(mu, ref.sub(v[1], v[0])))
            q_next = ref.add(v[2], ref.mul(mu, ref.sub(v[3], v[2])))
        else:
            p_next, q_next = ref.ZERO, q_root
        return tr, lref.LeafClaim(rho, lam, ref.add(p_next, ref.mul(lam, q_next))), tm
