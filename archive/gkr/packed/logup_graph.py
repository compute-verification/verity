"""The logUp fractional GKR prover of :mod:`logup_gpu` with the launch loop removed: every level's sumcheck (all its
rounds, folds included) is ONE CUDA graph replayed per proof; the tree build is another.  Bit-exact against
:mod:`logup_reference` (same coins -> same transcript): the messages are field elements and do not depend on where
the round constants were computed.

Why a graph per level is transcript-preserving here, and what Fiat-Shamir changes
---------------------------------------------------------------------------------
Inside a graph no host code runs, so a round cannot ask the verifier (or the sponge) for its challenge.  The graph
therefore reads the challenges ``r_1..r_lv`` of the level from a static device buffer ``coins (n, 6)`` that the host
fills *before* the replay, and derives everything else on the device (:mod:`kernels_graph`: ``eq`` tables of the
claim point ``rho``, the running prefix ``eq(rho_{<i-1}, r_{<i-1})``, the four ``eq(rho_{i-1}, x)``, the pre-scaled
fold challenge).  With the :class:`logup_reference.Coins` verifier the coins are independent of the transcript, so
pre-filling them is exactly what the eager prover did (it drew them in the same order) and the transcript is identical.
Under Fiat-Shamir ``r_i`` is a hash of the transcript up to ``S_i``, which is produced inside the graph: the level must
then be split at every round (``rounds_per_graph = 1``: one graph per round = ``[fold by r_{i-1}] + round i``, one
D2H of ``S_i`` and one H2D of ``r_i`` between replays, ``n (n-1) / 2`` host round trips of ~20 us each), or the sponge
runs on the device (a hash kernel writing ``coins[i]`` between the round's kernels -- not built here; it would keep
the whole level in one graph).  Both segmentations are supported by ``rounds_per_graph`` and produce the same
transcript; the measurement reports both.

Memory: all level buffers are static (the tree levels, the table leaves, ``w_e / w_o``), the fold outputs live in one
graph memory pool shared by every graph (replayed in capture order, as PyTorch requires).  Shapes are fixed at
:meth:`GraphLogUp.build`; a prover object serves one ``(Nw, Nt)`` shape.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from . import kernels_graph as kg
from . import logup_reference as lref
from . import reference as ref
from . import sumcheck_packed as sp

P = ref.P
Ext = ref.Ext


def to_dev(v, device) -> torch.Tensor:
    return sp.to_dev(v, device)


@dataclass
class GraphTiming:
    tree: float = 0.0                   # ms, histogram + leaves + tree levels (device, one graph)
    levels: dict = field(default_factory=dict)   # lv -> ms (device, graph replay(s) of the level)
    device_total: float = 0.0           # ms, sum of the above (CUDA events around each replay)
    wall: float = 0.0                   # s, host wall time of prove() including the coin bookkeeping and the D2H
    graphs: int = 0                     # graphs replayed
    host_round_trips: int = 0           # D2H syncs (0 in level mode; n(n-1)/2 in per-round mode)


class GraphLogUp:
    """CUDA-graphed logUp prover for one shape.  ``rounds_per_graph``: ``None`` = whole level per graph (coins pre-filled;
    the transcript-independent-coins setting), ``1`` = one graph per round (the Fiat-Shamir-compatible segmentation)."""

    def __init__(self, device: str = "cuda", k: int = 3, k1_log2: int = sp.K1_LOG2_DEFAULT, rounds_per_graph: int | None = None):
        self.device = torch.device(device)
        self.k = k
        self.kernel = sp.Kernel(device, k1_log2=k1_log2, mode="triton")
        self.tk = self.kernel._tk
        self.pw16 = self.kernel.pw16
        self.one6 = self.kernel.one6
        self.rpg = rounds_per_graph
        self.built = False

    # -- static buffers ------------------------------------------------------------------------------------------------

    def build(self, Nw: int, Nt: int) -> None:
        """Allocate the static buffers for ``(Nw, Nt)``, warm the kernels once eagerly, capture the graphs."""
        dev = self.device
        n = lref.log2ceil(Nw + Nt)
        k_pack = min(self.k, n - 1)
        if Nw % (1 << (k_pack + 1)) or Nt % (1 << (k_pack + 1)):
            raise ValueError("Nw and Nt must be multiples of 2^(k+1)")
        self.Nw, self.Nt, self.n, self.k_pack = Nw, Nt, n, k_pack
        i32 = torch.int32
        self.w = torch.zeros(Nw, dtype=i32, device=dev)
        self.w_e = torch.zeros(Nw // 2, dtype=i32, device=dev)
        self.w_o = torch.zeros(Nw // 2, dtype=i32, device=dev)
        self.table = torch.zeros(Nt, dtype=i32, device=dev)
        self.mult = torch.zeros(Nt, dtype=i32, device=dev)
        self.T4_tab = torch.zeros((4, Nt // 2, 6), dtype=i32, device=dev)
        self.levels = {lv: torch.zeros((2, 1 << lv, 6), dtype=i32, device=dev) for lv in range(n)}
        self.zc = torch.zeros((3, 6), dtype=i32, device=dev)         # z, z R, z^2 (plain / pre-scaled / plain)
        self.rho = torch.zeros((n + 1, 6), dtype=i32, device=dev)
        self.coins = torch.zeros((n + 1, 6), dtype=i32, device=dev)
        self.coeff2 = torch.cat([self.one6, torch.zeros((1, 6), dtype=i32, device=dev)], 0)   # [1 | lambda]
        self.lam6 = self.coeff2[1:2]
        self.consts = torch.zeros((n + 1, 36), dtype=i32, device=dev)
        self.S = torch.zeros((n, n, 4, 6), dtype=i32, device=dev)   # S[lv, i - 1]
        self.vals = torch.zeros((n, 4, 6), dtype=i32, device=dev)
        half_total = 1 << (n - 2)
        x = torch.arange(Nw // 2, (Nw + Nt) // 2, device=dev)
        self.tab_dst = torch.where(x % 2 == 0, x // 2, half_total + x // 2)
        self.pool = torch.cuda.graph_pool_handle()
        # warm-up (compiles every kernel), then capture in replay order
        self._T4_cur = self._T4_tab_cur = None
        self.g_tree = self._capture(self._tree_ops)
        self.g_levels: dict[int, list] = {}
        for lv in range(n):
            self.g_levels[lv] = [(i0, i1, self._capture(lambda lv=lv, i0=i0, i1=i1, tail=tail: self._level_ops(lv, i0, i1, tail)))
                                 for (i0, i1, tail) in self._segments(lv)]
        torch.cuda.synchronize()
        self.built = True

    def _capture(self, fn):
        # warm run (compiles; segments mutate the level state, so it is restored before the capture run)
        state = (self._T4_cur, self._T4_tab_cur)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self._T4_cur, self._T4_tab_cur = state
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        torch.cuda.synchronize()
        return g

    def _segments(self, lv: int) -> list[tuple[int, int, bool]]:
        """Round ranges ``[i0, i1)`` (1-based rounds; a segment first runs the fold by ``r_{i0-1}`` when ``i0 > 1``; the
        segment ending at ``lv + 1`` also produces the level's final values)."""
        if lv == 0:
            return [(1, 1, True)]
        if self.rpg is None:
            return [(1, lv + 1, True)]
        # the tail (fold by r_lv + final values) is its own segment: r_lv is drawn after the last round's message
        return [(i, min(i + self.rpg, lv + 1), False) for i in range(1, lv + 1, self.rpg)] + [(lv + 1, lv + 1, True)]

    # -- the tree (one graph) --------------------------------------------------------------------------------------------

    def _tree_ops(self) -> None:
        n, Nt = self.n, self.Nt
        kg.histogram(self.w, Nt, self.mult)
        z6 = self.zc[0:1]
        pt0 = torch.where(self.mult == 0, torch.zeros_like(self.mult), P - self.mult)
        self.T4_tab[0, :, 0] = pt0[0::2]
        self.T4_tab[1, :, 0] = pt0[1::2]
        self.T4_tab[0:2, :, 1:] = 0
        q0 = sp.sub_mod(z6[0, 0].expand(Nt), self.table)
        self.T4_tab[2] = z6
        self.T4_tab[3] = z6
        self.T4_tab[2, :, 0] = q0[0::2]
        self.T4_tab[3, :, 0] = q0[1::2]
        pq = self.levels[n - 1]
        pq[0] = 0
        pq[1] = self.one6
        self.tk.leaf_level(self.w, self.zc[0].contiguous(), self.zc[1].contiguous(), self.zc[2].contiguous(), pq[0], pq[1], 1 << (n - 2))
        mul = self.tk.ext_mul_rows
        pq[0][self.tab_dst] = sp.add_mod(mul(self.T4_tab[0], self.T4_tab[3]), mul(self.T4_tab[1], self.T4_tab[2]))
        pq[1][self.tab_dst] = mul(self.T4_tab[2], self.T4_tab[3])
        for lv in range(n - 1, 0, -1):
            src, dst = self.levels[lv], self.levels[lv - 1]
            m = 1 << (lv - 1)
            self.tk._tree_level_kernel[(-(-m // 256),)](src[0], src[1], dst[0], dst[1], m, R2=self.tk.R2_MOD_P, BLOCK=256, FK=self.tk.FIELD_KEY)

    # -- one level (one graph, or one graph per segment) -----------------------------------------------------------------

    def _eq_inputs(self, i: int, nvars_total: int, K1: int):
        """``e_lo32 (K1, 32) int8`` = limbs of ``eq(rho[i:i+kappa]) R`` and ``e_hi (K2, 6)`` = ``eq(rho[i+kappa:nvars_total])``."""
        kappa = int(math.log2(K1))
        e_lo = kg.eq_table_vars(self.rho, i, kappa)
        e_lo32 = self.tk.eq_limbs32(e_lo)
        e_hi = kg.eq_table_vars(self.rho, i + kappa, nvars_total - i - kappa)
        return e_lo32, e_hi

    def _fold4(self, T4: torch.Tensor, r_R: torch.Tensor) -> torch.Tensor:
        m = T4.shape[1] // 2
        out = torch.empty((4, m, 6), dtype=torch.int32, device=self.device)
        return self.tk.fold_ext4(T4, m, r_R, out)

    def _ext_ip(self, T4, e_lo32, e_hi, K1, K2):
        split = sp.Kernel.split_b1(K1, K2)
        Zs = self.tk.logup_ext_inner_product(T4, e_lo32, K1, K2, split)
        Z = Zs.to(torch.int64).sum(0) if split > 1 else Zs[0]
        return self.tk.reduce_contract(Z, e_hi.contiguous(), self.pw16, K2, 6, 6).view(3, 2, 6)

    def _round_consts(self, i: int, c: torch.Tensor) -> None:
        kg.round_consts(self.rho, self.coins, i, c)

    def _generic_round(self, T4: torch.Tensor, lv: int, i: int) -> None:
        """Round ``i`` of level ``lv`` on ``T4 (4, 2^(lv-i+1), 6)``: ``S[lv, i-1]``."""
        c = self.consts[i]
        self._round_consts(i, c)
        prefix, eq_x = c[0:6], c[6:30].view(4, 6)
        m = 1 << (lv - i)
        K1, K2 = self.kernel.split(m)
        if K1 % 128 == 0:
            e_lo32, e_hi = self._eq_inputs(i, lv, K1)
            Q = self._ext_ip(T4, e_lo32, e_hi, K1, K2)
            self.tk.finish_round(Q, self.coeff2, prefix, eq_x, out=self.S[lv, i - 1])
        else:
            e = kg.eq_table_vars(self.rho, i, lv - i)
            lam_R = kg.scale_R(self.lam6)[0]
            Q = self.tk.logup_small_round(T4, e, lam_R)
            self.tk.finish_round(Q.view(3, 1, 6), self.one6, prefix, eq_x, out=self.S[lv, i - 1])

    def _packed_round(self, lv: int, i: int, T4_tab: torch.Tensor) -> torch.Tensor:
        """Big level, round ``i <= k``: the witness region packed on ``w_e, w_o``, the table region generic, the pad analytic."""
        Nw, Nt = self.Nw, self.Nt
        c = self.consts[i]
        self._round_consts(i, c)
        prefix, eq_x = c[0:6], c[6:30].view(4, 6)
        half = 1 << (i - 1)
        hh = half * half
        rows = 3 * (hh + half)
        Rw, Rt = Nw >> (i + 1), Nt >> (i + 1)
        Rp = (1 << (lv - i)) - Rw - Rt
        K1 = 1 << self.kernel.k1_log2
        for R_ in (Rw, Rt, Rp):
            if R_:
                K1 = min(K1, R_ & -R_)
        if K1 % 128:
            raise ValueError(f"round {i}: region ranges {(Rw, Rt, Rp)} are not aligned to 128 eq-chunks")
        e_lo32, e_hi = self._eq_inputs(i, lv, K1)
        Bw, Bt = Rw // K1, Rt // K1
        mul = self.tk.ext_mul_rows
        split = sp.Kernel.split_b1(K1, Bw)
        Zs = self.tk.logup_packed_inner_product(self.w_e, self.w_o, e_lo32, i, K1, Bw, split)
        Z = Zs.to(torch.int64).sum(0) if split > 1 else Zs[0]
        Tw = self.tk.reduce_contract(Z, e_hi[:Bw].contiguous(), self.pw16, Bw, rows, 1).view(3, hh + half, 6)
        eq_r = kg.eq_table_vars(self.coins, 0, i - 1)                                   # (half, 6) = eq(r_1..r_{i-1}, a)
        c_prod = mul(eq_r.repeat_interleave(half, 0), eq_r.repeat(half, 1))              # (hh, 6): index a . half + a'
        pi = (mul(Tw[:, :hh].reshape(3 * hh, 6), c_prod.repeat(3, 1)).view(3, hh, 6).to(torch.int64).sum(1) % P).to(torch.int32)
        s = (mul(Tw[:, hh:].reshape(3 * half, 6), eq_r.repeat(3, 1)).view(3, half, 6).to(torch.int64).sum(1) % P).to(torch.int32)
        Ew = (e_hi[:Bw].to(torch.int64).sum(0) % P).to(torch.int32)[None]               # sum_b e(b) over the witness (sum_b1 e_lo = 1)
        z6 = self.zc[0:1]
        z2 = self.zc[2:3]
        c1 = sp.add_mod(sp.add_mod(z6, z6), mul(self.lam6, z2))                          # 2 z + lambda z^2
        c2 = sp.add_mod(self.one6, mul(self.lam6, z6))                                   # 1 + lambda z
        Q = sp.add_mod(sp.sub_mod(mul(Ew, c1).expand(3, 6), mul(s, c2)), mul(pi, self.lam6))
        if Bt:
            Qt = self._ext_ip(T4_tab, e_lo32, e_hi[Bw:Bw + Bt], K1, Bt)
            Q = sp.add_mod(Q, sp.add_mod(Qt[:, 0], mul(Qt[:, 1], self.lam6)))
        if Rp:
            Ep = (e_hi[Bw + Bt:].to(torch.int64).sum(0) % P).to(torch.int32)[None]
            Q = sp.add_mod(Q, mul(Ep, self.lam6).expand(3, 6))
        self.tk.finish_round(Q.contiguous().view(3, 1, 6), self.one6, prefix, eq_x, out=self.S[lv, i - 1])
        return T4_tab

    def _assemble_big(self, lv: int) -> torch.Tensor:
        """After the ``k`` packed rounds: ``fold_k`` of the base queries and the four extension tables of the level."""
        k, Nw, Nt = self.k_pack, self.Nw, self.Nt
        dev = self.device
        M = 1 << (lv - k)
        Mw, Mt = Nw >> (k + 1), Nt >> (k + 1)
        T4 = torch.empty((4, M, 6), dtype=torch.int32, device=dev)
        if k:
            coeffs_R = kg.scale_R(kg.eq_table_vars(self.coins, 0, k))
            We = self.tk.fold_k(self.w_e, k, coeffs_R)
            Wo = self.tk.fold_k(self.w_o, k, coeffs_R)
        else:
            We = torch.zeros((Mw, 6), dtype=torch.int32, device=dev)
            We[:, 0] = self.w_e
            Wo = torch.zeros((Mw, 6), dtype=torch.int32, device=dev)
            Wo[:, 0] = self.w_o
        z6 = self.zc[0:1]
        T4[0, :Mw] = self.one6
        T4[1, :Mw] = self.one6
        T4[2, :Mw] = sp.sub_mod(z6, We)
        T4[3, :Mw] = sp.sub_mod(z6, Wo)
        T4[:, Mw:Mw + Mt] = self._T4_tab_cur
        T4[0:2, Mw + Mt:] = 0
        T4[2:4, Mw + Mt:] = self.one6
        return T4

    def _fold_step(self, lv: int, i: int) -> None:
        """The fold by ``r_i`` after round ``i`` (``r_i R`` recomputed from ``coins`` so a per-round segmentation, which
        writes ``coins[i-1]`` after round ``i``'s graph ran, still folds by the right challenge)."""
        r_R = kg.scale_R(self.coins[i - 1:i])[0]
        if self._big and i <= self.k_pack:
            if self.Nt:
                self._T4_tab_cur = self._fold4(self._T4_tab_cur, r_R)
            if i == self.k_pack:
                self._T4_cur = self._assemble_big(lv)
        else:
            self._T4_cur = self._fold4(self._T4_cur, r_R)

    def _level_ops(self, lv: int, i0: int, i1: int, tail: bool) -> None:
        """Rounds ``i0..i1-1`` of level ``lv`` (with the fold by ``r_{i0-1}`` first when ``i0 > 1``, and the tail -- last
        fold + final values -- when ``tail``).  The level's working tables are kept on ``self`` between segments
        (graph-pool memory, stable across replays)."""
        self._big = lv == self.n - 1
        if i0 == 1:
            self._T4_tab_cur = self.T4_tab
            if self._big:
                self._T4_cur = self._assemble_big(lv) if self.k_pack == 0 else None
            else:
                self._T4_cur = self.levels[lv + 1].view(4, 1 << lv, 6)
        else:
            self._fold_step(lv, i0 - 1)
        for i in range(i0, i1):
            if self._big and i <= self.k_pack:
                self._packed_round(lv, i, self._T4_tab_cur)
            else:
                self._generic_round(self._T4_cur, lv, i)
            if i + 1 < i1:
                self._fold_step(lv, i)
        if tail:
            if lv and i0 <= lv:
                self._fold_step(lv, lv)
            self.vals[lv] = self._T4_cur[:, 0]

    # -- the proof ----------------------------------------------------------------------------------------------------

    @staticmethod
    def _h2d(buf: torch.Tensor, vals: list[Ext]) -> None:
        """Stream-ordered host write: pinned staging + ``non_blocking`` so the host never waits on the stream (a pageable
        ``copy_`` would synchronise the stream first and serialise instances that run on parallel streams)."""
        if vals:
            buf[:len(vals)].copy_(to_dev(vals, "cpu").pin_memory(), non_blocking=True)

    def launch(self, w: torch.Tensor, table: torch.Tensor, z: Ext, coins: lref.Coins) -> dict:
        """Enqueue the whole proof on the current stream and return without synchronising (``finish`` completes it).
        Several instances launched on different streams overlap (the small levels are latency-bound chains)."""
        if not self.built or w.numel() != self.Nw or table.numel() != self.Nt:
            self.build(w.numel(), table.numel())
        n = self.n
        tm = GraphTiming()
        t_wall = time.perf_counter()
        def ev():
            return torch.cuda.Event(enable_timing=True)

        self.w.copy_(w)
        self.w_e.copy_(w.view(-1, 2)[:, 0])
        self.w_o.copy_(w.view(-1, 2)[:, 1])
        self.table.copy_(table)
        self._h2d(self.zc, [z, tuple(x * sp.R_MOD_P % P for x in z), ref.mul(z, z)])
        e0, e1 = ev(), ev()
        e0.record()
        self.g_tree.replay()
        e1.record()
        lam = coins.next()
        rho: list[Ext] = []
        per_level = []
        mu = None
        for lv in range(n):
            self._h2d(self.rho, rho)
            self._h2d(self.lam6, [lam])
            rs: list[Ext] = []
            a, b = ev(), ev()
            a.record()
            if self.rpg is None:
                rs = [coins.next() for _ in range(lv)]
                self._h2d(self.coins, rs)
                for (_i0, _i1, g) in self.g_levels[lv]:
                    g.replay()
                    tm.graphs += 1
            else:
                for (i0, i1, g) in self.g_levels[lv]:
                    g.replay()
                    tm.graphs += 1
                    # Fiat-Shamir segmentation: the challenges of rounds i0..i1-1 are drawn after their messages
                    if i1 > i0:
                        _S = self.S[lv, i0 - 1:i1 - 1].cpu()      # D2H sync (the sponge would absorb these)
                        tm.host_round_trips += 1
                        new = [coins.next() for _ in range(i0, i1)]
                        self._h2d(self.coins[i0 - 1:i1 - 1], new)
                        rs += new
            b.record()
            per_level.append((lv, a, b))
            mu = coins.next()
            lam = coins.next()
            rho = [mu] + rs
        return {"tm": tm, "t_wall": t_wall, "e0": e0, "e1": e1, "per_level": per_level, "mu": mu, "rho": rho, "lam": lam}

    def prove(self, w: torch.Tensor, table: torch.Tensor, z: Ext, coins: lref.Coins
              ) -> tuple[lref.Transcript, lref.LeafClaim, GraphTiming]:
        """Same messages as ``logup_reference.prove_table(*leaves(w, table, mult, z), coins)``.  ``w (Nw,)``, ``table (Nt,)``
        int32 on the device (copied into the static buffers)."""
        st = self.launch(w, table, z, coins)
        torch.cuda.synchronize()
        return self.finish(st)

    def finish(self, st: dict) -> tuple[lref.Transcript, lref.LeafClaim, GraphTiming]:
        """Timing + transcript of a finished :meth:`launch` (the device must have completed it: synchronise first)."""
        n = self.n
        tm, t_wall, e0, e1, per_level, mu, rho, lam = (st[k] for k in ("tm", "t_wall", "e0", "e1", "per_level", "mu", "rho", "lam"))
        tm.tree = e0.elapsed_time(e1)
        for lv, a, b in per_level:
            tm.levels[lv] = a.elapsed_time(b)
        tm.device_total = tm.tree + sum(tm.levels.values())
        S = self.S.cpu().tolist()
        vals = self.vals.cpu().tolist()
        q_root = tuple(int(v) for v in self.levels[0][1][0].tolist())
        tr = lref.Transcript(q_root)
        for lv in range(n):
            tr.levels.append({"rounds": [[tuple(int(x) for x in row) for row in S[lv][i]] for i in range(lv)],
                              "vals": [tuple(int(x) for x in row) for row in vals[lv]]})
        tm.wall = time.perf_counter() - t_wall
        v = tr.levels[-1]["vals"] if n else None
        if n:
            p_next = ref.add(v[0], ref.mul(mu, ref.sub(v[1], v[0])))
            q_next = ref.add(v[2], ref.mul(mu, ref.sub(v[3], v[2])))
        else:
            p_next, q_next = ref.ZERO, q_root
        return tr, lref.LeafClaim(rho, lam, ref.add(p_next, ref.mul(lam, q_next))), tm
