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
try:
    from packed import kernels_graph as _kg
    from packed import logup_graph as _lgr
    GRAPHS = AVAILABLE
except ImportError:  # pragma: no cover
    try:
        from backends.gkr.packed import kernels_graph as _kg
        from backends.gkr.packed import logup_graph as _lgr
        GRAPHS = AVAILABLE
    except ImportError:
        GRAPHS = False

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


# ---------------------------------------------------------------------------------------------------------------------
# CUDA-graphed levels (a-fusion's GraphLogUp, one graph per round = the Fiat-Shamir segmentation)
# ---------------------------------------------------------------------------------------------------------------------
#
# Why the transcript is unchanged: a graph replays the same kernels the eager prover launched, on static buffers; the
# round constants (eq tables of rho, the prefix, eq(rho_{i-1}, x), r R) are recomputed ON DEVICE from the coin buffer
# by kernels_graph, so a replay with the Fiat-Shamir challenges written into ``coins`` evaluates exactly the eager
# arithmetic.  The host loop below does what ``_walk`` did -- absorb S_i, squeeze r_i, write it -- with one D2H per
# round (the sponge is on the host) and one per level (the four final values).  Launches per round: 1 (the replay).

if GRAPHS:

    class _FSGraph(_lgr.GraphLogUp):
        """Range table (R16 layout: ``[queries | table | pad]``) with the a-gpu transcript driving the coins."""

        def __init__(self, device, k: int) -> None:
            super().__init__(str(device), k=k, rounds_per_graph=1)

        def prove_fs(self, w: torch.Tensor, table: torch.Tensor, z: Ext, tr: Transcript, out: list[Ext]) -> LeafClaim:
            if not self.built or w.numel() != self.Nw or table.numel() != self.Nt:
                self.build(w.numel(), table.numel())
            self.w.copy_(w)
            self.w_e.copy_(w.view(-1, 2)[:, 0])
            self.w_o.copy_(w.view(-1, 2)[:, 1])
            self.table.copy_(table)
            self._h2d(self.zc, [z, tuple(x * _sp.R_MOD_P % P for x in z), _lg.ref.mul(z, z)])
            self.g_tree.replay()
            return self._walk_fs(tr, out)

        def _walk_fs(self, tr: Transcript, out: list[Ext]) -> LeafClaim:
            n = self.n
            root = self.levels[0][:, 0].to(torch.int64).cpu()                 # sync: the tree is done
            if bool((root[0] % P).any()):
                raise ValueError("LogUp: fractional sum is not zero (a query is not in its table)")
            root_q = py_ext(root[1])
            if not any(root_q):
                raise ValueError("LogUp: q_root is zero")
            out.append(root_q)
            tr.absorb_ext(root_q)
            lam = tr.challenge()
            rho: list[Ext] = []
            syncs = 1
            for lv in range(n):
                self._h2d(self.rho, rho)
                self._h2d(self.lam6, [lam])
                rs: list[Ext] = []
                for (i0, i1, g) in self.g_levels[lv]:
                    g.replay()
                    if i1 > i0:                                             # one round: absorb S_i, draw r_i, write it
                        hs = py_exts(self.S[lv, i0 - 1:i1 - 1].to(torch.int64).cpu())
                        syncs += 1
                        out.extend(hs)
                        tr.absorb_exts(hs)
                        new = [tr.challenge() for _ in range(i0, i1)]
                        self._h2d(self.coins[i0 - 1:i1 - 1], new)
                        rs += new
                vals = py_exts(self.vals[lv].to(torch.int64).cpu())
                syncs += 1
                out.extend(vals)
                tr.absorb_exts(vals)
                mu = tr.challenge()
                lam = tr.challenge()
                rho = [mu] + rs
            p_next = logup.e_add(vals[0], logup.e_mul(mu, logup.e_sub(vals[1], vals[0])))
            q_next = logup.e_add(vals[2], logup.e_mul(mu, logup.e_sub(vals[3], vals[2])))
            tr.logup_syncs = getattr(tr, "logup_syncs", 0) + syncs
            return logup.leaf_claim_top(rho, lam, logup.e_add(p_next, logup.e_mul(lam, q_next)))

    class _FSGraphExt(_FSGraph):
        """Vector-valued table: leaves ``p (2^n,)`` base / ``q (2^n, 6)`` in static buffers, level ``n-1`` by the a-gpu
        ``_combine_level`` kernel, every level generic (no packed base rounds)."""

        def __init__(self, device) -> None:
            super().__init__(device, k=0)

        def build_ext(self, n: int) -> None:
            dev = self.device
            self.n, self.k_pack, self.Nw, self.Nt = n, 0, 0, 0
            i32 = torch.int32
            self.p_leaf = torch.zeros(1 << n, dtype=i32, device=dev)
            self.q_leaf = torch.zeros((1 << n, 6), dtype=i32, device=dev)
            self.T4_big = torch.zeros((4, 1 << (n - 1), 6), dtype=i32, device=dev)
            self.levels = {lv: torch.zeros((2, 1 << lv, 6), dtype=i32, device=dev) for lv in range(n)}
            self.rho = torch.zeros((n + 1, 6), dtype=i32, device=dev)
            self.coins = torch.zeros((n + 1, 6), dtype=i32, device=dev)
            self.coeff2 = torch.cat([self.one6, torch.zeros((1, 6), dtype=i32, device=dev)], 0)   # [1 | lambda]
            self.lam6 = self.coeff2[1:2]
            self.consts = torch.zeros((n + 1, 36), dtype=i32, device=dev)
            self.S = torch.zeros((n, n, 4, 6), dtype=i32, device=dev)
            self.vals = torch.zeros((n, 4, 6), dtype=i32, device=dev)
            self.pool = torch.cuda.graph_pool_handle()
            self._T4_cur = self._T4_tab_cur = None
            self.g_tree = self._capture(self._tree_ops)
            self.g_levels = {}
            for lv in range(n):
                self.g_levels[lv] = [(i0, i1, self._capture(lambda lv=lv, i0=i0, i1=i1, tail=tail: self._level_ops(lv, i0, i1, tail)))
                                     for (i0, i1, tail) in self._segments(lv)]
            torch.cuda.synchronize()
            self.built = True

        def _tree_ops(self) -> None:
            n = self.n
            np_, nq = logup._combine_level(self.p_leaf, self.q_leaf)
            pq = self.levels[n - 1]
            pq[0] = _deinterleave(np_)
            pq[1] = _deinterleave(nq)
            for lv in range(n - 1, 0, -1):
                src, dst = self.levels[lv], self.levels[lv - 1]
                m = 1 << (lv - 1)
                self.tk._tree_level_kernel[(-(-m // 256),)](src[0], src[1], dst[0], dst[1], m, R2=self.tk.R2_MOD_P, BLOCK=256, FK=self.tk.FIELD_KEY)
            self.T4_big[0, :, 0] = self.p_leaf[0::2]
            self.T4_big[1, :, 0] = self.p_leaf[1::2]
            self.T4_big[2] = self.q_leaf[0::2]
            self.T4_big[3] = self.q_leaf[1::2]

        def _level_ops(self, lv: int, i0: int, i1: int, tail: bool) -> None:
            self._big = False
            if i0 == 1:
                self._T4_cur = self.T4_big if lv == self.n - 1 else self.levels[lv + 1].view(4, 1 << lv, 6)
            else:
                self._fold_step(lv, i0 - 1)
            for i in range(i0, i1):
                self._generic_round(self._T4_cur, lv, i)
                if i + 1 < i1:
                    self._fold_step(lv, i)
            if tail:
                if lv and i0 <= lv:
                    self._fold_step(lv, lv)
                self.vals[lv] = self._T4_cur[:, 0]

        def prove_fs_ext(self, p_leaf: torch.Tensor, q_leaf: torch.Tensor, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
            if not self.built or self.n != n:
                self.build_ext(n)
            self.p_leaf.copy_(p_leaf)
            self.q_leaf.copy_(q_leaf)
            self.g_tree.replay()
            return self._walk_fs(tr, out)

    class _DeviceFS:
        """Mixin: the sponge runs on the device inside the graphs (``fs_cuda``), one graph per level.  After every round
        the graph absorbs ``S[lv, i-1]`` and writes ``r_i`` to ``coins[i-1]``; after the tree it absorbs ``q_root`` and
        writes ``lam``; at a level's tail it absorbs the four values and writes ``mu`` / ``lam`` and the next level's
        ``rho = [mu] + r``.  The messages and the challenge order are ``_walk_fs``'s, so the transcript is unchanged;
        the host reads everything back once per table (``_finish_fs``)."""

        def _fs_setup(self) -> None:
            from .fs_cuda import DeviceTranscript
            self.fs = DeviceTranscript(self.device)
            # prod_{j < i-1} eq(rho_j, r_{j+1}) for the current round, kept by the sponge steps (round_consts_pfx)
            self.pfx = torch.zeros(6, dtype=torch.int32, device=self.fs.device)

        def _round_consts(self, i: int, c: torch.Tensor) -> None:
            _kg.round_consts_pfx(self.rho, self.pfx, self.coins, i, c)

        def _generic_round(self, T4, lv: int, i: int) -> None:
            super()._generic_round(T4, lv, i)
            self.fs.step(self.S[lv, i - 1], self.coins[i - 1], rho=self.rho[i - 1], pfx=self.pfx)

        def _packed_round(self, lv: int, i: int, T4_tab):
            out = super()._packed_round(lv, i, T4_tab)
            self.fs.step(self.S[lv, i - 1], self.coins[i - 1], rho=self.rho[i - 1], pfx=self.pfx)
            return out

        def _level_ops(self, lv: int, i0: int, i1: int, tail: bool) -> None:
            super()._level_ops(lv, i0, i1, tail)
            if tail:
                self.fs.step(self.vals[lv], self.rho[0], self.lam6[0], pfx=self.pfx, reset=True)
                if lv:
                    self.rho[1:lv + 1] = self.coins[0:lv]

        def _tree_ops(self) -> None:
            super()._tree_ops()
            self.fs.step(self.levels[0][1, 0], self.lam6[0], pfx=self.pfx, reset=True)

        def _replay_fs(self, tr: Transcript, out: list[Ext]) -> LeafClaim:
            self.fs.load(tr)
            self.g_tree.replay()
            for lv in range(self.n):
                for (_, _, g) in self.g_levels[lv]:
                    g.replay()
            return self._finish_fs(tr, out)

        def _finish_fs(self, tr: Transcript, out: list[Ext]) -> LeafClaim:
            n = self.n
            parts = [self.levels[0][:, 0], self.S, self.vals, self.rho[:n], self.lam6, self.fs.buf]
            h = torch.cat([t.reshape(-1) for t in parts]).cpu().numpy().astype("int64")   # the one sync of the table
            sizes = [t.numel() for t in parts]
            root, S, vals, rho, lam6, fsw = (h[a:a + s] for a, s in zip([sum(sizes[:j]) for j in range(len(sizes))], sizes))
            root = root.reshape(2, DEG)
            S = S.reshape(n, n, 4, DEG)
            vals = vals.reshape(n, 4, DEG)
            if bool((root[0] % P).any()):
                raise ValueError("LogUp: fractional sum is not zero (a query is not in its table)")
            root_q = tuple(int(v) for v in root[1])
            if not any(root_q):
                raise ValueError("LogUp: q_root is zero")
            out.append(root_q)
            S_l, vals_l = S.tolist(), vals.tolist()
            for lv in range(n):
                for i in range(lv):
                    out.extend(map(tuple, S_l[lv][i]))
                out.extend(map(tuple, vals_l[lv]))
            self.fs.store(tr, fsw.astype("int32"), 1 + sum(lv + 2 for lv in range(n)))
            tr.logup_syncs = getattr(tr, "logup_syncs", 0) + 1
            rho = [tuple(int(v) for v in r) for r in rho.reshape(n, DEG)]
            lam = tuple(int(v) for v in lam6)
            mu = rho[0]
            v = [tuple(int(x) for x in row) for row in vals[n - 1]]
            p_next = logup.e_add(v[0], logup.e_mul(mu, logup.e_sub(v[1], v[0])))
            q_next = logup.e_add(v[2], logup.e_mul(mu, logup.e_sub(v[3], v[2])))
            return logup.leaf_claim_top(rho, lam, logup.e_add(p_next, logup.e_mul(lam, q_next)))

    class _FSGraphDev(_DeviceFS, _FSGraph):
        def __init__(self, device, k: int) -> None:
            _lgr.GraphLogUp.__init__(self, str(device), k=k, rounds_per_graph=None)
            self._fs_setup()

        def prove_fs(self, w: torch.Tensor, table: torch.Tensor, z: Ext, tr: Transcript, out: list[Ext]) -> LeafClaim:
            if not self.built or w.numel() != self.Nw or table.numel() != self.Nt:
                self.build(w.numel(), table.numel())
            self.w.copy_(w)
            self.w_e.copy_(w.view(-1, 2)[:, 0])
            self.w_o.copy_(w.view(-1, 2)[:, 1])
            self.table.copy_(table)
            self._h2d(self.zc, [z, tuple(x * _sp.R_MOD_P % P for x in z), _lg.ref.mul(z, z)])
            return self._replay_fs(tr, out)

    class _FSGraphExtDev(_DeviceFS, _FSGraphExt):
        def __init__(self, device) -> None:
            _lgr.GraphLogUp.__init__(self, str(device), k=0, rounds_per_graph=None)
            self._fs_setup()

        def prove_fs_ext(self, p_leaf: torch.Tensor, q_leaf: torch.Tensor, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
            if not self.built or self.n != n:
                self.build_ext(n)
            self.p_leaf.copy_(p_leaf)
            self.q_leaf.copy_(q_leaf)
            return self._replay_fs(tr, out)

    _GRAPHS: dict = {}

    def device_fs() -> bool:
        import os
        from . import fs_cuda
        return fs_cuda.available() and not os.environ.get("VERITY_GPU_HOST_FS")

    def graph_range(device, k: int, Nw: int, Nt: int) -> "_FSGraph":
        dev_fs = device_fs()
        key = ("range", k, Nw, Nt, dev_fs)
        if key not in _GRAPHS:
            _GRAPHS[key] = _FSGraphDev(device, k) if dev_fs else _FSGraph(device, k)
        return _GRAPHS[key]

    def graph_ext(device, n: int) -> "_FSGraphExt":
        dev_fs = device_fs()
        key = ("ext", n, dev_fs)
        if key not in _GRAPHS:
            _GRAPHS[key] = _FSGraphExtDev(device) if dev_fs else _FSGraphExt(device)
        return _GRAPHS[key]


def use_graphs(device) -> bool:
    import os
    return GRAPHS and device.type == "cuda" and not os.environ.get("VERITY_GPU_NO_GRAPHS")


def prove_range_table_graphed(w: torch.Tensor, table: torch.Tensor, z: Ext, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
    """:func:`prove_range_table` on CUDA graphs (same messages); falls back to the eager path when the region alignment
    does not allow the packed rounds the graph was built for."""
    Nw, Nt = w.numel(), table.numel()
    if n != _lg.lref.log2ceil(Nw + Nt):
        raise ValueError("hypercube dimension mismatch")
    k = _pick_k(Nw, Nt, n, 3)
    return graph_range(w.device, k, Nw, Nt).prove_fs(w, table, z, tr, out)


def prove_ext_table_graphed(p_leaf: torch.Tensor, q_leaf: torch.Tensor, n: int, tr: Transcript, out: list[Ext]) -> LeafClaim:
    """:func:`prove_ext_table` on CUDA graphs (same messages)."""
    return graph_ext(q_leaf.device, n).prove_fs_ext(p_leaf.to(torch.int32), q_leaf.to(torch.int32), n, tr, out)


__all__ = ["AVAILABLE", "GRAPHS", "prove_range_table", "prove_ext_table", "prove_range_table_graphed",
           "prove_ext_table_graphed", "use_graphs", "FSCoins", "DEG"]
