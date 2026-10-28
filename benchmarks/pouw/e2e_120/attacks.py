"""E2E-120's attack library: each attack is a prover that departs from `window.Honest` at one step, played against today's
round. `ATTACKS` names each one, how to play it, and whether today's check (`PouwSegCheckDither_v2{S, duo}`, the e2e slice's)
catches it. benchmarks/pouw/tests/test_e2e_120.py asserts each is rejected, and marks xfail (with a reason) the ones today's
check doesn't catch.

    skip-fraction      skips a fraction of the GEMM's 32-steps (a tile's step j left out, T_j = T_(j-1)) and folds its
                       receipts over the words it did compute, so they are self-consistent: caught when the draw lands on a
                       skipped step (`play_until`, which draws until it does)
    forge-committed    commits receipt words that are no fold of any products (every word random)
    forge-opened       commits the honest receipts, then opens the drawn leaf with the thread's word changed by a bit
    mitm               computes and commits random receipts; at the draw it builds the window's S + 1 witness sets with the
                       drawn step right and solves two chains' words of a free set by meet-in-the-middle so the fold ends at
                       its committed word (art:22e9a0398b2d's open_any.py, generalised to any drawn step). Today's check
                       accepts it; PIN's pinned spot-check Definition (v3) is to bind every absorbed word
    swap-served        serves other weights (another model's W) and opens the registered weights' leaves
    swap-opened        serves other weights and opens their own tree's leaves (a root the registered one isn't)
    skip-filler        the served design is H1 + fx1 (decision 83, one keyed IMAD a word before duo); this prover leaves the
                       filler out and folds duo alone. Today's check folds duo, so duo receipts pass it; PIN's v3, folding
                       fx1, is to reject them
    replay             replays its previous round's receipts and witness (re-committed under this round's tree) in a round
                       that serves the same values under a fresh coin
    commit-after-coin  commits the activations, sees the coin, then commits a second activation tree
    commit-after-draw  commits the receipts, sees the draw, then commits a second receipt tree
"""
from __future__ import annotations

import numpy as np

import window as WD
from window import SG, RoundCoins, TilePick, Honest, Round, Refused, Shape, Tree, ServedValues, label, tile_rows

M32 = WD.M32
KA, KK, ROT = SG.KA, SG.KK, SG.ROT


def _rotl(x: np.ndarray, r: int) -> np.ndarray:
    return ((x << np.uint64(r)) | (x >> np.uint64(32 - r))) & np.uint64(M32)


# --- the GEMM / receipt attacks ----------------------------------------------------------------------------------------

class SkipFraction(Honest):
    """Skips a fraction of the tiles' 32-steps and folds over the rest."""

    def __init__(self, *a, frac: float = 0.3, mask_seed: int = 7, **k):
        super().__init__(*a, **k)
        rng = np.random.default_rng(mask_seed)
        self.skip = rng.random((self.shape.tiles, self.shape.ks)) < frac

    def products(self, coins: RoundCoins) -> np.ndarray:
        Wp, Xp = WD.noised(ServedValues(self.served_weights(), self.window.X), coins, self.shape)
        return WD.prefix_words(Wp, Xp, self.shape, self.skip)

    def skipped(self, d: TilePick) -> bool:
        return bool(self.skip[d.tile, d.seg])


class ForgeCommitted(Honest):
    """Commits random receipt words, no fold of anything (and opens an honest witness for them)."""

    def receipts(self, P, coins):
        rng = np.random.default_rng(int.from_bytes(coins.sigma[:8], "big"))
        return rng.integers(0, 1 << 32, (self.shape.tiles, self.shape.nw, WD.NCT), dtype=np.uint32)


class ForgeOpened(Honest):
    """Commits the honest receipts, then opens the drawn leaf with the thread's word changed."""

    def open(self, draw, coins, P, R, xt, rt):
        op = super().open(draw, coins, P, R, xt, rt)
        rc = np.ascontiguousarray(op.rows["rc"], "<u2").view("<u4").copy()
        rc[draw.thread] ^= 1
        op.rows["rc"] = rc.view("<u2")
        return op


def _chain_final(ci: int, y0: int, z0: int, sets: list[list[int]], start: int, var_set: int, var_entry: int,
                 vals: np.ndarray) -> np.ndarray:
    """Duo chain ``ci``'s contribution y ^ rotl(z, 16), run over candidate words ``vals`` for entry ``var_entry`` of set
    ``var_set``, from state (y0, z0) before set ``start`` (chains are independent, so one chain's final is a function of its
    own words alone)."""
    n = len(vals)
    y = np.full(n, y0 & M32, np.uint64)
    z = np.full(n, z0 & M32, np.uint64)
    e = ci % 4
    ts = range(0, 8, 2) if ci < 4 else range(1, 8, 2)
    for s in range(start, len(sets)):
        for t in ts:
            ent = 4 * t + e
            p = vals if (s == var_set and ent == var_entry) else np.uint64(sets[s][ent] & M32)
            u = (y ^ p ^ np.uint64(KK[t])) & np.uint64(M32)
            y = (u * np.uint64(KA[t]) + z) & np.uint64(M32)
            z = (_rotl(u, ROT[t]) + y) & np.uint64(M32)
    return y ^ _rotl(z, 16)


class Mitm(Honest):
    """The meet-in-the-middle forgery (art:22e9a0398b2d): commit random receipts, then at the draw build a witness that opens
    the thread's committed word for any drawn step. It computes no products; it serves honest operands so their leaves open."""

    def __init__(self, *a, candidates: int = 1 << 17, mitm_seed: int = 1, **k):
        super().__init__(*a, **k)
        self.candidates, self.rng = candidates, np.random.default_rng(mitm_seed)
        self.matches = 0

    def products(self, coins: RoundCoins) -> np.ndarray:
        # it computes nothing; the honest products only feed the operand leaves, opened straight from the window
        return np.zeros((self.shape.tiles, self.shape.ks + 1, WD.NCT, 32), np.uint32)

    def receipts(self, P, coins):
        return self.rng.integers(0, 1 << 32, (self.shape.tiles, self.shape.nw, WD.NCT), dtype=np.uint32)

    def witness(self, P, R, coins, draw, rows):
        sh, S = self.shape, self.shape.S
        pl = draw.place(sh)
        i, first = pl["i"], pl["w"] == 0
        target = int(R[draw.tile, pl["w"], draw.thread])
        Wp, Xp = WD.noised(ServedValues(self.served_weights(), self.window.X), coins, sh)
        j = draw.seg
        ap = (Wp[pl["a_rows"], 32 * j:32 * j + 32] & 0xFF).tolist()
        bp = (Xp[pl["b_rows"], 32 * j:32 * j + 32] & 0xFF).tolist()
        p = [[int(v) for v in self.rng.integers(0, 1 << 32, 32)] for _ in range(S + 1)]
        if first:
            p[0] = [0] * 32
        p[i + 1] = [SG.seg_dot(p[i][q], ap[SG.row_slot(q)], bp[SG.col_slot(q)]) for q in range(32)]
        ksol = S if i < S - 1 else S - 2         # the last set free of the drawn step (chains 0, 4; entries 24, 28)
        st = SG.fold_keys("duo", coins.seed, draw.tile, draw.thread, pl["w"], sh.nw)
        state = list(st)
        for s in range(ksol):
            state = SG._duo_absorb(state, p[s])
        rest = list(state)
        for s in range(ksol, S + 1):
            rest = SG._duo_absorb(rest, p[s])
        other = 0
        for c in range(SG.NCH):
            if c not in (0, 4):
                other ^= rest[c] ^ SG.rotl(rest[SG.NCH + c], 16)
        w0 = self.rng.integers(0, 1 << 32, self.candidates, dtype=np.uint64)
        w4 = self.rng.integers(0, 1 << 32, self.candidates, dtype=np.uint64)
        g0 = _chain_final(0, state[0], state[SG.NCH + 0], p, ksol, ksol, 24, w0)
        g4 = _chain_final(4, state[4], state[SG.NCH + 4], p, ksol, ksol, 28, w4) ^ np.uint64(other ^ target)
        common, i0, i4 = np.intersect1d(g0, g4, return_indices=True)
        self.matches = len(common)
        if len(common) == 0:
            raise RuntimeError("mitm: no collision; raise `candidates`")
        p[ksol][24], p[ksol][28] = int(w0[i0[0]]), int(w4[i4[0]])
        return p


class SkipFiller(Honest):
    """The served design is H1 + fx1; this leaves the filler out and folds duo (what today's duo check verifies)."""

    def __init__(self, *a, **k):
        super().__init__(*a, receipt_fold="duo", **k)


# --- the swap attacks --------------------------------------------------------------------------------------------------

class SwapServed(Honest):
    """Serves other weights, opens the registered weights' leaves."""

    def __init__(self, *a, other: np.ndarray, **k):
        super().__init__(*a, **k)
        self.other = other

    def served_weights(self):
        return self.other


class SwapOpened(Honest):
    """Serves other weights and opens their own tree's leaves (whose root is not the registered one)."""

    def __init__(self, *a, other: np.ndarray, **k):
        super().__init__(*a, **k)
        self.other = other
        self.open_w_tree = Tree(label(self.session, 0, "W"), tile_rows(other), self.salt_key)

    def served_weights(self):
        return self.other


# --- the ordering attacks ----------------------------------------------------------------------------------------------

class CommitAfterCoin(Honest):
    """Commits the activations, then (having seen the coin) commits a second activation tree."""

    def run(self, rd: Round):
        coins = RoundCoins(rd.operands(self.x_tree(rd.rnd, self.window.X).root))
        X2 = self.window.X ^ np.int8(int.from_bytes(coins.sigma[:1], "big") & 1)
        rd.operands(self.x_tree(rd.rnd, X2).root)             # Refused: once a round, before its coin
        raise AssertionError("commit-after-coin: the round let a second activation commitment through")


class CommitAfterDraw(Honest):
    """Commits the receipts, then (having seen the draw) commits a second receipt tree."""

    def run(self, rd: Round):
        xt = self.x_tree(rd.rnd, self.window.X)
        coins = RoundCoins(rd.operands(xt.root))
        P = self.products(coins)
        R = self.receipts(P, coins)
        rt = Tree(label(self.session, rd.rnd, "R"), WD.receipt_rows(R), self.salt_key)
        rd.receipts(rt.root)
        rd.receipts(rt.root)                                  # Refused: once a round, after its coin
        raise AssertionError("commit-after-draw: the round let a second receipt commitment through")


# --- the attack registry -----------------------------------------------------------------------------------------------

def honest(shape: Shape | None = None, seed: int = 0, fold: str = "duo") -> Honest:
    shape = shape or Shape()
    return Honest(ServedValues.stand_in(shape, seed), shape, receipt_fold=fold)


def _play_until(prover: SkipFraction, shape: Shape, base_seed: int = 0, tries: int = 200) -> WD.RoundResult:
    """Play ``prover`` until the draw lands on a step it skipped (the only draws its forgery is exposed on)."""
    for s in range(base_seed, base_seed + tries):
        rd = Round(shape, prover.w_tree.root, prover.session, 1, __import__("random").Random(s), "duo")
        op = prover.run(rd)
        if prover.skipped(rd.draw_):
            return WD.RoundResult(rd.judge(op) is None, rd.judge(op), rd.draw_, rd.sigma, op, rd.log)
    raise RuntimeError("skip-fraction: no draw landed on a skipped step")


def run_attack(name: str, shape: Shape | None = None, seed: int = 1) -> WD.RoundResult:
    """Play attack ``name`` once against today's check and return its verdict."""
    shape = shape or Shape()
    win = ServedValues.stand_in(shape)
    other = np.roll(win.W, 1, axis=0)
    makers = {
        "skip-fraction": lambda: _play_until(SkipFraction(win, shape), shape, seed),
        "forge-committed": lambda: WD.play(ForgeCommitted(win, shape), shape, seed=seed),
        "forge-opened": lambda: WD.play(ForgeOpened(win, shape), shape, seed=seed),
        "mitm": lambda: WD.play(Mitm(win, shape, mitm_seed=seed), shape, seed=seed),
        "swap-served": lambda: WD.play(SwapServed(win, shape, other=other), shape, seed=seed),
        "swap-opened": lambda: WD.play(SwapOpened(win, shape, other=other), shape, seed=seed),
        "skip-filler": lambda: WD.play(SkipFiller(win, shape), shape, seed=seed),
        "replay": lambda: play_replay(win, shape, seed),
        "commit-after-coin": lambda: WD.play(CommitAfterCoin(win, shape), shape, seed=seed),
        "commit-after-draw": lambda: WD.play(CommitAfterDraw(win, shape), shape, seed=seed),
    }
    return makers[name]()


class Replay(Honest):
    """Replays a captured round's products and receipts in a later round."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.prev: dict | None = None

    def products(self, coins):
        if self.prev is None:
            self._P = super().products(coins)
            return self._P
        return self.prev["P"]

    def receipts(self, P, coins):
        if self.prev is None:
            self._R = super().receipts(P, coins)
            return self._R
        return self.prev["R"]


def play_replay(win: ServedValues, shape: Shape, seed: int) -> WD.RoundResult:
    """Round 1 honest (captured), then round 2 replays round 1's receipts and witness under round 2's fresh coin."""
    import random
    pr = Replay(win, shape)
    WD.play(pr, shape, rnd=1, seed=seed)
    pr.prev = {"P": pr._P, "R": pr._R}
    rd = Round(shape, pr.w_tree.root, pr.session, 2, random.Random(seed + 1), "duo")
    op = pr.run(rd)
    why = rd.judge(op)
    return WD.RoundResult(why is None, why, rd.draw_, rd.sigma, op, rd.log)


#: each attack and whether today's check (duo) catches it (False: xfail, with the reason PIN's v3 closes it)
ATTACKS = {
    "skip-fraction": (True, None),
    "forge-committed": (True, None),
    "forge-opened": (True, None),
    "mitm": (False, "today's check constrains only the drawn step; PIN's v3 binds every absorbed prefix word"),
    "swap-served": (True, None),
    "swap-opened": (True, None),
    "skip-filler": (False, "today's check folds duo, so H1 + fx1's duo-without-filler receipts pass; PIN's v3 folds fx1"),
    "replay": (True, None),
    "commit-after-coin": (True, None),
    "commit-after-draw": (True, None),
}
