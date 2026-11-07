"""A gate-level prototype of the DSA top-k's mask form, bit-sliced over test vectors, with its ANDs counted.

The form: no `run` value.  From a row's N f32 logits the circuit computes
  1. the mask m_j = [ext_j >= theta], ext_j = (topk_key(s_j), N-1-j) (a strict total order: key descending, index ascending is
     dsa_topk's tie rule), theta the TOPK-th largest ext, by a radix select MSB first over the 32 + b bits of ext;
  2. each member's rank r_j = #{i < j : m_i} (a Brent-Kung prefix count);
  3. the order-preserving concentrator: an inverse butterfly (stage t exchanges positions p and p ^ 2^t, t = 0..L-1), each
     member routed to output r_j, its settings computed from (valid, r_j's bits) carried through the same network.
Each of the N history rows crosses the concentrator's live switches at one AND a payload bit, as through a Waksman switch.

Conflict-freeness: two members j < j' meet at stage t only if j, j' lie in one block of 2^(t+1) and r_j = r_j' mod 2^(t+1); but
0 < r_j' - r_j <= j' - j < 2^(t+1).  The prototype asserts it on every vector anyway.

Wires: a bool is a structural constant, an int a value bit-sliced over the vectors (bit v is vector v's).  Counted: two-input
ANDs between values (an OR or a mux is one AND, XOR and NOT none), as `definition_ands` counts a Definition.
"""
from __future__ import annotations

import random
import struct


class Sim:
    def __init__(self, V: int):
        self.full = (1 << V) - 1
        self.ands = 0

    def AND(self, a, b):
        if type(a) is bool:
            return b if a else False
        if type(b) is bool:
            return a if b else False
        self.ands += 1
        return a & b

    def NOT(self, a):
        return (not a) if type(a) is bool else a ^ self.full

    def X(self, a, b):
        if type(a) is bool:
            return self.NOT(b) if a else b
        if type(b) is bool:
            return self.NOT(a) if b else a
        return a ^ b

    def X3(self, a, b, c):
        return self.X(self.X(a, b), c)

    def OR(self, a, b):
        return self.X3(a, b, self.AND(a, b))

    def mux(self, s, a, b):
        """s ? b : a."""
        return self.X(a, self.AND(s, self.X(a, b)))

    def add(self, x: list, y: list, width: int, cin=False) -> list:
        """x + y + cin mod 2^width, LSB first: one AND a bit (carry ((a^c)&(b^c))^c)."""
        x = x + [False] * (width - len(x))
        y = y + [False] * (width - len(y))
        out, c = [], cin
        for i in range(width):
            a, b = x[i], y[i]
            out.append(self.X3(a, b, c))
            if i + 1 < width:
                c = self.X(self.AND(self.X(a, c), self.X(b, c)), c)
        return out

    def sub(self, x: list, y: list, width: int) -> list:
        y = y + [False] * (width - len(y))
        return self.add(x, [self.NOT(b) for b in y], width, cin=True)

    def ge(self, x: list, y: list):
        """x >= y unsigned: the carry out of x + ~y + 1, one AND a bit."""
        n = max(len(x), len(y))
        x = x + [False] * (n - len(x))
        y = y + [False] * (n - len(y))
        c = True
        for i in range(n):
            a, b = x[i], self.NOT(y[i])
            c = self.X(self.AND(self.X(a, c), self.X(b, c)), c)
        return c

    def popcount(self, bits: list) -> list:
        nums = [[b] for b in bits]
        while len(nums) > 1:
            nxt = [self.add(nums[i], nums[i + 1], max(len(nums[i]), len(nums[i + 1])) + 1) for i in range(0, len(nums) - 1, 2)]
            nums = nxt + nums[len(nums) - len(nums) % 2:]
        return nums[0]


def const_word(value: int, width: int) -> list:
    return [bool(value >> i & 1) for i in range(width)]


def topk_key(bits: int) -> int:
    bits &= 0xFFFFFFFF
    return ~bits & 0xFFFFFFFF if bits & 0x80000000 else bits | 0x80000000


def topk_set(scores, TOPK):
    return set(sorted(range(len(scores)), key=lambda j: (-topk_key(scores[j]), j))[:TOPK])


def select(sim: Sim, keys: list, N: int, TOPK: int) -> list:
    """The mask.  keys[j] is topk_key(s_j) as 32 wires LSB first (free from the f32's bits).  Radix select over ext's bits MSB
    first; eq_j: ext_j's bits so far equal theta's; gt_j: greater; r: the members still to take."""
    b = max(1, (N - 1).bit_length())
    rb = TOPK.bit_length() + 1
    ext = [const_word(N - 1 - j, b) + keys[j] for j in range(N)]
    eq, gt = [True] * N, [False] * N
    r = const_word(TOPK, rb)
    for t in reversed(range(32 + b)):
        a = [sim.AND(eq[j], ext[j][t]) for j in range(N)]
        cnt = sim.popcount(a)
        p = sim.ge(cnt, r)                                   # theta's bit t: enough members with a 1 here
        r = [sim.mux(p, x, y) for x, y in zip(sim.sub(r, cnt, rb), r)]
        pe = [sim.AND(p, e) for e in eq]
        ap = [sim.AND(pe[j], ext[j][t]) for j in range(N)]  # a & p
        gt = [sim.X3(gt[j], a[j], ap[j]) for j in range(N)]  # gt | (a & ~p); gt and eq are disjoint
        eq = [sim.X3(eq[j], a[j], pe[j]) for j in range(N)]  # eq & (bit == p): p ? a : eq ^ a
    return [sim.X(gt[j], eq[j]) for j in range(N)]


def prefix_ranks(sim: Sim, mask: list, width: int) -> list:
    """Exclusive prefix counts mod 2^width (Brent-Kung up and down sweeps)."""
    N = len(mask)
    L = 1
    while L < N:
        L *= 2
    v = [[m] for m in mask] + [[False]] * (L - N)
    d = 1
    while d < L:
        for i in range(2 * d - 1, L, 2 * d):
            v[i] = sim.add(v[i - d], v[i], min(width, max(len(v[i - d]), len(v[i])) + 1))
        d *= 2
    v[L - 1] = [False] * width
    d = L // 2
    while d >= 1:
        for i in range(2 * d - 1, L, 2 * d):
            left = v[i - d]
            v[i - d] = v[i]
            v[i] = sim.add(v[i], left, width)
        d //= 2
    return v[:N]


def concentrate(sim: Sim, mask: list, ranks: list, payload: list, TOPK: int):
    """The inverse butterfly over P = 2^ceil(log2 N) positions (pads constant invalid).  Returns (outputs 0..TOPK-1, switches with a
    live output, control ANDs (settings and tags), payload ANDs)."""
    N = len(mask)
    L = max(1, (N - 1).bit_length())
    kb = max(1, (TOPK - 1).bit_length())
    P = 1 << L
    W = len(payload[0])
    dead = (False, [False] * kb, [False] * W)
    pos = [(mask[j], ranks[j][:kb], payload[j]) for j in range(N)] + [dead] * (P - N)
    switches = pay_ands = 0
    start = sim.ands
    for t in range(L):
        def live(p, t=t):
            return all(not (p >> u & 1) for u in range(kb, t + 1))
        nxt = list(pos)
        for p in range(P):
            if p >> t & 1:
                continue
            q = p | (1 << t)
            lo_live, hi_live = live(p), live(q)
            (vl, dl, xl), (vh, dh, xh) = pos[p], pos[q]
            if not (lo_live or hi_live):
                nxt[p] = nxt[q] = dead
                continue
            if vl is False and vh is False:
                nxt[p] = nxt[q] = dead
                continue
            if t < kb:
                s = sim.mux(vl, sim.AND(vh, sim.NOT(dh[t])), dl[t])
                wl, wh = dl[t], dh[t]
            else:
                s, wl, wh = vh, False, False
            both = sim.full if type(vl) is int and type(vh) is int else 0
            if both:
                clash = vl & vh & ~(sim.X(wl, wh) if type(sim.X(wl, wh)) is int else (sim.full if sim.X(wl, wh) else 0))
                assert clash & sim.full == 0, f"conflict at stage {t}"
            switches += 1
            before = sim.ands
            lo_x = [sim.mux(s, a, c) for a, c in zip(xl, xh)] if lo_live else None
            if hi_live:
                hi_x = [sim.X3(a, c, y) for a, c, y in zip(xl, xh, lo_x)] if lo_live else [sim.mux(s, c, a) for a, c in zip(xl, xh)]
            pay_ands += sim.ands - before
            tv, th = [vl] + dl[t + 1:], [vh] + dh[t + 1:]
            lo_t = [sim.mux(s, a, c) for a, c in zip(tv, th)]
            hi_t = [sim.X3(a, c, y) for a, c, y in zip(tv, th, lo_t)]
            pad = [False] * (t + 1)
            nxt[p] = (lo_t[0], pad + lo_t[1:], lo_x) if lo_live else dead
            nxt[q] = (hi_t[0], pad + hi_t[1:], hi_x) if hi_live else dead
        pos = nxt
    return [pos[s][2] for s in range(TOPK)], switches, sim.ands - start - pay_ands, pay_ands


def run(N: int, TOPK: int, V: int, seed: int) -> dict:
    """The form on V random rows (scores from a pool of about N/3 floats, so ties are common, with +-0 and +-inf), checked
    against `topk_set` in ascending order, and its ANDs."""
    rng = random.Random(seed)
    pool = [struct.unpack("<I", struct.pack("<f", rng.uniform(-4, 4)))[0] for _ in range(max(4, N // 3))]
    pool += [0x80000000, 0, 0x7F800000, 0xFF800000]
    rows = [[rng.choice(pool) for _ in range(N)] for _ in range(V)]
    sim = Sim(V)
    keys = []
    for j in range(N):
        ks = [topk_key(rows[v][j]) for v in range(V)]
        keys.append([sum(((k >> i) & 1) << v for v, k in enumerate(ks)) for i in range(32)])
    mask = select(sim, keys, N, TOPK)
    sel = sim.ands
    kb = max(1, (TOPK - 1).bit_length())
    ranks = prefix_ranks(sim, mask, kb + 1)
    rank = sim.ands - sel
    b = max(1, (N - 1).bit_length())
    payload = [[sim.full if j >> i & 1 else 0 for i in range(b)] for j in range(N)]  # values, as history rows are
    outs, switches, ctrl, pay = concentrate(sim, mask, ranks, payload, TOPK)
    for v in range(V):
        want = sorted(topk_set(rows[v], TOPK))
        bit = lambda w: (w >> v & 1) if type(w) is int else int(w)  # noqa: E731
        assert sorted(j for j in range(N) if bit(mask[j])) == want, (N, TOPK, v)
        got = [sum(bit(w[i]) << i for i in range(b)) for w in outs]
        assert got == want, (N, TOPK, v, got[:8], want[:8])
    return {"N": N, "TOPK": TOPK, "vectors": V, "select_ands": sel, "rank_ands": rank, "live_switches": switches,
            "control_ands": ctrl, "index_payload_ands": pay, "index_bits": b}


if __name__ == "__main__":
    import json
    print(json.dumps([run(N, K, V=48, seed=N * 7 + K) for N, K in ((5, 2), (16, 4), (33, 8), (64, 16), (100, 32), (257, 64))],
                     indent=1))
