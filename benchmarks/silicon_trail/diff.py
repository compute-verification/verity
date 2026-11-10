"""Differences through the chain's step, and cancelling trails.

A difference is a state (25 lanes) of XOR differences.  Through one step:

    a   the difference entering the round: the previous round's output difference xor the placed input difference
    b   = L(a), L = pi . rho . theta: linear, so it maps differences with probability 1; iota drops out
    c   an output difference of chi compatible with b

chi works on the 5w rows of 5 bits.  For a row's input difference the compatible output differences form an affine
space of dimension `ROW_WEIGHT[difference]`, each equally likely over a uniform row value.  The weight w(b) of a
round is the sum over rows, and 2^-w(b) is the probability, over a uniform state, that b goes to one given compatible c.

A k-step cancelling trail is input differences d_1 != 0, d_2, ..., d_k and round differences (a_i, b_i, c_i),
i = 1..k-1, with a_1 = P(d_1), a_{i+1} = c_i xor P(d_{i+1}), and c_{k-1} = P(d_k): after the k-th absorb the two
chains hold the same state.  Its weight is sum_i w(b_i); 2^-weight is its probability if the rounds' events are
independent, which the experiment (`collide.c`) tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from keccak import PLACEMENTS, chi, linear, place, unplace


def _row_tables() -> tuple[list[frozenset[int]], list[int]]:
    outs: list[frozenset[int]] = []
    weight: list[int] = []
    for d in range(32):
        seen: dict[int, int] = {}
        for v in range(32):
            lanes_v = [(v >> x) & 1 for x in range(5)] + [0] * 20
            lanes_u = [((v ^ d) >> x) & 1 for x in range(5)] + [0] * 20
            cv, cu = chi(lanes_v, 1), chi(lanes_u, 1)
            o = sum((cv[x] ^ cu[x]) << x for x in range(5))
            seen[o] = seen.get(o, 0) + 1
        counts = set(seen.values())
        assert len(counts) == 1, "chi's output differences are equally likely"
        n = len(seen)
        assert n & (n - 1) == 0
        outs.append(frozenset(seen))
        weight.append(n.bit_length() - 1)
    return outs, weight


ROW_OUT, ROW_WEIGHT = _row_tables()
#: for an output difference of a row, the input differences it is compatible with
ROW_IN = [frozenset(d for d in range(32) if o in ROW_OUT[d]) for o in range(32)]


def rows(s: list[int], w: int):
    """The 5w rows of a state as ((y, z), 5-bit value), value bit x = a[x, y, z]."""
    for y in range(5):
        for z in range(w):
            v = 0
            for x in range(5):
                v |= ((s[x + 5 * y] >> z) & 1) << x
            yield (y, z), v


def weight(b: list[int], w: int) -> int:
    return sum(ROW_WEIGHT[v] for _, v in rows(b, w))


def active_rows(b: list[int], w: int) -> int:
    return sum(1 for _, v in rows(b, w) if v)


def compatible(b: list[int], c: list[int], w: int) -> bool:
    return all(vc in ROW_OUT[vb] for (_, vb), (_, vc) in zip(rows(b, w), rows(c, w)))


def is_outer(s: list[int], placement: str) -> bool:
    lanes = {x + 5 * y for x, y in PLACEMENTS[placement]}
    return all(v == 0 for i, v in enumerate(s) if i not in lanes)


def inner(s: list[int], placement: str) -> tuple[int, ...]:
    lanes = {x + 5 * y for x, y in PLACEMENTS[placement]}
    return tuple(v for i, v in enumerate(s) if i not in lanes)


def xor(a: list[int], b: list[int]) -> list[int]:
    return [u ^ v for u, v in zip(a, b)]


@dataclass
class Trail:
    """A k-step cancelling trail: k input differences and the k - 1 rounds between them."""

    w: int
    placement: str
    d: list[int]          # k input differences, each 4w bits
    a: list[list[int]]    # k - 1 round input differences
    b: list[list[int]]
    c: list[list[int]]

    @property
    def k(self) -> int:
        return len(self.d)

    @property
    def weights(self) -> list[int]:
        return [weight(b, self.w) for b in self.b]

    @property
    def weight(self) -> int:
        return sum(self.weights)

    def check(self) -> None:
        """Raise unless this is a cancelling trail: every a_i nonzero, each round linear then chi-compatible, the
        injections consistent, and the last output difference cancelled by d_k."""
        w, pl = self.w, self.placement
        assert len(self.a) == len(self.b) == len(self.c) == self.k - 1 >= 1
        assert self.a[0] == place(self.d[0], w, pl) and any(self.a[0]), "a_1 = P(d_1) != 0"
        for i in range(self.k - 1):
            assert any(self.a[i]), f"a_{i + 1} is zero: the trail ends earlier"
            assert self.b[i] == linear(self.a[i], w), f"b_{i + 1} != L(a_{i + 1})"
            assert compatible(self.b[i], self.c[i], w), f"c_{i + 1} is not an output difference of b_{i + 1}"
            nxt = xor(self.c[i], place(self.d[i + 1], w, pl))
            if i + 1 < self.k - 1:
                assert self.a[i + 1] == nxt, f"a_{i + 2} != c_{i + 1} xor P(d_{i + 2})"
            else:
                assert not any(nxt), "the last absorb does not cancel the difference"

    def to_json(self) -> dict:
        """Hex strings: an input difference as 4w bits (lane j at bits wj..), a state as 25w bits (bit w(5y+x)+z)."""
        w = self.w
        nb = (25 * w + 3) // 4

        def lanes(s: list[int]) -> str:
            return f"{sum(v << (w * i) for i, v in enumerate(s)):0{nb}x}"

        return {"w": w, "placement": self.placement, "k": self.k, "weight": self.weight, "weights": self.weights,
                "active_rows": [active_rows(b, w) for b in self.b], "d": [f"{d:0{w}x}" for d in self.d],
                "a": [lanes(s) for s in self.a], "b": [lanes(s) for s in self.b], "c": [lanes(s) for s in self.c]}

    @staticmethod
    def from_json(j: dict) -> "Trail":
        w = j["w"]
        m = (1 << w) - 1

        def lanes(h: str) -> list[int]:
            return [(int(h, 16) >> (w * i)) & m for i in range(25)]

        return Trail(w, j["placement"], [int(d, 16) for d in j["d"]], [lanes(s) for s in j["a"]],
                     [lanes(s) for s in j["b"]], [lanes(s) for s in j["c"]])


def trail_from_states(w: int, placement: str, a: list[list[int]], c: list[list[int]]) -> Trail:
    """The trail with round input differences a_1..a_{k-1} and output differences c_1..c_{k-1}; the input differences
    follow: d_1 = a_1's outer part, d_{i+1} = (a_{i+1} xor c_i)'s outer part, d_k = c_{k-1}."""
    d = [unplace(a[0], w, placement)]
    for i in range(len(a)):
        nxt = a[i + 1] if i + 1 < len(a) else [0] * 25
        d.append(unplace(xor(nxt, c[i]), w, placement))
    t = Trail(w, placement, d, a, [linear(s, w) for s in a], c)
    t.check()
    return t
