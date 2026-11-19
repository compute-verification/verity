"""A Beneš network on bits: `IoTieRoute{W, D}` takes 2^D W-bit inputs to the outputs a permutation sends them to, by
(2D − 1)·2^(D − 1) switch bits (`IoTieSwitch{W}`, W ANDs each), every bit a value; `benes` sets the switches of a
permutation (the looping algorithm) and `benes_apply` runs the network in the clear."""
from __future__ import annotations

from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.trace import part as _part
from verity.circuits.defs import CompositeDefinition, bind
from verity.circuits.refs import Coll
from verity.circuits.types import Array

BIT = BL.BIT


def _refs(c: Coll) -> list:
    return [c.refs[k] for k in range(c.refs.count)]


def _coll(t, refs: list) -> Coll:
    return Coll(t, TR._refs(list(refs)))


def route_depth(cells: int, entries: int) -> int:
    """D of the routing over M = 2^D ≥ max(cells, entries) positions."""
    return max(1, (max(cells, entries) - 1).bit_length())


def benes(perm: Sequence[int]) -> list[int]:
    """The switch bits of a Beneš network that takes input i to output perm[i] (the looping algorithm), in
    `IoTieRoute`'s order: the input column, the upper half's network, the lower half's, the output column."""
    n = len(perm)
    if n < 2 or n & (n - 1) or sorted(perm) != list(range(n)):
        raise ValueError("a Beneš network routes a permutation of a power of two at least 2")
    if n == 2:
        return [perm[0]]
    inv = [0] * n
    for i, d in enumerate(perm):
        inv[d] = i
    color: list[int | None] = [None] * n
    for start in range(0, n, 2):
        i = start
        # i goes up and its switch partner down; the input sharing that partner's output switch goes up next
        while color[i] is None:
            color[i], color[i ^ 1] = 0, 1
            i = inv[perm[i ^ 1] ^ 1]
    upper, lower, out = [0] * (n // 2), [0] * (n // 2), [0] * (n // 2)
    for i in range(n):
        (upper if color[i] == 0 else lower)[i // 2] = perm[i] // 2
        if color[i] == 0:
            out[perm[i] // 2] = perm[i] & 1
    return [color[2 * k] for k in range(n // 2)] + benes(upper) + benes(lower) + out  # type: ignore[list-item]


def benes_apply(xs: Sequence, bits: Sequence[int]) -> list:
    """`IoTieRoute` in the clear: the network's outputs for its inputs under the switch bits."""
    it = iter(bits)

    def run(v: list) -> list:
        if len(v) == 2:
            return v[::-1] if next(it) else v
        up, down = [], []
        for k in range(len(v) // 2):
            a, b = (v[2 * k + 1], v[2 * k]) if next(it) else (v[2 * k], v[2 * k + 1])
            up.append(a)
            down.append(b)
        up, down = run(up), run(down)
        res = []
        for a, b in zip(up, down):
            res += [b, a] if next(it) else [a, b]
        return res

    out = run(list(xs))
    if next(it, None) is not None:
        raise ValueError("more switch bits than the network has")
    return out


def _check_route(S) -> None:
    if type(S.D) is not int or not 1 <= S.D <= 32 or type(S.W) is not int or S.W < 1:
        raise ValueError(f"a Beneš network has 1 <= D <= 32 and W >= 1, not D={S.D!r}, W={S.W!r}")


def _sig_route(S):
    _check_route(S)
    return ((("xs", Array(1 << S.D, Array(S.W, BIT))), ("switches", Array((2 * S.D - 1) << (S.D - 1), BIT))),
            Array(1 << S.D, Array(S.W, BIT)))


def _sig_switch(S):
    if type(S.W) is not int or S.W < 1:
        raise ValueError(f"a switch has W >= 1, not W={S.W!r}")
    return (("a", Array(S.W, BIT)), ("b", Array(S.W, BIT)), ("s", BIT)), Array(2, Array(S.W, BIT))


def _build_switch(C, S, a, b, s):
    t = [C.AND(s[0], x ^ y) for x, y in zip(a, b)]
    return [[x ^ u for x, u in zip(a, t)], [y ^ u for y, u in zip(b, t)]]


IoTieSwitch = _part(
    "IoTieSwitch", ("W",), _sig_switch, _build_switch,
    "(a, b), swapped iff s: W ANDs")


def _body_route(B, S, xs, switches_):
    W, h = S.W, 1 << (S.D - 1)
    sw, half = _refs(switches_), Array(h, Array(W, BIT))

    def words(refs: list) -> list:
        return [refs[W * k:W * (k + 1)] for k in range(len(refs) // W)]

    def column(pairs: list, bits: list) -> list:
        """The h switches' (a, b) pairs, swapped by their bits: one batch of `IoTieSwitch`."""
        out = words(_refs(B.batch(bind(IoTieSwitch, W=W), _coll(half, [r for a, _ in pairs for r in a]),
                                     _coll(half, [r for _, b in pairs for r in b]), _coll(Array(h, BIT), bits),
                                     axes=(0, 0, 0))))
        return list(zip(out[0::2], out[1::2]))

    x = words(_refs(xs))
    if S.D == 1:
        last = column([(x[0], x[1])], sw)
    else:
        # `benes`' bit order: the input column, the upper half's, the lower half's, the output column
        sub = (2 * S.D - 3) << (S.D - 2)
        first = column([(x[2 * k], x[2 * k + 1]) for k in range(h)], sw[:h])
        net = bind(IoTieRoute, W=W, D=S.D - 1)
        up, down = (words(_refs(B.call(net, _coll(half, [r for p in first for r in p[i]]),
                                          _coll(Array(sub, BIT), sw[h + i * sub:h + (i + 1) * sub]))))
                    for i in (0, 1))
        last = column(list(zip(up, down)), sw[h + 2 * sub:])
    return _coll(Array(2 * h, Array(W, BIT)), [r for p in last for w in p for r in w])


IoTieRoute = CompositeDefinition(
    "IoTieRoute", 1, ("W", "D"), _sig_route, _body_route, conformance="exact-model-tested",
    doc="a Beneš network on 2^D W-bit inputs: the input column's 2^(D − 1) switches (`IoTieSwitch`: s swaps "
        "inputs 2k and 2k + 1 into the halves), the upper half's `IoTieRoute{W, D − 1}`, the lower half's, the "
        "output column's switches; W ANDs a switch")


__all__ = ["IoTieRoute", "IoTieSwitch", "benes", "benes_apply", "route_depth"]
