"""The sumcheck for Σ_x f(x)·g(x) over x in {0,1}^m, f and g multilinear and given by their tables (entry Σ_i x_i 2^i).

Round i binds variable i: the prover sends the round polynomial's values at 0, 1 and 2 (it has degree 2), the verifier checks
that the first two sum to the running claim, and a coin r_i moves the claim to the polynomial's value at r_i. After m rounds
the claim is f(r)·g(r), whose two factors the caller proves.
"""
from __future__ import annotations

from collections.abc import Sequence

from .field import ONE, Ext, add, ext_bytes, ext_from_bytes, inv, mul, scale, sub, total
from .session import ProverSession, Refused, VerifierSession, ext_coin

_HALF = inv((2, 0, 0))


def _at(v0: Ext, v1: Ext, v2: Ext, r: Ext) -> Ext:
    """The degree-2 polynomial through (0, v0), (1, v1), (2, v2), at r."""
    r1, r2 = sub(r, ONE), sub(r, (2, 0, 0))
    l0 = mul(mul(r1, r2), _HALF)
    l1 = scale(mul(r, r2), -1)
    l2 = mul(mul(r, r1), _HALF)
    return total((mul(l0, v0), mul(l1, v1), mul(l2, v2)))


def _fold(t: list[Ext], r: Ext) -> list[Ext]:
    return [add(t[2 * j], mul(r, sub(t[2 * j + 1], t[2 * j]))) for j in range(len(t) // 2)]


def prove(s: ProverSession, f: Sequence[Ext], g: Sequence[Ext], label: str) -> tuple[list[Ext], Ext, Ext]:
    """The point r and f(r), g(r)."""
    f, g, point = list(f), list(g), []
    while len(f) > 1:
        h = len(f) // 2
        v0 = total(mul(f[2 * j], g[2 * j]) for j in range(h))
        v1 = total(mul(f[2 * j + 1], g[2 * j + 1]) for j in range(h))
        v2 = total(mul(sub(scale(f[2 * j + 1], 2), f[2 * j]), sub(scale(g[2 * j + 1], 2), g[2 * j])) for j in range(h))
        s.message(f"{label}/round", ext_bytes(v0) + ext_bytes(v1) + ext_bytes(v2))
        r = ext_coin(s)
        point.append(r)
        f, g = _fold(f, r), _fold(g, r)
    return point, f[0], g[0]


def verify(s: VerifierSession, m: int, claim: Ext, label: str) -> tuple[list[Ext], Ext]:
    """The point r and the claim f(r)·g(r) the rounds leave."""
    point = []
    for _ in range(m):
        vs = ext_from_bytes(s.message(f"{label}/round"))
        if vs is None or len(vs) != 3:
            raise Refused("encoding", f"{label}/round")
        if add(vs[0], vs[1]) != claim:
            raise Refused("sumcheck-round", label)
        r = ext_coin(s)
        point.append(r)
        claim = _at(vs[0], vs[1], vs[2], r)
    return point, claim
