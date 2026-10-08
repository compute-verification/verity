"""The pipeline against Lean's own definitions, evaluated by Lean on small cases (`fixtures/lean_vectors.json`,
`lean/PousRefCheck.lean`): `c1'`, `c2'`/`topLabel'`, `transpose`, `decodeLower`, `P3Meets.tagOf` and `decodeW`, the
band graph's parents, and the column game `colRound` with its key-gated reverse.

The primitive is the same computable toy on both sides: `toy_h` is a monolithic `H` (`RandomOracleKey`), and `ToyP`
ignores the tweak, since the Lean model's `P` has none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.core.protocols.pous import adversary as A
from verity.core.protocols.pous.labels import component, join, transpose, w_labels
from verity.core.protocols.pous.oracle import KeyQuery, Tweak, run
from verity.core.protocols.pous.params import Params
from verity.core.protocols.pous.primitives import Primitives, RandomOracleKey, key_query
from verity.core.protocols.pous.scheme import decode_prog, encode_prog

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "lean_vectors.json").read_text())
P61 = 2305843009213693951


def mix(acc: int, x: int) -> int:
    return (acc * 1000003 + x + 1) % P61


def bits_mix(acc: int, value: int, width: int) -> int:
    for i in range(width):
        acc = mix(acc, (value >> i) & 1)
    return acc


def toy_h(n: int, wc: int):
    def h(q: KeyQuery) -> int:
        a = mix(mix(bits_mix(7, q.tag, 256), q.layer), q.node)
        present = dict(q.parents)
        for u in range(n):
            if u in present:
                a = bits_mix(mix(a, 3), present[u], n * wc)
            else:
                a = mix(a, 2)
        out = 0
        for c in range(n):
            for j in range(wc):
                out |= (((mix(a, c * wc + j + 5) * 11400714819323198485) >> 33) & 1) << (c * wc + j)
        return out
    return h


class ToyP:
    """Lean `toyP`: `P x = rotate (x ⊕ toyConst)`, bit `j` of component `v` coming from bit `j - 1` of component `v - 1`."""

    name = "lean-toy-p"

    def __init__(self, n: int, wc: int) -> None:
        self.n, self.wc, self.width = n, wc, n * wc
        self.const = join([sum(((c * 7 + j * 3) % 5 < 2) << j for j in range(wc)) for c in range(n)], wc)

    def _move(self, x: int, dv: int, dj: int) -> int:
        n, wc = self.n, self.wc
        out = 0
        for v in range(n):
            for j in range(wc):
                out |= ((component(x, (v - dv) % n, wc) >> ((j - dj) % wc)) & 1) << (v * wc + j)
        return out

    def forward(self, t: Tweak, x: int) -> int:
        return self._move(x ^ self.const, 1, 1)

    def inverse(self, t: Tweak, y: int) -> int:
        return self._move(y, -1, -1) ^ self.const


@pytest.mark.parametrize("case", FIXTURE["encode"], ids=lambda c: f"n{c['n']}-wc{c['wc']}-k{c['k']}")
def test_encoding_matches_lean(case):
    n, wc, k = case["n"], case["wc"], case["k"]
    params = Params(n, wc, k)
    prims = Primitives(RandomOracleKey(toy_h(n, wc)), ToyP(n, wc))
    w, salt, tag = int(case["w"]), int(case["salt"]), int(case["tag"])
    assert tag == salt
    assert [list(params.parents(v)) for v in range(n)] == case["parents"]
    [(c1, c2)], _ = run(encode_prog(params, prims.h, w, salt), prims)
    assert [str(x) for x in c1] == case["lower"]
    assert [str(x) for x in c2] == case["top"]
    assert [str(x) for x in transpose(c1, n, wc)] == case["transpose"]
    decoded, _ = run(decode_prog(params, prims.h, salt, c2), prims)
    assert str(decoded) == case["decode"] == case["w"]
    keys = [prims.h.fn(key_query(params, tag, 1, v, c2)) for v in range(n)]
    lower = transpose([prims.p.inverse(Tweak(tag, 1, v), c2[v]) ^ keys[v] for v in range(n)], n, wc)
    assert [str(x) for x in lower] == case["decode_lower"]
    assert str(keys[n - 1]) == case["h_sample"]
    first = w_labels(params, w)[0][0]
    assert str(prims.p.forward(Tweak(0, 0, 0), first)) == case["p_sample"]
    assert str(prims.p.inverse(Tweak(0, 0, 0), first)) == case["p_inverse_sample"]


def _holding_of(n: int, seed: int) -> A.Holding:
    return A.Holding(frozenset(w for w in range(n) if (seed // 2 ** w) % 5 == 0),
                     frozenset((w, u) for w in range(n) for u in range(n) if (seed // 3 ** (w * n + u)) % 3 == 0),
                     frozenset(u for u in range(n) if (seed // 2 ** (u + n)) % 3 != 0))


@pytest.mark.parametrize("case", FIXTURE["game"], ids=lambda c: f"n{c['n']}-k{c['k']}-{c['seed']}")
def test_column_game_matches_lean(case):
    params = Params(case["n"], 8, case["k"])
    for r, state in enumerate(case["reach"]):
        c = A.col_reach(params, _holding_of(case["n"], case["seed"]), r)
        assert sorted(c.lower_whole) == state["lower_whole"]
        assert sorted(c.lower_col) == [tuple(p) for p in state["lower_col"]]
        assert sorted(c.top_whole) == state["top_whole"]
        assert c.size(case["n"]) == state["size"]
