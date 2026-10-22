"""`weights.weights_roots`, the streaming b/<K> commitment the auditor registers a model with, against
`registered.Registered.commit` over the same rows and salts."""
from __future__ import annotations

import random
import struct

import pytest

from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw.audit import hm96_salt
from verity.core.protocols.sampled_proofs.one_stage import registered as G

CIRCUIT = "ab" * 64
PREFIX = "h1/"
KEY = bytes(range(32))


def linear(rng: random.Random, n: int, k: int) -> list[bytes]:
    return [Wt.row_bytes(bytes(rng.randrange(256) for _ in range(k)), struct.pack("<f", rng.uniform(1e-4, 1.0)))
            for _ in range(n)]


def model(seed: int = 0) -> tuple[list[tuple[int, int]], list[list[bytes]]]:
    rng = random.Random(seed)
    shapes = [(6, 8), (3, 8), (5, 12), (4, 8)]
    return shapes, [linear(rng, n, k) for n, k in shapes]


def committed(shapes, rows, key=KEY):
    """Each K's root as `Registered.commit` takes the rows, with `B_NONCE`'s salts."""
    out = {}
    for i, k in enumerate(Wt.ks(shapes)):
        words = [Wt.row_words(r) for w, (_n, kw) in enumerate(shapes) if kw == k for r in rows[w]]
        out[k] = G.Registered.commit(CIRCUIT, Wt.value(PREFIX, k), words,
                                     [hm96_salt(key, Wt.B_NONCE + i, j) for j in range(len(words))], schema=G.SCHEMA_V2)
    return out


def test_roots_and_entries_are_registered_commits():
    shapes, rows = model()
    roots = Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: iter(rows[w]), KEY)
    ref = committed(shapes, rows)
    assert sorted(roots) == [8, 12]
    for k, r in ref.items():
        assert roots[k] == r.root
        assert Wt.entry(CIRCUIT, PREFIX, k, r.leaves, roots[k]) == r.entry()
        assert r.words == Wt.b_words(k)


def test_weights_root_folds_by_ascending_k():
    shapes, rows = model()
    roots = Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY)
    assert Wt.weights_root(roots) == Wt.weights_root(dict(reversed(list(roots.items()))))
    assert Wt.weights_root(roots) != Wt.weights_root({**roots, 8: roots[12]})


def test_a_changed_code_scale_or_key_moves_the_root():
    shapes, rows = model()
    base = Wt.weights_root(Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY))
    code = [list(r) for r in rows]
    code[1][2] = bytes([code[1][2][0] ^ 1]) + code[1][2][1:]
    scale = [list(r) for r in rows]
    scale[3][0] = scale[3][0][:-1] + bytes([scale[3][0][-1] ^ 0x80])
    for other, key in ((code, KEY), (scale, KEY), (rows, bytes(32))):
        assert Wt.weights_root(Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: other[w], key)) != base


def test_refuses_rows_of_another_width_or_count():
    shapes, rows = model()
    with pytest.raises(ValueError, match="row 0 is"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: [r + b"\0\0" for r in rows[w]], KEY)
    with pytest.raises(ValueError, match="gave 5 rows"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w][:-1], KEY)
    with pytest.raises(ValueError, match="32 bytes"):
        Wt.weights_roots(CIRCUIT, PREFIX, shapes, lambda w: rows[w], KEY[:16])
    with pytest.raises(ValueError, match="positive even"):
        Wt.ks([(3, 7)])
    with pytest.raises(ValueError, match="whole u16"):
        Wt.row_bytes(b"\1\2\3", b"\0" * 4)
