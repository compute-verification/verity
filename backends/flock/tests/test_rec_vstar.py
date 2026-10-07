"""The reference for V*'s hashing (`verity_flock.rec_vstar`): query positions from coins (PROTOCOL.md §13.5), salted
``hm96-sha512/v1`` openings climbing to a cap node, and how the coin server's record yields each level's cap and query
round. The pod job (``pod/80-rec-inner.sh``) runs `check` over a real inner session."""
from __future__ import annotations

import hashlib

import pytest
from verity_flock import rec_vstar as V


def _salt(i: int) -> bytes:
    return hashlib.shake_256(b"rec-vstar-salt" + i.to_bytes(4, "little")).digest(V.SALT)


def _tree(depth: int, width: int):
    rows = [[(i * 1000 + k) * 0x9E3779B97F4A7C15 % (1 << 128) for k in range(width)] for i in range(1 << depth)]
    salts = [_salt(i) for i in range(1 << depth)]
    layers = [[V.leaf(r, s) for r, s in zip(rows, salts)]]
    while len(layers[-1]) > 1:
        lo = layers[-1]
        layers.append([hashlib.sha512(lo[i] + lo[i + 1]).digest() for i in range(0, len(lo), 2)])
    return rows, salts, layers


def _query(rows, salts, layers, pos: int, c: int) -> V.Query:
    d = len(layers) - 1
    sibs = [layers[h][(pos >> h) ^ 1] for h in range(d - c)]
    return V.Query(0, 0, 0, pos, d, c, layers[d - c][pos >> (d - c)], rows[pos], salts[pos], sibs)


def test_positions_fill_each_stratum():
    Q, d = 11, 4
    coins = [0x0123456789ABCDEF * (j + 1) for j in range(Q)]
    pos = V.positions(Q, d, coins)
    assert V.strata(Q, d) == [3, 1, 0]
    at = 0
    for c in V.strata(Q, d):
        for s in range(1 << c):
            assert pos[at] >> (d - c) == s
            assert pos[at] & ((1 << (d - c)) - 1) == coins[at] & ((1 << (d - c)) - 1)
            at += 1
    with pytest.raises(ValueError):
        V.positions(Q, d, coins[:-1])


def test_an_opening_climbs_to_its_cap_node_and_nothing_else_does():
    rows, salts, layers = _tree(depth=4, width=16)
    for pos in range(16):
        q = _query(rows, salts, layers, pos, c=1)
        assert V.opens(q)
        bad_row = list(q.row)
        bad_row[pos % 16] ^= 1
        bad_salt = bytes([q.salt[0] ^ 1]) + q.salt[1:]
        bad_sib = [bytes([q.sibs[0][0] ^ 1]) + q.sibs[0][1:]] + q.sibs[1:]
        for forged in (
            V.Query(**{**q.__dict__, "row": bad_row}),
            V.Query(**{**q.__dict__, "salt": bad_salt}),
            V.Query(**{**q.__dict__, "sibs": bad_sib}),
            V.Query(**{**q.__dict__, "sibs": q.sibs[:-1]}),
            V.Query(**{**q.__dict__, "pos": pos ^ 1}),
        ):
            assert not V.opens(forged)


def _round(stream, k, caps=(), commits=(), n=0):
    return {"stream": stream, "k": k, "caps": list(caps), "commits": list(commits), "n": n}


def test_ligeritos_copy_of_the_initial_cap_is_not_a_level():
    a, b, c = (bytes([x]) * 128 for x in (1, 2, 3))
    s = "circuit/rep0"
    public = {"rounds": [_round(s, 0, [a.hex()], n=6), _round(s, 1, [a.hex()], ["00"], 18), _round(s, 2, [b[:64].hex()], ["00"], 3),
                         _round("circuit/rep1", 0, [c.hex()], n=6)]}
    assert V.public_caps(public, s) == [[a[:64], a[64:]], [b[:64]]]
    public["rounds"][1]["caps"] = [c.hex()]
    with pytest.raises(ValueError):
        V.public_caps(public, s)


def test_query_rounds_follow_the_levels_in_order():
    s = "circuit/rep0"
    public = {"rounds": [_round(s, 0, ["aa"], n=7), _round(s, 1, n=9), _round(s, 2, ["bb"], ["00"], 5), _round(s, 3, n=7),
                         _round(s, 4, n=5), _round(s, 5, commits=["00"], n=3)]}
    assert [r["k"] for r in V.query_rounds(public, s, [7, 5, 3])] == [3, 4, 5]
    with pytest.raises(ValueError):
        V.query_rounds(public, s, [7, 5, 3, 2])
