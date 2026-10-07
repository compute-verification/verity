"""The reference for V*'s hashing (`verity_flock.rec_vstar`): query positions from coins (PROTOCOL.md §13.5), openings with
plain SHA-512 leaves (and rec-step2's salted ``hm96-sha512/v1`` ones) climbing to a top's node, each top opening the salted
commitment the coin server received or refused by name, and how the coin server's record yields each level's query round. The
pod job (``pod/85-rec-reprice.sh``) runs `check` over a real inner session."""
from __future__ import annotations

import hashlib
import json

import pytest
from verity_flock import rec_live as RL
from verity_flock import rec_vstar as V


def _salt(i: int) -> bytes:
    return hashlib.shake_256(b"rec-vstar-salt" + i.to_bytes(4, "little")).digest(V.SALT)


def _tree(depth: int, width: int, salted: bool = False):
    rows = [[(i * 1000 + k) * 0x9E3779B97F4A7C15 % (1 << 128) for k in range(width)] for i in range(1 << depth)]
    salts = [_salt(i) if salted else None for i in range(1 << depth)]
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


@pytest.mark.parametrize("salted", [False, True])
def test_an_opening_climbs_to_its_top_node_and_nothing_else_does(salted):
    rows, salts, layers = _tree(depth=4, width=16, salted=salted)
    assert (layers[0][3] == hashlib.sha512(V.row_bytes(rows[3])).digest()) != salted
    for pos in range(16):
        q = _query(rows, salts, layers, pos, c=1)
        assert V.opens(q)
        bad_row = list(q.row)
        bad_row[pos % 16] ^= 1
        bad_sib = [bytes([q.sibs[0][0] ^ 1]) + q.sibs[0][1:]] + q.sibs[1:]
        forgeries = [{"row": bad_row}, {"sibs": bad_sib}, {"sibs": q.sibs[:-1]}, {"pos": pos ^ 1}]
        if salted:
            forgeries.append({"salt": bytes([q.salt[0] ^ 1]) + q.salt[1:]})
        for f in forgeries:
            assert not V.opens(V.Query(**{**q.__dict__, **f}))


def _round(stream, k, tops=(), commits=(), n=0, level0=0):
    return {"stream": stream, "k": k, "tops": list(tops), "level0": level0, "commits": list(commits), "n": n}


def _committed(cap: bytes, at: int):
    """A top's opening and its ``b || c`` a node, under deterministic salts."""
    salts = [_salt(at + i) for i in range(len(cap) // V.HASH)]
    return ({"nodes": [cap[i:i + V.HASH].hex() for i in range(0, len(cap), V.HASH)], "salts": [s.hex() for s in salts]},
            [b.hex() for b in RL.commit_top(cap, salts)])


def _session():
    """A rep stream as the firewall records it: level 0's top at Commit, its copies counted, a level-1 top salted."""
    a, b = bytes([1]) * 64 + bytes([3]) * 64, bytes([2]) * 64
    (oa, ca), (ob, cb) = _committed(a, 100), _committed(b, 200)
    s = "circuit/rep0"
    public = {"tops": RL.TOPS, "link": {"roots": {"circuit": ca}},
              "rounds": [_round(s, 0, n=6, level0=1), _round(s, 1, commits=["00"], n=18, level0=1), _round(s, 2, [cb], ["00"], 3),
                         _round("circuit/rep1", 0, n=6, level0=1)]}
    private = {"tops": RL.TOPS, "roots": {"circuit": oa}, "rounds": [{"stream": s, "k": 2, "tops": [ob]}]}
    return public, private, a, b


def test_the_tops_are_the_committed_ones_level_0s_copies_are_not_levels():
    public, private, a, b = _session()
    t0, t1 = V.tops(public, private, "circuit/rep0")
    assert (t0.nodes, t1.nodes) == ((a[:64], a[64:]), (b,))
    assert [t.nodes for t in V.tops(public, private, "circuit/rep1")] == [t0.nodes]
    assert list(t1.bcs) == [bytes.fromhex(x) for x in public["rounds"][2]["tops"][0]] == RL.commit_top(b, list(t1.salts))


@pytest.mark.parametrize("forge", ["node", "salt", "root-node", "missing", "short"])
def test_a_top_that_does_not_open_its_commitment_is_refused(forge):
    """A wrong top (one bit of a node) or a top under another salt than its commitment's, of a later level or of level 0,
    and a committed top the prover does not open."""
    public, private, _, _ = _session()
    op = private["rounds"][0]["tops"][0]
    if forge == "node":
        op["nodes"][0] = ("%02x" % (int(op["nodes"][0][:2], 16) ^ 1)) + op["nodes"][0][2:]
    elif forge == "salt":
        op["salts"][0] = _salt(999).hex()
    elif forge == "root-node":
        private["roots"]["circuit"]["nodes"].reverse()
    elif forge == "missing":
        private["rounds"][0]["tops"] = []
    else:
        op["salts"] = []
    with pytest.raises(ValueError, match="FIREWALL-REFUSED: tree top"):
        V.tops(public, private, "circuit/rep0")


@pytest.mark.parametrize("side", ["public", "private"])
@pytest.mark.parametrize("name", [None, "", "hm96-sha512/v0/node-row"])
def test_a_record_whose_tops_are_named_otherwise_is_refused_by_name(side, name):
    """A record without `rec_live.TOPS` as its tops' name, in ``public.json`` or ``private.json`` (no name, as
    ``rec_inner.py forge --what unsalted`` leaves it, an empty one, or v0's), is refused for that, though its tops open."""
    public, private, _, _ = _session()
    rec = {"public": public, "private": private}[side]
    if name is None:
        del rec["tops"]
    else:
        rec["tops"] = name
    with pytest.raises(ValueError, match=r"^FIREWALL-REFUSED: tree top: the record's tops are not salted \(rec_live\.TOPS\)$"):
        V.tops(public, private, "circuit/rep0")


@pytest.mark.parametrize("have, streams", [((), 2), ((0,), 2), ((0, 1, 2), 2), ((0, 1), 1)])
def test_a_session_without_each_reps_stream_and_proof_is_refused(have, streams, tmp_path, capsys):
    """A prover that stopped mid-session leaves rounds and no proof files; V* refuses it by name (it read no top and would
    otherwise accept zero queries), as it does a proof of an undeclared rep and a declared rep with no stream."""
    public, private, _, _ = _session()
    public |= {"hello": {"reps": 2}, "streams": {f"circuit/rep{r}": {} for r in range(streams)}}
    with pytest.raises(ValueError, match="FIREWALL-REFUSED: proofs"):
        V.check(public, private, {r: b"" for r in have})
    (tmp_path / "session").mkdir()
    (tmp_path / "public.json").write_text(json.dumps(public))
    (tmp_path / "private.json").write_text(json.dumps(private))
    for r in have:
        (tmp_path / "session" / f"circuit.rep{r}.bin").write_bytes(b"")
    assert V.main([str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["refused"].startswith("FIREWALL-REFUSED: proofs")


def test_query_rounds_follow_the_levels_in_order():
    s = "circuit/rep0"
    public = {"rounds": [_round(s, 0, n=7, level0=1), _round(s, 1, n=9), _round(s, 2, [["bb"]], ["00"], 5), _round(s, 3, n=7),
                         _round(s, 4, n=5), _round(s, 5, commits=["00"], n=3), _round(s, 6, n=3, level0=1)]}
    assert [r["k"] for r in V.query_rounds(public, s, [7, 5, 3])] == [3, 4, 5]
    with pytest.raises(ValueError):
        V.query_rounds(public, s, [7, 5, 3, 2])
