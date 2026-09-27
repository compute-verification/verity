"""The private-circuit prototype (verity_flock.recursion): gadgets, lookups, commitments, and V[B] on a recorded session.

The end-to-end tests need the recorded RoPE statement and session (art:d9837120, art:bd9f7efb) and the compression
circuit (art:04259cdd): `research data fetch <art> --to $VERITY_RECURSION_ARTIFACTS/<8 hex>`.
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pytest

from verity_flock.recursion import gadgets, gf, hm, memcheck, program as P

ART = os.environ.get("VERITY_RECURSION_ARTIFACTS")
NEEDS_ART = pytest.mark.skipif(not ART, reason="set VERITY_RECURSION_ARTIFACTS to a directory holding art:d9837120, "
                                               "art:bd9f7efb and art:04259cdd")


def test_pinned_constants():
    assert f"{gf.PINNED[0]:032x}" == "1cf7a0fe8922c83fd8a5ae31928b4da1"
    assert f"{gf.PINNED[6]:032x}" == "d3d3d3d3d3d3d3d3d3d3d3d3d3d3d3b9"
    assert f"{gf.PHI8[2]:032x}" == "0dcb364640a222fe6b8330483c2e9849"


def test_vectorized_field_product():
    rng = random.Random(1)
    a = [rng.getrandbits(128) for _ in range(64)]
    b = [rng.getrandbits(128) for _ in range(64)]
    assert gf.from_arr(gf.vmul(gf.to_arr(a), gf.to_arr(b))) == [gf.mul(x, y) for x, y in zip(a, b)]


@pytest.mark.parametrize("algo,ands", [("sha256", 22573), ("sha512", 57947)])
def test_sha_gadget(algo, ands):
    assert gadgets.sha_ands(algo) == ands
    msg = bytes(range(40))
    block = np.frombuffer(msg + gadgets.md_pad(algo, len(msg)), np.uint8)
    iv = np.frombuffer(gadgets.iv_bytes(algo), np.uint8)
    want = getattr(hashlib, algo)(msg).digest()
    assert bytes(gadgets.sha_native(algo, iv, block[None, :])) == want
    inp = np.concatenate([np.unpackbits(iv[None, :], axis=1, bitorder="little").T,
                          np.unpackbits(block[None, :], axis=1, bitorder="little").T])
    got = np.packbits(gadgets.run_sliced(gadgets.sha_circuit(algo), inp).T, axis=1, bitorder="little")
    assert bytes(got[0]) == want


def test_field_gadget():
    c = gadgets.gf_mul_circuit()
    assert c.ands == gadgets.GF_MUL_ANDS == 3 ** 7
    rng = random.Random(2)
    a = [rng.getrandbits(128) for _ in range(100)]
    b = [rng.getrandbits(128) for _ in range(100)]
    bits = lambda xs: np.unpackbits(gf.to_arr(xs).view(np.uint8), axis=1, bitorder="little").T
    assert np.array_equal(gadgets.run_sliced(c, np.concatenate([bits(a), bits(b)])), bits([gf.mul(x, y) for x, y in zip(a, b)]))


@pytest.mark.parametrize("n", [2, 8, 64, 1024])
def test_benes_routes_every_permutation(n):
    rng = np.random.default_rng(n)
    for _ in range(3):
        d = rng.permutation(n)
        y = P.benes_apply(np.arange(n)[:, None], memcheck.benes_route(d))
        assert np.array_equal(y[d, 0], np.arange(n))


def test_lookup_accepts_the_table_and_nothing_else():
    rng = np.random.default_rng(3)
    prog = P.Program("lookup")
    N, Q, wv, wi = 50, 120, 12, 6
    tab = prog.public("tab", "bit", (N, wv))
    qi = prog.witness("qi", "bit", (Q, wi))
    memcheck.lookup(prog, "L", tab, qi)
    tv = rng.integers(0, 2, (N, wv), dtype=np.uint8)
    q = rng.integers(0, N, Q)
    wit = memcheck.lookup_witness("L", N, q, tv)
    ins = {"tab": tv, "qi": ((q[:, None] >> np.arange(wi)) & 1).astype(np.uint8)}
    assert P.evaluate(prog, {**ins, **wit}).accepted
    vals = wit["L.vals"].copy()
    vals[7, 2] ^= 1
    assert not P.evaluate(prog, {**ins, **wit, "L.vals": vals}).accepted


def test_hm96_sha512_matches_core_pr93():
    """hm96-sha512/v1 vectors from PR #93 (6338450b, hm96/vectors_sha512.json)."""
    assert hashlib.sha512(hm.default_key()).hexdigest().startswith("afc5a60c85d8aa75cb4c7f726588d299")
    salt = bytes.fromhex(
        "eaf9ec3bac2b1b0001d0579fcc042b7a0af026e15d39d8516a06dfb731bcc712c643ad9a9db8ea9cf34b17326c6523ef8e73d5e0f8571fd1410dd523"
        "1417681ddc135dbf85030966f869447b21ef841355712ccbd128e31cf327deb1241ef2dd5e9ab606549176e045694864418dd6b7aa12c2f1a676951a"
        "d68f33d98ec7a70c0a8f3e5289248dbdde2fb936f89b7526da8c1ff611a8ca96dd3684b8282063f47cbe272b3b36aef453ce9c6b601a692d19ad85"
        "69e982e20d02d5e69c6d0a4367")
    x = bytes.fromhex("04a5e1fbe7b16087f9a001081ad4a566a344a053a5e59648a1f2a7bd9bb3a3a8c413c95db02ab68b2b779dbbe90973a5f4720b769f5650"
                      "05a0ae17f1d3cf3a13")
    cs = hm.commit_string(x, salt)
    assert cs[:64].hex().startswith("7527876b8cf382ae4488bfb89398c1bc") and cs[64:].hex().startswith("4778a6a14b1b3f553782e1b3")
    assert hm.tree_leaf(cs).hex().startswith("b6fb89eb80df32f7e5c679e2201ea3ac")


def test_commitment_gadget_is_the_native_commitment():
    prog = P.Program("hm")
    m = prog.witness("m", "byte", (2, 200))
    s = prog.witness("s", "byte", (2, hm.SALT_BYTES))
    out = hm.commit_circuit(prog, "x", m, s)
    rng = np.random.default_rng(4)
    M = rng.integers(0, 256, (2, 200), dtype=np.uint8)
    S = rng.integers(0, 256, (2, hm.SALT_BYTES), dtype=np.uint8)
    got = P.evaluate(prog, {"m": M, "s": S}, keep=True).vals[out.id]
    assert all(bytes(got[i]) == hm.commit("x", bytes(M[i]), bytes(S[i])) for i in range(2))


# ---------------------------------------------------------------------------------------------------------------------
# V[B] on the recorded RoPE session
# ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rope():
    from verity_flock.recursion import coins, session as S, statement as T, vb, witness as W
    art = Path(ART)
    circ = T.parse_circuit((art / "d9837120/circuit.txt").read_text())
    pub = T.load_public(art / "d9837120/pub-8.bin", circ)
    st = T.Statement(circ, pub, T.load_comp(art / "04259cdd/comp.rows"))
    fr = vb.Frame.of(st)
    prog = vb.build(fr)
    sess = S.load_session(art / "bd9f7efb/honest")
    tabs = coins.tables(fr, st, sess)

    def run(C=None, c_of=None, edit=None, session=None):
        s = session or sess
        t = tabs if session is None else coins.tables(fr, st, s)
        pubi, wit = W.convert(fr, st, C or st.C, pub, s, t, c_of=c_of, rng=np.random.default_rng(5))
        ins = {**t, **pubi, **wit}
        if edit:
            edit(ins)
        return P.evaluate(prog, ins)

    return dict(st=st, fr=fr, prog=prog, run=run, art=art)


@NEEDS_ART
def test_vb_accepts_the_honest_session(rope):
    r = rope["run"]()
    assert r.accepted, r.failed


@NEEDS_ART
def test_vb_rejects_rbreak(rope):
    from verity_flock.recursion import session as S
    r = rope["run"](session=S.load_session(rope["art"] / "bd9f7efb/r-break"))
    assert [w for w, _ in r.failed] == ["S18/R1: the reps' caps differ"]


@NEEDS_ART
def test_vb_rejects_a_wrong_circuit(rope):
    C = copy.deepcopy(rope["st"].C)
    r0, c0 = C.b[-1]
    C.b[-1] = (r0, (c0 + 1) % r0)
    whys = {w for w, _ in rope["run"](C=C, c_of=rope["st"].C).failed}
    assert "C does not open c" in whys
    assert {w for w, _ in rope["run"](C=C, c_of=C).failed} == {"lincheck: ConsistencyFailed (sumcheck-final)"}


@NEEDS_ART
def test_vb_rejects_phi_violations(rope):
    C = copy.deepcopy(rope["st"].C)
    r0, _ = C.b[5]
    C.b += [(r0, r0 + 1)] * 2                           # a later column, twice: the fold is unchanged
    assert {w for w, _ in rope["run"](C=C, c_of=C).failed} == {"Φ: an entry is not in circuit form"}
    C = copy.deepcopy(rope["st"].C)
    C.leaves_out[1][0] = C.leaves_out[0][0]
    assert "phi.lout: not a permutation" in {w for w, _ in rope["run"](C=C, c_of=C).failed}


@NEEDS_ART
def test_vb_rejects_a_message_that_does_not_open(rope):
    from verity_flock.recursion import vb
    off = vb.field_offsets(rope["fr"])[("zc_ab",)][0][0]

    def flip(ins):
        m = ins["m"].copy()
        m[off] ^= 1
        ins["m"] = m

    assert "R2: a message does not open its commitment" in {w for w, _ in rope["run"](edit=flip).failed}


@NEEDS_ART
def test_vb_depends_only_on_the_bounds(rope):
    """The swap test's structural half: other leaf maps under the same bounds give the same V[B]."""
    from verity_flock.recursion import statement as T, vb
    import json
    text = (rope["art"] / "d9837120/circuit.txt").read_text().split("\n")
    meta = json.loads(text[1][5:])
    meta["leaves_in"] = [list(reversed(x)) for x in meta["leaves_in"]]
    text[1] = "META " + json.dumps(meta)
    circ = T.parse_circuit("\n".join(text))
    st2 = T.Statement(circ, T.load_public(rope["art"] / "d9837120/pub-8.bin", circ), rope["st"].comp)
    assert st2.C.leaves_in != rope["st"].C.leaves_in
    assert vb.build(vb.Frame.of(st2)).digest() == rope["prog"].digest()
