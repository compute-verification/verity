"""The end-to-end Ligerito backend (``prove.py`` / ``proof.py``): the multi-claim open on pcs-fast's device path against the
reference path (same coins -> same proof bytes, materialised and streamed round-1 commit), the file formats round trip, and
honest proofs verify from bytes.  The torch tests are pod only (``pytest backends/direct/ligerito/prove_test.py -q``); the format
tests at the bottom are numpy only."""
from __future__ import annotations

import numpy as np
import pytest

from .proof import P, Key, PcsProof, Proof, Statement, read_key, read_proof, read_statement

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001  (the laptop's torch is an import-error stub)
    torch, _CUDA = None, False

gpu = pytest.mark.skipif(not _CUDA, reason="GPU (pod) only")


@pytest.fixture(scope="module")
def fp8():
    from backends.direct.ligero.relchain import instances

    from .prove import Prover
    pv = Prover("fp8-ada", "cuda", 256)
    return pv, instances(pv.rel, 12)


def _prove_bytes(pv, bt, fast, seed=5):
    from .prove import make_coins
    pv.pcs_fast = fast
    try:
        return pv.prove(bt, make_coins("local", seed)).proof
    finally:
        pv.pcs_fast = None


@gpu
def test_fast_open_matches_reference(fp8, monkeypatch):
    pv, data = fp8
    bt = pv.batch(data, "local")
    ref = _prove_bytes(pv, bt, False)
    fast = _prove_bytes(pv, bt, True)
    assert fast.to_bytes() == ref.to_bytes()
    pv.pcs_impl = "own"
    try:
        own = _prove_bytes(pv, bt, True)
    finally:
        pv.pcs_impl = "pcs-fast"
    assert own.to_bytes() == ref.to_bytes()
    monkeypatch.setenv("LIGERITO_STREAM", "1")
    streamed = _prove_bytes(pv, bt, True)
    assert streamed.to_bytes() == ref.to_bytes()


@gpu
def test_honest_verifies_from_bytes(fp8):
    from .prove import make_coins
    pv, data = fp8
    for kind in ("fiat-shamir", "local"):
        bt = pv.batch(data, kind)
        res = pv.prove(bt, make_coins(kind, 3))
        pf = read_proof(res.proof.to_bytes())
        st = read_statement(bt.stmt.to_bytes())
        ok, why, _ = pv.verify(st, pf, make_coins(kind, 3))
        assert ok, why
        ok, why, _ = pv.verify(st, pf, make_coins(kind, 4)) if kind == "local" else (False, "", None)
        assert not ok


@gpu
def test_stream_coins_replay(fp8):
    """Live coins from live-2's stream (played locally): the verifier replays the recorded slots, and every slot is the
    stream's ``expand_challenge`` of an opening that matches its commitment."""
    from backends.direct.ligero import live

    from .prove import StreamCoins, statement_for_stream
    from .transcript import LiveCoins
    pv, data = fp8
    bt = pv.batch(data, "live")
    pc = StreamCoins(live.LocalChallenges(256), statement_for_stream(bt, pv.key_sha256))
    res = pv.prove(bt, pc)
    ok, why, _ = pv.verify(read_statement(bt.stmt.to_bytes()), read_proof(res.proof.to_bytes()), LiveCoins(pc.slots_out))
    assert ok, why
    b = pc.binding()
    ctx = live.batch_context(bytes.fromhex(b["statement"]), [bytes.fromhex(c) for c in b["commitments"]])
    for k, ((r, s), label) in enumerate(zip(b["openings"], b["labels"])):
        c = bytes.fromhex(b["commitments"][k])
        assert live.pm.coin_commit(bytes.fromhex(r), bytes.fromhex(s)) == c
        assert live.expand_challenge(ctx, k, label.encode(), bytes.fromhex(r), 32) == pc.slots_out[k]
    slots = list(pc.slots_out)
    slots[-1] = bytes(32)
    assert not pv.verify(bt.stmt, res.proof, LiveCoins(slots))[0]


@gpu
def test_session_replay_reproduces_the_stream_messages(fp8):
    """``SessionReplayCoins`` (run.verify_session's side) recomputes, from the proof alone, the label and MSG digest the
    prover sent before every coin -- what the issuing verifier records as ``msg_sha256`` per round."""
    from backends.direct.ligero import live

    from .prove import SessionReplayCoins, StreamCoins, statement_for_stream

    class Recording(live.LocalChallenges):
        def __init__(self, *a):
            super().__init__(*a)
            self.sent = []

        def challenge(self, label, message, n):
            self.sent.append((label, message))
            return super().challenge(label, message, n)

    pv, data = fp8
    bt = pv.batch(data, "live")
    ch = Recording(256)
    pc = StreamCoins(ch, statement_for_stream(bt, pv.key_sha256))
    res = pv.prove(bt, pc)
    vc = SessionReplayCoins(pc.slots_out)
    ok, why, _ = pv.verify(read_statement(bt.stmt.to_bytes()), read_proof(res.proof.to_bytes()), vc)
    assert ok, why
    assert list(zip(vc.labels, vc.msgs)) == ch.sent and len(ch.sent) == pc.rounds


@gpu
def test_statement_words_off_the_chain_ends_are_rejected(fp8):
    """A claimed word at a column whose end constraint is masked binds nothing (a proof for that statement is otherwise
    valid): the canonical statement has 0 there."""
    import copy

    from .prove import make_coins
    pv, data = fp8
    bt = copy.copy(pv.batch(data, "fiat-shamir"))
    _, _, end = bt.lay.chain_layout(0).masks("cpu")
    bt.stmt = copy.deepcopy(bt.stmt)
    bt.stmt.y[0][int(np.flatnonzero(end.numpy() == 0)[0])] = 1
    bt.stmt_digest = bt.stmt.digest()
    res = pv.prove(bt, make_coins("fiat-shamir", 1), ypub_override=bt.stmt.y[0])
    ok, why, _ = pv.verify(bt.stmt, res.proof, make_coins("fiat-shamir", 1))
    assert not ok and "non-canonical" in why, why


@gpu
def test_key_is_a_function_of_the_relation(fp8):
    from .prove import Prover
    pv, _ = fp8
    assert Prover("fp8-ada", "cuda", 16384).key_bytes == pv.key_bytes
    assert read_key(pv.key_bytes).to_bytes() == pv.key_bytes


# ------------------------------------------------------------------------------------------------ numpy only

def test_statement_round_trip():
    rng = np.random.default_rng(0)
    l, K = 16, 4
    st = Statement("fp8-ada", bytes(32), bytes(range(32)), l, 2, K, [3, 8, 0], 1, 4,
                   [rng.integers(0, 256, (l, K)) for _ in range(2)], [rng.integers(0, 256, (l, K)) for _ in range(2)],
                   [rng.integers(0, 2 ** 22, (l,)) for _ in range(2)])
    b = st.to_bytes()
    st2 = read_statement(b)
    assert st2.to_bytes() == b and st2.n_vus == [3, 8, 0] and st2.S == 3
    with pytest.raises(ValueError):
        read_statement(b + b"\0")


def test_key_round_trip():
    coo = {s: (np.arange(5), np.arange(5) * 3, (np.arange(5) * 7 - 3) % P) for s in "abc"}
    k = Key("fp8-ada", bytes(32), 3577, 12, 12, 3936, [10, 11, 12], {"const": 3577, "y16": 3578}, coo)
    b = k.to_bytes()
    assert read_key(b).to_bytes() == b


def test_proof_round_trip_and_framing():
    rng = np.random.default_rng(1)
    kps, qs = [3, 2], [5, 4]
    n = 9
    pcs = PcsProof([bytes([i]) * 32 for i in range(2)], [[rng.integers(0, P, (3, 6)) for _ in range(kp)] for kp in kps],
                   [rng.integers(0, P, (5, 8)), rng.integers(0, P, (4, 24))], [b"\1" * 64, b"\2" * 96],
                   rng.integers(0, P, (1 << (n - sum(kps)), 6)))
    params = {"relation": "fp8-ada", "l": 16, "n": n, "kprime": kps, "queries": qs, "rate_log2": [1, 2],
              "splits": [[64, 1], [16, 1]], "order": "row-major", "coins": "local", "zk": False, "n_claims": 4}
    pf = Proof(params, b"LGSC0002" + bytes(10), pcs)
    b = pf.to_bytes()
    pf2 = read_proof(b)
    assert pf2.to_bytes() == b
    nb = pf.n_bytes()
    assert nb["total"] == len(b) and nb["header"] > 0
    with pytest.raises(ValueError):
        read_proof(b[:-1])
    # the framing is not absorbed: a re-encoding of the same values must not be a second valid proof (red-team-3 R3-2)
    fr = b'{"sib_len":[64,96],"final_len":16}'
    at = b.index(len(fr).to_bytes(4, "little") + fr)
    for alt in (b'{"sib_len": [64, 96], "final_len": 16}', b'{"final_len":16,"sib_len":[64,96]}',
                b'{"sib_len":[64,96],"final_len":16.0}', b'{"sib_len":[64,96],"final_len":16,"x":0}'):
        with pytest.raises(ValueError, match="framing"):
            read_proof(b[:at] + len(alt).to_bytes(4, "little") + alt + b[at + 4 + len(fr):])
