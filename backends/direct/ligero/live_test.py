"""The live B-Ligero verifier (lane live-verifier; ``live.py``) on the loopback:

* the wire frames and the (w, h, q, v) packing round-trip;
* a server with the Python core (``--python-verifier``: the Rust binary is not on every test machine; with
  ``LIGERO_VERIFY`` set to a built ``ligero-verify`` the Rust core is used and the verdict is marked independent)
  accepts an honest 2-VU fp8-ada session and refuses the five malicious provers of ``run_negatives`` (a tampered
  column, a swapped root, a swapped test message, prover-chosen coins, stale coins) -- each refusal logged with its
  reason and, for the three the transcript alone cannot show, caught by the LIVE check before any cryptography;
* the hook does not change the transcript: a proof made with a live coin source is byte-identical to one made locally
  with the same coins (non-ZK interactive, so the prover is deterministic given the coins);
* the timing model: ``adjust_timings`` takes exactly the network waits out of the phase clocks;
* four concurrent sessions are served.

Needs torch (``importorskip``): runs on the pod, skips on the laptop.  CPU only, small l.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path  # noqa: F401  (LIGERO_VERIFY lookup)

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("blake3")

from . import live
from . import protocol as pm
from .serialize import proof_bytes


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    out = tmp_path_factory.mktemp("live")
    lv = os.environ.get("LIGERO_VERIFY")
    core = live.RustVerifier(lv) if lv and Path(lv).is_file() else live.PythonVerifier("cpu")
    port = _free_port()
    srv = live.Server(f"127.0.0.1:{port}", out, core, jobs=2, log=lambda m: None)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    time.sleep(0.2)
    yield srv, f"tcp://127.0.0.1:{port}", out
    srv.stop.set()


def test_frames_and_array_packing_round_trip():
    a, b = socket.socketpair()
    n = live.send_frame(a, live.K_ROOT, b"\x01" * 40)
    assert n == 5 + 40
    kind, payload = live.recv_frame(b)
    assert kind == live.K_ROOT and payload == b"\x01" * 40
    w = np.arange(12, dtype=np.uint32).reshape(3, 4)
    h = np.arange(6, dtype=np.uint32).reshape(2, 3) + 7
    packed = live._pack_arrays([w, h, None, None])
    w2, h2, q2, v2 = live._unpack_arrays(packed)
    assert np.array_equal(w, w2) and np.array_equal(h, h2) and q2 is None and v2 is None
    assert live._arr_eq(w, w2) and not live._arr_eq(w, h2) and live._arr_eq(None, None) and not live._arr_eq(w, None)
    with pytest.raises(ValueError):
        live._unpack_arrays(packed + b"\x00")


def test_honest_session_accepted_and_negatives_refused(server):
    srv, url, out = server
    res = live.run_negatives(url, "fp8-ada", "cpu")
    assert res["all_ok"], res
    assert res["honest"]["accepted"]
    for case in ("tamper-column", "root-swap", "tests-swap", "prover-coins", "stale-coins"):
        assert not res[case]["accepted"], case
    # the three the transcript alone cannot show are caught by the live check, before any cryptography
    for case in ("root-swap", "tests-swap", "prover-coins", "stale-coins"):
        assert res[case]["live_rejections"], case
    assert "root committed before coin slot 1" in res["root-swap"]["reason"]
    assert "test message" in res["tests-swap"]["reason"]
    assert "not this session's step-0 coins" in res["prover-coins"]["reason"]
    assert "not this session's step-0 coins" in res["stale-coins"]["reason"]
    # the dump tree: system.bin, sub_00.{stmt,proof,coins}, verdict.json per session, and the index
    sd = out / res["honest"]["session_id"]
    for f in ("system.bin", "sub_00.stmt", "sub_00.proof", "sub_00.coins", "verdict.json", "session.json", "hello.json"):
        assert (sd / f).is_file(), f
    v = json.loads((sd / "verdict.json").read_text())
    assert v["accepted"] and v["mode"] == "interactive" and v["accepted_subbatches"] == 1
    assert v["independent"] == isinstance(srv.rust, live.RustVerifier)
    assert len((sd / "sub_00.coins").read_bytes()) == 4 * pm.COIN_BYTES
    index = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    assert {x["session_id"] for x in index} >= {r["session_id"] for r in res.values() if isinstance(r, dict)}
    # the refusals are logged
    log = (out / "serve.log").read_text()
    assert "REJECTED by the live check" in log and "VERDICT REJECT" in log and "VERDICT ACCEPT" in log


def test_live_hook_does_not_change_the_transcript(server):
    """Same coins -> same bytes: a proof made through the live coin source equals the local-coins proof (non-ZK
    interactive: the prover is deterministic given the coins)."""
    srv, url, out = server
    from .serialize import system_bytes
    # non-ZK at l=256 gives 2^-127.7, so the target (the prover's and the verifier's, in HELLO) is 100 bits here
    R, data = live._small_runner("fp8-ada", "cpu", zk=False, target=-100.0)
    hello = {"relation": "fp8-ada", "mode": pm.MODE_INTERACTIVE, "zk": False, "batch": 2 * R.steps, "target": -100.0,
             "target_bits": 100.0, "run_id": "bytes-test", "device": "cpu", "n_proofs": 1, "total_vus": 2}
    V = live.Verifier(url, hello, system_bytes(R.sys))
    sess = V.session(1, 1)
    coins = sess.coins(0)
    proof_live, pubs, lay = R.prove_vus(data, check=False, min_l=256, coins=coins)
    sess.submit(0, proof_live)
    verdict = sess.end()
    assert verdict["accepted"], verdict["reason"]
    waits = (coins.wait_commit, coins.wait_open1, coins.wait_open2)
    assert all(w > 0 for w in waits) and len(coins.rtt) == 2
    # replay: the verifier's coins as a local Coins object
    replay = coins.opened()
    proof_local, _, _ = R.prove_vus(data, check=False, min_l=256, coins=replay)
    assert proof_bytes(proof_local) == proof_bytes(proof_live)
    assert proof_bytes(proof_live) == (out / verdict["session_id"] / "sub_00.proof").read_bytes()
    # the timing model: adjust_timings removes exactly the waits
    t_before = dict(proof_live.timings)
    t = live.adjust_timings(proof_live, coins)
    assert t["net_wait"] == pytest.approx(sum(waits))
    assert t["total"] == pytest.approx(max(0.0, t_before["total"] - sum(waits)))
    assert live.adjust_timings(proof_local, replay) is proof_local.timings   # a local Coins: untouched


@pytest.mark.parametrize("window", [1, 2])
def test_pipelined_live_matches_sequential_and_is_accepted(server, window):
    """``--pipeline N --verifier`` (lane live-pipeline): several sub-batches proved at once, their coin exchanges
    non-blocking (the stage yields a Future the pipeline driver polls; the session sends each message as soon as 8c allows
    and a reader thread fulfils the futures).  ``window`` = the HELLO field: 1 is a verifier that commits one sub-batch at
    a time (the stream degrades to the sequential order), 2 commits two ahead so two openings are in flight.  The verifier
    accepts every sub-batch (its wire checks: only committed sub-batches, ROOT before TESTS, each once), and each pipelined
    proof is byte-identical to the sequential proof with the same (verifier-chosen) coins.  CPU, small l: runs in CI."""
    srv, url, out = server
    from .relations import RELATIONS
    from .relchain import RelationChainRunner, instances
    from .serialize import system_bytes

    n_proofs = 3
    rel = RELATIONS["fp8-ada"]
    # target -100 sizes t; the union over 3 sub-batches then reaches ~2^-99.8, so the verifier's target_bits is 96 here
    # (the real bench sizes t for target 128 over its 49 sub-batches -- baseline r...-073016-bf7d accepted 147/147)
    R = RelationChainRunner(rel, "cpu", -100.0, 2, n_proofs=n_proofs, zk=False, mode=pm.MODE_INTERACTIVE)
    data = instances(rel, 2 * n_proofs, procs=1)
    subs = [(2 * i, 2 * i + 2) for i in range(n_proofs)]
    batches = [data[lo:hi] for lo, hi in subs]
    hello = {"relation": "fp8-ada", "mode": pm.MODE_INTERACTIVE, "zk": False, "batch": 2 * R.steps, "target": -100.0,
             "target_bits": 96.0, "run_id": "pipelined", "device": "cpu", "n_proofs": n_proofs, "total_vus": 2 * n_proofs}
    sess = live.Verifier(url, hello, system_bytes(R.sys)).session(n_proofs, 1, {"window": window} if window > 1 else None)
    sess.start_pipeline(n_proofs)
    many, wall = R.prove_vus_many(batches, lambda i: sess.coins(i), depth=2, min_l=256)
    assert len(many) == n_proofs
    for si, (_coins, proof, _pubs, _lay) in enumerate(many):
        sess.submit(si, proof)
    verdict = sess.end()
    assert verdict["accepted"], verdict["reason"]
    assert verdict["accepted_subbatches"] == n_proofs
    assert not verdict.get("live_rejections")
    assert len(sess.rtts) == 2 * n_proofs                    # every opening's round trip was timed
    # the verifier's own timeline: with window 2, COMMIT 1 went out before ROOT 0 came in (two sub-batches in flight);
    # with window 1, COMMIT 1 only after OPEN2 0 (the sequential order)
    sj = json.loads((out / verdict["session_id"] / "session.json").read_text())
    sub = {s["sub"]: s for s in sj["subbatches"]}
    if window > 1:
        assert sub[1]["t_commit_sent"] <= sub[0]["t_root_recv"]
    else:
        assert sub[1]["t_commit_sent"] >= sub[0]["t_open2_sent"]
    for i in range(n_proofs):                                # 8c's order per sub-batch, whatever the interleaving
        assert sub[i]["t_commit_sent"] <= sub[i]["t_root_recv"] <= sub[i]["t_open1_sent"] <= sub[i]["t_tests_recv"] <= sub[i]["t_open2_sent"]
    # byte-identity: re-prove each sub-batch sequentially with the verifier's coins (replayed as a local Coins)
    for si, (coins, proof, _pubs, _lay) in enumerate(many):
        replay = coins.opened()
        proof_local, _, _ = R.prove_vus(batches[si], check=False, min_l=256, coins=replay)
        assert proof_bytes(proof_local) == proof_bytes(proof), f"sub-batch {si} bytes differ"
        assert proof_bytes(proof) == (out / verdict["session_id"] / f"sub_{si:02d}.proof").read_bytes()
        # the timing model: t.total (prover) excludes the network wait, t.total_live = t.total + net_wait >= t.total
        assert proof.timings.get("net_wait", 0.0) >= 0.0
        assert proof.timings["total"] >= 0.0


def test_four_concurrent_sessions(server):
    srv, url, out = server
    from .serialize import system_bytes
    R, data = live._small_runner("fp8-ada", "cpu", zk=False, target=-100.0)
    hello = {"relation": "fp8-ada", "mode": pm.MODE_INTERACTIVE, "zk": False, "batch": 2 * R.steps, "target": -100.0,
             "target_bits": 100.0, "run_id": "concurrent", "device": "cpu", "n_proofs": 1, "total_vus": 2}
    # four sessions open at once on the server (one prover here proves them in turn: torch on one CPU runner is not
    # something to run from four threads); the server's session threads and its verify pool overlap
    sessions = [live.Verifier(url, {**hello, "lane": f"c{i}"}, system_bytes(R.sys)).session(1, 1) for i in range(4)]
    coins = [s.coins(0) for s in sessions]
    proofs = [R.prove_vus(data, check=False, min_l=256, coins=c)[0] for c in coins]
    for s, pf in zip(sessions, proofs):
        s.submit(0, pf)
    with srv.lock:
        assert sum(1 for s in srv.sessions.values() if not s.done.is_set()) >= 4
    verdicts = [s.end() for s in sessions]
    assert all(v["accepted"] for v in verdicts), [v["reason"] for v in verdicts]
    assert len({v["session_id"] for v in verdicts}) == 4


def _small_shared():
    """A committed 2x4 tile (8 VUs) under row sharing, one pair per session (lane shared-live)."""
    from .relations import RELATIONS
    from .relchain import SharedHashedRunner, tile_instances
    rel = RELATIONS["fp8-ada"]
    data, xr, wc = tile_instances(rel, 2, 4)
    R = SharedHashedRunner(rel, "cpu", -100.0, 2, n_proofs=1, zk=False, mode=pm.MODE_INTERACTIVE)
    R.commit_vus(data, tile=(2, 4), x_rows=xr, w_cols=wc)
    from .serialize import system_bytes
    sg, sh = system_bytes(R.sys), system_bytes(R.sys_h)
    hello = {"relation": "fp8-ada", "mode": pm.MODE_INTERACTIVE, "zk": False, "target": -100.0, "target_bits": 100.0,
             "run_id": "pair", "device": "cpu", "n_proofs": 1, "total_vus": len(data), "pair": True, "system_h_bytes": len(sh)}
    return R, data, hello, sg + sh


@pytest.mark.parametrize("negative", [None, "own-coins-h"])
def test_shared_pair_every_coin_from_the_verifier(server, monkeypatch, negative):
    """G3: a row-sharing pair takes G's AND H's coins from the live verifier (H on wire index PAIR_H); a prover that
    proves H on its own coins (``LIVE_NEGATIVE=own-coins-h``) is refused by the live check."""
    srv, url, out = server
    if negative:
        monkeypatch.setenv("LIVE_NEGATIVE", negative)
    R, data, hello, system = _small_shared()
    sess = live.Verifier(url, hello, system).session(1, 1)
    coins = sess.coins(0)
    proof, pubs, lay = R.prove_vus(data, check=False, min_l=256, coins=coins)
    assert isinstance(coins.h, live.LiveCoins) and len(coins.rtt) == 2 and len(coins.h.rtt) == 2
    assert coins.fp is not None                               # F went out with G's TESTS
    sess.submit(0, proof)
    v = sess.end()
    sd = out / v["session_id"]
    for f in ("system.bin", "system_h.bin", "sub_00.stmt", "sub_00.proof", "sub_00.hproof", "sub_00.coins", "sub_00.hcoins"):
        assert (sd / f).is_file(), f
    hcoins = (sd / "sub_00.hcoins").read_bytes()
    if negative is None:
        assert v["accepted"], v["reason"]
        assert hcoins == coins.h.r1 + coins.h.s1 + coins.h.r2 + coins.h.s2
        ph = proof.h.coins
        assert ph.r1 + ph.s1 + ph.r2 + ph.s2 == hcoins        # H's transcript is on the verifier's coins
    else:
        assert not v["accepted"]
        assert "H: coins are not this session's step-0 coins" in v["reason"], v["reason"]
        assert v["live_rejections"]


def test_shared_pair_refuses_prover_sampled_coins_h(server):
    srv, url, out = server
    R, data, hello, system = _small_shared()
    sess = live.Verifier(url, hello, system).session(1, 1)
    with pytest.raises(ValueError, match="prover-sampled coins_h refused"):
        R.prove_vus(data, check=False, min_l=256, coins=sess.coins(0), coins_h=pm.Coins.sample())
    for s in (sess.ctrl, sess.data):
        s.close()


def test_fiat_shamir_session_is_refused(server):
    srv, url, out = server
    with pytest.raises(live.LiveError, match="interactive"):
        live.ProverSession(*live.parse_url(url), {"relation": "fp8-ada", "mode": "fiat-shamir", "n_proofs": 1}, b"")


@pytest.mark.parametrize("window", [1, 4])
def test_challenge_stream_many_rounds_windowed(server, window):
    """The many-round challenge stream (lane live-2, for Ligerito): 6 batches x 7 rounds with ``window`` in flight against
    the live verifier -- every batch completes, the verifier accepts (its wire checks: committed batch, STMT first, rounds
    in order, each once), each opened coin matches its commitment, and the challenge bytes are what the verifier's record
    of (r_k) reproduces; the local honest verifier has the same interface and the same derivation."""
    srv, url, out = server
    res = live.run_challenge_bench(url, 6, 7, window, msg_bytes=256, compute_ms=0.0, note=f"test-w{window}")
    assert res["verdict"]["accepted"], res["verdict"]
    assert res["verdict"]["complete_batches"] == 6 and res["round_ms"]["n"] == 42
    assert res["batch_latency_s"]["median"] > 0 and res["batches_per_s"] > 0
    rec = json.loads((out / res["session_id"] / "session.json").read_text())
    assert [b["batch"] for b in rec["batches"]] == list(range(6))
    assert all(len(b["rounds"]) == 7 and [r["k"] for r in b["rounds"]] == list(range(7)) for b in rec["batches"])
    # the derivation: prover and verifier agree on the challenge bytes given the coins
    host, port = live.parse_url(url)
    sess = live.ChallengeSession(host, port, {"note": "derivation"}, 1, 3, 1)
    st = sess.stream(0)
    st.begin(b"claim")
    got = [st.challenge(b"L%d" % k, b"m%d" % k, 16) for k in range(3)]
    v = sess.end()
    assert v["accepted"]
    ctx = live.batch_context(b"claim", st.commitments())
    assert got == [live.expand_challenge(ctx, k, b"L%d" % k, r, 16) for k, (r, s) in enumerate(st.openings())]
    assert all(pm.coin_commit(r, s) == c for (r, s), c in zip(st.openings(), st.commitments()))
    loc = live.LocalChallenges(3)
    loc.begin(b"claim")
    lc = [loc.challenge(b"L%d" % k, b"m%d" % k, 16) for k in range(3)]
    lctx = live.batch_context(b"claim", loc.commitments())
    assert lc == [live.expand_challenge(lctx, k, b"L%d" % k, r, 16) for k, (r, s) in enumerate(loc.openings())]


def test_challenge_stream_refuses_out_of_order_rounds(server):
    """MSG i 1 before MSG i 0, and a message of an uncommitted batch, are refused before any coin is opened."""
    srv, url, out = server
    host, port = live.parse_url(url)
    sess = live.ChallengeSession(host, port, {"note": "neg"}, 2, 3, 1)
    st = sess.stream(0)
    st.begin(b"claim")
    st.commitments()
    sess.provide_msg(0, 1, b"L", b"skip round 0")
    f = sess.round_future(0, 1)
    with pytest.raises(live.LiveError):
        f.result(timeout=10)
    sess2 = live.ChallengeSession(host, port, {"note": "neg2"}, 2, 3, 1)
    sess2.ccom_future(0).result(timeout=10)
    sess2._send(live.K_MSG, live.struct.pack("<III", 1, 0, 1) + b"L" + b"uncommitted batch")
    with pytest.raises(live.LiveError):
        sess2.round_future(1, 0).result(timeout=10)
