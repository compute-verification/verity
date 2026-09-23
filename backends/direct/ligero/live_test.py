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


def test_pipelined_live_matches_sequential_and_is_accepted(server):
    """``--pipeline N --verifier`` (lane live-pipeline): several sub-batches proved at once, their coin exchanges
    non-blocking (the stage yields a Future the pipeline driver polls; a session network thread runs the wire sequence
    in sub-batch order).  The verifier accepts every sub-batch (a strict-order wire check), and each pipelined proof is
    byte-identical to the sequential proof with the same (verifier-chosen) coins.  CPU, small l: runs in CI on the pod."""
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
    sess = live.Verifier(url, hello, system_bytes(R.sys)).session(n_proofs, 1)
    sess.start_pipeline(n_proofs)
    many, wall = R.prove_vus_many(batches, lambda i: sess.coins(i), depth=2, min_l=256)
    assert len(many) == n_proofs
    for si, (_coins, proof, _pubs, _lay) in enumerate(many):
        sess.submit(si, proof)
    verdict = sess.end()
    assert verdict["accepted"], verdict["reason"]
    assert verdict["accepted_subbatches"] == n_proofs
    assert not verdict.get("live_rejections")
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


def test_fiat_shamir_session_is_refused(server):
    srv, url, out = server
    with pytest.raises(live.LiveError, match="interactive"):
        live.ProverSession(*live.parse_url(url), {"relation": "fp8-ada", "mode": "fiat-shamir", "n_proofs": 1}, b"")
