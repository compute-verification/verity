"""PoUS's block run (`benchmarks/pous/block/block.py`) at a tiny size: a store in MEM-69's layout round-trips through
`make_fixture` and `load`, C* held in host memory or mmapped answers every block as setup encoded it, and over a socket
pair with the challenger's side in a thread (a test signer, the clock injected) the run's check accepts the honest prover
and refuses the one that drops 10.2% of the blocks for wrong answers, none late. A pinned key that isn't the signer's
refuses the honest session too: the verifier's list is never the report's."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("verity.core.protocols.pous")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verity/core/protocols/pous/tests"))

import pous_test_signer as ts  # noqa: E402
from verity.core.protocols.pous import challenger_party as CP  # noqa: E402
from verity.core.protocols.pous import memory_challenger as mc  # noqa: E402
from verity.core.protocols.pous import period  # noqa: E402
from verity.core.protocols.pous.schemes import p2 as P2  # noqa: E402

_spec = importlib.util.spec_from_file_location("pous_block", ROOT / "benchmarks/pous/block/block.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

COINS = hashlib.sha512(b"the block's live coins").digest()


class Clock:
    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1_000
        return self.now


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("block") / "store"
    q = P2.P2Params(period.W, period.C0, 128, 2)
    B.make_fixture(d, q, store=hashlib.shake_256(b"a model's weights").digest(20_000), coins=COINS)
    return d


def _session(point, root: bytes, respond, signer, d: Path) -> dict:
    """One session with the challenger's side in a thread, its report as `challenger_party.session` returns it."""
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["s"] = CP.serve(f, CP.request_of("n0", point, root), signer=signer, log=ts.session_log(d), coins=COINS,
                                domain=period.SESSION_DOMAIN, clock=Clock())

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        CP.answer(f, point.params.block_bytes, respond)
    t.join()
    sid, s = got["s"]
    return CP.report(sid, s, signer.public_key)


def test_the_layout_round_trips_and_every_holding_answers_c_star(store) -> None:
    point, published, doc = B.load(store)
    q = point.params
    assert (q.w, q.c0, q.blocks) == (period.W, period.C0, 256) and published.blocks == 256
    assert json.loads((store / "store.json").read_text())["manifest"] == q.manifest() == doc["manifest"]
    digests = B.table(store, q)
    blocks = [(store / "seg" / f"{i // 128:05d}.c").read_bytes()[q.block_bytes * (i % 128):q.block_bytes * (i % 128 + 1)]
              for i in range(q.blocks)]
    assert b"".join(mc.block_digest(b) for b in blocks) == digests
    for memory in ("host", "disk"):
        held = B.Held(store, q, memory)
        try:
            assert [held.respond(i) for i in range(q.blocks)] == blocks
            assert B.spot(held, digests) == []
            with pytest.raises(ValueError, match="outside C"):
                held.respond(q.blocks)
        finally:
            held.close()


def test_honest_accepts_and_the_drop_is_refused_for_wrong_answers(store, tmp_path) -> None:
    point, published, _ = B.load(store)
    q = point.params
    digests = B.table(store, q)
    held = B.Held(store, q, "host")
    signer = ts.TestSigner(b"challenger host", 4, tmp_path)
    try:
        honest = _session(point, published.root, B.prover(held.respond, False, q.block_bytes), signer, tmp_path)
        drop = _session(point, published.root, B.prover(held.respond, True, q.block_bytes), signer, tmp_path)
    finally:
        held.close()
    a, rec = B.judge(point, published, digests, honest, [signer.public_key], ts.SCHEMES)
    assert a["verdict"] == "accept" and rec["late"] == 0 and rec["trusted"] == "pinned", a
    assert a["public"] == period.certify(point) and a["profile"]["setup"] == "cited"
    d, drec = B.judge(point, published, digests, drop, [signer.public_key], ts.SCHEMES)
    assert d["verdict"] == "reject" and d["reason"].endswith(B.WRONG) and drec["late"] == 0, d
    assert drec["dropped_drawn"] > 0
    checks = B.checks_of({"honest": [rec | {"public": {"certified": True}}], B.DROP: [drec]})
    assert list(checks.values()) == [True, True]
    assert "VERDICT honest-0: accept" in B.verdict_line("honest-0", a, rec)
    other = ts.public_key(b"not the auditor's", 4)
    x, xrec = B.judge(point, published, digests, honest, [other], ts.SCHEMES)
    assert x["verdict"] == "reject" and xrec["trusted"] == "pinned", x
    r, rrec = B.judge(point, published, digests, honest, [], ts.SCHEMES)
    assert r["verdict"] == "accept" and rrec["trusted"] == "reported"


def test_the_drop_is_a_tenth_of_the_blocks() -> None:
    share = sum(B.dropped(i) for i in range(1 << 14)) / (1 << 14)
    assert 0.095 < share < 0.11
