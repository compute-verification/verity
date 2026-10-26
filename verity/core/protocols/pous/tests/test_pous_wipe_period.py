"""`wipe_period` and `wipe_party`: one wipe in one process, on an injected clock, and the memory challenger's side and
the prover's over a socket pair, each verdict of `verity/wipe/v0` (the Lean's is `MemoryChallenger.WipeProtocol`,
difftested in `verity/core/devices/memory_challenger/lean/tests/test_wipe_protocol.py`): an honest segment certified at
⌈M/19⌉ + E, its bound vacuous when the egress is uncounted, a prover keeping more than f·M of foreign content and a late
read rejected, a session the prover left undecided. The leg runs whole against a challenger side on a host of its own
(`wipe_side.py`), and the bound's numeric step holds at the reads it is stated for."""

from __future__ import annotations

import json
import math
import socket
import sys
import threading
from pathlib import Path

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W

TEST_ID = mc.TEST_PREFIX + W.PROTOCOL
SMALL = sw.WipeParams(64, W.READS, 1, 1_000, (("gpu", 64 * sw.LABEL_BYTES),))
M = W.SEGMENT_LABELS * sw.LABEL_BITS


class Clock:
    """Each reading 1 ns after the last."""

    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1
        return self.now


def _run(params: sw.WipeParams = SMALL, secret: bytes = b"wipe period", **kw) -> W.Wipe:
    return W.run("n0", params, signer=ts.TestSigner(secret, 2), log=ts.session_log(), schemes=ts.SCHEMES,
                 protocol=TEST_ID, clock=kw.pop("clock", Clock()), coin_secret=b"coins", **kw)


def test_an_honest_segment_is_certified_at_f_m() -> None:
    w = _run(W.params_for(), egress_bits=3 * sw.EGRESS_UNIT_BITS - 5)
    assert w.verdict == "accept" and w.report == {"ended": sw.COMPLETE, "host": "n0", "reads": W.READS}
    assert w.answer["public"] == {"certified": True, "egress_mib": 3, "eps": f"1/{2 ** 128}",
                                  "foreign_bits": -(-M // 19) + 3 * sw.EGRESS_UNIT_BITS, "memory_bits": M}
    assert w.answer["profile"]["gaps"] == [] and w.answer["profile"]["tiers"] == [[W.TIER, 1 << 25]]
    evidence = json.loads(bytes.fromhex(w.case["evidence"]))
    assert evidence["signed"]["ended"] == sw.COMPLETE and w.case["params"]["server"] == b"n0".hex()


def test_an_uncounted_egress_bounds_nothing() -> None:
    w = _run(W.params_for())
    assert w.verdict == "accept"
    assert w.answer["public"] == {"certified": False, "egress_mib": sw.EGRESS_MAX_UNITS, "eps": "0", "foreign_bits": M,
                                  "memory_bits": M}
    assert w.answer["profile"]["gaps"] == [f"the egress count is {sw.EGRESS_MAX_UNITS} MiB, outside 0 to "
                                           f"{sw.EGRESS_MAX_UNITS - 1}: it bounds nothing"]


def test_a_prover_keeping_more_than_f_m_is_rejected() -> None:
    """A prover holding ⌈N/19⌉ + 1 labels' worth of prior content in place of labels misses a read with probability
    at least `1 − (18/19)^2152`, past `1 − 2^-128`; on these coins it misses one."""
    w = _run(W.params_for(), egress_bits=0, prover=W.keeping(W.SEGMENT_LABELS // 19 + 1))
    assert w.verdict == "reject" and w.report["ended"] == sw.WRONG, w.answer
    assert "wrong answers" in w.answer["reason"] and w.answer["public"] is None
    assert (18 / 19) ** W.READS < 2.0 ** -128


def test_a_late_read_is_rejected() -> None:
    """The sixth read's answer arrives a deadline after it was sent; the session ends there."""
    clock = Clock()

    class Late(sw.Prover):
        reads = 0

        def answer(self, indices):
            self.reads += 1
            clock.now += SMALL.limit_ns if self.reads == 6 else 0
            return super().answer(indices)

    w = _run(clock=clock, prover=Late, egress_bits=0)
    assert w.verdict == "reject" and w.report == {"ended": sw.LATE, "host": "n0", "reads": 6}
    assert "batch 5 is past its deadline of 1000 ns" in w.answer["reason"]


def test_a_session_the_prover_left_is_undecided() -> None:
    w = _run(stop_after=7)
    assert w.verdict == "undecided" and w.answer["reason"] == W.UNSIGNED and w.report["reads"] == 7
    assert json.loads(bytes.fromhex(w.case["evidence"])) == {"signed": None}


def _split(params: sw.WipeParams, prover: sw.Prover, signer, stop_after: int | None = None):
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["c"] = WP.serve(f, WP.request_of("n0", params), signer=signer, log=ts.session_log(), egress_bits=0,
                                clock=Clock())

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        answered = WP.answer(f, prover, stop_after)
    t.join()
    return got["c"], answered


@pytest.mark.parametrize("how", ["honest", "foreign", "left"])
def test_the_split_wipe(how: str) -> None:
    signer = ts.TestSigner(b"challenger host " + how.encode(), 1)
    prover = W.Keeping(SMALL, SMALL.labels // 19 + 1) if how == "foreign" else sw.Prover(SMALL)
    c, answered = _split(SMALL, prover, signer, stop_after=3 if how == "left" else None)
    a = W.answer(b"n0", c.commitment.session, (signer.public_key,), c.commitment, c.signed, ts.SCHEMES, TEST_ID)
    want = {"honest": "accept", "foreign": "reject", "left": "undecided"}[how]
    assert a["verdict"] == want, a
    if how == "honest":
        assert c.ended == sw.COMPLETE and answered == W.READS and a["public"]["egress_mib"] == 0
    elif how == "foreign":
        assert c.ended == sw.WRONG and answered == len(c.batches)
    else:
        assert answered == 3 and c.ended is None and c.signed is None


def test_the_party_follows_period_json(tmp_path) -> None:
    local = WP.Party.of({}, "vy-nebius-cpu-4", tmp_path)
    assert (local.host, local.address) == ("vy-nebius-cpu-4", None) and str(tmp_path) in local.command
    with pytest.raises(RuntimeError, match="vy-auditor-1 isn't wired for the wipe"):
        WP.Party.of({"challenger_host": "vy-auditor-1", "challenger": {"pous": {"command": ["x"], "address": "h:1"}}},
                    "vy-nebius-cpu-4", tmp_path)
    wired = WP.Party.of({"challenger_host": "vy-auditor-1", "challenger": {"wipe": {
        "command": ["vy-verifier-gate", "wipe"], "address": "127.0.0.1:7962"}}}, "vy-nebius-cpu-4", tmp_path)
    assert wired == WP.Party("vy-auditor-1", ("vy-verifier-gate", "wipe"), ("127.0.0.1", 7962))
    assert WP.placement("n0", "n0")[WP.PREMISE].startswith("unmet: ")


def test_the_leg_with_the_challenger_on_its_own_host(tmp_path, monkeypatch) -> None:
    """The leg whole: period.json read, the wipe of one segment against a challenger side that isn't this node's
    (`wipe_side.py`), its case and record written, the premise recorded as held. No network certifier counted the
    egress, so its bound is vacuous."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor"}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d: side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    assert answer["public"]["certified"] is False and answer["public"]["memory_bits"] == M
    assert answer["profile"]["placement"]["trusted-nebius-host"] == "holds: the memory challenger ran on auditor, not the prover's n0"
    case = json.loads((out / "wipe.case.json").read_text())
    assert sorted(case) == ["commitment", "evidence", "params"]
    record = json.loads((out / "wipe.json").read_text())
    assert record["lean"] is None and record["report"]["reads"] == W.READS and record["report"]["ended"] == sw.COMPLETE


@pytest.mark.parametrize("labels", [16, 1000, 1 << 19, 3 << 20, 1 << 30, 1_610_612_736, 1 << 50])
def test_the_bound_is_within_f_m_at_2152_reads(labels: int) -> None:
    """The streamed wipe's own bound (`retained_bits`, Pous.Guarantees.StreamedWipe before its PRF term) at k = 2152
    and δ = 2^-128, egress 0, is at most ⌈M/19⌉ from 16 labels up: the numeric step `EXPERIMENTAL` names."""
    m = labels * sw.LABEL_BITS
    q, _ = sw.retained_bits(labels, W.READS, m, 0, -128)
    assert q <= math.ceil(m / 19), (labels, q, m / 19)
