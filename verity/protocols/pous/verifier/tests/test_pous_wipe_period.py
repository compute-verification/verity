"""`wipe_period` and `wipe_party`: one wipe in one process, on an injected clock, and the memory challenger's side and
the prover's over a socket pair, each verdict of `verity/wipe/v0` (the Lean's is `MemoryChallenger.WipeProtocol`,
difftested in `verity/protocols/pous/device/lean/tests/test_wipe_protocol.py`): an honest segment certified at
`StreamedWipeMeetsReads`' threshold at the challenger's `k` (11,709,069 + E bits at `k = 2152`, `r = 24`) with `E` the
egress its evidence gives, its bound vacuous when the evidence gives none, an egress out of range rejected, a prover
keeping one label past it and a late read rejected, a session the prover left undecided; the challenger chooses `k` (`wipe_party --reads`) and refuses a request that names it. The certificate
is checked at `k = 5231`, the H200's (`StreamedWipeMeets141GB5`), and against the streamed wipe's own bound. The leg runs
whole against a challenger side on a host of its own (`wipe_side.py`), its record naming what the bound doesn't cover
(the links leaving the node, the undeclared tiers, an uncounted egress, what a joint store's disks leave outside the
boundary), the reads and the session's window. The verifier trusts the one key its pin names (`wipe.trusted_sha512`): a
session signed under another is rejected, and a challenger on its own host with no key pinned is refused. The record
says `trusted-host` holds only for a session signed under the key `wipe_party.AUDITOR_KEYS` pins for the challenger's
host: a run's own pin, or a rehearsal's host name, gives a rehearsal with no bound."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

from pous_fakes import Clock
import pous_test_signer as ts
import wipe_side
from verity.configuration.deploy.stream import Side
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import streamed_wipe as sw
from verity.protocols.pous.verifier import wipe_party as WP
from verity.protocols.pous.verifier import wipe_period as W
from verity.protocols.pous.verifier import wipe_store as WS

TEST_ID = mc.TEST_PREFIX + W.PROTOCOL
SMALL = sw.WipeParams(64, W.READS, 1, 1_000, (("gpu", 64 * sw.LABEL_BYTES),))
M = W.SEGMENT_LABELS * sw.LABEL_BITS


def _run(params: sw.WipeParams = SMALL, secret: bytes = b"wipe period", **kw) -> W.Wipe:
    return W.run("n0", params, signer=ts.TestSigner(secret, 2), log=ts.session_log(), schemes=ts.SCHEMES,
                 protocol=TEST_ID, clock=kw.pop("clock", Clock(10 ** 9, 1)), coin_secret=b"coins", **kw)


SEGMENT_2152 = 11_709_069
"""`512·(⌈2^19/24⌉ − 1) + 2^19 + 11 + 130`: one segment's threshold at `k = 2152`, `r = 24`."""


def test_an_honest_segment_is_certified_at_its_reads() -> None:
    w = _run(W.params_for(), egress_mib=sw.egress_units(3 * sw.EGRESS_UNIT_BITS - 5))
    key, = w.case["params"]["trusted"]
    assert w.verdict == "accept" and w.report == {"ended": sw.COMPLETE, "host": "n0", "reads": W.READS,
                                                  "key_sha512": hashlib.sha512(bytes.fromhex(key)).hexdigest(),
                                                  "key_pinned": False}
    assert w.answer["public"] == {"certified": True, "egress_mib": 3, "eps": f"1/{2 ** 128}",
                                  "foreign_bits": SEGMENT_2152 + 3 * sw.EGRESS_UNIT_BITS, "k": W.READS,
                                  "memory_bits": M, "r": 24}
    assert w.answer["profile"]["gaps"] == [] and w.answer["profile"]["tiers"] == [[W.TIER, 1 << 25]]
    assert w.answer["profile"]["certificate"] == {"eps": f"1/{2 ** 128}", "max_reads": W.MAX_READS, "outside_bits": 0,
                                                  "r": 24, "reads": W.READS}
    evidence = json.loads(bytes.fromhex(w.case["evidence"]))
    assert evidence["signed"]["ended"] == sw.COMPLETE and w.case["params"]["server"] == b"n0".hex()
    assert sorted(evidence) == ["egress_mib", "signed"] and evidence["egress_mib"] == 3
    assert "egress_bits" not in evidence["signed"] and w.signed.to_json() == evidence["signed"]


def test_an_uncounted_egress_bounds_nothing() -> None:
    """With no egress in the evidence the check takes `EGRESS_MAX_UNITS`: the wipe is accepted and not certified, as
    it is when the evidence says `EGRESS_MAX_UNITS` itself; one MiB less is certified."""
    vacuous = {"certified": False, "egress_mib": sw.EGRESS_MAX_UNITS, "eps": "0", "foreign_bits": M, "k": W.READS,
               "memory_bits": M, "r": 24}
    gaps = [f"the egress count is {sw.EGRESS_MAX_UNITS} MiB, outside 0 to {sw.EGRESS_MAX_UNITS - 1}: it bounds nothing"]
    for egress in (None, sw.EGRESS_MAX_UNITS):
        w = _run(W.params_for(), egress_mib=egress)
        assert w.verdict == "accept" and w.answer["public"] == vacuous and w.answer["profile"]["gaps"] == gaps
        assert json.loads(bytes.fromhex(w.case["evidence"]))["egress_mib"] == egress
    assert W.egress_of(None) == sw.EGRESS_MAX_UNITS and W.egress_of(0) == 0
    last = _run(W.params_for(), egress_mib=sw.EGRESS_MAX_UNITS - 1)
    assert last.verdict == "accept" and last.answer["public"]["certified"] and last.answer["profile"]["gaps"] == []


@pytest.mark.parametrize("egress", [-1, -(1 << 40), sw.EGRESS_MAX_UNITS + 1, 10 ** 15])
def test_an_egress_out_of_range_is_refused(egress: int) -> None:
    """An egress outside 0 to `EGRESS_MAX_UNITS` rejects the wipe, signed or not, and publishes nothing."""
    for stop_after in (None, 7):
        w = _run(egress_mib=egress, stop_after=stop_after)
        assert (w.signed is None) == (stop_after is not None)
        assert w.verdict == "reject" and w.answer["public"] is None and w.answer["profile"] is None
        assert w.answer["reason"] == f"the evidence's egress is {egress} MiB, outside 0 to {sw.EGRESS_MAX_UNITS}"
        assert json.loads(bytes.fromhex(w.case["evidence"]))["egress_mib"] == egress


def test_a_prover_keeping_one_label_past_the_threshold_is_rejected() -> None:
    """A prover holding `f` labels' worth of prior content in place of labels, one label past the published threshold,
    misses a read with probability `1 − (1 − f/N)^2152`, past `1 − 2^-128`; on these coins it misses one."""
    p = W.params_for()
    f = W.foreign_labels("fM", p)
    assert f == SEGMENT_2152 // sw.LABEL_BITS + 1 and (W.SEGMENT_LABELS - f) ** W.READS << 128 < W.SEGMENT_LABELS ** W.READS
    w = _run(p, egress_mib=0, prover=W.keeping(f))
    assert w.verdict == "reject" and w.report["ended"] == sw.WRONG, w.answer
    assert "wrong answers" in w.answer["reason"] and w.answer["public"] is None


def test_the_certificate_at_the_challengers_reads() -> None:
    """`ratio(k)` is the largest `r` with `(r − 1)^k · 2^129 ≤ r^k`, and `certify` publishes the threshold at it: one
    segment at `k = 2152` and at `k = 5231`, and `StreamedWipeMeets141GB5`'s H200 (its fill and 2.3 GB of driver reserve
    declared, `R = 18.4·10^9` bits) at its `39,721,768,854` bits; past `MAX_READS` the certificate is vacuous."""
    assert [W.ratio(k) for k in (1, 90, 91, 2151, 2152, 5230, 5231, 74_000)] == [1, 1, 1, 24, 24, 58, 59, 828]
    for k in (1, 91, 400, 2152, 5231, 10_000, W.MAX_READS):
        r = W.ratio(k)
        assert (r - 1) ** k << 129 <= r ** k and r ** k << 129 > (r + 1) ** k, k
    assert (W.certify(W.params_for(), 0)["foreign_bits"], W.certify(W.params_for(k=5231), 0)["foreign_bits"]) == (
        SEGMENT_2152, 5_074_062)
    h200 = sw.WipeParams(2_203_125_000, 5231, 1, W.LIMIT_NS, (("gpu-hbm", 143_300_000_000),))
    assert h200.outside_bits == 18_400_000_000
    assert W.certify(h200, 2) == {"certified": True, "egress_mib": 2, "eps": f"1/{2 ** 128}",
                                  "foreign_bits": 39_721_768_854 + 2 * sw.EGRESS_UNIT_BITS, "k": 5231,
                                  "memory_bits": 1_146_400_000_000, "r": 59}
    past = W.params_for(k=W.MAX_READS + 1)
    assert W.certify(past, 0) == {"certified": False, "egress_mib": 0, "eps": "0", "foreign_bits": M,
                                  "k": W.MAX_READS + 1, "memory_bits": M, "r": 0}
    assert W.gaps(past, 0) == [f"k = {W.MAX_READS + 1}: the certificate is computed for at most {W.MAX_READS} reads"]


def test_the_challenger_chooses_the_reads(tmp_path) -> None:
    """The request carries the fill, the deadline and the tiers; `k` is the challenger's (`--reads`), and a request
    naming it is refused."""
    request = WP.request_of("n0", W.params_for(k=5231))
    assert set(request) == WP.REQUEST_KEYS and WP.params_of(request).k == W.READS
    assert WP.params_of(request, 5231) == W.params_for(k=5231)
    with pytest.raises(ValueError, match="the challenger chooses the reads"):
        WP.params_of(request | {"k": 9})
    assert W.config({"wipe": {"reads": 5231}})["reads"] == 5231
    for bad in (0, "5231", 1 << 32):
        with pytest.raises(ValueError, match="reads null or 1 to 2"):
            W.config({"wipe": {"reads": bad}})


def test_a_late_read_is_rejected() -> None:
    """The sixth read's answer arrives a deadline after it was sent; the session ends there."""
    clock = Clock(10 ** 9, 1)

    class Late(sw.Prover):
        reads = 0

        def answer(self, indices):
            self.reads += 1
            clock.now += SMALL.limit_ns if self.reads == 6 else 0
            return super().answer(indices)

    w = _run(clock=clock, prover=Late, egress_mib=0)
    assert w.verdict == "reject" and w.report == {"ended": sw.LATE, "host": "n0", "reads": 6,
                                                  "key_sha512": w.report["key_sha512"], "key_pinned": False}
    assert "batch 5 is past its deadline of 1000 ns" in w.answer["reason"]


def test_a_session_the_prover_left_is_undecided() -> None:
    w = _run(stop_after=7)
    assert w.verdict == "undecided" and w.answer["reason"] == W.UNSIGNED and w.report["reads"] == 7
    assert json.loads(bytes.fromhex(w.case["evidence"])) == {"egress_mib": None, "signed": None}


def _split(params: sw.WipeParams, prover: sw.Prover, signer, stop_after: int | None = None):
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["c"] = WP.serve(f, WP.request_of("n0", params), signer=signer, log=ts.session_log(),
                                clock=Clock(10 ** 9, 1))

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        answered = WP.answer(f, prover, stop_after)
    t.join()
    return got["c"], answered


@pytest.mark.parametrize("how", ["honest", "foreign", "left"])
def test_the_split_wipe(how: str) -> None:
    signer = ts.TestSigner(b"challenger host " + how.encode(), 1)
    prover = W.Keeping(SMALL, W.foreign_labels("fM", SMALL)) if how == "foreign" else sw.Prover(SMALL)
    c, answered = _split(SMALL, prover, signer, stop_after=3 if how == "left" else None)
    a = W.answer(b"n0", c.commitment.session, (signer.public_key,), c.commitment, c.signed, ts.SCHEMES, TEST_ID,
                 egress_mib=0)
    want = {"honest": "accept", "foreign": "reject", "left": "undecided"}[how]
    assert a["verdict"] == want, a
    if how == "honest":
        assert c.ended == sw.COMPLETE and answered == W.READS and a["public"]["egress_mib"] == 0
    elif how == "foreign":
        assert c.ended == sw.WRONG and answered == len(c.batches)
    else:
        assert answered == 3 and c.ended is None and c.signed is None


def test_the_party_follows_period_json(tmp_path) -> None:
    local = WP.side_of({}, "cpu-node", tmp_path)
    assert (local.host, local.address) == ("cpu-node", None) and str(tmp_path) in local.command
    with pytest.raises(RuntimeError, match="auditor-host isn't wired for the wipe"):
        WP.side_of({"challenger_host": "auditor-host", "challenger": {"pous": {"command": ["x"], "address": "h:1"}}},
                   "cpu-node", tmp_path)
    wired = WP.side_of({"challenger_host": "auditor-host", "challenger": {"wipe": {
        "command": ["gate", "wipe"], "address": "127.0.0.1:7962"}}}, "cpu-node", tmp_path,
        ("--streamer", "build"))
    assert wired == Side("auditor-host", ("gate", "wipe"), ("127.0.0.1", 7962))
    native = WP.side_of({}, "n0", tmp_path, ("--streamer", "build", "--threads", "8"))
    assert native.command[-4:] == ("--streamer", "build", "--threads", "8")
    assert WP.placement("n0", "n0", OTHER)["kind"] == "prover-node" and WP.placement("n0", "a", OTHER)["kind"] == "own-host"
    simulated = WP.placement("n0", "n0", OTHER, simulated=True)
    assert simulated["kind"] == "simulated" and simulated[WP.PREMISE] == (
        f"{REHEARSAL}; the memory challenger ran as a process of its own on the developer's node n0, standing in for a "
        "host the developer doesn't control")
    with pytest.raises(ValueError, match="simulated placement runs the memory challenger on the developer's node"):
        WP.placement("n0", "a", OTHER, simulated=True)


AUDITOR = "vy-auditor-2"
REHEARSAL = "rehearsal: key not the auditor's pinned key, no bound"
OTHER = hashlib.sha512(b"a key that is no auditor's").hexdigest()


def test_the_auditor_keys_are_sha512s_in_lowercase_hex() -> None:
    assert WP.AUDITOR_KEYS and all(re.fullmatch("[0-9a-f]{128}", k) for k in WP.AUDITOR_KEYS.values())
    assert WP.REHEARSAL == REHEARSAL


def test_a_rehearsal_on_the_prover_node_names_another_host_and_has_no_bound() -> None:
    """Every party on the prover's node, the challenger's host named `on-node-rehearsal` and its key the run's own: the
    names say `own-host`, and the premise is a rehearsal's all the same, with the key custody left out of the bound."""
    where = WP.placement("node-a", "on-node-rehearsal", OTHER)
    assert where["kind"] == "own-host" and not WP.holds(where)
    assert where[WP.PREMISE] == f"{REHEARSAL}; the memory challenger ran on on-node-rehearsal, not the developer's node-a"
    assert W.uncertified(where, None) == [f"key custody: {where[WP.PREMISE]}"]
    assert not WP.holds(WP.placement("node-a", "on-node-rehearsal", None))


def test_the_auditor_host_under_its_pinned_key_holds() -> None:
    where = WP.placement("vy-dc-h200", AUDITOR, WP.AUDITOR_KEYS[AUDITOR])
    assert where["kind"] == "own-host" and WP.holds(where) and W.uncertified(where, None) == []
    assert where[WP.PREMISE] == ("holds: the session is signed under vy-auditor-2's pinned key; the memory challenger ran "
                                 "on vy-auditor-2, not the developer's vy-dc-h200")


def test_the_auditor_hosts_name_under_another_key_is_a_rehearsal() -> None:
    """The name `vy-auditor-2` is a label: under any key but its pinned one, the premise is a rehearsal's."""
    for key in (OTHER, WP.AUDITOR_KEYS[AUDITOR].upper(), None):
        where = WP.placement("vy-dc-h200", AUDITOR, key)
        assert where["kind"] == "own-host" and not WP.holds(where) and where[WP.PREMISE].startswith(f"{REHEARSAL}; ")
    assert not WP.holds(WP.placement("n0", "on-node-rehearsal", WP.AUDITOR_KEYS[AUDITOR]))


CONTROL = {"name": "control", "to": "outside", "via": "eth0 192.0.2.13: ssh"}


def _side(tmp_path, monkeypatch, wipe: dict) -> Path:
    """A period whose challenger is `wipe_side.py` on a host of its own, its key the pinned one unless `wipe` says."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor",
                                                    "wipe": {"trusted_sha512": wipe_side.key_sha512()} | wipe}))
    device = tmp_path / "device"
    device.mkdir()
    side = Side("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP, "side_of", lambda doc, node, d, extra=(): side)
    return period


def test_the_leg_with_the_challenger_on_its_own_host(tmp_path, monkeypatch) -> None:
    """The leg whole: period.json read, the wipe of one segment against a challenger side that isn't this node's
    (`wipe_side.py`, its key here the one `AUDITOR_KEYS` pins for its host), its case and record written, the premise
    recorded as held. No network certifier counted the
    egress and period.json names no count, so the check is given none and its bound is vacuous. The record names what the bound doesn't cover: the egress, the host's DRAM and
    the two links leaving the node (period.json's control link, and the challenger's as the prover connected), and
    carries the session's window on the challenger's clock."""
    monkeypatch.setitem(WP.AUDITOR_KEYS, "auditor", wipe_side.key_sha512())
    period = _side(tmp_path, monkeypatch, {"links": [CONTROL]})
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    assert answer["public"]["certified"] is False and answer["public"]["memory_bits"] == M
    assert answer["profile"]["placement"]["trusted-host"] == ("holds: the session is signed under auditor's pinned key; "
                                                              "the memory challenger ran on auditor, not the developer's n0")
    case = json.loads((out / "wipe.case.json").read_text())
    assert sorted(case) == ["commitment", "evidence", "params"]
    assert json.loads(bytes.fromhex(case["evidence"]))["egress_mib"] is None
    record = json.loads((out / "wipe.json").read_text())
    assert record["lean"] is None and record["report"]["reads"] == W.READS and record["report"]["ended"] == sw.COMPLETE
    assert record["prover"] is None and record["session"]["reads"]["n"] == W.READS
    assert record["wipe"] == W.CONFIG | {"foreign_labels": 0, "k": W.READS, "links": [CONTROL], "memory_bits": M,
                                         "outside_bits": 0, "segments": W.SEGMENTS, "tier": W.TIER,
                                         "trusted_sha512": wipe_side.key_sha512()}
    control, challenger = record["placement"]["links"]
    assert control == WP.link("control", "n0", "outside", CONTROL["via"])
    assert challenger["name"] == "challenger" and challenger["ends"] == ["n0", "auditor"]
    assert challenger["via"] == f"{record['report']['connection']['local']} to {record['report']['connection']['peer']}"
    assert challenger["via"].startswith("127.0.0.1:") and challenger["counted_by"] is None
    assert answer["profile"]["placement"] == record["placement"]
    dram = [u for u in record["placement"]["undeclared_tiers"] if u["tier"] == "host-dram"]
    assert len(dram) == 1 and dram[0]["what"] == "the host's DRAM outside the declared tier" and dram[0]["bytes"] > 0
    window = {"t0": "2026-10-27T08:00:00.123456Z", "t1": "2026-10-27T08:03:03.000001Z"}
    assert {k: record["report"]["window"][k] for k in window} == window
    assert record["window"]["challenger"] == record["report"]["window"] and set(record["window"]["prover"]) == {
        "clock", "t0", "t1"}
    gaps = record["uncertified"]
    assert not any(g.startswith("key custody") for g in gaps) and gaps[0].startswith("egress: uncounted, the check was")
    assert any(g.startswith("tier host-dram: the host's DRAM outside the declared tier (") for g in gaps)
    assert [g.split(":")[0] for g in gaps if g.startswith("link ")] == ["link control", "link challenger"]
    s = W.summary(record)
    assert (s["t0"], s["t1"]) == (window["t0"], window["t1"]) and s["uncertified"] == gaps and s["k"] == W.READS
    key, = case["params"]["trusted"]
    assert s["decided_by"] == record["decided_by"] == {"program": "python", "host": socket.gethostname()}
    assert s["key_sha512"] == record["report"]["key_sha512"] == hashlib.sha512(bytes.fromhex(key)).hexdigest()
    assert s["key_sha512"] == wipe_side.key_sha512() and s["key_pinned"] is record["report"]["key_pinned"] is True


def test_who_decided_is_the_leans_file_name_or_python_on_this_host(tmp_path) -> None:
    """`period.decided_by`, what a piece's summary says of its verdict: the Lean by its executable's name, wherever it
    was built, and `python` when none answered, each on the host that ran it."""
    host = socket.gethostname()
    assert W.decided_by(tmp_path / "scratch" / ".lake" / "build" / "bin" / "wipe-protocol") == {
        "program": "wipe-protocol", "host": host}
    assert W.decided_by("pous-run-protocol") == {"program": "pous-run-protocol", "host": host}
    assert W.decided_by(None) == {"program": "python", "host": host}


def test_the_leg_refuses_a_session_at_other_reads(tmp_path, monkeypatch) -> None:
    """`wipe.reads` starts a challenger on this node with `--reads`; one on its own host chooses its own (here
    `wipe_side.py`'s default), and the leg refuses a session at another count."""
    period = _side(tmp_path, monkeypatch, {"reads": 5231})
    extras, of = [], WP.side_of
    monkeypatch.setattr(WP, "side_of", lambda doc, node, d, extra=(): extras.append(extra) or of(doc, node, d, extra))
    with pytest.raises(RuntimeError, match="ran 2152 reads, not period.json's wipe.reads 5231"):
        W.leg(period, "n0", tmp_path / "out")
    assert extras == [("--reads", "5231")]


def test_the_periods_egress_goes_to_the_check_and_never_to_the_challenger(tmp_path, monkeypatch) -> None:
    """period.json's `wipe.egress_bits` is evidence: the check is given it in whole MiB, wherever the
    challenger runs, and the challenger's command names no egress. The check publishes `certify` at it."""
    period = _side(tmp_path, monkeypatch, {"egress_bits": 3 * sw.EGRESS_UNIT_BITS - 5})
    extras, of = [], WP.side_of
    monkeypatch.setattr(WP, "side_of", lambda doc, node, d, extra=(): extras.append(extra) or of(doc, node, d, extra))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert extras == [()]
    assert answer["verdict"] == "accept" and answer["public"] == W.certify(W.params_for(), 3) and answer["public"]["certified"]
    evidence = json.loads(bytes.fromhex(json.loads((out / "wipe.case.json").read_text())["evidence"]))
    assert evidence["egress_mib"] == 3 and "egress_bits" not in evidence["signed"]
    assert not any(g.startswith("egress") for g in json.loads((out / "wipe.json").read_text())["uncertified"])


UNTRUSTED = "the session is signed by a key that is not a trusted memory challenger's"


def test_the_verifier_trusts_the_pinned_key_alone() -> None:
    """`run`'s `trusted_sha512`: under the digest of the key the session is signed under the wipe is accepted and the
    key reported pinned; under another digest the same session is rejected, its case naming no trusted key; with no
    pin the key the session came with is trusted and reported unpinned."""
    unpinned = _run()
    own = unpinned.report["key_sha512"]
    pinned, other = _run(trusted_sha512=own), _run(trusted_sha512=hashlib.sha512(b"another key").hexdigest())
    assert (unpinned.verdict, unpinned.report["key_pinned"]) == ("accept", False)
    assert (pinned.verdict, pinned.report["key_pinned"]) == ("accept", True)
    key, = pinned.case["params"]["trusted"]
    assert unpinned.case["params"]["trusted"] == [key] and hashlib.sha512(bytes.fromhex(key)).hexdigest() == own
    assert (other.verdict, other.answer["reason"], other.answer["public"]) == ("reject", UNTRUSTED, None)
    assert other.report == pinned.report | {"key_pinned": False} and other.case["params"]["trusted"] == []


def test_a_session_signed_under_another_key_than_the_pinned_one_is_rejected(tmp_path, monkeypatch, capsys) -> None:
    """`main` on a period whose challenger, on its own host, signs under another key than the one `wipe.trusted_sha512`
    pins: the session, honest in every read, is rejected, by a case that trusts no key, and the record and the printed
    summary say whose key it was and that it wasn't the pinned one."""
    other = hashlib.sha512(b"another key").hexdigest()
    period = _side(tmp_path, monkeypatch, {"trusted_sha512": other})
    out = tmp_path / "out"
    argv = ["--period", str(period), "--out", str(out), "--node", "n0"]
    assert W.main(argv) == 1
    s = json.loads(capsys.readouterr().out)
    assert (s["verdict"], s["reason"], s["public"]) == ("reject", UNTRUSTED, None)
    assert (s["key_pinned"], s["key_sha512"]) == (False, wipe_side.key_sha512())
    assert json.loads((out / "wipe.case.json").read_text())["params"]["trusted"] == []
    record = json.loads((out / "wipe.json").read_text())
    assert record["report"]["ended"] == sw.COMPLETE and record["wipe"]["trusted_sha512"] == other
    assert record["answer"]["verdict"] == "reject" and W.summary(record) == s


def test_a_challenger_on_its_own_host_with_no_key_pinned_is_refused(tmp_path, monkeypatch) -> None:
    """With no pin the verifier trusts the key the session came with, which only a challenger whose key the run made
    may be given: one on its own host is refused before its session starts."""
    period = _side(tmp_path, monkeypatch, {"trusted_sha512": None})
    started = []
    monkeypatch.setattr(W, "run", lambda *a, **kw: started.append(a))
    with pytest.raises(ValueError, match="wipe.trusted_sha512 is unset with the memory challenger on auditor"):
        W.main(["--period", str(period), "--out", str(tmp_path / "out"), "--node", "n0"])
    assert not started and not (tmp_path / "out").exists()


def test_a_run_that_pins_its_own_key_is_accepted_and_still_a_rehearsal(tmp_path, monkeypatch) -> None:
    """The run pins its own challenger's key (`wipe.trusted_sha512`, as `--trusted-sha512` and `WIPE_KEY_SHA512` do), so
    the verifier accepts its session; the challenger's host is not one `AUDITOR_KEYS` pins that key for, so the record
    says rehearsal, no bound, and leaves the key custody out of the bound."""
    period = _side(tmp_path, monkeypatch, {})
    answer = W.leg(period, "n0", tmp_path / "out")
    assert answer["verdict"] == "accept", answer
    record = json.loads((tmp_path / "out" / "wipe.json").read_text())
    assert record["report"]["key_pinned"] is True and record["report"]["key_sha512"] == wipe_side.key_sha512()
    where = record["placement"]
    assert answer["profile"]["placement"] == where and where["kind"] == "own-host" and not WP.holds(where)
    assert where[WP.PREMISE] == f"{REHEARSAL}; the memory challenger ran on auditor, not the developer's n0"
    assert record["uncertified"][0] == f"key custody: {where[WP.PREMISE]}"


@pytest.mark.parametrize("pin", ["ab" * 63, "AB" * 64, "ab" * 64 + "\n", 7])
def test_a_pin_is_a_sha512_in_lowercase_hex(pin) -> None:
    with pytest.raises(ValueError, match="wipe.trusted_sha512 is null or the SHA-512"):
        W.config({"wipe": {"trusted_sha512": pin}})
    assert W.config({"wipe": {"trusted_sha512": "ab" * 64}})["trusted_sha512"] == "ab" * 64
    assert W.config({})["trusted_sha512"] is None


def test_what_the_bound_does_not_cover() -> None:
    """The memory outside the declared tier, a placement off its own host, and a counted link left out."""
    gpu = W.undeclared("gpu-hbm", 100, {"tier": "gpu-hbm", "total_bytes": 1000, "device": "NVIDIA H200"})
    assert gpu[0] == {"tier": "gpu-hbm", "bytes": 900, "what": "NVIDIA H200's memory outside the declared tier"}
    assert gpu[1]["tier"] == "host-dram" and gpu[1]["what"] == "the host's DRAM"
    whole = W.undeclared("gpu-hbm", 1000, {"tier": "gpu-hbm", "total_bytes": 1000, "device": "NVIDIA H200"})
    assert [u["tier"] for u in whole if u["tier"] == "gpu-hbm"] == []
    host = W.undeclared("host-dram", 100)
    assert host[0]["tier"] == "host-dram" and host[0]["bytes"] == gpu[1]["bytes"] - 100
    counted = WP.link("relay", "n0", "outside", "eth1", mbps=10_000, counted_by="network certifier")
    where = WP.placement("n0", "n0", OTHER, simulated=True, links=[counted, WP.link("control", "n0", "outside", "eth0")],
                         undeclared=gpu[:1])
    gaps = W.uncertified(where, None)
    assert gaps[0] == f"key custody: {where[WP.PREMISE]}" and len(gaps) == 3
    assert gaps[1] == ("tier gpu-hbm: NVIDIA H200's memory outside the declared tier (900 bytes) is undeclared: the fill "
                       "doesn't cover it and M doesn't count it")
    assert gaps[2] == ("link control: n0 to outside (eth0) is counted by no network certifier, so what leaves on it is "
                       "outside E")
    assert WP.utc(1_793_088_000_123_456_789) == "2026-10-27T08:00:00.123456Z"


def test_what_a_joint_store_and_its_disks_leave_out() -> None:
    """A joint store declares each part at its own bytes: DRAM outside a host-dram part, nothing of a device declared
    whole; a virtual disk's backing store, a drive's undeclared spare area and buffered I/O are named."""
    disk = {"tier": "disk", "total_bytes": 10 ** 12, "spare_bytes": 0, "direct": True,
            "disks": {"vda": {"bytes": 10 ** 12, "driver": "virtio_blk"}, "nvme0n1": {"bytes": 10 ** 12, "driver": "nvme"}}}
    joint = {"tier": "disk+host-dram", "parts": [disk, {"tier": "host-dram", "bytes": 100}]}
    dram = W.undeclared("disk+host-dram", 10 ** 12 + 100, joint)[0]
    assert dram == {"tier": "host-dram", "bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") - 100,
                    "what": "the host's DRAM outside the declared tier"}
    gpu = {"tier": "disk+gpu-hbm", "parts": [disk, {"tier": "gpu-hbm", "total_bytes": 1000, "bytes": 900}]}
    assert [u["tier"] for u in W.undeclared("disk+gpu-hbm", 0, gpu) if u["tier"] == "gpu-hbm"] == []
    gaps = W.disk_gaps(joint)
    assert gaps == [
        "disk nvme0n1: its spare area and cache are undeclared (spare_bytes 0), so M doesn't count what they keep",
        "disk vda: a virtual disk (virtio_blk), its backing store the hypervisor's: a snapshot or replica the cloud keeps "
        "of it is outside the boundary, which Covered assumes away"]
    assert W.disk_gaps(disk | {"spare_bytes": 7, "direct": False}) == [gaps[1], (
        "disk: buffered I/O, not O_DIRECT: the page cache can answer a read the disk doesn't keep")]
    where = WP.placement("n0", "auditor", OTHER, store=joint, links=[])
    assert W.uncertified(where, None)[-2:] == gaps and W.disk_gaps(None) == []


def test_an_unmounted_block_device_no_target_is_on_is_undeclared_whole(tmp_path, monkeypatch) -> None:
    """Every `/sys/block` device, mounted or not (vy-dc-h200-2's 1 MiB `vdb`, the cloud's seed disk, is on no mount),
    but loop and RAM devices, an empty one and a device-mapper device over a disk counted itself; the declared disks
    are left out."""
    block = tmp_path / "block"
    for name, sectors in (("vda", 1 << 20), ("vdb", 2048), ("nvme0n1", 1 << 21), ("dm-0", 1 << 21), ("loop0", 8),
                          ("ram0", 8), ("zram0", 8), ("sr0", 0)):
        (block / name / "slaves").mkdir(parents=True)
        (block / name / "size").write_text(f"{sectors}\n")
    (block / "dm-0" / "slaves" / "nvme0n1").touch()
    assert W.disks(block) == {"nvme0n1": 1 << 30, "vda": 1 << 29, "vdb": 1 << 20}
    with pytest.raises(FileNotFoundError):
        W.disks(tmp_path / "missing")
    real = W.disks
    monkeypatch.setattr(W, "disks", lambda skip=(): real(block, skip))
    disk = {"tier": "disk", "total_bytes": 1 << 29, "bytes": 1 << 28, "spare_bytes": 0, "direct": True,
            "disks": {"vda": {"bytes": 1 << 29, "driver": "virtio_blk"}}}
    joint = {"tier": "disk+host-dram", "parts": [disk, {"tier": "host-dram", "bytes": 100}]}
    got = [u for u in W.undeclared("disk+host-dram", 0, joint) if u["tier"] == "disk"]
    assert got == [{"tier": "disk", "bytes": 1 << 30, "what": "the host's block device nvme0n1, outside the declared disks"},
                   {"tier": "disk", "bytes": 1 << 20, "what": "the host's block device vdb, outside the declared disks"}]
    where = WP.placement("n0", "auditor", OTHER, store=joint, links=[], undeclared=got)
    assert ("tier disk: the host's block device vdb, outside the declared disks (1048576 bytes) is undeclared: the fill "
            "doesn't cover it and M doesn't count it") in W.uncertified(where, None)


def test_the_dram_a_fill_can_take_is_memavailable(tmp_path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  100 kB\nMemFree:  10 kB\nMemAvailable:  75 kB\n")
    assert W.dram_available(meminfo) == 75 * 1024 and W.dram_available(tmp_path / "missing") > 0


def test_the_config_and_the_device_fit() -> None:
    assert W.config({}) == W.CONFIG and W.config({"wipe": {"store": "cuda"}})["store"] == "cuda"
    with pytest.raises(ValueError, match="unknown keys"):
        W.config({"wipe": {"segment": 4}})
    with pytest.raises(ValueError, match="store is python, host, cuda or disk"):
        W.config({"wipe": {"store": "tpu"}})
    disk = W.config({"wipe": {"store": "cuda", "disk": {"paths": ["/scratch"], "spare_bytes": 7}}})["disk"]
    assert disk == W.DISK | {"paths": ["/scratch"], "spare_bytes": 7}
    for bad in ({"store": "disk"}, {"disk": {"paths": ["/scratch"]}},
                {"store": "host", "tier": "x", "disk": {"paths": ["/scratch"]}}):
        with pytest.raises(ValueError, match="store disk takes wipe.disk"):
            W.config({"wipe": bad})
    for bad in ({"paths": []}, {"paths": "/scratch"}, {"paths": ["/s"], "bytes": 0}, {"paths": ["/s"], "direct": 1},
                {"paths": ["/s"], "spare_bytes": -1}):
        with pytest.raises(ValueError, match="wipe.disk: paths a list"):
            W.config({"wipe": {"store": "disk", "disk": bad}})
    with pytest.raises(ValueError, match="wipe.disk takes"):
        W.config({"wipe": {"store": "disk", "disk": {"paths": ["/s"], "size": 1}}})
    assert W.config({"wipe": {"store": "cuda", "dram": {"bytes": 1 << 30}}})["dram"] == W.DRAM | {"bytes": 1 << 30}
    for bad in ({"store": "host", "dram": {}}, {"dram": {}}, {"store": "cuda", "tier": "x", "dram": {}}):
        with pytest.raises(ValueError, match="wipe.dram joins a cuda or disk store"):
            W.config({"wipe": bad})
    for bad in ({"bytes": 0}, {"bytes": "1"}, {"reserve_bytes": -1}):
        with pytest.raises(ValueError, match="wipe.dram: bytes null"):
            W.config({"wipe": {"store": "cuda", "dram": bad}})
    with pytest.raises(ValueError, match="wipe.dram takes"):
        W.config({"wipe": {"store": "cuda", "dram": {"size": 1}}})
    for bad in ({"foreign_bytes": -1}, {"foreign_bytes": "f"}, {"foreign_bytes": 1.5}, {"placement": "far"},
                {"egress_bits": -1}, {"egress_bits": "0"}):
        with pytest.raises(ValueError, match="foreign_bytes is a byte count"):
            W.config({"wipe": bad})
    assert W.config({"wipe": {"links": [CONTROL | {"mbps": 1000, "counted_by": None}]}})["links"][0]["mbps"] == 1000
    for bad in ({"name": "x", "to": "y"}, CONTROL | {"ends": ["a", "b"]}, CONTROL | {"mbps": "1"}, "control"):
        with pytest.raises(ValueError, match="wipe.links: each is"):
            W.config({"wipe": {"links": [bad]}})
    p = W.params_for()
    assert W.foreign_labels("fM", p) == SEGMENT_2152 // sw.LABEL_BITS + 1
    assert W.foreign_labels(100, p) == 2 and W.foreign_labels(0, p) == 0 and W.foreign_labels(1 << 40, p) == p.labels

    class Free(WS.Cuda):
        def __init__(self, free: int) -> None:
            self.free_bytes = free

        def mem_info(self) -> tuple[int, int]:
            return self.free_bytes, 0

    c = W.config({"wipe": {"store": "cuda", "reserve_bytes": 1 << 20}})
    assert W.segments_of(c, Free(5 * W.SEGMENT_BYTES + 100)) == 4
    assert W.segments_of(c | {"segments": 9}, Free(0)) == 9 and W.segments_of(W.CONFIG, WS.HostMemory()) == W.SEGMENTS
    with pytest.raises(RuntimeError, match="less than one segment"):
        W.segments_of(c, Free(W.SEGMENT_BYTES))
    h200_after_context = Free(149_557_477_376)
    assert W.segments_of(W.config({"wipe": {"store": "cuda"}}), h200_after_context) * W.SEGMENT_BYTES == 148_478_361_600
    assert W.segments_of(W.config({"wipe": {"store": "cuda", "reserve_bytes": 64 << 20}}), h200_after_context) == 4455


@pytest.mark.parametrize("k", [W.READS, 5231])
@pytest.mark.parametrize("labels", [1, 16, 1000, 1 << 19, 3 << 20, 1 << 30, 2_203_125_000, 1 << 50])
def test_the_threshold_is_past_the_streamed_wipes_own_bound(labels: int, k: int) -> None:
    """The streamed wipe's own bound (`retained_bits`, `Pous.Guarantees.StreamedWipe` before its PRF term, its best
    `g` and exact count) at `δ = 2^-128`, egress 0 and a GiB outside the fill, is at most the threshold `certify`
    publishes (`StreamedWipeMeetsReads`', its `g = n − ⌈n/r⌉` and crude count), at every fill."""
    p = sw.WipeParams(labels, k, 1, W.LIMIT_NS, (("gpu", labels * sw.LABEL_BYTES + (1 << 30)),))
    q, _ = sw.retained_bits(labels, k, W.memory_bits(p), 0, -128)
    assert q <= W.certify(p, 0)["foreign_bits"] < W.memory_bits(p), (labels, k, q)
