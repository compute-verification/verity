"""`gate_whatif`: a session replayed at its own round trip to the challenger is its record, to the nanosecond, on both
clocks, and so is its line of record; replayed at another it is the record the k = 1 loop would have left there; a
record of another form is refused; and the gate sessions' peer is the run's record's."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auditor_line as al  # noqa: E402
import gate_whatif as gw  # noqa: E402
import verifier  # noqa: E402
import wire  # noqa: E402
from test_challengers import PINS, analysed, host  # noqa: E402,F401
from test_session import run as run_sessions  # noqa: E402

MS = 1_000_000
THETA = 7_000 * MS
"""The challenger's clock less the verifier's."""


def loop(rtt: int, passes: list[int], *, tau: int = MS // 2, gap: int = MS // 5) -> tuple[dict, list[str], dict]:
    """The record of a k = 1 session whose every exchange with the challenger is `rtt` ns, half each way, and whose
    passes take `passes`: verifier.json's session, the challenger's release log and the prover's session. σ_0's fetch
    holds the loop until it returns, then σ_1's; at root g the verifier asks openings coin g, and σ_(g+2) behind it,
    once it has sent window g's boundary, draws for d after the reply and sends the boundary s after that."""
    n, h = len(passes), rtt // 2
    c0, po, fsync, ps, d, s = 40_000, 30_000, 2 * MS, 20_000, 300_000, 100_000
    R, A, D, C, b = [0] * n, [0] * n, [0] * n, [0] * n, [0] * n
    log = [("challenge", 0, "f" * 128, R[0] + h, R[0] + h + c0)]
    C[0] = R[0] + 2 * h + c0 + tau
    R[1] = b[0] = R[0] + 2 * h + c0
    log.append(("challenge", 1, "f" * 128, R[1] + h, R[1] + h + c0))
    C[1] = R[1] + 2 * h + c0 + tau
    free = None
    for g in range(n):
        start = C[g] if free is None else max(C[g], free)
        A[g] = start + passes[g] + gap + tau
        free = A[g] - tau
        root, got = f"{g:0128x}", max(A[g], b[g] if g else R[1] + 2 * h + c0) + h
        log.append(("openings", g, root, got, got + po))
        end = got + po
        if g + 2 < n:
            log.append(("challenge", g + 2, root, end + fsync, end + fsync + ps))
            end += fsync + ps
        D[g] = end + h + d
        if g + 1 < n:
            b[g + 1] = D[g] + s
        if g + 2 < n:
            R[g + 2] = b[g + 1]
            C[g + 2] = R[g + 2] + tau
    lines = [json.dumps({"event": "commitment", "root": "0" * 128})]
    for label, i, root, got, out in log:
        lines.append(json.dumps({"event": "root", "label": label, "index": i, "root": root,
                                 "received_at": got + THETA}))
        lines.append(json.dumps({"event": "release", "label": label, "index": i, "after": root,
                                 "released_at": out + THETA}))
    seen = [{"t": g, "released_ns": R[g], "boundary_ns": b[g], "passes": 1, "arrivals_ns": [A[g]],
             "draws_sent_ns": D[g]} for g in range(n)]
    proved = {"k": 0, "pass_host_ns": [A[g] - tau - gap - C[g] for g in range(n)], "gap_host_ns": [gap] * n}
    return {"k": 0, "pace": {"k": 1}, "window_ns": 1, "windows_seen": seen}, lines, proved


def recorded(rtt: int, passes: list[int], tau: int = MS // 2) -> tuple[gw.Record, dict]:
    s, lines, proved = loop(rtt, passes, tau=tau)
    auditor, _ = al.auditor_seen(lines, s["windows_seen"])
    rec = gw.record(s, lines, auditor, proved, tau)
    return rec, {"R": rec.R, "A": rec.A, "D": rec.D, "Ra": rec.Ra, "Aa": rec.Aa}


#: passes of 110 ms with every fifth a short 50 ms one, whose root waits on the verifier's exchange at 100 ms
PASSES = [50 * MS if g % 5 == 4 else 110 * MS for g in range(24)]


def test_a_record_replayed_at_its_own_round_trip_is_itself_and_at_another_is_the_loops_record_there():
    """The session's round trip is read off both clocks (the least exchange less the challenger's own time, σ_0's fetch
    here, which has no draws to compute), whatever the clocks' offset. At it the replay is the record; at 2 ms it is
    the record of the same loop at 2 ms, on both clocks, where the short passes' roots no longer wait on the verifier."""
    rec, own = recorded(100 * MS, PASSES)
    assert rec.r0 == 100 * MS and min(rec.net) == 100 * MS + 300_000
    assert sum(x > 0 for x in rec.queue()) == 4
    assert gw.replay(rec, rec.r0) == own
    near, there = recorded(2 * MS, PASSES)
    assert gw.replay(rec, 2 * MS) == there and gw.replay(near, 100 * MS) == own
    assert sum(x > 0 for x in near.queue()) == 0


def test_a_record_replayed_at_faster_passes_is_the_loops_record_at_them_and_says_what_it_waits_on():
    """Each pass's own time halved, or every pass 40 ms, at the session's round trip and at 2 ms, is the record the
    loop leaves with those passes there. At the record's 100 ms exchange the halved passes wait on their coins, and the
    worst group stays the exchange's; at 2 ms nothing waits on a coin and it is the passes'."""
    rec, _ = recorded(100 * MS, PASSES)
    assert gw.passes_of(rec) == PASSES == gw.passes_of(rec, scale=1.0)
    half = gw.passes_of(rec, scale=0.5)
    assert half == [p // 2 for p in PASSES] and gw.passes_of(rec, ms=40) == [40 * MS] * len(PASSES)
    for rtt in (100 * MS, 2 * MS):
        assert gw.replay(rec, rtt, half) == recorded(rtt, half)[1]
        assert gw.replay(rec, rtt, [40 * MS] * len(PASSES)) == recorded(rtt, [40 * MS] * len(PASSES))[1]
    own, far, near = (gw.waits(rec, gw.replay(rec, r, p), r) for r, p in
                      ((rec.r0, None), (rec.r0, half), (2 * MS, half)))
    assert own == {"coin": 4, "verifier": 4}
    assert far["coin"] == len(PASSES) - 1 and near["coin"] == 0
    l_max = lambda t: max(al.group_l([{"released_ns": x, "arrivals_ns": [y]} for x, y in zip(t["Ra"], t["Aa"])]))
    assert l_max(gw.replay(rec, rec.r0, half)) > 100 * MS + 55 * MS > l_max(gw.replay(rec, 2 * MS, half))


@pytest.mark.parametrize("broken", ["alone", "k3", "short"])
def test_a_record_of_another_loop_is_refused_by_name(broken):
    """A σ_(g+2) asked on its own, a pace of k = 3 and a prover's record short of its groups are refused, each with
    what the replay models."""
    s, lines, proved = loop(100 * MS, PASSES)
    if broken == "alone":
        j = next(i for i, x in enumerate(map(json.loads, lines))
                 if x["event"] == "root" and x.get("label") == "openings" and x["index"] == 3)
        lines[j + 2:j + 6] = lines[j + 4:j + 6] + lines[j + 2:j + 4]
    elif broken == "k3":
        s = {**s, "pace": {"k": 3}}
    else:
        proved = {**proved, "pass_host_ns": proved["pass_host_ns"][:-1]}
    why = {"alone": "wasn't asked with openings coin g", "k3": "the replay models k = 1",
           "short": "the prover's record has 23 passes for its 24 groups"}[broken]
    with pytest.raises(gw.Refused, match=why):
        gw.record(s, lines, al.auditor_seen(lines, s["windows_seen"])[0], proved, MS // 2)


def test_a_host_runs_replay_at_its_own_round_trip_is_its_line_of_record_and_its_peer_is_the_records(
        host, tmp_path, monkeypatch):
    """A gate run of one honest paced session whose verifier ran on the auditor host, in the form gate run 7's record
    has: its verifier.json names no challenger peer, the request.json beside out/ does (--challenger-peer
    vy-auditor-1), its own `peer` being the audited node. The line of record `whatif` reads (`auditor_line`'s, as
    auditor_line.py writes it) takes the peer from that record, the session is of record and credited, and its replay
    at its own round trip gives back the release log's and the verifier's times to the nanosecond and that line's
    shares on both clocks."""
    host.seal(monkeypatch, PINS)
    out = tmp_path / "run" / "out"
    desc = al.descriptor("ab" * 20)
    host.period(circuit=hashlib.sha512(desc).hexdigest())
    got = {}
    pace = verifier.pace_of("1:4:160:2", 1.0)
    sessions, proved = run_sessions("honest", paces=[pace], challenger=host.gate(out / "challenger", peer="vy-auditor-1"),
                                    defer_openings=True,
                                    then=lambda v: got.update(reg=wire.registration_json(v.registration)))
    rc, closed = host.call(["period-close", "--name", "p-1"])
    assert rc == 0 and json.loads(closed)["body"]["complete"] is True, closed
    bare = [{**s, "challenger": {k: x for k, x in s["challenger"].items() if k != "peer"}} for s in sessions]
    (out.parent / "request.json").write_text(json.dumps({"peer": "vy-nebius-2", "argv": [
        "/usr/bin/python3", "-I", "verifier.py", "--challenger", "gate", "--challenger-peer", "vy-auditor-1"]}))
    line = analysed(out, host, bare, got["reg"], desc)
    v = json.loads((out / "verifier.json").read_text())
    (out / "verifier.json").write_text(json.dumps({**v, "rtt": {"median_ns": 0}}))
    doc = gw.whatif(out, {"sessions": proved}, json.loads(json.dumps(line)), rtt_ms=(0.001,))
    assert doc["peer"] == {"name": "vy-auditor-1", "from": "record"}
    (s,) = doc["sessions"]
    assert s["of_record"] and s["credited"] and "refused" not in s, s.get("refused")
    assert s["identity"] == {"holds": True, "groups_off": dict.fromkeys(("R", "A", "D", "Ra", "Aa"), 0),
                             "shares_equal": {"auditor": True, "verifier": True}}
    assert s["own"]["shares_auditor_clock"] == line["sessions"][0]["shares_auditor_clock"]
    assert doc["line_of_record"] == line["line"] and doc["identity_holds"]
    assert doc["whatif_line"][s["T"]]["credited_sessions"] == [0]
