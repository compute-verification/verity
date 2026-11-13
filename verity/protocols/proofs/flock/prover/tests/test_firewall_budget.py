"""A stream's failure budget on Python's ZK server (`verity_flock.firewall`, live PROTOCOL.md §10.5), driven as its callers
drive it: `rec_live` opens a recursive session and records its inner session's finish or stop; the relay seats each outer
session (`Firewall.seat`) and records its outcome, or ends while it holds the slot. At most b of a stream's recursive
sessions ever have a statement not released, b its budget: the bound the Lean proves at every order of the record's ops
(`card_unrel_le_budget`), and so in its model of the loop (`StopCount.run`, `card_unreleased_le_budget`), which these
tests check over sampled histories. It comes from the open's rule (refused while the stream's unended recursive sessions
and its failures reach b), and from no stop being recorded of a session not yet opened, so that each stop counts against
its stream. The worker and the challenger are `test_firewall_release`'s, on socket pairs;
the clock is never read."""
from __future__ import annotations

import json
import random
from collections import Counter

import pytest
from verity_flock import firewall as FW

import test_firewall_release as F

NAMES = ("L0", "L1", "alg")
#: the streams' budgets: b = 1 to 4, and one with none, which shares the slot with them
BUDGETS = {"p": 1, "q": 2, "s": 3, "t": 4, "u": None}
SEEDS, STEPS = range(16), 200
#: the one stream's budget in the tests of a single history
B = 3


def _unreleased(record: list[dict], stream: str) -> int:
    """How many of ``stream``'s recursive sessions (each one's first open on it) have a statement not released, read off
    the record alone."""
    first: dict[str, dict] = {}
    for e in record:
        if e["event"] == "open":
            first.setdefault(e["session"], e)
    released = {(e["session"], e["part"]) for e in record if e["event"] == "release"}
    return sum(o["stream"] == stream and any((s, p) not in released for p in o["parts"]) for s, o in first.items())


def _unended(record: list[dict]) -> dict[str, str]:
    """The recursive sessions opened, not stopped, with a statement not released, in the order they opened: each one's
    next statement."""
    stopped = {e["session"] for e in record if e["event"] == "stop"}
    released = {(e["session"], e["part"]) for e in record if e["event"] == "release"}
    return {e["session"]: nxt for e in record if e["event"] == "open" and e["session"] not in stopped
            and (nxt := next((p for p in e["parts"] if (e["session"], p) not in released), None)) is not None}


def _history(fw: FW.Firewall, seed: int, steps: int, bounded: bool = True) -> tuple[dict[str, int], Counter]:
    """At most `steps` ops on ``fw``, sampled from ``seed``, until each budgeted stream has stopped at its budget: an open on
    one of `BUDGETS`' streams; the inner session of a recursive session opened and not yet ended finishing, or stopping;
    or an outer session, at any statement, of an unended recursive session, of any one opened, of the next one to be
    opened or of one never opened, seated by the relay: released on an accepting verdict, stopped by a rejecting one, by
    ``Finish`` before its proofs or by the worker ending before ``Finish``, or the relay ending while it holds the slot.
    After each op, each budgeted stream's sessions with a statement not released, at most its budget if ``bounded``:
    their most, and what each op came to."""
    rng, opened, ended, finished, log, record = random.Random(seed), [], set(), set(), [], []
    most, seen = {t: 0 for t, b in BUDGETS.items() if b is not None}, Counter()
    for _ in range(steps):
        if all(fw.failures(t) >= BUDGETS[t] for t in most):
            break
        r = rng.random()
        if r < 0.25:
            stream = rng.choice(sorted(BUDGETS))
            try:
                opened.append(fw.open(stream, rng.sample(NAMES, rng.randint(1, len(NAMES))), BUDGETS[stream]))
                seen["open"] += 1
            except FW.Refused as e:
                seen["open:unended" if "unended" in str(e) else "open:stopped"] += 1
        elif r < 0.45:
            live = [s for s in opened if s not in ended]
            if live:
                s = rng.choice(live)
                ended.add(s)
                if rng.random() < 0.8:
                    fw.finish(s)
                    finished.add(s)
                else:
                    fw.stop(s, FW.INNER, "rec proxy: the session ended without Finish")
        else:
            unended = _unended(record)
            ready = [s for s in unended if s in finished]
            if ready and rng.random() < 0.7:
                s = rng.choice(ready)
                part = unended[s]
            else:
                some = [rng.choice(opened)] if opened else []
                s, part = rng.choice([*unended, *some, f"r{len(opened):04d}", "r9999"]), rng.choice(NAMES)
            how = rng.choice(["accept"] * 4 + ["reject", "send check", "cut", "ends"])
            if how == "ends":
                with fw.seat(s, part) as why:
                    seen["ends" if why is None else "refused"] += 1
            else:
                reqs = F.REFUSED if how == "send check" else F.HONEST
                out, _ = F._outer(fw, s, part, reqs, F.Challenger(log, s, accept=how != "reject"),
                                  cut_after=rng.randint(1, len(reqs) - 1) if how == "cut" else None)
                seen[out["outcome"]] += 1
        record = fw.record()
        for t in most:
            u = _unreleased(record, t)
            assert not bounded or u <= BUDGETS[t], (seed, t, u, len(record))
            most[t] = max(most[t], u)
    return most, seen


def test_at_most_its_budget_of_a_streams_recursive_sessions_have_a_statement_not_released(tmp_path):
    """Over `SEEDS`' histories, after every op: at most b of each budgeted stream's recursive sessions have a statement not
    released, and the bound is reached on each, at b = 1 to 4. Every outcome of an outer session occurs, and opens are
    refused for the unended rule."""
    most, seen = {t: 0 for t, b in BUDGETS.items() if b is not None}, Counter()
    for seed in SEEDS:
        m, s = _history(FW.Firewall(tmp_path / f"fw-{seed}"), seed, STEPS)
        most = {t: max(most[t], m[t]) for t in most}
        seen += s
    assert most == {t: BUDGETS[t] for t in most}, most
    for k in ("open", "open:unended", "release", "stop", "refused", "ends"):
        assert seen[k] >= 10, (k, seen)


def test_without_the_open_rule_the_same_histories_exceed_the_budget(tmp_path, monkeypatch):
    """The bound is the open's rule's: with the stream's unended recursive sessions read as none, the histories of
    `test_at_most_its_budget_of_a_streams_recursive_sessions_have_a_statement_not_released` leave more than b of a stream's
    sessions with a statement not released."""
    for cls in (FW._Index, FW._Scan):
        monkeypatch.setattr(cls, "unended", lambda self, stream: 0)
    most = {t: 0 for t, b in BUDGETS.items() if b is not None}
    for seed in SEEDS:
        m, _ = _history(FW.Firewall(tmp_path / f"fw-{seed}"), seed, STEPS, bounded=False)
        most = {t: max(most[t], m[t]) for t in most}
    assert all(most[t] > BUDGETS[t] for t in most), most


def test_a_relay_that_ends_while_it_is_seated_leaves_the_record_as_it_was(tmp_path, monkeypatch):
    """`Firewall.seat` decides an outer session's admission before anything of it reaches the record, and appends its
    events in one write: a relay that ends at its decision leaves the record as it was. A relay asked for the next
    session's id, not yet opened, is refused, its ``slot`` followed by its refusal in that write, before any challenger;
    the next holder of the slot records no stop of that session (one would not count against its stream's budget, the
    session not yet being the stream's), so it opens unstopped, one of the stream's unended, and the stream's next open
    past its budget is refused."""
    fw, log = FW.Firewall(tmp_path / "fw"), []
    first = fw.open("s", list(NAMES), budget=B)
    fw.finish(first)
    record = tmp_path / "fw" / "record.jsonl"
    before = record.read_bytes()

    def ends(*a, **k):
        raise SystemExit("the relay ended")

    with monkeypatch.context() as m:
        m.setattr(FW.Firewall, "_admit", ends)
        with pytest.raises(SystemExit, match="the relay ended"):
            with fw.seat("r0001", "L0"):
                pass
    assert record.read_bytes() == before
    writes = []
    write = FW.Firewall._write
    monkeypatch.setattr(FW.Firewall, "_write", lambda self, rec, data: (writes.append(data), write(self, rec, data))[1])
    out, _ = F._outer(fw, "r0001", "L0", F.HONEST, F.Challenger(log, "next"))
    assert out["why"] == "FIREWALL-REFUSED: recursive session r0001 was never opened"
    assert [line.count(b"\n") for line in writes] == [2] and [(e["event"], e["session"]) for e in fw.record()[-2:]] == [
        ("slot", "r0001"), ("refused", "r0001")]
    out, _ = F._outer(fw, first, "L0", F.HONEST, F.Challenger(log, "first"))
    assert out["outcome"] == "release" and [(e["event"], e["session"]) for e in fw.record()[-2:]] == [
        ("slot", first), ("release", first)]
    assert [fw.open("s", list(NAMES), budget=B) for _ in range(B - 1)] == ["r0001", "r0002"]
    assert fw.stopped("r0001") is None and fw.failures("s") == 0 and {name for name, _ in log} == {"first"}
    with pytest.raises(FW.Refused, match=r"stream s's unended recursive sessions and failures reach its failure budget "
                                         r"\(3 and 0 of 3\)"):
        fw.open("s", list(NAMES), budget=B)


def test_a_seat_on_a_record_whose_last_line_has_no_newline_writes_nothing(tmp_path):
    """The slot's line would join a last line without its newline, so the Lean ZK server's ``seat`` fails at its
    admission's read and writes nothing; `Firewall.seat` does the same, and holds no slot after."""
    fw = FW.Firewall(tmp_path / "fw")
    fw.finish(fw.open("s", list(NAMES), budget=B))
    record = tmp_path / "fw" / "record.jsonl"
    record.write_bytes(record.read_bytes()[:-1])
    before = record.read_bytes()
    with pytest.raises(json.JSONDecodeError, match="the record's last line has no newline"):
        with fw.seat("r0000", "L0"):
            pass
    assert record.read_bytes() == before
    record.write_bytes(before + b"\n")
    with fw.seat("r0000", "L0") as why:
        assert why is None


def test_no_stop_is_recorded_of_a_session_not_yet_opened(tmp_path):
    """The record's ops in any order, not only the relay's seat: a slot's holder for the next session's id, not yet
    opened, that ends before its outcome (`Firewall.slot` without `Firewall.admit`, which `seat` never leaves), and a
    ``stop`` of that id, record no stop of it, so it opens unstopped, one of its stream's unended, and the stream's open
    past its budget is refused; at most b of its sessions ever have a statement not released."""
    fw = FW.Firewall(tmp_path / "fw")
    for k in range(2 * B):
        assert fw.stop(f"r{k:04d}", FW.INNER, "rec proxy: the session ended without Finish") is False
        with fw.slot(f"r{k:04d}", "L0"):
            pass
        with fw.slot("r9999", "L0"):
            pass
        try:
            fw.open("s", list(NAMES), budget=B)
        except FW.Refused as e:
            assert "unended recursive sessions and failures reach its failure budget (3 and 0 of 3)" in str(e)
        assert _unreleased(fw.record(), "s") <= B
    assert not any(e["event"] == "stop" for e in fw.record()) and fw.failures("s") == 0
    assert _unreleased(fw.record(), "s") == B
