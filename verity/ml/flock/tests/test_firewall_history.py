"""The firewall's record as it grows (`verity_flock.firewall`): a call reads only what other processes appended since its
`Firewall` object's last call, never the whole record, and each call's result or refusal, and the record it leaves, are a
firewall's that reads its whole record at every call (`_Whole`). They are compared on a fixed stream of ops by three
processes, with blank lines, a torn line and its repair, a line without its newline, a line the firewall doesn't write,
the record replaced, rewritten in place and cut shorter."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

from verity_flock import firewall as FW

PARTS = (["L0", "L1", "alg"], ["L0"], ["L0", "L1"])


class _Whole(FW.Firewall):
    """A firewall that reads its whole record at every call, each rule's read a scan of it."""

    def _read(self):
        return FW._Scan(self._events())


class _Json:
    """`json` as the firewall module sees it, counting the lines it parses."""

    def __init__(self) -> None:
        self.lines = 0

    def loads(self, s):
        self.lines += 1
        return json.loads(s)

    def __getattr__(self, name):
        return getattr(json, name)


def _lines(state: Path) -> int:
    log = state / "record.jsonl"
    return log.read_bytes().count(b"\n") if log.exists() else 0


def test_a_call_reads_only_what_other_processes_appended(tmp_path, monkeypatch):
    """300 rounds by two processes on one state: one opens and finishes each recursive session, the other serves its
    statements' outer sessions (slot, admit, release), and every fifth session stops at its second statement, so the rest
    are refused. No call reads the record whole, and each parses only the lines the other process appended since its own
    last call: a call's work doesn't grow with the record. A firewall that reads its whole record parses all of it at
    every call."""
    parsed, whole = _Json(), []
    monkeypatch.setattr(FW, "json", parsed)
    events = FW.Firewall._events
    monkeypatch.setattr(FW.Firewall, "_events", lambda self: (whole.append(type(self)), events(self))[1])
    state = tmp_path / "fw"
    inner, outer = FW.Firewall(state), FW.Firewall(state)
    seen = {inner: 0, outer: 0}

    def call(fw, f):
        before, n = _lines(state), parsed.lines
        try:
            return f()
        finally:
            assert parsed.lines - n == before - seen[fw]
            seen[fw] = _lines(state)

    for i in range(300):
        sid = call(inner, lambda: inner.open("s", PARTS[0], budget=10 ** 6))
        call(inner, lambda: inner.finish(sid))
        for j, p in enumerate(PARTS[0]):
            slot = outer.slot(sid, p)
            call(outer, slot.__enter__)
            try:
                if i % 5 == 4 and j > 0:
                    refused = call(outer, lambda: _raised(lambda: outer.admit(sid, p)))
                    assert isinstance(refused, FW.Refused) or j == 1
                    if j == 1:
                        call(outer, lambda: outer.stop(sid, p, "cut"))
                    continue
                call(outer, lambda: outer.admit(sid, p))
                call(outer, lambda: outer.release(sid, p))
            finally:
                slot.__exit__(None, None, None)
    assert call(inner, lambda: inner.failures("s")) == 60 and whole == []
    assert parsed.lines == _lines(state) == 300 * 8  # each line once, by the process that didn't write it

    monkeypatch.setattr(FW, "json", counted := _Json())
    w = _Whole(state)
    w.failures("s")
    assert counted.lines == _lines(state) and whole[-1] is _Whole


def _raised(f):
    try:
        f()
    except FW.Refused as e:
        return e
    return None


def _drive(cls, state: Path, seed: int, steps: int) -> tuple[list, list[int]]:
    """A fixed stream of `steps` ops by three `cls` objects on `state`, with the record damaged and repaired between them
    as the stream says: each step's result or exception, and the record's digest after it; and of the reads of the record,
    how many the index answered and how many there were."""
    rng, fws = random.Random(seed), [cls(state) for _ in range(3)]
    log, out, sids = state / "record.jsonl", [], ["r9999"]
    reads = [0, 0]
    if cls is FW.Firewall:
        for fw in fws:
            read = fw._read

            def counted(read=read):
                reads[1] += 1
                r = read()
                reads[0] += isinstance(r, FW._Index)
                return r
            fw._read = counted
    mark = None  # the record's size before a damage that is to be repaired

    def damage(kind: str) -> None:
        nonlocal mark
        data = log.read_bytes() if log.exists() else b""
        if kind == "blank":
            log.write_bytes(data + b"\n")
        elif kind in ("torn", "unterminated", "foreign") and mark is None:
            mark = len(data)
            n = data.count(b"\n")
            line = {"torn": b'{"event": "sto', "foreign": b'{"event": "note", "seq": 7}\n',
                    "unterminated": json.dumps({"event": "finish", "part": "inner", "seq": n, "session": sids[-1]},
                                               sort_keys=True).encode()}[kind]
            with open(log, "ab") as f:
                f.write(line)
        elif kind == "repair" and mark is not None:
            os.truncate(log, mark)
            mark = None
        elif kind == "replace":
            tmp = state / "record.tmp"
            tmp.write_bytes(data)
            os.replace(tmp, log)
        elif kind == "rewrite":
            log.write_bytes(data)
        elif kind == "cut" and mark is None and data.count(b"\n") > 4:
            log.write_bytes(b"".join(data.splitlines(keepends=True)[:-3]))

    for _ in range(steps):
        fw, r = rng.choice(fws), rng.random()
        sid, part = rng.choice(sids[-4:] + ["r9999"]), rng.choice(["L0", "L1", "alg", "L9"])
        stream = rng.choice(["s", "t"])
        if r < 0.04:
            kind = rng.choice(["blank", "torn", "unterminated", "foreign", "replace", "rewrite", "cut"])
            damage(kind)
            out.append(("damage", kind))
        elif r < 0.10:
            damage("repair")
            out.append(("damage", "repair"))
            continue
        else:
            op = rng.choice(["open", "open", "finish", "admit", "admit", "admit", "release", "release", "stop", "failures",
                             "stopped", "slot", "slot", "record"])
            budget = {"s": 4, "t": None}[stream] if rng.random() > 0.05 else 9
            fn = {"open": lambda: fw.open(stream, rng.choice(PARTS), budget=budget),
                  "finish": lambda: fw.finish(sid), "admit": lambda: fw.admit(sid, part),
                  "release": lambda: fw.release(sid, part), "stop": lambda: fw.stop(sid, part, "cut"),
                  "failures": lambda: fw.failures(stream), "stopped": lambda: fw.stopped(sid), "record": fw.record,
                  "slot": lambda: _slot(fw, sid, part, rng.random())}[op]
            try:
                got = ("result", op, fn())
            except Exception as e:  # noqa: BLE001 - the two firewalls are compared on what they raise too
                got = ("raise", op, type(e).__name__, str(e))
            if op == "open" and got[0] == "result":
                sids.append(got[2])
            out.append(got)
        out.append(hashlib.sha256(log.read_bytes() if log.exists() else b"").hexdigest())
    return out, reads


def _slot(fw: FW.Firewall, sid: str, part: str, r: float):
    """An outer session in the slot: admitted and released, admitted and stopped, or its holder ending before its
    outcome (so the next to take the slot records its stop)."""
    with fw.slot(sid, part):
        if r < 0.3:
            return "ended"
        at = fw.admit(sid, part)
        if r < 0.8:
            fw.release(sid, part)
        else:
            fw.stop(sid, part, "the challenger's verdict rejects it")
        return at


def test_every_result_and_the_record_are_the_whole_records(tmp_path):
    """On a fixed stream of 1,500 ops (opens on two streams, one with a budget, finishes, admits, releases, stops, outer
    sessions in the slot and holders that end before their outcome) by three processes, between which the record is
    damaged and repaired, every result, refusal and error, and the record after each op, are those of the firewall that
    reads its whole record; the index answered most of the reads, the rest being those of a damaged record."""
    got, (fast, reads) = _drive(FW.Firewall, tmp_path / "new", 20261016, 1500)
    want, _ = _drive(_Whole, tmp_path / "whole", 20261016, 1500)
    assert got == want
    kinds = {c[0] if isinstance(c, tuple) else "digest" for c in got}
    raised = {c[2] for c in got if isinstance(c, tuple) and c[0] == "raise"}
    damaged = {c[1] for c in got if isinstance(c, tuple) and c[0] == "damage"}
    assert kinds == {"result", "raise", "damage", "digest"} and {"Refused", "JSONDecodeError", "KeyError"} <= raised
    assert damaged == {"blank", "torn", "unterminated", "foreign", "replace", "rewrite", "cut", "repair"}
    assert fast > 0.7 * reads
