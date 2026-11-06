"""The ZK server's record as it grows (`verity_flock.firewall`): a call reads only what was appended since the index
(``index-py/``) last read the record, never the whole record, whichever object or process makes it; and each call's
result or refusal, and the record it leaves, are a ZK server's that reads its whole record at every call (`_Whole`). They
are compared on fixed streams of ops, with the record damaged (blank lines, a torn line and its repair, a line without
its newline, a line the ZK server doesn't write, the record replaced as it was or by one of its length whose first open
is another event, rewritten, cut shorter, its last lines edited in place, halved) and the index removed or torn between
them."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil

import pytest
from pathlib import Path

from verity_flock import firewall as FW

PARTS = (["L0", "L1", "alg"], ["L0"], ["L0", "L1"])


class _Whole(FW.Firewall):
    """A ZK server that reads its whole record at every call, each rule's read a scan of it."""

    def _read(self, sessions=(), streams=()):
        return FW._Scan(self._events())


class _Json:
    """`json` as the ZK server module sees it, counting the record's lines it parses (``loads``; the index's files are read
    with ``load``)."""

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


def _appended(state: Path, ev: dict) -> None:
    """An event appended as another process writes it (the Lean ZK server's release side, say), outside this module's
    calls: its line, with its seq."""
    with open(state / "record.jsonl", "ab") as f:
        f.write((json.dumps({"seq": _lines(state), **ev}, sort_keys=True) + "\n").encode())


class _Work:
    """The work of this module's calls on one state: the record's lines parsed, the index's files read, the calls, those
    the index answered, and the record's whole reads."""

    def __init__(self, monkeypatch) -> None:
        self.json, self.loads, self.calls, self.indexed, self.whole = _Json(), 0, 0, 0, 0
        monkeypatch.setattr(FW, "json", self.json)
        get, read, events = FW._Index._get, FW.Firewall._read, FW.Firewall._events

        def counted_get(ix, name):
            self.loads += 1
            return get(ix, name)

        def counted_read(fw, *a):
            r = read(fw, *a)
            self.calls += 1
            self.indexed += isinstance(r, FW._Index)
            return r

        def counted_events(fw):
            self.whole += 1
            return events(fw)
        monkeypatch.setattr(FW._Index, "_get", counted_get)
        monkeypatch.setattr(FW.Firewall, "_read", counted_read)
        monkeypatch.setattr(FW.Firewall, "_events", counted_events)

    def now(self) -> tuple[int, int, int, int, int]:
        return self.json.lines, self.loads, self.calls, self.indexed, self.whole


def test_a_call_reads_only_what_was_appended_since_the_index_last_read(tmp_path, monkeypatch):
    """300 recursive sessions. Each one's inner side (open, finish) and its first statement's outer session (slot, admit,
    release) are this module's calls, each by a new `Firewall` (a process of its own); its other statements' outer
    sessions are appended as the Lean ZK server writes them, and every fifth stops at its second statement, so its third
    is refused. Each call parses only the lines appended outside since the index last read the record, never the whole
    record, and the index answers it; a round's work (lines parsed, files of the index read, calls) is the same in the
    last round as in the first of its kind. The record read whole parses all of it."""
    work, state = _Work(monkeypatch), tmp_path / "fw"
    seen, rounds = 0, []

    def call(f):
        nonlocal seen
        outside, before = _lines(state) - seen, work.now()
        try:
            return f(FW.Firewall(state))
        finally:
            after = work.now()
            assert (after[0] - before[0], after[2] - before[2], after[3] - before[3], after[4] - before[4]) == (outside, 1, 1, 0)
            seen = _lines(state)

    for i in range(300):
        start = work.now()
        sid = call(lambda fw: fw.open("s", PARTS[0], budget=10 ** 6))
        assert sid == f"r{i:04d}"
        call(lambda fw: fw.finish(sid))
        slot = FW.Firewall(state).slot(sid, "L0")
        call(lambda fw: slot.__enter__())
        assert call(lambda fw: fw.admit(sid, "L0")) is None
        call(lambda fw: fw.release(sid, "L0"))
        slot.__exit__(None, None, None)
        _appended(state, {"event": "slot", "session": sid, "part": "L1"})
        if i % 5 == 4:
            _appended(state, {"event": "stop", "session": sid, "part": "L1", "why": "cut"})
            _appended(state, {"event": "slot", "session": sid, "part": "alg"})
            _appended(state, {"event": "refused", "session": sid, "part": "alg"})
        else:
            _appended(state, {"event": "release", "session": sid, "part": "L1"})
            _appended(state, {"event": "slot", "session": sid, "part": "alg"})
            _appended(state, {"event": "release", "session": sid, "part": "alg"})
        end = work.now()
        rounds.append(tuple(b - a for a, b in zip(start, end)))
    for i in range(6, 300):
        assert rounds[i] == rounds[i - 5], i
    assert call(lambda fw: fw.failures("s")) == 60

    before = work.json.lines
    assert _Whole(state).failures("s") == 60 and work.json.lines - before == _lines(state)


def test_an_index_from_another_boot_or_left_mid_write_is_rebuilt(tmp_path, monkeypatch):
    """A head that another boot wrote, or that a writer left dirty (it ended between the head and the files it was
    writing), is not read: the next call rebuilds the index from the whole record, so a file of the index it would have
    trusted (a stream's failures, here) doesn't count, and the call after reads nothing of the record."""
    state = tmp_path / "fw"
    fw = FW.Firewall(state)
    sid = fw.open("s", ["L0", "L1"], budget=5)
    assert fw.stop(sid, "L0", "cut")
    work, index = _Work(monkeypatch), state / "index-py"
    head = json.loads((index / "head.json").read_text())
    for k, v in (("boot", "another boot"), ("dirty", True)):
        (index / "head.json").write_text(json.dumps({**head, k: v}))
        (index / f"t{b's'.hex()}.json").write_text(json.dumps({"budget": 5, "n": 4}))
        for parsed in (_lines(state), 0):
            before = work.json.lines
            assert FW.Firewall(state).failures("s") == 1, k
            assert work.json.lines - before == parsed, k


def _waits(lines: list[bytes], sid: str) -> list[bytes]:
    """Each line rewritten in place as a `wait` of its length, where one fits."""
    out = []
    for k, line in enumerate(lines):
        pad = len(line) - 1 - len(json.dumps({"event": "wait", "part": "", "seq": k, "session": sid}, sort_keys=True))
        out.append(line if pad < 0 else
                   (json.dumps({"event": "wait", "part": "x" * pad, "seq": k, "session": sid}, sort_keys=True) + "\n").encode())
    return out


def _drive(cls, state: Path, seed: int, steps: int, long: bool = False) -> tuple[list, list[int]]:
    """A fixed stream of `steps` ops by three `cls` objects on `state`, with the record and the index damaged and repaired
    between them as the stream says, and events another process appends: each step's result or exception, and the
    record's digest after it; and of the reads of the record, how many the index answered and how many there were.
    `long` draws session names past `KEY_MAX` too."""
    rng, fws = random.Random(seed), [cls(state) for _ in range(3)]
    log, index, out, sids = state / "record.jsonl", state / "index-py", [], ["r9999"]
    long_sid = "r" * (FW.KEY_MAX + 1)
    reads = [0, 0]
    if cls is FW.Firewall:
        for fw in fws:
            read = fw._read

            def counted(*a, read=read):
                reads[1] += 1
                r = read(*a)
                reads[0] += isinstance(r, FW._Index)
                return r
            fw._read = counted
    mark = None  # the record's size before a damage that is to be repaired

    def damage(kind: str, sid: str) -> None:
        nonlocal mark
        data = log.read_bytes() if log.exists() else b""
        lines = data.splitlines(keepends=True)
        if kind == "blank":
            log.write_bytes(data + b"\n")
        elif kind in ("torn", "unterminated", "foreign") and mark is None:
            mark = len(data)
            line = {"torn": b'{"event": "sto', "foreign": b'{"event": "note", "seq": 7}\n',
                    "unterminated": json.dumps({"event": "finish", "part": "inner", "seq": len(lines), "session": sid},
                                               sort_keys=True).encode()}[kind]
            with open(log, "ab") as f:
                f.write(line)
        elif kind == "repair" and mark is not None:
            os.truncate(log, mark)
            mark = None
        elif kind in ("replace", "swap"):  # a swap's record has its first open rewritten as a `wait`
            k = next((k for k, line in enumerate(lines) if b'"event": "open"' in line), None) if kind == "swap" else None
            tmp = state / "record.tmp"
            tmp.write_bytes(data if k is None else b"".join(lines[:k] + _waits(lines[k:k + 1], sid) + lines[k + 1:]))
            os.replace(tmp, log)
        elif kind == "rewrite":
            log.write_bytes(data)
        elif kind in ("cut", "edit", "halve") and mark is None and len(lines) > 4:
            kept = lines[:len(lines) // 2 if kind == "halve" else -3]
            log.write_bytes(b"".join(kept + (_waits(lines[-3:], sid) if kind == "edit" else [])))
        elif kind == "index-gone":
            shutil.rmtree(index, ignore_errors=True)
        elif kind in ("index-head", "index-entries") and index.exists():
            for p in index.iterdir():
                if (p.name == "head.json") == (kind == "index-head"):
                    p.write_bytes(b'{"v": ')

    for _ in range(steps):
        fw, r = rng.choice(fws), rng.random()
        sid = rng.choice(sids[-4:] + ["r9999", f"r{len(sids) - 1:04d}"] + ([long_sid] if long else []))
        part, stream = rng.choice(["L0", "L1", "alg", "L9"]), rng.choice(["s", "t"])
        if r < 0.04:
            kind = rng.choice(["blank", "torn", "unterminated", "foreign", "replace", "swap", "rewrite", "cut", "edit",
                               "halve", "index-gone", "index-head", "index-entries"])
            damage(kind, sid)
            out.append(("damage", kind))
        elif r < 0.07:
            damage("repair", sid)
            out.append(("damage", "repair"))
        else:
            op = rng.choice(["open", "open", "finish", "admit", "admit", "admit", "release", "release", "stop", "failures",
                             "stopped", "slot", "slot", "record", "appended"])
            budget = {"s": 4, "t": None}[stream] if rng.random() > 0.05 else 9
            parts = rng.choice(PARTS)
            kind = rng.choice(["open", "finish", "stop", "release", "refused", "wait", "slot"])
            ev = ({"event": "open", "session": sid, "stream": stream, "parts": parts, "budget": budget} if kind == "open"
                  else {"event": kind, "session": sid, "part": part})
            fn = {"open": lambda: fw.open(stream, parts, budget=budget),
                  "finish": lambda: fw.finish(sid), "admit": lambda: fw.admit(sid, part),
                  "release": lambda: fw.release(sid, part), "stop": lambda: fw.stop(sid, part, "cut"),
                  "failures": lambda: fw.failures(stream), "stopped": lambda: fw.stopped(sid), "record": fw.record,
                  "slot": lambda: _slot(fw, sid, part, rng.random()), "appended": lambda: _appended(state, ev)}[op]
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
    """On fixed streams of 1,500 ops (opens on two streams, one with a budget, finishes, admits, releases, stops, outer
    sessions in the slot and holders that end before their outcome, events another process appends) by three objects,
    between which the record and the index are damaged and repaired, every result, refusal and error, and the record
    after each op, are those of the ZK server that reads its whole record; the index answered most of the reads, the rest
    being those of a damaged record. With names past `KEY_MAX`, the same."""
    for seed, long in ((20261016, False), (7, False), (11, True)):
        got, (fast, reads) = _drive(FW.Firewall, tmp_path / f"new-{seed}", seed, 1500, long)
        want, _ = _drive(_Whole, tmp_path / f"whole-{seed}", seed, 1500, long)
        assert got == want, seed
        kinds = {c[0] if isinstance(c, tuple) else "digest" for c in got}
        raised = {c[2] for c in got if isinstance(c, tuple) and c[0] == "raise"}
        damaged = {c[1] for c in got if isinstance(c, tuple) and c[0] == "damage"}
        assert kinds == {"result", "raise", "damage", "digest"} and {"Refused", "JSONDecodeError", "KeyError"} <= raised
        assert damaged == {"blank", "torn", "unterminated", "foreign", "replace", "swap", "rewrite", "cut", "edit",
                           "halve", "index-gone", "index-head", "index-entries", "repair"}, seed
        assert long or fast * 2 > reads, (seed, fast, reads)


def test_a_rebuild_that_ends_mid_write_is_not_trusted(tmp_path, monkeypatch):
    """A writer of the index marks its head dirty before it touches a file of it, so one that ends mid-write leaves an
    index the next call rebuilds from the record, even once the record is back as the head from before read it: here the
    record is cut by its last line, a rebuild of the index ends after removing its files, and the record is restored."""
    state = tmp_path / "fw"
    fw = FW.Firewall(state)
    sid = fw.open("s", ["L0"], budget=5)
    assert fw.stop(sid, "L0", "cut") and fw.finish(sid) is None
    log, put = state / "record.jsonl", FW._put
    data = log.read_bytes()
    log.write_bytes(data[:data.rstrip(b"\n").rfind(b"\n") + 1])

    def ends(dir, name, v):
        if name != "head.json":
            raise OSError("the writer ended")
        put(dir, name, v)
    monkeypatch.setattr(FW, "_put", ends)
    assert fw.failures("s") == 1
    monkeypatch.setattr(FW, "_put", put)
    log.write_bytes(data)
    assert fw.failures("s") == 1 and fw.open("s", ["L0"], budget=5) == "r0001"


def test_what_a_call_reads_of_the_index_is_loaded_before_it_decides(tmp_path, monkeypatch):
    """A call loads every file of the index it may read or append for before it decides anything (`_Index.load`), so a
    file that won't read sends it to the whole record, never into an error after it appended: the file of the session an
    open names (which has events before its open), that of the session whose holder of the slot ended before its
    outcome, which the next to take the slot stops, and a stream's (here one no session it loads was opened on). The call
    after rebuilds the index, and the one after that reads nothing of the record. A session whose stream has no file in
    the index sends the call to the whole record too."""
    state = tmp_path / "fw"
    fw = FW.Firewall(state)
    sid = fw.open("s", ["L0", "L1"])
    fw.finish(sid)
    with fw.slot(sid, "L0"):
        pass
    _appended(state, {"event": "finish", "session": "r0001", "part": "inner"})
    assert fw.failures("s") == 0
    work = _Work(monkeypatch)

    def torn(name: str, kind: str = "s") -> None:
        (state / "index-py" / f"{kind}{name.encode().hex()}.json").write_bytes(b'{"v": ')

    def rebuilt(failures: int) -> None:
        for parsed in (_lines(state), 0):
            before = work.now()
            assert fw.failures("s") == failures
            after = work.now()
            assert (after[0] - before[0], after[4] - before[4]) == (parsed, 0)

    torn("r0001")
    whole = work.whole
    assert fw.open("s", ["L0"]) == "r0001" and work.whole == whole + 1
    rebuilt(0)
    torn(sid)
    with fw.slot("r0001", "L0"):
        assert work.whole == whole + 2
    rebuilt(1)
    assert [(e["event"], e["session"]) for e in fw.record()[-3:]] == [("open", "r0001"), ("stop", sid), ("slot", "r0001")]
    assert fw.open("u", ["L0"], budget=3) == "r0002"
    torn("u", "t")
    assert fw.failures("u") == 0 and work.whole == whole + 4
    rebuilt(1)
    (state / "index-py" / f"t{b'u'.hex()}.json").unlink()
    with pytest.raises(FW.Refused, match="has not finished"):
        fw.admit("r0002", "L0")
    assert work.whole == whole + 5


def test_a_record_edited_at_its_end_or_replaced_is_read_again(tmp_path):
    """The index reads the record again from its start when the record no longer holds the last bytes the index read
    (its last line, a stop, edited in place as a `wait` of its length), or when it was replaced (by a record of its length
    whose stop, further back than those bytes, is such a `wait`)."""
    for replaced in (False, True):
        state = tmp_path / f"fw-{replaced}"
        fw = FW.Firewall(state)
        sid = fw.open("s", ["L0"], budget=5)
        assert fw.stop(sid, "L0", "cut")
        if replaced:
            for k in range(8):
                _appended(state, {"event": "refused", "session": f"r{9000 + k}", "part": "L0"})
        log = state / "record.jsonl"
        lines = log.read_bytes().splitlines(keepends=True)
        assert fw.failures("s") == 1
        edited = b"".join(lines[:1] + _waits(lines[1:2], sid) + lines[2:])
        assert len(edited) == len(b"".join(lines)) and len(b"".join(lines[2:])) > (FW._LAST if replaced else -1)
        if replaced:
            (state / "record.tmp").write_bytes(edited)
            os.replace(state / "record.tmp", log)
        else:
            with open(log, "r+b") as f:
                f.write(edited)
        assert fw.failures("s") == 0 and fw.stopped(sid) is None
