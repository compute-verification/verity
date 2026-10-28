"""The ZK server's release side in Lean (`core_firewall`: `Firewall.Record` and `Firewall.Release`, served by
`core-firewall-difftest`) against Python's (`verity_flock.firewall`) and against Rust's (`flock_live::release`) as it was
frozen in `fixtures/` before the Lean replaced it: the shared record vectors, every op of the record on drawn records against
Python (results and the record each leaves, byte for byte), Rust's index cases op by op, Rust's outer sessions, with each
step's failure and fault, the challenger's verdict and the clock drawn, and the record of an end-to-end run, both through
the stages `core-firewall session` runs (`decide`, then `stopAt` or `settle`). The records drawn, and the domain the Lean is
held to, are `_release`'s.

The fixtures were frozen from `flock_live::release` at e64d2e1238: `release-index.jsonl.gz` from its index tests'
streams (`release::tests::drive`, with each line its damage rewrites as a `wait` numbered as the event it replaces, and
each event its `appended` op writes as the ZK server writes it, which changes nothing Rust reads: an `open` without a
null `budget`, a `stop` with a `why`, a `finish` of part `inner`; so the record stays the ZK server's own),
`release-outer.jsonl.gz` from `outer_session` on `_release.outer_input`'s first 1000 draws at seed 0, and
`release-e2e.jsonl` from `flock-circuit prove --firewall-state` on rope_head(64) (run r20261021-042230-834a)."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from collections import Counter

import pytest

import _release as R
import verity_difftest as D

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "core-firewall-difftest"
SEED = 0


def port(fn: str, reference, inputs) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=R.PKG)


lean = D.server_fixture(port("op", R.python_op, R.op_input()))


def _vectors() -> list[dict]:
    v = json.loads(R.VECTORS.read_text())
    assert v["kind"] == "firewall/v0/vectors" and {c["name"] for c in v["cases"]} == {"rules-1-2", "order", "budget", "schedule", "escapes"}
    return v["cases"]


def _replay(op, case: dict) -> tuple[list, str]:
    text, results = "", []
    for o in case["ops"]:
        out = op({"op": o, "record": text})
        text = out["record"]
        results.append(out["result"])
    return results, text


def _lean_op(server: D.Server):
    def op(value):
        kind, text = server.call("op", value)
        assert kind == "output", text
        return json.loads(text)

    return op


def test_the_lean_replays_the_shared_vectors(lean):
    """Each case of `firewall_record_vectors.json`, its ops run one by one from an empty record: the results and the record
    `test_firewall_release` holds Python to, which Python as this suite drives it (`_release.python_op`) gives too."""
    for case in _vectors():
        assert _replay(_lean_op(lean), case) == (case["results"], case["record"]), case["name"]
        assert _replay(R.python_op, case) == (case["results"], case["record"]), case["name"]


def _result(out: dict) -> str:
    r = out["result"]
    return next(iter(r)) if isinstance(r, dict) and set(r) != {"release_ms"} else "ok"


def _tallied(reference, tally: Counter, key):
    def ref(value):
        out = reference(value)
        tally[key(value, out)] += 1
        return out

    return ref


def _assert_ops_seen(tally: Counter) -> None:
    ops = {k.split(":")[0] for k in tally}
    assert ops == {"open", "finish", "stop", "release", "admit", "failures", "slot"}, tally
    for k in ("open:ok", "open:refused", "open:error", "admit:ok", "admit:refused"):
        assert tally[k] >= 10, (k, tally)


@pytest.mark.slow
def test_every_op_agrees_with_python(lean):
    tally: Counter = Counter()
    ref = _tallied(R.python_op, tally, lambda v, out: f"{v['op'][0]}:{_result(out)}")
    stats = D.run(port("op", ref, R.op_input()), examples=1500, seed=SEED, server=lean)
    print(f"op vs Python, seed {SEED}: {stats}, {dict(sorted(tally.items()))}")
    assert stats.rejects >= 50, stats
    _assert_ops_seen(tally)


def _outcome(value: dict, out: dict) -> str:
    why = out["why"] or ""
    kinds = {"the firewall's fault": "fault", "the send check refused": "send check", "missed its release time": "missed",
             "the challenger's verdict": "verdict"}
    return out["outcome"] + next((f":{v}" for k, v in kinds.items() if why.startswith(k)), "")


def _frozen(name: str) -> list[dict]:
    with gzip.open(R.FIXTURES / name, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _apply(text: str, edit: list) -> str:
    at, cut, put = edit
    b = text.encode()
    return (b[:at] + put.encode() + b[at + cut:]).decode()


#: why the Lean refuses a record holding a line the ZK server doesn't write (`Record.read`), with the line's number, and the
#: one such line Rust's `foreign` damage writes
FOREIGN = re.compile(r"the record's line (\d+) is not (an event of firewall/v0|the firewall's own)")
FOREIGN_LINE = '{"event": "note", "seq": 7}\n'


def test_the_lean_runs_rusts_index_cases(lean):
    """Rust's index cases, op by op: its three streams (seeds 20261016 and 7, and 11 with a session id past the index's key
    limit), 1500 draws each, an op or damage to the record (torn, cut, halved, edited, rewritten, a line foreign to it), a
    repair or an append, which leave 3979 ops. Of those, Rust rejected 762, and the Lean rejects each; Rust answered 3217,
    and at 2830 of them the Lean's result and the record it leaves are Rust's byte for byte. The other 387 are outside the
    domain: their record holds a line the ZK server doesn't write (`foreign`), which Rust read past and the Lean refuses the
    record for (`test_the_lean_reads_only_the_firewalls_lines`). A step's `edit` is the record before it as an edit `[AT, CUT, PUT]` of the one the step before left (CUT bytes
    at byte AT replaced by PUT), its `out` the record it leaves as an edit of the one before it, and `after` that record's
    SHA-256."""
    tally: Counter = Counter()
    text = ""
    for n, step in enumerate(_frozen("release-index.jsonl.gz")):
        if "case" in step:
            text = ""
            continue
        before = _apply(text, step["edit"]) if "edit" in step else text
        text = _apply(before, step["out"]) if "out" in step else before
        assert hashlib.sha256(text.encode()).hexdigest() == step["after"], (n, step)
        kind, out = lean.call("op", {"op": step["op"], "record": before})
        if "reject" in step:
            assert kind == "reject", (n, step, out)
        elif kind == "reject" and (m := FOREIGN.match(out)):
            lines = [x + "\n" for x in before.split("\n") if x]
            assert lines[int(m.group(1)) - 1] == FOREIGN_LINE, (n, step, out)
            tally["outside"] += 1
            continue
        else:
            assert kind == "output", (n, step, out)
            got = json.loads(out)
            assert (got["result"], got["record"]) == (step["result"], text), (n, step, got)
        tally[f"{step['op'][0]}:{'reject' if 'reject' in step else _result({'result': step['result']})}"] += 1
    print(dict(sorted(tally.items())))
    assert dict(tally) == {
        "admit:ok": 7, "admit:refused": 925, "admit:reject": 195, "failures:ok": 202, "failures:reject": 52,
        "finish:ok": 206, "finish:reject": 63, "open:error": 32, "open:ok": 212, "open:refused": 184, "open:reject": 126,
        "outside": 387, "release:ok": 434, "release:reject": 115, "slot:ok": 418, "slot:reject": 147, "stop:ok": 210,
        "stop:reject": 64}, tally


def test_the_loops_stages_run_rusts_outer_sessions(lean):
    """Rust's `outer_session` on a fake clock, against the stages the loop runs: the outcome, why, the release time it
    waited for and the record it leaves, or a rejection where Rust's was an error. A release, a refusal, and a stop of
    each kind are all seen."""
    tally: Counter = Counter()
    for n, case in enumerate(_frozen("release-outer.jsonl.gz")):
        kind, out = lean.call("outer", case["input"])
        if "reject" in case:
            assert kind == "reject", (n, case, out)
            continue
        assert kind == "output" and json.loads(out) == case["output"], (n, case, out)
        tally[_outcome(case["input"], case["output"])] += 1
    print(dict(sorted(tally.items())))
    for k in ("release", "refused", "stop", "stop:fault", "stop:send check", "stop:missed", "stop:verdict"):
        assert tally[k] >= 10, (k, tally)


def test_the_loops_stages_write_the_end_to_end_record(lean):
    """The end-to-end run's six outer sessions in turn, each proved and accepted, through the stages the loop runs: an
    immediate release, a release at its scheduled time (the clock read at it), a stop at a statement never opened and two
    refusals, and a missed release time (the clock read when the run missed it). The record they leave is the run's, byte
    for byte."""
    lines = R.e2e_record()
    text = "".join(lines[:7])
    sched = json.loads(lines[2])["release"]
    missed = int(re.search(r"missed its release time \((\d+) ms", lines[-1])[1])
    now = {1: sched["start_ms"] + sched["windows_ms"][0], 5: missed}
    got = []
    for k, (s, p) in enumerate(R.E2E):
        value = {"now_ms": now.get(k, 0), "part": p, "prove": "ok", "ready": "ok", "record": text, "send": True, "session": s}
        kind, res = lean.call("outer", value)
        assert kind == "output", res
        res = json.loads(res)
        text = res["record"]
        got.append((res["outcome"], res["sent_at"]))
    assert got == [("release", None), ("release", now[1]), ("refused", None), ("refused", None), ("refused", None),
                   ("stop", None)]
    assert text == "".join(lines)


def test_the_lean_reads_only_the_firewalls_lines(lean):
    """A line of the record that is JSON but not the line the ZK server writes for its event (not `json.dumps(ev,
    sort_keys=True)`, or its `seq` not its place) is refused by the Lean; Python reads it as the event. Only the ZK server
    writes its record, so this is outside the domain the two are compared on."""
    first, second = R.record_of([["open", "s", ["L0"], None, None], ["finish", "r0000"]]).splitlines()
    ev = json.loads(second)
    variants = {"compact": json.dumps(ev, sort_keys=True, separators=(",", ":")), "unsorted": json.dumps(dict(reversed(ev.items()))),
                "seq": json.dumps(ev | {"seq": 5}, sort_keys=True), "trailing space": second + " "}
    for name, line in variants.items():
        value = {"op": ["failures", "s"], "record": f"{first}\n{line}\n"}
        assert lean.call("op", value)[0] == "reject", name
        assert R.python_op(value)["result"] == 0, name
    assert _lean_op(lean)({"op": ["failures", "s"], "record": f"{first}\n{second}\n"})["result"] == 0
