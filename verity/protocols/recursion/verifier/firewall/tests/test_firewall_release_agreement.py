"""The ZK server's release side in Lean (`core_firewall`: `Firewall.Record` and `Firewall.Release`, served by
`core-firewall-difftest`) against Python's (`verity_flock.firewall`) and against Rust's (`flock_live::release`) as it was
frozen in `fixtures/` before the Lean replaced it: the shared record vectors, every op of the record on sampled records against
Python (results and the record each leaves, byte for byte), Rust's index cases op by op, Rust's outer sessions, with each
step's failure and fault, the challenger's verdict and the clock sampled, and the record of an end-to-end run, both through
the stages `core-firewall session` runs (`decide`, then `stopAt` or `settle`). The records sampled, and the domain the Lean is
held to, are `_release`'s.

The fixtures were frozen from `flock_live::release` at e64d2e1238: `release-index.jsonl.gz` from its index tests'
streams (`release::tests::drive`, with each line its damage rewrites as a `wait` numbered as the event it replaces, and
each event its `appended` op writes as the ZK server writes it, which changes nothing Rust reads: an `open` without a
null `budget`, a `stop` with a `why`, a `finish` of part `inner`; so the record stays the ZK server's own),
`release-outer.jsonl.gz` from `outer_session` on `_release.outer_input`'s first 1000 samples at seed 0, and
`release-e2e.jsonl` from `flock-circuit prove --firewall-state` on rope_head(64)."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import random
import re
import shutil
from collections import Counter

import pytest

import _release as R
import verity_difftest as D
from verity.protocols.spot.verifier import service as S
from verity_flock import firewall as FW

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "core-firewall-difftest"
SEED = 0


def port(fn: str, reference, inputs) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=R.PKG)


lean = D.server_fixture(port("op", R.python_op, R.op_input()))


def _vectors() -> list[dict]:
    v = json.loads(R.VECTORS.read_text())
    assert v["kind"] == "firewall/v0/vectors" and {c["name"] for c in v["cases"]} == {
        "rules-1-2", "order", "budget", "unended", "honest", "schedule", "escapes", "registrations", "prestop",
        "prestop-registrations", "budget-1", "budget-2"}
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


def _honest(rounds: int, budget: int, n: int, rng: random.Random | None) -> tuple[list[list], list, int]:
    """An honest stream's ZK server ops and their results, over `n` registrations of a challenger's stream at `rounds` and
    `budget` (`service._step`, its rules), and the most recursive sessions unended at once. Each step is one the honest
    flow allows next, `rng`'s choice (the first allowed without one): a registration once `ready` allows it, sampled at once;
    a proof session (`session`) over the 1 to `rounds` earliest sampled rounds no session covers (one round without `rng`);
    its recursive session opened after it (`rec_live` opens one when the session's inner session starts) and finished;
    each statement admitted, then released (the relay releases it on the challenger's verdict); the session delivered once
    its last statement is released (the ZK server delivers the records of sessions it saw end); each round's outcome
    accepted, which closes its registration."""
    params = {"budget": budget, "rounds": rounds, "proof_deadline": 1e9, "registration_deadline": None, "stream": "h"}
    state = copy.deepcopy(S._EMPTY)

    def step(op: dict):
        nonlocal state
        d, state = S._step(params, state, op)
        return d

    ops, results, uncovered, sessions, opened, most = [], [], [], [], 0, 0

    def fw(op: list, result) -> None:
        ops.append(op)
        results.append(result)

    while state["size"] < n or uncovered or any(s["stage"] != "done" for s in sessions):
        moves = []
        if state["size"] < n and step({"op": "ready"}) == {"ok": None}:
            moves.append(("register", None))
        if uncovered:
            moves.append(("cover", None))
        moves += [(s["stage"], s) for s in sessions if s["stage"] != "done"]
        move, s = moves[0] if rng is None else rng.choice(moves)
        if move == "register":
            k = state["size"]
            assert step({"op": "register", "checks": None, "registration": f"g{k}"}) == {"ok": k}
            assert step({"op": "select", "order": k, "registration": f"g{k}", "checks": None, "draw": f"d{k}", "proved": 0,
                         "at": 0.0}) == {"ok": None}
            uncovered.append([k, f"g{k}", f"d{k}"])
        elif move == "cover":
            m = 1 if rng is None else rng.randint(1, min(rounds, len(uncovered)))
            draws, uncovered[:m] = uncovered[:m], []
            assert step({"op": "session", "draws": draws}) == {"ok": [d[0] for d in draws]}
            sessions.append({"draws": draws, "stage": "open", "sid": None, "next": 0})
        elif move == "open":
            s["sid"], opened = f"r{opened:04d}", opened + 1
            fw(["open", "h", list(R.NAMES), budget, None], s["sid"])
            s["stage"] = "finish"
            most = max(most, sum(t["stage"] in ("finish", "admit", "release") for t in sessions))
        elif move == "finish":
            fw(["finish", s["sid"]], None)
            s["stage"] = "admit"
        elif move == "admit":
            fw(["admit", s["sid"], R.NAMES[s["next"]]], {"release_ms": None})
            s["stage"] = "release"
        elif move == "release":
            fw(["release", s["sid"], R.NAMES[s["next"]]], None)
            s["next"] += 1
            s["stage"] = "admit" if s["next"] < len(R.NAMES) else "deliver"
        elif move == "deliver":
            assert step({"op": "deliver_session", "stream": "h", "draws": s["draws"], "sessions": ["0" * 128], "now": 0.0}) == {"ok": None}
            s["stage"] = "outcome"
        else:
            for o, g, d in s["draws"]:
                assert step({"op": "outcome", "order": o, "registration": g, "draw": d, "verdicts": None, "accepted": True,
                             "now": 0.0}) == {"ok": S.ACCEPT}
            s["stage"] = "done"
    assert state["failures"] == 0 and state["opened"] == []
    fw(["failures", "h"], 0)
    return ops, results, most


@pytest.mark.parametrize("rounds", range(1, S.ROUNDS + 1))
def test_an_honest_stream_is_never_refused_an_open(lean, rounds):
    """Completeness of the budget's open rule: an honest stream at each `rounds` R (1 to `ROUNDS`) and budget b (1 to 4,
    from `BUDGET_MIN`), under the challenger's stream's rules, is refused nothing by the Lean and Python ZK servers
    alike, which give the same results and record at every op. The challenger holds at most min(R, b − failures)
    registrations open, each unended recursive session covers one of them, and a session is released whole before its
    rounds' outcomes close them; so at an open its stream has fewer than min(R, b) unended. Without `rng` each session
    covers one round and is opened as soon as it can be, so min(R, b) recursive sessions are unended at once: the most,
    which the rule allows."""
    for budget in range(FW.BUDGET_MIN, 5):
        most = 0
        for rng in (None, *(random.Random(seed) for seed in range(3))):
            ops, results, m = _honest(rounds, budget, 3 * rounds + 2, rng)
            text = ""
            for op, want in zip(ops, results):
                got = _lean_op(lean)({"op": op, "record": text})
                assert got == R.python_op({"op": op, "record": text}) and got["result"] == want, (rounds, budget, op, got)
                text = got["record"]
            most = max(most, m)
        assert most == min(rounds, budget), (rounds, budget, most)


def _result(out: dict) -> str:
    r = out["result"]
    return next(iter(r)) if isinstance(r, dict) and set(r) != {"release_ms"} else "ok"


def _tallied(reference, tally: Counter, key):
    def ref(value):
        out = reference(value)
        tally[key(value, out)] += 1
        return out

    return ref


def _op_key(value: dict, out: dict) -> str:
    """The op and its result's kind, an open refused under the unended rule (`_unended`) apart from the other refusals."""
    return f"{value['op'][0]}:{'unended' if _unended(out['result']) else _result(out)}"


def _assert_ops_seen(tally: Counter) -> None:
    ops = {k.split(":")[0] for k in tally}
    assert ops == {"open", "finish", "stop", "release", "admit", "failures", "slot", "register", "registrations"}, tally
    for k in ("open:ok", "open:refused", "open:error", "admit:ok", "admit:refused", "registrations:ok",
              "registrations:refused"):
        assert tally[k] >= 10, (k, tally)
    assert tally["open:unended"] >= 5, tally


@pytest.mark.slow
def test_every_op_agrees_with_python(lean):
    tally: Counter = Counter()
    ref = _tallied(R.python_op, tally, _op_key)
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
#: the refusal of an open whose stream's unended recursive sessions and failures reach its budget, a rule Rust's lacked
UNENDED = re.compile(r"(?s)stream .*'s unended recursive sessions and failures reach its failure budget \(\d+ and \d+ of \d+\)")


def _unended(result) -> bool:
    return isinstance(result, dict) and UNENDED.fullmatch(str(result.get("refused"))) is not None


def _events(text: str) -> list:
    out = []
    for x in text.split("\n"):
        if x:
            try:
                out.append(json.loads(x))
            except ValueError:
                out.append(x)
    return out


def _prestop(before: str, rust: str, lean: str) -> bool:
    """Whether Rust's record `rust` is the Lean's `lean` with one stop more, of a recursive session `before` never opened,
    the lines after it numbered one later: a stop by `stop` or by the slot's orphan rule, which the Lean records only of a
    recursive session opened, a rule Rust's lacked."""
    opened = {e.get("session") for e in _events(before) if isinstance(e, dict) and e.get("event") == "open"}
    r = _events(rust)
    for i, e in enumerate(r):
        if isinstance(e, dict) and e.get("event") == "stop" and e.get("session") not in opened:
            after = [{**x, "seq": x["seq"] - 1} if isinstance(x, dict) and "seq" in x else x for x in r[i + 1:]]
            if r[:i] + after == _events(lean):
                return True
    return False


def test_the_lean_runs_rusts_index_cases(lean):
    """Rust's index cases, op by op: its three streams (seeds 20261016 and 7, and 11 with a session id past the index's key
    limit), 1500 samples each, an op or damage to the record (torn, cut, halved, edited, rewritten, a line foreign to it), a
    repair or an append, which leave 3979 ops. Of those, Rust rejected 762, and the Lean rejects each; Rust answered 3217,
    and at 2775 of them the Lean's result and the record it leaves are Rust's byte for byte. At 8, opens Rust made, the
    Lean refuses the open and leaves the record as it was, as Python does: the stream's unended recursive sessions and
    failures reach its budget, a rule Rust's lacked. At 47, 36 stops and 11 slots, Rust recorded a stop of a recursive
    session never opened, by `stop` or by the slot's orphan rule, and the Lean records none, as Python does: a stop is
    recorded only of a recursive session opened, a rule Rust's lacked. The other 387 are outside the domain: their record
    holds a line the ZK server doesn't write (`foreign`), which Rust read past and the Lean refuses the record for
    (`test_the_lean_reads_only_the_firewalls_lines`). A step's `edit` is the record before it as an edit `[AT, CUT, PUT]`
    of the one the step before left (CUT bytes at byte AT replaced by PUT), its `out` the record it leaves as an edit of the
    one before it, and `after` that record's SHA-256."""
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
            if step["op"][0] == "open" and isinstance(step["result"], str) and _unended(got["result"]):
                assert got["record"] == before and R.python_op({"op": step["op"], "record": before}) == got, (n, step, got)
                tally["unended"] += 1
                continue
            if step["op"][0] in ("stop", "slot") and (got["result"], got["record"]) != (step["result"], text):
                assert _prestop(before, text, got["record"]), (n, step, got)
                assert got["result"] == (False if step["op"][0] == "stop" else step["result"]), (n, step, got)
                assert R.python_op({"op": step["op"], "record": before}) == got, (n, step, got)
                tally[f"{step['op'][0]}:prestop"] += 1
                continue
            assert (got["result"], got["record"]) == (step["result"], text), (n, step, got)
        tally[f"{step['op'][0]}:{'reject' if 'reject' in step else _result({'result': step['result']})}"] += 1
    print(dict(sorted(tally.items())))
    assert dict(tally) == {
        "admit:ok": 7, "admit:refused": 925, "admit:reject": 195, "failures:ok": 202, "failures:reject": 52,
        "finish:ok": 206, "finish:reject": 63, "open:error": 32, "open:ok": 204, "open:refused": 184, "open:reject": 126,
        "outside": 387, "release:ok": 434, "release:reject": 115, "slot:ok": 407, "slot:prestop": 11, "slot:reject": 147,
        "stop:ok": 174, "stop:prestop": 36, "stop:reject": 64, "unended": 8}, tally


def test_the_loops_stages_run_rusts_outer_sessions(lean):
    """Rust's `outer_session` on a fake clock, against the stages the loop runs: the outcome, why, the release time it
    waited for and the record it leaves, or a rejection where Rust's was an error. A release, a refusal, and a stop of
    each kind are all seen. Where Rust's slot recorded its orphan's stop of a recursive session never opened, the Lean's
    record lacks that one stop (`_prestop`), and all else is Rust's."""
    tally: Counter = Counter()
    for n, case in enumerate(_frozen("release-outer.jsonl.gz")):
        kind, out = lean.call("outer", case["input"])
        if "reject" in case:
            assert kind == "reject", (n, case, out)
            continue
        assert kind == "output", (n, case, out)
        got = json.loads(out)
        if got != case["output"]:
            assert {**got, "record": None} == {**case["output"], "record": None}, (n, case, got)
            assert _prestop(case["input"]["record"], case["output"]["record"], got["record"]), (n, case, got)
            tally["prestop"] += 1
        tally[_outcome(case["input"], case["output"])] += 1
    print(dict(sorted(tally.items())))
    for k in ("release", "refused", "stop", "stop:fault", "stop:send check", "stop:missed", "stop:verdict", "prestop"):
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


def test_the_lean_reads_registrations_only_in_their_order(lean):
    """The Lean reads each registration only at its place among the registrations, its record that number first, and each
    admission or skip only at its place among them, below the registrations before it: a record admitting one twice, out of
    its order or before it is registered, or holding a registration out of its place, is refused, so no registration is
    admitted twice. Python reads each as its events."""
    good = R.record_of([["open", "s", ["L0"], None, None], ["finish", "r0000"], ["register", "aa"], ["register", "bb"],
                        ["registrations", "r0000", "L0", [0]]])
    lines = good.splitlines(keepends=True)

    def line(seq: int, **ev) -> str:
        return json.dumps({"seq": seq, **ev}, sort_keys=True) + "\n"

    admit = {"event": "admit", "part": "L0", "session": "r0000"}
    variants = {"admitted twice": good + line(5, **admit, n=0), "out of its order": "".join(lines[:4]) + line(4, **admit, n=1),
                "before its registration": "".join(lines[:2]) + line(2, **admit, n=0),
                "registration out of its place": "".join(lines[:2]) + line(2, event="register", n=1, record="0000000000000001bb"),
                "record numbered otherwise": "".join(lines[:2]) + line(2, event="register", n=0, record="0000000000000001aa")}
    for name, text in variants.items():
        value = {"op": ["failures", "s"], "record": text}
        assert lean.call("op", value)[0] == "reject", name
        assert R.python_op(value)["result"] == 0, name
    assert _lean_op(lean)({"op": ["failures", "s"], "record": good})["result"] == 0
