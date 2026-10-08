"""The firewall's release side in Lean (`core_firewall`: `Firewall.Record` and `Firewall.Release`, served by
`core-firewall-difftest`) against Rust's (`flock_live::release`, compiled by `tests/rust`) and Python's
(`verity_flock.firewall`): the shared record vectors replayed by all three, then every op of the record on drawn records
against both (results and the record each leaves, byte for byte), and the outer session against Rust's `outer_session`,
with each step's failure and panic, the challenger's verdict and the clock drawn. The records drawn, and the domain the
three are held to, are `_release`'s."""

from __future__ import annotations

import json
import shutil
from collections import Counter

import pytest

import _release as R
import verity_difftest as D

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
needs_cargo = pytest.mark.skipif(shutil.which("cargo") is None, reason="no Rust toolchain")

EXE = "core-firewall-difftest"
SEED = 0


def port(fn: str, reference, inputs) -> D.Port:
    return D.Port(EXE, fn, reference, inputs, package=R.PKG)


lean = D.server_fixture(port("op", R.python_op, R.op_input()))


@pytest.fixture(scope="module")
def rust(tmp_path_factory):
    with D.Server([str(R.rust_reference()), str(tmp_path_factory.mktemp("rust-state"))]) as s:
        yield s


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
    `flock_live::release`'s `the_shared_vectors_hold` and `test_firewall_release` hold Rust and Python to."""
    for case in _vectors():
        assert _replay(_lean_op(lean), case) == (case["results"], case["record"]), case["name"]


@needs_cargo
def test_the_references_replay_the_shared_vectors(rust):
    """The two references as this suite drives them (`tests/rust`'s driver, `_release.python_op`) give the vectors too."""
    for case in _vectors():
        assert _replay(R.through(rust, "op"), case) == (case["results"], case["record"]), case["name"]
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


@needs_cargo
@pytest.mark.slow
def test_every_op_agrees_with_rust(lean, rust):
    tally: Counter = Counter()
    ref = _tallied(R.through(rust, "op"), tally, lambda v, out: f"{v['op'][0]}:{_result(out)}")
    stats = D.run(port("op", ref, R.op_input()), examples=1500, seed=SEED, server=lean)
    print(f"op vs Rust, seed {SEED}: {stats}, {dict(sorted(tally.items()))}")
    assert stats.rejects >= 50, stats
    _assert_ops_seen(tally)


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


@needs_cargo
@pytest.mark.slow
def test_the_outer_session_agrees_with_rust(lean, rust):
    """`outerSession` against `outer_session` on a fake clock: the outcome, why, the release time it waited for and the
    record it leaves; a release, a refusal, and a stop of each kind are all seen."""
    tally: Counter = Counter()
    ref = _tallied(R.through(rust, "outer"), tally, _outcome)
    stats = D.run(port("outer", ref, R.outer_input()), examples=1000, seed=SEED, server=lean)
    print(f"outer vs Rust, seed {SEED}: {stats}, {dict(sorted(tally.items()))}")
    for k in ("release", "refused", "stop", "stop:fault", "stop:send check", "stop:missed", "stop:verdict"):
        assert tally[k] >= 10, (k, tally)


def test_the_lean_reads_only_the_firewalls_lines(lean):
    """A line of the record that is JSON but not the line the firewall writes for its event (not `json.dumps(ev,
    sort_keys=True)`, or its `seq` not its place) is refused by the Lean; Python reads it as the event. Only the firewall
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
