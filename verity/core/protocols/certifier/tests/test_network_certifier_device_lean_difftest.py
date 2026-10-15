"""The difftest of `active.Proxy` against the Lean package `certifier_device`'s `step`: the Python half.

`verity/core/devices/certifier/lean/scripts/difftest_vectors.py` writes 1,500 traces and what this package's
real `active.Proxy` does on each, driven event by event in a world the script holds (every action in order, then the
declared syncs, hand times, frames read and counts). `check` runs the Lean half in its Lean audit (`runs` in that
package's `lean-audit.json`): `CertifierDevice/DifftestMain.lean` replays each trace through `step` and fails on any
action or result that differs. The audit checks the data against the record's SHA-256, and its pass is cached on the Lean
package's files alone, so the test here is what says the data is still what this package does: a change to the proxy, its
certifiers or its prover changes the data, fails this test, and `check.py --update` records the new data, which makes
`check` run the Lean half on it. The other test checks, on the first 300 traces, that the generator reaches what its
docstring says.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from collections import Counter
from pathlib import Path

from verity.core.protocols.certifier import active

LEAN = Path(__file__).resolve().parents[3] / "devices" / "certifier" / "lean"
SCRIPT = LEAN / "scripts" / "difftest_vectors.py"
_spec = importlib.util.spec_from_file_location("certifier_device_difftest_vectors", SCRIPT)
vectors = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)  # its dataclasses look themselves up there
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["CertifierDevice.DifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 1,500 traces and the proxy's actions on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]


def read_before_choice(inst: dict) -> bool:
    """Whether a response is read in bucket w·T − lag before that egress tick, which starts window w's choice: read by
    then, but ready at w·T − lag + 1, past what the choice reads."""
    c = inst["config"]
    T, tau, lag = c["egress"]["T"], c["egress"]["tau"] * vectors.MS, c["prover"]["lag"]
    ticks = [e[2] for e in inst["events"] if e[0] == "tick" and e[1] == "egress"]
    reads = [e[2] for e in inst["events"] if e[0] == "upstream" and e[3]]
    return any((b := (now - c["origin"]) // tau) >= 0 and (b + lag) % T == 0 and b < len(ticks) and now < ticks[b]
               for now in reads)


def test_the_generator_covers_what_it_says():
    insts = [vectors.instance(s) for s in range(300)]
    results = [vectors.python_results(i) for i in insts]
    outcomes, refused = Counter(), Counter()
    for inst, res in zip(insts, results, strict=True):
        statuses = {a[1]["status"] for a in res["actions"] if a[0] == "publish"} - {None}
        outcomes[inst["path"], "faulted" if statuses else "accepted"] += 1
        if inst["path"] == "duplicate-at-capacity":  # the duplicate is malformed, not the queue's overflow
            assert "malformed" in statuses
        if inst["program"] is not None:
            vectors.check_honest(inst, res)
            assert not statuses
        if inst["path"] == "early-tick":  # the prover's refusal, where the early tick can make one
            refused["monitor-unavailable" in statuses] += 1
    assert {p for p, _ in outcomes} == set(vectors.PATHS)
    assert refused[True] > refused[False]
    for path in set(vectors.PATHS) - {"honest", "prover", "link-garbage"}:
        assert outcomes[path, "faulted"] > outcomes[path, "accepted"], path
    late = [{a[1]["status"] for a in r["actions"] if a[0] == "publish"} - {None}
            for i, r in zip(insts, results, strict=True) if i["path"] == "slow-server"]
    assert all(s <= {"coverage-never-established"} for s in late)  # a frame handed after its h(k) fails closed
    reasons = {a[1]["status"] for r in results for a in r["actions"] if a[0] == "publish"}
    assert reasons == {None} | {r.value for r in vectors.Reason} - {"dropped"}
    assert all(r["actions"] for r in results)
    proved = [(i, r) for i, r in zip(insts, results) if i["config"]["prover"] is not None]
    lags = {(i["config"]["egress"]["T"], i["config"]["prover"]["lag"]) for i, _ in proved}
    assert {T for T, lag in lags if lag == T - 1} == {3, 4, 5} and {lag for _, lag in lags} == {1, 2, 3, 4}
    assert any(read_before_choice(i) for i, _ in proved)
    assert any(i["config"]["prover"]["prior"] for i, _ in proved)
    assert any(len({a[1] for a in r["actions"] if a[0] == "connect"}) >= 4 for r in results)  # several streams
    writes = [a for r in results for a in r["actions"] if a[0] == "upstream"]
    assert any(a[3] for a in writes) and any(not a[3] for a in writes)  # a request's LAST waits for its drain
    assert any(e[0] == "refused" for i in insts for e in i["events"])
    assert any(e[0] == "link" and e[2] == "" for i in insts for e in i["events"])  # the link's early end
    ticks = [(i, e) for i in insts for e in i["events"] if e[0] == "tick" and e[1] == "egress"]
    bucket = lambda i: i["config"]["egress"]["r"] * i["config"]["egress"]["B"]  # noqa: E731
    assert any(e[3] == bucket(i) for i, e in ticks) and any(e[3] == 4 * bucket(i) for i, e in ticks)
    assert any(e[3] > 4 * bucket(i) for i, e in ticks)
    assert any([f[0] for f in r["frames"]] != sorted(f[0] for f in r["frames"]) for r in results)  # read out of order


def test_the_proxys_shutdown_runs_on_the_harness_thread(monkeypatch):
    """A run with a prover ends in the proxy's executor shutdown (`asyncio.to_thread`). On a real thread it finishes
    when the host gets to it, possibly after `settle` has taken the loop as quiet, and the generator then raises "the
    trace leaves ticks unrun" under load (r20261010-073331-ef9b). The harness runs it inline, on the loop's thread."""
    threads = []
    monkeypatch.setattr(active.InlineExecutor, "shutdown", lambda self, *a, **k: threads.append(threading.get_ident()),
                        raising=False)
    inst = next(i for i in map(vectors.instance, range(len(vectors.PATHS))) if i["config"]["prover"] is not None)
    vectors.python_results(inst)
    assert threads == [threading.get_ident()]
