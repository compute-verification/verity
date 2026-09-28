"""The difftest of the Python audit against the certifier's Lean program: the Python half.

`lean/scripts/difftest_vectors.py` writes 10,000 random instances and this package's results on each
(`schedule.expected_grids`, `Audit.check_link` with the first differing bucket, `Audit.check_ingress_link`, and each
prefix's acceptance). `check` runs the Lean half in its Lean audit (`runs` in `lean/lean-audit.json`):
`lean/NetworkCertifier/DifftestMain.lean` evaluates every instance with the program's definitions (`NetworkCertifier`: `expectedRow`,
and `Accepted` of `constantRate` and `StatusComplete` by `decide`) and fails on any difference, and `security_proofs`
proves those definitions equal to `NetTiming`'s, which the guarantees read. The audit checks that data against the
record's SHA-256, and its pass is cached on the Lean package's files alone, so the test here is what says the data is
still what this package computes: a change to the audit's verdicts changes the data, fails this test, and
`check.py --update` records the new data, which makes `check` run the Lean half on it. The tests here also check, on the
first 200 instances, that the generator reaches what its docstring says, and that Python rejects what the generator
never emits.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from verity.core.protocols.certifier.grid import released
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, handing_buckets

LEAN = Path(__file__).resolve().parents[1] / "lean"
SCRIPT = LEAN / "scripts" / "difftest_vectors.py"
_spec = importlib.util.spec_from_file_location("difftest_vectors", SCRIPT)
vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vectors)
RUN = json.loads((LEAN / "lean-audit.json").read_text())["runs"]["NetworkCertifier.DifftestMain"]


def test_the_data_is_what_the_audit_pins(tmp_path):
    """All 10,000 instances and the Python audit's results on them, byte for byte as the record pins them."""
    made = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in vectors.write(tmp_path).items()}
    assert made == RUN["data"]
    assert RUN["args"] == ["{difftest/instances.jsonl}", "{difftest/python.jsonl}", str(vectors.COUNT)]


def test_the_generator_covers_what_it_says():
    insts = [vectors.instance(s) for s in range(200)]
    frames = [(i, f) for i in insts for fs in i["frames"].values() for f in fs]
    assert min(f.index for _, f in frames) == 0 and min(f.anchor for _, f in frames) == 0
    assert any(f.anchor >= i["params"].buckets * i["windows"] for i, f in frames)
    assert {len(i["egress"]) for i in insts} == {1, 2, 3} and {len(i["ingress"]) for i in insts} == {0, 1, 2}
    assert {i["params"].buckets for i in insts} == {1, 2, 3, 4, 5} and {i["windows"] for i in insts} == {1, 2, 3, 4}
    assert {i["params"].allocation for i in insts} == {1, 2, 3}
    assert any(s.rho.denominator > 1 for i in insts for ss in i["declared"].values() for s in ss)
    assert any(s not in i["sync_set"] for i in insts for ss in i["declared"].values() for s in ss)
    assert any(len(set(map(tuple, i["declared"].values()))) > 1 for i in insts)
    assert any(not r.status.complete for i in insts for r in i["records"])
    assert any(i["recorded"] < i["windows"] for i in insts)
    results = [vectors.python_results(i) for i in insts]
    verdicts = {v for r in results for vs in r["verdicts"] for v in vs}
    assert {None, "advice", "mismatch", "no-verdict"} <= verdicts
    assert any(v and v.startswith("status:") for v in verdicts)
    assert any(r["accepted"][-1] for r in results)
    assert any(any(r["accepted"][1:]) and not r["accepted"][-1] for r in results)
    assert any(_carried(i, link) for i in insts for link in i["egress"])


def _carried(inst: dict, link: str) -> bool:
    """More frames handed in one bucket than the allocation, and some still queued when a window closes."""
    p = inst["params"]
    horizon = p.buckets * inst["windows"]
    handed = [h for h in handing_buckets(inst["frames"][link], inst["declared"][link], p.buckets) if h < horizon]
    enqueued = [handed.count(t) for t in range(horizon)]
    queue = [sum(enqueued[: t + 1]) - sum(released(p.allocation, enqueued)[: t + 1]) for t in range(horizon)]
    closing = range(p.buckets - 1, horizon - 1, p.buckets)
    return max(enqueued, default=0) > p.allocation and any(queue[t] for t in closing)


def test_python_rejects_what_lean_cannot_represent():
    """Lean's anchors, indices and δ are ℕ, so the generator emits none below 0, and Python rejects each."""
    with pytest.raises(ValueError, match="logical index is non-negative"):
        LogicalFrame(b"x", 0, -1)
    with pytest.raises(ValueError, match="anchors are non-negative"):
        handing_buckets([LogicalFrame(b"x", -1, 0)], [ClockSync(0, 1)], 2)
    with pytest.raises(ValueError, match="delta ≥ 0 and rho > 0"):
        ClockSync(-1, 1)
