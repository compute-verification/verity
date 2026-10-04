"""Differential tests: `verity_pous.erasure.ErasureVerifier` against the Lean model's secure-erasure accept rule over
distinct reads of the page fill, the game `Pous.Guarantees.SecureErasurePagesFamily` and `SecureErasurePages96112` are
about: `Pous.eraseWinsDraw` with `Pous.distinctReads` (`lean/Pous/Protocol/Model/ReadDraw.lean`) on the fill of
`Pous.PageFill` (`lean/Pous/Protocol/Model/PageFill.lean`), evaluated by Lean (`tests/lean/PousErasureCheck.lean`) with
the fill check's digest `Pous.Digest.digest` (`lean/Pous/Protocol/Model/Digest.lean`).

`pous_erasure_cases.py` draws 12,000 cases, half of them whole pages, each a prover's script for the fill and every
read: right, wrong (a flipped or cut digest, a flipped byte, a byte off by a multiple of 256) or no answer, after a
delay at, one nanosecond past, or anywhere around the deadline, over a transport that gives up at the deadline or
delivers late answers anyway; and a caller that reveals each read at once, up to, exactly at, or past `T_round` after the last reveal.
Lean evaluated `eraseWinsDraw` on every case once: it builds the fill from each page's tweak and bytes with
`PageFill.tweak`, `page` and `byte`, so a page reaches the fill only if `KeyedFill.tweak` is Lean's `tweak K p`; it
refuses a case whose indices repeat (`distinctReads` is an embedding); and it computes the digest the fill must have
itself.  `fixtures/lean_erasure_verdicts.json` pins the verdicts with the digests of the cases (the prover's digest and
bytes among them, from `KeyedFill`) and of the Lean files that define the rule, so a change to either fails here until
the fixture is regenerated (`uv run python tests/pous_erasure_cases.py`, about a minute).  The verifier runs on every
case here on an injected clock, through `run_erasure` when the caller never waits, and must reach Lean's verdict on
each: on the accepted cases that also says `KeyedFill`'s pages, indexing and digest are `PageFill`'s and
`Pous.Digest.digest`.  Each mutant of `erasure.py`'s accept decisions must reach a different verdict on some case.
`VERITY_POUS_LEAN=1` rebuilds the Lean modules and reruns Lean on the cases.

Lean's game has no clock, so the check reads a missed deadline (the fill's, a read's, or a reveal's `T_round`) as an
answer no digest or block is; that reading is the one rule the check writes itself.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
from collections import Counter

import pous_erasure_cases as cases_mod
import pytest
from pous_erasure_cases import (
    DROP,
    WRONG,
    Case,
    cases_digest,
    lean_files_digests,
    line,
    reveal_gap,
    run,
    truth,
)
from verity_pous import erasure

FIXTURE = json.loads(cases_mod.FIXTURE.read_text())
REGENERATE = "regenerate Lean's verdicts with `uv run python tests/pous_erasure_cases.py`"


@pytest.fixture(scope="module")
def cases() -> list[Case]:
    return cases_mod.cases(FIXTURE["count"], FIXTURE["seed"])


def test_the_verdicts_are_lean_s_on_these_cases(cases):
    """The pinned verdicts are Lean's on exactly these cases, from exactly these Lean files: at least 10,000 of them,
    both verdicts well represented."""
    assert FIXTURE["count"] >= 10_000 and len(FIXTURE["verdicts"]) == FIXTURE["count"] == len(cases)
    assert cases_digest([line(c) for c in cases]) == FIXTURE["cases_sha256"], f"the cases changed: {REGENERATE}"
    assert lean_files_digests() == FIXTURE["lean_files"], f"the Lean rule changed: {REGENERATE}"
    accepted = FIXTURE["verdicts"].count("1")
    assert 0.2 * len(cases) < accepted < 0.8 * len(cases)


def _fault(case: Case) -> str | None:
    """The one step that fails in `case`, as `fill|read:kind`, or None for no fault or more than one."""
    faults = []
    for j, (name, step, deadline) in enumerate([("fill", case.fill, case.fill_deadline),
                                                *(("read", r, case.read_deadline) for r in case.reads)]):
        gap = reveal_gap(case, j - 1) if j > 1 else 0
        if gap > case.round_deadline:
            faults.append(f"read:lapse{'+1' if gap == case.round_deadline + 1 else ''}")
        if step.kind == DROP:
            faults.append(f"{name}:drop")
        elif step.delay > deadline:
            edge = "+1" if step.delay == deadline + 1 else ""
            faults.append(f"{name}:late{edge}{':eager' if case.eager else ''}")
        elif step.kind == WRONG:
            faults.append(f"{name}:wrong-{step.wrong if name == 'fill' else 'mod256' if step.wrong > 255 else 'xor'}")
    return faults[0] if len(faults) == 1 else None


REQUIRED = {"fill:drop", "fill:late+1", "fill:late+1:eager", "fill:late:eager", "fill:wrong-1", "fill:wrong-2",
            "fill:wrong-3", "fill:wrong-4", "read:drop", "read:late+1", "read:late+1:eager", "read:late:eager",
            "read:wrong-xor", "read:wrong-mod256", "read:lapse+1", "read:lapse"}
"""The single faults the cases must reach, each rejected by Lean at least 20 times."""


def test_the_verifier_reaches_lean_s_verdict_on_every_case(cases):
    """No disagreement on any of the cases, and the verifier reveals the indices `spot_indices` draws, in order, until
    it stops.  The cases reach each single fault of `REQUIRED`; among the accepted, honest provers answering exactly at
    both deadlines, a reveal exactly `T_round` after the last with `T_round` above `T_read`, a zero deadline, whole
    pages and part of a page, and audits driven by `run_erasure`."""
    seen, edges, disagree = Counter(), Counter(), []
    for n, (case, lean) in enumerate(zip(cases, FIXTURE["verdicts"], strict=True)):
        accepted, revealed = run(case)
        assert revealed == truth(case)[1][:len(revealed)], n
        if accepted != (lean == "1"):
            disagree.append(n)
        fault = _fault(case)
        if fault is not None and lean == "0":
            seen[fault] += 1
        if lean == "1":
            edges["fill-at-deadline"] += case.fill.delay == case.fill_deadline
            edges["read-at-deadline"] += any(r.delay == case.read_deadline for r in case.reads)
            edges["reveal-at-t-round"] += case.round_deadline > case.read_deadline and any(
                reveal_gap(case, j) == case.round_deadline for j in range(1, case.spot_reads))
            edges["zero-deadline"] += case.read_deadline == 0 or case.fill_deadline == 0
            edges["whole-pages"] += case.size % case.page_bytes == 0 and case.spot_reads > 1
            edges["part-of-a-page"] += case.size % case.page_bytes != 0
            edges["run-erasure"] += not any(r.gap for r in case.reads) and case.spot_reads > 1
    assert not disagree, f"{len(disagree)} disagreements, the first cases {disagree[:10]}"
    assert all(seen[k] >= 20 for k in REQUIRED), {k: seen[k] for k in REQUIRED}
    assert min(edges.values()) >= 100 and len(edges) == 7, edges


MUTANTS = {
    "fill deadline exclusive": [("on_time = elapsed <= self.fill_deadline_ns", "on_time = elapsed < self.fill_deadline_ns")],
    "read deadline exclusive": [("on_time = elapsed <= self.read_deadline_ns", "on_time = elapsed < self.read_deadline_ns")],
    "reads timed by T_fill": [("on_time = elapsed <= self.read_deadline_ns", "on_time = elapsed <= self.fill_deadline_ns")],
    "digest unchecked": [("type(digest) is bytes and digest == self.fill.digest()", "type(digest) is bytes")],
    "byte checked mod 256": [("value == self.fill.at(i)", "value % 256 == self.fill.at(i)")],
    "one read short": [("return self.rejected or len(self._reads) == self.spot_reads",
                        "return self.rejected or len(self._reads) == self.spot_reads - 1")],
    "no reads needed": [("return self.rejected or len(self._reads) == self.spot_reads", "return True"),
                        ("len(self.reads) == self.spot_reads\n                and ", "")],
    "late fill passes": [("on_time = elapsed <= self.fill_deadline_ns", "on_time = True")],
    "late read passes": [("on_time = elapsed <= self.read_deadline_ns", "on_time = True")],
    "one index for every read": [("self._open = (j, self._indices[j], now)\n        return j, self._indices[j]",
                                  "self._open = (j, self._indices[0], now)\n        return j, self._indices[0]")],
    "reveal unchecked": [("if self._revealed is not None and now - self._revealed > self.round_deadline_ns:",
                          "if False:")],
    "reveal deadline exclusive": [("now - self._revealed > self.round_deadline_ns",
                                   "now - self._revealed >= self.round_deadline_ns")],
    "reveal timed by T_read": [("now - self._revealed > self.round_deadline_ns",
                                "now - self._revealed > self.read_deadline_ns")],
    "reveal timed from the answer": [("now - self._revealed > self.round_deadline_ns",
                                      "now - self._reads[-1].elapsed_ns - self._revealed > self.round_deadline_ns")],
}
"""Seeded bugs in `erasure.py`, each a list of `(text, replacement)` edits whose text occurs exactly once."""


def _mutant(edits, monkeypatch) -> types.ModuleType:
    src = inspect.getsource(erasure)
    for old, new in edits:
        assert src.count(old) == 1, f"the mutant's text is gone from erasure.py: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType("verity_pous._erasure_mutant")
    mod.__package__ = "verity_pous"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(src, "<erasure mutant>", "exec"), mod.__dict__)  # noqa: S102 -- erasure.py's own source, edited
    return mod


@pytest.mark.parametrize("name", MUTANTS)
def test_every_mutant_disagrees_with_lean(cases, name, monkeypatch):
    """The difftest is not inert: each seeded bug in the verifier reaches a verdict other than Lean's on some case, or
    reveals other indices than its draw (a verifier that reads one byte `s` times agrees with Lean's verdicts)."""
    mod = _mutant(MUTANTS[name], monkeypatch)
    for n, (case, lean) in enumerate(zip(cases, FIXTURE["verdicts"], strict=True)):
        accepted, revealed = run(case, mod)
        if accepted != (lean == "1") or revealed != truth(case)[1][:len(revealed)]:
            return
    pytest.fail(f"mutant {name!r} agrees with Lean on all {len(cases)} cases")


@pytest.mark.skipif(os.environ.get("VERITY_POUS_LEAN") != "1",
                    reason="reruns Lean on the cases, about a minute with the toolchain: set VERITY_POUS_LEAN=1")
def test_lean_reaches_the_pinned_verdicts(cases):
    """Lean, run now on the cases, prints the pinned verdicts."""
    assert cases_mod.lean_verdicts([line(c) for c in cases]) == FIXTURE["verdicts"]
