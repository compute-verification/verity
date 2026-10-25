"""The network certifier over a period (`NetworkCheck.Period`, `verity/certifier/v0`, `network-period serve`) against its
Python mirror, through the difftest harness's `ProtocolPort`.

The mirror and the generator are `period_case.py`'s: `reference` (its `answer`) checks a period in the core interface's
answer format, `build` makes a period with one change (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases), and `case` a small period aimed at a verdict or at a decoding failure
(`KINDS`). This module re-exports what the composition's difftest reads of it (`reference`, `build`, `EXPECT`).

`FIXTURE` is a few of these cases with the mirror's answers: the composition's difftest answers onsite's certifier with
them (`verity/core/protocols/composition/tests/reference.py`, `Recorded`), since its suite runs neither this mirror nor
LMS. `test_the_composition_fixture_is_current` holds it to the mirror and to Lean; `python
test_network_period_protocol.py` writes it."""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import strategies as st

import period_case as PC
import verity_difftest as D
from period_case import (ACCEPT, EXPECT, INGRESS, KINDS, MUTATIONS, REJECT, UNDECIDED, Facts, Params, at_most, build,
                         clog2, public, reference, verdict_of)
from verity.core.protocols.certifier.schedule import SyncSet

__all__ = ["EXPECT", "MUTATIONS", "build", "reference"]

LEAN = Path(__file__).resolve().parents[1] / "lean"
FIXTURE = Path(__file__).resolve().parents[2] / "composition" / "tests" / "fixtures" / "certifier.json"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

#: Each fixture case for the composition's difftest, by how `build` makes it: accepted, rejected, undecided and refused
#: periods (no `case` break: the composition re-encodes a component's case from its bytes).
FIXTURE_CASES = [("honest", s) for s in range(4)] + [
    ("answer-undecided", 0), ("audit-undecided", 0), ("cert-forged", 0), ("advice", 0), ("audit-reject", 0),
    ("params", 0), ("commitment", 0), ("evidence", 0)]

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    m = draw(st.sampled_from(["honest"] * 3 + list(only) if "honest" in only else list(only)))
    case = build(m, draw(st.integers(0, 1 << 16)))
    TAGS[D.canonical(case)] = m
    return case


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), out["verdict"])] += 1
    return out


PORT = D.ProtocolPort("network-period", counted, cases(), package=LEAN)
server = D.server_fixture(PORT)


def fixture() -> dict:
    """The composition's fixture: each `FIXTURE_CASES` case with the mirror's answer."""
    return {"cases": [{"answer": reference(case), "case": case, "made": [m, n]}
                      for m, n in FIXTURE_CASES for case in [build(m, n, small=True)]]}


@pytest.mark.slow
def test_the_period_check_agrees_with_its_mirror(server):
    """Lean and the mirror agree on every case, and each change reaches its verdict at least once."""
    SEEN.clear()
    examples = 800
    stats = D.run(PORT, examples=examples, server=server)
    print("\n".join(f"{tag:22} {verdict:10} {k}" for (tag, verdict), k in sorted(SEEN.items())))
    missed = [tag for tag, verdict in EXPECT.items() if not SEEN[(tag, verdict)]]
    assert not missed, (missed, SEEN)
    assert min(stats.counts.get(v, 0) for v in (ACCEPT, REJECT, UNDECIDED)) > examples // 50, stats


def test_each_verdict_is_the_mirrors(server):
    """A short run over an honest period, an undecided link-window and a forged certificate: all three verdicts."""
    port = dataclasses.replace(PORT, inputs=cases(("honest", "answer-undecided", "cert-forged")))
    stats = D.run(port, examples=30, server=server)
    assert all(stats.counts.get(v) for v in (ACCEPT, REJECT, UNDECIDED)), stats


def test_a_mirror_that_skips_the_ingress_certificate_is_caught(server, monkeypatch):
    """A mirror that reads only the egress certificate accepts a period whose ingress certificate names another link,
    and Lean rejects it."""
    real = PC._certifies
    monkeypatch.setattr(PC, "_certifies", lambda p, d, w, c: d is INGRESS or real(p, d, w, c))
    with pytest.raises(D.Mismatch) as x:
        D.run(dataclasses.replace(PORT, inputs=cases(("cert-link",))), examples=20, server=server)
    assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT)


def test_a_small_period_reaches_its_verdict(server):
    """Each change's small period (`build(..., small=True)`, which the composition's difftest draws) gets the change's
    verdict from the mirror, and Lean agrees."""
    small = [build(m, n, small=True) for m in MUTATIONS if m != "case" for n in range(2)]
    assert [reference(c)["verdict"] for c in small] == [EXPECT[m] for m in MUTATIONS if m != "case" for _ in range(2)]
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(small)), examples=2 * len(small), server=server)


def test_each_kind_of_case_is_answered_as_lean_answers(server):
    """`case(kind, seed)` reaches its kind's verdict for each kind, the ones that fail to decode refused where they fail
    (a digest only once the commitment decodes); `cases(seed)` is one of each with its answer; and Lean gives every one
    of them the mirror's answer."""
    drawn = {(k, s): PC.case(k, s) for k in KINDS for s in range(6)}
    answers = {ks: PC.answer(c) for ks, c in drawn.items()}
    assert {ks: a["verdict"] for ks, a in answers.items()} == {(k, s): verdict_of(k) for k, s in drawn}
    assert all(a["reason"].startswith(k.removeprefix("broken-") + ":")
               and (a["commitment"] is None) == (k != "broken-evidence")
               for (k, _), a in answers.items() if k.startswith("broken-"))
    assert all(isinstance(a["public"]["K"], int) for (k, _), a in answers.items() if k == ACCEPT)
    assert PC.cases(3) == {k: (drawn[k, 3], answers[k, 3]) for k in KINDS}
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(list(drawn.values()))), examples=2 * len(drawn),
          server=server)


def test_the_composition_fixture_is_current(server):
    """`FIXTURE` is the mirror's answers on its cases, and Lean gives them too (`python
    test_network_period_protocol.py` writes it)."""
    recorded = json.loads(FIXTURE.read_text())
    assert recorded == fixture(), f"{FIXTURE} is stale: run `python {Path(__file__).name}` to write it"
    kinds = {e["answer"]["verdict"] for e in recorded["cases"]}
    assert kinds == {ACCEPT, REJECT, UNDECIDED}, kinds
    port = dataclasses.replace(PORT, inputs=st.sampled_from([e["case"] for e in recorded["cases"]]))
    D.run(port, examples=3 * len(FIXTURE_CASES), server=server)


def test_the_bound_counts_what_it_says():
    """K's terms on a period worked by hand: 10 units drawn subset:3 at δ = 1/10 with at most 4 wrong, 7 cells on one
    connection and none on another, 2 syncs over 3 link-windows, jitter below 2 over 3 steps (Lean's `#guard`)."""
    p = Params(((0, 3),), (7, 0), 2, "a", None, b"", "", SyncSet((0,), (Fraction(1), Fraction(2))), 3)
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)])
    assert (out["schedules"], out["cells"], out["audits"][0]["faulty_sets"], out["audits"][0]["faulty_bits"],
            out["audits"][0]["jitter"], out["K"]) == (3, 3, 9, 64, 3, 3 + 3 + 9 + 64 + 3)
    assert [clog2(x) for x in range(6)] == [0, 0, 1, 2, 2, 3]
    assert [at_most(10, 0), at_most(10, 1), at_most(10, 2), at_most(4, 9), at_most(0, 3)] == [1, 11, 56, 16, 1]


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture(), indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE} ({FIXTURE.stat().st_size} bytes)", file=sys.stderr)
