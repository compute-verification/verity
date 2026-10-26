"""Demo A's report: it renders IFACE-B's block records with every number they don't hold as a placeholder, in Daniel's
sections, refuses the words CMP-98 retires ("leg", "window", "template"), fills each number from exactly one source,
derives part 1's and part 3's figures from their inputs, and renders tonight's PyTorch dry run (r20261025-230005-b71a,
period 30, written by `block_record.from_period`) labelled as such."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from verity_e2e_demo import block_record as R
from verity_e2e_demo import report as D

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = json.loads((Path(R.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
SECTIONS = ["# Demo A: one H200, one block", "## 1. Ran and proved, on one H200, one block",
            "## 2. Measured on the demo: the inputs to the scaling claim",
            "## 3. Computed from the same theorem and the measured costs, not run", "## What the demo didn't show",
            "## Sources"]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _m(**values) -> dict:
    return {"format": D.MEASUREMENTS, "values": {k: {"value": v, "source": f"test: {k}"} for k, v in values.items()}}


def test_iface_b_records_render_with_placeholders() -> None:
    out = D.render([_load("block-record-r2-period0"), _load("block-record-r1-window10")])
    assert [line for line in out.splitlines() if line.startswith("#")] == SECTIONS
    template = D.TEMPLATE.read_text()
    for name in D.MEASURED:
        assert f"| ⟨{name}⟩ | unfilled |" in out, name
    assert "⟨H_A_SHARE⟩" in out and "⟨Y⟩" in out and "⟨PLANNER_RATIO⟩" in out
    assert "**What was checked.** 2 drawn units of 1,024" in out
    assert "at most K = 15,764 bits leaked" in out
    assert "W_cert = 262,144 (128 tiles credited)" in out
    assert "a deployment of D = 100,000 H200s (dc-12500-machines-600s" in out
    assert "| r20261025-203534-186b | computeinstance-e03nth5jqd47azxrz2 | [0, 2) | accept (real) | not run |" in out
    assert "⟨LABEL⟩" in template and "⟨LABEL⟩" not in out


def test_every_placeholder_is_one_number_of_the_report() -> None:
    names = set(D.PLACEHOLDER.findall(D.TEMPLATE.read_text()))
    record, measured, derived = set(D.RECORD), set(D.MEASURED), set(D.DERIVED)
    assert not record & measured and not record & derived and not measured & derived
    periods = {n for n in derived if n.split("_")[0] in ("BLOCKS", "H", "B") and n.endswith(("WEEK", "MONTH", "YEAR"))}
    assert names == (record | measured | derived | set(D.BLOCKS)) - periods - {"H_POOLED", "B_POOLED", "JOINT_LOCKED"}
    assert D.BANNED.search(D.TEMPLATE.read_text()) is None


@pytest.mark.parametrize("word", ["leg", "Legs", "window", "templates"])
def test_the_retired_words_are_refused(word: str) -> None:
    with pytest.raises(ValueError, match=f"says {word!r}"):
        D.vocabulary(f"PoUW's {word} accepted")
    doc = {**_m(), "caveats": [f"the certifier {word} was stubbed"]}
    with pytest.raises(ValueError, match=f"says {word!r}"):
        D.render([_load("block-record-r2-period0")], doc)


def test_the_word_leg_in_a_source_is_refused() -> None:
    doc = {"format": D.MEASUREMENTS, "values": {"ZK_SESSIONS": {"value": 3, "source": "the integrity leg's sessions"}}}
    with pytest.raises(ValueError, match="says 'leg'"):
        D.render([_load("block-record-r2-period0")], doc)


@pytest.mark.parametrize(("doc", "why"), [
    ({"format": "verity/demo-a-measurements/v0"}, "format"),
    (_m(NOT_A_NUMBER=1), "is no number of the report"),
    (_m(DRAWN_UNITS=64), "comes from the block record"),
    (_m(Y=3.0), "is derived"),
    ({"format": D.MEASUREMENTS, "values": {"H_A": {"value": 7}}}, "names no source"),
    ({"format": D.MEASUREMENTS, "values": {"H_A": {"value": None, "source": "s"}}}, "has no value"),
    ({"format": D.MEASUREMENTS, "subcircuits": [{"name": "s"}]}, "needs a name and a source"),
])
def test_a_measurement_it_may_not_hold_is_refused(doc: dict, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        D.render([_load("block-record-r2-period0")], doc)


def test_derived_figures_follow_their_inputs() -> None:
    doc = _m(PROVING_GPU_S=5400, BLOCK_GPU_S=60, H_A=10, BLOCK_ANDS=1000, PLANNER_PREDICTED_S=6000, H_POOLED=500,
             B_POOLED=40, BLOCK_OUTPUT_BITS=100)
    doc["subcircuits"] = [{"name": "a", "fixed_s": 36, "source": "test"},
                          {"name": "b", "fixed_s": 72, "source": "test"}]
    out = D.render([_load("block-record-r2-period0")], doc)
    assert "proving this block cost 90× the block's own GPU time" in out
    assert "at most 10 of the block's 1,000 ANDs (1%)" in out
    assert "actual / predicted = 0.9" in out
    assert "That's 2 H200-hours, which D H200s fill in 0.0012 minutes." in out
    assert "| week | 8,400,000 | 500 | 4.96e-08% | 40 | 3.97e-08% |" in out
    assert "across 2 subcircuits" in out


def test_the_bound_cites_the_joint_statement_only_when_locked() -> None:
    rec = _load("block-record-r2-period0")
    locked = D.render([rec], _m(JOINT_LOCKED=True)).split("## Sources")[0]
    assert "By LEAN-56's `Flock.Guarantees.JointStratified` (locked):" in locked
    assert "by LEAN-56's `JointStratifiedHorizon` (locked)" in locked and "unlocked" not in locked
    unlocked = D.render([rec], _m(JOINT_LOCKED=False))
    assert "By #4358's per-subcircuit bound, the joint figure unlocked:" in unlocked
    assert "by #4358's `Flock.Guarantees.IntegrityHorizon` per subcircuit, the joint figure unlocked" in unlocked
    assert "By ⟨BOUND_CITATION⟩:" in D.render([rec])


def test_demo_a_composes_its_three_protocols() -> None:
    rec = _load("block-record-r2-period0")
    assert D.demo_a_verdict(rec) == "accept"
    rejected = copy.deepcopy(rec)
    rejected["legs"]["pouw"]["answer"]["verdict"] = "reject"
    assert D.demo_a_verdict(rejected) == "reject (PoUW)"
    absent = copy.deepcopy(rec)
    absent["legs"]["certifier"] = None
    assert D.demo_a_verdict(absent) == "pending: no network accounting on the block's server"
    with pytest.raises(ValueError, match="block record r20261025-210358-993e"):
        D.render([rejected])


def test_the_pytorch_dry_run_renders_labelled() -> None:
    rec = _load("block-record-b71a-w30")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    out = D.render([rec], _load("demo-a-dry-run.measurements"), record_label="PyTorch dry run")
    assert out.splitlines()[2].startswith("> The PyTorch dry run's numbers are not marked in place")
    assert "sampled proofs, PoUW and network accounting: **accept**." in out.replace("\n", " ")
    assert "64 drawn units of 1,024 across 1 subcircuits" in out
    assert "in 53 sessions (r20261025-230005-b71a)" in out
    assert "at most 68 of its" in out and "at most K = 1,449 bits leaked" in out
    assert "Commit on the H200: 225 *(H200 bench HASH-7, not the demo block)* GB/s" in out
    assert "(15.7% *(HASH-20 bench, not the demo block)*)" in out
    assert "PoUW backing: this block's PoUW ran the v1 subset's tiles (PouwH1Tile_v1)" in out
    assert ("| ⟨DRAWN_UNITS⟩ | 64 | block record r20261025-230005-b71a › integrity › quantity.drawn "
            "*(PyTorch dry run)* |") in out
    assert "⟨FLIP_VERDICT⟩" in out and "⟨PROVING_GPU_S⟩" in out
