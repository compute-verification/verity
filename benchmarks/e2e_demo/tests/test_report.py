"""Demo A's report: it renders IFACE-B's block records with every number they don't hold as a placeholder, in Daniel's
sections, refuses the words CMP-98 retires ("leg", "window", "template") and the subprotocols' old names, fills each number from exactly one source,
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
SECTIONS = ["# Demo A: one H200, one block", "## 1. Ran and proved, on one H200, one block", "### Visible failures",
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
    assert "**What was checked.** A handful: k = 2 units, drawn uniformly over the whole partition (1,024" in out
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
    assert names == ((record | measured | derived | set(D.BLOCKS)) - periods - {"H_POOLED", "B_POOLED", "JOINT_LOCKED"}
                     - {"SECTIONS", "MISSES"})
    assert D.BANNED.search(D.TEMPLATE.read_text()) is None


@pytest.mark.parametrize("word", ["leg", "Legs", "window", "templates", "integrity", "Work", "space", "communication"])
def test_the_retired_words_are_refused(word: str) -> None:
    with pytest.raises(ValueError, match=f"says {word!r}"):
        D.vocabulary(f"PoUW's {word} accepted")
    doc = {**_m(), "caveats": [f"the certifier {word} was stubbed"]}
    with pytest.raises(ValueError, match=f"says {word!r}"):
        D.render([_load("block-record-r2-period0")], doc)


def test_the_word_leg_in_a_source_is_refused() -> None:
    doc = {"format": D.MEASUREMENTS, "values": {"ZK_SESSIONS": {"value": 3, "source": "the leg's sessions"}}}
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


def _costed() -> dict:
    doc = _m(PROVING_GPU_S=5400, BLOCK_GPU_S=60, H_A=10, BLOCK_ANDS=1000, PLANNER_PREDICTED_S=6000,
             BLOCK_OUTPUT_BITS=100, PROVE_S_PER_AND=1e-6, BLOCK_FLOP=200)
    doc["values"]["WORLD_FLOP_YEAR"] = {"value": 9.4e30, "source": "test", "label": "world-inference-2027"}
    doc["subcircuits"] = [
        {"name": "a", "units": 1000, "ands_per_unit": 20000, "out_bits": 16, "fixed_s": 36, "source": "test"},
        {"name": "b", "units": 250, "ands_per_unit": 500, "out_bits": 16, "fixed_s": 72, "source": "test"}]
    return doc


def test_derived_figures_follow_their_inputs() -> None:
    out = D.render([_load("block-record-r2-period0")], _costed())
    assert "proving this block cost 90× the block's own GPU time" in out
    assert "at most 10 of the block's 1,000 ANDs (1%)" in out
    assert "actual / predicted = 0.9" in out
    assert "That's 2 H200-hours, which D H200s fill in 0.0012 minutes." in out
    assert ("| week | 100,000 | 3,126,428 | 3.47e-22% *(world-inference-2027)* | 0.00031% | 177,688 | "
            "1.97e-22% *(world-inference-2027)* | 0.000176% |") in out
    assert "across 2 subcircuits" in out


def test_part_3s_h_and_b_are_the_planners_at_one_week() -> None:
    out = D.render([_load("block-record-r2-period0")], _costed())
    assert "budget 6,048 GPU-s = 1% of a one-week sampling block (604,800 H200-s" in out
    assert "a: n 10,080,000, 20,000 ANDs, 16 output bits, fixed 36 GPU-s, k 296,953" in out
    assert "in its\nAND-weighted form: a unit's cost is its circuit size, so the detection budget is counted in ANDs." in out
    doc = _costed()
    del doc["values"]["PROVE_S_PER_AND"]
    unmeasured = D.render([_load("block-record-r2-period0")], doc)
    assert "| week | 100,000 | ⟨H_POOLED⟩ | ⟨H_WORLD_SHARE_WEEK⟩ | ⟨H_SHARE_WEEK⟩ | ⟨B_POOLED⟩ |" in unmeasured
    with pytest.raises(ValueError, match="⟨H_POOLED⟩ is derived"):
        D.render([_load("block-record-r2-period0")], _m(H_POOLED=500))


def test_the_world_figure_carries_its_census_label() -> None:
    doc = _m(WORLD_FLOP_YEAR=9.4e30)
    with pytest.raises(ValueError, match="world-inference-2027"):
        D.render([_load("block-record-r2-period0")], doc)
    assert "compute, ⟨WORLD_FLOP_YEAR⟩ FP8 FLOP a year" in D.render([_load("block-record-r2-period0")])


def test_a_stub_caveat_drops_once_the_measurements_show_it_fixed() -> None:
    rec = _load("block-record-r2-period0")
    out = D.render([rec])
    assert "(#4405, #4426)" in out and "(NET-80)" in out and "ReLU and per-row quantize are not proved units" in out
    assert "This block drew k = 2, not M3's handful of 8 (PRF-171)." in out
    assert "run on the prover's node for the demo" in out
    assert "(RT-LEAN L22)" in out
    fixed = D.render([rec], {**_m(), "fixed": {"pouw-live-salt": "run r1", "torch-loop-wiring": "run r2"}})
    assert "(#4405, #4426)" not in fixed and "(NET-80)" in fixed and "ReLU and per-row" not in fixed
    assert "(RT-LEAN L22)" in fixed
    assert "(RT-LEAN L22)" not in D.render([rec], {**_m(), "fixed": {"pouw-salt-order": "run r3"}})
    with pytest.raises(ValueError, match="fixed 'salt'"):
        D.render([rec], {**_m(), "fixed": {"salt": "run r1"}})


def test_the_three_landed_fixes_drop_their_caveats_cleanly() -> None:
    rec = _load("block-record-r2-period0")
    landed = {"certifier-output-tie": "#4575, e2e r20261026-063606-3381",
              "torch-loop-wiring": "#4558 (25e140abcd), e2e r20261026-070934-56b3",
              "pouw-salt-order": "#4573, carried by #4594"}
    caveats = D.render([rec], {**_m(), "fixed": landed}).split("## What the demo didn't show")[1].split("## Sources")[0]
    assert "(NET-80)" not in caveats and "ReLU and per-row" not in caveats and "(RT-LEAN L22)" not in caveats
    assert "(#4405, #4426)" in caveats and "\n\n\n" not in caveats
    assert all(line.startswith(("- ", "  ")) for line in caveats.strip().splitlines())


def _failure(unit: int, **kw) -> dict:
    return {"subcircuit": "PouwH1Tile_v1", "unit": unit, "step": 12, "slot": 7,
            "why": "a PoUW tile lane: no proving Definition", "source": "r20261026-120000-aaaa", **kw}


def test_a_visible_failure_is_named_beside_the_verdict_and_fails_the_block() -> None:
    rec = _load("block-record-r2-period0")
    out = D.render([rec], {**_m(), "visible_failures": [_failure(4093), _failure(77, slot=3)]})
    part1 = out.split("## 2.")[0]
    assert "**fail (2 drawn units not proved: visible failures, no redraw)**" in part1
    assert ("| PouwH1Tile_v1 | 4,093 | 12 | 7 | a PoUW tile lane: no proving Definition | r20261026-120000-aaaa |"
            in part1)
    assert "| fail (2 drawn units not proved: visible failures, no redraw) |" in out.split("## 1.")[0]
    assert "| ⟨VISIBLE_FAILURES⟩ | 2 | r20261026-120000-aaaa |" in out
    none = D.render([rec], {**_m(), "visible_failures": []})
    assert "None: every drawn unit was proved." in none and "the network certifier: **accept**." in none
    unsaid = D.render([rec])
    assert "\n⟨VISIBLE_FAILURES⟩\n" in unsaid and "| ⟨VISIBLE_FAILURES⟩ | unfilled |" in unsaid
    with pytest.raises(ValueError, match="needs subcircuit, unit, step, slot, why, source"):
        D.render([rec], {**_m(), "visible_failures": [_failure(1, source="")]})


@pytest.mark.parametrize("name", ["AllGather_v1", "Gather_v1{N=8}"])
def test_a_gather_is_no_subcircuit(name: str) -> None:
    doc = {**_m(), "subcircuits": [{"name": name, "units": 64, "source": "test"}]}
    with pytest.raises(ValueError, match="plain wire reference"):
        D.render([_load("block-record-r2-period0")], doc)


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
    assert D.demo_a_verdict(absent) == "pending: no network certifier on the block's server"
    with pytest.raises(ValueError, match="block record r20261025-210358-993e"):
        D.render([rejected])


def test_the_pytorch_dry_run_renders_labelled() -> None:
    rec = _load("block-record-b71a-w30")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    out = D.render([rec], _load("demo-a-dry-run.measurements"), record_label="PyTorch dry run")
    assert out.splitlines()[2].startswith("> The PyTorch dry run's numbers are not marked in place")
    assert "sampled proofs, PoUW and the network certifier: **accept**." in out.replace("\n", " ")
    assert "k = 64 units, drawn uniformly over the whole partition (1,024 units across 1 subcircuits" in out.replace(
        "\n", " ")
    assert "53 sessions (r20261025-230005-b71a)" in out
    assert "This block drew k = 64, not M3's handful of 8 (PRF-171)." in out
    assert "at most 68 of its" in out and "at most K = 1,449 bits leaked" in out
    assert "Commit on the H200: 225 *(H200 bench HASH-7, not the demo block)* GB/s" in out
    assert "(15.7% *(HASH-20 bench, not the demo block)*)" in out
    assert ("13.9 *(M3 bench r20261026-085143-da39)* s a step at batch 512 (4.445 *(M3 bench r20261026-085143-da39)* "
            "ns per\nrow-slot)") in out
    assert "21.7 *(M3 bench r20261026-093524-01df)* s at batch 1,024" in out
    assert "512 to 9 *(kernels #4666 bench)* s." in out
    assert "PoUW backing: this block's PoUW ran the v1 subset's tiles (PouwH1Tile_v1)" in out
    assert ("| ⟨DRAWN_UNITS⟩ | 64 | block record r20261025-230005-b71a › `integrity` › quantity.drawn "
            "*(PyTorch dry run)* |") in out
    assert "⟨FLIP_VERDICT⟩" in out and "⟨PROVING_GPU_S⟩" in out


def _fallback(**row) -> dict:
    base = {"what": "commit overhead at 4,096 slots", "value": 0.424, "unit": "s", "run": "r20261026-191502-b4da",
            "verdict": "VERDICT best ...: overhead +0.334 s against <= 0.95 s", "target": {"op": "<=", "value": 0.95}}
    return {"format": D.MEASUREMENTS, "status": "M3 fallback: live-drawn, opened, replayed (Python); not proved in C-Flock",
            "sections": [{"title": "Commit overhead", "text": "On circuits' 1 KiB rows.", "rows": [{**base, **row}]}]}


def test_the_fallback_starts_with_its_status_and_needs_no_block_record() -> None:
    out = D.render([], _fallback())
    assert out.splitlines()[0] == "M3 fallback: live-drawn, opened, replayed (Python); not proved in C-Flock"
    assert "## 1. Commit overhead" in out and "ratio 0.446×, **met**). Run r20261026-191502-b4da." in out
    assert "None: every number above met its target." in out and "⟨H_POOLED⟩" in out
    with pytest.raises(ValueError, match="need a status"):
        D.render([], {**_fallback(), "status": ""})


def test_a_missed_target_is_a_visible_failure() -> None:
    out = D.render([], _fallback(value=1.046)).split("## Visible failures")[1]
    assert "**commit overhead at 4,096 slots**: 1.046 against target ≤ 0.95 s (ratio 1.1×). Run r20261026-191502-b4da." in out
    done = D.render([], _fallback(what="S1 on main", value="open", unit=None, target={"op": "==", "value": "merged"}))
    assert "**S1 on main**: open against target = merged. Run" in done


@pytest.mark.parametrize("field", ["run", "verdict", "target"])
def test_a_row_without_its_run_verdict_or_target_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match=f"names no {field}"):
        D.render([], _fallback(**{field: ""}))
    with pytest.raises(ValueError, match="has no run id"):
        D.render([], _fallback(run="PR #4768"))


def test_old_names_inside_a_quoted_verdict_line_are_kept() -> None:
    out = D.render([], _fallback(verdict="Integrity Work Space Comm: reject at window 0"))
    assert "`Integrity Work Space Comm: reject at window 0`" in out
    with pytest.raises(ValueError, match="says 'integrity'"):
        D.render([], _fallback(what="integrity bound"))
