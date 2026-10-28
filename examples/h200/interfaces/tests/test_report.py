"""Demo A's report: it renders block records with every number they don't hold as a placeholder, in the report's
sections, refuses the retired words ("leg", "window", "template") and the subprotocols' old names, fills each number from exactly one source,
derives part 1's and part 3's figures from their inputs, and renders a PyTorch dry run (one period, written by
`block_record.from_period`) labelled as such."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

import re

from verity_e2e_demo import block_record as R
from verity_e2e_demo import report as D

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = json.loads((Path(R.__file__).parent / "schemas" / "block-record.v1.schema.json").read_text())
SECTIONS = ["# Demo A: one H200, one block", "## 1. Ran and proved, on one H200, one block", "### Visible failures",
            "## 2. Measured on the demo: the inputs to the scaling claim",
            "## 3. Computed from the same theorem and the measured costs, not run", "## What the demo didn't show",
            "## Sources"]


def _draft(*a, **k) -> str:
    return D.render(*a, draft=True, **k)


def _load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _m(**values) -> dict:
    return {"format": D.MEASUREMENTS, "values": {k: {"value": v, "source": f"test: {k}"} for k, v in values.items()}}


def test_iface_b_records_render_with_placeholders() -> None:
    out = _draft([_load("block-record-r2-period0"), _load("block-record-r1-window10")])
    assert [line for line in out.splitlines() if line.startswith("#")] == SECTIONS
    template = D.TEMPLATE.read_text()
    for name in D.MEASURED:
        assert f"| ⟨{name}⟩ | unfilled |" in out, name
    assert "⟨H_A_SHARE⟩" in out and "⟨Y⟩" in out and "⟨PLANNER_RATIO⟩" in out
    assert "**What was checked.** A handful: k = 2 units, drawn uniformly over the whole partition (1,024" in out
    assert "at most K = 15,764 bits leaked" in out
    assert "W_cert = 262,144 (128 tiles credited)" in out
    assert "a deployment of\nD = 100,000 H200s (dc-12500-machines-600s" in out
    assert "| r20261025-203534-186b | computeinstance-e03nth5jqd47azxrz2 | [0, 2) | accept (real) | not run |" in out
    assert "⟨LABEL⟩" in template and "⟨LABEL⟩" not in out


def test_every_placeholder_is_one_number_of_the_report() -> None:
    names = set(D.PLACEHOLDER.findall(D.TEMPLATE.read_text()))
    record, measured, derived = set(D.RECORD), set(D.MEASURED), set(D.DERIVED)
    assert not record & measured and not record & derived and not measured & derived
    periods = {n for n in derived if n.split("_")[0] in ("BLOCKS", "H", "B") and n.endswith(("WEEK", "MONTH", "YEAR"))}
    assert names == ((record | measured | derived | set(D.BLOCKS)) - periods - {"H_POOLED", "B_POOLED", "JOINT_LOCKED"}
                     - {"SECTIONS", "MISSES"} - set(D.SERVE_NOISE))
    assert D.BANNED.search(D.TEMPLATE.read_text()) is None


@pytest.mark.parametrize("word", ["leg", "Legs", "window", "templates", "integrity", "Work", "space", "communication"])
def test_the_retired_words_are_refused(word: str) -> None:
    with pytest.raises(ValueError, match=f"says {word!r}"):
        D.vocabulary(f"PoUW's {word} accepted")
    doc = {**_m(), "caveats": [f"the certifier {word} was stubbed"]}
    with pytest.raises(ValueError, match=f"says {word!r}"):
        _draft([_load("block-record-r2-period0")], doc)


def test_the_word_leg_in_a_source_is_refused() -> None:
    doc = {"format": D.MEASUREMENTS, "values": {"ZK_SESSIONS": {"value": 3, "source": "the leg's sessions"}}}
    with pytest.raises(ValueError, match="says 'leg'"):
        _draft([_load("block-record-r2-period0")], doc)


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
        _draft([_load("block-record-r2-period0")], doc)


PLANNER_RUN = "r20261027-032427-c117"


def _planned(**kw) -> dict:
    return {"PLANNER": {"value": f"`demo_a.py` @ 3b0c90de27, run {PLANNER_RUN}", "source": f"run {PLANNER_RUN}"},
            "H_POOLED": {"value": 3_126_434, "source": f"run {PLANNER_RUN} › free_compute"},
            "B_POOLED": {"value": 177_688, "source": f"run {PLANNER_RUN} › leak"}} | kw


def _costed() -> dict:
    doc = _m(PROVING_GPU_S=5400, BLOCK_GPU_S=60, H_A=10, BLOCK_ANDS=1000, PLANNER_PREDICTED_S=6000,
             BLOCK_OUTPUT_BITS=100, PROVE_S_PER_AND=1e-6)
    doc["values"]["WORLD_GATES_YEAR"] = {"value": 4.7e31, "source": "test", "label": "world-inference-2027"}
    doc["values"] |= _planned()
    doc["subcircuits"] = [
        {"name": "a", "units": 1000, "ands_per_unit": 20000, "out_bits": 16, "fixed_s": 36, "source": "test"},
        {"name": "b", "units": 250, "ands_per_unit": 500, "out_bits": 16, "fixed_s": 72, "source": "test"}]
    return doc


def test_derived_figures_follow_their_inputs() -> None:
    out = _draft([_load("block-record-r2-period0")], _costed())
    assert "proving this block cost 90× the block's own GPU time" in out
    assert "at most 10 of the block's 1,000 ANDs (1%)" in out
    assert "actual / predicted = 0.9" in out
    assert "That's 2 H200-hours, which D H200s fill in 0.0012 minutes." in out
    assert ("| week | 100,000 | 3,126,434 | 3.47e-22% *(world-inference-2027)* | 0.00031% | 177,688 | "
            "1.97e-22% *(world-inference-2027)* | 0.000176% |") in out
    assert "across 2 subcircuits" in out


def test_part_3s_h_and_b_are_one_recorded_run_of_proofs_planner() -> None:
    out = _draft([_load("block-record-r2-period0")], _costed())
    assert f"H and B are proofs' planner's (`demo_a.py` @ 3b0c90de27, run {PLANNER_RUN})" in out
    assert "in its\nAND-weighted form: a unit's cost is its circuit size, so the detection budget is counted in AND gates." in out
    assert f"| ⟨H_POOLED⟩ | 3,126,434 | run {PLANNER_RUN} › free_compute |" in out
    for values, why in (({"H_POOLED": {"value": 500, "source": f"run {PLANNER_RUN}"}}, "filled together"),
                        (_planned(B_POOLED={"value": 1, "source": "the planner"}), "filled together"),
                        (_planned(B_POOLED={"value": 1, "source": "run r20261027-000000-0000"}), "filled together")):
        with pytest.raises(ValueError, match=why):
            _draft([_load("block-record-r2-period0")], {"format": D.MEASUREMENTS, "values": values})


def test_a_report_with_an_unfilled_number_is_refused() -> None:
    """A rendered report never carries a placeholder: only a draft shows ⟨NAME⟩, and `main` renders no draft."""
    doc = {**_proved(), "values": _planned(JOINT_LOCKED={"value": True, "source": "the lock"}) | _noise()}
    doc["sections"][0]["serve_noise"] = True
    out = D.render([], doc)
    assert out.splitlines()[0] == D.M3_STATUS and not re.search(r"⟨[A-Z][A-Z0-9_]*⟩", out.split("## Sources")[0])
    assert "a one-week sampling block of one H200\nhas at most H = 3,126,434 faulty AND gates and B = 177,688" in out
    for name in ("H_POOLED", "SERVE_NOISE_S"):
        partial = copy.deepcopy(doc)
        del partial["values"][name]
        if name == "H_POOLED":
            del partial["values"]["B_POOLED"], partial["values"]["PLANNER"]
        with pytest.raises(ValueError, match=f"unfilled number.*⟨{name}⟩"):
            D.render([], partial)
        assert f"⟨{name}⟩" in _draft([], partial)
    with pytest.raises(ValueError, match="unfilled number"):
        D.render([_load("block-record-r2-period0")])


def test_part_1_states_the_guarantee_the_draw_certifies() -> None:
    doc = {**_proved(), "values": _planned(JOINT_LOCKED={"value": True, "source": "the lock"})}
    doc["proved"]["rows"] = [{"what": "the guarantee at δ = 1%: wrong proof blocks at most", "value": 1_367_366_590,
                              "unit": "of 3,124,276,224 proof blocks (43.77%)", "run": "r20261027-013000-abcd",
                              "verdict": "W at δ = 10⁻²: 1,367,366,590 of 3,124,276,224 proof blocks (43.77%)",
                              "target": {"none": "weak at an 8-block draw"}}]
    part1 = D.render([], doc).split("## 2. ")[0]
    assert ("- **the guarantee at δ = 1%: wrong proof blocks at most:** 1,367,366,590 of 3,124,276,224 proof blocks "
            "(43.77%) (no target (weak at an 8-block draw)). Run r20261027-013000-abcd.") in part1
    doc["proved"]["rows"][0]["run"] = "proofs"
    with pytest.raises(ValueError, match="has no run id"):
        D.render([], doc)


def test_the_world_figure_carries_its_census_label() -> None:
    doc = _m(WORLD_GATES_YEAR=4.7e31)
    with pytest.raises(ValueError, match="world-inference-2027"):
        _draft([_load("block-record-r2-period0")], doc)
    assert "compute, ⟨WORLD_GATES_YEAR⟩ AND gates a year" in _draft([_load("block-record-r2-period0")])


def test_part_3_is_circuit_size_in_gates_with_no_flop_or_price() -> None:
    part3 = D.TEMPLATE.read_text().split("## 3.")[1].split("## What")[0]
    assert "counted in AND gates" in part3 and not re.search(r"FLOP|\$|price|pricing|cost of", part3, re.I)
    assert not [n for n in (*D.MEASURED, *D.DERIVED) if "FLOP" in n]
    out = _draft([_load("block-record-r2-period0")], _costed()).split("## 3.")[1].split("## What")[0]
    assert "FLOP" not in out and "| H faulty AND gates | share of the world's AND gates |" in out


#: a phrase from each of three stub caveats (`report.STUBS`)
LIVE_SALT = "PoUW's salt isn't yet the live coin read in its answer"
OUTPUT_TIE = "The network certifier's output tie isn't proved yet"
SALT_ORDER = "so a prover could choose x after seeing it"


def test_a_stub_caveat_drops_once_the_measurements_show_it_fixed() -> None:
    rec = _load("block-record-r2-period0")
    out = _draft([rec])
    assert LIVE_SALT in out and OUTPUT_TIE in out and "ReLU and per-row quantize are not proved units" in out
    assert "This block drew k = 2, not M3's handful of 8." in out
    assert "run on the prover's node for the demo" in out
    assert SALT_ORDER in out
    fixed = _draft([rec], {**_m(), "fixed": {"pouw-live-salt": "run r1", "torch-loop-wiring": "run r2"}})
    assert LIVE_SALT not in fixed and OUTPUT_TIE in fixed and "ReLU and per-row" not in fixed
    assert SALT_ORDER in fixed
    assert SALT_ORDER not in _draft([rec], {**_m(), "fixed": {"pouw-salt-order": "run r3"}})
    with pytest.raises(ValueError, match="fixed 'salt'"):
        _draft([rec], {**_m(), "fixed": {"salt": "run r1"}})


def test_the_three_landed_fixes_drop_their_caveats_cleanly() -> None:
    rec = _load("block-record-r2-period0")
    landed = {"certifier-output-tie": "e2e r20261026-063606-3381",
              "torch-loop-wiring": "25e140abcd, e2e r20261026-070934-56b3",
              "pouw-salt-order": "the salt-after-anchors change"}
    caveats = _draft([rec], {**_m(), "fixed": landed}).split("## What the demo didn't show")[1].split("## Sources")[0]
    assert OUTPUT_TIE not in caveats and "ReLU and per-row" not in caveats and SALT_ORDER not in caveats
    assert LIVE_SALT in caveats and "\n\n\n" not in caveats
    assert all(line.startswith(("- ", "  ")) for line in caveats.strip().splitlines())


def _failure(unit: int, **kw) -> dict:
    return {"subcircuit": "PouwH1Tile_v1", "unit": unit, "step": 12, "slot": 7,
            "why": "a PoUW tile lane: no proving Definition", "source": "r20261026-120000-aaaa", **kw}


def test_a_visible_failure_is_named_beside_the_verdict_and_fails_the_block() -> None:
    rec = _load("block-record-r2-period0")
    out = _draft([rec], {**_m(), "visible_failures": [_failure(4093), _failure(77, slot=3)]})
    part1 = out.split("## 2.")[0]
    assert "**fail (2 drawn units not proved: visible failures, no redraw)**" in part1
    assert ("| PouwH1Tile_v1 | 4,093 | 12 | 7 | a PoUW tile lane: no proving Definition | r20261026-120000-aaaa |"
            in part1)
    assert "| fail (2 drawn units not proved: visible failures, no redraw) |" in out.split("## 1.")[0]
    assert "| ⟨VISIBLE_FAILURES⟩ | 2 | r20261026-120000-aaaa |" in out
    none = _draft([rec], {**_m(), "visible_failures": []})
    assert "None: every drawn unit was proved." in none and "the network certifier: **accept**." in none
    unsaid = _draft([rec])
    assert "\n⟨VISIBLE_FAILURES⟩\n" in unsaid and "| ⟨VISIBLE_FAILURES⟩ | unfilled |" in unsaid
    with pytest.raises(ValueError, match="needs subcircuit, unit, step, slot, why, source"):
        _draft([rec], {**_m(), "visible_failures": [_failure(1, source="")]})


@pytest.mark.parametrize("name", ["AllGather_v1", "Gather_v1{N=8}"])
def test_a_gather_is_no_subcircuit(name: str) -> None:
    doc = {**_m(), "subcircuits": [{"name": name, "units": 64, "source": "test"}]}
    with pytest.raises(ValueError, match="plain wire reference"):
        _draft([_load("block-record-r2-period0")], doc)


def test_the_bound_cites_the_joint_statement_only_when_locked() -> None:
    rec = _load("block-record-r2-period0")
    locked = _draft([rec], _m(JOINT_LOCKED=True)).split("## Sources")[0]
    assert "By `Flock.Guarantees.JointStratified` (locked):" in locked
    assert "by `JointStratifiedHorizon` (locked)" in locked and "unlocked" not in locked
    unlocked = _draft([rec], _m(JOINT_LOCKED=False))
    assert "By the per-subcircuit bound, the joint figure unlocked:" in unlocked
    assert "by `Flock.Guarantees.IntegrityHorizon` per subcircuit, the joint figure unlocked" in unlocked
    assert "By ⟨BOUND_CITATION⟩:" in _draft([rec])


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
        _draft([rejected])


def test_the_pytorch_dry_run_renders_labelled() -> None:
    rec = _load("block-record-b71a-w30")
    jsonschema.Draft202012Validator(SCHEMA).validate(rec)
    assert R.check(rec) == {}
    out = _draft([rec], _load("demo-a-dry-run.measurements"), record_label="PyTorch dry run")
    assert out.splitlines()[2].startswith("> The PyTorch dry run's numbers are not marked in place")
    assert "sampled proofs, PoUW and the network certifier: **accept**." in out.replace("\n", " ")
    assert "k = 64 units, drawn uniformly over the whole partition (1,024 units across 1 subcircuits" in out.replace(
        "\n", " ")
    assert "53 sessions (r20261025-230005-b71a)" in out
    assert "This block drew k = 64, not M3's handful of 8." in out
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
    out = _draft([], _fallback())
    assert out.splitlines()[0] == "M3 fallback: live-drawn, opened, replayed (Python); not proved in C-Flock"
    assert "## 1. Commit overhead" in out and "ratio 0.446×, **met**). Run r20261026-191502-b4da." in out
    assert "None: every number above met its target." in out and "⟨H_POOLED⟩" in out
    with pytest.raises(ValueError, match="need a status"):
        _draft([], {**_fallback(), "status": ""})


def test_a_missed_target_is_a_visible_failure() -> None:
    out = _draft([], _fallback(value=1.046)).split("## Visible failures")[1]
    assert "**commit overhead at 4,096 slots**: 1.046 s against target ≤ 0.95 s (ratio 1.1×). Run r20261026-191502-b4da." in out
    done = _draft([], _fallback(what="S1 on main", value="open", unit=None, target={"op": "==", "value": "merged"}))
    assert "**S1 on main**: open against target = merged. Run" in done


@pytest.mark.parametrize("field", ["run", "verdict", "target"])
def test_a_row_without_its_run_verdict_or_target_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match=f"names no {field}"):
        _draft([], _fallback(**{field: ""}))
    with pytest.raises(ValueError, match="has no run id"):
        _draft([], _fallback(run="the commit-overhead bench"))


def test_old_names_inside_a_quoted_verdict_line_are_kept() -> None:
    out = _draft([], _fallback(verdict="Integrity Work Space Comm: reject at window 0"))
    assert "`Integrity Work Space Comm: reject at window 0`" in out
    with pytest.raises(ValueError, match="says 'integrity'"):
        _draft([], _fallback(what="integrity bound"))


def _proved(accepted=(True,) * 8, drawn=tuple(range(101, 109)), ok=True, refused=3, lines=None) -> dict:
    out = [f"VERDICT {json.dumps({'block': g, 'accepted': a, 'k_log': 27, 'lean': 'ACCEPT', 'm0': a})}"
           for g, a in zip(drawn, accepted)]
    out.append("PROVEBUNDLE " + json.dumps({"ok": ok, "negatives": 3, "negatives_refused": refused}))
    return {**_fallback(), "proved": {"run": "r20261027-013000-abcd", "drawn": list(drawn), "text": "DRAWPROVE.",
                                      "lines": out if lines is None else lines}}


def test_m3_drops_the_fallback_line_only_when_all_eight_are_proved_and_accepted() -> None:
    out = _draft([], _proved())
    assert out.splitlines()[0] == D.M3_STATUS and "fallback" not in out.lower()
    assert "## 1. M3 proper: the drawn H200 proof blocks proved in C-Flock" in out and "## 2. Commit overhead" in out
    assert "**drawn proof block 108: proved in C-Flock --zk behind the firewall, verified by M0 and Lean:** accepted" in out
    assert "None: every number above met its target." in out


@pytest.mark.parametrize("kw", [{"accepted": (True,) * 7 + (False,)}, {"drawn": tuple(range(101, 108))},
                                {"ok": False}, {"refused": 2}])
def test_m3_stays_the_fallback_until_then(kw: dict) -> None:
    out = _draft([], _proved(**kw))
    assert out.splitlines()[0] == D.FALLBACK_STATUS
    assert "None: every number above met its target." not in out.split("## Visible failures")[1]


def test_a_drawn_proof_block_with_no_verdict_line_is_a_visible_failure() -> None:
    doc = _proved()
    doc["proved"]["lines"] = [x for x in doc["proved"]["lines"] if '"block": 104' not in x]
    out = _draft([], doc)
    assert out.splitlines()[0] == D.FALLBACK_STATUS
    assert ("**drawn proof block 104: proved in C-Flock --zk behind the firewall, verified by M0 and Lean**: no VERDICT "
            "line against target = accepted. Run r20261027-013000-abcd.") in out.split("## Visible failures")[1]


def test_proved_refuses_another_status_and_an_undrawn_block() -> None:
    with pytest.raises(ValueError, match="the status is"):
        _draft([], {**_proved(), "status": D.M3_STATUS})
    doc = _proved()
    doc["proved"]["lines"].insert(0, 'VERDICT {"block": 999, "accepted": true}')
    with pytest.raises(ValueError, match="block 999, which the coin didn't draw"):
        _draft([], doc)
    with pytest.raises(ValueError, match="needs its run id"):
        _draft([], {**_proved(), "proved": {"run": "the proving run", "drawn": [1], "lines": []}})


NOISE = {"SERVE_NOISE_S": (8.5, "kraqwxsc r20261026-113133-5cb4 against r20261026-120015-3773"),
         "SERVE_NOISE_BATCH": (4096, "r20261026-113133-5cb4"), "SERVE_NOISE_UNITS_S": (5.59, "r20261026-113133-5cb4"),
         "SERVE_NOISE_COMMIT_S": (2.8, "r20261026-113133-5cb4"), "SERVE_NOISE_GAMMA_STEP": (1.115, "the noise run"),
         "SERVE_NOISE_BASE_S": (74.4, "r20261026-120015-3773"), "SERVE_STEP_4562_S": (3.78, "the kraq build"),
         "SERVE_NOISE_UNITS_PEAK": (0.10, "tensor peak r20261027-022021-3a48"),
         "SERVE_COMMIT_OVERHEAD_S": (0.42, "commit overhead r20261027-004614-d98c")}
NOISE_LINE = ("Per-call noise adds **+8.5 s a step at batch 4,096** (5.59 s of units plus 2.8 s of commit, "
              "r20261026-113133-5cb4 against r20261026-120015-3773): 1.115× against that commit-bound 74.4 s served "
              "step, which is mostly FAMILIES' slot commit (`commit_gpu`), and about 3.2× against the earlier per-call "
              "build's 3.78 s served step. The units are ALU-bound at about 10% of the tensor peak "
              "(r20261027-022021-3a48). The commit's own overhead is a separate line: +0.42 s at 1 KiB rows "
              "(r20261027-004614-d98c).")


def _noise() -> dict:
    return {k: {"value": v, "source": s} for k, (v, s) in NOISE.items()}


def test_per_call_noise_is_seconds_a_step_with_both_ratios_never_gamma_step_alone() -> None:
    """γ_step 1.115 is against a commit-bound 74.4 s step, so the line states +8.5 s a step
    with 1.115× and about 3.2× against the plain 3.78 s step, in M3's part 2 and in Demo A's."""
    doc = _proved()
    doc["values"] = _noise()
    doc["sections"][0]["serve_noise"] = True
    out = _draft([], doc)
    part2 = out.split("## 2. Commit overhead")[1].split("\n## ")[0]
    assert f"- {NOISE_LINE}" in part2 and "| ⟨SERVE_NOISE_RATIO_4562⟩ | 3.2 |" in out
    demo = _draft([_load("block-record-r2-period0")], {"format": D.MEASUREMENTS, "values": _noise()})
    assert " ".join(NOISE_LINE.split()) in " ".join(demo.split())
    assert ("⟨SERVE_NOISE_RATIO_4562⟩× against the earlier per-call build's ⟨SERVE_STEP_4562_S⟩ s"
            in " ".join(_draft([_load("block-record-r2-period0")]).split()))


def test_a_row_stating_per_call_noise_alone_is_refused() -> None:
    for what in ("per-call noise's γ_step", "gamma_step at batch 4,096"):
        with pytest.raises(ValueError, match="states per-call noise on its own"):
            _draft([], _fallback(what=what, value=1.115, unit="×", target={"none": "a ratio"}))
    doc = {"format": D.MEASUREMENTS, "values": {**_noise(), "SERVE_NOISE_S": {"value": 8.5, "source": "kraqwxsc"}}}
    with pytest.raises(ValueError, match="names its two runs"):
        _draft([_load("block-record-r2-period0")], doc)
