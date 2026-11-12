"""The `resolve-log` command (`verity_vllm.observe.fold.resolve_log`) on a recorded log: the lines it prints, the report it
writes as JSON and as Markdown, its exit status, and the trees `--dump-trees` prints.

The log is `tests/engine/fixtures/qwen3_30b_a3b_tp2_moe_l24.rank0.jsonl.gz`: one decoder layer's MoE scope and its all-reduce,
for the prefill step and the first decode step of rank 0 of a recorded TP2 capture (`tests/engine/test_gen_ov_moe_qwen3_recorded.py`
says what it holds).  It is resolved under that rank's derived profile, and once under the same profile without its patterns, which
leaves every root Unsupported.
"""
from __future__ import annotations

import dataclasses
import importlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    Root,
    RunHeader,
    StepEnd,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.fold import resolve_log as RL
from verity_vllm.observe.fold.resolver import Unsupported
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.log import read_log, split_steps
from verity_vllm.pipeline import cli

FIXTURE = Path(__file__).parents[1] / "engine" / "fixtures" / "qwen3_30b_a3b_tp2_moe_l24.rank0.jsonl.gz"
RANK_PROFILE = "derived_QWEN3_30B_A3B_tp2"
Q3 = importlib.import_module(f"verity_vllm.observe.fold.profiles.{RANK_PROFILE}")
#: `--profile` takes a dotted path too: this module, whose `PROFILE` is the rank's profile without a pattern
PROFILE = dataclasses.replace(Q3.PROFILE, patterns=[])
check_header = Q3.check_header

DOT = "HopperBF16WgmmaDot16_v1"
#: specialization -> (rows of the prefill step, rows of the decode step, the module that defines it)
SPECS = {"AllReduce2_v1{N=2048}": (256, 1, "verity.definitions.registry.b1_tp2"),
         f"Gemm_v2{{K=2048,N=128,DOT={DOT}}}": (256, 1, "verity.definitions.ops.gemm"),
         f"MoeExpertGemmW_v2{{E=128,K=384,N=2048,DOT={DOT}}}": (2048, 8, "verity.definitions.registry.moe"),
         f"MoeExpertGemm_v2{{E=128,K=2048,N=768,DOT={DOT}}}": (2048, 8, "verity.definitions.registry.moe"),
         "MoeRouterTopKOrderedNorm_v1{E=128,TOPK=8,VPT=8}": (256, 1, "verity.definitions.registry.moe"),
         "MoeSum_v1{TOPK=8,H=2048}": (256, 1, "verity.definitions.registry.moe"),
         "SiluMul_v2{I=384}": (2048, 8, "verity.definitions.registry.act")}
FAMILIES = {k.split("{")[0]: v[:2] for k, v in SPECS.items()}
SCOPE, REDUCE = "vllm.moe_forward.default", "vllm.all_reduce.default"


def _run(tmp_path: Path, *flags: str, profile: str = RANK_PROFILE) -> tuple[int, Path]:
    base = tmp_path / "report"
    return cli.main(["resolve-log", str(FIXTURE), "--profile", profile, "--out", str(base), *flags]), base


def test_the_command_prints_each_step_then_the_histogram_and_the_report(tmp_path, capsys):
    rc, base = _run(tmp_path, "--strict")
    assert rc == 0
    lines = capsys.readouterr().out.split("\n")
    for step, (line, phase, events, transparent) in enumerate(zip(lines, ("prefill", "decode "), (41, 39), (4, 3))):
        fam = " ".join(f"{k}={v[step]}" for k, v in FAMILIES.items())
        assert re.fullmatch(rf"step   {step} {phase} rows={(256, 1)[step]:<3} roots=2    events={events}    computes=7    "
                            rf"transparent={transparent}    unsupported=0   \d+\.\d\ds  {re.escape(fam)}", line), line
    assert lines[2:4] == ["", "## op histogram"]
    assert lines[4] == f"  {'aten.slice.Tensor':<60} {16:>6}  consumed by moe.fused_experts.pairwise ×16"
    assert f"  {REDUCE:<60} {2:>6}  Compute AllReduce2_v1{{N=2048}} ×2" in lines
    assert f"  {SCOPE:<60} {2:>6}  Group of 10: 6 Compute, 4 Transparent ×1; Group of 9: 6 Compute, 3 Transparent ×1" in lines
    primary, expanded = lines.index("## ResolutionReport (primary resolutions)"), lines.index("## expanded (grouped members)")
    assert lines[primary - 1] == lines[expanded - 1] == ""
    rep = json.loads(Path(f"{base}.json").read_text())
    assert json.loads("\n".join(lines[primary + 1:expanded - 1])) == rep["primary_report"]
    assert json.loads("\n".join(lines[expanded + 1:-3])) == {k: rep[k] for k in ("by_spec", "transparent_effects", "totals", "roots",
                                                                                 "header_mismatches")}
    assert lines[-3:] == ["", f"wrote {base}.json and {base}.md", ""]
    assert "## Unsupported examples" not in lines


def test_the_report_counts_the_steps_the_specializations_and_the_roots_and_names_the_target(tmp_path, capsys):
    rc, base = _run(tmp_path, "--quiet")
    assert rc == 0 and capsys.readouterr().out == ""
    rep = json.loads(Path(f"{base}.json").read_text())
    assert (rep["log"], rep["profile"], rep["log_profile_id"]) == (str(FIXTURE), Q3.PROFILE.id, "3d2f260c76c1")
    assert rep["header_mismatches"] == [] and rep["quarantine_specs"] == [] and rep["unsupported_examples"] == []
    assert rep["totals"] == {"steps": 2, "unsupported": 0, "errors": 0, "computes": 14, "compute_rows": 7196, "transparent": 7}
    assert rep["by_spec"] == {k: {"computes": 2, "rows": a + b, "module": m} for k, (a, b, m) in SPECS.items()}
    assert rep["spec_modules"] == {k: m for k, (_a, _b, m) in SPECS.items()} and rep["transparent_effects"] == {"alloc": 6, "noop": 1}
    for step, (s, phase, events, transparent) in enumerate(zip(rep["steps"], ("prefill", "decode"), (41, 39), (4, 3))):
        assert isinstance(s.pop("seconds"), float)
        assert s == {"step": step, "phase": phase, "rows": (256, 1)[step], "roots": 2, "events": events, "computes": 7,
                     "transparent": transparent, "unsupported": 0, "error": None, "spec_rows": {k: v[step] for k, v in SPECS.items()},
                     "spec_computes": dict.fromkeys(SPECS, 1), "family_rows": {k: v[step] for k, v in FAMILIES.items()}}
    assert rep["op_histogram"][SCOPE] == {"Group of 10: 6 Compute, 4 Transparent": 1, "Group of 9: 6 Compute, 3 Transparent": 1}
    assert rep["op_histogram"][REDUCE] == {"Compute AllReduce2_v1{N=2048}": 2} and len(rep["op_histogram"]) == 17
    assert rep["primary_report"]["by_pattern"] == {"moe.fused_experts.pairwise": 2, "tp.collective": 2}
    assert rep["primary_report"]["by_spec"] == {k: a + b for k, (a, b, _m) in SPECS.items()}
    assert rep["roots"] == {"classes": {"weight": 3}, "unknown": []}
    assert [n["text"] for n in rep["notes"]] == ["moe_scope_cut"] and rep["notes"][0]["data"]["layer"] == 24
    target = rep["target"]
    assert target["host_gpu"] == {"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "compute_capability": "12.0",
                                  "multi_processor_count": 188}
    assert target["target_profile"]["compute_capability"] == [12, 0] and target["target_profile"]["num_sms"] == 188
    pinned = target["num_sms_pinned_from_target"]
    assert {k: pinned[k] for k in ("launches", "steps", "pinned", "profile_default")} == {"launches": 2, "steps": 2, "pinned": 188,
                                                                                         "profile_default": 128}
    assert pinned["source"].startswith("RunHeader.profile.gpu.multi_processor_count") and pinned["basis"].startswith("X-01")
    gemm = target["gemm_target"]
    assert {k: gemm[k] for k in ("target", "dot", "refusal", "compute_capability", "launches", "steps")} == {
        "target": "blackwell_consumer", "dot": "HopperBF16WgmmaDot16", "refusal": None, "compute_capability": [12, 0], "launches": 4,
        "steps": 2}


def test_the_markdown_report_is_the_same_report_as_tables(tmp_path):
    rc, base = _run(tmp_path, "--quiet")
    md = Path(f"{base}.md").read_text().split("\n")
    assert md[:5] == [f"# Resolution report: `{FIXTURE.name}` × profile `{Q3.PROFILE.id}`", "",
                      "- log profile_id: `3d2f260c76c1`; header mismatches vs C: none",
                      "- steps 2, computes 14 (7196 rows), transparent 7, **Unsupported 0**, resolution errors 0",
                      "- roots: {'weight': 3}"]
    at = md.index("| specialization | computes | rows |")
    assert md[at + 2:at + 2 + len(SPECS)] == [f"| `{k}` | 2 | {a + b} |" for k, (a, b, _m) in SPECS.items()]
    assert "Transparent effects: {'alloc': 6, 'noop': 1}" in md
    at = md.index("| step | phase | rows | roots | events | computes | transparent | unsupported | family rows |")
    assert md[at + 2:at + 4] == [
        f"| {step} | {phase} | {rows} | 2 | {events} | 7 | {transparent} | 0 | " + ", ".join(f"{k}={v[step]}" for k, v in FAMILIES.items()) + " |"
        for step, (phase, rows, events, transparent) in enumerate((("prefill", 256, 41, 4), ("decode", 1, 39, 3)))]
    at = md.index("| op | count | resolutions |")
    assert md[at + 2] == "| `aten.slice.Tensor` | 16 | consumed by moe.fused_experts.pairwise ×16 |"
    assert f"| `{REDUCE}` | 2 | Compute AllReduce2_v1{{N=2048}} ×2 |" in md and len(md[at + 2:md.index("", at)]) == 17
    assert md[md.index("## Notes in the log (preamble)") + 2].startswith("- moe_scope_cut: `{")
    assert not any(line.startswith("## Unsupported") for line in md)


def test_one_step_is_resolved_alone(tmp_path, capsys):
    rc, base = _run(tmp_path, "--step", "1")
    assert rc == 0
    assert [line.split()[:3] for line in capsys.readouterr().out.split("\n") if line.startswith("step ")] == [["step", "1", "decode"]]
    rep = json.loads(Path(f"{base}.json").read_text())
    assert [s["step"] for s in rep["steps"]] == [1] and rep["totals"]["compute_rows"] == sum(b for _a, b, _m in SPECS.values())
    assert rep["target"]["num_sms_pinned_from_target"]["launches"] == 1 and rep["target"]["gemm_target"]["launches"] == 2


def test_strict_fails_a_log_with_a_root_no_pattern_claims_and_the_report_gives_it_as_an_example(tmp_path, capsys):
    """Under this module's `PROFILE`, the rank's without a pattern, both roots of both steps are Unsupported."""
    rc, base = _run(tmp_path, "--strict", profile=__name__)
    assert rc == 1
    lines = capsys.readouterr().out.split("\n")
    assert [line.split("unsupported=")[1].split()[0] for line in lines if line.startswith("step ")] == ["2", "2"]
    assert f"  {SCOPE:<60} {2:>6}  Unsupported: no pattern ×2" in lines
    rep = json.loads(Path(f"{base}.json").read_text())
    examples = rep["unsupported_examples"]
    assert [(u["step"], u["id"], u["op"], u["reason"], u["detail"]["phase"], len(u["detail"]["children"])) for u in examples] == [
        (0, 2301, SCOPE, "no pattern", "forward", 39), (0, 2341, REDUCE, "no pattern", "forward", 0),
        (1, 6658, SCOPE, "no pattern", "forward", 37), (1, 6696, REDUCE, "no pattern", "forward", 0)]
    assert lines[lines.index("## Unsupported examples") + 1:-3] == [
        f"  step {u['step']} id {u['id']} {u['op']}: no pattern  {json.dumps(u['detail'])[:400]}" for u in examples]
    assert lines[lines.index("## Unsupported examples") + 2] == f'  step 0 id 2341 {REDUCE}: no pattern  {{"phase": "forward", "children": []}}'
    assert rep["totals"] == {"steps": 2, "unsupported": 4, "errors": 0, "computes": 0, "compute_rows": 0, "transparent": 0}
    assert rep["by_spec"] == {} and rep["target"]["gemm_target"] is None
    assert rep["target"]["num_sms_pinned_from_target"] == {"launches": 0, "steps": 0}
    md = Path(f"{base}.md").read_text().split("\n")
    at = md.index("## Unsupported (examples, ≤3 per op)")
    assert md[at + 4:at + 6] == [f"- step 0 id 2341 `{REDUCE}`: no pattern", '  - `{"phase": "forward", "children": []}`']
    assert "- steps 2, computes 0 (0 rows), transparent 0, **Unsupported 4**, resolution errors 0" in md
    assert _run(tmp_path, "--quiet", profile=__name__)[0] == 0, "without --strict an Unsupported root is reported, not failed"


def _view(alloc: int, shape: tuple[int, ...] = (4, 8)) -> ViewDesc:
    return ViewDesc(alloc, 0, shape, (shape[-1], 1), "bfloat16", "cuda:0")


def test_a_report_without_a_header_or_without_a_gpu_document_names_no_target_and_classes_each_root():
    weight, other = "model.layers.24.mlp.gate.weight", "no.such.weight"
    roots = {1: [Root(_view(1), "weight", weight)], 2: [Root(_view(2), "weight", other), Root(_view(3), "buffer", other)],
             4: [Root(_view(4), "prompt", "r0")]}
    none = {"host_gpu": {"name": None, "compute_capability": None, "multi_processor_count": None}, "target_profile": None,
            "num_sms_pinned_from_target": {"launches": 0, "steps": 0}, "gemm_target": None}
    rep = RL.build_report(None, [], roots, {}, Q3.PROFILE, Q3, "x.jsonl.gz", [])
    assert rep["target"] == none and rep["log_profile_id"] is None and rep["header_mismatches"] == []
    assert rep["roots"] == {"classes": {"weight": 1, "unknown": 2, "prompt": 1}, "unknown": [other, other]}
    assert rep["totals"] == {"steps": 0, "unsupported": 0, "errors": 0, "computes": 0, "compute_rows": 0, "transparent": 0}
    recorded = next(e for e in read_log(FIXTURE) if isinstance(e, RunHeader))
    documents: list[dict] = [{}, {"gpu": None}, {"gpu": {}}, {k: v for k, v in recorded.profile.items() if k != "gpu"}]
    for document in documents:
        header = dataclasses.replace(recorded, profile=document)
        assert RL.build_report(header, [], {}, {}, Q3.PROFILE, Q3, "x.jsonl.gz", [])["target"] == none
    # a profile module with no `root_class`: a weight is one the profile gives a field
    bare = type("Module", (), {})()
    assert RL.build_report(None, [], roots, {}, Q3.PROFILE, bare, "x.jsonl.gz", [])["roots"] == rep["roots"]


def _with_profile(where: str, value: Any) -> RunHeader:
    """The recorded header with `value` as its profile, or as its profile's `gpu` document."""
    recorded = next(e for e in read_log(FIXTURE) if isinstance(e, RunHeader))
    return dataclasses.replace(recorded, profile=value if where == "profile" else {"gpu": value})


@pytest.mark.parametrize("where", ["profile", "gpu"])
@pytest.mark.parametrize("value", [None, 0, "", [], ()], ids=["None", "zero", "an empty string", "an empty list", "an empty tuple"])
def test_the_target_of_a_header_whose_profile_or_gpu_document_is_empty_names_no_device(where, value):
    """The target's facts test the header's profile and its `gpu` document for truth, so an empty value of any type reads as
    a header that carried no device."""
    assert RL._target_facts(_with_profile(where, value), []) == {
        "host_gpu": {"name": None, "compute_capability": None, "multi_processor_count": None}, "target_profile": None,
        "num_sms_pinned_from_target": {"launches": 0, "steps": 0}, "gemm_target": None}


@pytest.mark.parametrize("where", ["profile", "gpu"])
@pytest.mark.parametrize("value", [[("name", "x")], (("name", "x"),), "abc", 7],
                         ids=["pairs in a list", "pairs in a tuple", "a string", "a number"])
def test_the_target_of_a_header_whose_profile_or_gpu_document_is_no_mapping_raises_for_its_get(where, value):
    """A profile or a `gpu` document that is not empty is asked for `get` as it is: it is not copied into a dict first, so
    pairs raise like a string or a number."""
    with pytest.raises(AttributeError, match=f"^'{type(value).__name__}' object has no attribute 'get'$"):
        RL._target_facts(_with_profile(where, value), [])


def test_the_trees_of_the_recorded_decode_step_are_dumped_with_what_each_root_resolved_to():
    _header, results, _roots, _notes = RL.resolve_log(FIXTURE, Q3.PROFILE)
    se = [step_events(step) for _pre, step in split_steps(read_log(FIXTURE)) if step][1]
    out = io.StringIO()
    RL.dump_trees(se, results[1].resolutions, 0, out=out)
    lines = out.getvalue().split("\n")
    assert lines[0] == "# step 1 phase=decode rows=1 roots=2 logits_rows=1 block_size=16"
    assert lines[1] == f"# input_ids_view={se.start.input_ids_view!r} positions_view={se.start.positions_view!r}"
    module = "model.layers.24.mlp.experts"
    assert lines[2] == f"[0] phase=forward module={module}  => Group of 9: 6 Compute, 3 Transparent"
    assert lines[3] == se.roots[0].render(1, 0) and lines[3].startswith(f"  {SCOPE}  in=[")
    assert lines[4].startswith("    schema=vllm::moe_forward(") and lines[5].startswith("    scalar args=[None, None, ")
    assert lines[6:] == [f"[1] phase=forward module={module}  => Compute AllReduce2_v1{{N=2048}}", se.roots[1].render(1, 0),
                         "    schema=vllm::all_reduce(Tensor tensor, str group_name) -> Tensor", "    scalar args=['tp:0'] kwargs={}", ""]


def _launch(event_id: int, alloc: int) -> TritonLaunch:
    return TritonLaunch(event_id, None, 3, "forward", "pkg.kernel", "sha", {"x_ptr": _view(alloc), "n": 4}, {"BLOCK": 32}, [4, 1, 1], 8, 3, 7,
                        "model.norm")


def test_a_dump_prints_a_launch_with_its_arguments_and_a_repeated_shape_once():
    add = "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
    se = step_events([
        StepStart(3, [("r0", 0), ("r0", 1)], [5, 6], [0, 1], {"r0": 2}, {"r0": 0}, {"r0": [0]}, 16, [0, 1], [("r0", 1)], [1], 1, 2, 2,
                  phase_hint="prefill"),
        DispatchEnter(10, None, 3, "forward", "aten.add.Tensor", add, [_view(1), 2.5], {"alpha": 1}, 7, "model.layers.0"),
        DispatchExit(10, _view(2)),
        _launch(11, 2),
        _launch(12, 3),
        DispatchEnter(13, None, 3, "sample", "aten.argmax.default", "sig", [_view(4)], {}, 7),
        DispatchExit(13, _view(5)),
        StepEnd(3)])
    out = io.StringIO()
    RL.dump_trees(se, [Unsupported(10, "aten.add.Tensor", "no pattern")], 6, out=out)
    assert out.getvalue().split("\n") == [
        "# step 3 phase=prefill rows=2 roots=4 logits_rows=1 block_size=16",
        "# input_ids_view=None positions_view=None",
        "[0] phase=forward module=model.layers.0  => Unsupported: no pattern",
        f"  aten.add.Tensor  in={[repr(_view(1))]} out={[repr(_view(2))]}",
        f"    schema={add}",
        "    scalar args=[2.5] kwargs={'alpha': 1}",
        "[1] phase=forward module=model.norm",
        "  Triton pkg.kernel grid=[4, 1, 1] const={'BLOCK': 32}",
        f"    args={{'x_ptr': {repr(_view(2))!r}, 'n': 4}} num_warps=8 num_stages=3",
        "[2] pkg.kernel  (same shape as root #1; 1 events)",
        "[3] phase=sample module=None",
        f"  aten.argmax.default  in={[repr(_view(4))]} out={[repr(_view(5))]}",
        "    schema=sig",
        ""]
    without = io.StringIO()
    RL.dump_trees(se, None, 6, out=without)
    assert without.getvalue() == out.getvalue().replace("  => Unsupported: no pattern", "")


def test_dump_trees_dumps_the_step_asked_for_and_the_first_without_one(tmp_path, monkeypatch, capsys):
    dumped: list[tuple[int, int, int]] = []
    monkeypatch.setattr(RL, "dump_trees", lambda se, res, max_depth: dumped.append((se.start.step, len(res), max_depth)))
    assert _run(tmp_path, "--dump-trees")[0] == 0 and dumped == [(0, 2, 6)]
    lines = capsys.readouterr().out.split("\n")
    assert lines[0].startswith("step   0 prefill") and lines[1:3] == ["", ""] and lines[3].startswith("step   1 decode")
    assert _run(tmp_path, "--dump-trees", "--step", "1", "--max-depth", "2")[0] == 0 and dumped[1:] == [(1, 2, 2)]
