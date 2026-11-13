"""`properties.noninterference`'s comparison on written runs, with no engine: what the observer's log and snapshot index give for each
boundary, what the hooks' record gives, and what `compare` and its Markdown say of the two.

A value's digest here is its name ("hidden0.1" is step 0's hidden state of layer 1): the comparison only tests digests for equality.
"""
from __future__ import annotations

import gzip
import itertools
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.observe.events import (
    DispatchEnter,
    DispatchExit,
    Event,
    Note,
    SampleEnd,
    StepEnd,
    StepStart,
    TritonLaunch,
    ViewDesc,
)
from verity_vllm.observe.log import LogWriter
from verity_vllm.properties import noninterference as NI

LAYERS = 2
TOKENS = [7, 9]
NORM = "_C.fused_add_rms_norm.default"


def _start(step: int, reqs: tuple[str, ...] = ("r0",), named: bool = True) -> StepStart:
    n = len(reqs)
    rows = [(r, step) for r in reqs]
    return StepStart(step, rows, [1] * n, [step] * n, {}, {}, {}, 16, [0] * n, rows if named else [], list(range(n)), n, n, n)


def _launch(i: int, step: int, phase: str, module: str | None) -> TritonLaunch:
    return TritonLaunch(i, None, step, phase, "vllm.gemm.matmul_kernel_persistent", "", {}, {}, [1], None, None, 0, module)


def _call(i: int, step: int, op: str, module: str, args: list | None = None, outs: Any = None) -> list[Event]:
    return [DispatchEnter(i, None, step, "forward", op, "", args or [], {}, 0, module), DispatchExit(i, outs)]


def _view(alloc: int) -> ViewDesc:
    return ViewDesc(alloc, 0, (1, 4), (4, 1), "bfloat16", "cuda:0")


def _entry(step: int, event: int, path: str, sha: str) -> dict:
    return {"step": step, "event": event, "path": path, "sha256": sha}


def _llama_step(s: int, ids: Iterator[int], sampled: list[int], reqs: tuple[str, ...] = ("r0",), named: bool = True,
                layers: int = LAYERS) -> tuple[list[Event], list[dict]]:
    """One Llama-shaped step of the observer's side and its snapshot entries: per layer the down projection's launch and the fused norm,
    then the final norm, the lm_head launch and the sample."""
    events: list[Event] = [_start(s, reqs, named)]
    entries: list[dict] = []
    for layer in range(layers):
        events.append(_launch(i := next(ids), s, "forward", f"model.layers.{layer}.mlp.down_proj"))
        entries.append(_entry(s, i, "args.c_ptr", f"hidden{s}.{layer}"))
        events += _call(i := next(ids), s, NORM, f"model.layers.{layer}.post_attention_layernorm")
        entries.append(_entry(s, i, "args[1]", f"residual{s}.{layer}"))
    events += _call(i := next(ids), s, NORM, "model.norm")
    entries.append(_entry(s, i, "args[0]", f"final{s}"))
    events.append(_launch(i := next(ids), s, "sample", "logits_processor"))
    entries.append(_entry(s, i, "args.c_ptr", f"logits{s}"))
    return [*events, SampleEnd(s, None, sampled), StepEnd(s)], entries


def _record(module: str, index: int, sha: str, contiguous: bool = True) -> dict:
    return {"module": module, "index": index, "shape": [1, 4], "contiguous": contiguous, "sha256": sha}


def _hooks_step(k: int, s: int | None, new_tokens: dict, layers: int = LAYERS) -> dict:
    """Engine step ``k`` of the hooks' side: device step ``s``'s records (none where the engine step launched nothing) and the tokens
    the engine reported at it, which the device step before sampled."""
    records: list[dict] = []
    if s is not None:
        for layer in range(layers):
            records += [_record(f"model.layers.{layer}", 0, f"hidden{s}.{layer}"), _record(f"model.layers.{layer}", 1, f"residual{s}.{layer}")]
        records += [_record("model.norm", 0, f"final{s}"), _record("logits_processor", 0, f"logits{s}")]
    return {"engine_step": k, "step": s, "records": records, "new_tokens": new_tokens}


def _hooks(steps: list[dict]) -> dict:
    return {"steps": steps, "hooked_modules": ["model.layers.0", "model.layers.1", "model.norm", "logits_processor", "lm_head"],
            "hook_fire_counts": {"model.layers.0": 2, "model.layers.1": 2, "model.norm": 2, "lm_head": 0}, "method": "forward hooks alone"}


def _tokens(generated: list[int] | None = None, **more: Any) -> dict:
    request = {"request_id": "r0", "generated_token_ids": TOKENS if generated is None else generated, "max_tokens": 2}
    return {"requests": [request], "timing": {"engine_steps": 2}, **more}


def _write(d: Path, name: str, doc: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(doc))
    return d / name


def _observer(d: Path, events: list[Event], entries: list[dict]) -> Path:
    """(a)'s directory under ``d``: the log, with a note after its last step, and the snapshot index."""
    with LogWriter(d / "a_observer" / "log.jsonl.gz") as w:
        for ev in [*events, Note("closed")]:
            w.write(ev)
    _write(d / "a_observer", "snapshots.json", {"entries": entries})
    return d / "a_observer"


def _run(d: Path, layers: tuple[int, int] = (LAYERS, LAYERS)) -> Path:
    """A two-step run whose three arms agree, step ``s`` of ``layers[s]`` layers on both sides: with two layers a step, event ids 1 to 6
    are step 0's and 7 to 12 step 1's."""
    ids = itertools.count(1)
    events: list[Event] = []
    entries: list[dict] = []
    for s, token in enumerate(TOKENS):
        step, snapshots = _llama_step(s, ids, [token], layers=layers[s])
        events += step
        entries += snapshots
    _observer(d, events, entries)
    _write(d / "b_hooks", "hooks.json", _hooks([_hooks_step(0, 0, {}, layers[0]), _hooks_step(1, 1, {"r0": [7]}, layers[1]),
                                                _hooks_step(2, None, {"r0": [9]})]))
    for arm in ("a_observer", "b_hooks", "c_bare"):
        _write(d / arm, "tokens.json", _tokens())
    return d


def _edit(path: Path, change: Any) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def _one_step(d: Path, events: list[Event], entries: list[dict]) -> dict:
    """What the observer's side gives for a log of one step 0 that ``events`` fill."""
    hashes, problems = NI.observer_boundary_hashes(_observer(d, [_start(0), *events, StepEnd(0)], entries))
    return {**hashes["steps"][0], "problems": problems}


# ---- compare ----

def test_three_arms_that_agree_pass_with_every_boundary_compared(tmp_path):
    rep = NI.compare(str(_run(tmp_path)), None, LAYERS)
    assert rep["PASS"] is True and rep["problems"] == [] and rep["mismatches"] == []
    assert (rep["hash_comparisons"], rep["hash_equal"]) == (12, 12), "a step: two hidden states, two residuals, the final norm, the logits"
    assert rep["tokens_equal_accepted"] == {"a_observer": True, "b_hooks": True, "c_bare": True} and rep["tokens_equal_workload_accepted"] is None
    assert rep["tokens"]["accepted"] == {"r0": TOKENS}, "the bare arm's generated ids, with no control run named"
    assert (rep["expected_steps"], rep["expected_steps_source"], rep["layers"], rep["expected_layers"]) == (2, "c_bare timing.engine_steps", 2, 2)
    assert rep["steps"][1] == {"step": 1, "rows_a": 1, "rows_b": 1, "layers_a": 2, "layers_b": 2, "hidden_eq": 2, "residual_eq": 2, "final_eq": True,
                               "logits_eq": True, "compared": 6, "equal": 6, "noncontiguous_b": [], "sampled_a": [9], "sampled_b": [9],
                               "sampled_eq_by_request": True, "batch_order_is_request_order": True, "sampled_eq": True}
    assert rep["hooked_modules_b"] == _hooks([])["hooked_modules"] and rep["variants"]["b_hooks"] == "forward hooks alone"
    assert rep["schema"] == "verity-capture/noninterference/v1" and rep["dir"] == str(tmp_path)


@pytest.mark.parametrize("module,index,tensor,layer", [("model.layers.1", 0, "hidden", 1), ("model.layers.0", 1, "residual", 0),
                                                       ("model.norm", 0, "final", None), ("logits_processor", 0, "logits", None)])
def test_a_boundary_the_hooks_hash_otherwise_is_named_and_fails(tmp_path, module, index, tensor, layer):
    def change(doc):
        for r in doc["steps"][1]["records"]:
            if (r["module"], r["index"]) == (module, index):
                r["sha256"] = "other"

    _edit(_run(tmp_path) / "b_hooks" / "hooks.json", change)
    rep = NI.compare(str(tmp_path), None, LAYERS)
    name = f"{tensor}1" if layer is None else f"{tensor}1.{layer}"
    assert rep["mismatches"] == [{"step": 1, "layer": layer, "tensor": tensor, "a": name, "b": "other"}]
    assert rep["PASS"] is False and rep["problems"] == [] and (rep["hash_comparisons"], rep["hash_equal"]) == (12, 11)


def test_a_boundary_the_observer_has_no_snapshot_of_is_a_problem_and_a_mismatch(tmp_path):
    _edit(_run(tmp_path) / "a_observer" / "snapshots.json", lambda doc: doc["entries"].pop(2))     # event 3: layer 1's down projection, step 0
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["problems"] == ["step 0: no snapshot for down_proj launch id 3 (layer 1)"]
    assert rep["mismatches"] == [{"step": 0, "layer": 1, "tensor": "hidden", "a": None, "b": "hidden0.1"}] and rep["PASS"] is False
    assert rep["layers"] == ["(1, 2)", "(2, 2)"], "the steps do not show one layer count"


def test_the_layer_count_is_held_to_the_expected_one(tmp_path):
    unknown, other = NI.compare(str(_run(tmp_path)), None), NI.compare(str(tmp_path), None, 30)
    assert unknown["problems"] == ['expected_layers unknown (no --case / --expected-layers): gate I3 needs the profile\'s C["NL"]']
    assert other["problems"] == ['decoder layers per step 2 != profile C["NL"] 30']
    assert unknown["PASS"] is False and other["PASS"] is False and unknown["layers"] == other["layers"] == 2
    assert (unknown["hash_comparisons"], unknown["hash_equal"]) == (12, 12), "every digest and token agrees; the layer count alone fails it"


def test_a_run_of_one_layer_passes_and_one_whose_steps_differ_in_their_layer_count_does_not(tmp_path):
    rep = NI.compare(str(_run(tmp_path / "one", layers=(1, 1))), None, 1)
    assert rep["PASS"] is True and rep["layers"] == 1 and (rep["hash_comparisons"], rep["hash_equal"]) == (8, 8)
    rep = NI.compare(str(_run(tmp_path / "uneven", layers=(2, 1))), None, LAYERS)
    assert rep["PASS"] is False and rep["layers"] == ["(1, 1)", "(2, 2)"], "each step's two sides agree, and the steps do not"
    assert rep["problems"] == [] and rep["mismatches"] == [] and (rep["hash_comparisons"], rep["hash_equal"]) == (10, 10)


def test_a_layer_the_observer_alone_shows_at_every_step_is_no_layer_count_and_a_mismatch_a_step(tmp_path):
    def change(doc):
        for step in doc["steps"]:
            step["records"] = [r for r in step["records"] if (r["module"], r["index"]) != ("model.layers.1", 0)]

    _edit(_run(tmp_path) / "b_hooks" / "hooks.json", change)
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["layers"] == ["(2, 1)"] and rep["problems"] == [] and rep["PASS"] is False
    assert rep["mismatches"] == [{"step": s, "layer": 1, "tensor": "hidden", "a": f"hidden{s}.1", "b": None} for s in (0, 1)]


def test_a_row_names_the_records_the_hooks_hashed_from_a_tensor_that_was_not_contiguous(tmp_path):
    _edit(_run(tmp_path) / "b_hooks" / "hooks.json", lambda doc: doc["steps"][1]["records"][2].update(contiguous=False))
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["PASS"] is True and [row["noncontiguous_b"] for row in rep["steps"]] == [[], ["model.layers.1[0]"]]


@pytest.mark.parametrize("side", ["(a)", "(b)"])
def test_a_step_only_one_side_has_is_a_problem_and_a_row_of_its_own(tmp_path, side):
    _run(tmp_path)
    if side == "(a)":
        _write(tmp_path / "b_hooks", "hooks.json", _hooks([_hooks_step(0, 0, {}), _hooks_step(1, None, {"r0": [7]})]))
    else:
        step, snapshots = _llama_step(0, itertools.count(1), [7])
        _observer(tmp_path, step, snapshots)
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["problems"] == [f"step 1 present only in {side}"] and rep["steps"][1] == {"step": 1, "error": "missing on one side"}
    assert rep["PASS"] is False and (rep["hash_comparisons"], rep["hash_equal"]) == (6, 6) and rep["layers"] == ["(2, 2)", "(None, None)"]
    assert "| 1 | missing on one side | | | | | | |" in NI.compare_markdown(rep).splitlines()


def test_the_sampled_ids_are_compared_by_request_where_the_log_names_each_row(tmp_path):
    step, snapshots = _llama_step(0, itertools.count(1), [5, 4], reqs=("r1", "r0"))
    _observer(tmp_path, step, snapshots)
    hooks = _write(tmp_path / "b_hooks", "hooks.json", _hooks([_hooks_step(0, 0, {}), _hooks_step(1, None, {"r0": [4], "r1": [5]})]))
    row = NI.compare(str(tmp_path), None, LAYERS)["steps"][0]
    assert (row["sampled_a"], row["sampled_b"], row["rows_a"]) == ([5, 4], [4, 5], 2)
    assert row["sampled_eq"] is True and row["sampled_eq_by_request"] is True and row["batch_order_is_request_order"] is False
    _edit(hooks, lambda doc: doc["steps"][1]["new_tokens"].update(r1=[6]))
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["steps"][0]["sampled_eq"] is False
    assert "step 0: SampleEnd ids by request {'r1': 5, 'r0': 4} != hooks-run new tokens {'r0': 4, 'r1': 6}" in rep["problems"]


def test_the_sampled_ids_are_compared_in_order_where_the_log_names_no_row(tmp_path):
    step, snapshots = _llama_step(0, itertools.count(1), [7], named=False)
    _observer(tmp_path, step, snapshots)
    hooks = _write(tmp_path / "b_hooks", "hooks.json", _hooks([_hooks_step(0, 0, {}), _hooks_step(1, None, {"r0": [7]})]))
    row = NI.compare(str(tmp_path), None, LAYERS)["steps"][0]
    assert row["sampled_eq"] is True and "sampled_eq_by_request" not in row
    _edit(hooks, lambda doc: doc["steps"][1]["new_tokens"].update(r0=[8]))
    assert "step 0: SampleEnd ids [7] != hooks-run new tokens [8]" in NI.compare(str(tmp_path), None, LAYERS)["problems"]


def test_the_reference_tokens_are_the_control_run_s_generated_ids_not_its_authored_record(tmp_path):
    ctl = _write(tmp_path / "ctl", "tokens.json", _tokens())
    _edit(ctl, lambda doc: doc["requests"][0].update(accepted_record_token_ids=[1, 2]))
    rep = NI.compare(str(_run(tmp_path / "run")), str(ctl), LAYERS)
    arms = ("a_observer", "b_hooks", "c_bare", "accepted_ctl_file")
    assert rep["PASS"] is True and rep["tokens"]["accepted"] == {"r0": TOKENS} and rep["tokens_equal_accepted"] == dict.fromkeys(arms, True)
    assert rep["tokens_equal_workload_accepted"] == dict.fromkeys(arms, False), "the authored record is reported and gates nothing"
    _edit(ctl, lambda doc: doc["requests"][0].update(generated_token_ids=[7, 8]))
    rep = NI.compare(str(tmp_path / "run"), str(ctl), LAYERS)
    assert rep["PASS"] is False and rep["problems"] == [] and rep["hash_equal"] == 12
    assert rep["tokens_equal_accepted"] == {"a_observer": False, "b_hooks": False, "c_bare": False, "accepted_ctl_file": True}
    rep = NI.compare(str(tmp_path / "run"), str(tmp_path / "no-such-file.json"), LAYERS)
    assert rep["PASS"] is True and set(rep["tokens"]) == {"a_observer", "b_hooks", "c_bare", "accepted"}, "a control file that is absent: the bare arm's"


def test_the_authored_record_is_compared_with_each_arm_and_is_none_for_an_arm_it_cannot_be_compared_with(tmp_path):
    ctl = _write(tmp_path / "ctl", "tokens.json", _tokens())
    _edit(ctl, lambda doc: doc["requests"][0].update(accepted_record_token_ids=TOKENS))
    run = _run(tmp_path / "run")
    _edit(run / "a_observer" / "tokens.json", lambda doc: doc["requests"].append({"request_id": "r1", "generated_token_ids": [1]}))
    (run / "b_hooks" / "tokens.json").unlink()
    rep = NI.compare(str(run), str(ctl), LAYERS)
    assert rep["tokens_equal_workload_accepted"] == {"a_observer": None, "b_hooks": None, "c_bare": True, "accepted_ctl_file": True}, \
        "(a) generated for a request the record does not name, and (b) wrote no tokens"
    assert rep["tokens_equal_accepted"] == {"a_observer": False, "b_hooks": False, "c_bare": True, "accepted_ctl_file": True}
    assert rep["problems"] == ["missing tokens for b_hooks"] and rep["PASS"] is False


def test_the_expected_step_count_is_the_bare_arm_s_then_its_largest_max_tokens_then_16(tmp_path):
    bare = _run(tmp_path) / "c_bare" / "tokens.json"
    _edit(bare, lambda doc: doc.pop("timing"))
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["PASS"] is True and (rep["expected_steps"], rep["expected_steps_source"]) == (2, "max(max_tokens)")
    _edit(bare, lambda doc: doc.update(timing={"engine_steps": 3}))
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert rep["PASS"] is False and rep["problems"] == [] and rep["expected_steps"] == 3, "two steps compared where the bare arm ran three"
    bare.unlink()
    rep = NI.compare(str(tmp_path), None, LAYERS)
    assert (rep["expected_steps"], rep["expected_steps_source"]) == (16, "default") and rep["PASS"] is False
    assert rep["problems"] == ["missing tokens for c_bare",
                               "no clean reference tokens: neither --accepted-ctl (control run) nor c_bare/tokens.json has generated_token_ids"]
    assert rep["tokens_equal_accepted"] == {"a_observer": False, "b_hooks": False, "c_bare": False}


def test_the_markdown_of_a_comparison_that_passes_and_of_one_that_fails(tmp_path):
    rep = NI.compare(str(_run(tmp_path)), None, LAYERS)
    assert NI.compare_markdown(rep) == "\n".join([
        "# Non-interference: observer vs bare (M1 workload)",
        "",
        f"- compared {rep['utc']}; results in `{tmp_path}`; **PASS**",
        f"- (a) {rep['variants']['a_observer']}",
        "- (b) forward hooks alone",
        f"- (c) {rep['variants']['c_bare']}",
        "- hooks in (b): 5 modules; fire counts: {'model.layers.0': 2, 'model.norm': 2, 'lm_head': 0}",
        "",
        "## Tokens",
        "",
        "| variant | generated ids | == accepted record |",
        "| --- | --- | --- |",
        "| a_observer | `[7, 9]` | True |",
        "| b_hooks | `[7, 9]` | True |",
        "| c_bare | `[7, 9]` | True |",
        "| accepted | `[7, 9]` | (reference) |",
        "",
        "## Boundary hashes (a) vs (b)",
        "",
        "12 / 12 sha256 pairs equal (per step: 30 hidden + 30 residual + final norm + logits).",
        "",
        "| step | rows a/b | layers a/b | hidden eq | residual eq | final eq | logits eq | sampled a == b |",
        "| ---: | --- | --- | ---: | ---: | --- | --- | --- |",
        "| 0 | 1/1 | 2/2 | 2/2 | 2/2 | True | True | True ([7]) |",
        "| 1 | 1/1 | 2/2 | 2/2 | 2/2 | True | True | True ([9]) |",
        ""])
    _edit(tmp_path / "b_hooks" / "hooks.json", lambda doc: doc["steps"][1]["records"][-2].update(sha256="other"))
    text = NI.compare_markdown(NI.compare(str(tmp_path), None, 30))
    assert "**FAIL**" in text and "| 1 | 1/1 | 2/2 | 2/2 | 2/2 | False | True | True ([9]) |" in text.splitlines()
    assert text.endswith('## Mismatches\n\n- step 1 layer None final: a=final1 b=other\n\n## Problems\n\n- decoder layers per step 2 != profile C["NL"] 30\n')


@pytest.mark.parametrize("flags,code,problems", [({"expected_layers": LAYERS}, 0, []), ({"case": "B0"}, 1, ['decoder layers per step 2 != profile C["NL"] 30'])])
def test_the_compare_command_writes_the_report_and_its_markdown_and_exits_by_the_verdict(tmp_path, capsys, flags, code, problems):
    options = NI.Options(out=str(_run(tmp_path)), compare=True, accepted_ctl=str(tmp_path / "absent.json"), **flags)
    assert NI.main(options) == code
    rep = json.loads((tmp_path / "noninterference.json").read_text())
    assert rep["PASS"] is (code == 0) and rep["problems"] == problems and rep["hash_equal"] == 12
    assert (tmp_path / "noninterference.md").read_text() == NI.compare_markdown(rep)
    verdict = "PASS" if code == 0 else "FAIL"
    assert capsys.readouterr().out == f"[nonint] {verdict}: hashes 12/12, tokens {rep['tokens_equal_accepted']}, problems {problems}\n"


def test_the_compare_command_prints_the_first_five_problems(tmp_path, capsys):
    _edit(_run(tmp_path) / "a_observer" / "snapshots.json", lambda doc: doc.update(entries=[]))
    assert NI.main(NI.Options(out=str(tmp_path), compare=True, accepted_ctl=str(tmp_path / "absent.json"), expected_layers=LAYERS)) == 1
    rep = json.loads((tmp_path / "noninterference.json").read_text())
    assert len(rep["problems"]) > 5 and capsys.readouterr().out.endswith(f", problems {rep['problems'][:5]}\n")


# ---- the hooks' side ----

def test_the_hooks_record_gives_each_step_s_boundaries_and_the_tokens_the_next_engine_step_reports(tmp_path):
    steps = [_hooks_step(0, None, {}), _hooks_step(1, 0, {}), _hooks_step(2, 1, {"r0": [7]}), _hooks_step(3, None, {"r0": [9]})]
    steps[1]["records"] += [_record("model.layers.1", 2, "a third output"), _record("lm_head", 0, "never called"), _record("model.norm", 1, "second")]
    steps[2]["records"][2]["contiguous"] = False
    doc = _hooks(steps)
    out, problems = NI.hooks_boundary_hashes(_write(tmp_path / "b", "hooks.json", doc).parent)
    step0 = {"hidden": {0: "hidden0.0", 1: "hidden0.1"}, "residual": {0: "residual0.0", 1: "residual0.1"}, "final": "final0", "logits": "logits0",
             "rows": 1, "noncontiguous": []}
    assert problems == [] and out["steps"][0] == step0 and sorted(out["steps"]) == [0, 1], "an engine step with no record is no step"
    assert out["steps"][1]["noncontiguous"] == ["model.layers.1[0]"] and out["steps"][1]["hidden"][1] == "hidden1.1"
    assert out["new_tokens"] == {0: {"r0": [7]}, 1: {"r0": [9]}}, "device step s sampled what engine step s + 1 reports"
    assert (out["hooked_modules"], out["hook_fire_counts"], out["method"]) == (doc["hooked_modules"], doc["hook_fire_counts"], doc["method"])


def test_the_hooks_record_is_read_from_its_gzipped_twin_and_under_the_gpt_neox_names(tmp_path):
    records = [_record("gpt_neox.layers.0", 0, "sum0"), _record("gpt_neox.final_layer_norm", 0, "final"), _record("logits_processor", 0, "logits")]
    doc = {"steps": [{"engine_step": 0, "step": 0, "records": records, "new_tokens": {}}]}
    tmp_path.joinpath("hooks.json.gz").write_bytes(gzip.compress(json.dumps(doc).encode()))
    out, problems = NI.hooks_boundary_hashes(tmp_path)
    assert out["steps"] == {0: {"hidden": {0: "sum0"}, "residual": {}, "final": "final", "logits": "logits", "rows": 1, "noncontiguous": []}}
    assert problems == [] and out["new_tokens"] == {0: {}} and out["hooked_modules"] is None


# ---- the observer's side ----

def test_the_observer_s_side_of_a_llama_shaped_run(tmp_path):
    _run(tmp_path)
    _edit(tmp_path / "a_observer" / "snapshots.json", lambda doc: doc["entries"].extend([{"event": 1, "path": "args.c_ptr", "sha256": "no step"},
                                                                                         {"step": 0, "event": 1, "path": "args.c_ptr"}]))
    out, problems = NI.observer_boundary_hashes(tmp_path / "a_observer")
    assert problems == [] and sorted(out["steps"]) == [0, 1]
    assert out["steps"][1] == {"hidden": {0: "hidden1.0", 1: "hidden1.1"}, "residual": {0: "residual1.0", 1: "residual1.1"}, "final": "final1",
                               "logits": "logits1", "rows": 1}
    assert out["sample_ids"] == {0: [7], 1: [9]} and out["sample_rows"] == {0: ["r0"], 1: ["r0"]}


def test_a_step_that_shows_no_boundary_has_none_and_no_problem(tmp_path):
    assert _one_step(tmp_path, [], []) == {"hidden": {}, "residual": {}, "final": None, "logits": None, "rows": 1, "problems": []}


def test_a_moe_layer_s_hidden_state_is_the_experts_op_s_output(tmp_path):
    events = [*_call(1, 0, "vllm.moe_forward.default", "model.layers.0.mlp.experts"), *_call(2, 0, "vllm.moe_forward_shared.default", "model.layers.1.mlp.experts"),
              *_call(3, 0, "vllm.moe_forward.default", "model.layers.1.mlp.experts"), *_call(4, 0, "vllm.moe_forward.default", "model.layers.3.mlp.experts"),
              *_call(5, 0, "vllm.moe_forward.default", "model.layers.4.mlp")]
    rec = _one_step(tmp_path, events, [_entry(0, 1, "outs", "experts0"), _entry(0, 2, "outs", "experts1"), _entry(0, 3, "outs", "again")])
    assert rec["hidden"] == {0: "experts0", 1: "experts1"} and rec["logits"] is None and rec["final"] is None
    assert rec["problems"] == ["step 0: two MoE block outputs for layer 1", "step 0: no snapshot for moe_forward id 4 outs (layer 3)"]


def test_a_second_launch_for_one_boundary_and_a_norm_with_no_snapshot_are_problems(tmp_path):
    events = [_launch(1, 0, "forward", "model.layers.0.mlp.down_proj"), _launch(2, 0, "forward", "model.layers.0.mlp.down_proj"),
              *_call(3, 0, NORM, "model.layers.0.post_attention_layernorm"), *_call(4, 0, NORM, "model.layers.1.post_attention_layernorm"),
              *_call(5, 0, NORM, "model.norm"), _launch(6, 0, "sample", None), _launch(7, 0, "forward", "logits_processor"),
              _launch(8, 0, "forward", "model.layers.0.mlp.up_proj")]
    rec = _one_step(tmp_path, events, [*(_entry(0, i, "args.c_ptr", f"launch{i}") for i in (1, 2, 6, 8)), _entry(0, 3, "args[1]", "residual0")])
    assert rec["hidden"] == {0: "launch1"} and rec["residual"] == {0: "residual0"} and rec["final"] is None and rec["logits"] is None
    assert rec["problems"] == ["step 0: two down_proj launches for layer 0", "step 0: no snapshot for post_attention_layernorm id 4 args[1] (layer 1)",
                               "step 0: no snapshot for model.norm id 5 args[0]", "step 0: two sample-phase GEMM launches (ids ..., 7)",
                               "step 0: no snapshot for the lm_head launch id 7"]


def _unfused_step(softcap: bool = True) -> tuple[list[Event], list[dict]]:
    """A step of a family whose norms run unfused: per layer the fp32 residual sum and its bf16 cast in the pre-feedforward norm (events
    10L + 1 to 10L + 4) and two dispatches of the post-feedforward norm (10L + 5, 10L + 6); then the final norm (31, 32), the lm_head
    launch in the logits processor (33) and, with ``softcap``, the softcap there (34)."""
    events: list[Event] = []
    entries = [_entry(0, 33, "args.c_ptr", "raw logits"), _entry(0, 31, "outs", "before the final norm's last"), _entry(0, 32, "outs", "final"),
               _entry(0, 34, "outs", "softcapped")]
    for layer in range(LAYERS):
        i, pre, post = 10 * layer, f"model.layers.{layer}.pre_feedforward_layernorm", f"model.layers.{layer}.post_feedforward_layernorm"
        events += [*_call(i + 1, 0, "aten.add.Tensor", pre, [_view(100 + layer), _view(200 + layer)], _view(300 + layer)),
                   *_call(i + 2, 0, "aten.add.Tensor", pre, [_view(300 + layer), 1.0], _view(400 + layer)),
                   *_call(i + 3, 0, "aten.to.dtype", pre, [_view(300 + layer)], _view(500 + layer)),
                   *_call(i + 4, 0, "aten.to.dtype", pre, [_view(300 + layer)], _view(600 + layer)),
                   *_call(i + 5, 0, "aten.mul.Tensor", post), *_call(i + 6, 0, "aten.to.dtype", post)]
        entries += [_entry(0, i + 3, "outs", f"residual{layer}"), _entry(0, i + 4, "outs", "a later cast"), _entry(0, i + 5, "outs", "not the last"),
                    _entry(0, i + 6, "outs", f"hidden{layer}")]
    events += [*_call(31, 0, "aten.mul.Tensor", "model.norm"), *_call(32, 0, "aten.to.dtype", "model.norm"), _launch(33, 0, "sample", "logits_processor")]
    if softcap:
        events += _call(34, 0, "aten.tanh.default", "logits_processor")
    return events, entries


def test_unfused_norms_give_the_first_cast_of_the_residual_sum_and_each_module_s_last_dispatch(tmp_path):
    rec = _one_step(tmp_path, *_unfused_step())
    assert rec == {"hidden": {0: "hidden0", 1: "hidden1"}, "residual": {0: "residual0", 1: "residual1"}, "final": "final", "logits": "softcapped",
                   "rows": 1, "problems": []}


def test_unfused_norms_with_no_dispatch_in_the_logits_processor_keep_the_launch_s_logits(tmp_path):
    rec = _one_step(tmp_path, *_unfused_step(softcap=False))
    assert rec["logits"] == "raw logits" and rec["final"] == "final" and rec["hidden"] == {0: "hidden0", 1: "hidden1"} and rec["problems"] == []


def test_an_unfused_boundary_with_no_snapshot_is_a_problem(tmp_path):
    events, entries = _unfused_step()
    rec = _one_step(tmp_path, events, [e for e in entries if e["event"] not in (3, 16, 32)])
    assert rec["residual"] == {0: "a later cast", 1: "residual1"}, "layer 0's first cast has no snapshot, so the next cast of the sum is taken"
    assert rec["hidden"] == {0: "hidden0"} and rec["final"] is None and rec["logits"] == "softcapped"
    assert rec["problems"] == ["step 0: no snapshot for the residual cast id 3 outs (layer 0)",
                               "step 0: no snapshot for model.layers.1.post_feedforward_layernorm id 16 outs", "step 0: no snapshot for model.norm id 32 outs"]


def _parallel_step() -> tuple[list[Event], list[dict]]:
    """A GPT-NeoX-shaped step: two residual adds in each layer's own module (events 10L + 1 and 10L + 2), an add in a module under it,
    the final norm (31) and the lm_head launch (32)."""
    events: list[Event] = []
    for layer in range(LAYERS):
        i, module = 10 * layer, f"gpt_neox.layers.{layer}"
        events += [*_call(i + 1, 0, "aten.add.Tensor", module), *_call(i + 2, 0, "aten.add.Tensor", module), *_call(i + 3, 0, "aten.add.Tensor", module + ".mlp")]
    events += [*_call(31, 0, "aten.layer_norm.default", "gpt_neox.final_layer_norm"), _launch(32, 0, "sample", None)]
    entries = [_entry(0, 1, "outs", "the first add"), _entry(0, 2, "outs", "sum0"), _entry(0, 12, "outs[0]", "sum1"), _entry(0, 3, "outs", "the mlp's"),
               _entry(0, 31, "outs", "final"), _entry(0, 32, "args.c_ptr", "logits")]
    return events, entries


def test_the_parallel_residual_family_gives_each_layer_s_last_add_and_the_final_norm(tmp_path):
    rec = _one_step(tmp_path, *_parallel_step())
    assert rec == {"hidden": {0: "sum0", 1: "sum1"}, "residual": {}, "final": "final", "logits": "logits", "rows": 1, "problems": []}


def test_a_parallel_residual_boundary_with_no_snapshot_is_a_problem(tmp_path):
    events, entries = _parallel_step()
    rec = _one_step(tmp_path, events, [e for e in entries if e["event"] not in (12, 31)])
    assert rec["hidden"] == {0: "sum0"} and rec["final"] is None
    assert rec["problems"] == ["step 0: no snapshot for the layer-1 residual add id 12 outs", "step 0: no snapshot for gpt_neox.final_layer_norm id 31 outs"]


def test_the_parallel_residual_family_s_layers_and_its_final_norm_are_each_read_without_the_other(tmp_path):
    events, entries = _parallel_step()
    norm = [e for e in events if getattr(e, "id", None) == 31]
    rec = _one_step(tmp_path / "layers", [e for e in events if e not in norm], entries)
    assert rec["hidden"] == {0: "sum0", 1: "sum1"} and rec["final"] is None and rec["logits"] == "logits" and rec["problems"] == []
    rec = _one_step(tmp_path / "norm", norm, entries)
    assert rec["hidden"] == {} and rec["final"] == "final" and rec["problems"] == []


@pytest.mark.parametrize("found,hidden", [("hidden", {0: "launch"}), ("final", {})])
def test_the_parallel_residual_rule_is_asked_only_where_the_rules_before_it_found_no_hidden_state_and_no_final_norm(tmp_path, found, hidden):
    before = [_launch(1, 0, "forward", "model.layers.0.mlp.down_proj")] if found == "hidden" else _call(1, 0, NORM, "model.norm")
    entry = _entry(0, 1, "args.c_ptr", "launch") if found == "hidden" else _entry(0, 1, "args[0]", "final")
    rec = _one_step(tmp_path, [*before, *_call(2, 0, "aten.add.Tensor", "gpt_neox.layers.0")], [entry, _entry(0, 2, "outs", "a sum no rule reads")])
    assert rec["hidden"] == hidden and rec["final"] == ("final" if found == "final" else None) and rec["problems"] == []


# ---- the token files and the case ----

def test_a_variant_s_token_file_says_which_requests_generated_the_accepted_ids(tmp_path, capsys):
    wl = {"requests": [{"request_id": "r0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "arrive_step": 1},
                       {"request_id": "r1", "prompt_token_ids": [4], "max_tokens": 2}], "serving_profile": "p", "context_class": "c",
          "expected_engine_steps": 3}
    timing = {"engine_steps": 3, "requests": {"r0": {"first_step": 1, "finish_step": 2, "finish_reason": "length"}}}
    NI._write_tokens(tmp_path, wl, {"r0": [7, 9]}, timing, {"r0": [7, 9]}, {"0": [7]}, "observer")
    doc = json.loads((tmp_path / "tokens.json").read_text())
    assert doc["requests"] == [
        {"request_id": "r0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "generated_token_ids": [7, 9], "accepted_record_token_ids": [7, 9],
         "equals_accepted": True, "prompt_len": 3, "n_generated": 2, "arrive_step": 1, "first_step": 1, "finish_step": 2, "finish_reason": "length"},
        {"request_id": "r1", "prompt_token_ids": [4], "max_tokens": 2, "generated_token_ids": [], "accepted_record_token_ids": None,
         "equals_accepted": None, "prompt_len": 1, "n_generated": 0, "arrive_step": 0, "first_step": None, "finish_step": None, "finish_reason": None}]
    assert doc["all_equal_accepted"] is False, "a request with no accepted record is not equal to one"
    assert {k: v for k, v in doc.items() if k not in ("requests", "all_equal_accepted")} == {
        "per_step_sampled": {"0": [7]}, "timing": timing, "variant": "observer", "serving_profile": "p", "context_class": "c",
        "expected_engine_steps": 3, "engine_steps": 3}
    assert capsys.readouterr().out.splitlines() == ["[nonint/observer] r0: generated [7, 9] equals_accepted=True",
                                                    "[nonint/observer] r1: generated [] equals_accepted=None"]


@pytest.mark.parametrize("accepted,equal", [({"r0": [7, 9], "r1": []}, True), ({"r0": [7, 9], "r1": [3]}, False), ({"r0": [7], "r1": []}, False)])
def test_every_request_must_equal_its_accepted_ids_for_the_file_to_say_all_do(tmp_path, accepted, equal):
    wl = {"requests": [{"request_id": r, "prompt_token_ids": [1], "max_tokens": 2} for r in ("r0", "r1")]}
    NI._write_tokens(tmp_path, wl, {"r0": [7, 9]}, {}, accepted, None, "hooks")
    doc = json.loads((tmp_path / "tokens.json").read_text())
    assert doc["all_equal_accepted"] is equal and doc["engine_steps"] is None and doc["per_step_sampled"] is None


def test_the_reference_and_the_authored_record_are_read_from_the_first_file_that_has_them(tmp_path):
    ctl = _write(tmp_path / "ctl", "tokens.json", {"requests": [{"request_id": "r0", "generated_token_ids": [5], "accepted_record_token_ids": [6]},
                                                                {"request_id": "r1", "generated_token_ids": None}]})
    _write(tmp_path / "run" / "c_bare", "tokens.json", {"requests": [{"request_id": "r0", "generated_token_ids": [8], "accepted_record_token_ids": []}]})
    run = tmp_path / "run"
    assert NI._accepted_tokens(str(ctl), run) == {"r0": [5]} and NI._workload_accepted(str(ctl), run) == {"r0": [6]}
    assert NI._accepted_tokens(None, run) == NI._accepted_tokens(str(tmp_path / "absent.json"), run) == {"r0": [8]}
    assert NI._workload_accepted(None, run) == {} and NI._accepted_tokens(None) == {} and NI._accepted_tokens(None, tmp_path) == {}
    _edit(ctl, lambda doc: doc["requests"][0].update(generated_token_ids=None, accepted_record_token_ids=None))
    assert NI._accepted_tokens(str(ctl), run) == {"r0": [8]} and NI._workload_accepted(str(ctl), run) == {}, "a control file that states none: the bare arm's"
    assert NI._tokens_of(run / "c_bare" / "tokens.json") == {"r0": [8]} and NI._tokens_of(run / "absent.json") is None
    _edit(run / "c_bare" / "tokens.json", lambda doc: doc["requests"][0].update(accepted_record_token_ids=[4]))
    assert NI._workload_accepted(None, run) == NI._workload_accepted(str(ctl), run) == {"r0": [4]}, "the bare arm's, where no control file states one"


def test_the_case_is_the_flag_s_else_the_workload_file_s_else_b0(tmp_path):
    stated = _write(tmp_path, "stated.json", {"case": "B7"})
    silent = _write(tmp_path, "silent.json", {"requests": []})
    tmp_path.joinpath("list.json").write_text("[]")
    tmp_path.joinpath("text.json").write_text("not JSON")
    assert NI.case_of_args(SimpleNamespace(case="B1", workload=str(stated))) == "B1"
    assert NI.case_of_args(SimpleNamespace(case=None, workload=str(stated))) == NI.case_of_args(SimpleNamespace(workload=str(stated))) == "B7"
    for name in (silent.name, "list.json", "text.json", "absent.json"):
        assert NI.case_of_args(SimpleNamespace(case="", workload=str(tmp_path / name))) == "B0", name


def test_the_expected_layers_are_the_case_s_profile_s_and_none_for_no_case_or_an_unknown_one():
    assert (NI.expected_layers_for("B0"), NI.expected_layers_for("B1")) == (30, 28)
    assert NI.expected_layers_for(None) is None and NI.expected_layers_for("") is None and NI.expected_layers_for("no such case") is None
