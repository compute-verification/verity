"""te_core.capture.script: a program recorded from its start and measured at one step of its history (`Trace.region`),
on recorded/islands/toy/train.py: the update's dependence and cut are the same however the program writes it (in place,
rebound, by an optimizer), a compressed update's cut is its bits, a MeZO update depends on its losses through a Python
float, and a buffer rewritten from θ and this step's tokens is not the run's; a read-back that decides what the graph
can't carry (a Python clip, a branch, a float folded over steps) is listed; a storage the program replaced is not named
again."""
from __future__ import annotations

import io
import json
import math
import os
import re
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from te_core.capture import script
from te_core.capture.trace import CaptureRefused

aten = torch.ops.aten
PROGRAM = Path(__file__).parent / "recorded" / "islands" / "toy" / "train.py"
V, H, B, CTX = 16, 8, 4, 8
T = B * CTX
VALUES = 2 * V * H + H * H                 # emb, head; fc
FWD = H * H + H * V                        # fc and head, per token (the lookup is a gather)


@pytest.fixture
def starts(tmp_path, monkeypatch):
    monkeypatch.setattr(script, "STREAM_TOKENS", 10 ** 5)
    shapes = {"emb": [V, H], "fc": [H, H], "head": [V, H], "frozen": [V, H]}
    (tmp_path / "toy.start.json").write_text(json.dumps({"config": {"vocab": V, "d_model": H},
                                                         "tensors": {n: ["float32", s] for n, s in shapes.items()}}))
    return tmp_path


def recorded(starts, *args):
    return script.record(PROGRAM, ["--batch", str(B), *args], step=1, starts=starts, wall=[0.1] * 8, step_tokens=T)


def fwd_roles(g) -> list[str]:
    return [m.role for m in g.matmuls if m.phase == "forward"]


@pytest.mark.parametrize("update", ["inplace", "rebind", "adamw", "inplace --windows", "inplace --windows --own-ctx"])
def test_a_dense_update_measures_alike_however_it_is_written(starts, update):
    """``--windows``: batches the program reads from the stream itself (numpy, then ``torch.from_numpy``), in explicit
    steps."""
    g = recorded(starts, "--update", *update.split())
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD
    head = [m for m in g.matmuls if m.phase == "forward"][-1]
    assert head.upstream_macs == H * H
    assert g.cut_bits == 32 * VALUES and g.beta == 32 * VALUES / FWD


def test_a_sign_update_with_error_feedback_cuts_at_its_bits(starts):
    g = recorded(starts, "--update", "sign-ef")
    assert g.updated_macs() == T * FWD
    assert g.cut_bits == VALUES + 3 * 32 and sorted(b for _, b in g.cut) == [32] * 3 + [H * H, V * H, V * H]


@pytest.mark.parametrize("update", ["mezo", "mezo-quant"])
def test_a_mezo_update_depends_on_its_losses_through_a_float(starts, update):
    """The update is z · lr · float(L₊ − L₋): every forward GEMM of both perturbed passes is updated, and the cut is the
    one fp32 difference; so it is with the coefficient quantized through a tensor (``mezo-quant``, as the seed
    island's `prepare.quantize_coefficient` does), which decides nothing and cuts at the fp16 value."""
    g = recorded(starts, "--update", update)
    assert fwd_roles(g) == ["updated"] * 4 and g.updated_macs() == 2 * T * FWD
    assert g.cut_bits == (16 if update == "mezo-quant" else 32)
    assert g.formalism["data_dependent"] == [] and g.formalism["carried_scalars"] == []


def line_of(text: str) -> int:
    return next(i for i, s in enumerate(PROGRAM.read_text().splitlines(), 1) if text in s)


def test_a_python_clip_on_a_read_back_estimate_is_listed_where_it_drops_the_update(starts):
    """seed-mezo's 04:15Z LOOPHOLE: max(−m, min(m, c)) on the stand-in returns the bound, so the step's update reads no
    loss and the trace shows no update (affected share 0, u(1) 0); the comparison is listed at the program's line."""
    g = recorded(starts, "--update", "mezo-clip")
    f = g.formalism
    assert f["affected_share"] == 0 and f["u1_bits"] == 0
    assert [(e["kind"], e["where"]) for e in f["data_dependent"]] == [
        ("a comparison of a read-back float", f"train.py:{line_of('max(-CLIP / LR')}")]
    assert f["data_dependent"][0]["steps"] == [0, 2]


@pytest.mark.parametrize("update, kind", [("mezo-branch", "bool() of a tensor (a branch on its value)")])
def test_a_branch_on_a_loss_comparison_is_listed_though_the_update_still_reads_the_loss(starts, update, kind):
    f = recorded(starts, "--update", update).formalism
    assert f["affected_share"] > 0 and [e["kind"] for e in f["data_dependent"]] == [kind]


def test_the_same_clip_as_a_tensor_op_is_measured(starts):
    g = recorded(starts, "--update", "mezo-clamp")
    assert g.formalism["data_dependent"] == [] and fwd_roles(g) == ["updated"] * 4 and g.formalism["u1_bits"] > 0


def test_integer_reads_numpy_and_math_on_read_backs_are_listed(starts, monkeypatch):
    rec = script._Recorder(stop=1, wall=[])
    monkeypatch.setattr(script, "_ACTIVE", rec)
    rec.step = 0
    s = script.Scalar(2.5, ("node",))
    assert isinstance(s * 2 + 1, script.Scalar) and (s * 2).deps == {"node"}
    assert float(round(s, 1)) == 2.5 and round(s, 1).deps == {"node"} and isinstance(np.sqrt(s), script.Scalar)
    assert rec.listed("data_dependent") == []
    int(s), round(s), bool(s), np.clip(s, 0, 1) > 0, script._lift_math(math.isfinite)(s)
    with FakeTensorMode(allow_non_fake_inputs=True):
        t = torch.empty((), dtype=torch.int64)
        rec.stand_in(t, "item()")
    assert [e["kind"] for e in rec.listed("data_dependent")] == [
        "int() of a read-back float", "round() of a read-back float", "a truth test of a read-back float",
        "a comparison of a read-back float", "math.isfinite of a read-back float", "item() of a int64 tensor"]
    assert script.Scalar(1.0) < 2  # a float no tensor was read for decides nothing
    assert len(rec.listed("data_dependent")) == 6


def test_a_tensor_made_where_a_replaced_storage_was_is_a_constant_of_its_own():
    """The recorder names a storage by its address, so it holds every storage it maps. A program that replaces a mapped
    tensor's storage would otherwise free it, and the allocator would give its address to a later storage: no tensor
    made afterwards lands there, and each one the recorder has not seen is a constant of its own, no view of the
    replaced tensor."""
    rec = script._Recorder(stop=1, wall=[])
    p = torch.zeros(4)
    assert rec.node_of(p).op == "get_attr"
    was = script._skey(p)
    p.data = torch.ones(4)
    fresh = [torch.full((4,), float(i)) for i in range(64)]
    assert was not in {script._skey(q) for q in fresh}
    assert {rec.node_of(q).op for q in fresh} == {"get_attr"}


@pytest.mark.parametrize("unrecorded", [True, False])
def test_e_sized_through_prepares_quant_arrays_lists_no_read_back(starts, monkeypatch, unrecorded):
    """sparse's and shallow's E sizing (`prepare._quant_array` in the steps, no encode): its range and finiteness checks
    and its bytes are E's, unrecorded as the encoders are, and the update measures as without them; recorded, the same
    reads are listed."""
    if not unrecorded:
        monkeypatch.setattr(script, "_E_PARTS", ())
    g = recorded(starts, "--update", "inplace", "--size-e")
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD
    assert sorted({e["kind"] for e in g.formalism["data_dependent"]}) == ([] if unrecorded else [
        "bool() of a tensor (a branch on its value)", "int() of a tensor", "numpy() of a tensor"])


def test_a_python_float_folded_over_steps_is_listed_and_grows_delta_grad(starts):
    """seed-mezo's 04:15Z TOOL: an EMA kept as a Python float carries every earlier step's read-back as a value of its
    own, so Δ_grad grows with the step; kept as a 0-dim tensor it is one carried value at every step."""
    def at(step, update):
        return script.record(PROGRAM, ["--batch", str(B), "--steps", "6", "--update", update], step=step, starts=starts,
                             wall=[0.1] * 8, step_tokens=T).formalism

    floats, tensor = [at(s, "mezo-ema") for s in (1, 3)], [at(s, "mezo-ema-tensor") for s in (1, 3)]
    assert floats[1]["delta_grad_bits"] > floats[0]["delta_grad_bits"]
    assert tensor[1]["delta_grad_bits"] == tensor[0]["delta_grad_bits"]
    assert [e["kind"] for e in floats[1]["carried_scalars"]] == ["a Python number read back in several earlier steps"]
    assert floats[1]["carried_scalars"][0]["earlier_steps"] == 4 and floats[1]["data_dependent"] == []
    assert tensor[1]["carried_scalars"] == [] and tensor[1]["data_dependent"] == []


def test_the_stub_session_has_the_runs_step_and_progress_bucket(starts):
    log = io.StringIO()
    script.record(PROGRAM, ["--batch", str(B), "--update", "inplace", "--session"], step=1, starts=starts,
                  wall=[0.1] * 8, step_tokens=T, log=log)
    seen = [json.loads(s) for s in log.getvalue().splitlines() if s.startswith("{")]
    assert seen[:3] == [{"step": 0, "bucket": 5}, {"step": 1, "bucket": 10}, {"step": 2, "bucket": 15}]


@pytest.mark.parametrize("source", ["theta", "trained"])
def test_a_buffer_rewritten_from_theta_and_this_steps_tokens_is_not_dependent(starts, source):
    """The side GEMM reads θ's table scaled by the buffer: not updated when the buffer comes from θ, updated when it
    comes from the trained embedding."""
    g = recorded(starts, "--update", "inplace", "--cache", source)
    assert fwd_roles(g) == ["updated", "updated", "neither" if source == "theta" else "updated"]
    assert g.cut_bits == 32 * VALUES


def test_a_plan_read_off_eval_ctx_is_recorded_again_with_only_the_windows_halved(starts):
    """Halving prepare.EVAL_CTX doubles the warmup, so only the first recording halves fc in step 1; halving the stream's
    windows alone keeps the plan."""
    g = recorded(starts, "--update", "inplace", "--warm-tokens", str(2 * T))
    assert g.axes == {"window": CTX}
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD and g.cut_bits == 32 * VALUES


def test_windows_cut_at_eval_ctx_refuse_the_windows_only_recording(starts):
    with pytest.raises(CaptureRefused, match="keep their shape"):
        recorded(starts, "--update", "inplace", "--windows", "--own-ctx", "--warm-tokens", str(2 * T))


@pytest.mark.parametrize("step_tokens", [T, B * (CTX + 2)])
def test_a_generation_rounds_decode_positions_are_on_the_token_axis(starts, step_tokens):
    """A round of 3 tokens a prompt (step 1): its B · (CTX + 2) positions are its tokens, whether its record counts
    only the prompts (te_core 3's counter) or every position; each decode product indexes its sequences, so decode
    attention is attention's (left out of M) and the decode GEMMs read no activation as a weight. A training step
    after it is recorded on ctx alone."""
    from te_core import formalism

    g = script.record(PROGRAM, ["--batch", str(B), "--update", "inplace", "--generate", "3"], step=1, starts=starts,
                      wall=[0.1] * 8, step_tokens=step_tokens)
    assert g.tokens == step_tokens and g.axes == {"ctx": CTX, "sequences": B}
    assert g.formalism["tokens"] == B * (CTX + 2) and g.formalism["token_factor"] == B * (CTX + 2) / step_tokens
    fwd = [m for m in g.matmuls if m.phase == "forward"]
    assert len(fwd) == 12 and all(m.result == "token" for m in fwd)
    decode = [m for m in fwd if m.dims["m"] == 1]
    assert len(decode) == 4 and all(sum(d in m.token_dims for d in ("m", "k", "n")) == 2 for m in decode)
    assert all("m" in m.token_dims for m in fwd if m.op == "aten.mm.default")
    rows = g.formalism["products"]
    he = formalism.he_of_rows(rows["rows"], 2 ** 20, tokens=rows["tokens"])
    assert he["left_out_macs"] == sum(m.macs for m in decode)
    assert he["trained_weights"] <= H * H + V * H
    train = script.record(PROGRAM, ["--batch", str(B), "--update", "inplace", "--generate", "3"], step=2,
                          starts=starts, wall=[0.1] * 8, step_tokens=T)
    assert train.axes == {"ctx": CTX} and train.formalism["tokens"] == T and train.updated_macs() == T * FWD


def test_a_round_whose_prompts_the_capture_cant_batch_is_refused(starts):
    """Prompts the program reads from the stream itself (``--windows``): the recording with twice the windows leaves
    them, and so the sampled ids, as they were, and the round's decode positions would be on no token axis."""
    with pytest.raises(CaptureRefused, match="sampled ids .* keep their shape"):
        recorded(starts, "--update", "inplace", "--generate", "3", "--windows")


def test_the_stub_stream_has_the_parts_its_start_read_then_an_unread_part(starts):
    """ref-data's ``--windows fresh``: a data v3 start's provenance names the parts it read (data.parts, by sha256), which
    the program finds in data_v3.TRAIN's manifest and among the stream's parts and starts, and it reads only the windows
    outside them; a start whose provenance names none is refused by the program, as on a node. prepare's data_v3.TRAIN
    is its own again after."""
    path = starts / "toy.start.json"
    toy = json.loads(path.read_text())
    parts = [{"index": 0, "sha256": "a" * 64, "tokens": 40 * CTX + 1}, {"index": 1, "sha256": "b" * 64, "tokens": 24 * CTX + 1}]
    path.write_text(json.dumps(toy | {"tokenizer": "bpe32k", "provenance": {"data": {"parts": parts}}}))
    log = io.StringIO()
    g = script.record(PROGRAM, ["--batch", str(B), "--update", "inplace", "--windows", "--fresh"], step=1, starts=starts,
                      wall=[0.1] * 8, step_tokens=T, log=log)
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD and g.cut_bits == 32 * VALUES
    seen = [json.loads(s) for s in log.getvalue().splitlines() if s.startswith("{")]
    assert seen[0] == {"parts": 3, "windows": 40 + 24 + (10 ** 5 - 1) // CTX, "seen": 40 + 24}
    assert seen[1] == {"parts": 3, "windows": 80 + 48 + (10 ** 5 - 1) // (CTX // 2), "seen": 80 + 48}
    assert sys.modules["prepare"].data_v3.TRAIN == Path("/nonexistent/train/v3")
    path.write_text(json.dumps(toy | {"tokenizer": "bpe32k", "provenance": {}}))
    with pytest.raises(CaptureRefused, match="needs the start's provenance to name the data v3 parts it read"):
        recorded(starts, "--update", "inplace", "--windows", "--fresh")


def test_the_stub_streams_unread_part_is_as_long_as_the_run_record_reads(starts):
    """hybrid's grow.py refuses an arm that reads more windows than its stream leaves unread (``--unread``): the unread
    part is the record's steps' tokens, STREAM_HEADROOM times over, at least STREAM_TOKENS; a long scoring step adds its
    own tokens, not the step count times them."""
    run = {"steps": [{"tokens": T}] * 2000 + [{"tokens": 0}]}
    assert script.stream_tokens(run) == script.STREAM_HEADROOM * 2000 * T
    scored = {"steps": [{"tokens": T}] * 2000 + [{"tokens": 1000 * T}]}
    assert script.stream_tokens(scored) == script.STREAM_HEADROOM * 3000 * T
    assert script.stream_tokens({"steps": [{"tokens": T}]}) == script.stream_tokens({}) == script.STREAM_TOKENS
    need = str((script.stream_tokens(run) - 1) // CTX)
    with pytest.raises(CaptureRefused, match=rf"train: {need} windows to read, {(10 ** 5 - 1) // CTX} in the stream"):
        recorded(starts, "--update", "inplace", "--unread", need)
    g = script.record(PROGRAM, ["--batch", str(B), "--update", "inplace", "--unread", need], step=1, starts=starts,
                      wall=[0.1] * 8, step_tokens=T, stream_tokens=script.stream_tokens(run))
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD


def test_the_stub_stream_holds_its_window_count_until_a_program_reads_its_order(starts):
    """A start whose provenance parts hold 1e10 windows (2e10 at the halved ctx) is recorded through `TrainData.next`
    with no permutation of them, 80 GB of int64: the stand-in makes prepare's order only when a program reads it."""
    path = starts / "toy.start.json"
    toy = json.loads(path.read_text())
    parts = [{"index": 0, "sha256": "a" * 64, "tokens": 10 ** 10 * CTX + 1}]
    path.write_text(json.dumps(toy | {"tokenizer": "bpe32k", "provenance": {"data": {"parts": parts}}}))
    tracemalloc.start()
    try:
        g = recorded(starts, "--update", "inplace")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert fwd_roles(g) == ["updated", "updated"] and g.updated_macs() == T * FWD
    assert peak < 2 ** 28, f"{peak:,} bytes traced at the peak"


def test_a_boolean_mask_index_is_taken_at_its_largest():
    x = torch.empty(10, 4)
    with FakeTensorMode(allow_non_fake_inputs=True):
        rows, cols = torch.empty(10, dtype=torch.bool), torch.empty(4, dtype=torch.bool)
        both = torch.empty(10, 4, dtype=torch.bool)
        shapes = [script._dynamic(aten.index.Tensor, (x, idx), {}).shape for idx in ([rows], [None, cols], [both])]
    assert shapes == [(10, 4), (10, 4), (40,)]


@pytest.mark.parametrize("update", ["inplace", "adamw", "inplace --windows"])
def test_a_programs_last_step_ends_at_its_end(starts, update):
    """The last step ends when the program returns, as te_core.run's `close` ends it, and measures as a step with one
    after it (no next step reads its new values, so only 4.0's cut is 0); a step past the program's end is refused."""
    last, mid = (recorded(starts, "--update", *update.split(), "--steps", str(n)) for n in (2, 4))
    assert last.updated_macs() == mid.updated_macs() == T * FWD and last.cut_bits == 0 < mid.cut_bits
    keep = ("sigma", "affected_share", "token_factor", "delta_grad_bits", "grad_cut_bits", "partial_cut_bits", "u1_bits")
    assert {k: last.formalism[k] for k in keep} == {k: mid.formalism[k] for k in keep}
    assert last.formalism["delta_grad_bits"] > 0
    with pytest.raises(CaptureRefused, match="the program ended after 2 steps, before step 2 ended"):
        script.record(PROGRAM, ["--batch", str(B), "--update", *update.split(), "--steps", "2"], step=2, starts=starts,
                      wall=[0.1] * 8, step_tokens=T)


@pytest.mark.parametrize("tier", ["long", "search", None])
def test_a_record_replays_at_the_tier_it_was_scored_at(starts, tmp_path, monkeypatch, tier):
    """hybrid's 16:22Z LOOPHOLE: a scored run.json keeps its tier only at ``evaluation.budget.tier``, and a program that
    plans its steps from prepare.budget (``--from-budget``: the search tier pays for 2 steps, the long tier for 200) runs
    its own steps only at that tier. A long-tier record replays to step 2; at the search tier, or with no evaluation,
    the program ends before it. CW_TIER is the environment's again after."""
    monkeypatch.delenv("CW_TIER", raising=False)
    monkeypatch.setattr(torch, "__version__", f"{script.products.TORCH}+cpu")
    run = {"steps": [{"wall_s": 0.1, "tokens": T, "matmul_macs": 0, "by": {}}] * 6, "tier": "search"}
    if tier is not None:
        run["evaluation"] = {"budget": {"tier": tier, "train_flops": 0.0}}
    (tmp_path / "run.json").write_text(json.dumps(run))
    argv = ["--script", str(PROGRAM), "--run", str(tmp_path / "run.json"), "--step", "2", "--starts", str(starts),
            "--out", str(tmp_path / "out.json"), "--", "--batch", str(B), "--update", "inplace", "--steps", "6",
            "--from-budget"]
    assert script.tier_of(run) == tier
    if tier == "long":
        assert script.main(argv) == 0
        out = json.loads((tmp_path / "out.json").read_text())
        assert out["tier"] == "long" and out["updated_macs"] == T * FWD and out["cut_bits"] == 32 * VALUES
    else:
        with pytest.raises(CaptureRefused, match="the program ended after 2 steps, before step 2 ended"):
            script.main(argv)
    assert "CW_TIER" not in os.environ


@pytest.mark.parametrize("version", ["2.14.0+cpu", "2.15.1", "2.16.0+cu130"])
def test_the_trace_command_refuses_a_torch_other_than_the_trace_jobs_before_reading_anything(tmp_path, monkeypatch,
                                                                                               version):
    """A step's graph moves with torch's version (from 2.14 to 2.15 the counter's attention count moved), so the frozen
    capture traces only under the trace job's torch, whatever its build; the run.json here doesn't exist."""
    monkeypatch.setattr(torch, "__version__", version)
    argv = ["--script", str(PROGRAM), "--run", str(tmp_path / "absent.json"), "--step", "2", "--starts", str(tmp_path)]
    with pytest.raises(SystemExit, match=rf"torch {re.escape(version)} here, but the capture is frozen at 4\.12\.9 under torch "
                                         r"2\.15\.0.*trace under torch==2\.15\.0\+cpu"):
        script.main(argv)
