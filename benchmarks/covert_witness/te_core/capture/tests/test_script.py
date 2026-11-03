"""te_core.capture.script: a program recorded from its start and measured at one step of its history (`Trace.region`),
on recorded/islands/toy/train.py: the update's dependence and cut are the same however the program writes it (in place,
rebound, by an optimizer), a compressed update's cut is its bits, a MeZO update depends on its losses through a Python
float, and a buffer rewritten from θ and this step's tokens is not the run's; a read-back that decides what the graph
can't carry (a Python clip, a branch, a float folded over steps) is listed."""
from __future__ import annotations

import io
import json
import math
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
