"""te_core.capture.script: a program recorded from its start and measured at one step of its history (`Trace.region`),
on recorded/islands/toy/train.py: the update's dependence and cut are the same however the program writes it (in place,
rebound, by an optimizer), a compressed update's cut is its bits, a MeZO update depends on its losses through a Python
float, and a buffer rewritten from θ and this step's tokens is not the run's."""
from __future__ import annotations

import json
from pathlib import Path

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


def test_a_mezo_update_depends_on_its_losses_through_a_float(starts):
    """The update is z · lr · float(L₊ − L₋): every forward GEMM of both perturbed passes is updated, and the cut is the
    one fp32 difference."""
    g = recorded(starts, "--update", "mezo")
    assert fwd_roles(g) == ["updated"] * 4 and g.updated_macs() == 2 * T * FWD
    assert g.cut_bits == 32


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


def test_a_program_that_ends_before_the_step_after_is_refused(starts):
    with pytest.raises(CaptureRefused, match="before step 2 ended"):
        recorded(starts, "--update", "inplace", "--steps", "2")
