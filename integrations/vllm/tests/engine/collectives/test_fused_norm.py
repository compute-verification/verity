"""`collectives.fused_norm`: the Match recorder's view of `fused_allreduce_rms_norm`.

* `targets` finds the function's own module and every module of the model's classes that imported it, and nothing when no
  class does;
* a recorded call is one record at the norm's parent module, whose part is (partial, residual) as they were before the call
  (the fused kernel overwrites the partial), whose out is the new residual, and whose path says whether the norm ran;
* `expected_residual` is the MNNVL rule: f32 sum from +0 over ranks ascending, -0 payloads as +0, FTZ, bf16, then + residual;
* the Match checks a fused-path call against it and leaves an unfused one to the Commit.
"""
from __future__ import annotations

import sys
import types
from contextlib import nullcontext

import pytest
import torch

from verity_vllm.engine.collectives import fused_norm as FN
from verity_vllm.pipeline.tp import match as M


class _Norm(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=torch.bfloat16))
        self.variance_epsilon = 1e-6

    def forward(self, x, residual):
        r = (x.float() + residual.float()).to(torch.bfloat16)
        return r * 2, r


def _fused(hidden_states, residual, norm):
    """The FlashInfer path's contract: the new residual into hidden_states, norm never called."""
    hidden_states.copy_((hidden_states.float() + residual.float()).to(torch.bfloat16))
    return hidden_states * 3, hidden_states


def _unfused(hidden_states, residual, norm):
    return norm(hidden_states, residual)


@pytest.fixture
def mods(monkeypatch):
    home, user = types.ModuleType(FN.FAR_MODULE), types.ModuleType("fake_dsv32_model")
    home.fused_allreduce_rms_norm = user.fused_allreduce_rms_norm = _fused
    monkeypatch.setitem(sys.modules, FN.FAR_MODULE, home)
    monkeypatch.setitem(sys.modules, "fake_dsv32_model", user)
    return home, user


def _model(module_name):
    layer = type("DeepseekV32DecoderLayer", (torch.nn.Module,), {"__module__": module_name})()
    layer.post_attention_layernorm = _Norm(4)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer])
    return model


def test_targets_are_the_home_module_and_the_models_importers(mods):
    home, user = mods
    assert FN.targets(_model("fake_dsv32_model")) == [(home, FN.FAR_NAME), (user, FN.FAR_NAME)]
    assert FN.targets(_model("some_other_module")) == []


def _recorder():
    st = {"cur": None, "records": [], "handles": []}

    def record(orig, kind):
        def w(x):
            y = orig(x)
            st["records"].append({"kind": kind, "module": (st["cur"] or {}).get("name"), "part": x.clone(), "out": y.clone()})
            return y
        return w
    return st, record


@pytest.mark.parametrize("orig, path", [(_fused, FN.FUSED), (_unfused, FN.UNFUSED)])
def test_a_call_is_one_record_at_the_norms_parent_with_the_inputs_before_it(orig, path):
    model = _model("fake_dsv32_model")
    norm = model.layers[0].post_attention_layernorm
    st, record = _recorder()
    far = FN.recording(orig, model, st, record, nullcontext)
    x, res = torch.tensor([1.0, 2.0, -3.0, 0.5], dtype=torch.bfloat16), torch.tensor([0.25, 1.0, 1.0, -0.5], dtype=torch.bfloat16)
    x0 = x.clone()
    normed, new_res = far(x, res, norm)
    (rec,) = st["records"]
    assert rec["kind"] == FN.KIND and rec["module"] == "layers.0" and rec["path"] == path
    assert torch.equal(rec["part"], torch.stack([x0, res])) and torch.equal(rec["out"], new_res)
    assert torch.equal(new_res, (x0.float() + res.float()).to(torch.bfloat16)) and len(st["handles"]) == 1 and st["cur"] is None
    far(x0.clone(), res, norm)
    assert len(st["handles"]) == 1


def test_expected_residual_is_the_mnnvl_rule():
    bf = torch.bfloat16
    tiny = torch.tensor([2.0 ** -127], dtype=bf)                      # a bf16 subnormal: FADD.FTZ reads it as 0
    parts = [torch.stack([torch.tensor([1.0, -0.0, 3.0, 0.0], dtype=bf), torch.tensor([0.5, -0.0, 1.0, 1.0], dtype=bf)]),
             torch.stack([torch.tensor([1.0, -0.0, 2.0 ** -8, 0.0], dtype=bf), torch.tensor([0.5, -0.0, 1.0, 1.0], dtype=bf)])]
    got = FN.expected_residual(parts)
    assert got.tolist() == [2.5, 0.0, 4.0, 1.0]
    assert torch.signbit(got[1]).item() is False                      # -0 payloads summed from +0, plus a -0 residual
    sub = FN.expected_residual([torch.stack([tiny, torch.zeros(1, dtype=bf)])] * 2)
    assert sub.tolist() == [0.0]


def test_the_match_checks_a_fused_call_and_leaves_an_unfused_one_to_the_commit():
    bf = torch.bfloat16
    parts = [torch.stack([torch.tensor([1.0, 2.0], dtype=bf), torch.tensor([0.5, 0.5], dtype=bf)])] * 2
    prov = {"all_reduce_checked": 0, "all_gather_checked": 0, "alt_f32_once_equal": 0, "alt_ascending_equal": 0}
    exp = M._expected(FN.KIND, FN.FUSED, parts, [], (0, 1), prov)
    assert exp.tolist() == [2.5, 4.5] and prov["fused_norm_checked"] == 1
    assert M._expected(FN.KIND, FN.UNFUSED, parts, [], (0, 1), prov) is None and prov["fused_norm_unfused_left_to_commit"] == 1
    assert M._expected("reduce_scatter", None, parts, [], (0, 1), prov) is None
