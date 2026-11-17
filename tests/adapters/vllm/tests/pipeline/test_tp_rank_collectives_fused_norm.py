"""`verity_vllm.pipeline.tp.rank_worker`: the Match's recorder (`TP2CaptureWorkerExtension.tp2_coll_install` and `tp2_coll_close`) on a
rank whose model reduces in the norm after a layer (vLLM's `fused_allreduce_rms_norm`, DeepSeek-V3.2's).  The recorder wraps the
function at vLLM's own module and at every module of the model's classes that imported it, and each call is then one collective
record of kind `fused_allreduce_rms_norm` at the norm's parent module: the partial is the hidden state and the residual stacked
as they were before the call, the output the new residual.  CPU-only: vLLM is `tests.fake_vllm`, the function is a stand-in
with the fused path's contract or the unfused one's, and the rank and its device are those of `tests.pipeline.test_tp_rank_collectives`."""
from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests import fake_vllm
from tests.pipeline import test_tp_rank_collectives as RC
from verity_vllm.engine.collectives import fused_norm as FN
from verity_vllm.pipeline.tp import rank_worker as RW

USER = "fake_dsv32_model"
FAR = f"{FN.FAR_MODULE}:{FN.FAR_NAME}"
HIDDEN, RESIDUAL = [1.0, 2.0, -3.0, 0.5], [0.25, 1.0, 1.0, -0.5]


class _Norm(torch.nn.Module):
    variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = (x.float() + residual.float()).to(torch.bfloat16)
        return r * 2, r


def fused(hidden_states: torch.Tensor, residual: torch.Tensor, norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The FlashInfer path's contract: the new residual is written into `hidden_states`, and the norm is never called."""
    hidden_states.copy_((hidden_states.float() + residual.float()).to(torch.bfloat16))
    return hidden_states * 3, hidden_states


def unfused(hidden_states: torch.Tensor, residual: torch.Tensor, norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return norm(hidden_states, residual)


class DeepseekV32DecoderLayer(torch.nn.Module):
    """A layer of a class whose module imported the function: it reduces in its norm, through its module's name at each call."""
    __module__ = USER

    def __init__(self) -> None:
        super().__init__()
        self.post_attention_layernorm = _Norm()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(sys.modules[USER], FN.FAR_NAME)(x, residual, self.post_attention_layernorm)


class _Model(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([layer])

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[0](x, residual)


def _inputs(scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor(HIDDEN, dtype=torch.bfloat16) * scale, torch.tensor(RESIDUAL, dtype=torch.bfloat16)


@pytest.fixture
def bindings(monkeypatch) -> Callable[[Callable], tuple[types.ModuleType, types.ModuleType]]:
    """``bindings(fn)`` makes ``fn`` vLLM's function at its own module and at the model's module, which imported it -> the two
    modules.  The rank's facts and its device are the recorder tests' stand-ins."""
    fake = fake_vllm.install(monkeypatch, extra=[FAR])
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: RC.FACTS)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    def bind(fn: Callable) -> tuple[types.ModuleType, types.ModuleType]:
        fake.set(FAR, fn)
        user = types.ModuleType(USER)
        setattr(user, FN.FAR_NAME, fn)
        monkeypatch.setitem(sys.modules, USER, user)
        return fake[FN.FAR_MODULE], user

    return bind


def _records(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "coll.rank1.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("fn, path", [(fused, FN.FUSED), (unfused, FN.UNFUSED)], ids=["the fused path", "the unfused path"])
def test_a_reduction_in_the_norm_is_one_record_at_the_norms_parent_with_the_inputs_before_it(bindings, tmp_path, fn, path):
    bindings(fn)
    model = _Model(DeepseekV32DecoderLayer())
    plain = [model(*_inputs()), model(*_inputs(2.0))]
    w = RC._Rank(model)
    assert w.tp2_coll_install(out_dir=str(tmp_path)) == {"rank": 1, "world_size": 2, "n_hooked": 0, "mid_module_sites_ruled_out": 0}
    served = [model(*_inputs()), model(*_inputs(2.0))]
    res = w.tp2_coll_close()
    assert all(torch.equal(a, b) for got, want in zip(served, plain, strict=True) for a, b in zip(got, want, strict=True)), \
        "the run computes what it does unrecorded"
    assert res == {"rank": 1, "n_steps": 2, "n_collectives": 2, "by_kind": {FN.KIND: 2}, "n_stored_steps": 2, "witness": None}
    recs = _records(tmp_path)
    assert [(r["step"], r["k"], r["kind"], r["module"], r["module_class"], r["path"]) for r in recs] == [
        (0, 0, "fused_allreduce_rms_norm", "layers.0", "_Norm", path), (1, 0, "fused_allreduce_rms_norm", "layers.0", "_Norm", path)]
    for step, (rec, scale) in enumerate(zip(recs, (1.0, 2.0), strict=True)):
        x, residual = _inputs(scale)
        part, new_residual = torch.stack([x, residual]), (x.float() + residual.float()).to(torch.bfloat16)
        assert (rec["shape"], rec["dtype"], rec["out_shape"], rec["args"]) == ([2, 4], "bfloat16", [4], [])
        assert (rec["part_sha256"], rec["out_sha256"]) == (RC._sha(part), RC._sha(new_residual))
        kept = torch.load(tmp_path / f"coll.rank1.step{step}.pt")
        assert kept["step"] == step and len(kept["part"]) == len(kept["out"]) == 1
        assert torch.equal(kept["part"][0], part) and torch.equal(kept["out"][0], new_residual)


def test_the_function_is_wrapped_at_vllms_module_and_the_models_and_put_back_at_the_close(bindings, tmp_path):
    home, user = bindings(fused)
    w = RC._Rank(_Model(DeepseekV32DecoderLayer()))
    w.tp2_coll_install(out_dir=str(tmp_path))
    during = (getattr(home, FN.FAR_NAME), getattr(user, FN.FAR_NAME))
    w.tp2_coll_close()
    assert during[0] is not fused and during[1] is not fused
    assert getattr(home, FN.FAR_NAME) is fused and getattr(user, FN.FAR_NAME) is fused


def test_a_model_none_of_whose_classes_imported_the_function_leaves_it_as_it_is(bindings, tmp_path):
    home, user = bindings(fused)
    w = RC._Rank(_Model(torch.nn.Identity()))
    w.tp2_coll_install(out_dir=str(tmp_path))
    during = (getattr(home, FN.FAR_NAME), getattr(user, FN.FAR_NAME))
    w.tp2_coll_close()
    assert during == (fused, fused) and _records(tmp_path) == []


def test_a_call_made_through_vllms_own_module_is_recorded_too(bindings, tmp_path):
    home, _ = bindings(fused)
    layer = DeepseekV32DecoderLayer()
    model = _Model(layer)
    w = RC._Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path))
    model(*_inputs())
    getattr(home, FN.FAR_NAME)(*_inputs(), layer.post_attention_layernorm)
    assert w.tp2_coll_close()["by_kind"] == {FN.KIND: 2}
    assert [(r["step"], r["k"], r["module"]) for r in _records(tmp_path)] == [(0, 0, "layers.0"), (0, 1, "layers.0")]


def test_the_runs_observer_is_off_for_the_recorders_own_copies_and_on_for_the_reduction(bindings, monkeypatch, tmp_path):
    seen: list[bool] = []
    observer: list[Any] = []

    def observed(hidden_states: torch.Tensor, residual: torch.Tensor, norm: Any) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(observer[0].active)
        return fused(hidden_states, residual, norm)

    bindings(observed)
    model = _Model(DeepseekV32DecoderLayer())
    w = RC._Rank(model)
    observer.append(RC._observed(monkeypatch, w, tmp_path))
    w.tp2_coll_install(out_dir=str(tmp_path))
    model(*_inputs())
    w.tp2_coll_close()
    assert seen == [True], "the reduction itself stays observed"
    assert observer[0].switched == [False, True] * 2 and observer[0].active is True, "off for the stacked inputs' copy, then for the record's"
