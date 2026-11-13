"""`verity_vllm.pipeline.tp.rank_worker`: the Match's recorder of a rank's collectives
(`TP2CaptureWorkerExtension.tp2_coll_install` and `tp2_coll_close`).  Between the two, every collective the rank issues is
one record in call order: the model forward it belongs to, its place in that forward, its kind, the module it is issued
from, and the shape, dtype and SHA-256 of its partial and of what the rank received.  The values of the first steps are
kept for the cross-rank check, and the run computes what it would without the recorder.  On a rank whose `tp2_install`
put an observer on the same run, the observer is off for the recorder's own copies.  CPU-only: vLLM is `tests.fake_vllm`,
whose three layer modules and MoE runner module hold the collectives the test gives them, the model's modules carry the
class names the recorder hooks by, and the test sets the rank and `torch.cuda.synchronize`."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from tests import fake_vllm
from verity_vllm.acquire.sources import moe_align_source
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.observe import capture
from verity_vllm.observe.events import RunHeader
from verity_vllm.pipeline import m1_capture
from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension

LINEAR = "vllm.model_executor.layers.linear"
EMBEDDING = "vllm.model_executor.layers.vocab_parallel_embedding"
LOGITS = "vllm.model_executor.layers.logits_processor"
MOE_RUNNER = "vllm.model_executor.layers.fused_moe.runner.moe_runner"
ALL_REDUCE, ALL_GATHER, GATHER = "tensor_model_parallel_all_reduce", "tensor_model_parallel_all_gather", "tensor_model_parallel_gather"
FACTS = {"tp": {"rank": 1, "world_size": 2}, "pid": 7}
#: (kind, module, its class) of each collective of one forward and the logits after it, in call order
ONE_STEP = [("all_reduce", "embed", "VocabParallelEmbedding"), ("all_reduce", "layer.attn.o_proj", "RowParallelLinear"),
            ("all_reduce", "layer.mlp", "FusedMoE"), ("all_reduce", "layer.block.inner", "RowParallelLinear"),
            ("all_reduce", "layer.block", "_Reducing"), ("all_reduce", None, None),
            ("all_gather", "logits_processor", "LogitsProcessor"), ("gather", "logits_processor", "LogitsProcessor")]
HOOKED = ["embed", "layer.attn.o_proj", "layer.mlp", "layer.block", "layer.block.inner", "lm_head", "logits_processor"]


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """What a rank receives when the other rank's partial equals its own."""
    return x * 2


def all_gather(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.cat([x, x], dim)


def gather(x: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor:
    return torch.cat([x, x], dim)


class _Issues(torch.nn.Module):
    """A module that issues one collective through the name ``name`` of vLLM's module ``module``, read at each call."""

    def __init__(self, module: Any, name: str = ALL_REDUCE, *args: Any) -> None:
        super().__init__()
        self.site, self.name, self.args = [module], name, args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self.site[0], self.name)(x + 1, *self.args)


class VocabParallelEmbedding(_Issues):
    pass


class RowParallelLinear(_Issues):
    pass


class MoERunner(_Issues):
    pass


class FusedMoE(torch.nn.Module):
    """The MoE layer: the all-reduce is its runner's, which the recorder does not hook under it."""

    def __init__(self, module: Any) -> None:
        super().__init__()
        self.runner = MoERunner(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.runner(x)


class _Reducing(torch.nn.Module):
    """A module of no hooked class that reduces its own result, after a hooked module inside it has returned."""
    reduce_results = True

    def __init__(self, module: Any) -> None:
        super().__init__()
        self.inner, self.site = RowParallelLinear(module), [module]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self.site[0], ALL_REDUCE)(self.inner(x) - 3)


class ParallelLMHead(torch.nn.Module):
    """Hooked for its name; it issues nothing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LogitsProcessor(torch.nn.Module):
    def __init__(self, module: Any) -> None:
        super().__init__()
        self.site = [module]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = getattr(self.site[0], ALL_GATHER)(x, -1)
        return getattr(self.site[0], GATHER)(y, 0, 0) if hasattr(self.site[0], GATHER) else y


class _Attention(torch.nn.Module):
    def __init__(self, module: Any) -> None:
        super().__init__()
        self.o_proj = RowParallelLinear(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)


class _Layer(torch.nn.Module):
    def __init__(self, linear: Any, runner: Any) -> None:
        super().__init__()
        self.attn, self.mlp, self.block = _Attention(linear), FusedMoE(runner), _Reducing(linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.mlp(self.attn(x)))


class _Model(torch.nn.Module):
    """One decoder layer between an embedding and an LM head.  Its forward ends with a collective issued by no hooked
    module, and the logits' collectives run after the forward, as vLLM's do."""

    def __init__(self, fake: fake_vllm.FakeVllm) -> None:
        super().__init__()
        self.site = [fake[LINEAR]]
        self.embed, self.layer = VocabParallelEmbedding(fake[EMBEDDING]), _Layer(fake[LINEAR], fake[MOE_RUNNER])
        self.lm_head, self.logits_processor = ParallelLMHead(), LogitsProcessor(fake[LOGITS])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(self.site[0], ALL_REDUCE)(self.layer(self.embed(x)))

    def step(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head(self(x)))


class _Rank(TP2CaptureWorkerExtension):
    """A rank's worker as far as the recorder reads it: the model its runner holds."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model_runner, self.vllm_config = SimpleNamespace(get_model=lambda: model), SimpleNamespace()


@pytest.fixture
def vllm(monkeypatch) -> fake_vllm.FakeVllm:
    """The stand-in with a collective at every name the recorder wraps, the rank's facts, and a device that never waits."""
    fake = fake_vllm.install(monkeypatch)
    for module in (LINEAR, EMBEDDING, MOE_RUNNER):
        setattr(fake[module], ALL_REDUCE, all_reduce)
    setattr(fake[LOGITS], ALL_GATHER, all_gather)
    setattr(fake[LOGITS], GATHER, gather)
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: FACTS)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return fake


def _x(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(6, dtype=dtype).reshape(2, 3)


def _records(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / "coll.rank1.jsonl").read_text().splitlines()]


def _sha(t: torch.Tensor) -> str:
    raw = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    return hashlib.sha256(raw.contiguous().numpy().tobytes()).hexdigest()


def test_install_hooks_the_collective_modules_and_writes_what_it_hooked(vllm, tmp_path):
    out = tmp_path / "not" / "there" / "yet"
    w = _Rank(_Model(vllm))
    assert w.tp2_coll_install(out_dir=str(out), store_steps=4) == {"rank": 1, "world_size": 2, "n_hooked": 7, "mid_module_sites_ruled_out": 0}
    meta = json.loads((out / "coll_meta.rank1.json").read_text())
    assert (meta["rank"], meta["world_size"], meta["facts"], meta["store_steps"]) == (1, 2, FACTS, 4)
    assert meta["hooked"] == HOOKED and meta["nested_moe_sites_dropped"] == ["layer.mlp.runner"]
    assert meta["mid_module_sites"] == {} and meta["mid_module_sites_ruled_out"] == []
    assert meta["attribution_rule"].startswith("outermost hooked module") and len(meta) == 9
    w.tp2_coll_close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_each_collective_is_one_record_in_call_order_with_its_module_and_digests(vllm, tmp_path, dtype):
    model = _Model(vllm)
    plain = [model.step(_x(dtype)), model.step(_x(dtype) + 1)]
    w = _Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path))
    served = [model.step(_x(dtype)), model.step(_x(dtype) + 1)]
    res = w.tp2_coll_close()
    assert all(torch.equal(a, b) for a, b in zip(plain, served, strict=True))       # the run computes what it does unrecorded
    assert res == {"rank": 1, "n_steps": 2, "n_collectives": 16, "by_kind": {"all_reduce": 12, "all_gather": 2, "gather": 2},
                   "n_stored_steps": 2, "witness": None}
    recs = _records(tmp_path)
    assert [(r["step"], r["k"]) for r in recs] == [(s, k) for s in range(2) for k in range(8)]
    assert [(r["kind"], r["module"], r["module_class"]) for r in recs] == ONE_STEP * 2
    assert {r["rank"] for r in recs} == {1} and {r["dtype"] for r in recs} == {str(dtype).replace("torch.", "")}
    assert [r["args"] for r in recs[:8]] == [[]] * 6 + [["-1"], ["0", "0"]]
    assert [(r["shape"], r["out_shape"]) for r in recs[:8]] == [([2, 3], [2, 3])] * 6 + [([2, 3], [2, 6]), ([2, 6], [4, 6])]
    for step in range(2):
        kept = torch.load(tmp_path / f"coll.rank1.step{step}.pt", weights_only=True)
        assert kept["step"] == step and len(kept["part"]) == len(kept["out"]) == 8
        assert torch.equal(kept["part"][0], _x(dtype) + step + 1) and torch.equal(kept["out"][7], served[step])
        for r, part, got in zip(recs[8 * step:8 * step + 8], kept["part"], kept["out"], strict=True):
            assert (r["part_sha256"], r["out_sha256"]) == (_sha(part), _sha(got))
            assert torch.equal(got, {"all_reduce": all_reduce, "all_gather": all_gather}.get(r["kind"], lambda p: gather(p, 0, 0))(part))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["coll.rank1.jsonl", "coll.rank1.step0.pt", "coll.rank1.step1.pt", "coll_meta.rank1.json"]


@pytest.mark.parametrize("store_steps, kept", [(0, []), (1, [0]), (2, [0, 1]), (3, [0, 1, 2])])
def test_values_are_kept_for_the_steps_under_store_steps_and_every_step_is_recorded(vllm, tmp_path, store_steps, kept):
    model = _Model(vllm)
    w = _Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path), store_steps=store_steps)
    for i in range(3):
        model.step(_x() + i)
    assert len(w._tp2_coll["values"]["part"]) == (8 if store_steps > 2 else 0)      # a step it will not store costs the rank no memory
    res = w.tp2_coll_close()
    assert (res["n_steps"], res["n_collectives"], res["n_stored_steps"]) == (3, 24, len(kept))
    assert sorted(p.name for p in tmp_path.glob("coll.rank1.step*.pt")) == [f"coll.rank1.step{s}.pt" for s in kept]
    assert json.loads((tmp_path / "coll_meta.rank1.json").read_text())["store_steps"] == store_steps


def test_a_run_of_no_forward_records_nothing(vllm, tmp_path):
    w = _Rank(_Model(vllm))
    w.tp2_coll_install(out_dir=str(tmp_path))
    assert w.tp2_coll_close() == {"rank": 1, "n_steps": 0, "n_collectives": 0, "by_kind": {}, "n_stored_steps": 0, "witness": None}
    assert (tmp_path / "coll.rank1.jsonl").read_text() == ""


def test_close_puts_vllms_names_back_and_takes_the_hooks_off_the_model(vllm, tmp_path):
    model = _Model(vllm)
    w = _Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path))
    wrapped = [getattr(vllm[m], n) for m, n in ((LINEAR, ALL_REDUCE), (EMBEDDING, ALL_REDUCE), (MOE_RUNNER, ALL_REDUCE), (LOGITS, ALL_GATHER),
                                                (LOGITS, GATHER))]
    assert not any(f in (all_reduce, all_gather, gather) for f in wrapped)
    assert sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()) == 2 * len(HOOKED) + 1
    w.tp2_coll_close()
    assert [getattr(vllm[m], ALL_REDUCE) for m in (LINEAR, EMBEDDING, MOE_RUNNER)] == [all_reduce] * 3
    assert (getattr(vllm[LOGITS], ALL_GATHER), getattr(vllm[LOGITS], GATHER)) == (all_gather, gather)
    assert sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in model.modules()) == 0
    model.step(_x())
    assert _records(tmp_path) == []                                                 # a forward after the close is not recorded


def test_a_collective_name_vllm_lacks_is_not_wrapped(vllm, tmp_path):
    delattr(vllm[LOGITS], GATHER)                                                   # a vLLM whose logits processor has no gather
    model = _Model(vllm)
    w = _Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path))
    model.step(_x())
    assert w.tp2_coll_close()["by_kind"] == {"all_reduce": 6, "all_gather": 1} and not hasattr(vllm[LOGITS], GATHER)


def test_an_moe_module_is_one_that_imports_and_holds_the_all_reduce(vllm):
    assert RW._moe_collective_modules() == [vllm[MOE_RUNNER]]                       # the other two are no modules of this vLLM
    delattr(vllm[MOE_RUNNER], ALL_REDUCE)
    assert RW._moe_collective_modules() == []


class _Recorder:
    """The recorder of the run's aten observer: it says each time it is switched."""

    def __init__(self) -> None:
        self.switched: list[bool] = []
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self.switched.append(value)
        self._active = value


def _observed(monkeypatch, w: _Rank, out: Path) -> _Recorder:
    """The recorder of the observer that `tp2_install` puts on rank ``w`` before the collectives' recorder, as `tp-capture` does."""
    rec = _Recorder()
    header = RunHeader("log/v1", "2026-01-01T00:00:00Z", "id", {}, {}, {})
    monkeypatch.setattr(capture, "make_header", lambda engine, workload, **kw: header)
    monkeypatch.setattr(capture, "Capture", lambda engine, **kw: SimpleNamespace(rec=rec, roots=[], install=lambda: None))
    monkeypatch.setattr(m1_capture, "dump_cos_sin", lambda engine, rank_dir: None)
    monkeypatch.setattr(va, "versions_doc", lambda engine: {})
    w.tp2_install(out_dir=str(out), workload={}, case="C", cp={}, engine_kwargs={}, env_flags={}, snapshot_steps=[], snapshot_max_bytes=0,
                  adapter_revision="r", wheel_sha256=None, vllm_commit=None)
    return rec


def test_the_runs_observer_is_off_for_the_recorders_own_copies_and_on_for_the_collective(vllm, monkeypatch, tmp_path):
    model = _Model(vllm)
    w = _Rank(model)
    rec, seen = _observed(monkeypatch, w, tmp_path), []

    def observed(x: torch.Tensor) -> torch.Tensor:
        seen.append(rec.active)
        return all_reduce(x)

    setattr(vllm[EMBEDDING], ALL_REDUCE, observed)
    w.tp2_coll_install(out_dir=str(tmp_path))
    model.step(_x())
    w.tp2_coll_close()
    assert seen == [True] and rec.switched == [False, True] * 8 and rec.active is True


class _Witness:
    """The MoE align witness of a model with an MoE layer: told each forward, closed once into the recorder's directory."""

    def __init__(self, position: Any) -> None:
        self.position = position
        self.steps: list[tuple[int, int]] = []
        self.closed: list[Path] = []

    def step(self, step: int) -> None:
        self.steps.append((step, self.position()))

    def close(self, out: Path) -> dict[str, Any]:
        self.closed.append(out)
        return {"calls": len(self.steps)}


def test_the_align_witness_is_told_each_forward_and_its_close_is_in_the_result(vllm, monkeypatch, tmp_path):
    made: list[Any] = []

    def capture_witness(model: Any, rank: int, position: Any, quiet: Any) -> _Witness:
        made.append((model, rank))
        with quiet():                                                               # the recorder's way to hide the witness's copies
            return _Witness(position)

    monkeypatch.setattr(moe_align_source, "capture_witness", capture_witness)
    model = _Model(vllm)
    w = _Rank(model)
    w.tp2_coll_install(out_dir=str(tmp_path))
    witness = w._tp2_coll["witness"]
    assert made == [(model, 1)] and witness.position() == 0
    for i in range(2):
        model.step(_x() + i)
    assert witness.position() == 8                                                  # its place among the step's collectives
    assert w.tp2_coll_close()["witness"] == {"calls": 2}
    assert witness.steps == [(0, 0), (1, 0)] and witness.closed == [tmp_path]
