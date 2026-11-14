"""`verity_vllm.pipeline.tp.rank_worker`: the short calls a TP driver makes on each rank around a run, and what each leaves
behind (`TP2CaptureWorkerExtension`).  The probe reads the rank's own model: the shapes of its first layer's and its unlayered
parameters, the classes of eight named modules and the logits processor's three settings, and a model it cannot read is
recorded by its error beside the rest.  Closing the capture that the rank's install made waits for the device, ends the
observer and writes the rank's stats and its sampled ids by step.  The loaded-modules record lists what this process imported
from the adapter, the native collector's sources only once its extension is built, and a rank adds its number.  The
value-check install clears the committer, installs it and then the checker over it, and arms the collectives' reference tap.
The profiler start waits for the device and opens the session the census later ends.  CPU-only: the observer, the committer
and the profiler session are stand-ins, and the test sets the rank and `torch.cuda.synchronize`."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.engine import vllm_adapter as va
from verity_vllm.observe import capture, events
from verity_vllm.pipeline import m1_capture
from verity_vllm.pipeline.tp import rank_worker as RW

NATIVE = "verity_vllm.commit.committer.native_collect"
SOURCES = ["native_collect.cpp", "native_gather.cu", "native_rows.cu", "native_plans.h"]
COMMITTER_DIR = "verity_vllm/commit/committer"


class _Rank(RW.TP2CaptureWorkerExtension):
    _tp2_commit: dict[str, Any]
    _tp2_prof: Any


def _rank(monkeypatch, order: list[str] | None = None, rank: int = 1) -> _Rank:
    """Rank ``rank`` of two, whose device wait is written to ``order``."""
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: {"tp": {"rank": rank, "world_size": 2}})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: (order if order is not None else []).append("device waited for"))
    return _Rank()


# -- the probe -----------------------------------------------------------------------------------------------------------------


class _Attention(torch.nn.Module):
    pass


class _SelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = torch.nn.Linear(4, 6, bias=False)
        self.o_proj = torch.nn.Linear(2, 4)
        self.attn = _Attention()


class _Mlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Linear(4, 10, bias=False)
        self.down_proj = torch.nn.Linear(5, 4, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn, self.mlp = _SelfAttention(), _Mlp()
        self.norm = torch.nn.LayerNorm(4)


class _Inner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.norm = torch.nn.LayerNorm(4)


class _LogitsProcessor(torch.nn.Module):
    use_all_gather = False
    org_vocab_size = 8


class _Model(torch.nn.Module):
    def __init__(self, logits_processor: bool = True) -> None:
        super().__init__()
        self.model, self.lm_head = _Inner(), torch.nn.Linear(4, 4, bias=False)
        if logits_processor:
            self.logits_processor = _LogitsProcessor()


def _probe(monkeypatch, get_model: Any) -> tuple[dict[str, Any], list[tuple[Any, Any]]]:
    """(the probe of a rank whose runner hands out its model by ``get_model``, what the execution was read back from)."""
    read: list[tuple[Any, Any]] = []

    def effective_execution_doc(llm: Any, model: Any = None) -> dict[str, Any]:
        read.append((llm.llm_engine.vllm_config, model))
        return {"execution_label": "eager"}

    monkeypatch.setattr(va, "effective_execution_doc", effective_execution_doc)
    w = _rank(monkeypatch)
    w.vllm_config, w.model_runner = SimpleNamespace(name="this rank's config"), SimpleNamespace(get_model=get_model)
    out = w.tp2_probe()
    assert out["tp"] == {"rank": 1, "world_size": 2} and out["effective_execution"] == {"execution_label": "eager"}
    assert [cfg for cfg, _ in read] == [w.vllm_config] and json.loads(json.dumps(out)) == out
    return out, read


def test_a_ranks_probe_reads_its_first_layers_and_its_unlayered_parameters_and_eight_modules_classes(monkeypatch):
    model = _Model()
    out, read = _probe(monkeypatch, lambda: model)
    first = "model.layers.0."
    assert out["param_shapes"] == {
        "model.embed_tokens.weight": [8, 4], first + "self_attn.qkv_proj.weight": [6, 4], first + "self_attn.o_proj.weight": [4, 2],
        first + "self_attn.o_proj.bias": [4], first + "mlp.gate_up_proj.weight": [10, 4], first + "mlp.down_proj.weight": [4, 5],
        first + "norm.weight": [4], first + "norm.bias": [4], "model.norm.weight": [4], "model.norm.bias": [4], "lm_head.weight": [4, 4]}
    assert out["n_params"] == 11 + 7 and list(out["param_shapes"]) == [n for n, _ in model.named_parameters() if not n.startswith("model.layers.1.")]
    assert out["modules"] == {"model.embed_tokens": "Embedding", "lm_head": "Linear", "logits_processor": "_LogitsProcessor",
                              first + "self_attn.qkv_proj": "Linear", first + "self_attn.o_proj": "Linear", first + "mlp.gate_up_proj": "Linear",
                              first + "mlp.down_proj": "Linear", first + "self_attn.attn": "_Attention"}
    assert out["logits_processor"] == {"use_all_gather": False, "org_vocab_size": 8, "vocab_size": None}
    assert "model_error" not in out and read[0][1] is model


def test_a_model_with_no_logits_processor_says_none_of_its_settings(monkeypatch):
    out, _ = _probe(monkeypatch, lambda: _Model(logits_processor=False))
    assert "logits_processor" not in out and "logits_processor" not in out["modules"] and "model_error" not in out
    assert out["n_params"] == 18 and len(out["modules"]) == 7


def test_a_model_the_rank_cannot_read_is_recorded_by_its_error_beside_its_facts_and_its_execution(monkeypatch):
    """A runner that hands out no model, and a model whose parameters cannot be listed: the probe still answers, and the execution
    is read back with the model as far as the rank got it."""
    def no_model() -> Any:
        raise RuntimeError("the runner holds no model yet")

    out, read = _probe(monkeypatch, no_model)
    assert out["model_error"] == "RuntimeError('the runner holds no model yet')" and read[0][1] is None
    assert not {"param_shapes", "n_params", "modules", "logits_processor"} & set(out)
    opaque = object()
    out, read = _probe(monkeypatch, lambda: opaque)
    assert out["model_error"] == "AttributeError(\"'object' object has no attribute 'named_parameters'\")" and read[0][1] is opaque
    assert not {"param_shapes", "n_params", "modules", "logits_processor"} & set(out)


# -- closing the capture ---------------------------------------------------------------------------------------------------------


class _Recorder:
    """A rank's observer as its install and its close read it: its roots, what it sampled by step, the log it writes when it
    ends and the stats its close returns."""

    def __init__(self, order: list[str], stats: dict[str, Any], log_path: Path, log: bytes | None) -> None:
        self.order, self.stats, self.log_path, self.log = order, stats, log_path, log
        self.roots = [SimpleNamespace(name="model.embed_tokens.weight", view=torch.zeros(8, 4))]
        self.sample_ids = {0: [5, 6], 1: [7, 8]}
        self.step_rows = {0: [["r0", 0, 4], ["r1", 4, 9]], 1: [["r0", 0, 1], ["r1", 1, 2]]}

    def install(self) -> None:
        self.order.append("recorder installed")

    def close(self) -> dict[str, Any]:
        self.order.append("recorder closed")
        if self.log is not None:
            self.log_path.write_bytes(self.log)
        return self.stats


def _stats(out: Path, errors: int) -> dict[str, Any]:
    return {"events_written": 41, "steps": 2, "recorder": "native", "snapshot_bytes": 96, "snapshot_files": 3, "roots": 12,
            "errors": [f"error {i}" for i in range(errors)], "timing": {"forward_s": 0.5}, "storage": {"dir": out / "rank1", "bytes": 96},
            "kept only in the stats file": [1, 2]}


def _close(monkeypatch, out: Path, *, log: bytes | None = b"eleven byte", errors: int = 7) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """(what rank 1's `tp2_close` returns after its `tp2_install` put an observer on it whose log ends as ``log``, the order of
    what the two did, the stats the observer closed with)."""
    order: list[str] = []
    stats = _stats(out, errors)
    header = events.RunHeader("log/v1", "2026-01-01T00:00:00Z", "profile-of-this-rank", {}, {}, {})
    monkeypatch.setattr(capture, "make_header", lambda engine, workload, **kw: header)
    monkeypatch.setattr(capture, "Capture", lambda engine, *, log_path, **kw: _Recorder(order, stats, log_path, log))
    monkeypatch.setattr(m1_capture, "dump_cos_sin", lambda engine, rank_dir: None)
    monkeypatch.setattr(va, "versions_doc", lambda engine: {})
    w = _rank(monkeypatch, order)
    w.vllm_config, w.model_runner = SimpleNamespace(), SimpleNamespace(get_model=_Model)
    made = w.tp2_install(out_dir=str(out), workload={}, case="C", cp={}, engine_kwargs={}, env_flags={}, snapshot_steps=[],
                         snapshot_max_bytes=0, adapter_revision="r", wheel_sha256=None, vllm_commit=None)
    assert (made["rank"], made["log"], made["roots"]) == (1, str(out / "log.rank1.jsonl.gz"), 1) and not (out / "log.rank1.jsonl.gz").exists()
    return w.tp2_close(), order, stats


def test_closing_a_ranks_capture_waits_for_the_device_and_writes_its_stats_and_its_sampled_ids_by_step(monkeypatch, tmp_path, capsys):
    res, order, stats = _close(monkeypatch, tmp_path)
    assert order == ["recorder installed", "device waited for", "device waited for", "recorder closed"]
    said = json.loads(json.dumps(stats, default=str))
    assert said["storage"]["dir"] == str(tmp_path / "rank1") and json.loads((tmp_path / "stats.rank1.json").read_text()) == said
    sampled = {"0": [5, 6], "1": [7, 8]}
    assert json.loads((tmp_path / "tokens.rank1.json").read_text()) == {
        "rank": 1, "per_step_sampled": sampled, "profile_id": "profile-of-this-rank",
        "step_rows": {"0": [["r0", 0, 4], ["r1", 4, 9]], "1": [["r0", 0, 1], ["r1", 1, 2]]}}
    assert res == {"rank": 1, "events_written": 41, "steps": 2, "recorder": "native", "snapshot_bytes": 96, "snapshot_files": 3,
                   "errors": [f"error {i}" for i in range(5)], "n_errors": 7, "roots": 12, "log_bytes": 11, "per_step_sampled": sampled,
                   "timing": {"forward_s": 0.5}, "storage": said["storage"]}
    assert sorted(f.name for f in tmp_path.iterdir()) == ["log.rank1.jsonl.gz", "rank1", "stats.rank1.json", "tokens.rank1.json"]
    line = capsys.readouterr().out.splitlines()[-1]
    assert line.startswith("[tp2 rank 1] closed: 41 events, 2 steps, recorder=native, snapshot_bytes=96, errors=7, log 11 B, ")


def test_a_capture_whose_log_is_not_there_closes_with_no_size_and_every_error_it_has(monkeypatch, tmp_path, capsys):
    res, _, _ = _close(monkeypatch, tmp_path, log=None, errors=2)
    assert (res["log_bytes"], res["errors"], res["n_errors"]) == (None, ["error 0", "error 1"], 2)
    assert json.loads((tmp_path / "stats.rank1.json").read_text())["errors"] == ["error 0", "error 1"]
    assert ", errors=2, log None B, " in capsys.readouterr().out.splitlines()[-1]


# -- the loaded modules ------------------------------------------------------------------------------------------------------------


class _NoPath:
    """A module file whose path the file system does not give."""

    def __fspath__(self) -> str:
        raise OSError("stale file handle")


def _native(monkeypatch, **attrs: Any) -> None:
    """The native collector as loaded in this process, with ``attrs`` (`_EXT`: its built extension)."""
    mod = types.ModuleType(NATIVE)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, NATIVE, mod)


def test_the_native_collectors_sources_are_listed_once_its_extension_is_built(monkeypatch):
    _native(monkeypatch, _EXT=object())
    r = RW.repo_modules_loaded()
    jit = [f"{COMMITTER_DIR}/{n}" for n in SOURCES]
    assert r["native_collect_jit_sources"] == jit and set(jit) <= set(r["files"])
    assert (r["pid"], r["n"]) == (os.getpid(), len(r["files"])) and r["files"] == sorted(set(r["files"]))
    assert "verity_vllm/pipeline/tp/rank_worker.py" in r["files"]


@pytest.mark.parametrize("loaded", ["not loaded", "loaded and not built"])
def test_a_native_collector_that_built_no_extension_lists_none_of_its_sources(monkeypatch, loaded):
    if loaded == "not loaded":
        monkeypatch.delitem(sys.modules, NATIVE, raising=False)
    else:
        _native(monkeypatch, _EXT=None)
    r = RW.repo_modules_loaded()
    assert r["native_collect_jit_sources"] == [] and not [f for f in r["files"] if f.endswith((".cpp", ".cu", ".h"))]
    assert "verity_vllm/pipeline/tp/rank_worker.py" in r["files"]


def test_the_sources_are_read_where_the_native_collector_says_it_is_and_only_the_ones_there(monkeypatch):
    """The collector names its own directory (`HERE`); a directory of the adapter that holds none of the four lists none."""
    from verity_vllm.config import ROOT

    _native(monkeypatch, _EXT=object(), HERE=str(ROOT / COMMITTER_DIR))
    assert RW.repo_modules_loaded()["native_collect_jit_sources"] == [f"{COMMITTER_DIR}/{n}" for n in SOURCES]
    _native(monkeypatch, _EXT=object(), HERE=str(ROOT / "verity_vllm" / "pipeline" / "tp"))
    assert RW.repo_modules_loaded()["native_collect_jit_sources"] == []


def test_a_module_whose_file_cannot_be_resolved_is_left_out_and_the_rest_is_still_listed(monkeypatch):
    lost = types.ModuleType("a_module_on_a_lost_mount")
    lost.__dict__["__file__"] = _NoPath()
    monkeypatch.setitem(sys.modules, lost.__name__, lost)
    r = RW.repo_modules_loaded()
    assert "verity_vllm/pipeline/tp/rank_worker.py" in r["files"] and r["n"] == len(r["files"]) > 1


@pytest.mark.parametrize(("state", "rank"), [({"rank": "1"}, 1), ({"rank": 0}, 0), ({}, None), (None, None), ("unset", None)])
def test_a_rank_adds_its_number_to_its_loaded_modules_once_its_commit_state_is_made(monkeypatch, state, rank):
    monkeypatch.setattr(RW, "repo_modules_loaded", lambda: {"pid": 7, "n": 1, "files": ["verity_vllm/a.py"]})
    w = _rank(monkeypatch)
    if state != "unset":
        w._tp2_commit = state
    assert w.tp2_repo_modules() == {"pid": 7, "n": 1, "files": ["verity_vllm/a.py"], "rank": rank}


# -- the value-check install -------------------------------------------------------------------------------------------------------


class _Sampled(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _Committer:
    """A rank's committer as the value checker reads it: its modules, its runner's sampler and its two step boundaries."""

    def __init__(self, order: list[str], sampler: Any) -> None:
        self.order = order
        self.state: dict[str, Any] = {}
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Identity())
        self._modules = [("0", self.model[0]), ("1", self.model[1])]
        self.runner = SimpleNamespace(sampler=sampler)

    def reset(self) -> None:
        self.order.append("committer cleared")

    def install(self) -> None:
        self.order.append("committer installed")
        self.model[0].register_forward_hook(self._acquire)

    def _acquire(self, module: Any, args: Any, output: Any) -> None:
        tapped = self.state["vc"].cur["0"]["calls"]
        self.order.append("committer acquires " + ("after" if tapped else "before") + " the checker's tap")

    def _begin_step(self) -> None:
        self.order.append("committer begins a step")

    def _end_step(self) -> None:
        self.order.append("committer ends a step")


def _install(monkeypatch, *, sampler: Any = None, tap: Any = None) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """(what rank 1's `tp2_commit_value_check_install` returns, its commit state after, the order of what happened)."""
    order: list[str] = []
    w = _rank(monkeypatch)
    com = _Committer(order, sampler)
    w._tp2_commit = com.state = {"com": com, "rank": 1, "rc": "the run commitment of the pair before", "tps": tap}
    return w.tp2_commit_value_check_install(), w._tp2_commit, order


def test_the_install_clears_the_committer_and_installs_it_before_the_checker_that_reads_after_it(monkeypatch):
    """The committer's hooks are registered first, so at each module it acquires before the checker's tap clones, and the
    checker's step begins and ends inside the committer's."""
    res, st, order = _install(monkeypatch)
    assert res == {"rank": 1, "installed": True, "value_checker": True, "tp_reference_tap": False} and st["rc"] is None
    assert order == ["committer cleared", "committer installed"]
    vc, com = st["vc"], st["com"]
    assert type(vc).__name__ == "ValueChecker" and vc.c is com and len(vc.handles) == 4 and vc.steps == []
    seen: list[int] = []
    com.model[0].register_forward_hook(lambda module, args, output: seen.append(vc.cur["0"]["calls"]))
    com._begin_step()
    x = torch.ones(1, 2)
    y = com.model(x)
    com._end_step()
    assert order[2:] == ["committer begins a step", "committer acquires before the checker's tap", "committer ends a step"]
    assert seen == [1] and vc.cur is None
    (step,) = vc.steps
    assert sorted(step) == ["0", "1"] and step["0"]["calls"] == step["1"]["calls"] == 1
    assert torch.equal(step["0"]["in"][0][0], x) and torch.equal(step["0"]["out"][0][0], y) and torch.equal(step["1"]["out"][0][0], y)


def test_the_checker_taps_the_sampler_too_when_it_is_a_module(monkeypatch):
    sampler = _Sampled()
    _, st, _ = _install(monkeypatch, sampler=sampler)
    assert len(st["vc"].handles) == 5
    st["com"]._begin_step()
    ids = torch.tensor([[3], [4]])
    sampler(ids)
    assert torch.equal(st["vc"].steps[0]["runner.sampler"]["out"][0][0], ids)


def test_the_install_arms_the_collectives_reference_tap_and_drops_what_it_held(monkeypatch):
    tap = SimpleNamespace(reference=False, ref={"a part of the pair before": 1}, ref_in={"its input": 2})
    res, st, _ = _install(monkeypatch, tap=tap)
    assert res == {"rank": 1, "installed": True, "value_checker": True, "tp_reference_tap": True}
    assert (tap.reference, tap.ref, tap.ref_in) == (True, {}, {}) and tap.ref is not tap.ref_in and st["tps"] is tap


# -- the profiler start ------------------------------------------------------------------------------------------------------------


class _Session:
    """A `torch.profiler` session: what it was made with, that it was entered, and the trace it exports once ended."""

    def __init__(self, order: list[str], **made_with: Any) -> None:
        self.order, self.made_with = order, made_with
        order.append("session made")

    def __enter__(self) -> Self:
        self.order.append("session entered")
        return self

    def __exit__(self, *exc: object) -> None:
        self.order.append("session ended")

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text(json.dumps({"traceEvents": [{"ph": "X", "cat": "kernel", "name": "gemm", "ts": 1, "dur": 2.0}]}))


def test_the_profiler_start_waits_for_the_device_and_opens_the_session_the_census_ends(monkeypatch, tmp_path):
    from torch import profiler

    order: list[str] = []
    monkeypatch.setattr(profiler, "profile", lambda **made_with: _Session(order, **made_with))
    w = _rank(monkeypatch, order)
    assert w.tp2_profiler_start() == {"rank": 1, "started": True, "pid": os.getpid()}
    assert order == ["device waited for", "session made", "session entered"]
    assert w._tp2_prof.made_with == {"activities": [profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA], "record_shapes": False}
    res = w.tp2_profiler_stop(out_dir=str(tmp_path))
    assert order[3:] == ["device waited for", "session ended"] and (res["rank"], res["n_kernels"], res["device_us"]) == (1, 1, 2.0)
    assert json.loads((tmp_path / "census.rank1.json").read_text())["kernels"]["gemm"]["count"] == 1
