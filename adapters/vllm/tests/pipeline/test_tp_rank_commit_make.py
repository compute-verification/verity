"""`verity_vllm.pipeline.tp.rank_worker`: a rank makes its committer (`TP2CaptureWorkerExtension.tp2_commit_make`, what
`tp-commit` asks of every rank before the first run).  The rank makes the committer the driver names over its own model,
attaches the sources its model and the driver's flags call for, sizes the arena, registers its weights and keeps the Commit's
state for the calls that follow; it answers what it made.  A negative the driver names for another rank changes nothing here.
CPU-only: the committer is a stand-in that keeps what the rank did with it, made through the Commit's door for an outside
committer (`plugin:<module>:<factory>`); the partial and MoE sources are the library's, on a model whose modules carry vLLM's
class names; the hidden source, which loads a tap build, is a stand-in; and the test sets the rank's facts."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch

from verity_vllm.acquire import fa2_plane_classes as FP
from verity_vllm.acquire.sources import hidden_source as HS
from verity_vllm.acquire.sources import moe_source as MS
from verity_vllm.acquire.sources.partial_source import TPPartialSource
from verity_vllm.pipeline import commit as PC
from verity_vllm.pipeline.tp import rank_worker as RW
from verity_vllm.pipeline.tp.rank_worker import TP2CaptureWorkerExtension

FACTS = {"tp": {"rank": 1, "world_size": 2}}
#: a make's options that are tables by request (`{"<request>": number}`: each request's arrival step and its token limit)
TABLES = [name for name, option in inspect.signature(TP2CaptureWorkerExtension.tp2_commit_make).parameters.items()
          if option.annotation == "dict[str, int] | None"]
ENVELOPE = {"max_num_seqs": 2, "max_tokens": 4}
ROOT = "ab" * 64
UNPINNED = {"rank": 1, "result": None, "why": "weights unpinned (MAN-09): no checkpoint of record given to this rank"}
UNIFORM = {"rank": 1, "declarations": 0, "source": None, "rule": "uniformity (no rank Build record with attention_impls_observed)"}
PARTIALS = {"sites": [["embed", "VocabParallelEmbedding"], ["proj", "RowParallelLinear"]],
            "extra_step_bytes": {"per_token_bytes": 8, "per_seq_bytes": 0, "tensors_per_step": 2},
            "relabel_rank": None, "omit_site": None, "flip_site": None}
MLA = {"family": None, "why": "no attention impl carries vllm_flash_attn_version (MLA)"}
IMPLS = {"model.layers.0.self_attn.attn": {"num_heads": 4, "num_kv_heads": 2, "head_size": 8, "sliding_window": None},
         "model.layers.1.self_attn.attn": {"num_heads": 2, "num_kv_heads": 2, "head_size": 8, "sliding_window": [3, 0]},
         "model.norm": {"num_heads": 2, "num_kv_heads": 2, "head_size": 8}}


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 2, dtype=torch.bfloat16))
        self.embedding_dim = 2


class RowParallelLinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 4, dtype=torch.bfloat16))
        self.input_size, self.output_size, self.reduce_results, self.tp_size = 4, 2, True, 2


class FusedMoE(torch.nn.Module):
    """A fused-MoE module that all-reduces its output across the ranks, or leaves that to its caller."""
    tp_size = 2

    def __init__(self, reduces: bool = True) -> None:
        super().__init__()
        self.reduce_results = reduces


class _Model(torch.nn.Module):
    """An embedding and one row-parallel layer: the two sites whose partials a rank binds."""

    def __init__(self, moe_layers: int = 0, flash_attn: int | None = None, moe_reduces: bool = True) -> None:
        super().__init__()
        self.embed, self.proj = VocabParallelEmbedding(), RowParallelLinear()
        if moe_layers:
            self.model = torch.nn.ModuleDict({"layers": torch.nn.ModuleList(
                torch.nn.ModuleDict({"mlp": torch.nn.ModuleDict({"experts": FusedMoE(moe_reduces)})}) for _ in range(moe_layers))})
        if flash_attn is not None:
            self.attn = torch.nn.Identity()
            self.attn.impl = SimpleNamespace(vllm_flash_attn_version=flash_attn)


class _Committer:
    """A committer that keeps what the rank did with it; it has no arena and checks no identities."""
    name = "stand-in"

    def __init__(self, engine: Any, run_id: str) -> None:
        self.engine, self.run_id, self.model = engine, run_id, engine.model
        self.sources: list[Any] = []
        self.registered: list[Any] = []

    def attach_source(self, src: Any) -> None:
        self.sources.append(src)

    def register_weights(self) -> dict[str, Any]:
        self.registered.append(self.model)
        return {"root": ROOT, "tensors": len(list(self.model.named_parameters()))}


class _Arena(_Committer):
    """One that stages a step in an arena, which the rank sizes."""
    windowed = False

    def __init__(self, engine: Any, run_id: str) -> None:
        super().__init__(engine, run_id)
        self.arena: Any = object()
        self.arena_sizing: dict[str, Any] | None = None

    def size_arena(self, envelope: dict[str, Any], *, hidden_m1: bool) -> dict[str, Any]:
        self.arena_sizing = {"envelope": dict(envelope), "hidden_m1": hidden_m1, "sources": len(self.sources)}
        return self.arena_sizing

    def enable_windows(self, rec: dict[str, Any] | None) -> dict[str, Any]:
        return {"ring_of": rec}


class _Windowed(_Arena):
    windowed = True


class _Direct(_Arena):
    """One that could size an arena and has none."""

    def __init__(self, engine: Any, run_id: str) -> None:
        super().__init__(engine, run_id)
        self.arena = None


class _Checking(_Committer):
    """One that checks a run's identities, and so takes the rank's attention declarations; it names its own hidden sources."""
    attention_layer_decls: list | None = None
    hidden_source_cls, hidden_source_cls_fa3 = "the committer's FA2 source", "the committer's FA3 source"

    def identity_check(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        raise AssertionError("making the committer checks no run")


def committer(engine: Any, run_id: str) -> _Committer:
    return _Committer(engine, run_id)


def arena(engine: Any, run_id: str) -> _Committer:
    return _Arena(engine, run_id)


def windowed(engine: Any, run_id: str) -> _Committer:
    return _Windowed(engine, run_id)


def direct(engine: Any, run_id: str) -> _Committer:
    return _Direct(engine, run_id)


def checking(engine: Any, run_id: str) -> _Committer:
    return _Checking(engine, run_id)


class _Rank(TP2CaptureWorkerExtension):
    _tp2_commit: dict[str, Any]

    def __init__(self, model: torch.nn.Module) -> None:
        self.model_runner = SimpleNamespace(get_model=lambda: model)
        self.vllm_config = SimpleNamespace(model="org/model")


@pytest.fixture(autouse=True)
def _facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RW, "_tp_facts", lambda worker: FACTS)


def _make(rank: _Rank, kind: str = "committer", **kw: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"kind": f"plugin:{__name__}:{kind}", "run_id": "run", "cp": {"repo": "org/model"}, "case": "case",
                            "engine_kwargs": {"seed": 0}, "hidden_so": None, "hidden_ns": None, "envelope": ENVELOPE}
    return rank.tp2_commit_make(**(args | kw))


def _made(kind: str = "committer", model: torch.nn.Module | None = None, **kw: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """(the rank's answer, the state it keeps, its committer)."""
    rank = _Rank(model if model is not None else _Model())
    answer = _make(rank, kind, **kw)
    return answer, rank._tp2_commit, rank._tp2_commit["com"]


def test_a_rank_answers_what_it_made():
    answer, _, com = _made()
    assert answer == {"rank": 1, "committer": "stand-in", "post_load_tensors": [], "attention_hidden": MLA, "moe_source": None,
                      "tp_partial_source": PARTIALS, "flip_weight": None, "serve_shard_from": None, "weights_as_loaded": None,
                      "weight_registration": {"root": ROOT, "tensors": 2}, "weights_pin": UNPINNED, "attention_plane_classes": UNIFORM}
    assert json.loads(json.dumps(answer)) == answer
    assert [type(s) for s in com.sources] == [TPPartialSource] and com.registered == [com.model]


def test_the_committer_is_made_over_the_ranks_own_model_under_the_ranks_run_id():
    model = _Model()
    rank = _Rank(model)
    _make(rank, cp={"repo": "org/other"}, case="tp-case", engine_kwargs={"seed": 7})
    com = rank._tp2_commit["com"]
    eng = com.engine
    assert com.run_id == "run/rank1" and rank._tp2_commit["eng"] is eng
    assert eng.runner is rank.model_runner and eng.model is model
    assert (eng.cp, eng.kwargs, eng.env_flags, eng.build_s, eng.case) == ({"repo": "org/other"}, {"seed": 7}, {}, 0.0, "tp-case")
    assert sorted(vars(eng)) == ["build_s", "case", "cp", "env_flags", "kwargs", "llm", "model", "runner"]
    # the engine's LLM is the rank alone: a probe that asks every rank asks this one
    assert eng.llm.llm_engine.vllm_config is rank.vllm_config
    assert eng.llm.collective_rpc(lambda worker: worker) == [rank]


@pytest.mark.parametrize("given, options", [
    ({}, {"arena_mb": None, "chunk": None, "hash_threads": None, "profile_hooks": False, "commit_scheme": None}),
    ({"arena_mb": 96, "commit_scheme": "rows-sha512"},
     {"arena_mb": 96, "chunk": None, "hash_threads": None, "profile_hooks": False, "commit_scheme": "rows-sha512"}),
])
def test_the_commits_door_is_handed_the_drivers_kind_arena_and_scheme(monkeypatch, given, options):
    seen = []

    def door(kind: str, engine: Any, args: Any, run_id: str) -> _Committer:
        seen.append((kind, vars(args), run_id))
        return _Committer(engine, run_id)

    monkeypatch.setattr(PC, "make_committer", door)
    _Rank(_Model()).tp2_commit_make(kind="native_collect_v2b", run_id="r7", cp={}, case="case", engine_kwargs={}, hidden_so=None,
                                    hidden_ns=None, envelope=ENVELOPE, **given)
    assert seen == [("native_collect_v2b", options, "r7/rank1")]


def test_the_rank_keeps_the_commits_state_for_the_calls_that_follow():
    answer, st, com = _made()
    assert sorted(set(st) - set(TABLES)) == ["base", "com", "eng", "match_capture_dir", "match_tokens", "n_requests", "program_digest",
                                             "rank", "rc", "replay", "request_ids", "request_order", "required_manifest", "swap_requests",
                                             "swap_rule", "tps", "weights", "weights_pin", "world"]
    assert (st["rank"], st["world"], st["rc"], st["n_requests"]) == (1, 2, None, 1)
    assert st["weights"] is com.model and st["tps"] is com.sources[0] and st["weights_pin"] == answer["weights_pin"]
    assert st["base"] == PC.baseline_state(st["eng"]) == {"runner_had": {"execute_model": False, "sample": False}, "hooks": 0,
                                                           "wrapped_forwards": 0, "sampler_hooks": 0, "cuda_alloc_mib": 0}
    assert [st[k] for k in ("program_digest", "required_manifest", "match_tokens", "swap_requests", "swap_rule", "match_capture_dir",
                            "replay")] == [None] * 7
    assert (st["request_order"], st["request_ids"]) == ([], {}) and [st[table] for table in TABLES] == [{}, {}]


def test_the_state_holds_its_own_copy_of_what_the_driver_declared(tmp_path):
    order, tokens, replay = ["x", "y"], {"x": [5, 6]}, {"programs": {"1": "p"}, "seed": 9}
    tables: dict[str, Any] = {table: {"x": at, "y": 3} for at, table in enumerate(TABLES)}
    _, st, _ = _made(n_requests=2, request_order=order, match_tokens=tokens, replay=replay, match_capture_dir=tmp_path / "capture",
                     program_digest="d" * 128, **tables)
    assert (st["n_requests"], st["program_digest"], st["match_capture_dir"]) == (2, "d" * 128, str(tmp_path / "capture"))
    for key, given in (("request_order", order), ("match_tokens", tokens), ("replay", replay), *tables.items()):
        assert st[key] == given and st[key] is not given, key


@pytest.mark.parametrize("digests, own", [({"0": "p0", "1": "p1"}, "p1"), ({0: "p0", 1: "p1"}, "p1"), ({"0": "p0"}, None), ("p", "p")])
def test_a_program_digest_given_per_rank_is_read_by_the_ranks_index(digests, own):
    assert _made(program_digest=digests)[1]["program_digest"] == own


def test_a_request_is_labelled_by_the_drivers_map_or_by_its_place_in_the_declared_order():
    assert _made(request_order=["x", "y"])[1]["request_ids"] == {"x": "r0", "y": "r1"}
    assert _made(request_order=[7, 8])[1]["request_ids"] == {"7": "r0", "8": "r1"}
    given = {"x": "r1", "y": "r0"}
    labels = _made(request_order=["x", "y"], request_ids=given)[1]["request_ids"]
    assert labels == given and labels is not given


@pytest.mark.parametrize("swap, applied, rule", [
    ({"1": ["r0", "r1"]}, ["r0", "r1"], {"a": "r0", "b": "r1", "lo": None, "hi": None, "mode": "all"}),
    ({1: ["r0", "r1"]}, ["r0", "r1"], {"a": "r0", "b": "r1", "lo": None, "hi": None, "mode": "all"}),
    ({"1": ["r1", "r0", 2, 5]}, ["r1", "r0", 2, 5], {"a": "r1", "b": "r0", "lo": 2, "hi": 5, "mode": "steps"}),
    ({"1": ["r0", "r9"]}, None, None),              # no request of the run carries the second label
    ({"1": ["r0"]}, None, None),                    # no pair
])
def test_the_swap_negative_is_applied_on_the_rank_it_names_when_each_label_is_one_request(swap, applied, rule):
    answer, st, _ = _made(request_order=["x", "y"], tp_swap_requests=swap)
    assert answer["swap_requests_applied"] == applied
    assert (st["swap_requests"], st["swap_rule"]) == (applied, rule)


def test_a_swap_named_for_another_rank_is_not_this_ranks():
    answer, st, _ = _made(request_order=["x", "y"], tp_swap_requests={"0": ["r0", "r1"]})
    assert "swap_requests_applied" not in answer and (st["swap_requests"], st["swap_rule"]) == (None, None)


def test_the_rank_binds_its_partials_of_every_collective_site():
    answer, st, com = _made()
    src = com.sources[0]
    assert st["tps"] is src and (src.rank, src.world, src.label_rank, src.omit_site, src.flip_site) == (1, 2, 1, None, None)
    assert src.c is com and src.sites == [("embed", "VocabParallelEmbedding"), ("proj", "RowParallelLinear")]
    assert answer["tp_partial_source"] == PARTIALS


@pytest.mark.parametrize("flags", [{"tp_partials": False}, {"tp_partials_omit_rank": 1}, {"tp_partials_omit_rank": "1"}])
def test_a_rank_told_to_omit_its_partials_binds_none(flags):
    answer, st, com = _made(**flags)
    assert answer["tp_partial_source"] is None and st["tps"] is None and com.sources == []


@pytest.mark.parametrize("flags, block, read", [
    ({"tp_partials_omit_rank": 0}, {}, (1, None, None, "gross")),
    ({"tp_relabel": {"1": 0}}, {"relabel_rank": 0}, (0, None, None, "gross")),
    ({"tp_relabel": {1: "0"}}, {"relabel_rank": 0}, (0, None, None, "gross")),
    ({"tp_relabel": {"0": 1}}, {}, (1, None, None, "gross")),
    ({"tp_partials_omit_site": {"1": "proj"}}, {"omit_site": "proj"}, (1, "proj", None, "gross")),
    ({"tp_partials_omit_site": {"0": "proj"}}, {}, (1, None, None, "gross")),
    ({"tp_partials_flip_site": {"1": "proj"}}, {"flip_site": "proj"}, (1, None, "proj", "gross")),
    ({"tp_partials_flip_site": {1: "proj:ulp"}}, {"flip_site": "proj:ulp"}, (1, None, "proj", "ulp")),
    ({"tp_partials_flip_site": {"0": "proj"}}, {}, (1, None, None, "gross")),
])
def test_a_negative_on_the_partials_reaches_the_source_of_the_rank_it_names(flags, block, read):
    answer, _, com = _made(**flags)
    src = com.sources[0]
    assert answer["tp_partial_source"] == PARTIALS | block
    assert (src.label_rank, src.omit_site, src.flip_site, src.flip_mode) == read


def _experts(layers: int) -> list[str]:
    return [f"model.layers.{k}.mlp.experts" for k in range(layers)]


@pytest.mark.parametrize("layers, modules", [(1, _experts(1)), (4, _experts(4)), (6, _experts(4) + ["..."])])
def test_a_model_with_moe_layers_gets_the_moe_source_before_the_partials(layers, modules):
    answer, _, com = _made(model=_Model(moe_layers=layers))
    assert answer["moe_source"] == {"layers": layers, "tensors_per_layer": MS.TENSORS_PER_LAYER, "modules": modules}
    assert [type(s) for s in com.sources] == [MS.MoeBlockSource, TPPartialSource]
    moe = com.sources[0]
    assert moe.c is com and moe.layers == {name: k for k, name in enumerate(_experts(layers))} and moe.at_calls is False


def test_the_moe_source_is_made_for_the_ranks_world_and_the_manifest_of_record(tmp_path):
    # in a world of 2, a MoE module that leaves the all-reduce to its caller has its own output bound
    _, _, com = _made(model=_Model(moe_layers=1, moe_reduces=False))
    assert com.sources[0].outputs == {"model.layers.0.mlp.experts"} and com.sources[0].at_calls is False
    man = tmp_path / "required.json"
    man.write_text(json.dumps({"identities": [{"family": "tp_partial"}]}))
    _, _, com = _made(model=_Model(moe_layers=1), required_manifest=str(man))
    assert com.sources[0].outputs == set() and com.sources[0].at_calls is True      # a manifest that names no MoE plane


def test_a_model_without_moe_layers_gets_no_moe_source():
    answer, _, com = _made()
    assert answer["moe_source"] is None and [type(s) for s in com.sources] == [TPPartialSource]


def test_a_moe_source_that_cannot_be_made_is_reported_and_the_rank_goes_on(monkeypatch):
    def refuse(*a: Any, **k: Any) -> None:
        raise ImportError("no fused_moe in this " + "vLLM" * 80)

    monkeypatch.setattr(MS, "MoeBlockSource", refuse)
    answer, st, com = _made(model=_Model(moe_layers=1))
    assert answer["moe_source"] == {"error": repr(ImportError("no fused_moe in this " + "vLLM" * 80))[:200]}
    assert len(answer["moe_source"]["error"]) == 200
    assert [type(s) for s in com.sources] == [TPPartialSource] and answer["weight_registration"] == {"root": ROOT, "tensors": 2}
    assert st["com"] is com


def _hidden(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Stand in for the hidden source's maker; the list holds each (committer, tap build, its keywords) it is asked for."""
    asked: list[tuple] = []

    def make(com: Any, so_path: str, **kw: Any) -> str:
        asked.append((com, so_path, kw))
        return "hidden source"

    monkeypatch.setattr(HS, "make_hidden_source", make)
    return asked


@pytest.mark.parametrize("version, family", [(2, "fa2_hidden_m1_stream"), (3, "fa3_hidden_m1_stream")])
def test_a_flash_attention_model_given_a_tap_build_gets_the_hidden_source_first(monkeypatch, version, family):
    asked = _hidden(monkeypatch)
    answer, _, com = _made("arena", model=_Model(flash_attn=version), hidden_so="/taps/verity_fa2_matReq.so")
    assert answer["attention_hidden"] == {"family": family, "why": f"the attention impls' vllm_flash_attn_version [{version}]"}
    assert answer["hidden_source"] == "/taps/verity_fa2_matReq.so"
    assert com.sources[0] == "hidden source" and type(com.sources[1]) is TPPartialSource
    assert asked == [(com, "/taps/verity_fa2_matReq.so", {"ns": "verity_fa2_matReq", "manifest": None, "fa2_cls": HS.HiddenM1Source,
                                                          "fa3_cls": None})]
    assert answer["arena_sizing"]["hidden_m1"] is True


def test_the_hidden_source_takes_the_namespace_the_driver_gives_and_the_committers_own_source_classes(monkeypatch):
    asked = _hidden(monkeypatch)
    _made("checking", model=_Model(flash_attn=2), hidden_so="/taps/tap.so", hidden_ns="verity_fa2_matAll")
    assert [kw for _, _, kw in asked] == [{"ns": "verity_fa2_matAll", "manifest": None, "fa2_cls": "the committer's FA2 source",
                                           "fa3_cls": "the committer's FA3 source"}]


@pytest.mark.parametrize("model, hidden_so, hidden", [
    (lambda: _Model(flash_attn=2), None, {"family": "fa2_hidden_m1_stream", "why": "the attention impls' vllm_flash_attn_version [2]"}),
    (lambda: _Model(), "/taps/tap.so", MLA),
])
def test_without_a_tap_build_or_a_flash_attention_impl_no_hidden_source_is_attached(monkeypatch, model, hidden_so, hidden):
    asked = _hidden(monkeypatch)
    answer, _, com = _made("arena", model=model(), hidden_so=hidden_so)
    assert answer["attention_hidden"] == hidden and "hidden_source" not in answer
    assert asked == [] and [type(s) for s in com.sources] == [TPPartialSource]
    assert answer["arena_sizing"]["hidden_m1"] is False


def test_a_manifest_that_names_no_hidden_stream_leaves_the_hidden_source_out(monkeypatch, tmp_path):
    asked = _hidden(monkeypatch)
    man = tmp_path / "required.json"
    man.write_text(json.dumps({"identities": [{"family": "tp_partial"}]}))
    answer, _, _ = _made(model=_Model(flash_attn=2), hidden_so="/taps/tap.so", required_manifest=str(man))
    assert answer["attention_hidden"] == {"family": None, "why": "the required manifest names no attention hidden stream"}
    assert asked == [] and "hidden_source" not in answer


def test_the_hidden_source_is_handed_the_manifest_that_names_its_stream(monkeypatch, tmp_path):
    asked = _hidden(monkeypatch)
    doc = {"identities": [{"family": "fa2_hidden_m1_stream"}, {"family": "tp_partial"}]}
    man = tmp_path / "required.json"
    man.write_text(json.dumps(doc))
    answer, _, _ = _made(model=_Model(flash_attn=2), hidden_so="/taps/tap.so", required_manifest=str(man))
    assert answer["attention_hidden"] == {"family": "fa2_hidden_m1_stream", "why": "required manifest identities"}
    assert [kw["manifest"] for _, _, kw in asked] == [doc]


def test_the_taps_the_driver_binds_are_made_for_this_rank(tmp_path):
    man = tmp_path / "required.json"
    man.write_text(json.dumps({"identities": []}))
    seen = []

    def taps(com: Any, model: Any, manifest: str | None, rank: int) -> dict[str, Any]:
        seen.append((com, model, manifest, rank))
        return {"taps": ["rope"], "rank": rank}

    answer, _, com = _made(tap_sources=taps, required_manifest=str(man))
    assert answer["tap_sources"] == {"taps": ["rope"], "rank": 1} and seen == [(com, com.model, str(man), 1)]
    assert "tap_sources" not in _made()[0]


def test_a_committer_with_an_arena_has_it_sized_from_the_envelope_after_its_sources_are_attached():
    answer, _, com = _made("arena")
    assert answer["arena_sizing"] == {"envelope": ENVELOPE, "hidden_m1": False, "sources": 1} and "window_ring" not in answer
    assert com.registered == [com.model]


def test_a_windowed_committer_has_its_ring_enabled_with_the_sizing():
    answer, _, _ = _made("windowed")
    assert answer["window_ring"] == {"ring_of": answer["arena_sizing"]}


@pytest.mark.parametrize("kind", ["committer", "direct"])
def test_a_committer_without_an_arena_is_not_sized(kind):
    answer, _, _ = _made(kind)
    assert "arena_sizing" not in answer and "window_ring" not in answer


def _manifest(tmp_path: Path, **doc: Any) -> str:
    path = tmp_path / "required.json"
    path.write_text(json.dumps({"manifest_digest": "m" * 64, "complete": True, "populations": {"identities": 7}, "ranks": 2} | doc))
    return str(path)


@pytest.mark.parametrize("digests, given, for_rank, matches, kept", [
    ({"program_digest_per_rank": {"0": "p0", "1": "p1"}}, "p1", "p1", True, "p1"),
    ({"program_digest_per_rank": {"0": "p0", "1": "p1"}}, "p0", "p1", False, "p0"),
    ({"program_digest_per_rank": {"0": "p0", "1": "p1"}}, None, "p1", True, "p1"),          # the driver gives none: the manifest's
    ({"program_digest_per_rank": {"0": "p0", "1": "p1"}}, {"1": "p1"}, "p1", True, "p1"),
    ({"program_digest": "p"}, "p", "p", True, "p"),                                          # one Program for every rank
    ({"program_digest_per_rank": {"0": "p0"}, "program_digest": "p"}, None, "p", True, "p"),
    ({}, "p1", None, False, "p1"),
    ({}, None, None, True, None),
])
def test_the_manifest_of_record_is_read_for_this_ranks_program(tmp_path, digests, given, for_rank, matches, kept):
    path = _manifest(tmp_path, **digests)
    answer, st, _ = _made(required_manifest=path, program_digest=given)
    assert answer["required_manifest"] == {"path": path, "manifest_digest": "m" * 64, "complete": True, "program_digest_for_rank": for_rank,
                                           "program_digest_matches": matches, "identities": 7, "ranks": 2}
    assert st["program_digest"] == kept and st["required_manifest"] == path
    assert st["manifest_doc"] == json.loads(Path(path).read_text())


def test_a_manifest_that_states_no_population_reads_as_none(tmp_path):
    path = tmp_path / "required.json"
    path.write_text("{}")
    answer, _, _ = _made(required_manifest=str(path))
    assert answer["required_manifest"] == {"path": str(path), "manifest_digest": None, "complete": None, "program_digest_for_rank": None,
                                           "program_digest_matches": True, "identities": None, "ranks": None}


def test_a_manifest_that_is_no_document_is_reported_by_its_path(tmp_path):
    path = tmp_path / "required.json"
    path.write_text("[]")
    answer, st, _ = _made(required_manifest=str(path))
    assert answer["required_manifest"] == {"path": str(path), "error": repr(AttributeError("'list' object has no attribute 'get'"))}
    assert "manifest_doc" not in st and st["required_manifest"] == str(path)


def test_without_a_manifest_the_answer_has_no_manifest_block():
    answer, st, _ = _made()
    assert "required_manifest" not in answer and "manifest_doc" not in st


def _program_dir(tmp_path: Path, name: str, impls: Any) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "result.json").write_text(json.dumps({"attention_impls_observed": impls}))
    return d


@pytest.mark.parametrize("where", ["replay", "weights_pin"])
def test_a_committer_that_checks_identities_takes_the_attention_declarations_of_the_ranks_program(tmp_path, where):
    own, other = _program_dir(tmp_path, "rank1", IMPLS), _program_dir(tmp_path, "rank0", {})
    dirs = {"0": str(other), "1": str(own)}
    answer, _, com = _made("checking", **({"replay": {"programs": dirs}} if where == "replay" else {"weights_pin": {"program_dirs": dirs}}))
    assert com.attention_layer_decls == FP.declarations_from_attention_impls(IMPLS)
    assert [(d.layer, d.NH, d.KVH, d.D, d.window) for d in com.attention_layer_decls] == [(0, 4, 2, 8, None), (1, 2, 2, 8, 3)]
    assert answer["attention_plane_classes"] == {
        "rank": 1, "declarations": 2, "source": str(own / "result.json"), "classes": ["global-gqa", "windowed-mha"],
        "per_rank_heads": [[2, 2, 8], [4, 2, 8]],
        "rule": "per-layer plane class derived per step from the rank Program's attention statics (fa2_plane_classes)"}
    assert answer["weights_pin"] == UNPINNED


def test_the_replays_program_is_read_before_the_weights_pins(tmp_path):
    replayed, pinned = _program_dir(tmp_path, "replayed", IMPLS), _program_dir(tmp_path, "pinned", {})
    answer, _, _ = _made("checking", replay={"programs": {"1": str(replayed)}}, weights_pin={"program_dirs": {"1": str(pinned)}})
    assert answer["attention_plane_classes"]["source"] == str(replayed / "result.json")


@pytest.mark.parametrize("flags", [{}, {"replay": {"programs": {"0": "elsewhere"}}}, {"replay": {"programs": {"1": "nowhere"}}}])
def test_without_a_record_of_the_ranks_program_the_uniformity_rule_stays(flags):
    answer, _, com = _made("checking", **flags)
    assert answer["attention_plane_classes"] == UNIFORM and com.attention_layer_decls is None


def test_a_record_with_no_layer_of_attention_leaves_the_uniformity_rule(tmp_path):
    own = _program_dir(tmp_path, "rank1", {"model.norm": {"num_heads": 2, "num_kv_heads": 2, "head_size": 8}})
    answer, _, com = _made("checking", replay={"programs": {"1": str(own)}})
    assert answer["attention_plane_classes"] == UNIFORM and com.attention_layer_decls is None


def test_a_committer_that_checks_no_identities_reads_no_record(tmp_path):
    own = _program_dir(tmp_path, "rank1", IMPLS)
    answer, _, com = _made(replay={"programs": {"1": str(own)}})
    assert answer["attention_plane_classes"] == UNIFORM and not hasattr(com, "attention_layer_decls")


def test_a_record_whose_statics_are_no_integers_is_reported_beside_the_uniformity_rule(tmp_path):
    own = _program_dir(tmp_path, "rank1", {"model.layers.0.self_attn.attn": {"num_heads": "four", "num_kv_heads": 2, "head_size": 8}})
    answer, _, com = _made("checking", replay={"programs": {"1": str(own)}})
    assert answer["attention_plane_classes"] == UNIFORM | {"error": repr(ValueError("invalid literal for int() with base 10: 'four'"))}
    assert com.attention_layer_decls is None
