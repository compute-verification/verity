"""The route source on CPU against the GLM-5.3 capture map (`manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json`): a stand-in
model with the map's modules, whose attention layers decide their route the way the pinned vLLM's `MLAAttention._use_sparse_mha` does
and are asked the way `DeepseekV32Attention._sparse_indexer_and_attn` and `forward_impl` ask, on a dense prefill, a decode, a mixed and a
long prefill step.  Each step's committed names are the ones `capture-map ids --route ... --names` prints for its routes, and the
committer's identity check passes on them with the source attached and fails without it; a name of the other route, a missing one, an
unlisted hook name or a layer without a route fails the step."""
from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.commit import capture_map as CM
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TP_PARTIAL_RE, NativeHostCommitter, TensorMeta
from verity_vllm.pipeline import cli

ROOT = Path(__file__).parents[2]
MAP = ROOT / "manifests" / "capture-maps" / "glm-moe-dsa-eager-hopper-tp8-v0.json"
CONFIG_PATH = ROOT / "tests" / "check" / "fixtures" / "glm_moe_dsa" / "config.json"
CMAP = CM.load(MAP)
CONFIG = json.loads(CONFIG_PATH.read_text())
TP = CMAP["target"]["tensor_parallel_size"]
ENV = CM.symbols(CMAP, CONFIG, TP)
L = ENV["L"]
DENSE, SPARSE = ("dense-mha",), ("sparse-mqa",)
#: the map's module classes whose class hierarchy holds one of the committer's ACQUIRE_CLASSES on the pinned vLLM (GateLinear is a
#: ReplicatedLinear, the fused A projection a MergedColumnParallelLinear); ParallelLMHead is skipped
HOOKED = {"ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "ReplicatedLinear", "GateLinear",
          "DeepSeekV2FusedQkvAProjLinear", "RMSNorm", "LayerNorm", "SiluAndMul", "RotaryEmbedding", "VocabParallelEmbedding",
          "LogitsProcessor"}


class Attention(torch.nn.Module):
    """The route decision of the pinned vLLM's sparse MLA layer, with masked MHA unavailable (it needs compute capability 10.x)."""

    def _use_sparse_mha(self, attn_metadata: Any) -> bool:
        prefill = attn_metadata.prefill
        if prefill is None:
            return False
        return bool(prefill.use_dense_mha)

    def forward(self, md: Any) -> None:
        """`_sparse_indexer_and_attn`: dense MHA through `forward_impl`, which asks again when the step has prefill tokens."""
        if self._use_sparse_mha(md) and md.num_prefill_tokens > 0:
            self._use_sparse_mha(md)


def glm_model() -> torch.nn.Module:
    """A module tree with every module of the map at its layers, the root of the map's model class, attention layers that decide
    their route, linears that hold the tensor-parallel size, and the config the map's symbols read."""
    mod, _, qual = CMAP["model_class"].rpartition(".")
    root = type(qual, (torch.nn.Module,), {"__module__": mod})()
    root.config = types.SimpleNamespace(to_dict=lambda: dict(CONFIG))
    classes: dict[str, type] = {}
    for e in sorted(CMAP["modules"], key=lambda e: e["pattern"].count(".")):
        if not e["pattern"]:
            continue
        layers = CM.layer_set(CMAP, e.get("when", "all"), ENV) if "{i}" in e["pattern"] else [None]
        for i in layers[:1] if e.get("shared") else layers:
            name = CM.at_layer(e["pattern"], i)
            parent = root
            for part in name.split(".")[:-1]:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = getattr(parent, part)
            base = Attention if name.endswith(".self_attn") else torch.nn.Module
            cls = classes.setdefault(e["type"], type(e["type"], (base,), {}))
            m = cls()
            if e["type"].endswith("Linear"):
                m.tp_size = TP
            parent.add_module(name.rsplit(".", 1)[-1], m)
    return root


def metadata(*, prefill: bool, dense: bool = False, decodes: int = 0, prefill_tokens: int = 256) -> types.SimpleNamespace:
    pre = types.SimpleNamespace(use_dense_mha=dense) if prefill else None
    return types.SimpleNamespace(prefill=pre, num_decode_tokens=decodes, num_prefill_tokens=prefill_tokens if prefill else 0)


#: (attention metadata, the routes the step takes, its phase): a 256-token prefill, a decode, a mixed step, a prefill over TOPK keys
STEPS = [(metadata(prefill=True, dense=True), DENSE, "prefill"),
         (metadata(prefill=False, decodes=1), SPARSE, "decode"),
         (metadata(prefill=True, dense=True, decodes=1), ("dense-mha", "sparse-mqa"), "prefill"),
         (metadata(prefill=True, dense=False), SPARSE, "prefill")]


def map_names(capsys, routes: tuple[str, ...], phase: str, rank: int = 3) -> list[str]:
    """What `capture-map ids --names` prints for one step's module-hook, sampler and all-reduce-norm names (no rank-tagged partial)."""
    argv = ["capture-map", "ids", "--map", str(MAP), "--config", str(CONFIG_PATH), "--phase", phase, "--rank", str(rank), "--names"]
    argv += [a for r in routes for a in ("--route", r)] + [a for b in RS.BINDERS for a in ("--binder", b)]
    capsys.readouterr()
    assert cli.main(argv) == 0
    lines = [ln.split(" ", 1) for ln in capsys.readouterr().out.splitlines()]
    return [n for _cid, n in lines if not TP_PARTIAL_RE.search(n)]


class Run:
    def __init__(self) -> None:
        self.model = glm_model()
        self.com = NativeHostCommitter(None, run_id="glm53-routes", gpu_tree=True)
        self.src = RS.RouteSource(self.com, self.model, CMAP, rank=3)
        self.attn = dict(sorted(self.src.layers.items()))
        self.com._modules = [(n, m) for n, m in self.model.named_modules() if type(m).__name__ in HOOKED]
        self.com._modules.append(("runner.sampler", torch.nn.Module()))
        self.src.install()

    def step(self, k: int, md: Any, skip: tuple[int, ...] = ()) -> None:
        self.com._cur = types.SimpleNamespace(step=k)
        for layer, m in self.attn.items():
            if layer not in skip:
                m(md)
        self.com._cur = None

    def check(self, steps: dict[int, list[str]], *, sources: list[Any] | None = None) -> dict[str, Any]:
        self.com.sources = [self.src] if sources is None else sources
        self.com._layouts = {}
        for k, names in steps.items():
            self.com._layouts[k] = [TensorMeta(i, n, "bfloat16", (4, 8), 64, GPU_CHUNK_BYTES, i, 1) for i, n in enumerate(names)]
            for i, m in enumerate(self.com._layouts[k]):
                m.dev_off = m.stream_off = 64 * i
        return self.com.identity_check()


@pytest.fixture(scope="module")
def run():
    r = Run()
    for k, (md, _routes, _phase) in enumerate(STEPS):
        r.step(k, md)
    return r


def test_the_map_describes_the_model_and_the_source_wraps_every_attention_layer():
    model = glm_model()
    assert RS.world_of(model) == TP and RS.find_map(model) is not None and RS.find_map(model, world=2) is None
    assert RS.find_map(torch.nn.Linear(2, 2)) is None
    src = RS.RouteSource(None, model, RS.find_map(model))
    assert sorted(src.layers) == list(range(L))
    attn = src.layers[0]
    src.install()
    assert "_use_sparse_mha" in vars(attn) and attn._use_sparse_mha(metadata(prefill=True, dense=True)) is True
    assert src.routes == {}, "an answer outside a committer step is not recorded"
    src.uninstall()
    assert "_use_sparse_mha" not in vars(attn) and attn._use_sparse_mha.__func__ is Attention._use_sparse_mha


def test_each_step_records_the_routes_its_layers_took(run):
    for k, (_md, routes, phase) in enumerate(STEPS):
        assert run.src.step_routes(k) == (tuple(sorted(routes)), phase)
    assert run.src.calls == L * (2 + 1 + 2 + 1), "a dense step asks twice per layer (forward_impl again), a sparse step once"
    assert run.src.stats["steps_by_routes"] == {"prefill dense-mha": 1, "decode sparse-mqa": 1, "prefill dense-mha+sparse-mqa": 1,
                                                "prefill sparse-mqa": 1}


def test_capture_map_ids_and_the_committer_agree_on_every_route(run, capsys):
    steps = {k: map_names(capsys, routes, phase) for k, (_md, routes, phase) in enumerate(STEPS)}
    for k, (_md, routes, phase) in enumerate(STEPS):
        assert sorted(steps[k]) == sorted(run.src.names_at(tuple(sorted(routes)), phase))
    kv_b = [f"model.layers.{i}.self_attn.kv_b_proj/0" for i in range(L)]
    assert set(kv_b) <= set(steps[0]) and not set(kv_b) & set(steps[1]) and set(kv_b) <= set(steps[2])
    assert sorted(steps[2]) == sorted(steps[0]) and sorted(steps[3]) == sorted(steps[1])
    wq_b = "model.layers.0.self_attn.indexer.wq_b/0"
    assert wq_b in steps[0] and wq_b in steps[1], "DeepseekV32Attention.forward runs the indexer's wq_b before it decides the route"
    got = run.check(steps)
    assert got["ok"] is True, got["violations"][:2]
    without = run.check(steps, sources=[])
    first = without["violations"][0]
    assert without["ok"] is False and "model.layers.0.self_attn.q_a_layernorm" in first["missing_modules"]
    assert any(v["step"] == 1 and v["multiplicity_vs_step0"] for v in without["violations"]), "a sparse step differs from step 0"


def test_a_step_off_its_routes_fails(run, capsys):
    steps = {k: map_names(capsys, routes, phase) for k, (_md, routes, phase) in enumerate(STEPS)}
    extra = "model.layers.5.self_attn.kv_b_proj/0"
    dropped = "model.layers.7.self_attn.kv_b_proj/0"
    fused = "model.layers.0.self_attn.q_a_layernorm"
    bad = {**steps, 1: steps[1] + [extra], 0: [n for n in steps[0] if n != dropped] + [fused]}
    got = run.check(bad)
    by_step = {v["step"]: v for v in got["violations"]}
    assert got["ok"] is False and sorted(by_step) == [0, 1]
    assert by_step[1]["routes_bad"] == {extra: (1, 0)} and by_step[1]["routes_bad_n"] == 1
    assert by_step[0]["routes_bad"] == {dropped: (0, 1), fused: (1, 0)}
    assert by_step[0]["missing_modules"] == [dropped.split("/")[0]]


def test_a_step_without_a_route_on_every_layer_fails(run, capsys):
    run.step(9, STEPS[1][0], skip=(3,))
    steps = {0: map_names(capsys, DENSE, "prefill"), 9: map_names(capsys, SPARSE, "decode")}
    got = run.check(steps)
    assert got["ok"] is False and got["violations"][0]["step"] == 9
    assert got["violations"][0]["routes_bad"] == {f"<1 of {L} layers recorded no route>": (len(steps[9]), 0)}
    run.src.routes.pop(9)


def test_a_tp_rank_attaches_the_route_source_and_every_rank_must(tmp_path):
    class Com:
        def __init__(self) -> None:
            self.sources: list[Any] = []

        def attach_source(self, s: Any) -> None:
            self.sources.append(s)

    attach = TAPS.for_ranks(TAPS.TapSources())
    made = [{"rank": r, "tap_sources": attach(Com(), glm_model(), None, r)} for r in (0, 1)]
    assert all(m["tap_sources"]["route_source"]["stats"]["layers"] == L for m in made)
    versions: dict[str, Any] = {}
    assert TAPS.require_on_every_rank(made, ("a",), versions, TAPS.TapSources(), explicit_classes=False) == ("a",)
    assert sorted(versions["route_source"]) == [0, 1]
    other = attach(Com(), torch.nn.Linear(2, 2), None, 0)
    assert other == {"route_source": None}
    with pytest.raises(SystemExit, match=r"route source did not attach on ranks \[1\]"):
        TAPS.require_on_every_rank([made[0], {"rank": 1, "tap_sources": other}], (), {}, TAPS.TapSources(), explicit_classes=True)
