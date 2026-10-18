"""The route source on CPU against the GLM-5.3 capture map (`manifests/capture-maps/glm-moe-dsa-eager-hopper-tp8-v0.json`): a stand-in
model with the map's modules (`tests/acquire/glm53_model.py`), whose attention layers decide their route the way the pinned vLLM's
`MLAAttention._use_sparse_mha` does and are asked the way `DeepseekV32Attention._sparse_indexer_and_attn` and `forward_impl` ask, on a
dense prefill, a decode, a mixed and a long prefill step.  Each step's committed names are the ones `capture-map ids --route ... --names`
prints for its routes, and the committer's identity check passes on them with the source attached and fails without it; a name of the
other route, a missing one, an unlisted hook name or a layer without a route fails the step."""
from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from tests.acquire.glm53_model import CMAP, CONFIG_PATH, DENSE, HOOKED, L, MAP, SPARSE, STEPS, TP, Attention, glm_model, metadata, run_step
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TP_PARTIAL_RE, NativeHostCommitter, TensorMeta
from verity_vllm.pipeline import cli


def map_names(capsys, routes: tuple[str, ...], phase: str, rank: int = 3, binders: tuple[str, ...] = RS.BINDERS,
              off_target: bool = False) -> list[str]:
    """What `capture-map ids --names` prints for one step's names of `binders` (no rank-tagged partial)."""
    argv = ["capture-map", "ids", "--map", str(MAP), "--config", str(CONFIG_PATH), "--phase", phase, "--rank", str(rank), "--names"]
    argv += [a for r in routes for a in ("--route", r)] + [a for b in binders for a in ("--binder", b)]
    argv += ["--off-target"] if off_target else []
    capsys.readouterr()
    assert cli.main(argv) == 0
    lines = [ln.split(" ", 1) for ln in capsys.readouterr().out.splitlines()]
    return [n for _cid, n in lines if not TP_PARTIAL_RE.search(n)]


class Run:
    def __init__(self, target: bool = True) -> None:
        self.model = glm_model()
        self.com = NativeHostCommitter(None, run_id="glm53-routes", gpu_tree=True)
        self.src = RS.RouteSource(self.com, self.model, CMAP, rank=3, target=target)
        self.com._modules = [(n, m) for n, m in self.model.named_modules() if type(m).__name__ in HOOKED]
        self.com._modules.append(("runner.sampler", torch.nn.Module()))
        self.src.install()
        self.slot = 0

    def step(self, k: int, md: Any, skip: tuple[int, ...] = ()) -> None:
        self.com._cur = types.SimpleNamespace(step=k)
        self.slot = run_step(self.model, md, self.slot, skip)
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
    assert set(steps[2]) == set(steps[0]) | set(steps[1]) and sorted(steps[3]) == sorted(steps[1])
    assert {n.rsplit("/", 1)[1] for n in set(steps[2]) - set(steps[0])} == {"attn_mqa", "out", "topk"}, \
        "a mixed step adds what its decode rows' MQA produces"
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
    assert by_step[0]["missing_modules"] == [] and by_step[0]["unknown_modules"] == [], "the map's modules are held by name"


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
    assert other == {}
    with pytest.raises(SystemExit, match=r"route source did not attach on ranks \[1\]"):
        TAPS.require_on_every_rank([made[0], {"rank": 1, "tap_sources": other}], (), {}, TAPS.TapSources(), explicit_classes=True)
