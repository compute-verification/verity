"""The route source on CPU against the DeepSeek-V4-Pro capture map (`manifests/capture-maps/deepseek-v4-eager-hopper-tp8-v0.json`),
whose `route_site` decides three routes per step at four sites: a stand-in model with the map's modules (`tests/acquire/dsv4_model.py`)
runs the pinned vLLM's decoder layers, FlashMLA's `forward_mqa` split and the CSA indexer's short path on a decode, a scored decode, a
prefill, a short prefill and a mixed step.  Each step's routes are the union of its decisions, its committed names are the ones
`capture-map ids --route ... --names` prints for them, and the committer's identity check passes on them with the source attached and
fails without it; a layer that recorded no attention rows, or indexers that disagree, fail the step."""
from __future__ import annotations

import types
from typing import Any

import pytest
import torch

from tests.acquire.dsv4_model import CMAP, CONFIG_PATH, CSA, HOOKED, L, MAP, STEPS, TP, dsv4_model, metadata, run_step
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.acquire.sources import taps as TAPS
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TP_PARTIAL_RE, NativeHostCommitter, TensorMeta
from verity_vllm.pipeline import cli


def map_names(capsys, routes: tuple[str, ...], phase: str, rank: int = 3, binders: tuple[str, ...] = RS.BINDERS) -> list[str]:
    """What `capture-map ids --names` prints for one step's names of `binders` (no rank-tagged partial)."""
    argv = ["capture-map", "ids", "--map", str(MAP), "--config", str(CONFIG_PATH), "--phase", phase, "--rank", str(rank), "--names"]
    argv += [a for r in routes for a in ("--route", r)] + [a for b in binders for a in ("--binder", b)]
    capsys.readouterr()
    assert cli.main(argv) == 0
    lines = [ln.split(" ", 1) for ln in capsys.readouterr().out.splitlines()]
    return [n for _cid, n in lines if not TP_PARTIAL_RE.search(n)]


class Run:
    def __init__(self, target: bool = True) -> None:
        self.model = dsv4_model()
        self.com = NativeHostCommitter(None, run_id="dsv4-routes", gpu_tree=True)
        self.src = RS.RouteSource(self.com, self.model, CMAP, rank=3, target=target)
        self.com._modules = [(n, m) for n, m in self.model.named_modules() if type(m).__name__ in HOOKED]
        self.com._modules.append(("runner.sampler", torch.nn.Module()))
        self.src.install()

    def step(self, k: int, md: Any, skip: tuple[int, ...] = ()) -> None:
        self.com._cur = types.SimpleNamespace(step=k)
        try:
            run_step(self.model, md, 100 * k, skip)
        finally:
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


def test_the_map_describes_the_model_and_the_source_wraps_every_site():
    model = dsv4_model()
    assert RS.world_of(model) == TP and RS.find_map(model)["id"] == CMAP["id"] and RS.find_map(model, world=2) is None
    src = RS.RouteSource(None, model, RS.find_map(model))
    assert [s["method"] for s in src.sites] == ["_forward_decode", "_forward_prefill", "forward", "forward"]
    assert sorted(src.layers) == list(range(L)) and src.layers[5] is model.model.layers[5].attn
    assert {d: len(ls) for d, ls in src.decisions.items()} == {"rows": L, "hc": L, "indexer": len(CSA)} and len(CSA) == 30
    layer, attn = model.model.layers[4], model.model.layers[4].attn
    src.install()
    assert "forward" in vars(layer) and "forward" in vars(attn.indexer)
    assert {"_forward_decode", "_forward_prefill"} <= set(vars(attn))
    run_step(model, STEPS[0][0])
    assert src.routes == {}, "a call outside a committer step is not recorded"
    src.uninstall()
    assert not {"forward", "_forward_decode", "_forward_prefill"} & (set(vars(layer)) | set(vars(attn)) | set(vars(attn.indexer)))


def test_each_step_records_the_routes_of_its_three_decisions(run):
    for k, (_md, routes, phase) in enumerate(STEPS):
        assert run.src.step_routes(k) == (tuple(sorted(routes)), phase)
    rows = L * (1 + 1 + 1 + 1 + 2)
    assert run.src.calls == rows + len(STEPS) * (L + len(CSA)), "a mixed step's layers call both FlashMLA paths"
    assert run.src.stats["steps_by_routes"] == {"decode decode-tokens+hc-fused+indexer-short": 1,
                                                "decode decode-tokens+hc-fused+indexer-scored": 1,
                                                "prefill hc-split+indexer-scored+prefill-tokens": 1,
                                                "prefill hc-fused+indexer-short+prefill-tokens": 1,
                                                "prefill decode-tokens+hc-split+indexer-scored+prefill-tokens": 1}


def test_capture_map_ids_and_the_committer_agree_on_every_route(run, capsys):
    steps = {k: map_names(capsys, routes, phase) for k, (_md, routes, phase) in enumerate(STEPS)}
    for k, (_md, routes, phase) in enumerate(STEPS):
        assert sorted(steps[k]) == sorted(run.src.names_at(tuple(sorted(routes)), phase))
    wq_b = {f"model.layers.{i}.attn.indexer.wq_b/0" for i in sorted(CSA)}
    assert not wq_b & set(steps[0]) and wq_b <= set(steps[1]) and wq_b <= set(steps[2]), "the short path runs no indexer wq_b"
    got = run.check(steps)
    assert got["ok"] is True, got["violations"][:2]
    without = run.check(steps, sources=[])
    assert without["ok"] is False
    assert any(v["step"] == 1 and v["multiplicity_vs_step0"] for v in without["violations"]), "a scored step differs from step 0"


def test_a_step_off_its_routes_fails(run, capsys):
    steps = {k: map_names(capsys, routes, phase) for k, (_md, routes, phase) in enumerate(STEPS)}
    extra = "model.layers.2.attn.indexer.wq_b/0"
    bad = {**steps, 0: steps[0] + [extra]}
    got = run.check(bad)
    assert got["ok"] is False and [v["step"] for v in got["violations"]] == [0]
    assert got["violations"][0]["routes_bad"] == {extra: (1, 0)}


def test_a_layer_without_attention_rows_or_split_indexers_fail(run, capsys):
    run.step(9, STEPS[0][0], skip=(3,))
    names = map_names(capsys, STEPS[0][1], "decode")
    got = run.check({0: map_names(capsys, STEPS[0][1], "decode"), 9: names})
    assert got["ok"] is False and got["violations"][0]["routes_bad"] == {f"<1 of {L} layers recorded no route (rows)>": (len(names), 0)}
    run.src.routes.pop(9)
    run.com._cur = types.SimpleNamespace(step=9)
    for i in range(L):
        run.model.model.layers[i].attn.md = STEPS[0][0]
    for i in sorted(CSA):
        run.model.model.layers[i].attn.indexer.md = metadata(max_seq_len=STEPS[1][0].max_seq_len if i == 4 else 64)
    run.model.model(torch.arange(4), torch.arange(4))
    run.com._cur = None
    assert run.src.step_routes(9) == (None, "<layers took different routes (indexer)>")
    run.src.routes.pop(9)


def test_a_tp_rank_attaches_the_route_source_once_no_binder_is_unbuilt(monkeypatch):
    """While the map binds taps through a binder no source builds, a rank leaves the map alone: nothing would commit those taps, and
    the route source would release the modules the map knows from the module rule."""
    class Com:
        def __init__(self) -> None:
            self.sources: list[Any] = []

        def attach_source(self, s: Any) -> None:
            self.sources.append(s)

    attach = TAPS.for_ranks(TAPS.TapSources())
    missing = RS.unbuilt(CMAP)
    assert missing and missing <= {"mhc", "dsv4_attention"}
    made = [{"rank": r, "tap_sources": attach(Com(), dsv4_model(), None, r)} for r in (0, 1)]
    assert all("route_source" not in m["tap_sources"] for m in made), made
    monkeypatch.setattr(RS, "OWN_RULES", (*RS.OWN_RULES, *missing))
    made = [{"rank": r, "tap_sources": attach(Com(), dsv4_model(), None, r)} for r in (0, 1)]
    assert all(m["tap_sources"]["route_source"]["stats"]["layers"] == L for m in made), made
    assert TAPS.require_on_every_rank(made, ("a",), {}, TAPS.TapSources(), explicit_classes=False) == ("a",)
