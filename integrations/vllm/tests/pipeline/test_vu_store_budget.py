"""The VU export's two budgets: the population build over many request Programs, and the sample after it."""
from __future__ import annotations

import json

import numpy as np

from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS
from tests.check.test_sampled_replay import Honest, _bf16, _program, _weights
from tests.pipeline.test_vu_export import ROOT


def test_the_draw_budget_starts_after_a_slow_population_build(tmp_path, monkeypatch):
    """Many request Programs whose index build is slow (a B32 row's 33 Programs took > 18 min): the sample still runs its whole budget,
    and the build's own budget names the Programs it did not reach.  The export's clock is the test's: each index build takes 1 s."""
    import copy
    import types
    rng = np.random.default_rng(5)
    inst, woff = _program(attn=True)
    weights = _bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80
    lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
    weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    prompts = {f"r{k}": [1 + k % 5, 2 + k % 3] for k in range(12)}
    world = Honest(inst, woff, weights, prompts)
    programs = {rid: copy.deepcopy(inst) for rid in prompts}             # 12 distinct Program objects: 12 index builds
    now = [0.0]
    monkeypatch.setattr(VS, "time", types.SimpleNamespace(perf_counter=lambda: now[0]))
    slow = VS.ProgramIndex

    def slow_index(*a, **kw):
        now[0] += 1.0
        return slow(*a, **kw)

    monkeypatch.setattr(VS, "ProgramIndex", slow_index)
    bmap, committed = world.bmap_and_reader(order=tuple(prompts))
    run = lambda out, **lim: VS.export_vus(programs, bmap, committed, _weights(world), out_dir=tmp_path / out, run_root=ROOT,  # noqa: E731
                                           limits=VX.Limits(per_family=8, workers=1, **lim))
    s = run("a", max_seconds=5)                                          # the build alone (12 x 1 s) is longer than the sample's budget
    assert s["timing"]["population_s"] == 12 and s["population"]["programs_built"] == 12
    assert sum(c.get("drawn", 0) for c in s["by_family"].values()) > 0 and s["sets"]
    b = run("b", max_population_seconds=2.5)
    assert b["population"]["programs_built"] == 3 and b["population"]["programs_past_budget"] == list(prompts)[3:]
    assert {v["request"] for v in map(json.loads, (tmp_path / "b" / "vus.jsonl").read_text().splitlines())} <= set(list(prompts)[:b["population"]["programs_built"]])


def test_the_export_reuses_the_replays_population_and_never_keeps_it(tmp_path, monkeypatch):
    """The Commit's replay has already built every request Program's index and population: the export takes that one (no rebuild
    inside the Commit), and `after_replay` drops it whether or not it exports."""
    from verity_vllm.check.replay import driver as SR
    from tests.pipeline.test_vu_export import _world
    world = _world(attn=True)
    programs = {rid: world.inst for rid in world.prompts}
    bmap, committed = world.bmap_and_reader()
    SR.keep_population(True)
    rep = SR.sampled_replay(programs, bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    kept = SR.take_population()
    assert set(kept["indices"]) == set(programs) and len(kept["vus"]) == rep["population"]["vus"] and not SR.take_population()

    def no_build(*a, **kw):
        raise AssertionError("the export rebuilt a request Program's index")

    monkeypatch.setattr(VS, "ProgramIndex", no_build)
    s = VS.export_vus(programs, bmap, committed, _weights(world), out_dir=tmp_path / "a", run_root=ROOT,
                      limits=VX.Limits(per_family=8, workers=1), population_of_record=kept)
    assert s["population"]["source"].startswith("the sampled replay") and s["population"]["vus"] == rep["population"]["vus"] and s["sets"]
    SR.sampled_replay(programs, bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    assert VS.after_replay(None, {"result": True}, 0, ROOT, programs, bmap, committed, _weights(world), {}, None, {}, {}, {}, {}, None) is None
    assert SR.take_population() == {}                                     # dropped even when the export is off
