"""The VU export's two budgets: the population build over many request Programs, and the draw after it."""
from __future__ import annotations

import json

import numpy as np

from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS
from tests.check.test_sampled_replay import Honest, _bf16, _program, _weights
from tests.pipeline.test_vu_export import ROOT


def test_the_draw_budget_starts_after_a_slow_population_build(tmp_path, monkeypatch):
    """Many request Programs whose index build is slow (a B32 row's 33 Programs took > 18 min): the draw still runs its whole budget,
    and the build's own budget names the Programs it did not reach."""
    import copy
    import time as _time
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
    slow = VS.ProgramIndex

    def slow_index(*a, **kw):
        _time.sleep(0.1)
        return slow(*a, **kw)

    monkeypatch.setattr(VS, "ProgramIndex", slow_index)
    bmap, committed = world.bmap_and_reader(order=tuple(prompts))
    run = lambda out, **lim: VS.export_vus(programs, bmap, committed, _weights(world), out_dir=tmp_path / out, run_root=ROOT,  # noqa: E731
                                           limits=VX.Limits(per_family=8, workers=1, **lim))
    s = run("a", max_seconds=0.5)                                        # the build alone (12 x 0.1 s) is longer than the draw's budget
    assert s["timing"]["population_s"] > 0.5 and s["population"]["programs_built"] == 12
    assert sum(c.get("drawn", 0) for c in s["by_family"].values()) > 0 and s["sets"]
    b = run("b", max_population_seconds=0.25)
    assert b["population"]["programs_built"] < 12 and b["population"]["programs_past_budget"] == list(prompts)[b["population"]["programs_built"]:]
    assert {v["request"] for v in map(json.loads, (tmp_path / "b" / "vus.jsonl").read_text().splitlines())} <= set(list(prompts)[:b["population"]["programs_built"]])
