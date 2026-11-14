"""The VU export's two budgets: the population build over many request Programs, and the sample after it."""
from __future__ import annotations

import contextlib
import json
from collections import Counter
from typing import Any

import numpy as np

from tests.check.test_sampled_replay import Honest, _bf16, _program, _weights
from tests.pipeline.test_vu_export import ROOT, _world
from tests.pipeline.test_vu_store_export import EACH, _export
from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS


def test_the_sample_budget_starts_after_a_slow_population_build(tmp_path, monkeypatch, capsys):
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
    assert capsys.readouterr().out.splitlines()[0].endswith(" sampled, 12 s")
    assert sum(c.get("drawn", 0) for c in s["by_family"].values()) > 0 and s["sets"]
    b = run("b", max_population_seconds=2.5)
    assert b["population"]["programs_built"] == 3 and b["population"]["programs_past_budget"] == list(prompts)[3:]
    assert capsys.readouterr().out.splitlines()[0].endswith(" sampled, 3 s; 9 Programs past the population budget")
    assert {v["request"] for v in map(json.loads, (tmp_path / "b" / "vus.jsonl").read_text().splitlines())} <= set(list(prompts)[:b["population"]["programs_built"]])


def test_the_export_reuses_the_replays_population_and_never_keeps_it(tmp_path, monkeypatch, capsys):
    """The Commit's replay has already built every request Program's index and population: the export takes that one (no rebuild
    inside the Commit), and `after_replay` drops it whether or not it exports."""
    from verity_vllm.check.replay import driver as SR
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
    capsys.readouterr()
    s = VS.export_vus(programs, bmap, committed, _weights(world), out_dir=tmp_path / "a", run_root=ROOT,
                      limits=VX.Limits(per_family=8, workers=1), population_of_record=kept)
    assert s["population"]["source"] == "the sampled replay's own population (reused)"
    assert s["population"]["vus"] == rep["population"]["vus"] and s["sets"]
    assert capsys.readouterr().out.splitlines()[0].endswith(" s (the replay's own)")
    SR.sampled_replay(programs, bmap, committed, _weights(world), per_stratum=1, seed=3, workers=0)
    assert VS.after_replay(None, {"result": True}, 0, ROOT, programs, bmap, committed, _weights(world), {}, None, {}, {}, {}, {}, None) is None
    assert SR.take_population() == {}                                     # dropped even when the export is off


def test_a_spent_sample_budget_leaves_the_units_after_it_unsampled_and_counted_by_family(tmp_path, monkeypatch, capsys):
    """The serial sample's check of its budget, before each unit: a unit's replay takes `EACH` on the test's clock and the budget is two
    and a half of them, so three units are sampled and the rest are counted under their families and written nowhere."""
    e = _export(_world(attn=True), tmp_path, monkeypatch, capsys, limits=VX.Limits(per_family=2, workers=1, max_seconds=2.5 * EACH))
    families = [v["family"] for v in e.picked]
    sampled, left = Counter(families[:3]), Counter(families[3:])
    assert len(e.replayed) == 3 and [u["vu"] for u in e.units] == [u["vu"] for u in e.listed] == [0, 1, 2]
    assert set(e.summary["by_family"]) == set(families) and sum(left.values()) == len(families) - 3 > 3
    for family, c in e.summary["by_family"].items():
        assert {k: v for k, v in c.items() if k.endswith("_time_budget")} == ({"not_drawn_time_budget": left[family]} if left[family] else {})
        assert c.get("replayed_equal", 0) == c.get("stored", 0) == sampled[family] and c["seconds"] == round(sampled[family] * EACH, 3)
        assert len(c) == 1 + (4 if sampled[family] else 0) + (1 if left[family] else 0)


class _PoolHere:
    """A pool that runs each unit in this process, in the forked pool's place, and lists what it is asked."""

    def __init__(self, calls: list[Any]) -> None:
        self.calls = calls

    def map(self, f, ks):
        self.calls.append(("map", list(ks)))
        return [f(k) for k in ks]


def test_a_pooled_sample_checks_its_budget_between_chunks_of_four_units_a_worker(tmp_path, monkeypatch, capsys):
    """Two workers take the units eight at a time, so a budget the third unit spends is seen only before the second chunk, where the
    serial sample stops at the third unit.  The weights and the evaluators are warmed once, before the pool is made."""
    from verity_vllm.check.replay import driver as SR
    from verity_vllm.check.replay import fork
    calls: list[Any] = []

    @contextlib.contextmanager
    def pool_here(workers):
        calls.append(("pool", workers))
        yield _PoolHere(calls)

    monkeypatch.setattr(fork, "fork_pool", pool_here)
    monkeypatch.setattr(SR, "_prewarm_weights", lambda _indices, picked, _weights: calls.append(("weights", len(picked))))
    monkeypatch.setattr(SR, "_prewarm_evaluators", lambda picked: calls.append(("evaluators", len(picked))))
    e = _export(_world(attn=True), tmp_path, monkeypatch, capsys, limits=VX.Limits(per_family=2, workers=2, max_seconds=2.5 * EACH))
    assert len(e.picked) == 14 and calls == [("weights", 14), ("evaluators", 14), ("pool", 2), ("map", list(range(8)))]
    assert [u["vu"] for u in e.units] == list(range(8))
    assert sum(v for c in e.summary["by_family"].values() for k, v in c.items() if k.endswith("_time_budget")) == 6
