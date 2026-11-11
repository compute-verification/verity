"""The one-stage replay of a warm request whose Program recomputes its prefix-cache hit (`check.replay.recomputed_prefix`), on
the sampled replay's CPU-served world (`tests.check.test_sampled_replay`): r1 served with its first position from the cache
(C = 1), so serving committed its first step's per-position members from position 1 on and embedded prompt[1:]. Its prefix
rows leave its population, every sampled VU replays bit for bit, a read below C is its replay's and never a committed word
(with attention, a later position's and every decode's key and value reads of position 0), and without the recompute the
same record doesn't pass."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from verity_vllm.check.replay import driver as SR, index as IX, recomputed_prefix as RP
from verity_vllm.check.replay.opening import CommittedStore

C = 1


@pytest.fixture(scope="module", params=[False, True], ids=["mlp", "attention"])
def world(request):
    import tests.check.test_sampled_replay as T

    rng = np.random.default_rng(23)
    inst, woff = T._program(attn=request.param)
    weights = T._bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80
    if request.param:
        lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
        weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    out = T.Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})
    out.attention = request.param
    return out


def _weights(world):
    def read(field, lo, hi):
        flo, _fhi = world.woff[field]
        return world.weights[flo + lo:flo + hi]
    return read


def _warm(world):
    """The record of r1 served warm: its first step's per-position members without their first C rows."""
    per = RP.positions(IX.ProgramIndex(world.inst))
    warm = copy.copy(world)
    warm.blocks = {k: (blk[C * per[(k[2], k[3])]:] if k[1] == "r1" and k[0] == 0 and (k[2], k[3]) in per else blk)
                   for k, blk in world.blocks.items()}
    return warm.bmap_and_reader()


def test_the_prefix_rows_are_the_first_step_s_per_position_rows_below_c(world):
    pi = IX.ProgramIndex(world.inst)
    per = RP.positions(pi)
    assert (pi.embed_op_path(), "in") in per and ("runner.sampler", "sampled_token_ids") not in per
    rows = [pi.describe(pi.by_i[i]) for i in RP.prefix_rows(pi, C)]
    assert rows and {(d["step"], d["row"]) for d in rows} == {(0, 0)}
    assert {d["op_path"] for d in rows} == {op for op, _m in per if op != "logits_processor"}
    with pytest.raises(ValueError, match="recomputed in steps of its own"):
        RP.prefix_rows(pi, 2)


def test_a_read_below_c_is_the_replay_s_and_one_past_it_is_serving_s_c_rows_earlier(world):
    pi = IX.ProgramIndex(world.inst)
    bmap, committed = _warm(world)
    store = RP.RecomputedPrefix(CommittedStore(bmap, committed), pi, "r1", C, world.prompts["r1"], _weights(world))
    op, w = "model.layers.0.mlp.down_proj", RP.positions(pi)[("model.layers.0.mlp.down_proj", "out")]
    honest = world.blocks[(0, "r1", op, "out")]
    got, _ = store.words("r1", 0, op, "out", 0, 2 * w)
    assert np.array_equal(got, honest) and store.take() == [["r1", 0, op, "out", [0, w]]] and store.take() == []
    got, _ = store.words("r1", 0, op, "out", w, 2 * w)
    assert np.array_equal(got, honest[w:]) and store.take() == []
    ids, _ = store.words("r1", 0, pi.embed_op_path(), "in", 0, 2)
    assert list(ids) == world.prompts["r1"] and store.take() == [["r1", 0, pi.embed_op_path(), "in", [0, 1]]]
    other, _ = store.words("r0", 0, op, "out", 0, w)
    assert np.array_equal(other, world.blocks[(0, "r0", op, "out")][:w]) and store.take() == []
    with pytest.raises(ValueError, match="at least one, and short of it"):
        RP.RecomputedPrefix(CommittedStore(bmap, committed), pi, "r1", 2, world.prompts["r1"], _weights(world))


def test_the_warm_request_replays_bit_for_bit_with_its_prefix_out_of_its_population(world, monkeypatch):
    bmap, committed = _warm(world)
    progs = {rid: world.inst for rid in world.prompts}
    warm = {"r1": {"cached": C, "producers": ["r0"], "prompt": world.prompts["r1"]}}
    answered, take = [], RP.RecomputedPrefix.take
    monkeypatch.setattr(RP.RecomputedPrefix, "take", lambda self: answered.extend(got := take(self)) or got)
    rep = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0, cached_prefixes=warm)
    assert rep["result"] is True and rep["sample"]["mismatch_n"] == 0, rep.get("why")
    per = RP.positions(IX.ProgramIndex(world.inst))
    assert all(rid == "r1" and step == 0 and 0 <= lo < hi <= C * per[(op, m)] for rid, step, op, m, (lo, hi) in answered)
    assert not world.attention or any("rotary_emb" in op and m == "key" for _r, _s, op, m, _ in answered), answered[:4]
    pi = IX.ProgramIndex(world.inst)
    prefix = [k for k in rep["population"]["not_evaluable_by_family"] if "(recomputed prefix" in k]
    assert sum(rep["population"]["not_evaluable_by_family"][k] for k in prefix) == len(RP.prefix_rows(pi, C))
    assert all("(recomputed prefix" not in u["family"] or u["why"].startswith("recomputed prefix") for u in rep["unreplayable"])
    plan = SR.sampled_replay(progs, bmap, committed, None, per_stratum=8, seed=1, cached_prefixes=warm, plan=True)
    assert plan["unresolved_n"] == 0
    cold = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0)
    assert cold["result"] is not True
    with pytest.raises(ValueError, match="needs its certified prompt"):
        SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0,
                          cached_prefixes={"r1": {"cached": C, "producers": ["r0"]}})
