"""The one-stage replay of a warm request whose Program recomputes its prefix-cache hit (`check.replay.recomputed_prefix`), on
the sampled replay's CPU-served world (`tests.check.test_sampled_replay`): r1 served with its first position from the cache
(C = 1), so serving committed its first step's per-position members from position 1 on and embedded prompt[1:]. Its prefix
rows leave its population, every sampled VU replays bit for bit, a read below C is its replay's and never a committed word
(with attention, a later position's and every decode's key and value reads of position 0), and without the recompute the
same record doesn't pass. A hit its Program takes (a prefix layout beside the Program) is recomputed by nothing: the
population binds the request to its producers. A plane a row's output is committed under a second time is a position's
too, the replay's to read, and a second address of its row in the population."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from verity_vllm.check.replay import cached_prefix as CP
from verity_vllm.check.replay import driver as SR
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import population as POP
from verity_vllm.check.replay import recomputed_prefix as RP
from verity_vllm.check.replay.opening import CommittedStore
from verity_vllm.check.replay.request_replay import Replay

C = 1
#: the weights field only the world with attention registers
ROTARY = "model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"


@pytest.fixture(scope="module", params=[False, True], ids=["mlp", "attention"])
def world(request):
    import tests.check.test_sampled_replay as T

    rng = np.random.default_rng(23)
    inst, woff = T._program(attn=request.param)
    weights = T._bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80
    if request.param:
        lo, hi = woff[ROTARY]
        weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    assert (ROTARY in woff) == request.param
    return T.Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


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


def test_a_replay_given_no_weights_fails_at_the_first_it_reads_and_a_plan_runs_none_whatever_it_is_given(world):
    pi = IX.ProgramIndex(world.inst)
    bmap, committed = _warm(world)
    store = CommittedStore(bmap, committed)
    op, w = "model.layers.0.mlp.down_proj", RP.positions(pi)[("model.layers.0.mlp.down_proj", "out")]
    unweighted = RP.RecomputedPrefix(store, pi, "r1", C, world.prompts["r1"], None)
    assert unweighted.replay is not None and unweighted.replay.weights is None
    got, _ = unweighted.words("r1", 0, op, "out", w, 2 * w)                       # past C: serving's, no weight read
    assert np.array_equal(got, world.blocks[(0, "r1", op, "out")][w:])
    ids, _ = unweighted.words("r1", 0, pi.embed_op_path(), "in", 0, 1)           # below C, of the prompt: no weight read either
    assert list(ids) == world.prompts["r1"][:C]
    with pytest.raises(LookupError, match=r"of request r1: .*'NoneType' object is not callable"):
        unweighted.words("r1", 0, op, "out", 0, w)
    for weights in (None, _weights(world)):
        plan = RP.RecomputedPrefix(store, pi, "r1", C, world.prompts["r1"], weights, plan=True)
        got, _ = plan.words("r1", 0, op, "out", 0, w)
        assert plan.replay is None and got is not None and len(got) == w and not got.any()
        assert plan.take() == [["r1", 0, op, "out", [0, w]]]


def test_the_warm_request_replays_bit_for_bit_with_its_prefix_out_of_its_population(world, monkeypatch):
    bmap, committed = _warm(world)
    progs = {rid: world.inst for rid in world.prompts}
    warm = {"r1": {"cached": C, "producers": ["r0"], "prompt": world.prompts["r1"]}}
    answered, take = [], RP.RecomputedPrefix.take

    def taken(self):
        got = take(self)
        answered.extend(got)
        return got

    monkeypatch.setattr(RP.RecomputedPrefix, "take", taken)
    rep = SR.sampled_replay(progs, bmap, committed, _weights(world), per_stratum=8, seed=1, workers=0, cached_prefixes=warm)
    assert rep["result"] is True and rep["sample"]["mismatch_n"] == 0, rep.get("why")
    per = RP.positions(IX.ProgramIndex(world.inst))
    assert all(rid == "r1" and step == 0 and 0 <= lo < hi <= C * per[(op, m)] for rid, step, op, m, (lo, hi) in answered)
    assert ROTARY not in world.woff or any("rotary_emb" in op and m == "key" for _r, _s, op, m, _ in answered), answered[:4]
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


def test_a_hit_its_program_takes_is_bound_to_its_producers_and_recomputed_by_nothing(tmp_path):
    """r1's Program has a prefix layout beside it (`prefix_cache.json`, a build with `--cached 4`), r0's has none."""
    import tests.check.test_sampled_replay as T

    cold, warm = T._program()[0], T._program()[0]
    layout = {"input": 1, "C": 4, "row_leaves": 4, "layers": ["model.layers.0.self_attn.attn"]}
    (tmp_path / CP.PREFIX_FILE).write_text(json.dumps(layout))
    arrivals = {"r0": 0, "r1": 2}

    def indices(cached: int, producers: list[str]) -> dict:
        return SR.populate({"r0": cold, "r1": warm}, [0], arrivals, None, None, attention_paths=None, tp=None,
                           program_dirs={"r1": str(tmp_path)}, manifest=None,
                           cached_prefixes={"r1": {"cached": cached, "producers": producers, "prompt": [5, 3]}})[3]

    got = indices(4, ["r0", "r0"])
    taken = got["r1"].cached_prefix["r1"]
    assert got["r1"].kv_prefix == layout and got["r1"].recomputed == {} and got["r0"].cached_prefix == {}
    assert (taken.C, taken.rl, taken.layers, taken.producers, taken.block) == (4, 4, [0], ["r0", "r0"], 2)
    assert taken.indices is got and taken.arrivals is arrivals
    assert indices(0, ["r0"])["r1"].cached_prefix == {}                           # served no cached row: no hit to bind
    with pytest.raises(ValueError, match="^r1 is served 5 cached prompt rows but its Program of record takes a cached prefix of 4$"):
        indices(5, ["r0"])


class _Described(IX.ProgramIndex):
    """An index over rows described here, the first of which reads a prompt id."""

    def __init__(self, described: list[dict]) -> None:
        self.rows = [{"i": i, "spec": "X_v1{}", "name": f"x{i}", "args": [[["p", 0, i, i + 1]]] if i == 0 else []} for i in range(len(described))]
        self.by_i, self._described = {r["i"]: r for r in self.rows}, described
        self.pnames, self.kv_prefix, self.segs = ["prompt"], None, [None]
        self.supplied: set[int] = set()

    def describe(self, r: dict) -> dict | None:
        return self._described[int(r["i"])]


def test_a_plane_a_row_s_output_is_committed_under_a_second_time_is_a_position_s_the_replay_s_to_read_and_an_address_of_the_row():
    """Two positions of a first step: at each, an embedding row and a row whose four words are also a plane's."""
    def described(row: int) -> list[dict]:
        plane = {"op_path": "moe/L0", "member": "sum", "width": 4, "dtype": "f32", "row": row}
        return [{"step": 0, "row": row, "op_path": "embed", "members": [("out", 8, "bf16")]},
                {"step": 0, "row": row, "op_path": "moe", "members": [("out", 4, "bf16")], "copies": [plane]}]

    pi = _Described(described(0) + described(1))
    assert RP.positions(pi) == {("embed", "out"): 8, ("moe", "out"): 4, ("moe/L0", "sum"): 4, ("embed", IX.IDS_MEMBER): 1}
    replay = Replay(pi, "r0", [1, 2], None)
    assert replay.words("r0", 0, "moe/L0", "sum", 8, 12) == (None, "no row 2 of (moe/L0/sum) at request r0's step 0")
    assert replay.words("r0", 0, "moe/L1", "sum", 0, 4) == (None, "no row of request r0 commits (moe/L1/sum) at its step 0, and nothing gives it")
    addresses: dict[tuple, dict] = {}
    POP.population(pi, "r0", addresses=addresses)                                 # each row at both its addresses, the plane's under the plane's family
    assert {k: (a["manifest_family"], a["rows"]) for k, a in addresses.items()} == {
        ("r0", 0, "embed", "out"): ("instance_outputs", 2), ("r0", 0, "moe", "out"): ("instance_outputs", 2),
        ("r0", 0, "moe/L0", "sum"): ("moe_block_stream", 2)}
    twice = _Described(described(0) + described(0)[1:])
    with pytest.raises(ValueError, match=r"rows 1 and 2 both commit \(moe/out\) row 0 at step 0"):
        Replay(twice, "r0", [1], None)
