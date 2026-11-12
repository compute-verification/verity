"""The T1 check's recomputation path against the function it wraps, on synthetic inputs (laptop-safe, no fixture).

`replay_partition.recompute` re-derives the sampled-replay PARTITION from a `sampled_replay` record's stated parameters + the Programs; over
the synthetic worlds of `tests.check.test_sampled_replay` it must lift to exactly what `sampled_replay` itself recorded
(population, strata, sample, picks per (request, step), lifetime, phase anchors) -- and a changed seed / a changed Program must not.
"""
from __future__ import annotations

import copy
import gzip
import json

import pytest

from tests.check import test_sampled_replay as T
from tests.check.test_sampled_replay import moe_world, world  # noqa: F401  -- fixtures
from tests.regression import resolver
from tests.regression.checks import replay_partition as RP
from verity_vllm.check.replay import driver as SR


def _programs(w):
    return {rid: w.inst for rid in w.prompts}


def _weights(w):
    def read(field, lo, hi):
        flo, _fhi = w.woff[field]
        return w.weights[flo + lo: flo + hi]
    return read


def _manifest_for(w) -> dict:
    return T.v2_manifest(w.inst, tuple(sorted(w.prompts)))


def _record(w, **kw) -> tuple[dict, dict]:
    man = _manifest_for(w)
    bmap, committed = w.bmap_and_reader()
    rep = SR.sampled_replay(_programs(w), bmap, committed, _weights(w), workers=0, manifest=man, **kw)
    return rep, man


@pytest.mark.parametrize("seed,per_stratum,ring_slots", [(3, 2, None), (11, 1, 2)])
def test_the_partition_recomputed_from_the_record_lifts_to_the_record(world, seed, per_stratum, ring_slots):
    rep, man = _record(world, per_stratum=per_stratum, seed=seed, ring_slots=ring_slots)
    assert rep["population"]["vus"] > 0 and rep["population"]["strata_n"] > 1
    doc = RP.recompute(rep, _programs(world))
    exp, act = RP.lift(rep), RP.lift(doc)
    assert resolver.diff(exp, act) == []
    # the sample is the seed's: another seed moves the per-(request, step) picks (the strata sizes stay)
    other = RP.recompute(dict(rep, sample=dict(rep["sample"], seed=seed + 1)), _programs(world))
    d = resolver.diff(exp, RP.lift(other))
    assert d and all(".sample." in x or ".lifetime.sampled" in x for x in d), d
    assert all(".strata." not in x or x.endswith("sampled") or "population" not in x for x in d)


def test_the_partition_over_the_moe_world_lifts_to_the_record(moe_world):
    rep, man = _record(moe_world, per_stratum=2, seed=5)
    assert rep["population"]["strata_n"] > 1 and rep["population"]["by_family"]
    assert resolver.diff(RP.lift(rep), RP.lift(RP.recompute(rep, _programs(moe_world)))) == []


def test_a_changed_program_row_moves_a_count_by_name(world):
    rep, man = _record(world, per_stratum=2, seed=3)
    progs = {rid: copy.deepcopy(world.inst) for rid in world.prompts}
    inst = next(iter(progs.values()))
    inst["rows"] = inst["rows"][:-1]                                        # one Program row fewer
    for v in progs.values():
        v["rows"] = inst["rows"]
    d = resolver.diff(RP.lift(rep), RP.lift(RP.recompute(rep, progs)))
    assert any(x.startswith("$.population.vus") for x in d), d


def test_the_self_check_recomputes_the_partition_from_the_rows_own_programs(world, tmp_path, monkeypatch):
    """`actual` with no candidate: the record's `programs` name each request's Build dir, read by its basename under the row's Programs."""
    rep, _man = _record(world, per_stratum=2, seed=3)
    row_dir = tmp_path / "rows" / "stand-in-row"
    (row_dir / "commit").mkdir(parents=True)
    (row_dir / "build_request").mkdir()
    record = row_dir / "commit" / "sampled_replay_p0.json"
    record.write_text(json.dumps(dict(rep, programs={rid: {"dir": "/pod/row/build_request"} for rid in world.prompts})))
    with gzip.open(row_dir / "build_request" / "instances.json.gz", "wt") as f:
        json.dump(world.inst, f)
    (row_dir / "build_request" / "result.json").write_text("{}")
    monkeypatch.setenv(resolver.ROWS_ROOT_ENV, str(tmp_path / "rows"))
    monkeypatch.delenv(resolver.CANDIDATE_ENV, raising=False)
    row = resolver.Row("stand-in-row", {"row": 900}, fixtures={})
    exp, act = RP.expected(row), RP.actual(row)
    assert act["source"].startswith("recomputed") and act["population"]["vus"] > 0 and resolver.diff(exp, act) == []
    record.write_text(json.dumps(dict(rep, programs={rid: {"dir": "/pod/row/build_request_LP9_T9"} for rid in world.prompts})))
    with pytest.raises(resolver.NotResolvable, match="build_request_LP9_T9/instances.json.gz not under"):
        RP.actual(row)
