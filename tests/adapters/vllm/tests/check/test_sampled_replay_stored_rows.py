"""A rank Program loaded with its argument progressions as stored (`format.load_program(expand=False)`, `rank_replay.load`) replays
exactly as the per-leaf-run Program does: the same rows once read, the same index, population and record; and `rank_replay.evaluate`'s
seeded index (`index_memo`) decides the record the replay's own index decides."""
from __future__ import annotations

import copy
import gzip
import json

import pytest

from tests.check import test_sampled_replay as T
from tests.check.test_sampled_replay import (  # noqa: F401  -- fixtures
    _under_module_body,
    attn_world,
    biased_world,
    moe_world,
    rope_world,
    world,
)
from verity_vllm.check.replay import driver as SR
from verity_vllm.check.replay import index as IX
from verity_vllm.check.replay import population as POP
from verity_vllm.program.compact import file_items_from_runs
from verity_vllm.query.manifest import format as RM

WORLDS = ("world", "biased_world", "rope_world", "attn_world", "moe_world")


def _stored(inst: dict) -> dict:
    out = copy.deepcopy(inst)
    for r in out["rows"]:
        if r.get("args"):
            r["args"] = [file_items_from_runs(tuple(t) for t in a) for a in r["args"]]
    return out


def _write(tmp_path, inst: dict) -> str:
    d = tmp_path / "prog"
    d.mkdir()
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(_stored(inst), f)
    (d / "result.json").write_text(json.dumps({"program": {"digest": inst.get("digest")}}))
    return str(d)


def _strip(rep: dict) -> dict:
    return {k: v for k, v in rep.items() if k not in ("timing", "evaluator", "wall_s")}


def _n_progressions(inst: dict) -> int:
    return sum(1 for r in inst["rows"] for a in r.get("args") or () for it in a if len(it) == 8)


@pytest.mark.parametrize("name", WORLDS)
def test_stored_rows_replay_as_the_expanded_rows(request, tmp_path, name):
    w = request.getfixturevalue(name)
    d = _write(tmp_path, w.inst)
    stored, expanded = RM.load_program(d, expand=False)["instances"], RM.load_program(d)["instances"]
    assert expanded["rows"] == json.loads(json.dumps(w.inst["rows"]))
    assert [IX.row_args(r) for r in stored["rows"]] == [r.get("args") or [] for r in expanded["rows"]]
    if name in ("attn_world", "moe_world"):
        assert _n_progressions(stored) > 0
    pis, pie = IX.ProgramIndex(stored), IX.ProgramIndex(expanded)
    assert [pis.describe(r) for r in stored["rows"]] == [pie.describe(r) for r in expanded["rows"]]
    assert pis.embed_op_path() == pie.embed_op_path()
    for rid in w.prompts:
        assert POP.population(pis, rid) == POP.population(pie, rid)
    bmap, committed = w.bmap_and_reader()
    reps = [SR.sampled_replay({rid: inst for rid in w.prompts}, bmap, committed, T._weights(w), per_stratum=4, seed=5, workers=0)
            for inst in (stored, expanded)]
    assert reps[1]["result"] is True and reps[1]["sample"]["evaluated"] > 0
    assert _strip(reps[0]) == _strip(reps[1])


def test_a_stored_moe_program_that_is_not_pair_major_is_named_alike(moe_world):
    rows = [dict(r) for r in moe_world.inst["rows"]]
    ups = [k for k, r in enumerate(rows) if r["spec"].startswith("MoeExpertGemm_v1")][:2]
    rows[ups[0]]["args"], rows[ups[1]]["args"] = rows[ups[1]]["args"], rows[ups[0]]["args"]
    sm = next(k for k, r in enumerate(rows) if r["spec"].startswith("MoeSum"))
    rows[sm]["args"] = [list(reversed(rows[sm]["args"][0]))]
    bad = dict(moe_world.inst, rows=rows)
    pops = [POP.population(IX.ProgramIndex(inst), "r0") for inst in (_stored(bad), bad)]
    assert pops[0] == pops[1]
    assert any("not pair-major" in k for k in pops[0][1])


