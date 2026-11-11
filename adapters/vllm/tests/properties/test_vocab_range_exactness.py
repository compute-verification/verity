"""The vocabulary-range exactness property without a GPU: registered, a sealed record verifies and every clause fails it, and the IR side
it compares against reads the shard's committed gates."""
from __future__ import annotations

import copy

import pytest

from verity_vllm.properties import record as REC
from verity_vllm.properties import vocab_range_exactness as X


def _rank(r):
    e = {"rank": r, "edge_rows": 100, "edge_ne_ir": 0, "edge_kernel_disagrees": 0, "edge_out_equal": True, "reference_ne_ir": 0}
    e["ok"] = X.rank_ok(e)
    return e


def _live():
    v = {"tokens_equal_bare": True, "ranks": [{"start": s, "rows": 40, "rows_ne_ir": 0, "kernel_disagrees": 0, "one_per_step": True,
                                                "outside_step_acquired": 0, "restored": True, "errors": 0} for s in (0, 24576)]}
    v["ok"] = X.live_ok(v)
    return v


def _doc():
    return X.seal({"schema": X.SCHEMA, "ranks": [_rank(0), _rank(1)], "live": _live()})


def test_registered_and_sealed():
    assert REC.REGISTRY["vocab_range_exactness"].schema == X.SCHEMA
    doc = _doc()
    assert doc["ok"] and X.verify_record(doc) == []


@pytest.mark.parametrize("path,value", [(("ranks", 0, "edge_ne_ir"), 1), (("ranks", 1, "edge_kernel_disagrees"), 2), (("ranks", 0, "edge_out_equal"), False),
                                        (("live", "ranks", 1, "rows_ne_ir"), 1), (("live", "ranks", 0, "one_per_step"), False),
                                        (("live", "tokens_equal_bare"), False), (("live", "ranks", 1, "start"), 0)])
def test_every_clause_fails_the_record(path, value):
    doc = copy.deepcopy(_doc())
    ent = doc
    for k in path[:-1]:
        ent = ent[k]
    ent[path[-1]] = value
    body = {k: v for k, v in doc.items() if k not in ("ok", "digest")}
    for r in body["ranks"]:
        r["ok"] = X.rank_ok(r)
    body["live"]["ok"] = X.live_ok(body["live"])
    assert not X.seal(body)["ok"]


def test_ir_values_read_the_shards_gates():
    assert X.edge_ids(8, 8, 16) == [0, 1, 7, 8, 9, 14, 15]
    assert [X.ir_values(8, 8, t) for t in (7, 8, 15)] == [(-1, 0, 1), (0, 1, 1), (7, 1, 1)]
    assert X.ir_values(8, 0, 8) == (8, 1, 0)
