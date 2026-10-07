"""The router-tap exactness property without a GPU: it is registered; a sealed record verifies and every clause fails it; the IR side it
compares against is the ordered router's committed gates, whose outputs are the kernel-order router's (the record's) on every edge row, and
whose interior words are the kernel's where fast-math arithmetic would differ: the row max is fmaxf (NaN operands skipped, subnormals kept)
and a NaN exponential or reciprocal is 0x7FFFFFFF."""
from __future__ import annotations

import copy
import json
import random

import pytest

from tests.properties.router_tap_exactness_gpu import edge_rows, rows_of
from verity_vllm.properties import record as REC
from verity_vllm.properties import router_tap_exactness as X
from verity_vllm.query import router_softmax as RS


def _config(E=64, K=8, renorm=False, **kw):
    e = {"name": f"E{E}", "E": E, "TOPK": K, "VPT": 8, "renormalize": renorm, "rows": 10, "ir_rows": 10, "words_per_row": RS.words(E, K, renorm),
         "out_eq_installed": True, "out_eq_untapped": True, "unwritten": 0, "tap_ne_ir": 0, "out_ne_ir": 0, **kw}
    e["ok"] = X.config_ok(e)
    return e


def _doc(**kw):
    doc = {"schema": X.SCHEMA, "quick": False, "configs": [_config(E, K, r) for E, K, _V, r in X.CONFIGS],
           "source": {"outputs_equal": True, "outside_step_acquired": 0, "names_ok": True, "words_equal": True, "refused_other_geometry": True,
                      "restored": True, "ok": True},
           "live": {"models": [{"experts": E, "topk": K, "renormalize": r, "rows_checked": 100, "rows_ne_ir": 0, "tokens_equal_bare": True,
                                "every_layer_every_step": True, "errors": 0} for E, K, r in X.LIVE_SHAPES]}}
    doc["live"]["ok"] = X.live_ok(doc["live"])
    doc.update(kw)
    return X.seal(doc)


def test_registered():
    assert REC.REGISTRY["router_tap_exactness"].schema == X.SCHEMA


def test_a_sealed_record_verifies():
    doc = _doc()
    assert doc["ok"] and X.verify_record(doc) == []


@pytest.mark.parametrize("edit", [("configs", 0, "tap_ne_ir", 1), ("configs", 1, "unwritten", 3), ("configs", 2, "out_eq_installed", False),
                                  ("configs", 3, "out_eq_untapped", False), ("configs", 0, "ir_rows", 9), ("configs", 1, "words_per_row", 130),
                                  ("source", None, "restored", False), ("live", 0, "rows_ne_ir", 2), ("live", 1, "tokens_equal_bare", False),
                                  ("live", 1, "renormalize", False)])
def test_every_clause_fails_the_record(edit):
    where, i, key, value = edit
    doc = copy.deepcopy(_doc())
    ent = (doc["live"]["models"][i] if where == "live" else doc[where][i]) if i is not None else doc[where]
    ent[key] = value
    if where == "live":
        ent, where = doc["live"], "live"
    ent["ok"] = X.config_ok(ent) if where == "configs" else (X.source_ok(ent) if where == "source" else X.live_ok(ent))
    resealed = X.seal({k: v for k, v in doc.items() if k not in ("ok", "digest")})
    assert not resealed["ok"] and X.verify_record(resealed) == []
    doc["ok"] = True
    assert X.verify_record(doc), "a record claiming ok over a failing clause is named"


def test_a_missing_configuration_or_a_quick_run_without_live():
    doc = _doc(configs=[_config(64, 8, False)])
    assert not doc["ok"]
    assert _doc(quick=True, live={})["ok"] and not _doc(live={})["ok"]


def test_the_verify_cli(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("rte_gpu", Path(__file__).with_name("router_tap_exactness_gpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = tmp_path / X.RECORD
    p.write_text(json.dumps(_doc()))
    assert mod.main(["--verify", str(p)]) == 0
    bad = _doc()
    bad["configs"][0]["tap_ne_ir"] = 1
    p.write_text(json.dumps(bad))
    assert mod.main(["--verify", str(p)]) == 1


@pytest.mark.parametrize("E,renorm", [(64, False), (128, True)])
def test_the_irs_outputs_are_the_record_routers_on_every_edge_row(E, renorm):
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import moe

    kernel_order = bind(moe.MoeRouterTopKNorm if renorm else moe.MoeRouterTopK, E=E, TOPK=8, VPT=8)
    for name, row in edge_rows(E, random.Random(5)).items():
        words, w, ids = X.ir_row(E, 8, 8, renorm, row)
        assert len(words) == RS.words(E, 8, renorm)
        assert [int(v) & 0xFFFFFFFF for v in evaluate(kernel_order, row)] == w + ids, name


def test_the_interior_words_are_the_kernels_at_nan_and_subnormal_rows():
    rows = edge_rows(64, random.Random(5))
    o = RS.offsets(64, 8, False)
    words, _w, ids = X.ir_row(64, 8, 8, False, rows["NaN first"])
    assert words[o["max"]] & 0x7F800000 != 0x7F800000, "fmaxf skips the NaN at the head of thread 0's columns"
    assert words[o["ex"]] == 0x7FFFFFFF and words[o["rcp"]] == 0x7FFFFFFF and ids == list(range(8))
    words, _w, _ids = X.ir_row(64, 8, 8, False, rows["subnormal logits"])
    assert 0 < words[o["max"]] < 0x00800000, "fmaxf keeps a subnormal max"
    words, _w, _ids = X.ir_row(64, 8, 8, False, rows["all NaN"])
    assert words[o["max"]] == 0x7FFFFFFF and all(v == 0x7FFFFFFF for v in words[o["ex"]:o["ex"] + 64]) and words[o["rcp"]] == 0x7FFFFFFF
    words, _w, _ids = X.ir_row(64, 8, 8, False, rows["mixed +-0"])
    assert words[o["max"]] == 0, "max(+0, -0) is +0"


def test_the_rows_cover_the_edges_and_are_seeded():
    a, b = rows_of(64, 1, 8), rows_of(64, 1, 8)
    assert a == b and len(a) == len(edge_rows(64, random.Random(0))) + 8
    specials = {0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000}
    named = ("all +0", "all -0", "mixed +-0", "all -inf", "all NaN", "all +inf", "one +inf", "two +inf", "one NaN", "NaN first", "NaN heads thread 1")
    assert all(any(v in specials for v in a[n]) for n in named)
    assert X.classify(128, 8, True, 0) == "max" and X.classify(128, 8, True, 1) == "ex" and X.classify(128, 8, True, 129) == "rcp"
    assert X.classify(128, 8, True, 130) == "p" and X.classify(128, 8, True, 258) == "sel" and X.classify(128, 8, True, 266) == "scale"
