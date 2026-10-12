"""`points` splits a layer's y into base, decode, tag and commit so that they add up to y_total, per tag kind."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import points as P  # noqa: E402


def test_split_adds_up():
    lib = {"constants": {"n": 768, "PB": 96, "S": 96},
           "layer": {"1": {"plain": 1.0, "fused_k1": 1.16, "fused_k1_tagged": 2.5, "fused_k1_sketch16": 1.2}},
           "commit_step": {"y_commit": {"1": 0.03}}, "sketch_commit_step": {"y_commit": {"1": 0.02}}}
    gemm = {"libs": {"p768": lib}, "model": {"name": "70b"}, "device": {"name": "gpu"}}
    rates = {"libs": {"p768": {"analysis": {"k": {"1": {"x_cuda": 402.0, "delta_cuda": 0.62}}}}}}
    pts = P.points(gemm, "r1", "art:x", rates, 2092.0)
    assert [p["arithmetic"].split("role-split/")[1] for p in pts] == ["ro-blake3/v1", "ro-sketch-blake3/v0 k_s=16"]
    for p in pts:
        assert p["y_base"] + p["y_decode"] + p["y_tag"] + p["y_commit"] == pytest.approx(p["y_total"])
        assert p["schema"] == P.SCHEMA and p["model"] == "llama-3.1-70b" and p["payload_per_stored_bit"] == 1.0
    assert pts[1]["y_total"] == pytest.approx(1.22) and pts[1]["tau"] == 0.0


def test_rates_rows_take_the_last_run_and_the_best_config():
    def lib(n, sq):
        return {"constants": {"n": n}, "analysis": {"read_gbps": 1300.0, "words_per_plain_byte": 0.01,
                                                     "k": {"1": {"x_cuda": 1.0}}},
                "rates": {"square_cuda": {"a": {"per_s": sq}, "b": {"per_s": 2 * sq}}, "square_tc": {},
                          "decode": {"1": {"per_s": 5.0}}, "encode": {"1": {"per_s": 0.5}}}}
    r1 = {"device": {"name": "gpu"}, "libs": {"p768": lib(768, 1.0), "p512": {"constants": {"n": 512}}}}
    r2 = {"device": {"name": "gpu"}, "libs": {"p768": lib(768, 3.0)}}
    rows = P.rates_rows(P.merge_rates([("r1", r1), ("r2", r2)]), 2092.0)
    assert [(r["field"], r["run"], r["squarings_s_cuda"], r["squarings_s_tc"]) for r in rows] == [("p768", "r2", 6.0, None)]
    assert rows[0]["schema"] == P.RATES_SCHEMA and rows[0]["encode_words_s"] == {"1": 0.5}
