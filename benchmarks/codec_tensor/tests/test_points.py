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
