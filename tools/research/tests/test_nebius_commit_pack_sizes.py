"""Node 1 has one bundle-size estimate: the commit-pack pilot (sky/commit_pack.py) reads the per-model rates the pacer
(sky/release.py) learns from succeeded Commits, from the pacer's mirror under /workspace, which a pack pod mounts, before its own
60/120 GB table. Stdlib only, unlike test_nebius_commit_pack.py, so it runs where PyYAML isn't installed."""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from research.pods import registry

SKY = Path(registry.__file__).resolve().parent / "nebius" / "sky"


def row(model: str) -> dict:
    return {"env": {"ROW": f"{model}__bf16__rtxpro6000__tp1__b8__i1024__o128__mixed__greedy__bi-eager"}}


def test_the_pilot_estimates_a_bundle_as_the_pacer_does_once_the_pacer_has_learned_its_model(monkeypatch):
    sizes = Path(os.environ["VY_BUNDLE_SIZES"])               # conftest points it away from node 1's mirror
    spec = importlib.util.spec_from_file_location("commit_pack_sizes", SKY / "commit_pack.py")
    pilot = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "commit_pack_sizes", pilot)
    spec.loader.exec_module(pilot)
    assert pilot.SIZES == sizes and not sizes.exists()
    assert pilot.bundle_est(row("qwen25-3b")) == 120e9 and pilot.bundle_est(row("qwen3-06b")) == pilot.BUNDLE_EST   # no file yet
    sizes.write_text(json.dumps({"qwen25-3b": 0.005, "qwen3-06b": 0.0001}))
    assert pilot.bundle_est(row("qwen25-3b")) == pytest.approx(0.005 * 8 * 1152 * 1.25 * 1e9)   # 57.6 GB, as release.py has it
    assert pilot.bundle_est(row("qwen3-06b")) == 10e9                                           # the pacer's 10 GB floor
    assert pilot.bundle_est(row("phi3-mini")) == 120e9                                          # a model it hasn't learned
    sizes.write_text("{not json")
    assert pilot.bundle_est(row("qwen25-3b")) == 120e9


def test_the_pacer_mirrors_its_learned_table_where_the_pilot_reads_it(monkeypatch, tmp_path):
    monkeypatch.setenv("VY_PACER_DIR", str(tmp_path / "pacer"))
    spec = importlib.util.spec_from_file_location("release_sizes", SKY / "release.py")
    pacer = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "release_sizes", pacer)
    spec.loader.exec_module(pacer)
    spec = importlib.util.spec_from_file_location("commit_pack_sizes2", SKY / "commit_pack.py")
    pilot = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "commit_pack_sizes2", pilot)
    spec.loader.exec_module(pilot)
    assert pacer.SHARED == pilot.SIZES
    pacer._learned = {"yi15-6b": 0.0076}                     # as learn() leaves it after a succeeded yi15-6b Commit
    pacer.share(pacer.learned())
    assert pilot.learned() == {"yi15-6b": 0.0076} and not pacer.SHARED.with_name(pacer.SHARED.name + ".tmp").exists()
    r = "yi15-6b__bf16__rtxpro6000__tp1__b32__i1024__o128__g"
    assert pilot.bundle_est({"env": {"ROW": r}}) == pytest.approx(pacer.estimate_gb(r) * 1e9)   # the pacer once it has read it
