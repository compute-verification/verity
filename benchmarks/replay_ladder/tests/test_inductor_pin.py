"""The pin and the tiling log on stand-ins for Inductor's benchmarker and autotuner, and the plugin's site directory."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import inductor_pin as P


class Benchmarker:
    def benchmark(self, fn, *args, **kwargs):
        return 3.0

    def benchmark_gpu(self, fn, **kwargs):
        return {"wide": 2.0, "narrow": 1.5}[fn]


def test_pinned_timings_keep_every_choice_s_first_candidate():
    b = Benchmarker()
    assert min(["wide", "narrow"], key=b.benchmark_gpu) == "narrow"
    P.pin_timings(b)
    P.pin_timings(b)
    assert min(["wide", "narrow"], key=b.benchmark_gpu) == "wide"
    assert (b.benchmark(None), b.benchmark_gpu("narrow", rep=40)) == (1.0, 1.0)
    assert not hasattr(b, "benchmark_cpu")


def test_inductor_s_benchmarker_has_the_timings_the_pin_replaces():
    pytest.importorskip("torch")
    from torch._inductor.runtime.benchmarking import Benchmarker as Real

    assert all(callable(getattr(Real, name, None)) for name in P.TIMINGS)


class Config:
    def __init__(self, r0: int):
        self.kwargs, self.num_warps, self.num_stages = {"XBLOCK": 2, "R0_BLOCK": r0}, 8, 1


class Autotuner:
    """Inductor's CachingAutotuner as far as the tiling log reads it: a kernel with two candidate tilings keeps one."""

    def __init__(self, keep: int):
        self.launchers, self.keep = [SimpleNamespace(config=Config(1024)), SimpleNamespace(config=Config(512))], keep
        self.inductor_meta, self.fn = {"kernel_name": "triton_red_fused_add_rms_norm_1"}, None
        self.heuristic_type = SimpleNamespace(name="REDUCTION")

    def autotune_to_one_config(self, *args, **kwargs):
        self.launchers = [x for x in self.launchers if x.config.kwargs["R0_BLOCK"] == self.keep]


def test_the_tiling_log_records_each_runtime_choice(tmp_path, monkeypatch):
    class Tuner(Autotuner):
        pass

    monkeypatch.setenv(P.TILINGS, str(tmp_path / "replay-1"))
    P.log_tilings(Tuner)
    P.log_tilings(Tuner)
    Tuner(512).autotune_to_one_config()
    recs = [json.loads(x) for f in (tmp_path / "replay-1").glob("tilings.*.jsonl") for x in f.read_text().splitlines()]
    assert recs == [{"kernel": "triton_red_fused_add_rms_norm_1", "heuristic": "REDUCTION",
                     "kept": {"XBLOCK": 2, "R0_BLOCK": 512, "num_warps": 8, "num_stages": 1},
                     "candidates": [{"XBLOCK": 2, "R0_BLOCK": r, "num_warps": 8, "num_stages": 1} for r in (1024, 512)]}]


def test_the_site_declares_the_module_as_a_vllm_general_plugin(tmp_path):
    assert P.site(tmp_path) == tmp_path
    eps = [ep for dist in importlib.metadata.distributions(path=[str(tmp_path)]) for ep in dist.entry_points]
    assert [(ep.group, ep.name, ep.value) for ep in eps] == \
        [("vllm.general_plugins", "verity_replay_ladder", "inductor_pin:register")]
    assert (tmp_path / "inductor_pin.py").read_bytes() == Path(P.__file__).read_bytes()
