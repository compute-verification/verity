"""`served_match.build_here`, the served Build of a serving process's own compile artifact, on a CPU.

The Build (`acquire.served.build`) and the probe of cuBLAS's routes (`probe_routes`, which needs the GPU) are stand-ins
that record what they are asked, and the source is one that holds a launch record and a fused-MoE record.  So these cases
show what `build_here` hands the Build, without routes and then with the probed ones, which files it writes, and what it
refuses before it builds.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.acquire import served as SV
from verity_vllm.acquire.inductor_templates import Unmodelled
from verity_vllm.config import UsageError
from verity_vllm.engine import env as engine_env
from verity_vllm.pipeline import served_match as sm

CONFIGS = {"sha-a": {"config": {"num_warps": 4}}}
SCHEDULE = [{"request_id": "r0", "arrive": 0, "prompt_tokens": 2, "max_tokens": 3}]
MISSING = [[8, 16, 32], [1, 16, 32]]
ROUTES = {"8x16x32": {"route": "k16"}, "1x16x32": {"route": "k8"}}
MOE = {"layers.0.mlp": {"E": 4, "top_k": 2}}


class _Source:
    """The source as far as `build_here` reads it: the launch record and the fused-MoE layers' pieces and configs."""

    moe_pieces = (("layers.0.mlp", "c03"),)
    moe_configs = {"layers.0.mlp": {"BLOCK_SIZE_M": 16}}

    def launch_configs(self) -> dict[str, Any]:
        return CONFIGS


def _facts(tmp_path: Any, **over: Any) -> dict[str, Any]:
    """An engine's facts as far as `build_here` and `device_of` read them, its vLLM cache dir under the test's directory."""
    return {"cache_dir": str(tmp_path / "vllm_cache" / "torch_compile_cache"), "cudagraph_capture_sizes": [8, 4, 2, 1], "max_num_seqs": 8,
            "device": "NVIDIA H200", "cc": [9, 0], "sms": 132, "attention": [{"backend": "FLASH_ATTN", "fa_version": 3}],
            "max_model_len": 4096, "moe": {}, **over}


def _graph(tmp_path: Any, *under: str) -> str:
    """A split graph file under the vLLM cache dir, and its path."""
    path = tmp_path.joinpath("vllm_cache", "torch_compile_cache", *under, "computation_graph.py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# the split graph\n")
    return str(path)


@pytest.fixture
def process(monkeypatch, tmp_path):
    """A process with its own compile dirs under the test's directory, where the Build and the probe record what they are
    asked: the list of the Build's calls, each (inductor dir, keywords), then the probe's (shapes,)."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm_cache"))
    monkeypatch.setenv(engine_env.INDUCTOR_DIR_MARK, "1")
    (tmp_path / "out").mkdir()
    asked: list[tuple[Any, ...]] = []

    def build(inductor_dir: str, **kw: Any) -> dict[str, Any]:
        asked.append((inductor_dir, kw))
        return {"routes_missing": MISSING, "routes": kw["routes"]}

    def probe_routes(shapes: list[list[int]]) -> dict[str, Any]:
        asked.append((shapes,))
        return ROUTES

    monkeypatch.setattr(SV, "build", build)
    monkeypatch.setattr(SV, "probe_routes", probe_routes)
    return asked


def _files(tmp_path: Any) -> dict[str, Any]:
    return {f.name: json.loads(f.read_text()) for f in (tmp_path / "out").iterdir()}


def test_the_build_is_made_without_routes_then_with_the_ones_probed_for_the_shapes_it_missed(process, tmp_path):
    graph, facts = _graph(tmp_path, "rank_0_0", "backbone"), _facts(tmp_path)
    kw = {"split_graph": graph, "launch_configs": CONFIGS, "schedule": SCHEDULE, "steps": 3, "capture_sizes": [8, 4, 2, 1], "max_num_seqs": 8,
          "device": sm.device_of(facts)}
    assert sm.build_here(_Source(), facts, SCHEDULE, 3, str(tmp_path / "out")) == {"routes_missing": MISSING, "routes": ROUTES}
    inductor = str(tmp_path / "inductor")
    assert process == [(inductor, {**kw, "routes": {}}), (MISSING,), (inductor, {**kw, "routes": ROUTES})]
    assert _files(tmp_path) == {"launch_configs.json": CONFIGS, "routes.json": ROUTES}
    assert (tmp_path / "out" / "launch_configs.json").read_text() == json.dumps(CONFIGS, indent=1)
    assert (tmp_path / "out" / "routes.json").read_text() == json.dumps(ROUTES, indent=1)


def test_an_engine_with_fused_moe_layers_hands_the_build_its_moe_record_and_writes_it(process, tmp_path):
    _graph(tmp_path)
    sm.build_here(_Source(), _facts(tmp_path, moe=MOE), SCHEDULE, 3, str(tmp_path / "out"))
    moe = {"layers": MOE, "pieces": {"layers.0.mlp": "c03"}, "configs": {"layers.0.mlp": {"BLOCK_SIZE_M": 16}}, "cc": [9, 0]}
    assert [call[1]["moe"] for call in process if len(call) == 2] == [moe, moe]
    assert _files(tmp_path) == {"launch_configs.json": CONFIGS, "moe.json": moe, "routes.json": ROUTES}
    assert (tmp_path / "out" / "moe.json").read_text() == json.dumps(moe, indent=1)


@pytest.mark.parametrize("held", [{"moe": {}}, {"moe": None}, {}])
def test_an_engine_with_no_fused_moe_layer_hands_the_build_no_moe_record(process, tmp_path, held):
    _graph(tmp_path)
    facts = {k: v for k, v in _facts(tmp_path).items() if k != "moe"} | held
    sm.build_here(_Source(), facts, SCHEDULE, 3, str(tmp_path / "out"))
    assert ["moe" in call[1] for call in process if len(call) == 2] == [False, False] and "moe.json" not in _files(tmp_path)


@pytest.mark.parametrize("graphs", [0, 2])
def test_a_cache_dir_that_does_not_hold_one_split_graph_is_refused_after_the_launch_record_is_written(process, tmp_path, graphs):
    for i in range(graphs):
        _graph(tmp_path, f"rank_0_{i}")
    facts = _facts(tmp_path)
    with pytest.raises(UsageError, match=f"^{graphs} computation_graph.py under {facts['cache_dir']}: one split graph is required$"):
        sm.build_here(_Source(), facts, SCHEDULE, 3, str(tmp_path / "out"))
    assert process == [] and _files(tmp_path) == {"launch_configs.json": CONFIGS}


def test_a_process_that_was_not_given_its_own_compile_dirs_is_refused_before_the_build(monkeypatch, process, tmp_path):
    _graph(tmp_path)
    monkeypatch.delenv(engine_env.INDUCTOR_DIR_MARK)
    with pytest.raises(UsageError) as stop:
        sm.build_here(_Source(), _facts(tmp_path), SCHEDULE, 3, str(tmp_path / "out"))
    assert str(stop.value) == "this process was not given its own compile dirs (engine.env.own_compile_dirs, ops/served_commit.sh)"
    assert process == []


def test_a_device_that_is_not_registered_is_refused_before_the_build(process, tmp_path):
    _graph(tmp_path)
    with pytest.raises(Unmodelled, match="NVIDIA H100"):
        sm.build_here(_Source(), _facts(tmp_path, device="NVIDIA H100 80GB HBM3"), SCHEDULE, 3, str(tmp_path / "out"))
    assert process == []


def test_a_build_that_refuses_the_artifact_is_raised_and_no_route_is_probed(monkeypatch, process, tmp_path):
    _graph(tmp_path)

    def build(inductor_dir: str, **kw: Any) -> dict[str, Any]:
        raise Unmodelled("no template models kernel k9")

    monkeypatch.setattr(SV, "build", build)
    with pytest.raises(Unmodelled, match="no template models kernel k9"):
        sm.build_here(_Source(), _facts(tmp_path), SCHEDULE, 3, str(tmp_path / "out"))
    assert process == [] and _files(tmp_path) == {"launch_configs.json": CONFIGS}
