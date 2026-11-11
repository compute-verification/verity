"""A TP row's Build configures vLLM's meta engine with the ``disable_custom_all_reduce`` the row's engines run (its workload's), so the
``tp.engine_args_required`` and ``tp.collective_backend`` it records from that config are the row's, not `vllm_meta`'s default."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import build as B
from verity_vllm.pipeline import gc_tuning

WL = Path(__file__).resolve().parents[2] / "workloads"
GLM53 = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
L40S_TP2 = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"
RTX_TP8 = "qwen3-235b-a22b-fp8__fp8__rtxpro6000__tp8__b1__i256__o32__mixed__greedy__bi-eager"
L40S_TP1 = "llama32-1b__bf16__l40s__tp1__b1__i256__o32__mixed__greedy__bi-eager"


def _construct_engine_args(tmp_path: Path, monkeypatch, workload: str | None, tp: int) -> dict | None:
    """The ``engine_args`` `build.main` hands the construction, whose meta config takes them."""
    seen: dict = {}
    monkeypatch.setattr(B, "_construct", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(B, "_cuda_available", lambda: False)
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    B.main(B.Options(out=str(tmp_path / "out"), workload=workload, tp=tp))
    return seen["engine_args"]


@pytest.mark.parametrize("row, tp, engine_args", [
    (GLM53, 8, {"disable_custom_all_reduce": False}),     # an H box: vLLM's default, the custom all-reduce on
    (L40S_TP2, 2, {"disable_custom_all_reduce": True}),
    (RTX_TP8, 8, {"disable_custom_all_reduce": True}),
    (L40S_TP1, 1, {}),                                     # no TP group: vLLM's own default
])
def test_a_build_takes_the_custom_all_reduce_its_workload_states(tmp_path, monkeypatch, row, tp, engine_args):
    assert _construct_engine_args(tmp_path, monkeypatch, str(WL / f"{row}.json"), tp) == engine_args


def test_a_tp_build_of_a_workload_without_the_field_is_refused(tmp_path, monkeypatch):
    doc = json.loads((WL / f"{GLM53}.json").read_text())
    del doc["engine"]["args"]["disable_custom_all_reduce"]
    (tmp_path / "wl.json").write_text(json.dumps(doc))
    monkeypatch.setattr(B, "_construct", lambda *a, **k: pytest.fail("constructed"))
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    assert B.main(B.Options(out=str(tmp_path / "out"), workload=str(tmp_path / "wl.json"), tp=8)) == 2
    res = json.loads((tmp_path / "out" / "result.json").read_text())
    assert res["exception"] == "ValueError" and "disable_custom_all_reduce" in res["message"]


def test_a_tp_build_without_a_workload_leaves_vllm_meta_s_default(tmp_path, monkeypatch):
    assert _construct_engine_args(tmp_path, monkeypatch, None, 2) == {}


def test_a_tp_rank_s_dead_free_request_reaches_the_construction(tmp_path, monkeypatch):
    """`--dead-free` reaches `_construct`, whose derive takes it, and the result records it; a one-rank Build refuses it (one rank's
    request always is)."""
    seen: dict = {}
    monkeypatch.setattr(B, "_construct", lambda *a, **k: seen.update(k))
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(B, "_cuda_available", lambda: False)
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    B.main(B.Options(out=str(tmp_path / "tp2"), wrapper="request", tp=2, rank=1, dead_free=True))
    assert (seen["tp"], seen["rank"], seen["dead_free"]) == (2, 1, True)
    assert json.loads((tmp_path / "tp2" / "result.json").read_text())["tp"] == {"world": 2, "rank": 1, "dead_free": True}
    assert B.main(B.Options(out=str(tmp_path / "tp1"), wrapper="request", dead_free=True)) == 2
    assert "--dead-free" in json.loads((tmp_path / "tp1" / "result.json").read_text())["message"]
