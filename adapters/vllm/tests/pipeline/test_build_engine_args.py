"""A TP row's Build configures vLLM's meta engine with the ``disable_custom_all_reduce`` the row's engines run (its workload's), so the
``tp.engine_args_required`` and ``tp.collective_backend`` it records from that config are the row's, not `vllm_meta`'s default."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from verity_vllm import gc_tuning
from verity_vllm.pipeline import build as B

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


@pytest.mark.parametrize("text, code_rev", [
    ('{"commit": "c0ffee", "patch_sha256": null}', {"commit": "c0ffee", "patch_sha256": None}),     # an export: what its file says
    ("{not JSON", None),
    (None, None),                                                                                    # no export here
])
def test_a_build_records_the_export_it_ran_in(tmp_path, monkeypatch, text, code_rev):
    """A Build's ``code_rev`` is the ``EXPORT.json`` of the directory it ran in, and nothing where there is none or it is not JSON."""
    monkeypatch.chdir(tmp_path)
    if text is not None:
        (tmp_path / "EXPORT.json").write_text(text)
    _construct_engine_args(tmp_path, monkeypatch, None, 1)
    assert json.loads((tmp_path / "out" / "result.json").read_text())["code_rev"] == code_rev


def _logged(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "verity_vllm.pipeline.build"]


def test_a_build_of_a_declared_target_names_the_platform_it_selected(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(B, "_construct", lambda *a, **k: None)
    monkeypatch.setattr(B.engine_env, "apply_target", lambda env: None)
    monkeypatch.setattr(B, "_cuda_available", lambda: False)
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    monkeypatch.setattr("verity_vllm.program.frontend.export_compat.select_cuda_platform", lambda world, name: f"{world} of {name}")
    target = json.dumps({"compute_capability": [12, 0], "num_sms": 188})
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        B.main(B.Options(out=str(tmp_path / "declared"), target=target, tp=2, device_name="a device"))
        B.main(B.Options(out=str(tmp_path / "host"), tp=2, device_name="a device"))     # no declared target: the line is not written
    (rec,) = _logged(caplog)
    assert (rec.levelno, rec.getMessage()) == (logging.INFO, "[derive_step] declared target sm_120 x2: vLLM platform 2 of a device")


def test_a_build_that_raises_says_so_in_one_error_that_ends_with_its_traceback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        assert B.main(B.Options(out=str(tmp_path / "tp1"), wrapper="request", dead_free=True)) == 2
    (rec,) = _logged(caplog)
    res = json.loads((tmp_path / "tp1" / "result.json").read_text())
    head, _, trace = rec.getMessage().partition("\n")
    assert rec.levelno == logging.ERROR and head == f"[derive_step] CRASH {res['exception']}: {res['message']}"
    assert trace and res["traceback_tail"].endswith(trace) and trace.rstrip().splitlines()[-1] == f"{res['exception']}: {res['message']}"
