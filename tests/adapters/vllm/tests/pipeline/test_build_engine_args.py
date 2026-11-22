"""What `build.main` hands the construction and records: the engine arguments, the export it ran in, the platform it selected, and
its refusals."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from verity_vllm import gc_tuning
from verity_vllm.pipeline import build as B

WL = Path(__file__).resolve().parents[5] / "adapters" / "vllm" / "workloads"
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










def test_a_one_gpu_build_takes_no_tensor_parallel_engine_arguments(tmp_path, monkeypatch):
    assert _construct_engine_args(tmp_path, monkeypatch, str(WL / f"{L40S_TP1}.json"), 1) == {}


@pytest.mark.parametrize("tp, rank", [(2, 0), (8, 3), (1, 1)])
def test_a_multi_gpu_build_is_refused_by_name_before_its_construction(tmp_path, monkeypatch, tp, rank):
    monkeypatch.setattr(B, "_construct", lambda *a, **k: pytest.fail("constructed"))
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    assert B.main(B.Options(out=str(tmp_path / "out"), tp=tp, rank=rank)) == 2
    res = json.loads((tmp_path / "out" / "result.json").read_text())
    assert res["exception"] == "ValueError" and res["message"].startswith("the multi-GPU model is not served")


@pytest.mark.parametrize("text, code_rev", [
    ('{"commit": "c0ffee", "patch_sha256": null}', {"commit": "c0ffee", "patch_sha256": None}),     # an export: what its file says
    ("{not JSON", None),
    pytest.param("[" * 100_000 + "]" * 100_000, None, id="nested past the parser"),                  # `json` raises RecursionError, no ValueError
    (None, None),                                                                                    # no export here
])
def test_a_build_records_the_export_it_ran_in(tmp_path, monkeypatch, text, code_rev):
    """A Build's ``code_rev`` is the ``EXPORT.json`` of the directory it ran in, and nothing where there is none or it can't be parsed."""
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
    target = json.dumps({"card": "rtx-pro-6000-server"})
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        B.main(B.Options(out=str(tmp_path / "declared"), target=target))
        B.main(B.Options(out=str(tmp_path / "host")))     # no declared target: the line is not written
    (rec,) = _logged(caplog)
    assert (rec.levelno, rec.getMessage()) == (
        logging.INFO, "[derive_step] declared target sm_120 x1: vLLM platform 1 of NVIDIA RTX PRO 6000 Blackwell Server Edition")


def test_a_build_that_raises_says_so_in_one_error_that_ends_with_its_traceback(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gc_tuning, "tune", lambda: None)
    with caplog.at_level(logging.INFO, logger="verity_vllm"):
        assert B.main(B.Options(out=str(tmp_path / "tp1"), wrapper="request", dead_free=True)) == 2
    (rec,) = _logged(caplog)
    res = json.loads((tmp_path / "tp1" / "result.json").read_text())
    head, _, trace = rec.getMessage().partition("\n")
    assert rec.levelno == logging.ERROR and head == f"[derive_step] CRASH {res['exception']}: {res['message']}"
    assert trace and res["traceback_tail"].endswith(trace) and trace.rstrip().splitlines()[-1] == f"{res['exception']}: {res['message']}"
