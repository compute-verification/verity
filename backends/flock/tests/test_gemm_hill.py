"""The BF16 hillclimb's records (`backends/flock/pod/gemm_hill.py`, `gemm_slowdown.py`): the preflight check's verdict as a
point reads it, and the readers of records written before the check was named."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

POD = Path(__file__).resolve().parents[1] / "pod"
sys.path.insert(0, str(POD))
import gemm_slowdown as GS  # noqa: E402

DEF = "GemmCoordinate_v2{K=64,DOT=HopperBF16WgmmaDot16_v1}"


def _preflight_record(path: Path, passes=(True, True, True)) -> Path:
    cases = [{"case": c, "pass": p, "proofs_equal": p, "transcripts_equal": p} for c, p in zip(GS.PREFLIGHT_CASES, passes)]
    path.write_text(json.dumps({"definition": DEF, "live": {"statement_digest": "d" * 64}, "selftest": {"gpu_cases": cases}}) + "\n")
    return path


def test_the_preflight_check_passes_only_when_every_gpu_case_is_byte_equal(tmp_path):
    assert GS.preflight(_preflight_record(tmp_path / "a.jsonl"), shapes_needed=1)["status"] == "passed"
    assert GS.preflight(_preflight_record(tmp_path / "b.jsonl", (True, False, True)), shapes_needed=1)["status"] == "failed"
    assert GS.preflight(_preflight_record(tmp_path / "c.jsonl"), shapes_needed=2)["status"] == "failed"
    assert GS.preflight(tmp_path / "absent.jsonl")["status"] == "not_run"


def test_a_slowdown_row_reads_the_preflight_check_under_its_old_name_too(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    cfg = sorted(GS.CONFIGS)[0]
    row = {"configs": {cfg: {"phase": GS.CONFIGS[cfg][0], "slowdown": 2.0, "slowdown_prove_only": 1.5, "prover_s": 4.0, "native_s": 2.0}},
           "native": {"gpu": "fake"}}
    for key in ("preflight", "gate"):
        (tmp_path / f"{key}.json").write_text(json.dumps(row | {key: {"status": "passed"}}))
        out = tmp_path / f"{key}-result.json"
        GS.point(SimpleNamespace(slowdown=str(tmp_path / f"{key}.json"), config=cfg, from_run="r1", result=str(out)))
        assert json.loads(out.read_text())["validation"]["status"] == "passed"


def test_compute_and_metrics_take_the_preflight_record_under_its_old_flag(monkeypatch):
    import gemm_hill as GH

    seen = {}
    monkeypatch.setattr(GS, "compute", lambda a: seen.setdefault("compute", a.preflight))
    monkeypatch.setattr(GH, "metrics", lambda a: seen.setdefault("metrics", a.preflight))
    for flag in ("--gate", "--preflight"):
        seen.clear()
        monkeypatch.setattr(sys, "argv", ["gemm_slowdown.py", "compute", "--sweep", "s", flag, "p.jsonl", "--native", "n", "--result", "r"])
        GS.main()
        monkeypatch.setattr(sys, "argv", ["gemm_hill.py", "metrics", "--K", "64", flag, "p.jsonl", "--sweep", "s", "--stamps", "t",
                                          "--dmon", "d", "--out", "o", "--result", "r"])
        GH.main()
        assert seen == {"compute": "p.jsonl", "metrics": "p.jsonl"}
