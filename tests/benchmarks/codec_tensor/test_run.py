"""`verity bench codec_tensor` (`run.py`): its command line, and its numbers and its gates' validation written as the
run's `result.json` through the harness's kit."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "codec_tensor"
sys.path.insert(0, str(DIR))

import bench  # noqa: E402

_spec = importlib.util.spec_from_file_location("codec_tensor_run", DIR / "run.py")
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


class _Lib:
    def __init__(self, name: str, path: Path):
        self.name, self.path = name, str(path)


def test_the_command_is_run_py_and_bench_py_is_its_library(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run.py", "--help"])
    with pytest.raises(SystemExit) as ex:
        run.main()
    assert ex.value.code == 0 and "--lib" in capsys.readouterr().out
    assert not hasattr(bench, "main") and not hasattr(bench, "write_result")


def test_the_result_holds_each_number_and_the_gates_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    so = tmp_path / "wt.so"
    so.write_bytes(b"a build")
    res = {"ok": True, "read": {"gbps": 1500.0},
           "libs": {"p512": {"analysis": {"squarings_s": {"cuda": 2.5e9}, "k": {1: {"decode_x": 0.75}}},
                             "decode": {2: {"cuda": {"ok": True, "gbps_stored": 900.0}, "tc": {"ok": False}}},
                             "gemm": {"qkv": {"times": {1: {"ratio_vs_native": {"fused_k1": 1.25, "empty": 0.1}}}}}}}}
    (tmp_path / "out").mkdir()
    run.write_result(res, [_Lib("p512", so)], tmp_path / "out")
    doc = json.loads((tmp_path / "result.json").read_text())
    got = {(m["name"], m["value"], m["unit"], m["mark"]) for m in doc["measurements"]}
    assert got == {("read", 1500.0, "GB/s", "measured"), ("p512_squarings_cuda", 2.5e9, "1/s", "measured"),
                   ("p512_k1_decode_x", 0.75, "1", "measured"), ("p512_decode_k2_cuda", 900.0, "GB/s", "measured"),
                   ("p512_qkv_m1_fused_k1_vs_native", 1.25, "1", "measured")}
    files = ["out/codec-tensor.json", "out/bench.log"]
    assert doc["validation"] == {"status": "passed", "checker": "codec-tensor gates", "evidence": files,
                                 "detail": "device codec = wide.py, TC square = a^2 mod p, fused partial sums = plain"}
    assert doc["measurement_files"] == files and doc["workload_fingerprint"]["seed"] == bench.SEED.hex()
    assert set(doc["workload_fingerprint"]["libs"]) == {"p512"}


def test_a_failed_gate_writes_a_failed_validation_and_no_number(tmp_path):
    run.write_result({"ok": False, "libs": {"p512": {"gate": {}}}}, [], tmp_path)
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["validation"]["status"] == "failed" and doc["measurements"] == []
