"""floored_gemm's table.py on a made-up bench.jsonl and ncu CSVs: each arm's time over cuBLASLt's, ncu's tensor-pipe
share weighted by each kernel's duration, a metric ncu gave no number left out, and a CSV without ncu's columns shown as
no data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("floored_gemm_table", POUW / "floored_gemm" / "table.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)

TENSOR = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
ALU = "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"
HEAD = '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'
MS = {"lt": 1.0, "plain": 1.05, "b": 1.2, "c": 1.5, "d": 1.4, "c2": 1.6, "d2": 1.45, "c3": 1.55, "c4": 1.7, "form": 0.3}


def csv_of(rows: list[tuple[str, str, str, str]]) -> str:
    """ncu's CSV as run.sh captures it: its banner on the lines before the header (stderr goes to the same file)."""
    return "==PROF== Connected to process 1\n" + HEAD + "\n" + "".join(
        f'"{i}","1","{kernel}","{metric}","%","{value}"\n' for i, kernel, metric, value in rows)


def test_ncu_tensor_weights_by_duration(tmp_path):
    f = tmp_path / "ncu-o-c.csv"
    f.write_text(csv_of([("0", "form", "gpu__time_duration.sum", "1,000"), ("0", "form", TENSOR, "0"),
                         ("1", "gemm", "gpu__time_duration.sum", "3,000.0"), ("1", "gemm", TENSOR, "80.0"),
                         ("1", "gemm", ALU, "n/a")]))
    r = T.ncu_tensor(f)
    assert r == {"pipe_pct": 60.0, "ns": 4000.0, "kernels": 2, "gemm_pipe_pct": 80.0, "gemm_alu_pct": 0, "gemm_issue_pct": 0,
                 "gemm_dram_pct": 0}


@pytest.mark.parametrize("text", [
    "", "==ERROR== no kernels were profiled\n",
    '"ID","Kernel Name","Metric Value"\n"0","gemm","1"\n',
    HEAD + "\n" + '"0","1","gemm","gpu__time_duration.sum","ns","n/a"\n',
])
def test_ncu_tensor_has_no_data(tmp_path, text):
    f = tmp_path / "ncu-o-c.csv"
    f.write_text(text)
    assert T.ncu_tensor(f) is None and T.ncu_tensor(tmp_path / "absent.csv") is None


def test_the_table_of_a_run(tmp_path, capsys, monkeypatch):
    row = {"shape": "o", "m": 8192, "n": 4096, "k": 4096, "tile": "t0",
           "arms": {a: {"ms": ms, "eta": 0.5 / ms, "mhz": 2000.0} for a, ms in MS.items()}}
    (tmp_path / "bench.jsonl").write_text("built fgemm\n" + json.dumps(row) + "\n")
    (tmp_path / "ncu-o-c.csv").write_text(csv_of([("0", "gemm", "gpu__time_duration.sum", "2000"), ("0", "gemm", TENSOR, "75")]))
    (tmp_path / "ncu-o-b.csv").write_text("==ERROR== no kernels were profiled\n")
    (tmp_path / "ncu-stall-t0-o-c.csv").write_text(csv_of([("0", "gemm", "gpu__time_duration.sum", "2000")]))
    monkeypatch.setattr("sys.argv", ["table.py", str(tmp_path / "bench.jsonl"), str(tmp_path)])
    assert T.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert ("| o m=8192 | 0.500 | 0.417 | 0.333 | 0.357 | 0.312 | 0.345 | 0.323 | 0.294 | +20.0% | +50.0% | +40.0% | +60.0% | "
            "+45.0% | +55.0% | +70.0% |") in out
    assert ("| o | 8192 4096 4096 | 0.500 | 0.476 | 0.417 | 0.333 | 0.357 | 0.312 | 0.345 | 0.323 | 0.294 | +20.0% | +50.0% | "
            "+40.0% | +60.0% | +45.0% | +25.0% | 25.0% | 2000 |") in out
    assert out[-2:] == ["| o | b | (no data) | | | | | | |", "| o | c | 75.0 | 75.0 | 0.0 | 0.0 | 0.0 | 1 | 2000 |"]
    assert not any("once per challenge" in line for line in out)


def test_the_arms_that_form_b_once(tmp_path, capsys, monkeypatch):
    ms = dict(MS, bA=1.1, cA=1.32, c2A=1.21)
    row = {"shape": "o", "m": 8192, "n": 4096, "k": 4096, "tile": "t0",
           "arms": {a: {"ms": v, "eta": 0.5 / v, "mhz": 2000.0} for a, v in ms.items()}}
    (tmp_path / "bench.jsonl").write_text(json.dumps(row) + "\n")
    monkeypatch.setattr("sys.argv", ["table.py", str(tmp_path / "bench.jsonl")])
    assert T.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "| o | 0.455 | 0.379 | 0.413 | +10.0% | +32.0% | +21.0% | +20.0% |"
