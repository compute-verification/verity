"""h200_step.py: the H200 step at M = 16,384 from `v31 bench`'s JSON lines, and its refusal of an incomplete step."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accel_v31"))
import h200_step


def _line(n: int, k: int, ms: float, lp: int = 512, t: int = 8, row: int = 1024, m: int = 16384) -> str:
    arms = {a: {"ms": ms, "mhz": 1980 if a == "ovl" else None} for a in h200_step.ARMS}
    return json.dumps({"shape": f"h200-{n}-{k}", "m": m, "n": n, "k": k, "lp": lp, "t": t, "row": row, "S": k // (lp * t),
                       "arms": arms, "gamma_call": 1.0, "gamma_ovl": 1.0, "gamma_gemm": 1.0})


def test_step_counts_each_call_of_52_layers_and_the_head(tmp_path):
    f = tmp_path / "bench-row1024.jsonl"
    f.write_text("\n".join(["noise"] + [_line(n, k, 2.0) for n, k in h200_step.CALLS]
                           + [_line(n, k, 5.0, t=16) for n, k in h200_step.CALLS]) + "\n")
    g = tmp_path / "bench-row1024-wrap4.jsonl"
    g.write_text("\n".join(_line(n, k, 3.0) for n, k in h200_step.CALLS) + "\n")
    st = h200_step.steps(h200_step.rows([str(f), str(g)]))
    assert set(st) == {(512, 8, 1024, "row1024"), (512, 16, 1024, "row1024"), (512, 8, 1024, "row1024-wrap4")}
    assert st[(512, 8, 1024, "row1024")]["s"]["ovl"] == pytest.approx(261 * 2e-3)
    assert st[(512, 8, 1024, "row1024-wrap4")]["s"]["ovl"] == pytest.approx(261 * 3e-3)
    assert st[(512, 16, 1024, "row1024")]["s"]["gemm"] == pytest.approx(261 * 5e-3)
    macs = 16384 * (104 * 16384**2 + 52 * 128 * 16384 + 104 * 65536 * 16384 + 256 * 16384)
    assert st[(512, 8, 1024, "row1024")]["macs"] == macs
    assert st[(512, 8, 1024, "row1024")]["mhz"] == (1980, 1980)


def test_the_8b_step_is_32_layers_of_its_four_linears_against_cublaslt(tmp_path, capsys, monkeypatch):
    calls = h200_step.MODELS["8b"]
    f = tmp_path / "bench-row16384.jsonl"
    f.write_text("\n".join(_line(n, k, 2.0, lp=128, row=16384) for n, k in calls) + "\n")
    st = h200_step.steps(h200_step.rows([str(f)]), calls=calls)
    assert st[(128, 8, 16384, "row16384")]["s"]["ovl"] == pytest.approx(128 * 2e-3)
    assert st[(128, 8, 16384, "row16384")]["macs"] == 32 * 16384 * (6144 * 4096 + 4096 * 4096 + 28672 * 4096 + 4096 * 14336)
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 0
    out = capsys.readouterr().out
    assert "summary: 8b L 1024 row 16384 row16384: step 0.256 s overlapped, gamma_lt 1.000 against cuBLASLt's 0.256 s" in out
    assert "mqa16k52" not in out


def test_a_missing_call_or_another_m_gives_no_step(tmp_path, capsys, monkeypatch):
    calls = list(h200_step.CALLS)
    f = tmp_path / "bench-row1024.jsonl"
    f.write_text("\n".join([_line(n, k, 1.0) for n, k in calls[1:]] + [_line(*calls[0], 1.0, m=8192)]) + "\n")
    assert h200_step.steps(h200_step.rows([str(f)])) == {}
    monkeypatch.setattr(sys, "argv", ["h200_step.py", str(f)])
    assert h200_step.main() == 1
    assert "summary: no complete step" in capsys.readouterr().out


def test_pipe_duty_is_each_kernels_busy_time_over_the_overlapped_call(tmp_path, capsys):
    """h200_pipes.py: a pipe's busy time is its share of peak over the kernel's elapsed cycles (active / elapsed scales
    it) times the kernel's duration, the larger of its issue and cycles-active shares; the operands' generation is left
    out; and the duty is the busy time a call over fill-bench's overlapped call, so overlapping kernels add."""
    import h200_pipes

    cols = ["ID", "Kernel Name", "gpu__time_duration.sum", "sm__cycles_active.avg", "sm__cycles_elapsed.avg",
            "sm__pipe_tensor_op_gmma_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active",
            "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
            "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active"]
    rows = [["", "", "msecond", "cycle", "cycle", "%", "%", "%", "%"],
            ["0", "gen_kernel(signed char*, int, unsigned int, bool)", "1", "100", "100", "0", "50", "50", "10"],
            ["1", "void gemm_kernel<true, 2>(GArgs)", "2", "1,000", "1,000", "80", "5", "4", "2"],
            ["2", "void leaf_persist<V31Row>(Persist, int)", "1", "50", "100", "0", "90", "80", "20"],
            ["3", "fill_a_kernel(...)", "0.5", "100", "100", "0", "40", "40", "30"]]
    (tmp_path / "ncu-row16384-L4096-8B-o.raw.csv").write_text(
        "\n".join(",".join(f'"{c}"' for c in r) for r in [cols] + rows) + "\n")
    (tmp_path / "ncu-row16384-L4096-8B-o.log").write_text("8B-o: 1 calls, L 4096 row 16384, grid 132, gemm_bn 256\n")
    (tmp_path / "fill-bench-row16384-L4096.jsonl").write_text(
        json.dumps({"fill": True, "shape": "8B-o", "arms": {"ovl": {"ms": 2.5}, "lt": {"ms": 2.0}}}) + "\n")
    assert h200_pipes.main(["h200_pipes.py", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert ("| 8B-o | 4096 | row16384 | 2.500 | 2.000 | 1.250 | 64.0% | 30.0% | 0.0% | 11.6% | 36.0% | 2.000 | 1.000 | "
            "0.500 | 1.40 | - |") in out


def test_pipe_kinds_read_ncus_demangled_names_and_cublaslts_gemm():
    import h200_pipes

    assert h200_pipes.kind("void gemm_kernel<false, 1>(GArgs)") == "gemm"
    assert h200_pipes.kind("void cutlass::device_kernel<cutlass3x_sm90_tensorop_s64x256x32gemm_s8_s8_s32>(Params)") == "gemm"
    assert h200_pipes.kind("sm90_xmma_gemm_s8s8_s32_tn_n_tilesize128x256x128_cublasLt") == "gemm"
    assert h200_pipes.kind("void leaf_persist<LtRow>(Persist, int)") == "hash"
    assert h200_pipes.kind("node_kernel(unsigned long*, int)") == "hash"
    assert h200_pipes.kind("void gen_kernel<1>(signed char*)") is None
    assert h200_pipes.kind("fill_a_kernel(signed char const*)") == "other"
