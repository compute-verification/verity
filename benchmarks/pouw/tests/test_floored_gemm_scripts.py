"""floored_gemm's CPU scripts beside ref.py and table.py, each on made-up input: sass_mix.py's opcode counts, tb_table.py's
table of the tensor-core receipts, and diff.py's own checks of its layout and its fold against ref.py."""
from __future__ import annotations

import json

import diff
import numpy as np
import sass_mix
import tb_table

SASS = """
\t\tFunction : gemm_duo
        /*0000*/                   IMMA.16832.S8.S8 R0, R1, R2, R3 ;
        /*0010*/                   LOP3.LUT R4, R5, R6, R7, 0x96, !PT ;
        /*0020*/                   LEA.HI R4, R5, R6, R7, 0x3 ;
        /*0030*/                   MOV R1, R2 ;
\t\tFunction : form
        /*0000*/                   IMAD R4, R5, R6, R7 ;
        /*0010*/                   @P0 IMAD.MOV.U32 R4, RZ, RZ, R7 ;
"""


def test_sass_mix_counts_a_kernel_s_opcodes(tmp_path, capsys):
    (tmp_path / "fgemm.sass").write_text(SASS)
    assert sass_mix.main(str(tmp_path / "fgemm.sass")) == 0
    assert json.loads(capsys.readouterr().out) == {
        "gemm_duo": {"IMMA": 1, "IMMA.16832.S8.S8": 1, "LOP3": 1, "LEA": 1, "LEA.HI": 1, "total": 6},
        "form": {"IMAD": 2, "total": 2}}
    (tmp_path / "no-imma.sass").write_text("\t\tFunction : form\n        /*0000*/                   IMAD R4, R5, R6, R7 ;\n")
    assert sass_mix.main(str(tmp_path / "no-imma.sass")) == 5


def test_tb_table_of_a_run(tmp_path, capsys, monkeypatch):
    ms = {"lt": 1.0, "b": 1.2, "c": 1.5, "t4": 1.8, "t4g": 1.1, "g0": 1.0}
    row = {"shape": "o", "m": 8192, "n": 4096, "k": 4096, "tile": "t0",
           "arms": {a: {"ms": v, "eta": 0.5 / v, "mhz": 2000.0} for a, v in ms.items()}}
    (tmp_path / "bench.jsonl").write_text("built fgemm\n" + json.dumps(row) + "\n")
    (tmp_path / "ncu-o-t4.csv").write_text(
        '"ID","Kernel Name","Metric Name","Metric Value"\n"0","gemm","gpu__time_duration.sum","2,000"\n'
        '"0","gemm","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","75"\n')
    monkeypatch.setattr("sys.argv", ["tb_table.py", str(tmp_path / "bench.jsonl"), str(tmp_path)])
    assert tb_table.main() == 0
    out = capsys.readouterr().out.splitlines()
    assert "| o m=8192 | 0.500 | 0.417 | 0.333 | 0.278 | +20% | +50% | +80% |" in out
    assert "| o | t0 | 0.500 | 0.500 | 0.417 | 0.333 | 0.278 | 0.455 | 1.100 | 1.250 | 1.500 | 2000 |" in out
    assert out[-1] == "| o | t4 | 75.0 | 0.0 | 0.0 | 0.0 | 2000 |"


def test_diff_holds_its_layout_and_fold_to_ref():
    diff.check_slots()
    diff.check_sim()
    assert diff.dbytes(np.array([255]), 1, (0, 1)).tolist() == [[-255, 1]]
    deltas = diff.fixed_deltas()
    assert len(deltas) == 35 and deltas["+2^31"] == 1 << 31 and deltas["-121*121"] == -14641
    kind = diff.VARIANTS["tb4"]
    keys = {t: diff.tb_key(kind, t) for t in range(diff.MI * diff.NI)}
    states = diff.honest(np.random.default_rng(3), 64, 2).reshape(64, 1)
    assert diff.unchanged(kind, keys, [(0, 1, 5, 3)], states, [0]) == (1.0, 0.0, 0.0)
    p, cancel, changed = diff.unchanged(kind, keys, [(0, 1, 5, 3)], states, [1])
    assert (p, cancel, changed) == (0.0, 0.0, 1.0)
