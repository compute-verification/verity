"""floored_gemm's sass_mix.py, which run.sh runs on the build's SASS before any lease, on made-up input: its opcode counts,
and its refusal (exit 5) of a binary with no IMMA."""
from __future__ import annotations

import json

import sass_mix

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
