"""`sass_mix.py` on a listing written by hand in `cuobjdump -sass`'s format: which functions it reports, where it finds the
timed loop, and how it counts opcodes."""

from __future__ import annotations

import json
import sys

import sass_mix

LISTING = """
\t\tFunction : _Z5chainILi5EEvPmi
\t.headerflags\t@"EF_CUDA_SM120"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   ISETP.GE.U32 P1, PT, R0, 0x2, PT ;
        /*0020*/              @P1  BRA 0x40 ;
        /*0030*/                   MOV R2, RZ ;
        /*0040*/                   IMAD.WIDE.U32 R4, R2, R3, RZ ;
        /*0050*/                   IMAD.WIDE.U32.X R6, R2, R3, R5 ;
        /*0060*/                   ADD.X R7, R7, RZ ;
        /*0070*/              @P0  BRA 0x40 ;
        /*0080*/                   EXIT ;
\t\tFunction : _Z5probeILi5EEvPmi
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   EXIT ;
\t\tFunction : _Z4gatePmi
        /*0000*/                   IMAD.WIDE.U32 R4, R2, R3, RZ ;
        /*0010*/                   BRA.U UP0, 0x0 ;
"""


def test_the_loop_is_between_the_last_backward_branch_and_its_target(tmp_path):
    path = tmp_path / "sq.sass"
    path.write_text(LISTING)
    fns = sass_mix.parse(path)
    assert [len(ins) for ins in fns.values()] == [9, 2, 2]
    chain, probe, gate = fns.values()
    assert [i[0] for i in sass_mix.loop_body(chain)] == [0x40, 0x50, 0x60, 0x70]   # the forward branch at 0x20 is no loop
    assert sass_mix.loop_body(probe) == []
    assert [i[0] for i in sass_mix.loop_body(gate)] == [0x00, 0x10]


def test_opcodes_are_counted_with_and_without_their_suffixes_most_frequent_first(tmp_path):
    path = tmp_path / "sq.sass"
    path.write_text(LISTING)
    body = sass_mix.loop_body(sass_mix.parse(path)["_Z5chainILi5EEvPmi"])
    got = sass_mix.mix(body)
    assert got["total"] == 4
    assert list(got["base"].items()) == [("IMAD", 2), ("ADD", 1), ("BRA", 1)]
    assert got["full"] == {"IMAD.WIDE.U32": 1, "IMAD.WIDE.U32.X": 1, "ADD.X": 1, "BRA": 1}


def test_it_reports_the_chain_and_probe_kernels_only(tmp_path, capsys, monkeypatch):
    path = tmp_path / "sq.sass"
    path.write_text(LISTING)
    monkeypatch.setattr(sys, "argv", ["sass_mix.py", str(path)])
    sass_mix.main()
    out = json.loads(capsys.readouterr().out)
    assert sorted(out) == ["_Z5chainILi5EEvPmi", "_Z5probeILi5EEvPmi"]
    assert out["_Z5chainILi5EEvPmi"]["function"]["total"] == 9
    assert out["_Z5chainILi5EEvPmi"]["loop_body"]["total"] == 4
    assert out["_Z5probeILi5EEvPmi"]["loop_body"] == {"total": 0, "base": {}, "full": {}}
