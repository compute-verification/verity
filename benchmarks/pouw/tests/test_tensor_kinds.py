"""tensor_kinds.py's fit: kernel names decode, SASS loops count, and a form cheaper in issue a slot is flagged."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_tensor_kinds", Path(__file__).parents[1] / "tensor_kinds" / "tensor_kinds.py")
TK = importlib.util.module_from_spec(spec)
sys.modules["pouw_tensor_kinds"] = TK
spec.loader.exec_module(TK)


def test_kernel_key_reads_the_struct_by_its_length():
    # QmmaE4m3k32 holds an E, which a regex on the mangling's E terminators would split
    assert TK.kernel_key("_Z6k_specI11QmmaE4m3k32Li32EEviPjP8BlockClk") == ("spec", "qmma_e4m3_m16n8k32_f32", 32)
    assert TK.kernel_key("_Z7k_aloneI9ImmaS8k32EviPjP8BlockClk") == ("alone", "imma_s8_m16n8k32", 0)
    assert TK.kernel_key("_Z4mainv") is None


def _fn(name: str, *loop_bodies: list[str]) -> str:
    """A function whose loops are the given bodies, each closed by a branch back to its first instruction, ending as
    cuobjdump's do: EXIT, then an idle branch to itself."""
    lines, addr = [f"\tFunction : {name}"], 0
    for body in loop_bodies:
        start = addr
        for ins in body + [f"BRA 0x{start:x}"]:
            lines.append(f"        /*{addr:04x}*/                   {ins} ;")
            addr += 16
    lines.append(f"        /*{addr:04x}*/                   EXIT ;")
    lines.append(f"        /*{addr + 16:04x}*/                   BRA 0x{addr + 16:x};")
    return "\n".join(lines)


def test_loops_counts_a_backward_branch_by_address_and_by_label():
    sass = _fn("f", ["IMMA.16832.S8.S8 R8, R4.ROW, R2.COL, R8", "IADD3 R3, R3, R4, R5", "NOP", "ISETP.GE.AND P0, PT, R0, R9, PT"])
    [loop] = TK.loops(TK.parse_sass(sass)["f"])
    assert loop == {"mma": 1, "mma_ops": ["IMMA.16832.S8.S8"], "other": 3, "iadd3": 1}
    labelled = "\tFunction : g\n.L_x_0:\n        /*0000*/  HMMA.16816.F32 R8, R4, R2, R8 ;\n" \
               "        /*0010*/  @P0 BRA `(.L_x_0) ;\n"
    assert TK.loops(TK.parse_sass(labelled)["g"]) == [{"mma": 1, "mma_ops": ["HMMA.16816.F32"], "other": 1, "iadd3": 0}]


# 64 lane adds, the counter and its compare; _fn closes the loop with a BRA, 67 instructions an iteration
LANE = ["IADD3 R3, R3, R4, RZ"] * 64 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]


def _opalone(iters: int, cyc_per_iter: float):
    row = {"probe": "tensor_kinds", "mode": "opalone", "form": "iadd3_32", "macs": 0, "slots": 0, "ops_per_mma": 0,
           "iters": iters, "cyc_med": iters * cyc_per_iter, "distinct_sms": 188, "mhz": 2100.0}
    return row, [_fn("_Z9k_opaloneiPjP8BlockClk", LANE)]


def _write(tmp_path: Path, cases) -> None:
    (tmp_path / "sass").mkdir()
    rows = [c[0] for c in cases] + [{"probe": "tensor_kinds", "done": True, "num_sms": 188}]
    (tmp_path / "tensor_kinds.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp_path / "sass" / "tensor_kinds.sass").write_text("\n".join(s for c in cases for s in c[1]) + "\n")


def _case(mode, form, struct, R, iters, cyc, mma_loop, op_loop=None, slots=128):
    k = {"alone": "k_alone", "spec": "k_spec", "mixu": "k_mixu"}[mode]
    name = f"_Z{len(k)}{k}I{len(struct)}{struct}" + (f"Li{R}E" if mode != "alone" else "") + "EviPjP8BlockClk"
    sass = [_fn(name, mma_loop, op_loop) if op_loop else _fn(name, mma_loop)]
    row = {"probe": "tensor_kinds", "mode": mode, "form": form, "macs": 4096, "slots": slots,
           "ops_per_mma": R / 8 if mode != "alone" else 0, "iters": iters, "cyc_med": cyc, "distinct_sms": 188,
           "mhz": 2100.0}
    return row, sass


def test_a_form_cheaper_in_issue_a_slot_is_flagged(tmp_path: Path):
    imma = ["IMMA.16832.S8.S8 R8, R4.ROW, R2.COL, R8"] * 8 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    hmma = [x.replace("IMMA.16832.S8.S8", "HMMA.16816.F32") for x in imma]
    lane = ["IADD3 R3, R3, R4, R5"] * 32 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    # spec at 4 ops a MMA: 2 MMA warps a SMSP x 1000 iters x 8 MMAs = 16000 MMAs; 2 x 1000 x (3 + 35) others = 76000
    cases = [
        _opalone(1000, 8 * 67 / 0.98),
        _case("alone", "imma_s8_m16n8k32", "ImmaS8k32", 0, 1000, 16000 * 16.0, imma),
        _case("spec", "imma_s8_m16n8k32", "ImmaS8k32", 32, 1000, 16000 * 8.115 + 76000, imma, lane),
        _case("alone", "hmma_f16_m16n8k16_f32", "HmmaF16k16F32", 0, 1000, 16000 * 16.0, hmma),
        _case("spec", "hmma_f16_m16n8k16_f32", "HmmaF16k16F32", 32, 1000, 16000 * 6.0 + 76000, hmma, lane),
    ]
    _write(tmp_path, cases)
    summary, fails = TK.analyze(tmp_path)
    assert fails == []
    assert abs(summary["lane_op_alone"]["issue_per_smsp_cycle"] - 0.98) < 1e-9
    assert abs(summary["imma_s8_m16n8k32"]["a"] - 8.115) < 1e-9
    assert summary["imma_s8_m16n8k32"]["alone_smsp_cycles_per_mma"] == 16.0
    assert abs(summary["hmma_f16_m16n8k16_f32"]["vs_ref"] - 6.0 / 8.115) < 1e-9
    assert summary["hmma_f16_m16n8k16_f32"]["below_ref"] and not summary["imma_s8_m16n8k32"]["below_ref"]
    assert TK.main(["tensor_kinds.py", str(tmp_path)]) == 0
    assert "**below**" in (tmp_path / "tensor_kinds.md").read_text()


def test_a_lane_op_slower_than_issue_fails_the_gate(tmp_path: Path):
    # 0.45 a SMSP-cycle, as the three-source IADD3 ran: the mixed points would sit on its rate, not the issue facet
    _write(tmp_path, [_opalone(1000, 8 * 67 / 0.45)])
    _, fails = TK.analyze(tmp_path)
    assert fails == ["opalone: the lane op issues 0.450 a SMSP-cycle alone, below 0.9"]


def test_an_unfinished_run_fails_the_gate(tmp_path: Path):
    (tmp_path / "sass").mkdir()
    (tmp_path / "tensor_kinds.jsonl").write_text("")
    (tmp_path / "sass" / "tensor_kinds.sass").write_text("")
    _, fails = TK.analyze(tmp_path)
    assert fails == ["the run did not finish (no done line)", "opalone: no lane-op-alone row"]
