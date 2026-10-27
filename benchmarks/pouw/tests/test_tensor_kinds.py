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
    assert TK.kernel_key("_Z9k_opaloneILi2EEviPjP8BlockClk") == ("opalone", "iadd3_ffma", 64)
    assert TK.kernel_key("_Z6k_mixuI9ImmaS8k16Li64ELi0EEviPjP8BlockClk") == ("mixu_iadd3", "imma_s8_m16n8k16", 64)
    assert TK.kernel_key("_Z6k_specI9ImmaS8k16Li64ELi2EEviPjP8BlockClk") == ("spec", "imma_s8_m16n8k16", 64)
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
    assert loop == {"mma": 1, "mma_ops": ["IMMA.16832.S8.S8"], "other": 3, "iadd3": 1, "ffma": 0}
    labelled = "\tFunction : g\n.L_x_0:\n        /*0000*/  HMMA.16816.F32 R8, R4, R2, R8 ;\n" \
               "        /*0010*/  @P0 BRA `(.L_x_0) ;\n"
    assert TK.loops(TK.parse_sass(labelled)["g"]) == [
        {"mma": 1, "mma_ops": ["HMMA.16816.F32"], "other": 1, "iadd3": 0, "ffma": 0}]


# 64 lane ops, the counter and its compare; _fn closes the loop with a BRA, 67 instructions an iteration
LANE = ["IADD3 R3, R3, R4, RZ", "FFMA R5, R5, R6, 1"] * 32 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]


def _opalone(iters: int, cyc_per_iter: float, x: int = 2):
    row = {"probe": "tensor_kinds", "mode": "opalone", "form": TK.LANES[x], "macs": 0, "slots": 0, "ops_per_mma": 0,
           "iters": iters, "cyc_med": iters * cyc_per_iter, "distinct_sms": 188, "mhz": 2100.0}
    return row, [_fn(f"_Z9k_opaloneILi{x}EEviPjP8BlockClk", LANE)]


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


def test_an_add_moved_to_the_fma_pipe_still_counts_as_an_add():
    # sm_89's ptxas emits part of an add.u32 stream as IMAD.IADD ('s lane loops)
    sass = _fn("f", ["IADD3 R3, R3, R4, RZ", "IMAD.IADD R5, R5, 0x1, R6", "FFMA R7, R7, R8, 1"])
    [loop] = TK.loops(TK.parse_sass(sass)["f"])
    assert loop["iadd3"] == 2 and loop["ffma"] == 1 and loop["other"] == 4


def test_kernel_key_reads_alone4():
    assert TK.kernel_key("_Z8k_alone4I9ImmaS8k16EviPjP8BlockClk") == ("alone4", "imma_s8_m16n8k16", 0)
    assert TK.kernel_key("_Z6k_specI11BmmaAndK256Li32ELi0EEviPjP8BlockClk") == ("spec_iadd3", "bmma_and_m16n8k256", 32)


def test_a_form_cheaper_in_issue_a_slot_is_flagged(tmp_path: Path):
    imma = ["IMMA.16832.S8.S8 R8, R4.ROW, R2.COL, R8"] * 8 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    hmma = [x.replace("IMMA.16832.S8.S8", "HMMA.16816.F32") for x in imma]
    lane = ["IADD3 R3, R3, R4, RZ", "FFMA R5, R5, R6, 1"] * 16 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
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
    assert abs(summary["lane_op_alone"]["iadd3_ffma"]["issue_per_smsp_cycle"] - 0.98) < 1e-9
    assert abs(summary["imma_s8_m16n8k32"]["a"] - 8.115) < 1e-9
    assert summary["imma_s8_m16n8k32"]["alone_smsp_cycles_per_mma"] == 16.0
    assert abs(summary["hmma_f16_m16n8k16_f32"]["vs_ref"] - 6.0 / 8.115) < 1e-9
    assert summary["hmma_f16_m16n8k16_f32"]["below_ref"] and not summary["imma_s8_m16n8k32"]["below_ref"]
    assert TK.main(["tensor_kinds.py", str(tmp_path)]) == 0
    assert "**below**" in (tmp_path / "tensor_kinds.md").read_text()


def test_a_program_faster_than_a_joint_row_is_marked_over(tmp_path: Path):
    # a measured IMMA.16816 + IADD3 x8 (spec): 17.563 SMSP-cycles a MMA at 9.125 others a MMA, where
    # JI3e_1 (15.030 + 0.317 a other) asks at least 17.92
    k16 = ["IMMA.16816.S8.S8 R8, R4.ROW, R2.COL, R8"] * 8 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    lane = ["IADD3 R3, R3, R4, RZ"] * 64 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    # spec: 2 x 1000 x 8 = 16000 MMAs; 2 x 1000 x (3 + 67) = 140000 others, 8.75 a MMA
    cases = [_opalone(1000, 8 * 67 / 0.98),
             _case("spec", "imma_s8_m16n8k16", "ImmaS8k16", 64, 1000, 16000 * 17.4, k16, lane)]
    _write(tmp_path, cases)
    summary, fails = TK.analyze(tmp_path)
    assert fails == []
    j = summary["imma_s8_m16n8k16"]["ji3e"]
    a, b = TK.JI3E["JI3e_1"]
    assert j["JI3e_1"]["over"] and abs(j["JI3e_1"]["load"] - (a + b * 8.75) / 17.4) < 1e-9
    assert not j["JI3e_0"]["over"]
    TK.main(["tensor_kinds.py", str(tmp_path)])
    assert "**over** (spec R=64)" in (tmp_path / "tensor_kinds.md").read_text()

    # refit: JI3e_1's MMA comes down to what this program allows at the noise margin, then the row ÷ SCHED; JI3e_0
    # holds at VF3e's (8.302 allowed), and IMMA s8 m16n8k32, absent here, would keep VF3e's on every row
    rows = TK.refit([tmp_path])["rows"]
    assert abs(rows["JI3n_1"]["MMA"] - (0.98 * 17.4 - b * 8.75) / 1.07) < 1e-9
    assert abs(rows["JI3n_1"]["other"] - b / 1.07) < 1e-12
    assert rows["JI3n_1"]["IMMA16816"] == rows["JI3n_1"]["MMA"] and set(rows["JI3n_1"]) == {"MMA", "other", "IMMA16816"}
    assert abs(rows["JI3n_0"]["MMA"] - TK.JI3E["JI3e_0"][0] / 1.07) < 1e-9
    assert abs(rows["JI3n_0"]["IMMA16816"] - (0.98 * 17.4 - 8.75) / 1.07) < 1e-9


def test_a_spec_point_with_role_cycles_reads_each_role_over_its_own_time(tmp_path: Path):
    # the MMA warps finish at 8 SMSP-cycles a MMA, the lane warps at 0.5 lane instructions a SMSP-cycle; the block
    # cycles are the slower role's, so the MMA rate over them is only a lower bound
    imma = ["IMMA.16832.S8.S8 R8, R4.ROW, R2.COL, R8"] * 8 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    lane = ["IADD3 R3, R3, R4, RZ", "FFMA R5, R5, R6, 1"] * 16 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    row, sass = _case("spec", "imma_s8_m16n8k32", "ImmaS8k32", 32, 1000, 140000, imma, lane)
    row |= {"mma_cyc_med": 16000 * 8.0, "op_cyc_med": 2 * 1000 * 35 / 0.5}
    _write(tmp_path, [_opalone(1000, 8 * 67 / 0.98), (row, sass)])
    summary, fails = TK.analyze(tmp_path)
    [p] = summary["imma_s8_m16n8k32"]["points"]
    assert fails == [] and p["mma_cycles_per_mma"] == 8.0 and p["lane_per_smsp_cycle"] == 0.5
    assert p["cycles_per_mma"] == 140000 / 16000
    done = [json.loads(x) for x in (tmp_path / "tensor_kinds.jsonl").read_text().splitlines()]
    (tmp_path / "tensor_kinds.jsonl").write_text("\n".join(
        json.dumps(r | {"sm": 100, "group": 1} if r.get("done") else r) for r in done) + "\n")
    assert TK.analyze(tmp_path)[1] == ["sm_100: groups [1] finished, want 0 (int8)"]


def test_a_lane_op_slower_than_issue_fails_the_gate(tmp_path: Path):
    # 0.52 a SMSP-cycle, as IADD3 alone runs: the mixed points would sit on its pipe's rate, not the issue facet
    _write(tmp_path, [_opalone(1000, 8 * 67 / 0.52)])
    _, fails = TK.analyze(tmp_path)
    assert fails == ["opalone: the lane op issues 0.520 a SMSP-cycle alone, below 0.9"]


def test_an_unfinished_run_fails_the_gate(tmp_path: Path):
    (tmp_path / "sass").mkdir()
    (tmp_path / "tensor_kinds.jsonl").write_text("")
    (tmp_path / "sass" / "tensor_kinds.sass").write_text("")
    _, fails = TK.analyze(tmp_path)
    assert fails == ["the run did not finish (no done line)", "opalone: no iadd3_ffma row"]


def test_kernel_names_of_the_search_decode():
    assert TK.tpl_key("_Z7k_spec2I9ImmaS8k16Li64ELi0ELi8ELi1ELi1EEviPjP8BlockClk") == (
        "k_spec2", "imma_s8_m16n8k16", (64, 0, 8, 1, 1))
    assert TK.tpl_key("_Z6k_mix2I11ImmaS8m8k16Li32ELi4ELi16ELi6ELi2EEviPjP8BlockClk") == (
        "k_mix2", "imma_s8_m8n8k16", (32, 4, 16, 6, 2))
    assert TK.tpl_key("_Z6k_specI9ImmaS8k16Li64ELi0EEviPjP8BlockClk") == ("k_spec", "imma_s8_m16n8k16", (64, 0))
    assert TK.tpl_key("_Z9k_opaloneILi2EEviPjP8BlockClk") is None and TK.tpl_key("_Z4mainv") is None


def test_a_faster_schedule_joins_the_refit_and_measures_the_margin(tmp_path: Path):
    # M1: IMMA.16816 + IADD3 x8 (spec) at 17.4 SMSP-cycles a MMA; the search finds the same program at 17.0 (k_spec2,
    # 2 + 2 warps a SMSP) and re-times M1's own at 17.5 (the control). Both 8.75 others a MMA.
    k16 = ["IMMA.16816.S8.S8 R8, R4.ROW, R2.COL, R8"] * 8 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    lane = ["IADD3 R3, R3, R4, RZ"] * 64 + ["IADD3 R0, R0, 0x1, RZ", "ISETP.GE.AND P0, PT, R0, R9, PT"]
    m1 = tmp_path / "m1"
    m1.mkdir()
    _write(m1, [_opalone(1000, 8 * 67 / 0.98), _case("spec", "imma_s8_m16n8k16", "ImmaS8k16", 64, 1000, 16000 * 17.4, k16, lane)])
    srch = tmp_path / "m1c"
    (srch / "sass").mkdir(parents=True)
    names = {"k_spec2": "_Z7k_spec2I9ImmaS8k16Li64ELi0ELi8ELi2ELi2EEviPjP8BlockClk",
             "k_spec": "_Z6k_specI9ImmaS8k16Li64ELi0EEviPjP8BlockClk"}
    (srch / "sass" / "search-ImmaS8k16_R64.sass").write_text("\n".join(_fn(n, k16, lane) for n in names.values()) + "\n")
    recs = [{"kernel": k, "form": "imma_s8_m16n8k16", "tpl": tpl, "mma_warps": 2, "lane_warps": 2, "warps": 0, "iters": 1000,
             "cyc_med": 16000 * cyc, "distinct_sms": 188, "mhz": 2100.0}
            for k, tpl, cyc in [("k_spec", [64, 0], 17.5), ("k_spec2", [64, 0, 8, 2, 2], 17.0)]]
    (srch / "search-ImmaS8k16_R64.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs + [{"done": True, "num_sms": 188}]) + "\n")
    pts, fails = TK.search_points(srch)
    assert fails == [] and [round(p["cycles_per_mma"], 9) for _, p in pts] == [17.5, 17.0]
    assert all(abs(p["ops_per_mma"] - 8.75) < 1e-12 for _, p in pts) and [p["control"] for _, p in pts] == [True, False]

    # the margin is the time ratio the faster schedule beat M1's row by, 17.4 / 17.0 at the same counts, on every row
    a, b = TK.JI3E["JI3e_1"]
    fit = TK.refit([m1], search=[srch])
    assert all(abs(fit["sched"][f"JI3o_{i}"] - 17.4 / 17.0) < 1e-12 for i in range(3))
    assert abs(fit["rows"]["JI3o_1"]["MMA"] - (0.98 * 17.0 - b * 8.75) / (17.4 / 17.0)) < 1e-9
    assert abs(fit["rows"]["JI3o_1"]["other"] - b / (17.4 / 17.0)) < 1e-12
    assert abs(fit["rows"]["JI3o_0"]["MMA"] - (0.98 * 17.0 - 8.75) / (17.4 / 17.0)) < 1e-9
    # a search that finds nothing faster than M1's row allows leaves the 2% noise margin alone
    slow = json.loads((srch / "search-ImmaS8k16_R64.jsonl").read_text().splitlines()[1])
    (srch / "search-ImmaS8k16_R64.jsonl").write_text(
        json.dumps({**slow, "cyc_med": 16000 * 17.9}) + "\n" + json.dumps({"done": True, "num_sms": 188}) + "\n")
    assert TK.refit([m1], search=[srch])["sched"] == {f"JI3o_{i}": 1.0 for i in range(3)}
    assert TK.main(["tensor_kinds.py", "search", str(srch)]) == 0
    assert json.loads((srch / "search.json").read_text())["gate_failures"] == []
