"""tensor_kinds/sm80.py: kernel names decode, the gate's counts match sm80.cu's templates (the loop counter an IADD3 or a
UIADD3), the issue coefficient is the least bound over every gated program, a slow lane withholds every joint line, the
joint row takes a convergence margin, and the census is in the machine model's schema."""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
SM80 = POUW / "tensor_kinds" / "sm80.py"


def _load(name: str, path: Path):
    import importlib.util
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SM = _load("pouw_tensor_kinds_sm80", SM80)
G = _load("gpu_kit_sass_gate", POUW / "gpu_kit" / "sass_gate.py")
LANE_MIX, LANE_IADD3, LANE_LOP3 = 0 | 1 << 4, 0 | 15 << 4, 2 | 15 << 4


def test_kernel_names_round_trip():
    for kern, struct, targs in [("k_mixu", "ImmaS8k16", [32, LANE_MIX, 8, 4]), ("k_spec", "HmmaF16k16F32", [64, 245]),
                                ("k_alone", "ImmaS8k32", [2]), ("k_opalone", "", [LANE_MIX]), ("k_duofold", "", [4])]:
        assert SM.demangle(SM.mangle(kern, struct, targs)) == (kern, struct, targs)
    assert SM.mangle("k_mixu", "ImmaS8k16", [32, 16, 8]) == "_Z6k_mixuI9ImmaS8k16Li32ELi16ELi8EEviPjP8BlockClk"
    assert SM.demangle("_Z7k_exactI9ImmaS8k32EvjjPjS0_") is None
    assert SM.demangle("_Z4mainv") is None


def test_expect_follows_the_templates():
    assert SM.expect("k_mixu", "ImmaS8k16", [32, LANE_MIX, 8]) == [{"IMMA*": 32, "*IADD3": 65, "FFMA": 64}]
    assert SM.expect("k_mixu", "ImmaS8k32", [128, LANE_IADD3, 4]) == [{"IMMA*": 8, "*IADD3": 129}]   # before U
    assert SM.expect("k_mixu", "ImmaS8k32", [72, LANE_MIX, 8, 2]) == [{"IMMA*": 16, "*IADD3": 73, "FFMA": 72}]
    assert SM.expect("k_spec", "ImmaS8k32", [64, LANE_IADD3]) == [{"IMMA*": 8}, {"*IADD3": 65}]
    assert SM.expect("k_opalone", "", [LANE_LOP3]) == [{"LOP3*": 64}]
    assert SM.expect("k_alone", "HmmaTf32k8", [8]) == [{"HMMA*": 8}]   # before k_alone took U
    assert SM.expect("k_alone", "BmmaAndK256", [4, 4]) == [{"BMMA*": 32}]
    assert [SM.mixu_u(r) for r in (8, 16, 24, 32, 40, 48, 64, 128)] == [8, 8, 6, 4, 4, 3, 2, 1]


def test_sm80_cu_agrees_with_the_analysis():
    cu = (POUW / "tensor_kinds" / "sm80.cu").read_text()
    assert "constexpr int mixu_u(int r) { return 192 / (8 + r) > 8 ? 8 : 192 / (8 + r) < 1 ? 1 : 192 / (8 + r); }" in cu
    rs = {int(r) for r in re.findall(r"\bmixu<F, (\d+),", cu)}
    assert rs and all(r % 8 == 0 for r in rs), "k_mixu puts R / 8 lane ops after each MMA"
    ru = [(int(r), int(u)) for r, u in re.findall(r"\bmixu<F, (\d+), \w+, \w+, (\d+)>", cu)]
    assert ru and all(u * (8 + r) <= 192 for r, u in ru), "a longer loop exits through CALL.REL.NOINC"
    enum = re.search(r"enum \{ (OP_[^}]*)\}", cu).group(1)
    ops = {int(v): k.lower() for k, v in re.findall(r"OP_(\w+) = (\d+)", enum)}
    assert ops.pop(SM.OP_NONE) == "none" and ops == {p: name for p, (name, _) in SM.OPS.items()}


def _mixu_listing(name: str, iadd3: int) -> str:
    """A k_mixu at 8 warps a SMSP as ptxas lowers it there: 32 IMMAs, the lane ops, and a UIADD3 loop counter."""
    body = ["IMMA.16816.S8.S8 R16, R20.ROW, R24.COL, R16"] * 32
    body += ["IADD3 R3, R3, R4, 0x9e3779b9"] * iadd3 + ["FFMA R5, R5, R6, 1"] * 64
    body += ["UIADD3 UR6, UR6, 0x1, URZ", "UISETP.NE.AND UP0, UPT, UR6, URZ, UPT", "BRA.U UP0, `(.L_x_1)"]
    lines = [f'\t.section\t.text.{name},"ax",@progbits', f"        .global         {name}",
             f"        .type           {name},@function", f"        .size           {name},(.L_x_9 - {name})",
             f'        .other          {name},@"STO_CUDA_ENTRY STV_DEFAULT"', f"{name}:", f".text.{name}:",
             "        /*0000*/                   CS2R R2, SR_CLOCKLO ;", ".L_x_1:"]
    addr = 0x10
    for ins in body + ["CS2R R12, SR_CLOCKLO", "IADD3 R3, R3, R5, RZ", "STG.E [R10.64], R3", "EXIT"]:
        lines.append(f"        /*{addr:04x}*/                   {ins} ;")
        addr += 0x10
    return "\n".join(lines + [".L_x_9:", ""])


def test_a_uniform_loop_counter_passes_the_gate_and_a_merged_lane_op_does_not():
    # the gate once refused 9 HMMA kernels at 8 warps a SMSP: ptxas made their counter a UIADD3
    name = SM.mangle("k_mixu", "ImmaS8k16", [32, LANE_MIX, 8])
    for iadd3, ok in ((64, True), (63, False)):
        (f,) = [G.H.Function(x) for x in G.H.parse_nvdisasm(_mixu_listing(name, iadd3))]
        rec = G.judge_kernel(f, SM.expect(*SM.demangle(name)))
        assert rec["ok"] is ok, rec["reasons"]
    assert any(r.startswith("loops: *IADD3: 64 in the loop") for r in rec["reasons"])


ALONE_LOOP = {"IMMA.16832.S8.S8": 8, "IADD3": 1, "ISETP.GE.AND": 1, "BRA": 1}
MIXU_LOOP = {"IMMA.16832.S8.S8": 32, "IADD3": 65, "FFMA": 64, "ISETP.GE.AND": 1, "BRA": 1}


def _row(kern, struct, targs, T, mma_w=0, lane_w=0, mma_ptx=0, lane=-1, iters=1000):
    form = {"ImmaS8k32": ("imma_s8_m16n8k32", 4096, 128, 128)}.get(struct, ("", 0, 0, 0))
    return {"probe": "tensor_kinds_sm80", "mode": kern[2:], "kern": kern, "struct": struct, "form": form[0], "targs": targs,
            "macs": form[1], "slots": form[2], "words": form[3], "mma_warps_smsp": mma_w, "lane_warps_smsp": lane_w,
            "mma_ptx_iter": mma_ptx, "lane": lane, "iters": iters, "sink_gate": True, "cyc_med": T, "mhz": 1409.3,
            "nvml_mhz_before": 1410, "nvml_mhz_after": 1410, "throttle": "0x0"}


def _run(out: Path, mix_rate: float, a_mix: float = 5.125, a_iadd3: float | None = None,
         run_id: str = "r20261017-072600-1397") -> Path:
    """One run: the two lanes alone, IMMA.16832 alone at 8.375 SMSP-cycles a MMA, one mixu beside IADD3+FFMA at a_mix
    and, given a_iadd3, one beside IADD3 at that bound."""
    out.mkdir(parents=True, exist_ok=True)
    mix = {"IADD3": 33, "FFMA": 32, "ISETP.GE.AND": 1, "BRA": 1}
    iadd3 = {"IADD3": 65, "ISETP.GE.AND": 1, "BRA": 1}
    n_mma, n_other = 4 * 1000 * 32, 4 * 1000 * 131
    cases = [(_row("k_opalone", "", [LANE_MIX], 8000 * 67 / mix_rate, lane_w=8, lane=LANE_MIX), mix),
             (_row("k_opalone", "", [LANE_IADD3], 8000 * 67 / 0.508, lane_w=8, lane=LANE_IADD3), iadd3),
             (_row("k_alone", "ImmaS8k32", [2, 1], 2 * 1000 * 8 * 8.375, mma_w=2, mma_ptx=8), ALONE_LOOP),
             (_row("k_mixu", "ImmaS8k32", [32, LANE_MIX, 4, 4], n_mma * a_mix + n_other, mma_w=4, mma_ptx=32,
                   lane=LANE_MIX),
              MIXU_LOOP)]
    if a_iadd3 is not None:
        cases.append((_row("k_mixu", "ImmaS8k32", [32, LANE_IADD3, 4, 4], n_mma * a_iadd3 + n_other, mma_w=4, mma_ptx=32,
                           lane=LANE_IADD3),
                      {"IMMA.16832.S8.S8": 32, "IADD3": 129, "ISETP.GE.AND": 1, "BRA": 1}))
    keys = [f"g0:{SM.mangle(r['kern'], r['struct'], r['targs'])}" for r, _ in cases]
    (out / "gate.json").write_text(json.dumps({"kernels": {k: {"ok": True, "reasons": []} for k in keys}}))
    (out / "loops.json").write_text(json.dumps({k: [lp] for k, (_, lp) in zip(keys, cases)}))
    (out / "meta.json").write_text(json.dumps({"run_id": run_id}))
    rows = [r for r, _ in cases] + [{"probe": "tensor_kinds_sm80", "done": True}]
    (out / "g0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert SM.main(["sm80.py", "analyze", str(out)]) == 0
    return out


def test_the_coefficient_holds_on_every_program(tmp_path: Path):
    # measured: IMMA.8816's IADD3 programs bound a at 2.207, below its IADD3+FFMA programs' 2.374
    f = json.loads((_run(tmp_path, 0.973, a_iadd3=4.9) / "sm80.json").read_text())["summary"]["forms"]["ImmaS8k32"]
    assert abs(f["a"] - 4.9) < 1e-9 and abs(f["a_mix"] - 5.125) < 1e-9 and abs(f["a_iadd3"] - 4.9) < 1e-9


def test_the_coefficient_q0_and_census(tmp_path: Path):
    out = _run(tmp_path / "a", 0.973)
    rec = json.loads((out / "sm80.json").read_text())
    s, fails = rec["summary"], rec["gate_failures"]
    assert fails == []
    f = s["forms"]["ImmaS8k32"]
    assert abs(f["alone_smsp_cycles_per_mma"] - 8.375) < 1e-9 and abs(f["a"] - 5.125) < 1e-9
    assert abs(f["slots_per_sm_clk"] - 4 * 128 / 8.375) < 1e-9 and f["a_iadd3"] is None
    assert abs(s["qstar0"]["ImmaS8k32"]["q0"] - 5.125 / 4) < 1e-9
    assert "Run r20261017-072600-1397" in (out / "sm80.md").read_text()

    ext = _run(tmp_path / "b", 0.973, a_mix=5.1, run_id="r20261017-081119-98b5")
    rows_csv, census_csv = SM.census([out], [ext])
    rows = {d["row"]: d for d in csv.DictReader(io.StringIO(rows_csv))}
    assert set(rows) == {"issue", "M", "A", "JIM"} and rows["JIM"]["status"] == "joint"
    assert rows["M"]["status"] == rows["issue"]["status"] == "measured"
    assert "0.973 a SMSP-cycle alone" in rows["JIM"]["description"] and "/ 1.0049 " in rows["JIM"]["description"]
    assert "IADD3+FFMA alternating, 0.973 a SMSP-cycle" in rows["issue"]["description"]
    lines = {(d["kind"], d["row"]): d for d in csv.DictReader(io.StringIO(census_csv))}
    assert set(lines) == ({("IMMA16832", "M"), ("ALU", "A"), ("IMMA16832", "JIM")}
                          | {(k, r) for k in ["IMMA16832", *SM.SCALAR, *SM.ISSUE_ONLY] for r in ("JIM", "issue")})
    assert all(lines[k, "issue"]["cycles"] == "1" for k in ["IMMA16832", *SM.SCALAR, *SM.ISSUE_ONLY])
    assert lines["ALU", "issue"]["runs"] == "r20261017-072600-1397 r20261017-081119-98b5"
    scale = 0.98 / (5.125 / 5.1)
    assert abs(float(lines["IMMA16832", "JIM"]["cycles"]) - 5.1 * scale) < 1e-5
    assert abs(float(lines["FFMA", "JIM"]["cycles"]) - scale) < 1e-5
    assert abs(float(lines["IMMA16832", "M"]["rate"]) - 4 / 8.375) < 1e-5 and lines["IMMA16832", "M"]["noise"] == "0.02"
    assert abs(float(lines["ALU", "A"]["rate"]) - 4 * 0.508 * 65 / 67) < 1e-5 and lines["ALU", "A"]["noise"] == "0.02"
    assert lines["IMMA16832", "JIM"]["runs"] == "r20261017-081119-98b5"
    for d in lines.values():
        assert sum(d[c] != "" for c in ("rate", "cycles", "u")) == 1 and d["since"] == d["until"] == ""


def test_the_joint_row_waits_for_the_search_to_converge(tmp_path: Path):
    a = _run(tmp_path / "a", 0.973)
    with pytest.raises(SystemExit, match="--extension"):
        SM.census([a])
    far = _run(tmp_path / "b", 0.973, a_mix=4.9, run_id="r20261017-081119-98b5")
    with pytest.raises(SystemExit, match="moved IMMA16832's a 4.6% .* extend the search again"):
        SM.census([a], [far])


def test_a_lane_below_issue_withholds_every_joint_line(tmp_path: Path):
    # IADD3 with three register reads beside FFMA: 0.671 a SMSP-cycle, so the mixed points
    # would sit on register reads, not on the issue facet
    out = _run(tmp_path, 0.671)
    fails = json.loads((out / "sm80.json").read_text())["gate_failures"]
    assert fails == ["opalone: the IADD3+FFMA lane issues 0.671 a SMSP-cycle alone, below 0.9"]
    _, census_csv = SM.census([out])
    assert "JIM" not in census_csv and ",issue," not in census_csv and "IMMA16832,M," in census_csv
