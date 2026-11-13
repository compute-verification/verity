"""`sass_gate.py`, rows.sh's gate on the kernels a 64-row pass adds: it passes the builds as ptxas and cuobjdump
report them at each arch it has a table for (the default one, and the one capped at 128 registers, where sm_120a's
receipt grid spills up to 32 bytes), keeps each kernel's record and listing, and refuses a kernel of the five that is
missing, compiled for another arch or one with no table, holding local memory
past its bound, sketching with another tensor-core form or none, or holding floating point that isn't a whole
integer division. Every other kernel of the build is `same_sass.py`'s to judge, whatever its frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sass_gate

SYMBOL = {"sketch64_pass_kernel": "_ZN2rb20sketch64_pass_kernelEPK5uint4mmS2_PmPjjS4_",
          "pass_consumer_kernel<64>": "_ZN2rb20pass_consumer_kernelILi64EEEvPK5uint4mmPKjPmPjS6_mmiim",
          "subtree64_receipts_kernel": "_ZN2rb25subtree64_receipts_kernelEPK5uint4mPKmiS2_Pmm",
          "signs_kernel<64>": "_ZN2rb12signs_kernelILi64EEEvNS_10StreamHeadEPj",
          "chain_prepare_kernel<64>": "_ZN2rb20chain_prepare_kernelILi64EEEvjPKmPhPj"}
#: main's kernels beside them, which the gate leaves to same_sass.py: one at the register limit with a frame and
#: spills, and one sketching through another tensor-core form
OTHERS = {"_ZN2rb20pass_consumer_kernelILi16EEEvPK5uint4mmPKjPmPjS6_mmiim": (255, 160, 172, 224, ["STL [R1+0x10], R4"]),
          "_ZN2rb13sketch_kernelILb0EEEvPK5uint4S3_mS3_Pi": (100, 0, 0, 0, ["HFMA2.MMA R4, R5, R6, R7"])}
DIVISION = ["I2F.U32.RP R0, R9", "MUFU.RCP R0, R0", "F2I.FTZ.U32.TRUNC.NTZ R3, R0"]
#: each kernel's (registers, stack frame, spill stores, spill loads) and instructions, the default build's shape
DEFAULT = {"sketch64_pass_kernel": (149, 0, 0, 0, DIVISION + ["IMMA.16832.S8.S8 R56, R36.ROW, R116.COL, R56"] * 32),
           "pass_consumer_kernel<64>": (255, 160, 172, 224, DIVISION + ["I2F.U64.RP R2, R8", "MUFU.RCP R2, R2",
                                                                       "F2I.U64.TRUNC R4, R2", "LDL R4, [R1+0x8]"]
                                        + ["IMMA.16832.S8.S8 R56, R36.reuse.ROW, R116.COL, R56"] * 32),
           "subtree64_receipts_kernel": (141, 0, 0, 0, ["IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 32),
           "signs_kernel<64>": (79, 256, 0, 0, ["STL.64 [R1+0x40], R6"]),
           "chain_prepare_kernel<64>": (128, 304, 44, 44, ["LDL.64 R6, [R1+0x40]"])}
SPILL = ["STL [R1], R3", "LDL R3, [R1]"]
CAPPED = {**DEFAULT, "sketch64_pass_kernel": (128, *DEFAULT["sketch64_pass_kernel"][1:]),
          "subtree64_receipts_kernel": (127, *DEFAULT["subtree64_receipts_kernel"][1:])}
#: sm_120a's forms: a 32-bit divisor in a uniform register, a 16-bit division (on uniform registers, and on plain
#: ones), and a constant loaded through HFMA2
DIVISION_U = ["UI2F.U32.RP UR4, UR6", "MUFU.RCP R2, UR4", "F2I.FTZ.U32.TRUNC.NTZ R5, R2"]
DIVISION16_U = ["UI2F.U16 UR4, UR6", "I2F.U16.RZ R0, R0",
                "UFSETP.GEU.AND UP1, UPT, |UR4|, 1.175494350822287508e-38, UPT",
                "UFSETP.GT.AND UP0, UPT, |UR4|, 8.50705917302346158658e+37, UPT", "UFSEL UR5, UR5, 1, !UP1",
                "UFSEL UR5, UR5, 0.25, !UP0", "UFMUL UR4, UR4, UR5", "MUFU.RCP R2, UR4", "FMUL R2, R2, UR5",
                "FMUL.RZ R4, R0, R3", "F2I.U32.TRUNC.NTZ R4, R4"]
DIVISION16 = ["I2F.U16 R2, R3", "I2F.U16.RZ R4, R4", "FSETP.GEU.AND P1, PT, |R2|.reuse, 1.175494350822287508e-38, PT",
              "FSETP.GT.AND P0, PT, |R2|, 8.50705917302346158658e+37, PT", "FSEL R5, R5, 1, !P1",
              "@!P0 FSEL R5, R5, 0.25, !P0", "FMUL R6, R2, R5", "MUFU.RCP R6, R6", "FMUL R5, R5, R6",
              "FMUL.RZ R5, R4, R5", "F2I.U32.TRUNC.NTZ R5, R5"]
CONSTANTS = ["HFMA2 R3, -RZ, RZ, 0, 0", "HFMA2 R11, -RZ, RZ, 0, 7.62939453125e-06"]
IMMAS = ["IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 32
#: the default build's shape at sm_120a, and the capped one's, where the receipt kernels spill
SM120 = {"sketch64_pass_kernel": (162, 0, 0, 0, DIVISION_U + DIVISION16_U + IMMAS),
         "pass_consumer_kernel<64>": (255, 112, 128, 160, DIVISION_U + DIVISION + ["I2F.U64.RP R0, R4", "MUFU.RCP R0, R0",
                                                                                 "F2I.U64.TRUNC R2, R2",
                                                                                 "LDL R4, [R1+0x8]"] + IMMAS),
         "subtree64_receipts_kernel": (145, 0, 0, 0, DIVISION16 + IMMAS),
         "signs_kernel<64>": (128, 368, 0, 0, ["STL.64 [R1+0x40], R6"] + CONSTANTS),
         "chain_prepare_kernel<64>": (128, 352, 92, 92, ["LDL.64 R6, [R1+0x40]"] + CONSTANTS * 6)}
SM120_CAPPED = {**SM120, "sketch64_pass_kernel": (128, 24, 24, 32, SM120["sketch64_pass_kernel"][4] + SPILL),
                "subtree64_receipts_kernel": (128, *SM120["subtree64_receipts_kernel"][1:])}
BUILDS = {"default": (DEFAULT, "sm_90a"), "128 registers": (CAPPED, "sm_90a"), "sm_120a": (SM120, "sm_120a"),
          "sm_120a, 128 registers": (SM120_CAPPED, "sm_120a")}


def build(kernels: dict, arch: str = "sm_90a", others: dict = OTHERS) -> tuple[str, str]:
    """ptxas -v's report and cuobjdump -sass's listing of a library holding `kernels` and `others`."""
    ptxas, sass = ["ptxas info    : 16792 bytes gmem"], []
    for name, (regs, stack, stores, loads, body) in [*others.items(), *((SYMBOL[k], v) for k, v in kernels.items())]:
        ptxas += [f"ptxas info    : Compiling entry function '{name}' for '{arch}'",
                  f"ptxas info    : Function properties for {name}",
                  f"    {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads",
                  f"ptxas info    : Used {regs} registers, used 1 barriers" + (f", {stack} bytes cumulative stack size"
                                                                               if stack else "")]
        sass += [f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM90"']
        for at, text in enumerate(body):
            sass += [f"        /*{16 * at:04x}*/                   {text} ;          /* 0x0000007424387237 */",
                     "                                                              /* 0x000e220000000800 */"]
    return "\n".join(ptxas) + "\n", "\n".join(sass) + "\n"


def changed(kernels: dict, name: str, regs=None, frame=None, body=None) -> dict:
    was = kernels[name]
    return {**kernels, name: (was[0] if regs is None else regs, *(was[1:4] if frame is None else frame),
                              was[4] if body is None else body)}


@pytest.mark.parametrize("name", BUILDS)
def test_each_build_passes_with_each_kernels_record(name):
    kernels, arch = BUILDS[name]
    funcs, problems = sass_gate.gate(*build(kernels, arch), arch)
    assert problems == [] and set(funcs) == sass_gate.WANT
    for k, (regs, stack, stores, loads, body) in kernels.items():
        r = funcs[k]
        assert (r["regs"], r["stack"], r["spill_stores"], r["spill_loads"]) == (regs, stack, stores, loads)
        assert r["instructions"] == len(body) and r["symbol"] == SYMBOL[k]
        assert r["IMMA"] == (32 if k in sass_gate.SKETCHING else 0)
    far = arch == "sm_120a"
    assert funcs["pass_consumer_kernel<64>"]["divisions"] == {"u32": 1 + far, "u64": 1, "u16": 0}
    assert funcs["sketch64_pass_kernel"]["divisions"] == {"u32": 1, "u64": 0, "u16": int(far)}
    assert funcs["subtree64_receipts_kernel"]["float"] == ({
        "I2F.U16": 1, "I2F.U16.RZ": 1, "FSETP.GEU.AND": 1, "FSETP.GT.AND": 1, "FSEL": 2, "FMUL": 2, "MUFU.RCP": 1,
        "FMUL.RZ": 1, "F2I.U32.TRUNC.NTZ": 1} if far else {})
    assert funcs["chain_prepare_kernel<64>"]["constants"] == (12 if far else 0)


def test_the_cli_writes_the_record_and_each_kernels_listing_and_exits_0(tmp_path, capsys):
    ptxas, sass = build(CAPPED)
    (tmp_path / "p.txt").write_text(ptxas)
    (tmp_path / "s.txt").write_text(sass)
    out, listings = tmp_path / "gate.json", tmp_path / "sass64"
    assert sass_gate.main([str(tmp_path / "p.txt"), str(tmp_path / "s.txt"), str(out), "--listings", str(listings)]) == 0
    doc = json.loads(out.read_text())
    assert doc["problems"] == [] and set(doc["kernels"]) == sass_gate.WANT
    assert "sketch64_pass_kernel 128 registers, 32 IMMA.16832.S8.S8, 0 bytes local" in capsys.readouterr().out
    assert sorted(p.name for p in listings.iterdir()) == sorted(
        ["sketch64_pass_kernel.sass", "pass_consumer_kernel_64.sass", "subtree64_receipts_kernel.sass",
         "signs_kernel_64.sass", "chain_prepare_kernel_64.sass"])
    listing = (listings / "sketch64_pass_kernel.sass").read_text().splitlines()
    assert listing[1] == "/*0000*/ I2F.U32.RP R0, R9 ; /* 0x0000007424387237 */" and len(listing) == 1 + 2 * 35


REFUSED = {
    "a kernel missing": ({k: v for k, v in DEFAULT.items() if k != "subtree64_receipts_kernel"},
                         "kernels missing from the build: ['subtree64_receipts_kernel']"),
    "the pass's kernel with a frame": (changed(DEFAULT, "sketch64_pass_kernel", frame=(16, 0, 0)), "local memory"),
    "the pass's kernel storing to the stack": (changed(DEFAULT, "sketch64_pass_kernel", body=SPILL + DEFAULT[
        "sketch64_pass_kernel"][4]), "local memory"),
    "the pass's kernel spilling at 128 registers on sm_90a": (
        changed(CAPPED, "sketch64_pass_kernel", frame=(16, 16, 16), body=DEFAULT["sketch64_pass_kernel"][4] + SPILL),
        "local memory"),
    "the opening's receipts spilling at 128": (changed(CAPPED, "subtree64_receipts_kernel", frame=(16, 16, 16)),
                                               "local memory"),
    "a consumer past its frame": (changed(DEFAULT, "pass_consumer_kernel<64>", frame=(160, 172, 256)), "local memory"),
    "another tensor-core form": (changed(DEFAULT, "subtree64_receipts_kernel", body=DEFAULT[
        "subtree64_receipts_kernel"][4] + ["HMMA.16816.F32 R4, R8.ROW, R12.COL, R4"]),
                                 "MMA forms other than IMMA.16832.S8.S8"),
    "an unsigned operand": (changed(DEFAULT, "sketch64_pass_kernel", body=DEFAULT["sketch64_pass_kernel"][4] + [
        "IMMA.16832.U8.S8 R56, R36.ROW, R116.COL, R56"]), "MMA forms other than IMMA.16832.S8.S8"),
    "no tensor-core instruction": (changed(DEFAULT, "subtree64_receipts_kernel", body=["IMAD R4, R5, R6, R7"]),
                                   "0 IMMA.16832.S8.S8"),
    "30 tensor-core instructions": (changed(DEFAULT, "subtree64_receipts_kernel", body=[
        "IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 30), "30 IMMA.16832.S8.S8"),
    "a tensor-core instruction in the signs' kernel": (changed(DEFAULT, "signs_kernel<64>", body=[
        "IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 4), "MMA forms other than none"),
    "a flushed float": (changed(DEFAULT, "subtree64_receipts_kernel", body=DEFAULT[
        "subtree64_receipts_kernel"][4] + ["FADD.FTZ R2, R3, R4"]), "floating point other than integer division's"),
    "a reciprocal of no integer": (changed(DEFAULT, "sketch64_pass_kernel", body=DEFAULT["sketch64_pass_kernel"][4] + [
        "MUFU.RCP R0, R0"]), "floating point other than integer division's"),
    "a square root": (changed(DEFAULT, "chain_prepare_kernel<64>", body=["MUFU.SQRT R0, R0"]),
                      "floating point other than integer division's"),
    "a division short of its truncation": (changed(DEFAULT, "sketch64_pass_kernel", body=DEFAULT[
        "sketch64_pass_kernel"][4][:2] + IMMAS), "floating point other than integer division's"),
    "a float converted and kept": (changed(DEFAULT, "signs_kernel<64>", body=["I2F.S32 R2, R3"]),
                                   "floating point other than integer division's"),
}
REFUSED_120 = {
    "a 16-bit division with a product more": (changed(SM120, "subtree64_receipts_kernel", body=SM120[
        "subtree64_receipts_kernel"][4] + ["FMUL R6, R2, R5"]), "floating point other than integer division's"),
    "a half-float product of registers": (changed(SM120, "signs_kernel<64>", body=SM120["signs_kernel<64>"][4] + [
        "HFMA2 R3, R4, R5, R6"]), "floating point other than integer division's"),
    "the pass's kernel spilling above 128 registers": (changed(SM120_CAPPED, "sketch64_pass_kernel", regs=162),
                                                        "local memory"),
    "the pass's kernel spilling more at 128": (changed(SM120_CAPPED, "sketch64_pass_kernel", frame=(24, 40, 24)),
                                                "local memory"),
    "sm_90a's consumer frame": (changed(SM120, "pass_consumer_kernel<64>", frame=(160, 172, 224)), "local memory"),
}


@pytest.mark.parametrize("case", [*REFUSED, *(f"sm_120a: {c}" for c in REFUSED_120)])
def test_the_gate_refuses(case):
    far = case.startswith("sm_120a: ")
    kernels, why = REFUSED_120[case.removeprefix("sm_120a: ")] if far else REFUSED[case]
    arch = "sm_120a" if far else "sm_90a"
    _, problems = sass_gate.gate(*build(kernels, arch), arch)
    assert len(problems) == 1 and why in problems[0], problems


def test_a_kernel_compiled_for_another_arch_or_absent_from_the_listing_is_refused(tmp_path):
    _, problems = sass_gate.gate(*build(DEFAULT, arch="sm_120a"))
    assert len(problems) == 5 and all("compiled for sm_120a, want sm_90a" in p for p in problems)
    funcs, problems = sass_gate.gate(*build(DEFAULT, arch="sm_100a"), arch="sm_100a")
    assert funcs == {} and len(problems) == 1 and "no table for sm_100a" in problems[0]
    ptxas, _ = build(DEFAULT)
    _, sass = build({k: v for k, v in DEFAULT.items() if k != "signs_kernel<64>"})
    assert sass_gate.gate(ptxas, sass)[1] == ["signs_kernel<64>: no listing"]
    _, sass = build(DEFAULT)
    ptxas, _ = build({k: v for k, v in DEFAULT.items() if k != "signs_kernel<64>"})
    assert sass_gate.gate(ptxas, sass)[1] == [f"{SYMBOL['signs_kernel<64>']}: no ptxas resource line"]
    (tmp_path / "p.txt").write_text(ptxas)
    (tmp_path / "s.txt").write_text(sass)
    assert sass_gate.main([str(tmp_path / "p.txt"), str(tmp_path / "s.txt"), str(tmp_path / "gate.json")]) == 1
    assert json.loads((tmp_path / "gate.json").read_text())["problems"]


def test_a_kernels_name_is_the_sources():
    assert {sass_gate.short(s) for s in SYMBOL.values()} == sass_gate.WANT
    assert sass_gate.short("_ZN2rb20pass_consumer_kernelILi16EEEvPK5uint4mmPKjPmPjS6_mmiim") == "pass_consumer_kernel<16>"
    assert sass_gate.short("_ZN2rb20sketch16_pass_kernelEPK5uint4mmPKjPmPjjS6_") == "sketch16_pass_kernel"
