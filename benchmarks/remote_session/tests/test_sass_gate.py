"""`sass_gate.py`, rows.sh's gate on the kernels a 64-row pass adds: it passes the two builds as ptxas and cuobjdump
report them (the default one, and the one capped at 128 registers, where the in-place kernels spill 16 bytes), keeps
each kernel's record and listing, and refuses a kernel of the eight that is missing, compiled for another arch,
holding local memory past its bound, sketching with another tensor-core form or none, or holding floating point that
isn't integer division's. Every other kernel of the build is `same_sass.py`'s to judge, whatever its frame."""
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
          "mapped64_pass_kernel": "_ZN2rb20mapped64_pass_kernelEPKmS1_mPK5uint4PmPjS6_",
          "mapped64_receipts_kernel": "_ZN2rb24mapped64_receipts_kernelEPKmS1_mPK5uint4Pm",
          "mapped_consumer_kernel<64>": "_ZN2rb22mapped_consumer_kernelILi64EEEvPKmS2_mPKjPmPjS5_ii",
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
           "mapped64_pass_kernel": (151, 0, 0, 0, ["FLO.U32 R2, R3"]
                                    + ["IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 32),
           "mapped64_receipts_kernel": (150, 0, 0, 0, ["IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 32),
           "mapped_consumer_kernel<64>": (255, 200, 268, 288, ["STL [R1+0x20], R7"]
                                          + ["IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 32),
           "signs_kernel<64>": (79, 256, 0, 0, ["STL.64 [R1+0x40], R6"]),
           "chain_prepare_kernel<64>": (128, 304, 44, 44, ["LDL.64 R6, [R1+0x40]"])}
SPILL = ["STL [R1], R3", "LDL R3, [R1]"]
CAPPED = {**DEFAULT, "sketch64_pass_kernel": (128, *DEFAULT["sketch64_pass_kernel"][1:]),
          "subtree64_receipts_kernel": (127, *DEFAULT["subtree64_receipts_kernel"][1:]),
          "mapped64_pass_kernel": (128, 16, 16, 16, DEFAULT["mapped64_pass_kernel"][4] + SPILL),
          "mapped64_receipts_kernel": (128, 16, 16, 16, DEFAULT["mapped64_receipts_kernel"][4] + SPILL)}


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


@pytest.mark.parametrize("kernels", [DEFAULT, CAPPED], ids=["default", "128 registers"])
def test_both_builds_pass_with_each_kernels_record(kernels):
    funcs, problems = sass_gate.gate(*build(kernels))
    assert problems == [] and set(funcs) == sass_gate.WANT
    for name, (regs, stack, stores, loads, body) in kernels.items():
        r = funcs[name]
        assert (r["regs"], r["stack"], r["spill_stores"], r["spill_loads"]) == (regs, stack, stores, loads)
        assert r["instructions"] == len(body) and r["symbol"] == SYMBOL[name]
        assert r["IMMA"] == (32 if name in sass_gate.SKETCHING else 0)
    assert funcs["pass_consumer_kernel<64>"]["INT_DIV"] == 1 and funcs["pass_consumer_kernel<64>"]["MUFU_RCP"] == 2


def test_the_cli_writes_the_record_and_each_kernels_listing_and_exits_0(tmp_path, capsys):
    ptxas, sass = build(CAPPED)
    (tmp_path / "p.txt").write_text(ptxas)
    (tmp_path / "s.txt").write_text(sass)
    out, listings = tmp_path / "gate.json", tmp_path / "sass64"
    assert sass_gate.main([str(tmp_path / "p.txt"), str(tmp_path / "s.txt"), str(out), "--listings", str(listings)]) == 0
    doc = json.loads(out.read_text())
    assert doc["problems"] == [] and set(doc["kernels"]) == sass_gate.WANT
    assert "mapped64_pass_kernel 128 registers, 32 IMMA.16832.S8.S8, 16 bytes local" in capsys.readouterr().out
    assert sorted(p.name for p in listings.iterdir()) == sorted(
        ["sketch64_pass_kernel.sass", "pass_consumer_kernel_64.sass", "subtree64_receipts_kernel.sass",
         "mapped64_pass_kernel.sass", "mapped64_receipts_kernel.sass", "mapped_consumer_kernel_64.sass",
         "signs_kernel_64.sass", "chain_prepare_kernel_64.sass"])
    listing = (listings / "sketch64_pass_kernel.sass").read_text().splitlines()
    assert listing[1] == "/*0000*/ I2F.U32.RP R0, R9 ; /* 0x0000007424387237 */" and len(listing) == 1 + 2 * 35


REFUSED = {
    "a kernel missing": ({k: v for k, v in DEFAULT.items() if k != "subtree64_receipts_kernel"},
                         "kernels missing from the build: ['subtree64_receipts_kernel']"),
    "the pass's kernel with a frame": (changed(DEFAULT, "sketch64_pass_kernel", frame=(16, 0, 0)), "local memory"),
    "the pass's kernel storing to the stack": (changed(DEFAULT, "sketch64_pass_kernel", body=SPILL + DEFAULT[
        "sketch64_pass_kernel"][4]), "local memory"),
    "an in-place kernel spilling above 128 registers": (changed(CAPPED, "mapped64_pass_kernel", regs=151),
                                                         "local memory"),
    "an in-place kernel spilling more at 128": (changed(CAPPED, "mapped64_receipts_kernel", frame=(32, 32, 32)),
                                                "local memory"),
    "a consumer past its frame": (changed(DEFAULT, "pass_consumer_kernel<64>", frame=(160, 172, 256)), "local memory"),
    "another tensor-core form": (changed(DEFAULT, "mapped64_pass_kernel", body=DEFAULT["mapped64_pass_kernel"][4] + [
        "HMMA.16816.F32 R4, R8.ROW, R12.COL, R4"]), "MMA forms other than IMMA.16832.S8.S8"),
    "an unsigned operand": (changed(DEFAULT, "sketch64_pass_kernel", body=DEFAULT["sketch64_pass_kernel"][4] + [
        "IMMA.16832.U8.S8 R56, R36.ROW, R116.COL, R56"]), "MMA forms other than IMMA.16832.S8.S8"),
    "no tensor-core instruction": (changed(DEFAULT, "subtree64_receipts_kernel", body=["IMAD R4, R5, R6, R7"]),
                                   "0 IMMA.16832.S8.S8"),
    "30 tensor-core instructions": (changed(DEFAULT, "subtree64_receipts_kernel", body=[
        "IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 30), "30 IMMA.16832.S8.S8"),
    "a tensor-core instruction in the signs' kernel": (changed(DEFAULT, "signs_kernel<64>", body=[
        "IMMA.16832.S8.S8 R52, R36.ROW, R118.COL, R52"] * 4), "MMA forms other than none"),
    "a flushed float": (changed(DEFAULT, "mapped64_receipts_kernel", body=DEFAULT["mapped64_receipts_kernel"][4] + [
        "FADD.FTZ R2, R3, R4"]), "floating point other than integer division's"),
    "a reciprocal of no integer": (changed(DEFAULT, "sketch64_pass_kernel", body=DEFAULT["sketch64_pass_kernel"][4] + [
        "MUFU.RCP R0, R0"]), "floating point other than integer division's"),
    "a square root": (changed(DEFAULT, "chain_prepare_kernel<64>", body=["MUFU.SQRT R0, R0"]),
                      "floating point other than integer division's"),
}


@pytest.mark.parametrize("case", REFUSED)
def test_the_gate_refuses(case):
    kernels, why = REFUSED[case]
    _, problems = sass_gate.gate(*build(kernels))
    assert len(problems) == 1 and why in problems[0], problems


def test_a_kernel_compiled_for_another_arch_or_absent_from_the_listing_is_refused(tmp_path):
    _, problems = sass_gate.gate(*build(DEFAULT, arch="sm_120a"))
    assert len(problems) == 8 and all("compiled for sm_120a, want sm_90a" in p for p in problems)
    assert sass_gate.gate(*build(DEFAULT, arch="sm_120a"), arch="sm_120a")[1] == []
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
