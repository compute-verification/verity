"""The SASS gate (`harness/sass_gate.py`), on CUDA 12.9.1's and 13.0.1's listings of `fixtures/sass_gate/ftz_probe.cu` and
`ftz_refused.cu` (sm_120a, and 13.0.1's sm_90a): builds without flags pass, with every flagged instruction inside a pinned sequence; --use_fast_math, -ftz=true
and the approximate -prec-div=false -prec-sqrt=false fail, as does every kernel of `ftz_refused.cu`; a cubin whose ptxas
(version, build, arguments) isn't pinned with a numeric run exempts nothing; an instruction a toolkit excepts (MUFU.EX2
under 13.0.1-sm_90a, on kernels' units_softmax) passes alone, in its one form, and nothing else does; kernels' units.cu
at sm_90a passes, its calls reloading operands from their constant-bank slots, and fails under each mutation; each sequence
breaks under a mutation; a read the gate can't tell from a merge passes and is listed; the pins regenerate; the two disassemblies must agree; the fatbin
parser finds every cubin and refuses what it can't read; passes are cached and failures aren't; an arm must declare its
binaries; and, where a CUDA toolkit is present, real builds."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import sass_gate as SG  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sass_gate"
NOTES = json.loads((FIX / "tool_notes.json").read_text())
PLAIN_COVERED = {"int-seed": 10, "__cuda_sm20_rem_s64": 1, "__cuda_sm20_div_u64": 1, "__cuda_sm20_div_u16": 1,
                 "sqrt-fast-path": 6, "__cuda_sm20_sqrt_rn_f32_slowpath": 18, "div-fast-path": 4,
                 "__cuda_sm3x_div_rn_noftz_f32_slowpath": 30, "rcp-fast-path": 2, "__cuda_sm20_rcp_rn_f32_slowpath": 7}
INTEGER_COVERED = {"int-seed": 10, "__cuda_sm20_rem_s64": 1, "__cuda_sm20_div_u64": 1, "__cuda_sm20_div_u16": 1}
FP32_KERNELS = {"rho_k", "rcp_k", "div_k", "cdiv_k", "sqrt_k", "arith_k"}
TEST_RUN = {"run": "test"}


def pins(numeric: dict | None = TEST_RUN) -> dict:
    """sass_pins.json with each toolkit's numeric run replaced by `numeric`."""
    p = SG.load_pins()
    for tk in p["toolkits"].values():
        tk["numeric"] = numeric
    return p


def listing(name: str) -> str:
    return (FIX / f"{name}.nvdisasm.txt").read_text()


def verdict(name: str, text: str | None = None, notes: list | None = None, p: dict | None = None) -> dict:
    return SG.cubin_verdict(SG.parse_nvdisasm(listing(name) if text is None else text),
                            NOTES[name] if notes is None else notes, pins() if p is None else p)


def offending(v: dict) -> set[tuple[str, str]]:
    return {(o["function"], o["instruction"]) for o in v["offending"]}


def in_function(text: str, fn: str, old: str, new: str) -> str:
    """`text` with `old` replaced by `new` once, inside function `fn`'s section."""
    start = text.index(f"\n{fn}:\n")
    end = text.find("\t.section", start)
    end = len(text) if end < 0 else end
    assert text.count(old, start, end) == 1, (fn, old)
    return text[:start] + text[start:end].replace(old, new) + text[end:]


def after(text: str, fn: str, line: str, insn: str, addr: int) -> str:
    """`text` with `insn` inserted in `fn` after the instruction `line`, at address `addr`."""
    return in_function(text, fn, f"{line} ;\n", f"{line} ;\n        /*{addr:04x}*/                   {insn} ;\n")


@pytest.fixture(autouse=True)
def _list_everything(monkeypatch):
    monkeypatch.setattr(SG, "MAX_LISTED", 10_000)


# ---- the verdicts ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name,toolkit", [("plain", "12.9.1"), ("plain13", "13.0.1")])
def test_a_build_without_flags_passes_with_every_flagged_instruction_in_a_pinned_sequence(name, toolkit):
    v = verdict(name)
    assert v["toolkit"] == toolkit and v["unpinned"] is None
    assert v["offending_n"] == 0 and v["merged_reads_n"] == 0 and v["unmatched_calls"] == []
    assert v["covered"] == PLAIN_COVERED and v["flagged"] == sum(PLAIN_COVERED.values())


@pytest.mark.parametrize("name,flagged,offending_n", [("fast_math", 55, 42), ("ftz", 94, 81)])
def test_fast_math_and_ftz_builds_fail_on_every_fp32_kernel(name, flagged, offending_n):
    v = verdict(name)
    assert v["flagged"] == flagged and v["offending_n"] == offending_n
    assert {o["function"] for o in v["offending"]} == FP32_KERNELS
    assert v["covered"] == INTEGER_COVERED
    assert ("arith_k", "FADD.FTZ R0, R2, -R5") in offending(v)


def test_the_approximate_forms_fail_though_they_carry_no_ftz():
    """-prec-div=false -prec-sqrt=false: the bare MUFU the .FTZ rule alone would miss; __fdiv_rn and __fsqrt_rn stay exact."""
    v = verdict("prec")
    assert offending(v) == {("sqrt_k", "MUFU.SQRT R7, R0"), ("cdiv_k", "MUFU.RCP R0, R0"), ("div_k", "MUFU.RCP R9, R8")}
    assert "rho_k" not in {o["function"] for o in v["offending"]}


def test_every_refused_kernel_fails():
    """`ftz_refused.cu`, built without flags: __frsqrt_rn's two bare MUFU.RSQ (no .FTZ at default flags, so no entry), the
    approximate intrinsics, .ftz on add, min, setp and cvt, and a division whose slow path ptxas specialized."""
    v = verdict("refused")
    assert v["covered"] == {} and v["offending_n"] == v["flagged"] == 18
    assert {o["function"] for o in v["offending"]} == {"frsqrt_k", "rsqrtf_k", "expf_k", "fdividef_k", "sqrt_approx_k",
                                                       "add_ftz_k", "min_ftz_k", "setp_ftz_k", "cvt_ftz_k", "cdiv1_k"}
    assert [SG.opcode(i)[:2] for f, i in sorted(offending(v)) if f == "frsqrt_k"] == [["MUFU", "RSQ"]] * 2
    assert any("isn't a pinned one" in o.get("why", "") for o in v["offending"] if o["function"] == "cdiv1_k")


@pytest.mark.parametrize("text", ["FADD.FTZ R0, R2, R3", "@!P0 FFMA.FTZ R0, R1, R2, R3", "FFMA.RZ.FTZ R0, R1, R2, R3",
                                  "FADD32I.FTZ R0, R1, 0x3f800000", "FSETP.GTU.FTZ.AND P0, PT, |R0|, +INF , PT",
                                  "FMNMX.FTZ R0, R1, R2, !PT", "F2I.FTZ.U32.TRUNC.NTZ R5, R4", "F2F.FTZ.F32.F32 R0, R1",
                                  "HADD2.FTZ R0, R1, R2", "{ FMUL.FTZ R0, R1, R2", "MUFU.RCP R0, R1", "MUFU.RSQ R0, R1",
                                  "MUFU.SQRT R0, R1", "MUFU.EX2 R0, R1", "MUFU.LG2 R0, R1", "MUFU.SIN R0, R1",
                                  "MUFU.COS R0, R1", "MUFU.TANH R0, R1", "@P1 MUFU.RCP R0, R1"])
def test_every_ftz_and_every_fp32_mufu_is_flagged(text):
    assert SG.flagged(SG.opcode(text))


@pytest.mark.parametrize("text", ["FADD R0, R2, R3", "FFMA.RZ R0, R1, R2, R3", "FSETP.GTU.AND P0, PT, |R0|, +INF , PT",
                                  "F2I.U32.TRUNC.NTZ R5, R4", "DADD R0, R2, R4", "MUFU.RCP64H R1, R3",
                                  "MUFU.RSQ64H R1, R3", "I2F.U32.RP R0, R5"])
def test_the_rest_is_not(text):
    assert not SG.flagged(SG.opcode(text))


# ---- the toolkit key ------------------------------------------------------------------------------------------------------


def note(**kw) -> list[dict]:
    return [{**NOTES["plain"][0], **kw}]


@pytest.mark.parametrize("notes,why", [
    ([], "no ptxas record"),
    (note(version="Cuda compilation tools, release 12.8, V12.8.93"), "no pinned toolkit built it"),
    (note(branch="Build cuda_12.9.r12.9/compiler.00000000_0"), "no pinned toolkit built it"),
    (note(args="-arch sm_120a -m 64 -O2"), "with unpinned arguments '-arch sm_120a -m 64 -O2'"),
    (note(args="-arch sm_100a -m 64"), "with unpinned arguments"),
    (NOTES["plain"] + [{**NOTES["plain13"][0], "tool": "nvlink"}], "no pinned toolkit built it"),
])
def test_a_cubin_from_any_other_ptxas_exempts_nothing(notes, why):
    v = verdict("plain", notes=notes)
    assert v["toolkit"] is None and why in v["unpinned"]
    assert v["covered"] == {} and v["offending_n"] == v["flagged"] == sum(PLAIN_COVERED.values())


def test_one_ptxas_build_may_have_an_entry_per_target():
    """An entry for another target of the same ptxas (13.0.1-sm_90a beside 13.0.1) is found past the one whose arguments
    don't match, and exempts only with its own numeric run."""
    p = pins()
    p["toolkits"]["12.9.1-sm_90a"] = {**p["toolkits"]["12.9.1"], "args": ["-arch sm_90a -m 64", "-arch sm_90a -m 64 -fmad false"]}
    sm90 = note(args="-arch sm_90a -m 64 -fmad false")
    assert SG.toolkit_of(sm90, p) == ("12.9.1-sm_90a", None) and SG.toolkit_of(NOTES["plain"], p) == ("12.9.1", None)
    assert SG.toolkit_of(note(args="-arch sm_100a -m 64"), p) == (None, "ptxas Cuda compilation tools, release 12.9, V12.9.86 "
                                                                        "with unpinned arguments '-arch sm_100a -m 64'")
    p["toolkits"]["12.9.1-sm_90a"]["numeric"] = None
    assert SG.toolkit_of(sm90, p) == (None, "ptxas Cuda compilation tools, release 12.9, V12.9.86 is pinned without a numeric run")


def test_ptxas_verbose_flag_is_not_an_argument_that_matters():
    assert verdict("plain", notes=note(args="-v -arch sm_120a -m 64"))["offending_n"] == 0
    assert verdict("plain", notes=note(args="-arch sm_120a -m 64 -fmad false"))["offending_n"] == 0


def test_a_toolkit_pinned_without_a_numeric_run_exempts_nothing():
    v = verdict("plain", p=pins(numeric=None))
    assert v["toolkit"] is None and v["unpinned"] == "ptxas Cuda compilation tools, release 12.9, V12.9.86 is pinned " \
                                                     "without a numeric run"
    assert v["offending_n"] == v["flagged"]


def test_a_subroutine_is_exempt_only_as_its_own_toolkit_builds_it():
    """ptxas 13.0 builds __cuda_sm20_div_u16 two ways, as its call site lets it (ieee_pin.cu's u16 and u16_call): the one
    12.9 builds too, and one that converts the dividend as a u32, which 12.9.1 exempts nowhere."""
    v = verdict("plain13", notes=NOTES["plain"])
    assert {o["function"] for o in v["offending"]} == {"udiv16_k"}
    assert all("__cuda_sm20_div_u16's body" in o["why"] for o in v["offending"])
    assert verdict("plain", notes=NOTES["plain13"])["offending_n"] == 0
    tk = SG.load_pins()["toolkits"]
    assert tk["12.9.1"]["subroutines"]["__cuda_sm20_div_u16"] == ["2ada2310b05d870e"]
    assert tk["13.0.1"]["subroutines"]["__cuda_sm20_div_u16"] == ["2ada2310b05d870e", "f28777dafff5f987"]


# ---- the sequences --------------------------------------------------------------------------------------------------------

SQRT_K = {("sqrt_k", "MUFU.RSQ R5, R0"), ("sqrt_k", "FMUL.FTZ R3, R0, R5"), ("sqrt_k", "FMUL.FTZ R5, R5, 0.5")}


def test_a_fast_block_with_an_extra_instruction_is_not_exempt():
    v = verdict("plain", text=after(listing("plain"), "sqrt_k", "FMUL.FTZ R5, R5, 0.5", "FADD.FTZ R9, R9, R9", 0xf8))
    assert offending(v) == SQRT_K | {("sqrt_k", "FADD.FTZ R9, R9, R9")}
    assert any("sqrt: the fast block isn't four instructions" in m for m in v["unmatched_calls"])


def test_a_seed_read_outside_its_sequence_is_not_exempt():
    v = verdict("plain", text=after(listing("plain"), "sqrt_k", "ISETP.GT.U32.AND P0, PT, R4, 0x727fffff, PT",
                                    "STG.E desc[UR4][R2.64], R5", 0x98))
    assert offending(v) == SQRT_K
    assert any("R5 from 0x0080 is read outside the sequence at 0x0098" in m for m in v["unmatched_calls"])


def test_a_slow_path_of_another_name_is_not_exempt_nor_the_fast_path_around_it():
    text = listing("plain").replace("__cuda_sm20_sqrt_rn_f32_slowpath", "__cuda_sm20_sqrt_rn_ftz_f32_slowpath")
    v = verdict("plain", text=text)
    assert "sqrt-fast-path" not in v["covered"] and "__cuda_sm20_sqrt_rn_f32_slowpath" not in v["covered"]
    assert {o["function"] for o in v["offending"]} == {"sqrt_k", "rho_k"}


def test_a_fast_path_around_a_slow_path_that_isnt_its_pin_is_not_exempt():
    text = in_function(listing("plain"), "rcp_k", "@P0 FFMA R6, R5, R2, -1", "@P0 FFMA R6, R5, R2, -2")
    v = verdict("plain", text=text)
    assert {o["function"] for o in v["offending"]} == {"rcp_k"} and "rcp-fast-path" not in v["covered"]
    assert any("the __cuda_sm20_rcp_rn_f32_slowpath it calls isn't the pinned body" in m for m in v["unmatched_calls"])


def test_the_div_chain_must_be_div_rn():
    text = in_function(listing("plain"), "div_k", "FFMA R10, -R3, R8, R0", "FFMA R10, -R3, R8, R3")
    v = verdict("plain", text=text)
    assert ("div_k", "MUFU.RCP R8, R3") in offending(v) and v["covered"]["div-fast-path"] == 3


def test_a_constant_numerator_must_be_the_constant_fchk_checks():
    text = in_function(listing("plain"), "cdiv_k", "UMOV UR6, 0x40400000", "UMOV UR6, 0x40800000")
    v = verdict("plain", text=text)
    assert offending(v) == {("cdiv_k", "MUFU.RCP R0, R3")}
    assert "no FCHK $P, $n, $b in the block whose $n holds $a" in v["offending"][0]["why"]


PIPE = ("ISETP.GT.U32.AND P0, PT, R4, 0x727fffff, PT ;\n"
        "        /*0094*/               @P1 BRA `(.L_x_90) ;\n"
        "        /*0098*/                   {arm} ;\n"
        ".L_x_90:\n")


def pipelined(arm: str, text: str | None = None) -> str:
    """sqrt_k with a branch between its range check (and seed) and the branch to the slow path, as ptxas schedules a loop
    that records the last iteration's result there (ieee_pin.cu's sqrt_all)."""
    return in_function(listing("plain") if text is None else text, "sqrt_k",
                       "ISETP.GT.U32.AND P0, PT, R4, 0x727fffff, PT ;\n", PIPE.format(arm=arm))


def test_a_range_check_and_seed_in_an_earlier_block_are_found_on_every_path():
    assert verdict("plain", text=pipelined("STG.E desc[UR4][R2.64], R7"))["covered"] == PLAIN_COVERED


@pytest.mark.parametrize("arm,why", [("MOV R0, R7", "$x changes between the range check and the branch"),
                                     ("MOV R5, R7", "the fast block's $r isn't MUFU.RSQ $r, $x"),
                                     ("ISETP.NE.AND P0, PT, R7, RZ, PT", "the range check's $P isn't")])
def test_a_write_on_one_path_between_the_check_and_the_branch_is_not_exempt(arm, why):
    v = verdict("plain", text=pipelined(arm))
    assert offending(v) == SQRT_K and any(why in m for m in v["unmatched_calls"])


def test_a_path_to_the_branch_that_misses_the_range_check_is_not_exempt():
    text = after(pipelined("STG.E desc[UR4][R2.64], R7"), "sqrt_k", "LDG.E R0, desc[UR4][R2.64]", "@P2 BRA `(.L_x_90)",
                 0x58)
    v = verdict("plain", text=text)
    assert offending(v) == SQRT_K and any("the range check's $P isn't" in m for m in v["unmatched_calls"])


def test_integer_division_seed_must_round_up_by_two_ulps():
    text = in_function(listing("plain"), "udiv_k", "IADD3 R6, PT, PT, R0, 0xffffffe, RZ", "IADD3 R6, PT, PT, R0, 0xfffffff, RZ")
    v = verdict("plain", text=text)
    assert offending(v) == {("udiv_k", "MUFU.RCP R0, R0"), ("udiv_k", "F2I.FTZ.U32.TRUNC.NTZ R7, R6")}
    assert "the seed isn't rounded up by IADD3 +2 ulp" in v["offending"][0]["why"]


MERGE = """\t.section\t.text.merge_k,"ax",@progbits
merge_k:
        /*0000*/                   I2F.U32.RP R0, R5 ;
        /*0010*/                   MUFU.RCP R0, R0 ;
        /*0020*/                   IADD3 R6, PT, PT, R0, 0xffffffe, RZ ;
        /*0030*/                   F2I.FTZ.U32.TRUNC.NTZ R7, R6 ;
        /*0040*/               @P0 BRA `(.L_x_1) ;
        /*0050*/                   MOV R0, 0x5 ;
.L_x_1:
        /*0060*/                   STG.E desc[UR4][R2.64], R0 ;
        /*0070*/                   EXIT ;
"""


def test_a_read_the_gate_cant_tell_from_a_merge_passes_and_is_listed():
    """The seed reaches the store only when P0 holds; another write reaches it otherwise. The gate can't show that path
    infeasible (hash_leaf's 22 such reads are: its tail reads R10 only when a count it skipped the loop for is 1-3), so it
    passes the read and lists it; without the other write the read is direct, and refused."""
    v = SG.cubin_verdict(SG.parse_nvdisasm(MERGE), NOTES["plain"], pins())
    assert v["offending_n"] == 0 and v["covered"] == {"int-seed": 2}
    assert v["merged_reads"] == [{"function": "merge_k", "sequence": "int-seed", "value": "R0 from 0x0010",
                                  "reader": "0x0060 STG.E desc[UR4][R2.64], R0"}]
    v = SG.cubin_verdict(SG.parse_nvdisasm(MERGE.replace("MOV R0, 0x5", "MOV R9, 0x5")), NOTES["plain"], pins())
    assert v["offending_n"] == 2 and "the seed has no one next step" in v["offending"][0]["why"]


def test_moves_after_a_call_may_be_either_spelling_and_nothing_else():
    n = listing("plain")
    same = in_function(n, "cdiv_k", "MOV R4, R9 ;", "IMAD.MOV.U32 R4, RZ, RZ, R9 ;")
    assert verdict("plain", text=same)["covered"] == PLAIN_COVERED
    v = verdict("plain", text=in_function(n, "cdiv_k", "MOV R4, R9 ;", "IMAD R4, R9, R3, RZ ;"))
    assert v["covered"]["div-fast-path"] == 3 and {o["function"] for o in v["offending"]} == {"cdiv_k"}


def test_a_signature_ignores_how_an_immediate_is_loaded_and_nothing_else():
    n = listing("plain")
    hfma = "HFMA2 R3, -RZ, RZ, 15, 0"
    assert verdict("plain", text=in_function(n, "udiv16_k", hfma, "MOV R3, 0x4b800000"))["covered"] == PLAIN_COVERED
    v = verdict("plain", text=in_function(n, "udiv16_k", hfma, "MOV R3, 0x4b800001"))
    assert {o["function"] for o in v["offending"]} == {"udiv16_k"} and "__cuda_sm20_div_u16's body" in v["offending"][0]["why"]


def test_a_sections_flags_line_keeps_its_function():
    """nvdisasm writes `.sectionflags` after `.section` for a kernel that uses barriers (the sm_90a plainq)."""
    flagged = "".join(ln + ('\t.sectionflags\t@"SHF_BARRIERS=1"\n' if ln.lstrip().startswith(".section\t.text.") else "")
                      for ln in listing("plain").splitlines(keepends=True))
    assert flagged.count(".sectionflags") >= 3
    assert verdict("plain", text=flagged) == verdict("plain")


# ---- the pins -------------------------------------------------------------------------------------------------------------


@pytest.fixture
def records(tmp_path, monkeypatch):
    """`pin_record` of a listing: a stand-in cubin naming it, which `_disassemble` reads back as the listing and its notes."""
    def disassemble(cubins, tools):
        names = [c[off:off + size].decode() for c in cubins for _, off, size in [SG._sections(c)[".text.k"]]]
        return [(SG.parse_nvdisasm(listing(n)), NOTES[n]) for n in names]

    monkeypatch.setattr(SG, "_disassemble", disassemble)

    def record(name: str) -> list[dict]:
        p = tmp_path / f"{name}.cubin"
        p.write_bytes(elf(SG.EM_CUDA, {".text.k": name.encode()}))
        return SG.pin_record([p], {})
    return record


def test_the_probe_pins_a_part_of_each_entry(tmp_path, records):
    """The pins come from ieee_pin.cu's builds (ieee_pin.py checks they are what those builds pin); the probe's bodies are
    among them, and pinning the probe alone gives the same entry with those bodies."""
    shipped = SG.load_pins()
    out = tmp_path / "pins.json"
    for tk, name in (("12.9.1", "plain"), ("13.0.1", "plain13")):
        entry = SG.update_pins(records(name), tk, out)
        want = shipped["toolkits"][tk]
        assert entry == {**want, "args": ["-arch sm_120a -m 64"], "numeric": None, "subroutines": entry["subroutines"]}
        assert entry["subroutines"].keys() == want["subroutines"].keys()
        assert all(set(s) <= set(want["subroutines"][n]) for n, s in entry["subroutines"].items())
    assert json.loads(out.read_text())["signatures"].items() <= shipped["signatures"].items()


def test_a_numeric_run_survives_a_repin_only_while_the_subroutines_are_unchanged(tmp_path, records):
    out = tmp_path / "pins.json"
    rec = records("plain")
    SG.update_pins(rec, "12.9.1", out)
    p = json.loads(out.read_text())
    p["toolkits"]["12.9.1"]["numeric"] = TEST_RUN
    out.write_text(json.dumps(p))
    assert SG.update_pins(rec, "12.9.1", out)["numeric"] == TEST_RUN
    div_u16 = records("plain13")[0]["subroutines"]["__cuda_sm20_div_u16"]
    other = [{**r, "subroutines": {**r["subroutines"], "__cuda_sm20_div_u16": div_u16}} for r in rec]
    assert SG.update_pins(other, "12.9.1", out)["numeric"] is None


def test_pinning_refuses_two_ptxas_builds_and_keeps_every_body_of_a_subroutine(tmp_path, records):
    a, b = records("plain"), records("plain13")
    with pytest.raises(SG.SassGateFailed, match="the builds name 2 ptxas builds, not one"):
        SG.update_pins(a + b, "x", tmp_path / "pins.json")
    entry = SG.update_pins(a + [{**r, "tool_notes": NOTES["plain"]} for r in b], "x", tmp_path / "pins.json")
    assert entry["subroutines"]["__cuda_sm20_div_u16"] == ["2ada2310b05d870e", "f28777dafff5f987"]


def test_the_shipped_pins_are_consistent():
    """Every pinned subroutine has its signature's blocks, and every signature is some toolkit's."""
    p = SG.load_pins()
    live = {sid for tk in p["toolkits"].values() for sids in tk["subroutines"].values() for sid in sids}
    assert live == set(p["signatures"])
    assert all(SG.signature_id(p["signatures"][sid]) == sid for sid in live)


def test_pins_of_another_format_are_refused(tmp_path):
    """v1 named one signature per subroutine as a string, which `in` would read as a substring."""
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({**SG.load_pins(), "version": "sass-pins/v1"}))
    with pytest.raises(SG.SassGateFailed, match="is 'sass-pins/v1', not sass-pins/v2"):
        SG.load_pins(p)


# ---- sm_90a ---------------------------------------------------------------------------------------------------------------
# `plain90` and `refused90` are 13.0.1's sm_90a builds of the probes, by the sm_90a pin's ptxas. The shipped
# 13.0.1-sm_90a entry is ieee_pin.cu's, from its numeric run on the H200.


@pytest.fixture
def pins90(tmp_path, records) -> dict:
    """The probe's sm_90a build pinned as ``--pin --toolkit 13.0.1-sm_90a`` writes it, with a test numeric run."""
    entry = SG.update_pins(records("plain90"), "13.0.1-sm_90a", tmp_path / "pins90.json")
    return {"version": SG.PINS_VERSION, "toolkits": {"13.0.1-sm_90a": {**entry, "numeric": TEST_RUN}}}


def test_an_sm90a_build_without_flags_passes_under_its_own_entry_and_no_other(pins90):
    """ptxas for sm_90a spells the range checks' and the seeds' adds VIADD or IADD3 R, R, imm, RZ, and moves the constant
    numerator from a uniform register with IMAD.U32: every sequence is still found, as at sm_120a."""
    v = verdict("plain90", p=pins90)
    assert v["toolkit"] == "13.0.1-sm_90a" and v["offending_n"] == 0 and v["merged_reads_n"] == 0
    assert v["unmatched_calls"] == [] and v["covered"] == PLAIN_COVERED and v["flagged"] == sum(PLAIN_COVERED.values())
    pins90["toolkits"]["13.0.1-sm_90a"]["numeric"] = None
    assert verdict("plain90", p=pins90)["offending_n"] == v["flagged"]


def test_the_shipped_sm90a_entry_exempts_only_the_bodies_its_numeric_run_tested():
    """The shipped entry is found for an sm_90a build of its ptxas and exempts the seeds and the bodies ieee_pin.cu's builds
    pinned: sqrt_k's sqrt calls reach the body units_sqrt tests, and rho_k's a __cuda_sm20_sqrt_rn_f32_slowpath that ptxas
    specialized otherwise, which no numeric run tested, so they stay refused, and so does every refused kernel but
    __expf's, whose one flagged instruction is a bare MUFU.EX2 between IEEE FMULs, which the entry excepts."""
    entry = SG.load_pins()["toolkits"]["13.0.1-sm_90a"]
    assert entry["args"] == ["-arch sm_90a -m 64", "-arch sm_90a -m 64 -fmad false"]
    assert entry["numeric"]["run"] == "r20261026-085046-75c4"
    v = verdict("plain90")
    assert v["toolkit"] == "13.0.1-sm_90a" and v["unpinned"] is None and v["covered"]["int-seed"] == 12
    assert [c.split(" ")[0] for c in v["unmatched_calls"]] == ["rho_k"] and v["covered"]["sqrt-fast-path"] == 3
    assert "sqrt_k" not in {o["function"] for o in v["offending"]} and 0 < v["offending_n"] < v["flagged"]
    r = verdict("refused90")
    assert r["toolkit"] == "13.0.1-sm_90a" and r["covered"] == {"MUFU.EX2": 1} and r["flagged"] == 18
    assert r["offending_n"] == 17 and "expf_k" not in {o["function"] for o in r["offending"]}


def test_every_refused_kernel_fails_at_sm90a_too(pins90):
    v = verdict("refused90", p=pins90)
    assert v["covered"] == {} and v["offending_n"] == v["flagged"] == 18
    assert {o["function"] for o in v["offending"]} == {"frsqrt_k", "rsqrtf_k", "expf_k", "fdividef_k", "sqrt_approx_k",
                                                       "add_ftz_k", "min_ftz_k", "setp_ftz_k", "cvt_ftz_k", "cdiv1_k"}


RCP90 = {("rcp_k", "MUFU.RCP R7, R0"), ("rcp_k", "FADD.FTZ R2, -R2, -RZ")}
SQRT90 = {("sqrt_k", "MUFU.RSQ R5, R0"), ("sqrt_k", "FMUL.FTZ R3, R0, R5"), ("sqrt_k", "FMUL.FTZ R5, R5, 0.5")}


@pytest.mark.parametrize("fn,old,new,refused", [
    ("rcp_k", "VIADD R2, R0, 0x1800000", "VIADD R2, R0, 0x1800001", RCP90),
    ("rcp_k", "VIADD R2, R0, 0x1800000", "VIADD R2, R3, 0x1800000", RCP90),
    ("rcp_k", "VIADD R2, R0, 0x1800000", "IADD3 R2, R0, 0x1800000, R3", RCP90),
    ("sqrt_k", "IADD3 R4, R0, -0xd000000, RZ", "IADD3 R4, R0, -0xc000000, RZ", SQRT90),
    ("sqrt_k", "IADD3 R4, R0, -0xd000000, RZ", "IADD3 R4, R0, -0xd000000, R7", SQRT90),
    ("sqrt_k", "IADD3 R4, R0, -0xd000000, RZ", "VIADD R4, R0, -0xd000000", SQRT90),
    ("udiv_k", "IADD3 R6, R0, 0xffffffe, RZ", "IADD3 R6, R0, 0xfffffff, RZ",
     {("udiv_k", "MUFU.RCP R0, R0"), ("udiv_k", "F2I.FTZ.U32.TRUNC.NTZ R7, R6")}),
    ("udiv_k", "IADD3 R6, R0, 0xffffffe, RZ", "IADD3 R6, R0, 0xffffffe, R11",
     {("udiv_k", "MUFU.RCP R0, R0"), ("udiv_k", "F2I.FTZ.U32.TRUNC.NTZ R7, R6")}),
    ("srem_k", "VIADD R6, R9, 0xffffffe", "VIADD R6, R9, 0xfffffff",
     {("srem_k", "MUFU.RCP R9, R9"), ("srem_k", "F2I.FTZ.U32.TRUNC.NTZ R7, R6")}),
    ("cdiv_k", "UMOV UR6, 0x40400000", "UMOV UR6, 0x40800000", {("cdiv_k", "MUFU.RCP R0, R3")}),
    ("cdiv_k", "IMAD.U32 R8, RZ, RZ, UR6", "IMAD.U32 R8, R3, RZ, UR6", {("cdiv_k", "MUFU.RCP R0, R3")}),
    ("cdiv_k", "IMAD.U32 R8, RZ, RZ, UR6", "IMAD.U32 R8, RZ, RZ, UR7", {("cdiv_k", "MUFU.RCP R0, R3")}),
])
def test_an_sm90a_sequence_with_another_operand_or_constant_is_not_exempt(pins90, fn, old, new, refused):
    """A range check, a seed's +2 ulp or a constant numerator's move that isn't exactly what sm_90a's ptxas writes: the
    wrong constant, another register, a third addend, or a negative immediate VIADD (nvdisasm writes 0xf3000000)."""
    v = verdict("plain90", text=in_function(listing("plain90"), fn, f"{old} ;", f"{new} ;"), p=pins90)
    assert offending(v) == refused


@pytest.mark.parametrize("f2f,exempt", [("F2F.F32.F64 R2, R10", True), ("F2F.F64.F32 R2, R10", False),
                                        ("F2F.F64.F64 R2, R10", False)])
def test_a_conversion_may_write_only_its_destination_types_width(pins90, f2f, exempt):
    """sm_90a's ptxas schedules an F2F.F32.F64 inside ieee_pin.cu's cdiv_all div.rn fast path, its
    destination the register before the divisor: an F32 result writes one register, an F64 result two."""
    v = verdict("plain90", text=after(listing("plain90"), "div_k", "FCHK P0, R0, R3", f2f, 0xb8), p=pins90)
    if exempt:
        assert v["offending_n"] == 0
    else:
        assert offending(v) == {("div_k", "MUFU.RCP R8, R3")}
        assert "$b changes between its readers" in v["offending"][0]["why"]


# ---- exceptions -----------------------------------------------------------------------------------------------------------
# `units_softmax90` is units_softmax (benchmarks/torch_gpu/units/units.cu at 3cc8d5b6a6) as 13.0.1's ptxas built it for
# sm_90a: the cubin an H200 run ran (sha256 81d539e9...).
# Its 32 MUFU.EX2 are ex2.approx.ftz.f32, one per element of a row; the same run's units_sweep held that instruction to
# MufuEx2Ftz on all 2^32 operands.

def softmax90(text: str | None = None, notes: list | None = None, p: dict | None = None) -> dict:
    return SG.cubin_verdict(SG.parse_nvdisasm(listing("units_softmax90") if text is None else text),
                            NOTES["units_softmax90"] if notes is None else notes, SG.load_pins() if p is None else p)


def ex2s(v: dict) -> list[dict]:
    return [o for o in v["offending"] if SG.opcode(o["instruction"])[:2] == ["MUFU", "EX2"]]


def test_units_softmax_mufu_ex2_passes_under_the_shipped_sm90a_exception():
    """Every MUFU.EX2 of units_softmax passes under the shipped 13.0.1-sm_90a entry; its r = 1/s div.rn passes as a fast
    path around the slow path ieee_pin.cu's units_soft tests, which the exception doesn't touch."""
    v = softmax90()
    assert v["toolkit"] == "13.0.1-sm_90a" and v["covered"] == UNITS_COVERED["units_softmax90"] and v["flagged"] == 44
    assert v["offending_n"] == 0


def test_without_the_exception_or_for_another_target_every_mufu_ex2_is_refused():
    p = SG.load_pins()
    del p["toolkits"]["13.0.1-sm_90a"]["exceptions"]
    v = softmax90(p=p)
    assert "MUFU.EX2" not in v["covered"] and len(ex2s(v)) == 32 == v["offending_n"]
    assert {o["why"] for o in ex2s(v)} == {"13.0.1-sm_90a lists no exception for MUFU.EX2"}
    sm120 = softmax90(notes=[{**NOTES["units_softmax90"][0], "args": "-v -arch sm_120a -m 64 -fmad false"}])
    assert sm120["toolkit"] == "13.0.1" and sm120["covered"] == {} and len(ex2s(sm120)) == 32
    assert {o["why"] for o in ex2s(sm120)} == {"13.0.1 lists no exception for MUFU.EX2"}


@pytest.mark.parametrize("new,why", [("@P5 MUFU.RSQ R35, R42", None), ("@P5 MUFU.LG2 R35, R42", None),
                                     ("@P5 MUFU.TANH R35, R42", None), ("@P5 MUFU.EX2.F16 R35, R42", None),
                                     ("@P5 MUFU.EX2 R35, -R42", "MUFU.EX2 is excepted only as"),
                                     ("@P5 MUFU.EX2 R35, |R42|", "MUFU.EX2 is excepted only as")])
def test_another_mufu_or_another_form_of_ex2_is_still_refused(new, why):
    """The exception is MUFU.EX2 of a register, the form the sweep measured: another MUFU in its place, or MUFU.EX2 with
    another type or an operand modifier, stays flagged while the other 31 pass."""
    v = softmax90(in_function(listing("units_softmax90"), "units_softmax", "@P5 MUFU.EX2 R35, R42", new))
    hit = [o for o in v["offending"] if o["instruction"] == new]
    assert v["covered"]["MUFU.EX2"] == 31 and len(hit) == 1 and v["offending_n"] == 1
    assert hit[0].get("why", "").startswith(why) if why else "why" not in hit[0]


def test_the_exception_names_the_definition_and_table_its_runs_measured():
    """The shipped exception is MufuEx2Ftz_v1 over the library's ex2 table, the evaluator the sweep's digests came from;
    another id or table is another Definition, which needs its own run. No other toolkit excepts anything."""
    from verity.ml.circuits import library
    from verity.ml.circuits.ops import mufu

    p = SG.load_pins()
    exc = p["toolkits"]["13.0.1-sm_90a"]["exceptions"]
    assert set(exc) == set(SG.EXCEPTABLE) == {"MUFU.EX2"}
    assert exc["MUFU.EX2"]["definition"].startswith(f"verity.ml.circuits.ops.mufu.MufuEx2Ftz ({mufu.MufuEx2Ftz.id}, ")
    assert library.table("ex2").sha512 in exc["MUFU.EX2"]["definition"] and len(mufu.table("ex2")) == 1 << 23
    assert exc["MUFU.EX2"]["numeric"]["runs"] == ["r20261026-015355-aec6", "r20261026-032225-3cee"]
    assert [tk for tk, rec in p["toolkits"].items() if rec.get("exceptions")] == ["13.0.1-sm_90a"]


# ---- units.cu at sm_90a ------------------------------------------------------------------------------------------
# `units_sweep90` and `units_rmsnorm90` are the units_sweep and units_rmsnorm sections of the listing `units_softmax90` comes
# from (of the cubin 81d539e9...), with its tool notes. Their fast paths reload a divisor or
# numerator from constant bank 0 in the call block, FCHK reads a copy of it, and units_sweep switches with BRX; rmsnorm's
# and softmax's slow paths are the bodies ieee_pin.cu's units_rms, units_sqrt and units_soft test.

UNITS_COVERED = {
    "units_sweep90": {"sqrt-fast-path": 3, "div-fast-path": 6, "MUFU.EX2": 4, "__cuda_sm20_sqrt_rn_f32_slowpath": 9,
                      "__cuda_sm3x_div_rn_noftz_f32_slowpath": 10},
    "units_rmsnorm90": {"div-fast-path": 2, "sqrt-fast-path": 3, "__cuda_sm20_sqrt_rn_f32_slowpath": 9,
                        "__cuda_sm3x_div_rn_noftz_f32_slowpath": 10},
    "units_softmax90": {"MUFU.EX2": 32, "div-fast-path": 1, "__cuda_sm3x_div_rn_noftz_f32_slowpath": 11}}
SWEEP_BRX = "BRX R2 -0x220" + " " * 59 + '(*"BRANCH_TARGETS .L_x_385,.L_x_386,.L_x_10"*)'


def units90(name: str, text: str | None = None) -> dict:
    return SG.cubin_verdict(SG.parse_nvdisasm(listing(name) if text is None else text), NOTES["units_softmax90"],
                            SG.load_pins())


def at(name: str, addr: int, old: str, new: str) -> str:
    """`name`'s listing with the instruction at `addr`, `old`, replaced by `new`."""
    text, pad = listing(name), f"/*{addr:04x}*/" + " " * 19
    assert text.count(pad + old + " ;") + text.count(pad + old + ";") == 1, (addr, old)
    return text.replace(pad + old, pad + new)


@pytest.mark.parametrize("name", sorted(UNITS_COVERED))
def test_kernels_units_pass_with_every_flagged_instruction_covered(name):
    v = units90(name)
    assert v["toolkit"] == "13.0.1-sm_90a" and v["offending_n"] == 0 and v["unmatched_calls"] == []
    assert v["covered"] == UNITS_COVERED[name] and v["merged_reads_n"] == 0


def test_a_call_block_may_reload_an_operand_from_any_register_holding_its_slot():
    """units_sweep's reload of the divisor c[0x0][0x220] through UR7 passes as well through UR8, which the kernel loaded
    from the same slot before its loop (whose calls don't write it)."""
    v = units90("units_sweep90", at("units_sweep90", 0x450, "IMAD.U32 R4, RZ, RZ, UR7", "IMAD.U32 R4, RZ, RZ, UR8"))
    assert v["offending_n"] == 0 and v["covered"] == UNITS_COVERED["units_sweep90"]


@pytest.mark.parametrize("name,fn,addr,old,new,rcp,why", [
    ("units_sweep90", "units_sweep", 0x3b0, "FCHK P0, R13, R10", "NOP", "MUFU.RCP R3, R10", "no FCHK"),
    ("units_sweep90", "units_sweep", 0x3b0, "FCHK P0, R13, R10", "FCHK P0, R13, R11", "MUFU.RCP R3, R10", "no FCHK"),
    ("units_sweep90", "units_sweep", 0x4b0, "IMAD.U32 R12, RZ, RZ, UR8", "IMAD.U32 R12, RZ, RZ, UR6", "MUFU.RCP R2, R13",
     "no FCHK"),
    ("units_rmsnorm90", "units_rmsnorm", 0x6310, "MOV R8, UR4", "MOV R8, UR5", "MUFU.RCP R0, R3", "no FCHK"),
    ("units_sweep90", "units_sweep", 0x420, "ULDC UR7, c[0x0][0x220]", "ULDC UR7, c[0x0][0x224]", "MUFU.RCP R3, R10",
     "isn't passed the fast path's operands"),
    ("units_sweep90", "units_sweep", 0x450, "IMAD.U32 R4, RZ, RZ, UR7", "IMAD.U32 R4, RZ, RZ, UR6", "MUFU.RCP R3, R10",
     "isn't passed the fast path's operands"),
    ("units_sweep90", "units_sweep", 0x430, "IMAD.MOV.U32 R3, RZ, RZ, R13", "IMAD.MOV.U32 R3, RZ, RZ, R10",
     "MUFU.RCP R3, R10", "isn't passed the fast path's operands"),
    ("units_rmsnorm90", "units_rmsnorm", 0x61b0, "MOV R0, R15", "MOV R0, R6", "MUFU.RCP R0, R8",
     "isn't passed the fast path's operands"),
    ("units_rmsnorm90", "units_rmsnorm", 0x61a0, "ULDC UR4, c[0x0][0x254]", "ULDC UR4, c[0x0][0x258]", "MUFU.RCP R0, R8",
     "isn't passed the fast path's operands"),
    ("units_softmax90", "units_softmax", 0x2b10, "ULDC UR4, c[0x0][0x23c]", "ULDC UR4, c[0x0][0x240]", "MUFU.RCP R0, R3",
     "isn't passed the fast path's operands"),
    ("units_sweep90", "units_sweep", 0x440, "MOV R12, 0x470", "MOV R12, 0x480", "MUFU.RCP R3, R10",
     "isn't passed its return address"),
    ("units_sweep90", "units_sweep", 0x400, "FFMA R4, R2, R3, R4", "FFMA.RZ R4, R2, R3, R4", "MUFU.RCP R3, R10",
     "no FFMA $z"),
    ("units_sweep90", "units_sweep", 0x3c0, "FFMA R2, R3, -R10, 1", "FFMA.RP R2, R3, -R10, 1", "MUFU.RCP R3, R10",
     "no FFMA $t"),
])
def test_each_reload_and_copy_breaks_under_a_mutation(name, fn, addr, old, new, rcp, why):
    """The shapes the gate takes for units.cu are tight: with its check dropped, its FCHK reading another divisor or a copy
    of another numerator, the call passing another slot, register or return address, or a step of the chain rounding
    other than to nearest, the division is refused and nothing else is."""
    v = units90(name, at(name, addr, old, new))
    assert offending(v) == {(fn, rcp)} and why in v["offending"][0]["why"]


@pytest.mark.parametrize("name,fn,edits,rcp,why", [
    ("units_sweep90", "units_sweep", [(0x430, "IMAD.MOV.U32 R3, RZ, RZ, R13", "IMAD.U32 R3, RZ, RZ, UR7"),
                                      (0x450, "IMAD.U32 R4, RZ, RZ, UR7", "IMAD.MOV.U32 R4, RZ, RZ, R13")],
     "MUFU.RCP R3, R10", "in their roles"),
    ("units_rmsnorm90", "units_rmsnorm", [(0x61b0, "MOV R0, R15", "IMAD.U32 R0, RZ, RZ, UR4"),
                                          (0x61c0, "IMAD.U32 R3, RZ, RZ, UR4", "MOV R3, R15")],
     "MUFU.RCP R0, R8", "in their roles"),
    ("units_sweep90", "units_sweep", [(0x3b0, "FCHK P0, R13, R10", "FCHK P0, R10, R13")], "MUFU.RCP R3, R10", "no FCHK"),
    ("units_rmsnorm90", "units_rmsnorm", [(0x6130, "FCHK P0, R15, R8", "FCHK P0, R8, R15")], "MUFU.RCP R0, R8", "no FCHK"),
])
def test_a_div_call_or_fchk_with_its_operands_swapped_is_refused(name, fn, edits, rcp, why):
    """The call passes the numerator and the divisor each in its role: the body's divisor is the input its MUFU.RCP takes
    (`_sources`), so a call block that moves the fast path's a and b into each other's registers fails, as does an FCHK
    that reads them swapped."""
    text = listing(name)
    for addr, old, new in edits:
        pad = f"/*{addr:04x}*/" + " " * 19
        assert text.count(pad + old + " ;") == 1, (addr, old)
        text = text.replace(pad + old, pad + new)
    v = units90(name, text)
    assert offending(v) == {(fn, rcp)} and why in v["offending"][0]["why"]


def test_the_div_bodies_take_their_divisor_from_one_input():
    """`_sources` of each units.cu div body's MUFU.RCP operand is exactly the register or slot the call's divisor goes in."""
    want = {"units_sweep90": {("in", "R4")}, "units_rmsnorm90": {("in", "R3")}, "units_softmax90": {("in", "R0")}}
    for name, divisor in want.items():
        fns = [SG.Function(d) for d in SG.parse_nvdisasm(listing(name))]
        (fn, lo, hi), = [(f, lo, hi) for f in fns for n, lo, hi in f.subroutines.values() if "div_rn" in n]
        rcp, = [t for t in range(lo, hi) if fn.insns[t].name == "MUFU.RCP"]
        assert SG._sources(fn, lo, hi, rcp, fn.insns[rcp].args[1]) == divisor, name


@pytest.mark.parametrize("old,new,why", [
    ("FFMA R3, -R4, R4, R13", "FFMA.RZ R3, -R4, R4, R13", "the fast block at 0x02a0 isn't sqrt.rn's"),
    (SWEEP_BRX, SWEEP_BRX.replace(".L_x_10", ".L_x_10,.L_x_8"), "another branch enters the fast block"),
    (SWEEP_BRX, "BRX R2 -0x220", "the fast block's $r isn't MUFU.RSQ"),
])
def test_the_sweeps_sqrt_breaks_under_a_mutation_or_a_brx_the_gate_cant_follow(old, new, why):
    """units_sweep's sqrt.rn sits behind a BRX: it matches through the BRX's BRANCH_TARGETS, and not with a fast block
    that rounds toward zero, a BRX that may enter its fast block, or a BRX without its targets (then every division
    after it fails too)."""
    addr = 0x210 if old == SWEEP_BRX else 0x2c0
    v = units90("units_sweep90", at("units_sweep90", addr, old, new))
    assert ("units_sweep", "MUFU.RSQ R2, R13") in offending(v) and any(why in m for m in v["unmatched_calls"])
    assert v["offending_n"] == (9 if new == "BRX R2 -0x220" else 3)


def test_the_sqrt_call_must_pass_the_fast_paths_x():
    text = listing("units_sweep90").replace("        /*0270*/                   MOV R16, 0x290 ;\n",
                                            "        /*0270*/                   MOV R16, 0x290 ;\n"
                                            "        /*0278*/                   IMAD.MOV.U32 R13, RZ, RZ, R2 ;\n")
    v = units90("units_sweep90", text)
    assert ("units_sweep", "MUFU.RSQ R2, R13") in offending(v)
    assert v["unmatched_calls"] == ["units_sweep 0x0280 sqrt: the slow path isn't passed the fast path's operands"]


def test_instructions_name_what_they_read_and_where_they_branch():
    """A fixed slot's load keys each register by its own 32-bit slot, and nothing else is a slot; P2R reads only the
    predicates its mask selects; a BRX branches to its BRANCH_TARGETS, and without them is indirect."""
    pair = SG.Insn(0, "ULDC.64 UR4, c[0x0][0x268]")
    assert [SG._slot(pair, r) for r in ("UR4", "UR5", "UR6")] == ["c[0x0][0x268]", "c[0x0][0x26c]", None]
    assert SG._slot(SG.Insn(0, "LDC R7, c[0x0][0x23c]"), "R7") == "c[0x0][0x23c]"
    for text in ("LDC R2, c[0x0][R3+0x10]", "LDC.U8 R2, c[0x0][0x10]", "LDC R2, c[0x3][0x10]", "LDC.128 R4, c[0x0][0x10]"):
        assert SG._slot(SG.Insn(0, text), "R2" if "R2" in text else "R4") is None, text
    assert SG.Insn(0, "P2R R41, PR, RZ, 0x40").reads == {"P6"}
    assert SG.Insn(0, "P2R.B1 R41, PR, RZ, 0x40").reads == set(SG._PREDS)
    brx = SG.Insn(0x210, SWEEP_BRX)
    assert brx.targets == [".L_x_385", ".L_x_386", ".L_x_10"] and not brx.indirect and brx.reads == {"R2"}
    assert SG.Insn(0x210, "BRX R2 -0x220").indirect and SG.Insn(0, "JMX R2").indirect


@pytest.mark.parametrize("exc,why", [({"MUFU.RSQ": {"definition": "d", "numeric": {"runs": ["r"]}}},
                                      "excepts MUFU.RSQ, which sass_gate.EXCEPTABLE doesn't list"),
                                     ({"MUFU.EX2": {"definition": "d", "numeric": {"runs": []}}}, "no numeric runs"),
                                     ({"MUFU.EX2": {"numeric": {"runs": ["r"]}}}, "names no definition")])
def test_pins_refuse_an_exception_the_gate_doesnt_list_or_without_its_evidence(tmp_path, exc, why):
    p = SG.load_pins()
    p["toolkits"]["13.0.1-sm_90a"]["exceptions"] = exc
    (tmp_path / "pins.json").write_text(json.dumps(p))
    with pytest.raises(SG.SassGateFailed, match=why):
        SG.load_pins(tmp_path / "pins.json")


def test_a_repin_keeps_the_exceptions_only_while_the_ptxas_is_unchanged(tmp_path, records):
    out = tmp_path / "pins.json"
    out.write_text(SG.PINS.read_text())
    exc = SG.load_pins()["toolkits"]["13.0.1-sm_90a"]["exceptions"]
    assert SG.update_pins(records("plain90"), "13.0.1-sm_90a", out)["exceptions"] == exc
    assert "exceptions" not in SG.update_pins(records("plain"), "13.0.1-sm_90a", out)


# ---- tool notes -----------------------------------------------------------------------------------------------------------


def tkinfo(notes: list[dict]) -> bytes:
    """A .note.nv.tkinfo section's bytes, as ptxas writes it."""
    out = b""
    for n in notes:
        strs, offs = b"\0", []
        for s in (n["tool"], n["version"], n["branch"], n["args"]):
            offs.append(len(strs))
            strs += s.encode() + b"\0"
        desc = struct.pack("<II4I", 0x81, 0, *offs) + strs
        name = b"NVIDIA Corp\0"
        out += struct.pack("<III", len(name), len(desc), 2000) + name + desc + bytes(-len(desc) % 4)
    return out


def test_tool_notes_reads_every_record_and_refuses_a_short_one():
    notes = NOTES["plain"] + [{"tool": "nvlink", "version": "v", "branch": "b", "args": "-arch  sm_120a"}]
    got = SG.tool_notes(elf(SG.EM_CUDA, {".text.k": b"\x01" * 8, ".note.nv.tkinfo": tkinfo(notes)}))
    assert got == NOTES["plain"] + [{"tool": "nvlink", "version": "v", "branch": "b", "args": "-arch sm_120a"}]
    assert SG.tool_notes(elf(SG.EM_CUDA, {".text.k": b"\x01" * 8})) == []
    short = struct.pack("<III", 12, 8, 2000) + b"NVIDIA Corp\0" + bytes(8)
    with pytest.raises(SG.SassGateFailed, match="a .note.nv.tkinfo record of 8 bytes"):
        SG.tool_notes(elf(SG.EM_CUDA, {".note.nv.tkinfo": short}))


# ---- the binary's verdict -------------------------------------------------------------------------------------------------


def judge(name: str, cuobjdump: str | None = None, ptx: list[str] = ()) -> dict:
    c = cuobjdump if cuobjdump is not None else (FIX / f"{name}.cuobjdump.txt").read_text()
    return SG.judge([(SG.parse_nvdisasm(listing(name)), NOTES[name])], c, list(ptx), pins())


def test_the_two_disassemblies_must_agree():
    ok = judge("plain")
    assert ok["ok"] and ok["cross_check"] == {"cuobjdump_flagged": 80, "agree": True}
    lines = (FIX / "plain.cuobjdump.txt").read_text().splitlines(keepends=True)
    drop = next(i for i, ln in enumerate(lines) if "MUFU.RSQ" in ln)
    r = judge("plain", cuobjdump="".join(lines[:drop] + lines[drop + 1:]))
    assert r["offending_n"] == 0 and not r["ok"] and r["cross_check"] == {"cuobjdump_flagged": 79, "agree": False}
    assert "disagree" in SG.why_failed(r)


def test_why_failed_names_the_first_instruction_and_why():
    r = judge("fast_math")
    assert not r["ok"] and r["cross_check"]["agree"] and r["offending_n"] == 42
    assert SG.why_failed(r).startswith("42 flagged instruction(s) (.FTZ or FP32 MUFU) outside a pinned IEEE sequence or "
                                       "exception, first ")
    unpinned = SG.judge([(SG.parse_nvdisasm(listing("plain")), [])], (FIX / "plain.cuobjdump.txt").read_text(), [], pins())
    assert "(cubin 0: no ptxas record (.note.nv.tkinfo), so nothing is exempt)" in SG.why_failed(unpinned)


def test_ftz_or_approximate_ptx_fails():
    r = judge("plain", ptx=["\tadd.rn.f32 %f1, %f2, %f3;\n\tfma.rn.ftz.f32 %f4, %f1, %f2, %f3;\n\tdiv.full.f32 %f5, %f1, %f2;"
                            "\n\tsqrt.approx.f32 %f6, %f1;\n\tadd.rn.f32 %f7, %f1, %f2; // .ftz\n"])
    assert not r["ok"] and r["ptx_bad"] == ["fma.rn.ftz.f32 %f4, %f1, %f2, %f3;", "div.full.f32 %f5, %f1, %f2;",
                                            "sqrt.approx.f32 %f6, %f1;"]


# ---- the binaries ---------------------------------------------------------------------------------------------------------


def elf(machine: int, sections: dict[str, bytes]) -> bytes:
    """An ELF64 with the named sections (a .note.* one a note) and a .shstrtab, its section headers last."""
    names, offsets = b"\0", {}
    for name in [*sections, ".shstrtab"]:
        offsets[name] = len(names)
        names += name.encode() + b"\0"
    body, heads, off = b"", [(0, 0, 0, 0)], 64
    for name, data in [*sections.items(), (".shstrtab", names)]:
        typ = 3 if name == ".shstrtab" else SG.SHT_NOTE if name.startswith(".note") else 1
        heads.append((offsets[name], typ, off, len(data)))
        body += data
        off += len(data)
    ident = b"\x7fELF" + bytes([2, 1, 1]) + bytes(9)
    header = ident + struct.pack("<HHIQQQIHHHHHH", 2, machine, 1, 0, 0, off, 0, 64, 0, 0, 64, len(heads), len(heads) - 1)
    table = b"".join(struct.pack("<IIQQQQIIQQ", n, t, 0, 0, o, s, 0, 0, 1, 0) for n, t, o, s in heads)
    return header + body + table


def fatbin(entries: list[tuple[int, bytes, int]]) -> bytes:
    """A fatbin of (kind, payload, flags) entries: 2 an ELF, 1 PTX."""
    body = b""
    for kind, payload, flags in entries:
        padded = payload + bytes(-len(payload) % 8 + 8)
        body += struct.pack("<HHIQ", kind, 0x101, 64, len(padded)) + bytes(24) + struct.pack("<Q", flags) + bytes(16) + padded
    return struct.pack("<IHHQ", SG.FATBIN_MAGIC, 1, 16, len(body)) + body


def test_the_fatbin_parser_finds_every_cubin_of_every_fatbin_even_same_named_ones():
    a, b = elf(SG.EM_CUDA, {".text.k": b"\x01" * 40}), elf(SG.EM_CUDA, {".text.k": b"\x02" * 24})
    two = fatbin([(2, a, 0x11), (1, b"add.rn.f32 %f1, %f2, %f3;\0", 0x11)]) + bytes(8) + fatbin([(2, b, 0x11)])
    cubins, ptx = SG.device_code(elf(62, {".text": b"\x90" * 16, ".nv_fatbin": two}), "lib.so")
    assert cubins == [a, b] and ptx == ["add.rn.f32 %f1, %f2, %f3;"]
    assert SG.device_code(two, "x.fatbin") == ([a, b], ["add.rn.f32 %f1, %f2, %f3;"])
    assert SG.device_code(a, "x.cubin") == ([a], [])


@pytest.mark.parametrize("data,match", [
    (fatbin([(2, elf(SG.EM_CUDA, {".text.k": b"\x01" * 8}), SG.FATBIN_COMPRESSED)]), "compressed"),
    (fatbin([(3, b"\x01" * 8, 0)]), "unknown fatbin entry"),
    (elf(62, {".text": b"\x90" * 16}), "no device code"),
    (b"#!/bin/sh\n", "not a cubin"),
])
def test_what_the_parser_cannot_read_fails(data, match):
    with pytest.raises(SG.SassGateFailed, match=match):
        SG.device_code(data, "x")


def test_has_device_code_reads_the_section_headers(tmp_path):
    (tmp_path / "dev.so").write_bytes(elf(62, {".nv_fatbin": fatbin([(2, elf(SG.EM_CUDA, {".text.k": b"\x01"}), 0)])}))
    (tmp_path / "host.so").write_bytes(elf(62, {".text": b"\x90" * 16}))
    (tmp_path / "script").write_text("#!/bin/sh\n")
    assert [SG.has_device_code(str(tmp_path / f)) for f in ("dev.so", "host.so", "script", "absent")] == [True] + [False] * 3


# ---- the gate -------------------------------------------------------------------------------------------------------------

TOOLS = {"cuobjdump": {"path": "cuobjdump", "version": "fake"}, "nvdisasm": {"path": "nvdisasm", "version": "fake"}}


def fake_tools(monkeypatch, name: str) -> list:
    calls: list = []
    n, c = listing(name), (FIX / f"{name}.cuobjdump.txt").read_text()
    monkeypatch.setattr(SG, "_run", lambda cmd: calls.append(cmd[0]) or (n if cmd[0] == "nvdisasm" else c))
    return calls


def a_library(path: Path, notes: list[dict] = NOTES["plain"]) -> Path:
    cubin = elf(SG.EM_CUDA, {".text.k": b"\x01" * 8, ".note.nv.tkinfo": tkinfo(notes)})
    path.write_bytes(elf(62, {".nv_fatbin": fatbin([(2, cubin, 0)])}))
    return path


def pins_file(path: Path, numeric: dict | None = TEST_RUN) -> Path:
    path.write_text(json.dumps(pins(numeric)))
    return path


def test_a_pass_is_cached_by_the_binary_and_a_failure_is_not(tmp_path, monkeypatch):
    lib, cache, p = a_library(tmp_path / "libarm.so"), tmp_path / "cache", pins_file(tmp_path / "pins.json")
    calls = fake_tools(monkeypatch, "plain")
    first = SG.scan(lib, TOOLS, cache, p)
    assert first["ok"] and not first["cached"] and calls == ["nvdisasm", "cuobjdump"]
    assert first["cubins"][0]["tool_notes"] == NOTES["plain"] and first["cubins"][0]["toolkit"] == "12.9.1"
    again = SG.scan(lib, TOOLS, cache, p)
    assert again["cached"] and calls == ["nvdisasm", "cuobjdump"] and again["covered"] == PLAIN_COVERED
    assert not SG.scan(lib, TOOLS, cache, pins_file(tmp_path / "unpinned.json", None))["ok"]
    lib.write_bytes(lib.read_bytes() + b"\0")
    calls = fake_tools(monkeypatch, "fast_math")
    assert not SG.scan(lib, TOOLS, cache, p)["ok"]
    assert not SG.scan(lib, TOOLS, cache, p)["cached"] and calls == ["nvdisasm", "cuobjdump"] * 2


def test_the_gate_names_the_failing_binary_function_and_instruction(tmp_path, monkeypatch):
    lib = a_library(tmp_path / "libarm.so")
    fake_tools(monkeypatch, "fast_math")
    logged: list[str] = []
    g = SG.SassChecker(log=logged.append, tools=TOOLS, cache=None, pins=pins_file(tmp_path / "pins.json"))
    with pytest.raises(SG.SassGateFailed, match=r"libarm\.so: 42 flagged instruction\(s\) .* first \w+ 0x[0-9a-f]+: "):
        g.check([str(lib)], "arm:x")
    assert logged[0].startswith(f"sass gate: FAIL {lib} (ptxas 12.9.1; 55 flagged, covered ")
    rep = g.report()
    assert not rep["ok"] and rep["binaries"][0]["source"] == "arm:x"
    with pytest.raises(SG.SassGateFailed, match="is not a file"):
        g.check([str(tmp_path / "absent.cubin")], "arm:x")


def test_an_arm_must_declare_its_binaries():
    assert SG.declared("x", {"binaries": ["k.cubin"]}) == ["k.cubin"]
    assert SG.declared("x", {"binaries": []}) == []
    for build in ({}, {"binaries": "k.cubin"}, {"binaries": [1]}, None):
        with pytest.raises(SG.SassGateFailed, match="arm x: build\\(\\) returns no 'binaries'"):
            SG.declared("x", build)


def test_discovery_takes_mapped_device_code_and_leaves_nvidias_libraries_unscanned(tmp_path, monkeypatch):
    libs = [a_library(tmp_path / "libcublasLt.so.12"), a_library(tmp_path / "libarm.so"), tmp_path / "libhost.so"]
    libs[2].write_bytes(elf(62, {".text": b"\x90" * 16}))
    monkeypatch.setattr(SG, "mapped", lambda: [str(p) for p in libs])
    g = SG.SassChecker(tools=TOOLS, cache=None)
    assert g.discover() == [str(libs[1])]
    assert g.report()["not_scanned"] == [{"path": str(libs[0]), "reason": "NVIDIA's library"}]
    for gone in (str(tmp_path / "libgone.so") + " (deleted)", "/memfd:lib (deleted)", "/dev/shm/x (deleted)"):
        monkeypatch.setattr(SG, "mapped", lambda gone=gone: [gone])
        with pytest.raises(SG.SassGateFailed, match="mapped but deleted"):
            g.discover()


def test_shared_anonymous_memory_is_not_a_library(tmp_path, monkeypatch):
    """The CUDA driver maps shared anonymous memory, which /proc/self/maps names '/dev/zero (deleted)'."""
    anon = ["/dev/zero (deleted)", "/SYSV0000a1b2 (deleted)"]
    monkeypatch.setattr(SG, "mapped", lambda: [*anon, str(a_library(tmp_path / "libarm.so"))])
    g = SG.SassChecker(tools=TOOLS, cache=None)
    assert g.discover() == [str(tmp_path / "libarm.so")]
    assert g.report()["not_scanned"] == [{"path": p, "reason": "shared anonymous memory, not a file"} for p in anon]


def test_a_deleted_posix_semaphore_is_not_a_library_but_a_bigger_or_code_bearing_one_fails(monkeypatch):
    """multiprocessing's locks map an unlinked POSIX semaphore, '/dev/shm/sem.<name> (deleted)': one page with a sem_t."""
    import multiprocessing as mp

    lock = mp.Lock()  # noqa: F841 - keeps the semaphore mapped
    sems = [p for p in SG.mapped() if SG.POSIX_SEM.fullmatch(p)]
    assert sems, "a multiprocessing lock maps no /dev/shm/sem.* (deleted)"
    monkeypatch.setattr(SG, "mapped", lambda: sems[:1])
    g = SG.SassChecker(tools=TOOLS, cache=None)
    assert g.discover() == [] and g.report()["not_scanned"][0]["reason"].startswith("a POSIX semaphore")
    fake = "/dev/shm/sem.mp-planted (deleted)"
    monkeypatch.setattr(SG, "mapped", lambda: [fake])
    for size, data in ((2 * SG.PAGE, b"\0"), (SG.PAGE, b"\x7fELF"), (SG.PAGE, b"\0" * 8 + SG.FATBIN_MAGIC.to_bytes(4, "little"))):
        monkeypatch.setattr(SG, "regions", lambda p, size=size: [(0x1000, 0x1000 + size)])
        monkeypatch.setattr(SG, "read_mapped", lambda s, n, data=data: data.ljust(n, b"\0"))
        with pytest.raises(SG.SassGateFailed, match="mapped but deleted"):
            SG.SassChecker(tools=TOOLS, cache=None).discover()


def test_this_process_maps_the_python_interpreter():
    assert any(Path(p).name.startswith("python") or "libpython" in p for p in SG.mapped())


# ---- real builds ----------------------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["sm_120a", "sm_90a"])
@pytest.mark.parametrize("src,flags,ok", [
    ("ftz_probe.cu", (), True), ("ftz_probe.cu", ("-fmad=false",), True), ("ftz_probe.cu", ("--use_fast_math",), False),
    ("ftz_probe.cu", ("-ftz=true",), False), ("ftz_probe.cu", ("-prec-div=false", "-prec-sqrt=false"), False),
    ("ftz_refused.cu", (), False)])
def test_a_real_build(tmp_path, arch, src, flags, ok):
    """Compiles the probe as a cubin and as a shared library with the toolkit the gate finds (CUDA_HOME, POUW_CUDA_BIN or
    /usr/local/cuda) and gates both with the shipped pins; skipped without nvcc, cuobjdump and nvdisasm. It catches what the
    listings can't: a toolkit whose expansions or notes differ, and a real host ELF's fatbin. A build passes only from a
    toolkit pinned for its target with a numeric run; an sm_90a cubin from any other fails closed (12.9 writes it no ptxas
    note, 13.0 unpinned arguments), and one from 13.0.1 fails only on the div and sqrt slow-path bodies its numeric run
    didn't pin (`test_the_shipped_sm90a_entry_exempts_only_the_bodies_its_numeric_run_tested`)."""
    try:
        tools = SG.find_tools()
    except SG.SassGateFailed as ex:
        pytest.skip(str(ex))
    nvcc = Path(tools["cuobjdump"]["path"]).with_name("nvcc")
    if not nvcc.exists():
        pytest.skip(f"no nvcc beside {tools['cuobjdump']['path']}")
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
    pinned = any(t["version"] in version and t["branch"] in version and t["numeric"] and f"-arch {arch} -m 64" in t["args"]
                 for t in SG.load_pins()["toolkits"].values())
    base = [str(nvcc), "-O3", "-std=c++17", "-gencode", f"arch=compute_{arch[3:]},code={arch}", *flags]
    for out, extra in (("probe.cubin", ["-cubin"]), ("libprobe.so", ["-Xcompiler", "-fPIC", "-shared", "--no-compress"])):
        subprocess.run([*base, *extra, "-o", str(tmp_path / out), str(FIX / src)], check=True)
        rec = SG.scan(tmp_path / out, tools, None)
        code = [c for c in rec["cubins"] if c["functions"]]
        assert rec["cross_check"]["agree"] and len(code) == 1, (out, rec["cross_check"], len(code))
        assert (code[0]["toolkit"] is not None) is pinned, (out, code[0]["unpinned"])
        if arch == "sm_90a" and not pinned:
            assert code[0]["unpinned"] == "no ptxas record (.note.nv.tkinfo)" \
                or "with unpinned arguments '-arch sm_90a -m 64" in code[0]["unpinned"], code[0]["unpinned"]
        assert rec["ok"] is (ok and pinned and arch == "sm_120a"), (out, rec["offending"][:3])
        if rec["ok"]:
            assert rec["covered"] == PLAIN_COVERED, (out, rec["covered"])
        elif pinned and ok:
            assert {o["function"] for o in rec["offending"]} <= {"sqrt_k", "rho_k", "cdiv_k"} and all(
                "isn't the pinned body" in o["why"] or "isn't a pinned one" in o["why"]
                for o in rec["offending"] if "why" in o), (out, rec["offending"][:3])
        elif pinned:
            fns = {o["function"] for o in rec["offending"]}
            assert fns and (fns <= FP32_KERNELS or src == "ftz_refused.cu"), (out, fns)
