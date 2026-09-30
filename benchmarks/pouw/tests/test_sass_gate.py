"""The SASS gate (`harness/sass_gate.py`): a --use_fast_math build fails and the same kernels built without it pass, on
CUDA 12.9.1's listings of `fixtures/sass_gate/ftz_probe.cu`; every spelling of an FP32 .FTZ add, multiply or FMA is refused;
only ptxas's IEEE expansions are exempt; the two disassemblies must agree; the fatbin parser finds every cubin and refuses
what it can't read; passes are cached and failures aren't; an arm must declare its binaries; and, where a CUDA toolkit is
present, real builds with and without the flags."""

from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "harness"))

import sass_gate as SG  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "sass_gate"
PLAIN_EXEMPT = {"rcp-fast-path": 1, "__cuda_sm20_rcp_rn_f32_slowpath": 2, "sqrt-fast-path": 2,
                "__cuda_sm20_sqrt_rn_f32_slowpath": 5, "__cuda_sm3x_div_rn_noftz_f32_slowpath": 2}


def listings(name: str) -> tuple[str, str]:
    return (FIX / f"{name}.nvdisasm.txt").read_text(), (FIX / f"{name}.cuobjdump.txt").read_text()


def judge(name: str, nvdisasm: str | None = None, cuobjdump: str | None = None) -> dict:
    n, c = listings(name)
    return SG.judge(SG.parse_nvdisasm(nvdisasm or n), cuobjdump or c, [])


# ---- the verdicts ---------------------------------------------------------------------------------------------------------


def test_a_fast_math_build_fails_and_the_same_kernels_built_without_it_pass():
    plain = judge("plain")
    assert plain["ok"] and plain["ftz_fp32"] == 12 and plain["offending_n"] == 0
    assert plain["exempt"] == PLAIN_EXEMPT
    assert plain["cross_check"] == {"cuobjdump_ftz_fp32": 12, "agree": True}
    fast = judge("fast_math")
    assert not fast["ok"] and fast["exempt"] == {} and fast["offending_n"] == fast["ftz_fp32"] == 16
    assert {o["function"] for o in fast["offending"]} == {"arith_k", "rho_k", "rcp_k"}
    assert any(o["function"] == "arith_k" and o["instruction"].startswith("FADD.FTZ") for o in fast["offending"])
    assert SG.why_failed(fast).startswith("16 FP32 .FTZ instruction(s) outside ptxas's IEEE expansions, first arith_k 0x")


@pytest.mark.parametrize("text", ["FADD.FTZ R0, R2, R3", "@!P0 FFMA.FTZ R0, R1, R2, R3", "@P1 FMUL.FTZ R0, R3, 0.5",
                                  "FFMA.RZ.FTZ R0, R1, R2, R3", "FADD.FTZ.SAT R0, R1, R2", "FFMA2.FTZ R0, R2, R4, R6",
                                  "FADD32I.FTZ R0, R1, 0x3f800000", "FMUL32I.FTZ.SAT R0, R1, 2", "{ FADD.FTZ R0, R1, R2"])
def test_every_spelling_of_an_fp32_ftz_add_multiply_or_fma_is_refused(text):
    assert SG.is_ftz_arith(text)


@pytest.mark.parametrize("text", ["FADD R0, R2, R3", "FFMA.RZ R0, R1, R2, R3", "FSETP.GTU.FTZ.AND P0, PT, |R0|, +INF , PT",
                                  "F2I.FTZ.U32.TRUNC.NTZ R5, R4", "HADD2.FTZ R0, R1, R2", "DADD R0, R2, R4",
                                  "FMNMX.FTZ R0, R1, R2, !PT"])
def test_other_opcodes_are_not_the_rule(text):
    assert not SG.is_ftz_arith(text)


def test_a_fast_path_is_exempt_only_when_its_arithmetic_is_the_intrinsics():
    n, c = listings("plain")
    at = n.index("FMUL.FTZ R0, R3, 0.5 ;")
    line_end = n.index("\n", at) + 1
    extra = n[:line_end] + "        /*0248*/                   FADD.FTZ R9, R9, R9 ;\n" + n[line_end:]
    r = judge("plain", nvdisasm=extra)
    assert not r["ok"]
    assert {o["instruction"] for o in r["offending"]} == {"FMUL.FTZ R7, R3.reuse, R4", "FMUL.FTZ R0, R3, 0.5",
                                                         "FADD.FTZ R9, R9, R9"}
    assert "sqrt-fast-path" not in r["exempt"]


def test_sm90s_move_after_the_call_is_a_move_and_nothing_else_is():
    """sm_90a moves the slow path's result with IMAD.MOV.U32 (#449's pearl_c.cubin); any other instruction there ends the
    idiom, so the fast path after it is not exempt."""
    n, _ = listings("plain")
    assert "MOV R7, R3 ;" in n
    assert judge("plain", nvdisasm=n.replace("MOV R7, R3 ;", "IMAD.MOV.U32 R7, RZ, RZ, R3 ;")) == judge("plain")
    r = judge("plain", nvdisasm=n.replace("MOV R7, R3 ;", "IMAD R7, R3, R4, RZ ;"))
    assert not r["ok"] and "sqrt-fast-path" not in r["exempt"]


def test_an_ftz_slow_path_is_not_exempt_nor_the_fast_path_around_it():
    n, _ = listings("plain")
    r = judge("plain", nvdisasm=n.replace("__cuda_sm20_sqrt_rn_f32_slowpath", "__cuda_sm20_sqrt_rn_ftz_f32_slowpath"))
    assert not r["ok"] and r["offending_n"] == 7
    assert "sqrt-fast-path" not in r["exempt"] and "__cuda_sm20_sqrt_rn_f32_slowpath" not in r["exempt"]


def test_the_two_disassemblies_must_agree():
    n, c = listings("plain")
    lines = c.splitlines(keepends=True)
    drop = next(i for i, ln in enumerate(lines) if "FADD.FTZ" in ln)
    r = judge("plain", cuobjdump="".join(lines[:drop] + lines[drop + 1:]))
    assert r["offending_n"] == 0 and not r["ok"]
    assert r["cross_check"] == {"cuobjdump_ftz_fp32": 11, "agree": False}
    assert "disagree" in SG.why_failed(r)


def test_a_sections_flags_line_keeps_its_function():
    """nvdisasm writes `.sectionflags` after `.section` for a kernel that uses barriers (#449's sm_90a plainq)."""
    n, _ = listings("plain")
    flagged = "".join(ln + ('\t.sectionflags\t@"SHF_BARRIERS=1"\n' if ln.lstrip().startswith(".section\t.text.") else "")
                      for ln in n.splitlines(keepends=True))
    assert flagged.count(".sectionflags") >= 3
    assert judge("plain", nvdisasm=flagged) == judge("plain")


def test_ftz_f32_ptx_fails():
    n, c = listings("plain")
    r = SG.judge(SG.parse_nvdisasm(n), c, ["\tadd.rn.f32 %f1, %f2, %f3;\n\tfma.rn.ftz.f32 %f4, %f1, %f2, %f3;\n"])
    assert not r["ok"] and r["ptx_ftz"] == ["fma.rn.ftz.f32 %f4, %f1, %f2, %f3;"]


# ---- the binaries ---------------------------------------------------------------------------------------------------------


def elf(machine: int, sections: dict[str, bytes]) -> bytes:
    """An ELF64 with the named sections and a .shstrtab, its section headers last."""
    names, offsets = b"\0", {}
    for name in [*sections, ".shstrtab"]:
        offsets[name] = len(names)
        names += name.encode() + b"\0"
    body, heads, off = b"", [(0, 0, 0, 0)], 64
    for name, data in [*sections.items(), (".shstrtab", names)]:
        heads.append((offsets[name], 3 if name == ".shstrtab" else 1, off, len(data)))
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
    n, c = listings(name)
    monkeypatch.setattr(SG, "_run", lambda cmd: calls.append(cmd[0]) or (n if cmd[0] == "nvdisasm" else c))
    return calls


def a_library(path: Path) -> Path:
    path.write_bytes(elf(62, {".nv_fatbin": fatbin([(2, elf(SG.EM_CUDA, {".text.k": b"\x01" * 8}), 0)])}))
    return path


def test_a_pass_is_cached_by_the_binary_and_a_failure_is_not(tmp_path, monkeypatch):
    lib, cache = a_library(tmp_path / "libarm.so"), tmp_path / "cache"
    calls = fake_tools(monkeypatch, "plain")
    first = SG.scan(lib, TOOLS, cache)
    assert first["ok"] and not first["cached"] and calls == ["nvdisasm", "cuobjdump"]
    again = SG.scan(lib, TOOLS, cache)
    assert again["cached"] and calls == ["nvdisasm", "cuobjdump"] and again["exempt"] == PLAIN_EXEMPT
    lib.write_bytes(lib.read_bytes() + b"\0")
    calls = fake_tools(monkeypatch, "fast_math")
    assert not SG.scan(lib, TOOLS, cache)["ok"]
    assert not SG.scan(lib, TOOLS, cache)["cached"] and calls == ["nvdisasm", "cuobjdump"] * 2


def test_the_gate_names_the_failing_binary_function_and_instruction(tmp_path, monkeypatch):
    lib = a_library(tmp_path / "libarm.so")
    fake_tools(monkeypatch, "fast_math")
    g = SG.Gate(tools=TOOLS, cache=None)
    with pytest.raises(SG.SassGateFailed, match=r"libarm\.so: 16 FP32 \.FTZ .* first arith_k 0x[0-9a-f]+: FADD\.FTZ"):
        g.check([str(lib)], "arm:x")
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
    g = SG.Gate(tools=TOOLS, cache=None)
    assert g.discover() == [str(libs[1])]
    assert g.report()["not_scanned"] == [{"path": str(libs[0]), "reason": "NVIDIA's library"}]
    monkeypatch.setattr(SG, "mapped", lambda: [str(tmp_path / "libgone.so") + " (deleted)"])
    with pytest.raises(SG.SassGateFailed, match="mapped but deleted"):
        g.discover()


def test_this_process_maps_the_python_interpreter():
    assert any(Path(p).name.startswith("python") or "libpython" in p for p in SG.mapped())


# ---- real builds ----------------------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["sm_120a", "sm_90a"])
@pytest.mark.parametrize("flags,ok", [((), True), (("-fmad=false",), True), (("--use_fast_math",), False),
                                      (("-ftz=true",), False)])
def test_a_real_build_passes_without_fast_math_or_ftz_and_fails_with_either(tmp_path, arch, flags, ok):
    """Compiles `ftz_probe.cu` as a cubin and as a shared library with the toolkit the gate finds (CUDA_HOME, POUW_CUDA_BIN or
    /usr/local/cuda) and gates both, for sm_120a and #449's sm_90a; skipped without nvcc, cuobjdump and nvdisasm. It catches
    what the listings can't: a toolkit or an architecture whose expansions differ, and the fatbin of a real host ELF."""
    try:
        tools = SG.find_tools()
    except SG.SassGateFailed as ex:
        pytest.skip(str(ex))
    nvcc = Path(tools["cuobjdump"]["path"]).with_name("nvcc")
    if not nvcc.exists():
        pytest.skip(f"no nvcc beside {tools['cuobjdump']['path']}")
    base = [str(nvcc), "-O3", "-std=c++17", "-gencode", f"arch=compute_{arch[3:]},code={arch}", *flags]
    for out, extra in (("probe.cubin", ["-cubin"]), ("libprobe.so", ["-Xcompiler", "-fPIC", "-shared"])):
        subprocess.run([*base, *extra, "-o", str(tmp_path / out), str(FIX / "ftz_probe.cu")], check=True)
        rec = SG.scan(tmp_path / out, tools, None)
        assert rec["ok"] is ok and rec["cross_check"]["agree"], (out, rec)
        if ok:
            assert rec["exempt"] == PLAIN_EXEMPT, (out, rec["exempt"])
        else:
            assert any(o["function"] == "arith_k" for o in rec["offending"]), (out, rec["offending"])
