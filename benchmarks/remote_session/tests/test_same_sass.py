"""`same_sass.py`: a kernel is the same when its listing is, whatever its name and however wide cuobjdump made the
encodings' column; a kernel whose instructions changed is named, with its listing beside the new build's, and the
comparison exits 1, so `host_read.sh` runs nothing on that build. A kernel only the new build has leaves its listing. cuobjdump is a stub that prints each library's canned
listing. A cuobjdump that fails is said on one line, with what it said. With nvcc's log of the new build, a kernel
only the new build has is recorded with its registers and spilled bytes, and one that spills is refused."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import same_sass


def listing(kernels: dict[str, list[str]], column: int) -> str:
    """cuobjdump's -sass of `kernels`, each instruction's encoding at `column`."""
    out = ["Fatbin elf code:", "arch = sm_90", "\tcode for sm_90"]
    for name, instructions in kernels.items():
        out += [f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM90"']
        out += [f"        /*{16 * k:04x}*/ {text} ;".ljust(column) + "/* 0x00 */" for k, text in enumerate(instructions)]
    return "\n".join(out) + "\n"


def compare(tmp_path, monkeypatch, capsys, base: dict, new: dict, exits: int = 0, ptxas: str = "") -> tuple[dict, str]:
    libraries = {"base.so": listing(base, 60), "new.so": listing(new, 61)}
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, libraries[argv[-1]]))
    (tmp_path / "new.ptxas").write_text(ptxas)
    monkeypatch.setattr(sys, "argv", ["same_sass.py", "base.so", "new.so", "--out", str(tmp_path / "same.json"),
                                      "--listings", str(tmp_path / "sass")]
                        + (["--ptxas", str(tmp_path / "new.ptxas")] if ptxas else []))
    assert same_sass.main() == exits
    return json.loads((tmp_path / "same.json").read_text()), capsys.readouterr().out


def test_a_wider_encoding_column_and_a_new_name_leave_a_kernel_the_same(tmp_path, monkeypatch, capsys):
    read, signs = ["LDC R1, c[0x0][0x28]", "EXIT"], ["MOV R2, R3", "EXIT"]
    doc, said = compare(tmp_path, monkeypatch, capsys, {"_ZN2rb11read_kernelEv": read, "_ZN2rb14signs16_kernelEv": signs},
                        {"_ZN2rb11read_kernelEv": read, "_ZN2rb12signs_kernelILi16EEEvv": signs,
                         "_ZN2rb12signs_kernelILi64EEEvv": ["MOV R2, R4", "EXIT"]})
    assert doc["same"] == doc["base_kernels"] == 2 and doc["changed"] == []
    assert doc["renamed"] == {"_ZN2rb14signs16_kernelEv": "_ZN2rb12signs_kernelILi16EEEvv"}
    assert doc["only_new"] == ["_ZN2rb12signs_kernelILi64EEEvv"]
    assert said.startswith("2 of BASE's 2 kernels") and "changed: none" in said
    assert [(f.name, f.read_text()) for f in (tmp_path / "sass").iterdir()] == [
        ("signs_kernel-64.only-new.sass",
         '.headerflags @"EF_CUDA_SM90"\n/*0000*/ MOV R2, R4 ; /* 0x00 */\n/*0010*/ EXIT ; /* 0x00 */\n')]


def test_a_changed_kernel_exits_1_and_is_named_with_its_listing_beside_its_16_row_instance(tmp_path, monkeypatch,
                                                                                           capsys):
    name = "_ZN2rb20chain_prepare_kernelEv"
    doc, said = compare(tmp_path, monkeypatch, capsys, {name: ["LEA.HI R13, R0, R5", "EXIT"]},
                        {"_ZN2rb20chain_prepare_kernelILi64EEEvv": ["NOP", "EXIT"],
                         "_ZN2rb20chain_prepare_kernelILi16EEEvv": ["LEA.HI R15, R0, R5", "EXIT"]}, exits=1)
    assert doc["same"] == 0 and doc["changed"] == [name] and f"changed: {name}" in said
    diff = next((tmp_path / "sass").glob("chain_prepare_kernel-*.diff")).read_text()
    assert "+++ _ZN2rb20chain_prepare_kernelILi16EEEvv" in diff and "-/*0000*/ LEA.HI R13, R0, R5 ; /* 0x00 */" in diff


def test_a_cuobjdump_that_fails_is_said_on_one_line_with_its_message(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(
        argv, 1, "", "cuobjdump fatal   : no disassembler\nfor this file\n"))
    monkeypatch.setattr(sys, "argv", ["same_sass.py", "base.so", "new.so"])
    assert same_sass.main() == 2
    said = capsys.readouterr()
    assert said.out == "" and said.err == ("no comparison: cuobjdump -sass base.so exits 1: cuobjdump fatal : no "
                                           "disassembler for this file\n")


def ptxas(kernels: dict[str, tuple[int, int]]) -> str:
    """nvcc's `-Xptxas -v` lines for `kernels` (registers, spilled bytes each way), a helper's properties among them."""
    out = ["ptxas info    : 6 bytes gmem", "ptxas info    : Function properties for _ZN2rb9help_tileEv",
           "    104 bytes stack frame, 144 bytes spill stores, 176 bytes spill loads"]
    for name, (registers, spill) in kernels.items():
        out += [f"ptxas info    : Compiling entry function '{name}' for 'sm_90a'",
                f"ptxas info    : Function properties for {name}",
                f"    {spill} bytes stack frame, {spill} bytes spill stores, {spill} bytes spill loads",
                f"ptxas info    : Used {registers} registers, used 1 barriers, 24576 bytes smem",
                "ptxas info    : Compile time = 111.229 ms"]
    return "\n".join(out) + "\n"


def test_a_kernel_only_the_new_build_has_is_recorded_as_built_and_refused_when_it_spills(tmp_path, monkeypatch, capsys):
    read = ["LDC R1, c[0x0][0x28]", "EXIT"]
    one, two = "_ZN2rb12fresh_kernelILi1EEEvv", "_ZN2rb12fresh_kernelILi3EEEvv"
    new = {"_ZN2rb11read_kernelEv": read, one: ["MOV R2, R4", "EXIT"], two: ["MOV R2, R5", "EXIT"]}
    doc, said = compare(tmp_path, monkeypatch, capsys, {"_ZN2rb11read_kernelEv": read}, new,
                        ptxas=ptxas({"_ZN2rb11read_kernelEv": (20, 8), one: (124, 0), two: (128, 0)}))
    assert doc["only_new_built"] == {one: {"registers": 124, "spill_bytes": 0}, two: {"registers": 128, "spill_bytes": 0}}
    assert "fresh_kernel<1> 124 registers, 0 spilled bytes" in said and "refused" not in said
    doc, said = compare(tmp_path, monkeypatch, capsys, {"_ZN2rb11read_kernelEv": read}, new, exits=1,
                        ptxas=ptxas({one: (124, 0), two: (128, 16)}))
    assert doc["changed"] == [] and doc["only_new_built"][two] == {"registers": 128, "spill_bytes": 32}
    assert said.rstrip().endswith(f"refused: {two}")
    doc, said = compare(tmp_path, monkeypatch, capsys, {"_ZN2rb11read_kernelEv": read}, new, exits=1,
                        ptxas=ptxas({one: (124, 0)}))
    assert doc["only_new_built"][two] == {} and said.rstrip().endswith(f"refused: {two}")
