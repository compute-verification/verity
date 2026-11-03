"""Every build gate on ptxas's report and cuobjdump's SASS (`harness/sass_gate.py`'s `mma_gate`, through each kernel's own
`sass_gate.py`): its CLI's exit code, OUT_JSON and printed lines on each case are pinned in `fixtures/mma_gate/verdicts.json`
(sha256s of the bytes), so a refactor of the shared gate can't move a verdict.

The cases: a clean listing of every kernel the gate wants (`pass`, and `pass_sm120a` for a gate given ARCH), the same with
one of each refusal (`refused`), and the harness's real sm_120a `cuobjdump -sass` listings (`fixtures/sass_gate/plain` and
`fast_math`) under a ptxas report of their functions. The listings are generated here from each gate's own tables, so a
change that means to move a verdict (a kernel added to WANT, say) rewrites the pins: `python test_mma_gate.py --write`.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIX = Path(__file__).resolve().parent / "fixtures"
VERDICTS = FIX / "mma_gate" / "verdicts.json"
GATES = {"pearl_c_sm90": ("benchmarks/pouw/pearl_c_sm90/sass_gate.py", None),
         "pearl_c_sm120": ("benchmarks/pouw/pearl_c_sm120/sass_gate.py", None),
         "s8_linear": ("benchmarks/torch_gpu/s8_linear/sass_gate.py", None),
         "attn_bf16": ("benchmarks/torch_gpu/attn_bf16/sass_gate.py", "sm_90a")}
#: the arch each gate's real-listing cases are reported for
REAL_ARCH = {"pearl_c_sm90": "sm_90a", "pearl_c_sm120": "sm_120a", "s8_linear": "sm_90a", "attn_bf16": "sm_120a"}


def _load(gate: str):
    name = f"mma_gate_case_{gate}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / GATES[gate][0])
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


def entry(name: str, arch: str, regs: int = 64, stack: int = 0, spills: int = 0, smem: int = 1024, used: bool = True) -> str:
    return (f"ptxas info    : Compiling entry function '{name}' for '{arch}'\nptxas info    : Function properties for {name}\n"
            f"    {stack} bytes stack frame, {spills} bytes spill stores, {spills} bytes spill loads\n"
            + (f"ptxas info    : Used {regs} registers, used 1 barriers, {smem} bytes smem, 528 bytes cmem[0]\n" if used else ""))


class Listing:
    """A cuobjdump -sass listing, written the way cuobjdump writes one."""

    def __init__(self):
        self.lines, self.addr = ["\t\t.........."], 0

    def fn(self, name: str):
        self.lines += [f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM90 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90)"']
        self.addr = 0

    def ins(self, text: str, n: int = 1):
        for _ in range(n):
            self.lines.append(f"        /*{self.addr:04x}*/                   {text} ;          /* 0x000000ff05057208 */")
            self.addr += 16

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def pearl_c_sm90(arch: str, bad: bool) -> tuple[str, str]:
    G = _load("pearl_c_sm90")
    p, s = [], Listing()
    for name, (n, q, h) in G.WANT.items():
        if bad and name == "gemm_words":
            continue
        p.append(entry(name, "sm_90a", regs=168))
        s.fn(name)
        s.ins("UTMALDG.2D [UR8], [UR4]")
        s.ins("SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [UR4], RZ")
        s.ins(f"QGMMA.64x{n}x32.F32.E4M3.E4M3 R24, gdesc[UR4], R24, gsb0", 3 if bad and name == "gemm_plain" else
              2 * q if name == "gemm_plain" else q)
        s.ins(f"HGMMA.64x{n}x16.F32.BF16 R24, gdesc[UR8], R24, gsb0", h)
        s.ins("QGMMA.64x32x32.F32.E4M3.E4M3 R88, gdesc[UR12], R88, gsb0", G.WINDOW_ATOMS.get(name, 0) + (bad and name in G.WINDOW_ATOMS))
        if bad and name == "gemm_ct":
            s.ins("HMMA.16816.F32.BF16 R4, R8, R12, R4")
        if bad and name == "gemm_fast":
            s.ins("LDL R1, [R1]")
        s.ins("F2FP.BF16.F32.PACK_AB R3, R5, R4")
        s.ins("EXIT")
    p.append(entry("s512_rows", "sm_90a", regs=255, stack=24 if bad else 8, spills=4, smem=0))
    s.fn("s512_rows")
    s.ins("STL [R1], R2")
    s.ins("LDL R2, [R1]")
    p.append(entry("s512_pads", "sm_90a", regs=32, smem=0))
    s.fn("s512_pads")
    s.ins("STG.E desc[UR4][R2.64], R5")
    if bad:
        p += ["ptxas warning : Registers are spilled\n",
              "ptxas info    : (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized\n"]
    return "".join(p), s.text()


def pearl_c_sm120(arch: str, bad: bool) -> tuple[str, str]:
    G = _load("pearl_c_sm120")
    p, s = [], Listing()
    for name, (q, h) in G.WANT.items():
        if bad and name == "words_k":
            continue
        p.append(entry(name, "sm_120a", regs=96))
        s.fn(name)
        s.ins("LDGSTS.E.BYPASS.128 [R3], desc[UR4][R4.64]")
        s.ins("LDSM.16.M88.4 R8, [R3]")
        s.ins("QMMA.16832.F32.E4M3.E4M3 R24, R8, R12, R24", q + (bad and name == "gemm_plain"))
        s.ins("HMMA.16816.F32.BF16 R40, R8, R12, R40", h)
        if bad and name == "gemm_ct":
            s.ins("QMMA.16832.F32.E5M2.E5M2 R24, R8, R12, R24")
        if bad and name == "words":
            s.ins("STL [R1], R2")
        s.ins("EXIT")
    p.append(entry("s512_pads", "sm_120a", regs=32, smem=0))
    s.fn("s512_pads")
    s.ins("STG.E desc[UR4][R2.64], R5")
    if bad:
        p += ["ptxas warning : Registers are spilled\n", entry("lines", "sm_120a", regs=40, spills=8, smem=0)]
    return "".join(p), s.text()


def s8_linear(arch: str, bad: bool) -> tuple[str, str]:
    G = _load("s8_linear")
    want = [f"s8mm{c}_{tm}x{bn}" for c in ("", "2") for tm in (64, 128) for bn in (64, 128, 256)]
    reduce = sorted({f"{f}_128x{bn}_r{r}" for f in ("s8mr", "s8mrb") for bn in (128, 256) for r in (8, 16, 32, 64, 128)} - G.UNBUILT)
    p, s = [], Listing()
    for name in want + reduce:
        if bad and name == "s8mm2_128x256":
            continue
        rm = G.REDUCE.match(name)
        bn = int((G.KERNEL.match(name) or rm).group(2))
        p.append(entry(name, "sm_90a", regs=168, used=not (bad and name == "s8mm_64x128")))
        s.fn(name)
        s.ins("UTMALDG.2D [UR8], [UR4]")
        s.ins(f"IGMMA.64x{bn}x32.{'U8.S8' if bad and name == 's8mm_128x64' else 'S8.S8'} R24, gdesc[UR4], R24, gsb0",
              8 if name.endswith("64x64") else 4)
        if rm:
            s.ins(f"IGMMA.64x{rm.group(3)}x32.S8.S8 R200, R24, gdesc[UR8], R200", bn // 8)
        s.ins("F2I.FTZ.U32.TRUNC.NTZ R9, R8")
        s.ins("I2FP.F32.S32 R21, R25")
        if not (bad and name == "s8mm_64x256"):
            s.ins("FMUL R60, R21, R12")
        if bad and name == "s8mm2_64x64":
            s.ins("FFMA R1, R2, R3, R4")
            s.ins("FMUL.FTZ R1, R2, R3")
        if bad and name == "s8mm2_64x128":
            s.ins("LDL R1, [R2]")
            s.ins("HGMMA.64x64x16.F32.BF16 R24, gdesc[UR4], R24")
        s.ins("F2FP.BF16.F32.PACK_AB R61, R60, R61")
        s.ins("EXIT")
    if bad:
        p += [entry("s8mm_64x64", "sm_100a", regs=80),
              "ptxas info    : (C7515) Potential Performance Loss: wgmma.mma_async instructions are serialized\n"]
    return "".join(p), s.text()


def attn_bf16(arch: str, bad: bool) -> tuple[str, str]:
    G = _load("attn_bf16")
    p, s = [], Listing()
    for name, want in G.WANT.items():
        if bad and name == "attn_pv_d64_ref":
            continue
        p.append(entry(name, arch, regs=40, used=not (bad and name == "attn_qk_d64_h8"), spills=8 if bad and name == "attn_pv_d128" else 0))
        s.fn(name)
        s.ins("HFMA2.MMA R4, -RZ, RZ, 0, 0")
        s.ins("FSEL R5, R5, RZ, !P0")
        s.ins("HMMA.16816.F32.BF16 R4, R8, R12, R4", (want or 1) - (bad and name == "attn_qk_d64"))
        if bad and name == "attn_pv_d64":
            s.ins("FFMA R3, R4, R5, R6")
            s.ins("FSEL R3, -R4, R5, P0")
        if bad and name == "attn_qk_d128":
            s.ins("HMMA.1688.F32.TF32 R4, R8, R12, R4")
            s.ins("IGMMA.64x8x32.S8.S8 R4, R8")
        if not (bad and name == "attn_pv_d128_s16"):
            s.ins("F2FP.BF16.F32.PACK_AB R3, R5, R4")
        s.ins("EXIT")
    if bad:
        p.append("ptxas warning : Registers are spilled\n")
    return "".join(p), s.text()


def real(gate: str, listing: str) -> tuple[str, str]:
    sass = (FIX / "sass_gate" / f"{listing}.cuobjdump.txt").read_text()
    names = dict.fromkeys(re.findall(r"Function : (\w+)", sass))
    return "".join(entry(n, REAL_ARCH[gate], regs=24 + 2 * i, smem=0) for i, n in enumerate(names)), sass


def cases() -> list[tuple[str, str]]:
    out = []
    for gate, (_, arch) in GATES.items():
        out += [(gate, "pass"), (gate, "refused")] + ([(gate, "pass_sm120a")] if arch else []) + [(gate, "plain"), (gate, "fast_math")]
    return out


def run(gate: str, case: str, tmp: Path) -> dict:
    """The gate's CLI on the case, as build.sh runs it: its exit code and the sha256s of OUT_JSON, stdout and stderr."""
    path, takes_arch = GATES[gate]
    if case in ("plain", "fast_math"):
        arch = REAL_ARCH[gate]
        ptxas, sass = real(gate, case)
    else:
        arch = "sm_120a" if case == "pass_sm120a" else "sm_90a"
        ptxas, sass = globals()[gate](arch, case == "refused")
    (tmp / "ptxas.txt").write_text(ptxas)
    (tmp / "sass.txt").write_text(sass)
    out = tmp / "out.json"
    out.unlink(missing_ok=True)
    argv = [sys.executable, str(ROOT / path)] + ([arch] if takes_arch else []) + [str(tmp / "ptxas.txt"), str(tmp / "sass.txt"), str(out)]
    r = subprocess.run(argv, capture_output=True, text=True, cwd=tmp)
    digest = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    return {"rc": r.returncode, "out_json": digest(out.read_bytes()) if out.exists() else None,
            "stdout": digest(r.stdout.encode()), "stderr": digest(r.stderr.encode()), "_said": (r.stdout + r.stderr)[:3000]}


@pytest.mark.parametrize("gate, case", cases())
def test_the_gates_verdict_and_out_json_are_pinned(gate, case, tmp_path):
    got = run(gate, case, tmp_path)
    said = got.pop("_said")
    assert got == json.loads(VERDICTS.read_text())[gate][case], said


def test_sass_functions_takes_each_functions_lines_and_a_repeated_header_starts_over():
    H = _load("pearl_c_sm90").H
    listing = "code for sm_90a\n  Function : a\n  /*0000*/ X ;\n  Function : b\n  /*0000*/ Y ;\n  Function : a\n  /*0000*/ Z ;\n"
    assert H.sass_functions(listing) == {"a": ["  /*0000*/ Z ;"], "b": ["  /*0000*/ Y ;"]}


def test_every_case_is_pinned_and_every_pin_is_a_case():
    pins = json.loads(VERDICTS.read_text())
    assert sorted((g, c) for g in pins for c in pins[g]) == sorted(cases())
    assert {g: sorted(c for c in pins[g] if pins[g][c]["rc"] == 0) for g in pins} == {
        g: ["pass"] + (["pass_sm120a"] if GATES[g][1] else []) for g in GATES}


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    import tempfile
    pins: dict = {}
    with tempfile.TemporaryDirectory() as d:
        for gate, case in cases():
            v = run(gate, case, Path(d))
            v.pop("_said")
            pins.setdefault(gate, {})[case] = v
    VERDICTS.parent.mkdir(parents=True, exist_ok=True)
    VERDICTS.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
