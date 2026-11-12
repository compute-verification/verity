"""Pearl-C4's offline gate (`pearl_c4/sass.py`) by its command line, on a ptxas log and a SASS listing written here: the
kernels the cubin must hold pass with the tensor-core and cast instructions each is meant to issue, and each defect the
gate names (a warning, a spill, a stack frame, local memory, an FP32 instruction with .FTZ, the other format's OMMA, a
count that is no whole number of copies, a missing kernel) is refused with its reason, exit 1 and the same JSON."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "pearl_c4" / "sass.py"

OMMA = {"NV": "OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X", "MX": "OMMA.SF.16864.F32.E2M1.E2M1.E8"}
HMMA, E2M1 = "HMMA.16816.F32.BF16", "F2FP.SATFINITE.E2M1"
GEMMS = [f"gemm_{f}_{m}{h}" for f in ("nv", "mx") for m in ("hash", "nohash", "plain") for h in ("", "_m64", "_m64n64")]
CHAINS = [f"chain32_{f}{r}" for f in ("nv", "mx") for r in ("", "_r2")]
FORMS = ["form_nv", "form_mx", "form_nv_v0", "form_nv_v1"]
OTHERS = ["lines", "transpose", "commit_rows", "pb", "leaves", "form_nv_smem", "form_mx_smem"]


def issued(name: str) -> list[str]:
    """One copy of what a kernel of this name is meant to issue: a GEMM's 32 OMMA of its format and, unless plain, its
    peel's 32 HMMA; a chain's 8 OMMA; forming's 8 NVFP4 OMMA (the noise atoms, in either format) and 16 E2M1 casts."""
    fmt = "NV" if "_nv" in name else "MX"
    if name in GEMMS:
        return [OMMA[fmt]] * 32 + ([] if "_plain" in name else [HMMA] * 32)
    if name in CHAINS:
        return [OMMA[fmt]] * 8
    return [OMMA["NV"]] * 8 + [E2M1] * 16 if name in FORMS else []


def cubin() -> dict[str, list[str]]:
    return {name: issued(name) for name in GEMMS + CHAINS + FORMS + OTHERS}


def texts(kernels: dict[str, list[str]], frames: dict[str, tuple[int, int, int]]) -> tuple[str, str]:
    """A `ptxas -v` log and a `cuobjdump -sass` listing of these kernels (name -> its instructions' opcodes); `frames`
    gives a kernel's (stack frame, spill stores, spill loads) in bytes where they are not zero."""
    ptxas, sass = [], []
    for i, (name, ops) in enumerate(kernels.items()):
        stack, stores, loads = frames.get(name, (0, 0, 0))
        ptxas += [f"ptxas info    : Compiling entry function '{name}' for 'sm_120a'",
                  f"ptxas info    : Function properties for '{name}'",
                  f"ptxas         . {stack} bytes stack frame, {stores} bytes spill stores, {loads} bytes spill loads",
                  f"ptxas info    : Used {40 + i} registers, used 1 barriers"]
        sass += [f"\t\tFunction : {name}", '\t.headerflags\t@"EF_CUDA_SM120"']
        sass += [f"        /*{16 * j:04x}*/                   {op} R0, R1, R2 ;" for j, op in enumerate(ops)]
    return "\n".join(ptxas) + "\n", "\n".join(sass) + "\n"


def gate(tmp_path: Path, kernels: dict[str, list[str]], frames: dict[str, tuple[int, int, int]] | None = None,
         log: str = "") -> tuple[int, dict, str, str]:
    """Run the gate as `build.sh` does; its exit status, the JSON it wrote, and what it printed."""
    ptxas, sass = texts(kernels, frames or {})
    (tmp_path / "ptxas.txt").write_text(ptxas + log)
    (tmp_path / "pearl_c4.sass").write_text(sass)
    p = subprocess.run([sys.executable, str(GATE), str(tmp_path / "ptxas.txt"), str(tmp_path / "pearl_c4.sass"),
                        str(tmp_path / "sass.json")], capture_output=True, text=True)
    return p.returncode, json.loads((tmp_path / "sass.json").read_text()), p.stdout, p.stderr


def test_the_cubin_s_kernels_pass(tmp_path):
    kernels = cubin()
    rc, out, stdout, stderr = gate(tmp_path, kernels)
    assert (rc, out["problems"], stderr) == (0, [], "")
    assert sorted(out["kernels"]) == sorted(kernels) and len(kernels) == 33
    assert out["kernels"]["gemm_mx_hash"] == {
        "regs": 40 + list(kernels).index("gemm_mx_hash"), "stack": 0, "spill_stores": 0, "spill_loads": 0,
        "OMMA_NV": 0, "OMMA_MX": 32, "HMMA": 32, "E2M1": 0, "LDL": 0, "STL": 0, "FTZ": 0}
    assert out["kernels"]["form_mx"]["OMMA_NV"] == 8 and out["kernels"]["form_mx"]["E2M1"] == 16
    assert stdout.startswith("pearl_c4.cubin: 33 kernels, no local memory; gemm_mx_hash ")
    assert f"gemm_nv_plain {40 + list(kernels).index('gemm_nv_plain')} regs 32 OMMA 0 HMMA" in stdout
    assert stdout.count(" regs ") == len(GEMMS)


def test_an_unrolled_loop_counts_each_copy(tmp_path):
    kernels = cubin()
    kernels["gemm_nv_hash"] = issued("gemm_nv_hash") * 2
    kernels["form_nv"] = issued("form_nv") * 3
    rc, out, _, _ = gate(tmp_path, kernels)
    assert (rc, out["problems"]) == (0, [])
    assert (out["kernels"]["gemm_nv_hash"]["OMMA_NV"], out["kernels"]["form_nv"]["E2M1"]) == (64, 48)


def _more(name: str, *ops: str):
    return lambda k: k[name].extend(ops)


def _fewer(name: str, n: int):
    return lambda k: k.__setitem__(name, k[name][n:])


REFUSED = {
    "spill": (None, {"gemm_nv_hash": (0, 8, 8)}, "", "gemm_nv_hash: local memory {"),
    "stack frame": (None, {"pb": (16, 0, 0)}, "", "pb: local memory {"),
    "local load": (_more("lines", "LDL"), {}, "", "lines: local memory {"),
    "local store": (_more("commit_rows", "STL"), {}, "", "commit_rows: local memory {"),
    "ftz": (_more("transpose", "FADD.FTZ", "FFMA.FTZ", "FMUL.FTZ"), {}, "",
            "transpose: 3 FP32 FADD, FFMA or FMUL with .FTZ (a shipped kernel has none)"),
    "the other format's OMMA": (_more("gemm_nv_hash", OMMA["MX"]), {}, "", "gemm_nv_hash: 1 OMMA_MX (want none)"),
    "part of a copy": (_fewer("gemm_mx_nohash", 1), {}, "",
                       "gemm_mx_nohash: 31 OMMA_MX (want a multiple of 32, at least 32)"),
    "a peel in a plain GEMM": (_more("gemm_nv_plain", HMMA), {}, "", "gemm_nv_plain: 1 HMMA (want none)"),
    "a chain with no OMMA": (_fewer("chain32_nv", 8), {}, "",
                             "chain32_nv: 0 OMMA_NV (want a multiple of 8, at least 8)"),
    "forming in MXFP4's atoms": (lambda k: k.__setitem__("form_mx", [OMMA["MX"]] * 8 + [E2M1] * 16), {}, "",
                                 "form_mx: 0 OMMA_NV (want a multiple of 8, at least 8)"),
    "forming without its casts": (lambda k: k.__setitem__("form_nv_v1", [OMMA["NV"]] * 8), {}, "",
                                  "form_nv_v1: 0 E2M1 (want a multiple of 16, at least 16)"),
    "a missing kernel": (lambda k: k.pop("chain32_mx_r2"), {}, "", "chain32_mx_r2: missing"),
    "a warning": (None, {}, "ptxas warning : Stack size for entry function 'pb' cannot be statically determined\n",
                  "ptxas warning : Stack size for entry function 'pb' cannot be statically determined"),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_a_defect_is_refused_with_its_reason(tmp_path, case):
    change, frames, log, reason = REFUSED[case]
    kernels = cubin()
    if change:
        change(kernels)
    rc, out, stdout, stderr = gate(tmp_path, kernels, frames, log)
    first = out["problems"][0]
    assert rc == 1 and stdout == "" and first.startswith(reason)
    assert stderr == "REFUSED: " + "; ".join(out["problems"]) + "\n"
    extra = {"forming in MXFP4's atoms": ["form_mx: 8 OMMA_MX (want none)"]}.get(case, [])
    assert out["problems"][1:] == extra
