"""Lane tcgen05, stage 0: instruction reality on a datacenter Blackwell part (sm_100 / sm_103), before any semantics.

    python benchmarks/tc_probe_fp4/stage0_sm100.py --out $RESEARCH_RUN_DIR [--arch sm_100a] [--cutlass /workspace/cutlass]

Records, into ``stage0.json`` and a ``bench-result/v1`` ``result.json``:
  * the toolchain and device: nvcc / ptxas versions, driver, device name, compute capability;
  * whether ptxas accepts the sm_120 block-scaled ``mma.sync`` mnemonics for the datacenter target (the PTX ISA says
    ``.kind::mxf4nvf4`` / ``.kind::mxf4`` on ``mma`` are sm_120a / sm_121a only) -- ``mma_fp4.cu`` is compiled with
    ``-arch=<arch>`` and the diagnostic captured; the same for a one-instruction PTX file through ``ptxas`` directly, and the
    sm_120a control (must assemble);
  * what ``tcgen05_fp4.cu`` lowers to on this target: ``cuobjdump -sass`` opcodes per kernel (the UTC*MMA family), the number
    of MMA instructions per kernel (must be one);
  * optionally, the SASS of a CUTLASS sm_100 block-scaled GEMM (example 72a, compiled with nvcc against ``--cutlass``): the
    opcodes its main loop uses, so the probe's instruction is the one production kernels use.
Nothing here executes a tensor-core instruction; stage 1/2 (probe.py --instruction tcgen05_*) do.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MMA_SYNC_PTX = """\
.version 8.7
.target {target}
.address_size 64
.visible .entry k(.param .u64 p) {{
    .reg .b32 %a<4>; .reg .b32 %b<2>; .reg .f32 %c<4>; .reg .f32 %d<4>; .reg .b32 %s<2>;
    mov.b32 %a0, 0; mov.b32 %a1, 0; mov.b32 %a2, 0; mov.b32 %a3, 0; mov.b32 %b0, 0; mov.b32 %b1, 0;
    mov.f32 %c0, 0f00000000; mov.f32 %c1, 0f00000000; mov.f32 %c2, 0f00000000; mov.f32 %c3, 0f00000000;
    mov.b32 %s0, 0; mov.b32 %s1, 0;
    mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
        {{%d0,%d1,%d2,%d3}}, {{%a0,%a1,%a2,%a3}}, {{%b0,%b1}}, {{%c0,%c1,%c2,%c3}}, {{%s0}}, {{0, 0}}, {{%s1}}, {{0, 0}};
    ret;
}}
"""


def run(cmd: list[str], timeout: int = 1800) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": cmd, "rc": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:], "seconds": round(time.time() - t0, 1)}
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:  # noqa: PERF203
        return {"cmd": cmd, "rc": None, "error": repr(e), "seconds": round(time.time() - t0, 1)}


def cuda_bin(name: str) -> str:
    return shutil.which(name) or f"/usr/local/cuda/bin/{name}"


def gencode(arch: str) -> str:
    """explicit compute_XXXa PTX + sm_XXXa SASS (nvcc 12.9's -arch=sm_100a also embeds compute_100 PTX, which tcgen05 refuses)"""
    m = re.fullmatch(r"sm_(\d+)a", arch)
    return f"-gencode=arch=compute_{m.group(1)}a,code=sm_{m.group(1)}a" if m else f"-arch={arch}"


def sass_opcodes(cubin_or_so: Path) -> dict:
    """{function: {mma_count, mma_opcodes, tcgen05_opcodes, instructions}} from cuobjdump -sass."""
    text = subprocess.run([cuda_bin("cuobjdump"), "-sass", str(cubin_or_so)], capture_output=True, text=True).stdout
    out: dict = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1)
            out[fn] = {"mma_count": 0, "mma_opcodes": [], "tcgen05_opcodes": [], "instructions": 0}
            continue
        if fn is None or "*/" not in line:
            continue
        body = line.split("*/", 1)[1].strip()
        if not body or body.startswith("/*"):
            continue
        out[fn]["instructions"] += 1
        op = body.split()[0].rstrip(";")
        if op.startswith("@"):
            parts = body.split()
            op = parts[1].rstrip(";") if len(parts) > 1 else op
        if re.search(r"MMA", op):
            out[fn]["mma_count"] += 1
            if op not in out[fn]["mma_opcodes"]:
                out[fn]["mma_opcodes"].append(op)
        elif op.startswith("UTC") or op.startswith("UTMA") or op.startswith("UBLK") or "TMEM" in op:
            if op not in out[fn]["tcgen05_opcodes"]:
                out[fn]["tcgen05_opcodes"].append(op)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="sm_100a", help="the datacenter target under test (sm_100a on a B200, sm_103a on a B300)")
    ap.add_argument("--cutlass", default=None, help="CUTLASS checkout; compiles examples/72_blackwell_narrow_precision_gemm/72a for its SASS")
    args = ap.parse_args()
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rec: dict = {"schema": "tcgen05/stage0/v1", "arch": args.arch, "hostname": platform.node()}

    rec["nvcc_version"] = run([cuda_bin("nvcc"), "--version"])
    rec["ptxas_version"] = run([cuda_bin("ptxas"), "--version"])
    rec["nvidia_smi"] = run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid,memory.total", "--format=csv,noheader"])
    rec["nvidia_smi_full"] = run(["nvidia-smi"])

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # (a) the sm_120 kernel source against the datacenter target: expected to be refused by ptxas
        rec["mma_sync_kernel_on_arch"] = run([cuda_bin("nvcc"), "-O2", gencode(args.arch), "-Xcompiler", "-fPIC", "-shared",
                                              "-o", str(tdp / "mma_fp4.so"), str(HERE / "mma_fp4.cu")])
        # (b) one mma.sync instruction as PTX through ptxas: datacenter target, then the sm_120a control
        for target in (args.arch, "sm_120a"):
            ptx = tdp / f"mma_{target}.ptx"
            ptx.write_text(MMA_SYNC_PTX.format(target=target))
            rec[f"ptxas_mma_sync_{target}"] = run([cuda_bin("ptxas"), f"-arch={target}", str(ptx), "-o", str(tdp / f"mma_{target}.cubin")])
        # (c) the tcgen05 probe kernel on this target: must build; its SASS is the instruction under test.  Also recorded: what the
        #     plain `-arch=<arch>` spelling does with this nvcc (12.9 adds a compute_100 PTX stage that refuses tcgen05).
        rec["tcgen05_kernel_build_plain_arch"] = run([cuda_bin("nvcc"), "-O2", f"-arch={args.arch}", "-Xcompiler", "-fPIC", "-shared",
                                                      "-o", str(tdp / "plain.so"), str(HERE / "tcgen05_fp4.cu")])
        so = out / f"libtcgen05_fp4_{args.arch}.so"
        rec["tcgen05_kernel_build"] = run([cuda_bin("nvcc"), "-O2", gencode(args.arch), "-Xcompiler", "-fPIC", "-shared", "-o", str(so),
                                           str(HERE / "tcgen05_fp4.cu")])
        if rec["tcgen05_kernel_build"]["rc"] == 0:
            rec["tcgen05_kernel_sass"] = sass_opcodes(so)
            ptx_out = out / f"tcgen05_fp4_{args.arch}.ptx"
            run([cuda_bin("nvcc"), "-O2", gencode(args.arch), "-ptx", "-o", str(ptx_out), str(HERE / "tcgen05_fp4.cu")])
            if ptx_out.exists():
                rec["tcgen05_kernel_ptx_mnemonics"] = sorted({m.group(0) for m in re.finditer(r"tcgen05\.[A-Za-z0-9_.:]+", ptx_out.read_text())})
        # (d) CUTLASS example 72a: the production block-scaled GEMM's SASS
        if args.cutlass:
            cut = Path(args.cutlass)
            src = cut / "examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu"
            if src.exists():
                obj = tdp / "72a.o"
                rec["cutlass_72a_build"] = run([cuda_bin("nvcc"), "-O2", gencode(args.arch), "-std=c++17", "--expt-relaxed-constexpr",
                                                "-I", str(cut / "include"), "-I", str(cut / "tools/util/include"), "-I", str(cut / "examples/common"),
                                                "-c", "-o", str(obj), str(src)], timeout=3600)
                if rec["cutlass_72a_build"]["rc"] == 0:
                    sass = sass_opcodes(obj)
                    # keep only functions with MMAs (the mainloop kernels), full opcode sets
                    rec["cutlass_72a_sass"] = {fn: v for fn, v in sass.items() if v["mma_count"]}
                    rec["cutlass_72a_functions"] = len(sass)
                git = run(["git", "-C", str(cut), "describe", "--tags", "--always"])
                rec["cutlass_version"] = git.get("stdout", "").strip()
            else:
                rec["cutlass_72a_build"] = {"error": f"{src} not found"}

    rec["seconds"] = round(time.time() - t_all, 1)
    (out / "stage0.json").write_text(json.dumps(rec, indent=1))

    # the findings in one place
    findings = {
        "mma_sync_mxf4nvf4_assembles_for_arch": rec[f"ptxas_mma_sync_{args.arch}"]["rc"] == 0,
        "mma_sync_mxf4nvf4_assembles_for_sm_120a": rec["ptxas_mma_sync_sm_120a"]["rc"] == 0,
        "mma_sync_kernel_builds_for_arch": rec["mma_sync_kernel_on_arch"]["rc"] == 0,
        "ptxas_diagnostic_on_arch": (rec[f"ptxas_mma_sync_{args.arch}"].get("stderr") or "").strip().splitlines()[:3],
        "tcgen05_kernel_builds": rec["tcgen05_kernel_build"]["rc"] == 0,
        "tcgen05_kernel_builds_with_plain_arch_flag": rec["tcgen05_kernel_build_plain_arch"]["rc"] == 0,
        "plain_arch_first_diagnostic": (rec["tcgen05_kernel_build_plain_arch"].get("stderr") or "").strip().splitlines()[:1],
        "tcgen05_kernel_sass": {fn: {"mma_count": v["mma_count"], "mma_opcodes": v["mma_opcodes"], "tcgen05_opcodes": v["tcgen05_opcodes"]}
                                for fn, v in rec.get("tcgen05_kernel_sass", {}).items()},
        "cutlass_72a_mma_opcodes": sorted({op for v in rec.get("cutlass_72a_sass", {}).values() for op in v["mma_opcodes"]}) if "cutlass_72a_sass" in rec else None,
    }
    rec["findings"] = findings
    (out / "stage0.json").write_text(json.dumps(rec, indent=1))
    tile_kernels = {fn: v for fn, v in rec.get("tcgen05_kernel_sass", {}).items() if "tiles_kernel" in fn}
    ok = bool(findings["tcgen05_kernel_builds"]) and bool(tile_kernels) and all(v["mma_count"] == 1 for v in tile_kernels.values())
    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        smi = (rec["nvidia_smi"].get("stdout") or "").strip().split(",")
        result = {
            "schema": "research/result/v0.1", "kind": "bench-result/v1", "run_id": run_id,
            "stage": "tcgen05/stage0", "arch": args.arch, "sku": smi[0].strip() if smi else None, "driver": smi[1].strip() if len(smi) > 1 else None,
            "compute_capability": smi[2].strip() if len(smi) > 2 else None,
            "nvcc": (rec["nvcc_version"].get("stdout") or "").strip().splitlines()[-1:],
            "findings": findings,
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/stage0_sm100.py", "arch": args.arch, "device": smi[0].strip() if smi else None},
            "validation": {"status": "passed" if ok else "failed", "checker": "stage0: tcgen05 kernel builds, one MMA per tile kernel",
                           "detail": json.dumps(findings["tcgen05_kernel_sass"])[:1500], "evidence": ["stage0.json"]},
            "measurements": [{"name": "wall", "value": rec["seconds"], "unit": "s"}],
            "measurement_files": ["stage0.json"],
            "artifacts": [p.name for p in out.iterdir() if p.suffix in (".ptx",)],
        }
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out / "result.json")
    print(json.dumps(findings, indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
