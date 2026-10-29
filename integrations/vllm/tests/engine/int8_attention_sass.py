"""The int8 attention's two Triton kernels (`int8_attention_triton.kv_rows` and `lut_rows`) held to Int8KvQuant_v1 and
Int8LutAttentionRow_v1 at the PTX and SASS level, as `int8_fused.ptx_check` holds the fused linears at the PTX level and
benchmarks/pouw/harness/sass_gate.py holds the PoUW kernels' SASS.  Each kernel is compiled by Triton for sm_120 and sm_90
(no GPU needed: Triton's own ptxas and nvdisasm) under `int8_fused.OPTS`, the options it is launched with, or taken from a
GPU process's JIT cache after a launch (`launched`).

The Definitions round every product before it is added (lut_rows' epilogue RN32(RN32(f32(Y) RN32(1/L)) + RN32(z f32(V'(0))))
times s_v, and A as mul.rn.f64 left to right); kv_rows computes no product and no sum in floating point at all.  So:

  PTX   no float fma or mad (a contraction LLVM made); every float mul, add and sub carries an explicit rounding mode (one
        without may be contracted by ptxas); no .ftz, .approx or div.full, and no ex2, lg2, sin, cos, tanh, sqrt or rsqrt;
        div.rn.f32 and cvt.rni.s32.f32 present, and for lut_rows cvt.rni.s64.f64, mul.rn.f64, mul.rn.f32, add.rn.f32 and
        an s8 x s8 mma.
  fmad  ptxas builds the same SASS from the kernel's PTX with contraction allowed as with --fmad=false: nothing in the
        PTX is left uncontracted only because Triton passes the flag.
  SASS  outside ptxas's IEEE subroutines (div.rn's slow path, 64-bit integer division) the only .FTZ or FP32 MUFU
        instructions are MUFU.RCP and integer division's F2I(.FTZ).U32/U64.TRUNC seed, with no more MUFU.RCP than div.rn
        checks (FCHK) and seeds; no DFMA or DADD and no float MMA; and the products and sums are the Definition's, each
        its own instruction: kv_rows has no FMUL and no FADD, lut_rows exactly three FMUL and one FADD per bf16 output
        (FADD x, -RZ is an exact abs or move and doesn't count), at least one DMUL and an integer MMA.  ptxas's FFMAs
        are then div.rn's Newton steps: the only FFMA a contraction-free build has.

The slow-path subroutines are taken by name, not by the harness's pinned signatures: Triton bundles ptxas 12.8 (sm_90) and
13.1 (sm_120), which `sass_pins.json` doesn't pin, so their exactness on subnormals rests on ptxas's IEEE div.rn, not on a
numeric pin here.

    cd integrations/vllm; PYTHONPATH=.:../.. python -m tests.engine.int8_attention_sass --out DIR [--launch]

writes ``<out>/int8_attention_sass.json``: each kernel variant per architecture, the negatives (lut_rows compiled with
contraction allowed, its PTX with the rounding modes stripped through ptxas, kv_rows dividing with Triton's `/`) refused, and
kv_rows with contraction allowed the same SASS; with --launch (a GPU) also every kv_rows and lut_rows the process launched on
the batched int8 attention's conformance cases.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA = "verity-vllm/int8-attention-sass/v1"
HARNESS = Path(__file__).resolve().parents[4] / "benchmarks" / "pouw" / "harness" / "sass_gate.py"
ARCHES = (120, 90)
#: ptxas's IEEE subroutines (div.rn.f32's slow path; div.s64 / div.u64 / rem for integer division by a 64-bit divisor)
IEEE_SUBROUTINES = frozenset({"__cuda_sm3x_div_rn_noftz_f32_slowpath", "__cuda_sm20_div_s64", "__cuda_sm20_div_u64",
                              "__cuda_sm20_rem_s64", "__cuda_sm20_rem_u64", "__cuda_sm20_div_u16"})
FLOAT = re.compile(r"\.(?:b?f16(?:x2)?|f32|f64)\b")
ROUNDING = frozenset({"rn", "rz", "rm", "rp"})
TRANSCENDENTAL = frozenset({"ex2", "lg2", "sin", "cos", "tanh", "sqrt", "rsqrt"})
NEED = {"kv_rows": ("div.rn.f32", "cvt.rni.s32.f32"),
        "lut_rows": ("div.rn.f32", "cvt.rni.s32.f32", "cvt.rni.s64.f64", "mul.rn.f64", "mul.rn.f32", "add.rn.f32", ".s8.s8")}

KV_SIG = {"k_ptr": "*bf16", "v_ptr": "*bf16", "bt_ptr": "*i32", "start_ptr": "*i64", "len_ptr": "*i64", "ks_ptr": "*fp32",
          "kq_ptr": "*i8", "vq_ptr": "*i8", "sv_ptr": "*fp32", "v0_ptr": "*fp32", "floor": "fp32", "D": "i32", "S_MAX": "i32",
          "BS": "i32", "sk_n": "i32", "sk_o": "i32", "sk_g": "i32", "sv_n": "i32", "sv_o": "i32", "sv_g": "i32", "sbt": "i32",
          "sq_b": "i32", "sq_g": "i32", "sq_s": "i32", "ss_b": "i32", "ss_g": "i32",
          "PAGED": "constexpr", "BLOCK_S": "constexpr", "BLOCK_C": "constexpr"}
LUT_SIG = {"q_ptr": "*bf16", "k_ptr": "*i8", "v_ptr": "*i8", "sv_ptr": "*fp32", "v0_ptr": "*fp32", "ks_ptr": "*fp32",
           "req_ptr": "*i64", "pos_ptr": "*i64", "f64_ptr": "*fp64", "lut_ptr": "*i32", "o_ptr": "*bf16", "floor": "fp32",
           "H": "i32", "D": "i32", "S_MAX": "i32", "REP": "i32", "sq_t": "i32", "sq_h": "i32", "sk_b": "i32", "sk_g": "i32",
           "sk_s": "i32", "ss_b": "i32", "ss_g": "i32", "so_t": "i32", "so_h": "i32",
           "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_D": "constexpr"}


class ToolMissing(RuntimeError):
    pass


@functools.cache
def harness() -> ModuleType:
    """benchmarks/pouw/harness/sass_gate.py: its nvdisasm parser, `Function` (subroutine regions) and `flagged`."""
    spec = importlib.util.spec_from_file_location("pouw_harness_sass_gate", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tool(name: str, cap: int = 120) -> str:
    """Triton's bundled CUDA tool: nvdisasm, or the ptxas Triton compiles `cap` with."""
    import triton
    from triton.backends.nvidia import compiler as nv
    t = nv.get_ptxas(cap) if name == "ptxas" else getattr(triton.knobs.nvidia, name)
    path = getattr(t, "path", None)
    if not path or not Path(path).is_file():
        raise ToolMissing(f"Triton has no {name} for sm_{cap}")
    return path


def variants(T: ModuleType) -> dict[str, tuple[Any, dict, dict]]:
    """Each kernel as `int8_attention_triton` launches it: kv_rows paged and packed, lut_rows at head dims 128 and 64."""
    kv = {"BLOCK_S": T.BLOCK_S, "BLOCK_C": T.BLOCK_C}
    return {"kv_rows/paged": (T.kv_rows, KV_SIG, {"PAGED": True, **kv}),
            "kv_rows/packed": (T.kv_rows, KV_SIG, {"PAGED": False, **kv}),
            "lut_rows/d128": (T.lut_rows, LUT_SIG, {"BLOCK_M": T.BLOCK_M, "BLOCK_N": T.BLOCK_N, "BLOCK_D": 128}),
            "lut_rows/d64": (T.lut_rows, LUT_SIG, {"BLOCK_M": T.BLOCK_M, "BLOCK_N": T.BLOCK_N, "BLOCK_D": 64})}


def compile_variant(T: ModuleType, variant: str, cap: int, **opts):
    """Triton's CompiledKernel of one variant for sm_`cap` under `int8_fused.OPTS` (`opts` overriding them)."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    fn, sig, consts = variants(T)[variant]
    return triton.compile(ASTSource(fn, sig, consts), target=GPUTarget("cuda", cap, 32), options={**T.OPTS, **opts})


def ptx_ops(ptx: str) -> Counter:
    """The PTX instructions' opcodes (with their modifiers), counted."""
    ops: Counter = Counter()
    for line in ptx.splitlines():
        s = re.sub(r"^@!?%\w+\s+", "", line.split("//")[0].strip())
        if s and s[0].isalpha() and not s.endswith(":"):
            ops[s.split(None, 1)[0].rstrip(";")] += 1
    return ops


def ptx_rules(kernel: str, ptx: str) -> dict[str, Any]:
    ops = ptx_ops(ptx)
    fused = {o: n for o, n in ops.items() if o.split(".")[0] in ("fma", "mad") and FLOAT.search(o)}
    loose = {o: n for o, n in ops.items() if o.split(".")[0] in ("mul", "add", "sub") and FLOAT.search(o)
             and not ROUNDING & set(o.split(".")[1:])}
    ruled_out = {o: n for o, n in ops.items() if re.search(r"\.(ftz|approx)\b", "." + o) or o.startswith("div.full")
                 or o.split(".")[0] in TRANSCENDENTAL}
    float_mma = {o: n for o, n in ops.items() if o.split(".")[0] in ("mma", "wgmma") and ".s8.s8" not in o and "kind::i8" not in o}
    missing = [n for n in NEED[kernel] if not any(n in o for o in ops)]
    refused = []
    if fused:
        refused.append(f"PTX: a product contracted into an add {fused}")
    if loose:
        refused.append(f"PTX: float mul/add/sub without a rounding mode, which ptxas may contract {loose}")
    if ruled_out:
        refused.append(f"PTX: flush to zero, an approximation or a function the Definition doesn't compute {ruled_out}")
    if float_mma:
        refused.append(f"PTX: a float MMA {float_mma}")
    if missing:
        refused.append(f"PTX: none of the Definition's {missing}")
    return {"fma": fused, "unrounded": loose, "ruled_out": ruled_out, "float_mma": float_mma, "missing": missing,
            "census": {o: n for o, n in sorted(ops.items()) if FLOAT.search(o) or o.split(".")[0] in ("div", "mma", "wgmma")},
            "refused": refused}


def listing(cubin: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "k.cubin"
        p.write_bytes(cubin)
        r = subprocess.run([tool("nvdisasm"), "-c", str(p)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"nvdisasm exited {r.returncode}: {r.stderr[-400:]}")
    return r.stdout


def _int_seed(op: list[str]) -> bool:
    return op[0] in ("F2I", "UF2I") and "TRUNC" in op and bool({"U32", "U64"} & set(op))


def sass_rules(kernel: str, text: str) -> dict[str, Any]:
    G = harness()
    c: Counter = Counter()
    flagged_out, foreign, mma = Counter(), Counter(), Counter()
    for fd in G.parse_nvdisasm(text):
        fn = G.Function(fd)
        sub = {k: name for name, s, e in fn.subroutines.values() for k in range(s, e)}
        for k, ins in enumerate(fn.insns):
            op = ins.op
            if k in sub:
                if G.flagged(op) and sub[k] not in IEEE_SUBROUTINES:
                    foreign[f"{sub[k]}: {ins.name}"] += 1
                continue
            if G.flagged(op):
                if ins.name == "MUFU.RCP":
                    c["MUFU.RCP"] += 1
                elif _int_seed(op):
                    c["seed"] += 1
                else:
                    flagged_out[ins.name] += 1
            if op[0] == "FADD":
                c["FADD" if not (len(ins.args) == 3 and ins.args[2] in ("RZ", "-RZ")) else "FADD.zero"] += 1
            elif op[0] in ("FMUL", "FFMA", "DMUL", "DFMA", "DADD", "FCHK"):
                c[op[0]] += 1
            elif op[0] == "F2FP" and "BF16" in op:
                c["bf16"] += 2 if "PACK_AB" in op else 1
            elif op[0] == "F2F" and op[1:2] == ["BF16"]:
                c["bf16"] += 1
            if "MMA" in op[0]:
                mma[ins.name] += 1
    refused = []
    if flagged_out:
        refused.append(f"SASS: .FTZ or FP32 MUFU outside div.rn and integer division {dict(flagged_out)}")
    if foreign:
        refused.append(f"SASS: .FTZ or FP32 MUFU in a subroutine that isn't ptxas's IEEE division {dict(foreign)}")
    if c["MUFU.RCP"] > c["FCHK"] + c["seed"]:
        refused.append(f"SASS: {c['MUFU.RCP']} MUFU.RCP for {c['FCHK']} div.rn checks and {c['seed']} integer seeds (an "
                       "approximate reciprocal)")
    if c["DFMA"] or c["DADD"]:
        refused.append(f"SASS: DFMA {c['DFMA']} / DADD {c['DADD']} (A is mul.rn.f64 only)")
    float_mma = {m: n for m, n in mma.items() if not m.startswith(("IMMA", "IGMMA"))}
    if float_mma:
        refused.append(f"SASS: a float MMA {float_mma}")
    if kernel == "kv_rows":
        if c["FMUL"] or c["FADD"] or c["DMUL"] or mma:
            refused.append(f"SASS: kv_rows computes FMUL {c['FMUL']} / FADD {c['FADD']} / DMUL {c['DMUL']} / MMA {dict(mma)}, "
                           "where Int8KvQuant_v1 has no product or sum in floating point")
    else:
        n = c["bf16"]
        if not n or c["FMUL"] != 3 * n or c["FADD"] != n:
            refused.append(f"SASS: {c['FMUL']} FMUL and {c['FADD']} FADD for {n} bf16 outputs, where Int8LutAttentionRow_v1 "
                           f"rounds three products and one sum per output ({3 * n} and {n}): a product contracted into its add")
        if not c["DMUL"] or not any(m.startswith(("IMMA", "IGMMA")) for m in mma):
            refused.append(f"SASS: DMUL {c['DMUL']}, integer MMA {dict(mma)}: not A's f64 products and the s8 dots")
    return {"census": dict(sorted(c.items())), "mma": dict(mma), "refused": refused}


def instructions(text: str) -> list[str]:
    G = harness()
    return [t for fd in G.parse_nvdisasm(text) for _, t in fd["insns"]]


def ptxas(ptx: str, cap: int, fmad: bool) -> bytes:
    from triton.backends.nvidia.compiler import sm_arch_from_capability
    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "k.ptx", Path(td) / "k.cubin"
        src.write_text(ptx)
        cmd = [tool("ptxas", cap), "-lineinfo", *([] if fmad else ["--fmad=false"]), f"--gpu-name={sm_arch_from_capability(cap)}",
               str(src), "-o", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"ptxas exited {r.returncode}: {r.stderr[-400:]}")
        return out.read_bytes()


def fmad_rule(ptx: str, cap: int) -> dict[str, Any]:
    on, off = (instructions(listing(ptxas(ptx, cap, f))) for f in (True, False))
    ffma = [sum(t.split()[0].startswith("FFMA") for t in x) for x in (on, off)]
    same = on == off
    return {"same": same, "ffma_contraction_allowed": ffma[0], "ffma_fmad_false": ffma[1],
            "refused": [] if same else [f"fmad: ptxas contracts this PTX when allowed ({ffma[0]} FFMA against {ffma[1]} with "
                                        "--fmad=false)"]}


def gate(kernel: str, ptx: str, cubin: bytes, cap: int) -> dict[str, Any]:
    """The verdict on one compiled kernel: its PTX, ptxas's build of it with and without contraction, and its SASS."""
    p, f, s = ptx_rules(kernel, ptx), fmad_rule(ptx, cap), sass_rules(kernel, listing(cubin))
    refused = p["refused"] + f["refused"] + s["refused"]
    return {"kernel": kernel, "arch": f"sm_{cap}", "ptx": p, "fmad": f, "sass": s, "refused": refused, "ok": not refused}


def gate_compiled(compiled) -> dict[str, Any]:
    return gate(compiled.metadata.name, compiled.asm["ptx"], compiled.asm["cubin"], int(compiled.metadata.target.arch))


def div_full_mutant(where: Path) -> ModuleType:
    """`int8_attention_triton` with `_q8` dividing by Triton's `/` (div.full.f32) in place of tl.div_rn."""
    from verity_vllm.engine import int8_attention_triton as T
    src = Path(T.__file__).read_text()
    old = "tl.div_rn(x, tl.broadcast_to(s, x.shape))"
    if old not in src:
        raise RuntimeError(f"_q8 no longer divides by {old}")
    path = where / "int8_attention_triton_div_full.py"
    path.write_text(src.replace(old, "x / s"))
    spec = importlib.util.spec_from_file_location("int8_attention_triton_div_full", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def unrounded(ptx: str) -> str:
    """The PTX with f32 mul/add/sub's explicit rounding modes stripped: what ptxas may contract."""
    return re.sub(r"\b(mul|add|sub)\.rn\.f32\b", r"\1.f32", ptx)


def negatives(T: ModuleType, where: Path, cap: int) -> dict[str, Any]:
    """Each must be refused, except kv_rows with contraction allowed, which must compile to the same SASS and pass."""
    lut = compile_variant(T, "lut_rows/d128", cap, enable_fp_fusion=True)
    shipped = compile_variant(T, "lut_rows/d128", cap)
    stripped = unrounded(shipped.asm["ptx"])
    kv_on, kv_off = (compile_variant(T, "kv_rows/paged", cap, enable_fp_fusion=f) for f in (True, False))
    mutant = compile_variant(div_full_mutant(where), "kv_rows/paged", cap)
    return {"lut_rows contraction allowed": gate_compiled(lut),
            "lut_rows PTX unrounded through ptxas": gate("lut_rows", stripped, ptxas(stripped, cap, True), cap),
            "kv_rows div.full": gate_compiled(mutant),
            "kv_rows contraction allowed": {**gate_compiled(kv_on), "same_sass": instructions(listing(kv_on.asm["cubin"]))
                                            == instructions(listing(kv_off.asm["cubin"]))}}


def launched() -> list:
    """Every kv_rows and lut_rows this process compiled for a launch (Triton's per-device JIT caches)."""
    from verity_vllm.engine import int8_attention_triton as T
    return [k for fn in (T.kv_rows, T.lut_rows) for entry in fn.device_caches.values() for k in entry[0].values()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="int8 attention's Triton kernels against their Definitions at the PTX and SASS level")
    ap.add_argument("--out", required=True)
    ap.add_argument("--launch", action="store_true", help="also gate the kernels launched on this GPU by the conformance cases")
    a = ap.parse_args(argv)
    from verity_vllm.engine import int8_attention_triton as T
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "triton": __import__("triton").__version__, "compiled": {},
                           "negatives": {}, "launched": []}
    with tempfile.TemporaryDirectory() as td:
        for cap in ARCHES:
            for v in variants(T):
                doc["compiled"][f"{v}/sm_{cap}"] = r = gate_compiled(compile_variant(T, v, cap))
                print(f"[sass] {v} sm_{cap} ok={r['ok']} {r['sass']['census']} {r['refused'][:2]}", flush=True)
            doc["negatives"][f"sm_{cap}"] = neg = negatives(T, Path(td), cap)
            for name, r in neg.items():
                print(f"[negative] {name} sm_{cap} ok={r['ok']} {r.get('same_sass', '')} {[x[:120] for x in r['refused']]}", flush=True)
    if a.launch:
        import torch
        from tests.engine import int8_batch_gpu as B
        for i, case in enumerate([(32, 8, 128, 8, 600, 16, 1.0), (8, 2, 64, 6, 200, 16, 1.0)]):
            B.run_case(7000 + i, case)
        torch.cuda.synchronize()
        doc["device"] = torch.cuda.get_device_name(0)
        for k in launched():
            r = gate_compiled(k)
            doc["launched"].append({**r, "constants": {str(x): str(y) for x, y in getattr(k.src, "constants", {}).items()}
                                    if hasattr(k, "src") else None})
            print(f"[launched] {r['kernel']} {r['arch']} ok={r['ok']} {r['sass']['census']} {r['refused'][:2]}", flush=True)
    neg_ok = all(not r["ok"] for n in doc["negatives"].values() for k, r in n.items() if k != "kv_rows contraction allowed") \
        and all(n["kv_rows contraction allowed"]["ok"] and n["kv_rows contraction allowed"]["same_sass"] for n in doc["negatives"].values())
    doc["ok"] = all(r["ok"] for r in doc["compiled"].values()) and neg_ok and all(r["ok"] for r in doc["launched"]) \
        and (bool(doc["launched"]) or not a.launch)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "int8_attention_sass.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"INT8-ATTN-SASS ok={doc['ok']} compiled={len(doc['compiled'])} negatives_fire={neg_ok} launched={len(doc['launched'])}",
          flush=True)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
