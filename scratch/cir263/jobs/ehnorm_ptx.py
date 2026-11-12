"""Compile vLLM d9105ea80's `_fused_eh_norm_kernel` (deepseek_v32/common/kernels.py:25-31 and 987-1019, copied verbatim) for
sm_90 with the Triton this process imports, at the specialisation GLM-5.3's draft step launches it with, and write its TTGIR,
PTX and SASS under OUT.  No GPU is used."""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _rms_norm(x, w, eps, HIDDEN_SIZE: tl.constexpr):
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    rrms = tl.rsqrt(mean_sq + eps)
    w = w.to(tl.float32)
    return (x * rrms) * w


@triton.jit
def _fused_eh_norm_kernel(
    pos_ptr,
    embeds_ptr,
    embeds_stride,
    prev_ptr,
    prev_stride,
    enorm_w_ptr,
    hnorm_w_ptr,
    eps,
    out_ptr,
    out_stride,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """MTP input fusion: zero embeds at position 0, RMSNorm(embeds) with enorm
    and RMSNorm(prev_hidden) with hnorm, written side-by-side into ``out``
    ([N, 2H]) ready for the eh_proj GEMM. Replaces where + 2x RMSNorm + cat."""
    tok = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    mask = off < H

    pos = tl.load(pos_ptr + tok)
    e = tl.load(embeds_ptr + tok * embeds_stride + off, mask=mask, other=0.0)
    e = tl.where(pos == 0, 0.0, e.to(tl.float32))
    ew = tl.load(enorm_w_ptr + off, mask=mask)
    e_normed = _rms_norm(e, ew, eps, H)
    tl.store(out_ptr + tok * out_stride + off, e_normed, mask=mask)

    p = tl.load(prev_ptr + tok * prev_stride + off, mask=mask, other=0.0)
    hw = tl.load(hnorm_w_ptr + off, mask=mask)
    p_normed = _rms_norm(p, hw, eps, H)
    tl.store(out_ptr + tok * out_stride + H + off, p_normed, mask=mask)


SIG = {"pos_ptr": "*i64", "embeds_ptr": "*bf16", "embeds_stride": "i32", "prev_ptr": "*bf16", "prev_stride": "i32",
       "enorm_w_ptr": "*bf16", "hnorm_w_ptr": "*bf16", "eps": "fp32", "out_ptr": "*bf16", "out_stride": "i32",
       "H": "constexpr", "BLOCK": "constexpr"}
DIV16 = ("pos_ptr", "embeds_ptr", "embeds_stride", "prev_ptr", "prev_stride", "enorm_w_ptr", "hnorm_w_ptr", "out_ptr", "out_stride")


def source(H: int, BLOCK: int) -> ASTSource:
    names = list(SIG)
    attrs = {(names.index(n),): [["tt.divisibility", 16]] for n in DIV16}
    return ASTSource(fn=_fused_eh_norm_kernel, signature=SIG, constexprs={(names.index("H"),): H, (names.index("BLOCK"),): BLOCK},
                     attrs=attrs)


def sass(cubin: bytes, work: Path) -> str:
    nv = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
    path = work / "k.cubin"
    path.write_bytes(cubin)
    for tool, args in (("cuobjdump", ["-sass"]), ("nvdisasm", ["-c"])):
        exe = nv / tool
        if exe.exists():
            r = subprocess.run([str(exe), *args, str(path)], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return f"// {tool} {' '.join(args)}\n" + r.stdout
    return "// no cuobjdump or nvdisasm beside Triton's ptxas\n"


def main() -> int:
    out = Path(os.environ.get("OUT") or sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rec = {"triton": triton.__version__, "python": sys.version, "cases": {}}
    for H, BLOCK, warps in ((6144, 8192, 4),):
        k = triton.compile(source(H, BLOCK), target=GPUTarget("cuda", 90, 32), options={"num_warps": warps})
        tag = f"H{H}_B{BLOCK}_w{warps}"
        ptx = k.asm["ptx"]
        code = ptx[:ptx.index("\t.file")] if "\t.file" in ptx else ptx
        (out / f"{tag}.ptx").write_text(ptx)
        (out / f"{tag}.ttgir").write_text(k.asm["ttgir"])
        (out / f"{tag}.sass").write_text(sass(k.asm["cubin"], out))
        ops = {}
        for op in ("ld.global.v4.b32", "ld.global.v2.b32", "ld.global.b32", "ld.global.b64", "fma.rn.f32", "mul.f32", "add.f32",
                   "div.full.f32", "rsqrt.approx.ftz.f32", "shfl.sync.bfly.b32", "cvt.rn.bf16x2.f32", "st.global.v4.b32",
                   "selp.f32", "selp.b32", "setp.eq.s64"):
            ops[op] = len(re.findall(re.escape(op) + r"(?=[\s;])", code))
        rec["cases"][tag] = {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "counts": ops,
                             "reqntid": re.findall(r"\.reqntid\s+([\d, ]+)", code), "shared": k.metadata.shared,
                             "num_warps": k.metadata.num_warps}
    (out / "k.cubin").unlink(missing_ok=True)
    (out / "ehnorm_ptx.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
