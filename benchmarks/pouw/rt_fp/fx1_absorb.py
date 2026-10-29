"""An ISA check: on sm_120, does fx1's multiply absorb an FP route's extraction at no op?

    python benchmarks/pouw/rt_fp/fx1_absorb.py OUT        # PTXAS, CUOBJDUMP: the tools (default /usr/local/cuda-13.0/bin)

One PTX kernel per tail, H1's shape: each loop iteration fires 8 m16n8k32 MMAs (tile positions J < 8), and each of
their 4 words a lane goes through fx1's entry t = w·m, m = key_mul(1, (4J + e)·32), then duo into chain 4(J % 2) + e.

- `int8`: IMMA s8 from 0, t = w·m (the int8 record's tail).
- `fp-split`: QMMA e4m3 from the binade 1.5·2^23, t = (w − 0x4B400000)·m written as a subtract and a multiply.
- `fp-absorbed`: QMMA e4m3 from the binade, t = w·m − 0x4B400000·m written as one mad.lo with a constant addend.
- `fp-absorbed-reg`: as `fp-absorbed`, the 32 addends loaded into registers before the loop, so ptxas can't
  rematerialize them (left to itself it emits a MOV of the addend a word inside the loop).

It assembles each for sm_120a, disassembles it, and counts the loop body's SASS opcodes a word (the body's count over
its MMAs' words), with ptxas's registers and spills. Each position has its own B fragment, so no MMA is merged.
OUT/fx1_absorb.json holds the counts, OUT/<tail>.sass and OUT/<tail>.ptx the code.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

BIN = Path(os.environ.get("CUDA_BIN", "/usr/local/cuda-13.0/bin"))
PTXAS = os.environ.get("PTXAS", str(BIN / "ptxas"))
CUOBJDUMP = os.environ.get("CUOBJDUMP", str(BIN / "cuobjdump"))
M32 = 0xFFFFFFFF
NPOS, POSITIONS, BINADE = 16, 8, 0x4B400000


def mix32(x: int) -> int:
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def key_mul(lane: int, t: int) -> int:
    return mix32((0xA0000000 + 2 * (lane * NPOS + t)) & M32) | 1


def key_xor(lane: int, t: int) -> int:
    return mix32((0xB0000000 + 2 * (lane * NPOS + t) + 1) & M32)


def rot_of(t: int) -> int:
    return (t * 7 + 5) % 31 + 1


def ptx(tail: str) -> str:
    fp = tail != "int8"
    acc = ".f32.e4m3.e4m3.f32" if fp else ".s32.s8.s8.s32"
    init = "%fc" if fp else "%zero"
    body = []
    for j in range(POSITIONS):
        w = [f"%w{4 * j + e}" for e in range(4)]
        body.append(f"  mma.sync.aligned.m16n8k32.row.col{acc} {{{', '.join(w)}}}, {{%a0, %a1, %a2, %a3}}, "
                    f"{{%b{2 * j}, %b{2 * j + 1}}}, {{{init}, {init}, {init}, {init}}};")
        for e in range(4):
            m, h = key_mul(1, (4 * j + e) * 32), 4 * (j % 2) + e
            if tail == "fp-split":
                body.append(f"  sub.u32 %t, {w[e]}, {BINADE};\n  mul.lo.u32 %t, %t, {m};")
            elif tail == "fp-absorbed":
                body.append(f"  mad.lo.u32 %t, {w[e]}, {m}, {(-BINADE * m) & M32};")
            elif tail == "fp-absorbed-reg":
                body.append(f"  mad.lo.u32 %t, {w[e]}, {m}, %k{4 * j + e};")
            else:
                body.append(f"  mul.lo.u32 %t, {w[e]}, {m};")
            body.append(f"  xor.b32 %u, %y{h}, %t;\n  xor.b32 %u, %u, {key_xor(0, j)};\n"
                        f"  mad.lo.u32 %y{h}, %u, {key_mul(0, j)}, %z{h};\n"
                        f"  shf.l.wrap.b32 %s, %u, %u, {rot_of(j)};\n  add.u32 %z{h}, %s, %y{h};")
    chains = "\n".join(f"  mov.b32 %y{h}, {h};\n  mov.b32 %z{h}, {h + 8};" for h in range(8))
    frags = "\n".join(f"  ld.global.v4.u32 {{%b{4 * q}, %b{4 * q + 1}, %b{4 * q + 2}, %b{4 * q + 3}}}, [%rp+{16 + 16 * q}];"
                      for q in range(POSITIONS // 2))
    addends = "\n".join(f"  ld.global.v4.u32 {{%k{4 * q}, %k{4 * q + 1}, %k{4 * q + 2}, %k{4 * q + 3}}}, "
                        f"[%rp+{16 + 8 * POSITIONS + 16 * q}];" for q in range(POSITIONS)) if tail == "fp-absorbed-reg" else ""
    store = "\n".join(f"  st.global.v2.u32 [%ro+{8 * h}], {{%y{h}, %z{h}}};" for h in range(8))
    return f""".version 8.7
.target sm_120a
.address_size 64
.visible .entry k(.param .u64 pa, .param .u64 pout, .param .u32 pn)
{{
  .reg .b32 %a<4>, %b<{2 * POSITIONS}>, %k<{4 * POSITIONS}>, %w<32>, %y<8>, %z<8>, %t, %u, %s, %i, %n, %zero;
  .reg .f32 %fc;
  .reg .b64 %rp, %ro;
  .reg .pred %p;
  ld.param.u64 %rp, [pa];
  ld.param.u64 %ro, [pout];
  ld.param.u32 %n, [pn];
  ld.global.v4.u32 {{%a0, %a1, %a2, %a3}}, [%rp];
{frags}
{addends}
  mov.b32 %zero, 0;
  mov.b32 %fc, 0f4B400000;
{chains}
  mov.u32 %i, 0;
LOOP:
  xor.b32 %a0, %a0, %y0;
{chr(10).join(body)}
  add.u32 %i, %i, 1;
  setp.lt.u32 %p, %i, %n;
  @%p bra LOOP;
{store}
  ret;
}}
"""


def loop_ops(sass: str) -> tuple[Counter, int]:
    """The opcodes of the kernel's loop body (from the back edge's target to the back edge), and its MMA count."""
    ins = []
    for line in sass.splitlines():
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);", line)
        if m:
            ins.append((int(m.group(1), 16), m.group(3), m.group(4)))
    back = [(i, a, op, args) for i, (a, op, args) in enumerate(ins) if op.startswith("BRA")
            and (t := re.search(r"0x([0-9a-f]+)", args)) and int(t.group(1), 16) < a]
    if not back:
        raise SystemExit("no loop back edge in the SASS")
    i, _a, _, args = back[-1]
    target = int(re.search(r"0x([0-9a-f]+)", args).group(1), 16)
    body = [op.split(".")[0] for addr, op, _ in ins[: i + 1] if addr >= target]
    return Counter(body), sum(1 for op in body if op in ("IMMA", "QMMA", "HMMA", "OMMA"))


def main(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    ver = subprocess.run([PTXAS, "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    rows = {}
    with tempfile.TemporaryDirectory() as d:
        for tail in ("int8", "fp-split", "fp-absorbed", "fp-absorbed-reg"):
            src, cubin = Path(d) / f"{tail}.ptx", Path(d) / f"{tail}.cubin"
            src.write_text(ptx(tail))
            r = subprocess.run([PTXAS, "-arch=sm_120a", "-O3", "-v", str(src), "-o", str(cubin)], capture_output=True,
                               text=True, check=True)
            regs = re.search(r"Used (\d+) registers", r.stderr)
            spill = re.search(r"(\d+) bytes spill stores, (\d+) bytes spill loads", r.stderr)
            sass = subprocess.run([CUOBJDUMP, "-sass", str(cubin)], capture_output=True, text=True, check=True).stdout
            (out / f"{tail}.sass").write_text(sass)
            (out / f"{tail}.ptx").write_text(src.read_text())
            ops, mmas = loop_ops(sass)
            words = 4 * mmas
            rows[tail] = {"registers": int(regs.group(1)) if regs else None,
                              "spill_bytes": [int(spill.group(1)), int(spill.group(2))] if spill else None,
                              "mmas": mmas, "words": words, "ops": dict(ops),
                              "lane_ops_a_word": (sum(ops.values()) - mmas) / words}
    res = {"ptxas": ver, "positions": POSITIONS, "tails": rows}
    (out / "fx1_absorb.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
