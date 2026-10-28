"""The exhaustive sweeps: units.cu's helpers on every operand word, against the twins' instructions (words.py) and circuits'
MUFU models.

Each op is one helper.  The GPU's units_sweep and the CPU's `digest` both reduce the outputs of operand chunk c, the
operands i in [c 2^24, (c + 1) 2^24) below n, to sum_i splitmix64((i << 32) | out(i)) mod 2^64, so the chunks agree iff
(up to a 2^-64 collision) every output word does, and a chunk that disagrees says where to look: units_sweep_words (sm_120a)
gives its words, and `first_difference` the first one that differs.

  op  name        n      out(i)                                         what reads it
  0   ex2         2^32   ex2.approx.ftz.f32(i), against mufu.MufuEx2Ftz   softmax (MufuEx2Ftz is measured on sm_89; this
                                                                          holds it on the H200 and the RTX PRO 6000)
  1   s8          2^32   cvt.rni.sat.s8.f32(i)                            quant
  2   sqrt        2^32   sqrt.rn.f32(i)                                   rmsnorm
  3   rcp         2^32   div.rn.f32(1, i)                                 rmsnorm's rs, softmax's r
  4   div_n       2^32   div.rn.f32(i, 16384)                             rmsnorm's mean at the frozen width
  5   quant_div   2^31   qdiv(i & 0xFFFF, scale(i >> 16)), against         quant's codes, every bf16 word by every amax
                         div.rn.f32(f32(i & 0xFFFF), scale(i >> 16))     (qdiv: RN(1 / scale) and one FMA correction)
  6   bf16        2^32   cvt.rn.bf16.f32(i)                               rmsnorm, softmax
  7   scale       2^16   quant's scale(i)                                 quant
  8   add         2^32   bf16(add.rn.f32(widen(i >> 16), widen(i)))       add: every pair of bf16 words
  9   rsq         2^32   rsqrt.approx.ftz.f32(i) (bare MUFU.RSQ), against  no unit: circuits' RSQ table on sm_120 (measured
                         mufu.mufu_rsq_words over mufu.table("rsq")       on sm_89)
  10  rsqrt       2^32   rsqrt.approx.f32(i), against mufu.RsqrtApprox    no unit: rsqrtf's MUFU.RSQ with ptxas's subnormal
                                                                          scaling, as the CUDA RMSNorm Definitions read it
The constants (1, 16384, 127) come in as the kernel's arguments, as they do into the units.  Ops 9 and 10 are in the
sm_120a cubin only (`ARCH_OPS`).
"""
from __future__ import annotations

import numpy as np

import words as W

CHUNK = 1 << 24
SUB = 1 << 22
OPS = {0: ("ex2", 1 << 32, 0), 1: ("s8", 1 << 32, 0), 2: ("sqrt", 1 << 32, 0), 3: ("rcp", 1 << 32, W.ONE),
       4: ("div_n", 1 << 32, W.f32word(16384)), 5: ("quant_div", 1 << 31, W.F127), 6: ("bf16", 1 << 32, 0),
       7: ("scale", 1 << 16, W.F127), 8: ("add", 1 << 32, 0), 9: ("rsq", 1 << 32, 0), 10: ("rsqrt", 1 << 32, 0)}
#: the ops each cubin's units_sweep runs
ARCH_OPS = {"sm_90a": tuple(range(9)), "sm_120a": tuple(OPS)}


def chunks(op: int) -> int:
    return -(-OPS[op][1] // CHUNK)


def splitmix64(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def model(op: int, i: np.ndarray) -> np.ndarray:
    """The twins' (or circuits' model's) out(i) on operand words i (uint32)."""
    import quant
    i = np.asarray(i, np.uint32)
    arg = OPS[op][2]
    if op == 0:
        return W.ex2(i)
    if op == 1:
        return W.s8(i).astype(np.uint32)
    if op == 2:
        return W.fsqrt(i)
    if op == 3:
        return W.fdiv(np.uint32(arg), i)
    if op == 4:
        return W.fdiv(i, np.uint32(arg))
    if op == 5:
        return W.fdiv(W.widen(i & np.uint32(0xFFFF)), quant.scale((i >> np.uint32(16)).astype(np.uint16)))
    if op == 6:
        return W.bf16(i).astype(np.uint32)
    if op == 7:
        return quant.scale(i.astype(np.uint16))
    if op == 8:
        return W.bf16(W.fadd(W.widen(i >> np.uint32(16)), W.widen(i & np.uint32(0xFFFF)))).astype(np.uint32)
    if op == 9:
        from verity.ml.circuits.ops import mufu
        return mufu.mufu_rsq_words(i, mufu.table("rsq"))
    if op == 10:
        from verity.ml.circuits.evaluation import evaluate_batch
        from verity.ml.circuits.ops import mufu
        return np.asarray(evaluate_batch(mufu.RsqrtApprox, i), np.uint32)
    raise ValueError(f"no sweep op {op}")


def digest(op: int, c: int) -> int:
    """Chunk c's sum on the CPU, from `model`."""
    n = OPS[op][1]
    total = np.uint64(0)
    for lo in range(c * CHUNK, min(n, (c + 1) * CHUNK), SUB):
        i = np.arange(lo, min(n, lo + SUB), dtype=np.uint64)
        out = model(op, i.astype(np.uint32)).astype(np.uint64)
        with np.errstate(over="ignore"):
            total = total + np.sum(splitmix64((i << np.uint64(32)) | out), dtype=np.uint64)
    return int(total)


def first_difference(op: int, c: int, got: np.ndarray) -> dict:
    """Where chunk c's words ``got`` (from `gpu_words`) differ from `model`: how many, and the first (operand, GPU, model)."""
    lo = c * CHUNK
    want = model(op, np.arange(lo, lo + len(got), dtype=np.uint64).astype(np.uint32))
    bad = np.flatnonzero(got != want)
    first = None if not len(bad) else {"i": f"{lo + int(bad[0]):08x}", "gpu": f"{int(got[bad[0]]):08x}",
                                       "model": f"{int(want[bad[0]]):08x}"}
    return {"chunk": c, "words": len(got), "differing": len(bad), "first": first}


def gpu(op: int) -> list[int]:
    """Every chunk's sum from units_sweep on the current device."""
    import torch

    import lib as L
    nc = chunks(op)
    out = torch.zeros(nc, dtype=torch.int64, device="cuda")
    L.lib().launch("units_sweep", nc, 1024, [L.i32(op), L.u64(OPS[op][1]), L.u(OPS[op][2]), L.u(W.ONE), L.ptr(out)])
    torch.cuda.synchronize()
    return [int(v) & 0xFFFFFFFFFFFFFFFF for v in out.cpu().tolist()]


def gpu_words(op: int, c: int) -> np.ndarray:
    """Chunk c's output words from units_sweep_words (the sm_120a cubin's) on the current device."""
    import torch

    import lib as L
    count = min(CHUNK, OPS[op][1] - c * CHUNK)
    out = torch.zeros(count, dtype=torch.int32, device="cuda")
    L.lib().launch("units_sweep_words", -(-count // 256), 256,
                   [L.i32(op), L.u64(c * CHUNK), L.u(count), L.u(OPS[op][2]), L.u(W.ONE), L.ptr(out)])
    torch.cuda.synchronize()
    return out.cpu().numpy().view(np.uint32)
