#!/usr/bin/env python
"""Verity-semantics reference for the FA2 attention chain (lane B-sem, R8) -- the P0 tap oracle.

Implements `AttentionHead_v3{T, D, BN}` -> `AttnBlock_v3{D, NVIS, FIRST, BN}` (verity_vllm/program/registry/b1.py:727-822) with
the registry primitives themselves (verity_vllm/program/registry/prims.py), so every emitted word is *the Definition's* word:
FTZ f32 ops, `x > y ? x : y` max, the measured MUFU.EX2 / MUFU.RCP tables, the measured Ampere m16n8k16 model,
RNE bf16.  Emits every online-state version the kernel holds in registers (mapping: p0-tap-spec.md).

Run:  cd integrations/vllm; PYTHONPATH=. python tests/acquire/fa2_attn_oracle.py --case p0b --out /tmp/p0b_ref.json
      PYTHONPATH=. python tests/acquire/fa2_attn_oracle.py --compare /tmp/p0b_ref.json /tmp/p0b_dump.json
Needs the MUFU tables (core's, `verity.ml.circuits.ops.mufu.TABLE_DIR`).
Deterministic: inputs come from numpy PCG64(seed); no torch, no GPU.  All words are hex in the JSON (f32: 8 digits,
bf16: 4 digits); ints `n_block`, `nvis`, `M`, `T`, `D`, `BN` are decimal.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
from verity_vllm.program.registry import prims as P  # PYTHONPATH=.

NEG_INF = 0xFF800000
BF16_KEYS = {"q", "k", "v", "pb", "out"}
PLAIN = {"n_block", "nvis", "M", "T", "D", "BN", "case", "causal", "softmax_scale", "is_first", "lse_approx", "softcap", "window"}
CASES = {  # 1 head; BN = fa2_kblock_n(D); every case has row-schedule == kernel tile-schedule (checked at run time)
    "p0a": dict(M=16, T=128, D=64, causal=True, seed=11, spread=1.0),   # one block: Is_first only, prefix mask
    "p0b": dict(M=16, T=144, D=64, causal=True, seed=12, spread=1.0),   # two blocks: rescale path, mask, K/V zero pad
    "p0c": dict(M=3, T=160, D=64, causal=False, seed=13, spread=1.0),   # decode shape (3 rows = 3 heads of a KV group)
    "p0d": dict(M=8, T=200, D=64, causal=True, seed=14, spread=6.0),    # large |scores|: MUFU / FTZ ranges, scale << 1
    # [R15 capacity] Gemma-2 shape: hd 256 (kBlockN 64), softcap 50, sliding window 4096 >= T (Is_local kernel, masks nothing)
    "g2a": dict(M=8, T=150, D=256, causal=True, seed=15, spread=1.0, softcap=50.0, window=4096),
    "g2b": dict(M=8, T=150, D=256, causal=True, seed=15, spread=30.0, softcap=50.0, window=4096),  # |s*scale| ~ 30-60: tanh table + saturation
}


def lane(c: int) -> int:                       # b1.py:409 -- quad thread that owns key column c
    return (c % 8) // 2


def scale_log2_bits(D: int) -> int:            # b1.py:121-124 == flash_api.cpp:164 (double product, narrowed to f32)
    return P.bits_of(np.float32(float(np.float32(D ** -0.5)) * 1.4426950408889634))


def fa2_kblock_n(D: int) -> int:               # b1.py:364-371 (the num_splits == 1 "align" kernel, kBlockN)
    return 128 if D <= 64 else 64


# [R15 capacity] softcap (Gemma-2: attn_logit_softcapping 50.0).  flash_api.cpp:156-159 with softcap > 0:
#   params.softcap = softmax_scale / softcap (f32 / f32), params.scale_softmax = softcap, params.scale_softmax_log2 = softcap * M_LOG2E
#   (double product narrowed to f32); the kernel then runs apply_softcap(acc_s, params.softcap): s = tanh.approx.f32(s * params.softcap)
#   (cutlass::fast_tanh on sm >= 75 = MUFU.TANH, the measured `MufuTanh` word table) BEFORE the mask / online softmax, which therefore
#   sees the capped scores with scale softcap (exp2((s - max) * softcap * log2e) == exp(softcap * (tanh(...) - tanh(...)))).
def softcap_params(D: int, cap: float, scale: float | None = None) -> tuple[int, int]:
    """-> (params.softcap bits, scale_softmax_log2 bits) exactly as flash_api.cpp derives them for softmax_scale (default D^-0.5) and cap."""
    sc = np.float32(D ** -0.5) if scale is None else np.float32(scale)
    return P.bits_of(np.float32(sc / np.float32(cap))), P.bits_of(np.float32(float(np.float32(cap)) * 1.4426950408889634))


def apply_softcap(s_bits: int, softcap_bits: int) -> int:
    """One kernel word of FA2 apply_softcap: MUFU.TANH(FTZ f32 mul(s, params.softcap))."""
    return P.MufuTanh.evaluate(P.F32MulFtz.evaluate(s_bits, softcap_bits))


def mma16(acc: int, a: list[int], b: list[int]) -> int:
    return P.AmpereBF16TcDot16.evaluate(acc, *a, *b)


def dot_bf16(q: list[int], k: list[int]) -> int:   # DotBf16_v1{K=D}: ascending k-steps from acc 0
    acc = 0
    for s in range(len(q) // 16):
        acc = mma16(acc, q[16 * s:16 * s + 16], k[16 * s:16 * s + 16])
    return acc


def attn_block(q, k, v, nvis, first, m_prev, l_prev, o_prev, sl2, D, BN, rec, softcap_bits=0):
    """AttnBlock_v3 body (b1.py:740-783) for one query row; k, v are BN rows (zero-padded); records every stage.
    softcap_bits != 0 (Gemma-2): every acc_s word (pad columns included: tanh(0 * c) == 0 by the identity rule) is softcapped before
    the mask, and sl2 is the caller's softcap * log2e (softcap_params)."""
    s_all = [dot_bf16(q, k[c]) for c in range(BN)]          # kernel acc_s before apply_mask (pad rows give s == 0)
    if softcap_bits:
        s_all = [apply_softcap(x, softcap_bits) for x in s_all]
    s = s_all[:nvis]                                         # A: the Definition sees visible columns only
    rec["s_pre_mask"], rec["s_masked"] = s_all, s + [NEG_INF] * (BN - nvis)
    bmax = s[0]
    for c in range(1, nvis):
        bmax = P.F32Max.evaluate(bmax, s[c])                 # B1
    m_new = bmax if first else P.F32Max.evaluate(m_prev, bmax)
    m_use = P.GuardNegInfZero.evaluate(m_new)
    rec["bmax"], rec["m_new"], rec["m_use"] = bmax, m_new, m_use
    if not first:                                            # B2
        scores_scale = P.MufuEx2Ftz.evaluate(P.F32MulFtz.evaluate(P.F32SubFtz.evaluate(m_prev, m_use), sl2))
        rec["scores_scale"] = scores_scale
    max_scaled = P.F32MulFtz.evaluate(m_use, sl2)
    rec["max_scaled"] = max_scaled
    p = [P.MufuEx2Ftz.evaluate(P.F32FmaSubFtz.evaluate(s[c], sl2, max_scaled)) for c in range(nvis)]   # B3
    rec["p"] = p + [0] * (BN - nvis)                         # kernel: exp2(-inf - max_scaled) == +0 on masked cols
    lanes = []                                               # B4: per-lane sequential sums; first add contracted
    for ln in range(4):
        cols = [c for c in range(nvis) if lane(c) == ln]
        acc = (p[cols[0]] if cols else 0) if first else P.F32FmaFtz.evaluate(scores_scale, l_prev[ln], p[cols[0]] if cols else 0)
        for c in cols[1:]:
            acc = P.F32AddFtz.evaluate(acc, p[c])
        lanes.append(acc)
    rec["l"] = lanes
    pb = [P.F2fpBf16.evaluate(x) for x in p] + [0] * (BN - nvis)   # B5
    rec["pb"] = pb
    o = [0] * D if first else [P.F32MulFtz.evaluate(o_prev[d], scores_scale) for d in range(D)]
    rec["o_rescaled"], o_steps = list(o), []
    for ks in range(BN // 16):                               # C: PV in ascending k-steps of 16 keys
        k0 = 16 * ks
        if k0 >= nvis:
            break                                            # all-masked k-step: P == 0, accumulator unchanged
        o = [mma16(o[d], pb[k0:k0 + 16], [v[k0 + j][d] for j in range(16)]) for d in range(D)]
        o_steps.append(list(o))
    rec["o_ksteps"], rec["o"] = o_steps, list(o)
    return m_new, lanes, o


def attention_head(q, K, V, T, D, sl2, BN, limit, nb, softcap_bits=0, left=0):
    """AttentionHead_v3 (b1.py:790-822) for one query row: keys 0..limit-1 visible, `nb` blocks visited nb-1 -> 0.
    softcap_bits: FA2 softcap (see softcap_params; sl2 must then be softcap * log2e).  left: the sliding-window lower column bound
    (FA2 Is_local, col_idx_limit_left = max(0, row + seqlen_k - seqlen_q - window_left)); only left == 0 is in this oracle's domain
    (every panel row of Gemma-2 / Phi-3 has seqlen <= window, so the local kernel masks nothing below: identical values to causal)."""
    if left:
        raise SystemExit(f"sliding window shorter than the context (left={left}): trailing all-masked blocks are outside this oracle's domain")
    if (limit + BN - 1) // BN != nb:
        raise SystemExit("row-level schedule != kernel tile schedule (a row with no visible key in the tile's last "
                         "block): outside AttentionHead_v3's domain -- pick M, T so every row sees >= 1 key there")
    m = l = o = None
    blocks = []
    for b in range(nb - 1, -1, -1):
        lo, nvis = b * BN, min(limit, (b + 1) * BN) - b * BN
        kb = [K[j] if j < T else [0] * D for j in range(lo, lo + BN)]   # Clear_OOB_MN: key rows >= T zero-filled
        vb = [V[j] if j < T else [0] * D for j in range(lo, lo + BN)]
        first = b == nb - 1
        rec = {"n_block": b, "nvis": nvis, "is_first": first}
        m, l, o = attn_block(q, kb, vb, nvis, first, 0 if first else m, [0] * 4 if first else l, [0] * D if first else o, sl2, D, BN, rec,
                             softcap_bits)
        blocks.append(rec)
    total = P.F32AddFtz.evaluate(P.F32AddFtz.evaluate(l[0], l[2]), P.F32AddFtz.evaluate(l[1], l[3]))
    inv = P.Fa2InvSum.evaluate(total)
    y = [P.F32MulFtz.evaluate(o[d], inv) for d in range(D)]
    out = [P.F2fpBf16.evaluate(x) for x in y]
    scale = np.float32(D ** -0.5) if not softcap_bits else np.float32(P.f32_of(sl2)) / np.float32(1.4426950408889634)
    lse = float(np.float32(P.f32_of(m)) * scale + np.float32(np.log(np.float32(P.f32_of(total)))))
    return {"blocks": blocks, "sum": total, "inv_sum": inv, "y": y, "out": out, "lse_approx": lse}  # lse: not Verity, ~only


def make_inputs(M, T, D, seed, spread):
    rng = np.random.default_rng(seed)
    bf16 = lambda a: [int(w) for w in (np.asarray(a, np.float32).view(np.uint32) >> 16)]  # truncation: fine for inputs
    return ([bf16(rng.normal(0, spread, D)) for _ in range(M)], [bf16(rng.normal(0, spread, D)) for _ in range(T)],
            [bf16(rng.normal(0, 1.0, D)) for _ in range(T)])


def run_case(name):
    c = CASES[name]
    M, T, D, causal = c["M"], c["T"], c["D"], c["causal"]
    cap, window = c.get("softcap", 0.0), c.get("window", -1)
    BN = fa2_kblock_n(D)
    softcap_bits, sl2 = softcap_params(D, cap) if cap else (0, scale_log2_bits(D))
    q, k, v = make_inputs(M, T, D, c["seed"], c["spread"])
    kBlockM = 64                                             # one m_block tile (M <= 64); kernel n_block_max, :539-543
    nb = (T + BN - 1) // BN
    if causal:
        nb = min(nb, (kBlockM + T - M + BN - 1) // BN)
    rows = [attention_head(q[r], k, v, T, D, sl2, BN, min(T, r + 1 + T - M) if causal else T, nb, softcap_bits,
                           left=max(0, r + T - M - window) if window >= 0 else 0) for r in range(M)]
    return {"case": name, "M": M, "T": T, "D": D, "BN": BN, "causal": causal, "softmax_scale": float(np.float32(D ** -0.5)),
            "scale_softmax_log2": sl2, "softcap": cap, "softcap_bits": softcap_bits, "window": window, "q": q, "k": k, "v": v, "rows": rows}


def hexify(x, bf=False):
    if isinstance(x, bool) or isinstance(x, (float, str)):
        return x
    if isinstance(x, int):
        return f"{x:04x}" if bf else f"{x:08x}"
    if isinstance(x, list):
        return [hexify(e, bf) for e in x]
    return {k: (v if k in PLAIN else hexify(v, k in BF16_KEYS)) for k, v in x.items()}


def unhex(x):
    if isinstance(x, str):
        return int(x, 16)
    if isinstance(x, list):
        return [unhex(e) for e in x]
    if isinstance(x, dict):
        return {k: (v if k in PLAIN else unhex(v)) for k, v in x.items()}
    return x


def compare(ref, dump, zero_equal=("bmax", "m_new", "m_use", "max_scaled")):
    """Bit-compare a kernel dump with the reference; +0/-0 count as equal on the max chain (correspondence finding 4)."""
    bad = 0
    def check(path, a, b, key):
        nonlocal bad
        a, b = (a if isinstance(a, list) else [a]), (b if isinstance(b, list) else [b])
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y and not (key in zero_equal and {x, y} <= {0, 0x80000000}):
                bad += 1
                if bad <= 40:
                    print(f"MISMATCH {path}/{key}[{i}] ref {x:08x} dump {y:08x}")
    for r, (rr, dr) in enumerate(zip(ref["rows"], dump["rows"])):
        for br, bd in zip(rr["blocks"], dr["blocks"]):
            for key in ("s_masked", "bmax", "m_new", "scores_scale", "max_scaled", "p", "pb", "l", "o_rescaled", "o_ksteps", "o"):
                if key in br and key in bd:
                    check(f"row{r}/n_block{br['n_block']}", br[key], bd[key], key)
        for key in ("sum", "inv_sum", "y", "out"):
            if key in dr:
                check(f"row{r}", rr[key], dr[key], key)
    print("BIT-EXACT" if bad == 0 else f"{bad} mismatching words")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="p0b", choices=sorted(CASES))
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("REF_JSON", "DUMP_JSON"))
    a = ap.parse_args()
    if a.compare:
        ref, dump = (unhex(json.load(open(f))) for f in a.compare)
        sys.exit(1 if compare(ref, dump) else 0)
    res = run_case(a.case)
    if a.out:
        open(a.out, "w").write(json.dumps(hexify(res)))
    r0 = res["rows"][-1]
    print(f"{a.case}: M={res['M']} T={res['T']} BN={res['BN']} sl2={res['scale_softmax_log2']:08x} blocks/row={len(r0['blocks'])}; "
          f"last row: m={r0['blocks'][-1]['m_new']:08x} sum={r0['sum']:08x} inv={r0['inv_sum']:08x} out[0..3]="
          + " ".join(f"{w:04x}" for w in r0["out"][:4]) + (f"  -> {a.out}" if a.out else ""))
