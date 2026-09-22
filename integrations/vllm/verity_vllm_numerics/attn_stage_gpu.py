"""Stage-by-stage check of the FA2 CPU model against the *hardware instructions* (GPU host only).

For selected fixture rows the kernel's per-row arithmetic is re-executed with the instructions
the kernel uses -- chained ``mma.sync.m16n8k16`` (W4b's single-instruction probe, one instruction per
tile, D fed back as C) for S = Q K^T and O += P V, and nvcc ``--use_fast_math`` ``exp2f`` / ``1.f/x``
(FFMA+MUFU.EX2, MUFU.RCP) for the softmax scalars -- and every intermediate is compared bit-for-bit
with the CPU model's:

  stage A  direct QK MMA        S words per (row, key)
  stage B  online softmax step  P words per (row, key), the block rescale factors, the row-sum lanes
  stage C  PV MMA               O accumulator words per (row, d) after the last block
  stage D  epilogue             inv = rcp(sum), the BF16 output word (also against the production output)

    PYTHONPATH=. python -m verity_vllm_numerics.attn_stage_gpu FIXTURE_DIR OUT.json [--rows N] [--seqs 5,7,8,784]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from . import fa2_model as M
from . import probe

HERE = Path(__file__).resolve().parent
SM_SRC = HERE / "cuda" / "fa2_softmax_probe.cu"
SM_LIB = HERE / "cuda" / "libfa2_softmax_probe.so"
_sm = None


def sm_lib() -> ctypes.CDLL:
    global _sm
    if _sm is None:
        nvcc = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
        if not SM_LIB.exists() or SM_LIB.stat().st_mtime < SM_SRC.stat().st_mtime:
            subprocess.run([nvcc, "-O2", "--use_fast_math", "-arch=sm_89", "-Xcompiler", "-fPIC", "-shared", "-o", str(SM_LIB), str(SM_SRC)], check=True)
        _sm = ctypes.CDLL(str(SM_LIB))
        _sm.fa2_p.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        _sm.fa2_scale.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        _sm.fa2_rcp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        _sm.fa2_maxscaled.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        _sm.fa2_lanesum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_void_p]
        for f in (_sm.fa2_p, _sm.fa2_scale, _sm.fa2_rcp, _sm.fa2_maxscaled, _sm.fa2_lanesum):
            f.restype = ctypes.c_int
    return _sm


def gpu_p(s: np.ndarray, scale: float, max_scaled: float) -> np.ndarray:
    s = np.ascontiguousarray(s, np.float32)
    p = np.zeros_like(s)
    assert sm_lib().fa2_p(s.ctypes.data, ctypes.c_float(scale), ctypes.c_float(max_scaled), p.ctypes.data, s.size) == 0
    return p


def gpu_scale(mprev: np.ndarray, mcur: np.ndarray, scale: float) -> np.ndarray:
    a = np.ascontiguousarray(mprev, np.float32)
    b = np.ascontiguousarray(mcur, np.float32)
    c = np.zeros_like(a)
    assert sm_lib().fa2_scale(a.ctypes.data, b.ctypes.data, ctypes.c_float(scale), c.ctypes.data, a.size) == 0
    return c


def gpu_rcp(x: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(x, np.float32)
    b = np.zeros_like(a)
    assert sm_lib().fa2_rcp(a.ctypes.data, b.ctypes.data, a.size) == 0
    return b


def gpu_maxscaled(m: np.ndarray, scale: float) -> np.ndarray:
    a = np.ascontiguousarray(m, np.float32)
    b = np.zeros_like(a)
    assert sm_lib().fa2_maxscaled(a.ctypes.data, ctypes.c_float(scale), b.ctypes.data, a.size) == 0
    return b


def gpu_lanesum(p: np.ndarray, l_in: np.ndarray, sc: float, rescale: int) -> np.ndarray:
    a = np.ascontiguousarray(p, np.float32)
    b = np.ascontiguousarray(l_in, np.float32)
    out = np.zeros(4, np.float32)
    assert a.size == 64 and b.size == 4
    assert sm_lib().fa2_lanesum(a.ctypes.data, b.ctypes.data, ctypes.c_float(sc), int(rescale), out.ctypes.data) == 0
    return out


def ftz(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    u = x.view(np.uint32)
    sub = (u & 0x7F800000) == 0
    return np.where(sub, (u & 0x80000000).view(np.float32), x).astype(np.float32)


def f32(u):
    return np.asarray(u, np.uint32).view(np.float32)


def u32(f):
    return np.asarray(f, np.float32).view(np.uint32)


def bf16_rne(words32: np.ndarray) -> np.ndarray:
    u = words32.astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def gpu_qk(q: np.ndarray, K: np.ndarray, ncols: int) -> np.ndarray:
    """S[col] for col < ncols with chained mma.sync (8 k-steps of 16 over D=128; 8 keys per tile)."""
    ntiles = (ncols + 7) // 8
    A = np.zeros((ntiles, 16, 16), np.uint16)
    Bt = np.zeros((ntiles, 8, 16), np.uint16)
    C = np.zeros((ntiles, 16, 8), np.uint32)
    for i in range(8):
        A[:] = 0
        A[:, 0, :] = q[16 * i : 16 * i + 16]
        Bt[:] = 0
        for t in range(ntiles):
            lo, hi = 8 * t, min(8 * t + 8, ncols)
            Bt[t, : hi - lo, :] = K[lo:hi, 16 * i : 16 * i + 16]
        D = probe.mma_tiles("bf16", A, Bt, C)
        C = D
    return C[:, 0, :].reshape(-1)[:ncols]


def gpu_pv_block(pb: np.ndarray, V: np.ndarray, o_bits: np.ndarray) -> np.ndarray:
    """One 64-key block: O[d] += P_bf16 . V[:, d] as 4 chained mma.sync k-steps (16 d per... 8 d per tile)."""
    D = o_bits.shape[0]
    ntiles = D // 8
    C = np.zeros((ntiles, 16, 8), np.uint32)
    C[:, 0, :] = o_bits.reshape(ntiles, 8)
    for i in range(4):
        A = np.zeros((ntiles, 16, 16), np.uint16)
        A[:, 0, :] = pb[16 * i : 16 * i + 16]
        Bt = np.zeros((ntiles, 8, 16), np.uint16)
        vs = V[16 * i : 16 * i + 16, :]  # (16 keys, D)
        Bt[:, :, :] = vs.T.reshape(ntiles, 8, 16)
        C = probe.mma_tiles("bf16", A, Bt, C)
    return C[:, 0, :].reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture")
    ap.add_argument("out")
    ap.add_argument("--rows", type=int, default=24, help="number of (call,row,head) samples")
    ap.add_argument("--seqs", default=None, help="comma-separated call seqs to include (all their saved rows x heads sampled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", default=None, help="attn_check report: include its mismatching rows")
    args = ap.parse_args()
    fx = Path(args.fixture)
    idx = json.loads((fx / "index.json").read_text())
    consts = idx["consts"]
    H, KVH, D = consts["num_heads"], consts["num_kv_heads"], consts["head_size"]
    scale = float(consts["softmax_scale"])
    sl2 = float(M.scale_log2(scale))
    tables = M.MufuTables.load(fx / "mufu")
    calls = {c["seq"]: c for c in idx["calls"]}
    rng = np.random.default_rng(args.seed)
    samples = []
    if args.check:
        rep = json.loads(Path(args.check).read_text())
        for b in rep["first_mismatches"]:
            samples.append((b["seq"], b["row"], b["head"], "mismatch"))
    pool = [calls[int(s)] for s in args.seqs.split(",")] if args.seqs else list(calls.values())
    for _ in range(args.rows):
        c = pool[rng.integers(len(pool))]
        samples.append((c["seq"], int(rng.choice(c["rows"])), int(rng.integers(H)), "random"))

    kvc = {}

    def kv(f):
        if f not in kvc:
            kvc[f] = np.load(fx / f)
        return kvc[f]

    stages = {k: {"compared": 0, "exact": 0} for k in ("A_qk_mma_S", "B_softmax_P", "B_rescale_factor", "B_row_sum", "C_pv_mma_O", "D_inv", "D_out_word", "D_out_vs_production")}
    rows_out = []
    t0 = time.perf_counter()
    for seq, row, h, why in samples:
        c = calls[seq]
        q = np.load(fx / c["q_file"])
        gpu_out = np.load(fx / c["out_file"])
        K = kv(c["k_file"])
        V = kv(c["v_file"])
        rows = np.array(c["rows"])
        ri = int(np.where(rows == row)[0][0])
        kvh = h // (H // KVH)
        Lk = c["Lk"]
        col_limit, nb = M.kernel_geometry(c["Lq"], Lk, [row])
        col_limit, nb = int(col_limit[0]), int(nb[0])
        res = M.attention_rows(q[ri, h][None, :], K[:Lk], V[:Lk], np.array([kvh]), np.array([col_limit]), np.array([nb]), scale, tables, debug=True)
        dbg = res.dbg
        Kh = K[:Lk, kvh, :]
        Vh = V[:Lk, kvh, :]
        # --- stage A: S via hardware mma chain
        S_gpu = np.full(nb * 64, np.float32(-np.inf), np.float32)
        S_gpu[:col_limit] = f32(gpu_qk(q[ri, h], Kh, col_limit))
        S_mod = f32(dbg["s"][0])
        cmpA = u32(S_gpu[:col_limit]) == u32(S_mod[:col_limit])
        stages["A_qk_mma_S"]["compared"] += col_limit
        stages["A_qk_mma_S"]["exact"] += int(cmpA.sum())
        # --- stage B: P, rescale factors, lane sums, using the GPU S as input
        m = np.float32(-np.inf)
        first = True
        P_gpu = np.zeros(nb * 64, np.float32)
        l = np.zeros(4, np.float32)
        o_bits = np.zeros(D, np.uint32)
        sc_cmp = []
        for b in range(nb - 1, -1, -1):
            s = S_gpu[64 * b : 64 * b + 64]
            m_new = np.float32(max(float(s.max()), float(m))) if not first else np.float32(s.max())
            if not first:
                cur = np.float32(0) if m_new == -np.inf else m_new
                sc = gpu_scale(np.array([m]), np.array([cur]), sl2)[0]
                sc_mod = M.mufu_ex2(ftz(ftz(np.float32(m) - cur) * np.float32(sl2)), tables)[0]
                sc_cmp.append(u32(sc) == u32(sc_mod))
                o_bits = u32(ftz(f32(o_bits) * sc))
            else:
                sc = np.float32(1.0)
            ms = gpu_maxscaled(np.array([m_new]), sl2)[0]
            p = gpu_p(s, sl2, ms)
            P_gpu[64 * b : 64 * b + 64] = p
            # per-lane sums on the GPU, source-literal code (`l *= sc; l += p...`) so nvcc contracts exactly as in the kernel
            l = gpu_lanesum(p, l, sc, 0 if first else 1)
            pb = bf16_rne(u32(p))
            vblk = np.zeros((64, D), np.uint16)
            hi = min(col_limit, 64 * b + 64)
            if hi > 64 * b:
                vblk[: hi - 64 * b] = Vh[64 * b : hi]
            o_bits = gpu_pv_block(pb, vblk, o_bits)
            m = m_new
            first = False
        P_mod = f32(dbg["p"][0])
        cmpB = u32(P_gpu) == u32(P_mod)
        stages["B_softmax_P"]["compared"] += P_gpu.size
        stages["B_softmax_P"]["exact"] += int(cmpB.sum())
        stages["B_rescale_factor"]["compared"] += len(sc_cmp)
        stages["B_rescale_factor"]["exact"] += int(sum(bool(x) for x in sc_cmp))
        cmpL = u32(l) == dbg["l"][0]
        stages["B_row_sum"]["compared"] += 4
        stages["B_row_sum"]["exact"] += int(cmpL.sum())
        cmpC = o_bits == dbg["o_pre"][0]
        stages["C_pv_mma_O"]["compared"] += D
        stages["C_pv_mma_O"]["exact"] += int(cmpC.sum())
        # --- stage D
        ssum = ftz(np.float32(ftz(np.float32(l[0] + l[2])) + ftz(np.float32(l[1] + l[3]))))
        inv_gpu = gpu_rcp(np.array([ssum]))[0]
        inv_mod = np.float32(1) if (ssum == 0 or ssum != ssum) else M.mufu_rcp(np.array([ssum]), tables)[0]
        stages["D_inv"]["compared"] += 1
        stages["D_inv"]["exact"] += int(u32(inv_gpu) == u32(inv_mod))
        out_hw = bf16_rne(u32(ftz(f32(o_bits) * inv_gpu)))
        cmpD = out_hw == res.out[0]
        stages["D_out_word"]["compared"] += D
        stages["D_out_word"]["exact"] += int(cmpD.sum())
        cmpP = out_hw == gpu_out[ri, h]
        stages["D_out_vs_production"]["compared"] += D
        stages["D_out_vs_production"]["exact"] += int(cmpP.sum())
        rec = {"seq": seq, "phase": c["phase"], "layer": c["layer"], "Lk": Lk, "row": row, "head": h, "why": why, "n_blocks": nb,
               "S_mismatch": int((~cmpA).sum()), "P_mismatch": int((~cmpB).sum()), "rescale_mismatch": int(len(sc_cmp) - sum(bool(x) for x in sc_cmp)),
               "lane_sum_mismatch": int((~cmpL).sum()), "O_mismatch": int((~cmpC).sum()), "inv_match": bool(u32(inv_gpu) == u32(inv_mod)),
               "out_word_mismatch_vs_model": int((~cmpD).sum()), "hw_chain_vs_production_mismatch": int((~cmpP).sum()),
               "model_vs_production_mismatch": int((res.out[0] != gpu_out[ri, h]).sum())}
        if rec["S_mismatch"]:
            bad = np.argwhere(~cmpA)[:4].reshape(-1)
            rec["S_examples"] = [{"col": int(cc), "gpu": f"{int(u32(S_gpu[cc])):08x}", "model": f"{int(u32(S_mod[cc])):08x}"} for cc in bad]
        if rec["O_mismatch"]:
            bad = np.argwhere(~cmpC)[:4].reshape(-1)
            rec["O_examples"] = [{"d": int(dd), "gpu": f"{int(o_bits[dd]):08x}", "model": f"{int(dbg['o_pre'][0][dd]):08x}"} for dd in bad]
        if rec["P_mismatch"]:
            bad = np.argwhere(~cmpB)[:4].reshape(-1)
            rec["P_examples"] = [{"col": int(cc), "gpu": f"{int(u32(P_gpu[cc])):08x}", "model": f"{int(u32(P_mod[cc])):08x}", "S_gpu": f"{int(u32(S_gpu[cc])):08x}"} for cc in bad]
        rows_out.append(rec)
        print(json.dumps(rec))
    report = {"fixture": str(fx), "device": probe.device_info(), "scale_log2": f"{int(u32(np.float32(sl2))):08x}", "stages": stages, "rows": rows_out,
              "seconds": time.perf_counter() - t0}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(stages, indent=1))


if __name__ == "__main__":
    main()
