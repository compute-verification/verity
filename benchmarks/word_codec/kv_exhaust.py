"""The KV cache's read and write sides at an HBM-exhaustion shape (HBM holds the encoded weights, the KV cache and
filler, and every step reads all of it), on one GPU: `kv.cu`'s libraries (`--lib NAME=PATH`, built with
KV_QHEADS the model's query heads) gated, then timed. Writes `kv-exhaust.json` and `kv.log` under `--out`, and, with
`RESEARCH_RUN_DIR` set, its `result.json`.

Nothing is timed before its gate passed:
- `gate`: kv_bench's codec gate at every k of `--ks`;
- `attn_gate`: at each batch x context of `--gate-cells`, with the model's layers, every `kv_attn` form at every split
  count of SPLITS, at layers 0 and L - 1: each encoded form's output equals its plain form's bit for bit (the CUDA-core
  forms 1 and 2 form 0's, the tensor-core forms 4 and 5 form 3's, 7, at the standalone decode's occupancy, its own
  plain form 6's, 9, persistent, its own plain form 8's, and 12, mma-ws's work items handed out by a counter, form
  3's), and each plain form is within RTOL of float64 attention
  over the same bf16 values.

Then, at each batch x context of `--cells`, one store holding the cell's whole KV cache (batch x context x layers
token-layers in kv.py's slot layout, so a layer's call reads 4 KB a token at the slots' stride) of random canonical
words:
- `read`: R over the store (kv_bench's read kernel), and the store's read-and-decode at each k, the floor where the
  decode overlaps the read and nothing else runs (gated first: its checksum at every timed grid, where each thread
  steps its tweak, equals the checksum at one word a thread);
- `attn`: a pass over every layer, each form at each split count, queued behind a spin, its best split count's time.
  Plain forms read the words as bf16 (their arithmetic's time doesn't depend on the values); encoded forms decode them.
  The read side's overhead a KV byte, in reads of the stored bytes at R, is (encoded - read) / read, and against the
  same family's plain form (encoded - plain) / read; each is fitted over k as c k - h. Against the floor: c(k) is the
  floor's time in reads over k (the decode a byte a round, where the decode is compute-bound), and each encoded form's
  h = c k - its overhead, the share of the decode hidden under the read. The floor is also timed at OCC_CTAS CTAs an
  SM (fewer decoder warps), against the encoded forms' occupancy.

`write`: the step's new tokens' encode (batch x layers x a token-layer's words) at each k: alone (queued behind a spin),
at the compute-only rate, and beside the step's weight GEMM (cuBLASLt FP8 at the model's linears, every layer's own
cold weights) on a second stream of lower priority, at several grids and at each of ENC_SMEM_CARVEOUTS (the encode's
preferred shared memory carveout: an SM the encode reaches first is set to the encode's carveout and takes the GEMM's
CTAs only if it fits their shared memory, so at the driver's choice whether the two overlap depends on which reaches
an SM first), against the GEMM alone; then with CARVEOUTS SMs taken from the GEMM (torch's SM carveout, cuBLASLt's SM
count target) and the encode at CARVE_GRIDS CTAs a carved-out SM and the most shared memory: the GEMM alone on the
rest, and both.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import kv as K  # noqa: E402
import kv_bench as B  # noqa: E402
import kv_receipt as R  # noqa: E402
import kv_twin as T  # noqa: E402

MODELS = {
    "8b": {"heads": 32, "layers": 32,
           "shapes": {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}},
    "70b": {"heads": 64, "layers": 80, "shapes": B.SHAPES_70B},
}
FORMS = {0: "fma-plain", 1: "fma-phased", 2: "fma-ws", 3: "mma-plain", 4: "mma-phased", 5: "mma-ws",
         6: "mma-occ-plain", 7: "mma-occ", 8: "mma-pers-plain", 9: "mma-pers", 10: "mma-ws-receipt",
         11: "mma-ws-bound", 12: "mma-dyn"}
PLAIN_OF = {1: 0, 2: 0, 4: 3, 5: 3, 7: 6, 9: 8, 10: 3, 11: 3, 12: 3}
RECEIPT = {10: 1, 11: 2}  # kv_attn's receipt forms and their rc (kv_receipt.NAMES)
RC_SEED = hashlib.sha256(b"ra-word-codec/kv-receipt-seed").digest()
SPLITS = (1, 2, 4, 8, 16, 32)
PERSISTENT = (8, 9)  # forms whose `splits` is the CTAs over the SMs' slots
PERS_SPLITS = (1, 2, 4)
RTOL = {0: 1e-3, 3: 8e-3, 6: 8e-3, 8: 8e-3}  # the tensor-core forms round P to bf16
SPIN = 60_000_000
WRITE_GRIDS = (0, 1, 2, 4)  # the encode's grid beside the GEMM: a thread a word (0), or this many CTAs an SM
ENC_SMEM_CARVEOUTS = (100,)  # the encode's preferred shared memory carveout (-1 the driver's choice): the most
CARVEOUTS = (24, 47, 94)  # SMs taken from the GEMM (cuBLASLt's SM count target) for the encode beside it
CARVE_GRIDS = (1, 2, 4, 6)  # the encode's CTAs a carved-out SM
OCC_CTAS = (2, 3, 4)  # the read-and-decode's floor at this many CTAs an SM
FLOOR_CTA_WARPS = 4  # kv.cu's WPB / 32


def shape(lib: B.Lib) -> dict:
    a = (ctypes.c_int * 10)()
    lib.so.kv_attn_shape(ctypes.cast(a, ctypes.c_void_p))
    sh = dict(zip(("heads", "kv_heads", "head_dim", "tlw", "tle", "tt", "part", "mw", "mma", "mt"), a))
    sh["receipts"] = rc_shape(lib)["forms"]
    return sh


def forms_of(sh: dict, only=None) -> list[int]:
    """The forms a build runs: the tensor-core ones where whole words fill a head, and a receipt form where its build
    runs it (form 11 only at 64-byte words)."""
    mma = [3, 4, 5, 6, 7, 8, 9] + [fm for fm, rc in RECEIPT.items() if sh["receipts"] >> (rc - 1) & 1] + [12]
    return [f for f in [0, 1, 2] + (mma if sh["mma"] else []) if only is None or f in only]


def rc_shape(lib: B.Lib) -> dict:
    a = (ctypes.c_int * 7)()
    lib.so.kv_rc_shape(ctypes.cast(a, ctypes.c_void_p))
    return dict(zip(("rt", "lag", "zw", "mdw", "mt", "steps", "forms"), a))


def rc_setup(lib: B.Lib, seed: bytes) -> dict:
    """The receipt forms' sign rows and keys under `seed`, each form the build runs; their sign bits by rc."""
    v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    so = lib.so
    so.kv_rc_set.argtypes, so.kv_rc_buffers.argtypes, so.kv_rc_tag.argtypes = [i, v, v], [v, v, v], [i, q, v]
    so.kv_rc_l2.argtypes, so.kv_rc_tag16.argtypes, so.kv_rc_tag8.argtypes = [q, v, v], [i, q, v], [i, q, v]
    rs = rc_shape(lib)
    bits = {}
    for rc in (rc for rc in (1, 2) if rs["forms"] >> (rc - 1) & 1):
        bits[rc] = R.sign_bits(seed, rc, 32 * rs["steps"] * rs["rt"])
        table, key = R.rows(bits[rc], rs["rt"], rc), R.key_words(seed, rc)
        lib.call("kv_rc_set", rc, table.ctypes.data, key.ctypes.data)
    so.kv_rc_sha_set.argtypes, so.kv_rc_sha8.argtypes = [ctypes.c_char_p], [q, v]
    lib.call("kv_rc_sha_set", R.sha_prefix(seed))
    return bits


def sha_equal(sk, tg, nr: int, ends: int) -> bool:
    """The first and last `ends` of kv_rc_sha8's nr receipts (64 bytes each in the receipt slots) equal kv_receipt's
    over the same sketches."""
    skn, tgn = sk.cpu().numpy(), tg.cpu().numpy().view(np.uint8).reshape(-1)
    return all(R.sha_receipt(RC_SEED, u, skn[8 * u: 8 * u + 8]) == tgn[64 * u: 64 * u + 64].tobytes()
               for u in sorted(set(range(min(ends, nr))) | set(range(max(0, nr - ends), nr))))


def split_tokens(ctx: int, splits: int, tile: int) -> int:
    """kv_attn's tokens a split at `splits` for a form of `tile` tokens a tile (or a receipt block)."""
    per = -(-ctx // splits)
    return -(-per // tile) * tile


def receipt_vectors(sh: dict, rs: dict, b: int, ctx: int) -> int:
    return b * sh["kv_heads"] * -(-ctx // (rs["rt"] * rs["mt"])) * rs["mdw"] * 16


def cta_lanes(encl, seq: int, g: int, t0: int, t1: int, rs: dict):
    """A CTA's raw words as its decoder lanes hold them, (tiles padded to whole blocks, MDW, 32, WB) uint8: lane l of
    warp i holds word l % HW of token (32 / HW) i + l // HW, its KV head's K words, then its V words."""
    mt, rt, mdw, wb = rs["mt"], rs["rt"], rs["mdw"], 16 * rs["steps"]
    kw = 2 * K.HEAD_DIM // wb
    nt = -(-(t1 - t0) // mt)
    tp = -(-nt // rt) * rt
    toks = np.zeros((tp * mt, 2 * kw, wb), np.uint8)
    words = encl[seq, t0:t1].reshape(t1 - t0, -1, wb)
    half = K.TOKEN_LAYER_BYTES // wb // 2
    toks[: t1 - t0] = words[:, [kw * g + j for j in range(kw)] + [half + kw * g + j for j in range(kw)]]
    return toks.reshape(tp, mdw, R.LANES, wb)


def receipt_gate(lib: B.Lib, torch, m: dict, cells, ks, log, layers_override: int | None = None) -> dict:
    """Forms 10 and 11 at each gate cell, k of ks and split count of SPLITS, at layer L - 1: the attention output equals
    form 5's bit for bit at every split count where the two split the tokens alike; every sketch equals kv_receipt's
    from the raw words (form 11's from the z it dumped); the z after each CTA's last tile equals its partials bit for
    bit; the first and last 64 receipts equal kv_receipt's; and so do the first and last 32 block receipts
    (kv_rc_tag16 over the same sketches), half-block receipts (kv_rc_tag8) and SHA-512 half-block receipts
    (kv_rc_sha8). Form 11 runs only at 64-byte words."""
    sh, rs, f = shape(lib), rc_shape(lib), lib.field
    hw = 4 * K.HEAD_DIM // f.word_bytes
    if not sh["mma"] or sh["kv_heads"] != K.KV_HEADS or rs["mdw"] * R.LANES != rs["mt"] * hw or not sh["receipts"]:
        return {}
    layers = layers_override or m["layers"]
    layer, st, tl, tle = layers - 1, B.stream(torch), K.TOKEN_LAYER_BYTES, sh["tle"]
    mt, rt, zw, mw, group = rs["mt"], rs["rt"], rs["zw"], sh["mw"], sh["heads"] // sh["kv_heads"]
    bits = rc_setup(lib, RC_SEED)
    zl = group // 2
    w_, gid, t4, mt_, e = np.meshgrid(np.arange(mw), np.arange(8), np.arange(zl), np.arange(8), np.arange(4),
                                      indexing="ij")
    z_word = ((w_ * 8 * zl + gid * zl + t4) * 32 + 4 * mt_ + e).ravel()
    z_part = (w_ * sh["part"] + (2 * t4 + (e & 1)) * sh["head_dim"] + 16 * mt_ + gid + 8 * (e >> 1)).ravel()
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    out = {}
    for b, ctx in cells:
        g = torch.Generator(device="cuda").manual_seed(b * 100003 + ctx + 1)
        plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.float32, device="cuda", generator=g).to(
            torch.bfloat16)
        pad = torch.zeros(b * ctx * layers, tle, dtype=torch.uint8, device="cuda")
        pad[:, :tl] = plain.view(torch.uint8).view(-1, tl)
        enc = torch.empty_like(pad)
        q = (torch.randn(b, sh["heads"], K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16)
        nvec = receipt_vectors(sh, rs, b, ctx)
        sk = torch.zeros(nvec, 16, dtype=torch.int32, device="cuda")
        tg = torch.zeros(nvec, 8, dtype=torch.int32, device="cuda")
        part = part_buffer(torch, sh, b)
        o5, orc = (torch.empty(b, sh["heads"], K.HEAD_DIM, dtype=torch.float32, device="cuda") for _ in range(2))
        cell = {"batch": b, "context": ctx, "layer": layer, "equal": {}, "sketch": {}, "z": {}, "receipts": {},
                "block_receipts": {}, "half_block_receipts": {}, "sha_receipts": {}}
        for k in ks:
            B.set_key(lib, B.lib_key(lib, k))
            lib.call("kv_encode", B.ptr(pad), B.ptr(enc), pad.numel() // f.word_bytes, 0, k, 0, st)
            encl = enc.view(b, ctx, layers, tle)[:, :, layer].cpu().numpy()
            for sp in SPLITS:
                per, per5 = split_tokens(ctx, sp, rt * mt), split_tokens(ctx, sp, mt)
                nsp = -(-ctx // per)
                attn_call(lib, st, enc, 5, q, o5, part, b, ctx, layers, layer, sp, k)
                for fm, rc in RECEIPT.items():
                    if fm not in forms_of(sh):
                        continue
                    key = f"{FORMS[fm]}@k{k}"
                    zd = (torch.zeros(b * nsp * K.KV_HEADS * (per // mt) * zw, dtype=torch.int32, device="cuda")
                          if rc == 2 else None)
                    sk.zero_()
                    tg.zero_()
                    lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None if zd is None else B.ptr(zd))
                    orc.fill_(float("nan"))
                    attn_call(lib, st, enc, fm, q, orc, part, b, ctx, layers, layer, sp, k)
                    torch.cuda.synchronize()
                    if per == per5 or min(per, per5) >= ctx:
                        cell["equal"][key] = cell["equal"].get(key, True) and bool(torch.equal(orc, o5))
                    skn, zdn = sk.cpu().numpy(), None if zd is None else zd.cpu().numpy().view(np.uint32)
                    partn = part.cpu().numpy().view(np.uint32)
                    sk_ok = z_ok = True
                    for seq in range(b):
                        for split in range(nsp):
                            t0, t1 = split * per, min(ctx, (split + 1) * per)
                            nt = -(-(t1 - t0) // mt)
                            for hd in range(K.KV_HEADS):
                                lanes = cta_lanes(encl, seq, hd, t0, t1, rs)
                                cta = (seq * nsp + split) * K.KV_HEADS + hd
                                z = None
                                if zdn is not None:
                                    zc = zdn.reshape(-1, per // mt, zw)[cta, :nt]
                                    z = np.zeros((lanes.shape[0], zw), np.uint32)
                                    z[:nt] = zc
                                    want = partn[(seq * nsp + split) * mw * sh["part"] + hd * group * sh["head_dim"]
                                                 + z_part]
                                    z_ok &= bool(np.array_equal(zc[nt - 1][z_word], want))
                                ref = R.sketch(lanes, bits[rc], rt, z, rs["lag"], zw)
                                nb = ref.shape[0]
                                v0 = ((seq * K.KV_HEADS + hd) * -(-ctx // (rt * mt)) + t0 // (rt * mt)) * rs["mdw"] * 16
                                got = skn[v0: v0 + nb * rs["mdw"] * 16].reshape(ref.shape)
                                sk_ok &= bool(np.array_equal(got, ref))
                    cell["sketch"][key] = cell["sketch"].get(key, True) and sk_ok
                    if zdn is not None:
                        cell["z"][key] = cell["z"].get(key, True) and z_ok
                    tgn = tg.cpu().numpy()
                    rc_ok = all(R.receipt(RC_SEED, rc, v, skn[v]) == tgn[v].astype("<i4").tobytes()
                                for v in list(range(64)) + list(range(nvec - 64, nvec)))
                    cell["receipts"][key] = cell["receipts"].get(key, True) and rc_ok
                    nblk = nvec // 16
                    lib.call("kv_rc_tag16", rc, nblk, st)
                    torch.cuda.synchronize()
                    tbn = tg[:nblk].cpu().numpy()
                    blk_ok = all(R.receipt(RC_SEED, rc, u, skn[16 * u: 16 * u + 16]) == tbn[u].astype("<i4").tobytes()
                                 for u in sorted(set(range(min(32, nblk))) | set(range(max(0, nblk - 32), nblk))))
                    cell["block_receipts"][key] = cell["block_receipts"].get(key, True) and blk_ok
                    lib.call("kv_rc_tag8", rc, 2 * nblk, st)
                    torch.cuda.synchronize()
                    thn = tg[:2 * nblk].cpu().numpy()
                    half_ok = all(R.receipt(RC_SEED, rc, u, skn[8 * u: 8 * u + 8]) == thn[u].astype("<i4").tobytes()
                                  for u in sorted(set(range(min(32, 2 * nblk))) | set(range(max(0, 2 * nblk - 32),
                                                                                              2 * nblk))))
                    cell["half_block_receipts"][key] = cell["half_block_receipts"].get(key, True) and half_ok
                    lib.call("kv_rc_sha8", 2 * nblk, st)
                    torch.cuda.synchronize()
                    cell["sha_receipts"][key] = cell["sha_receipts"].get(key, True) and sha_equal(sk, tg, 2 * nblk, 32)
                    del zd
        names = ("equal", "sketch", "receipts", "block_receipts", "half_block_receipts", "sha_receipts")
        cell["ok"] = all(all(cell[n].values()) and cell[n] for n in names) and all(
            cell["z"].values()) and (bool(cell["z"]) or 11 not in forms_of(sh))
        log(f"receipt gate {lib.name} B={b} ctx={ctx} L={layers} layer {layer}: {'ok' if cell['ok'] else 'FAIL'}; "
            + "; ".join(f"{n} " + ", ".join(f"{a} {v}" for a, v in cell[n].items()) for n in names + ("z",)))
        out[f"b{b}_ctx{ctx}"] = cell
        del plain, pad, enc, q, sk, tg, part, o5, orc
        torch.cuda.empty_cache()
    return out


def attn_occ(lib: B.Lib) -> dict:
    a = (ctypes.c_int * 5)()
    if lib.so.kv_attn_occ(ctypes.cast(a, ctypes.c_void_p)) != 0:
        return {}
    return {FORMS[5]: {"ctas_per_sm": a[0], "decoder_warps_per_sm": a[0] * a[1]},
            FORMS[7]: {"ctas_per_sm": a[2], "decoder_warps_per_sm": a[2] * a[3]},
            FORMS[10]: {"ctas_per_sm": a[4], "decoder_warps_per_sm": a[4] * a[1]}}


def attn_call(lib, st, store, form, q, o, part, b, ctx, layers, layer, splits, k) -> None:
    lib.call("kv_attn", B.ptr(store), form, B.ptr(q), B.ptr(o), B.ptr(part), b, ctx, layers, layer, splits, k, st)


def part_buffer(torch, sh: dict, b: int):
    return torch.empty(b * max(SPLITS) * sh["mw"] * sh["part"], dtype=torch.float32, device="cuda")


def attn_gate(lib: B.Lib, torch, m: dict, cells, ks, log, layers_override: int | None = None) -> dict:
    sh, f = shape(lib), lib.field
    heads, layers = sh["heads"], layers_override or m["layers"]
    if heads != m["heads"] or sh["tlw"] != K.token_layer_words(f):
        raise RuntimeError(f"{lib.name}: kv_attn_shape {sh} is not the model's ({m['heads']} heads)")
    st = B.stream(torch)
    tl, tle = K.TOKEN_LAYER_BYTES, sh["tle"]
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    forms = [fm for fm in forms_of(sh) if fm not in RECEIPT]
    out = {}
    for b, ctx in cells:
        g = torch.Generator(device="cuda").manual_seed(b * 100003 + ctx)
        plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.float32, device="cuda", generator=g).to(
            torch.bfloat16)
        pad = torch.zeros(b * ctx * layers, tle, dtype=torch.uint8, device="cuda")
        pad[:, :tl] = plain.view(torch.uint8).view(-1, tl)
        enc = torch.empty_like(pad)
        q = (torch.randn(b, heads, K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16).contiguous()
        part = part_buffer(torch, sh, b)
        o = {fm: torch.empty(b, heads, K.HEAD_DIM, dtype=torch.float32, device="cuda") for fm in forms}
        kvl = plain.view(b, ctx, layers, 2, K.KV_HEADS, K.HEAD_DIM)
        cell = {"batch": b, "context": ctx, "layers": layers, "splits": list(SPLITS), "equal": {}, "rel_err": {},
                "mma_vs_fma": 0.0}
        for k in ks:
            B.set_key(lib, B.lib_key(lib, k))
            lib.call("kv_encode", B.ptr(pad), B.ptr(enc), pad.numel() // f.word_bytes, 0, k, 0, st)
            for layer in (0, layers - 1):
                kk, vv = kvl[:, :, layer, 0].double(), kvl[:, :, layer, 1].double()
                qq = q.double().view(b, K.KV_HEADS, heads // K.KV_HEADS, K.HEAD_DIM)
                s = torch.einsum("bgqd,btgd->bgqt", qq, kk)
                p = torch.exp2(s - s.amax(-1, keepdim=True))
                ref = torch.einsum("bgqt,btgd->bgqd", p / p.sum(-1, keepdim=True), vv).reshape(b, heads, K.HEAD_DIM)
                del kk, vv, s, p
                for sp in SPLITS:
                    for fm in forms:
                        o[fm].fill_(float("nan"))
                        attn_call(lib, st, plain if fm in RTOL else enc, fm, q, o[fm], part, b, ctx, layers, layer, sp, k)
                    torch.cuda.synchronize()
                    for fm in forms:
                        if fm in RTOL:
                            err = float((o[fm].double() - ref).abs().max() / ref.abs().max())
                            cell["rel_err"][FORMS[fm]] = max(cell["rel_err"].get(FORMS[fm], 0.0), err)
                        else:
                            key = f"{FORMS[fm]}@k{k}"
                            cell["equal"][key] = cell["equal"].get(key, True) and bool(torch.equal(o[fm], o[PLAIN_OF[fm]]))
                    if 3 in forms:
                        d = float((o[3].double() - o[0].double()).abs().max() / ref.abs().max())
                        cell["mma_vs_fma"] = max(cell["mma_vs_fma"], d)
                del ref
        cell["ok"] = all(cell["equal"].values()) and all(
            cell["rel_err"].get(FORMS[fm], 1.0) <= tol for fm, tol in RTOL.items() if fm in forms)
        log(f"attn gate {lib.name} B={b} ctx={ctx} L={layers}: {'ok' if cell['ok'] else 'FAIL'}; equal "
            + ", ".join(f"{n} {v}" for n, v in cell["equal"].items()) + "; rel err "
            + ", ".join(f"{n} {v:.1e}" for n, v in cell["rel_err"].items()) + f"; mma vs fma {cell['mma_vs_fma']:.1e}")
        out[f"b{b}_ctx{ctx}"] = cell
        del plain, pad, enc, q, part, o, kvl
        torch.cuda.empty_cache()
    return out


def fit(ks, ys) -> dict:
    """y = c k - h by least squares."""
    n = len(ks)
    if n < 2:
        return {}
    mk, my = sum(ks) / n, sum(ys) / n
    c = sum((a - mk) * (b - my) for a, b in zip(ks, ys)) / sum((a - mk) ** 2 for a in ks)
    return {"c": c, "h": c * mk - my}


def prim_gate(lib: B.Lib, torch, log) -> dict:
    """kv_prim's ops at the twin's rare_cases (each op's rare carry or borrow, edges, random words) against
    kv_twin.prim_want, bit for bit."""
    v, i = ctypes.c_void_p, ctypes.c_int
    lib.so.kv_prim.argtypes = [i, v, v, v, i, v]
    f = lib.field
    out, st = {}, B.stream(torch)
    for op, cases in T.rare_cases(f, random.Random(f.n + 7)).items():
        a = B.to_device(torch, [x for x, _ in cases], f)
        b = B.to_device(torch, [y for _, y in cases], f)
        y = torch.empty_like(a)
        lib.call("kv_prim", op, B.ptr(a), B.ptr(b), B.ptr(y), len(cases), st)
        torch.cuda.synchronize()
        got = B.from_device(y, f)
        bad = [j for j, ((x, z), g) in enumerate(zip(cases, got)) if g != T.prim_want(f, op, x, z)]
        out[op] = {"cases": len(cases), "bad": bad, "ok": not bad}
    ok = all(r["ok"] for r in out.values())
    log(f"prim gate {lib.name}: " + ("ok" if ok else "FAIL") + " (" + ", ".join(
        f"{T.PRIMS[op]} {r['cases']}" + (f" bad {r['bad']}" if r["bad"] else "")
        for op, r in out.items()) + ")")
    return {"ok": ok, "ops": out}


def floor_gate(lib: B.Lib, torch, store, n_words: int, k: int, grids, st) -> list:
    """The grids whose read-and-decode checksum differs from one word a thread's (no tweak stepping)."""
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")

    def checksum(grid):
        xr.zero_()
        lib.call("kv_decode", B.ptr(store), 0, n_words, 0, k, grid, B.ptr(xr), st)
        torch.cuda.synchronize()
        return int(xr.item()) & 0xFFFFFFFF

    want = checksum(0)
    return [g for g in grids if checksum(g) != want]


def floor_occ(lib: B.Lib, torch, store, n_words: int, k: int, sms: int, reps: int, r: float, st) -> dict:
    """The read-and-decode at OCC_CTAS CTAs an SM, each checksum-gated against one word a thread's; r the read in
    bytes/s."""
    v, q, i = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int
    lib.so.kv_decode_occ.argtypes = [v, q, q, i, i, v, i, v, v]
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    xr.zero_()
    lib.call("kv_decode", B.ptr(store), 0, n_words, 0, k, 0, B.ptr(xr), st)
    torch.cuda.synchronize()
    want = int(xr.item()) & 0xFFFFFFFF
    out = {}
    for c in OCC_CTAS:
        got, best = ctypes.c_int(0), None
        for mult in (1, 2, 4):
            def run():
                lib.call("kv_decode_occ", B.ptr(store), n_words, 0, k, sms * c * mult, B.ptr(xr), c, ctypes.byref(got),
                         st)
            xr.zero_()
            run()
            torch.cuda.synchronize()
            if int(xr.item()) & 0xFFFFFFFF != want:
                return {"ok": False, "ctas": c, "grid_per_sm": c * mult}
            ts = B.timed(torch, run, reps)
            if best is None or statistics.median(ts) < best[1]:
                best = (mult, statistics.median(ts), ts)
        out[c] = {"ctas_per_sm": got.value, "warps_per_sm": got.value * FLOOR_CTA_WARPS, "grid_per_sm": c * best[0],
                  "s": best[1], "times_s": best[2], "over_read": best[1] * r / store.numel()}
    return {"ok": True, "by_ctas": out}


def exhaust_cell(lib: B.Lib, torch, m: dict, b: int, ctx: int, ks, reps: int, sms: int, log, only=None) -> dict:
    sh, f = shape(lib), lib.field
    heads, layers, tle = sh["heads"], m["layers"], sh["tle"]
    st = B.stream(torch)
    n_words = b * ctx * layers * sh["tlw"]
    store = B.random_words(torch, n_words, f, b * 7919 + ctx)
    per_layer = b * ctx * tle
    rd = B.read_rate(lib, torch, store, sms, reps)
    r = rd["gbps"] * 1e9
    t_read = per_layer / r
    cell = {"batch": b, "context": ctx, "layers": layers, "store_bytes": store.numel(), "layer_bytes": per_layer,
            "read": rd, "read_layer_s": t_read, "floor": {}, "forms": {}}
    log(f"exhaust {lib.name} B={b} ctx={ctx} L={layers}: store {store.numel() / 1e9:.1f} GB, read {rd['gbps']:.0f} GB/s")
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    for k in ks:
        B.set_key(lib, B.lib_key(lib, k))
        bad = floor_gate(lib, torch, store, n_words, k, [sms * mult for mult in (4, 8, 16)], st)
        if bad:
            cell["floor"][k] = {"ok": False, "bad_grids": bad}
            log(f"exhaust {lib.name} B={b} ctx={ctx} k={k}: read-and-decode checksum FAIL at grids {bad}")
            continue
        best = None
        for mult in (4, 8, 16):
            ts = B.timed(torch, lambda: lib.call("kv_decode", B.ptr(store), 0, n_words, 0, k, sms * mult, B.ptr(xr), st),
                         reps)
            if best is None or statistics.median(ts) < best[1]:
                best = (mult, statistics.median(ts), ts)
        occ = floor_occ(lib, torch, store, n_words, k, sms, reps, r, st)
        cell["floor"][k] = {"grid_per_sm": best[0], "s": best[1], "times_s": best[2],
                            "over_read": best[1] * r / store.numel(), "occupancy": occ}
        if not occ["ok"]:
            cell["floor"][k]["ok"] = False
            log(f"exhaust {lib.name} B={b} ctx={ctx} k={k}: read-and-decode checksum FAIL at {occ['ctas']} CTAs an SM")
        else:
            log(f"exhaust {lib.name} B={b} ctx={ctx} k={k}: read-and-decode by decoder warps an SM, " + ", ".join(
                f"{v['warps_per_sm']} {v['over_read']:.3f}x" for v in occ["by_ctas"].values()))
    log(f"exhaust {lib.name} B={b} ctx={ctx}: read-and-decode " + ", ".join(
        f"k={k} {v['over_read']:.3f}x the read" for k, v in cell["floor"].items() if v.get("ok", True)))
    q = (torch.randn(b, heads, K.HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
    o = torch.empty(b, heads, K.HEAD_DIM, dtype=torch.float32, device="cuda")
    part = part_buffer(torch, sh, b)
    if any(fm in RECEIPT for fm in forms_of(sh, only)):
        rc_setup(lib, RC_SEED)
        nvec = receipt_vectors(sh, rc_shape(lib), b, ctx)
        sk = torch.empty(nvec, 16, dtype=torch.int32, device="cuda")
        sk2 = torch.empty_like(sk)
        tg = torch.empty(nvec, 8, dtype=torch.int32, device="cuda")
    for fm in forms_of(sh, only):
        for k in ([1] if fm in RTOL else ks):
            B.set_key(lib, B.lib_key(lib, k))
            if fm in RECEIPT:
                lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
            per = {}
            ctas = (lambda s: b * K.KV_HEADS * s) if fm >= 3 else (lambda s: b * s)
            sweep = PERS_SPLITS if fm in PERSISTENT else [s for s in SPLITS if ctas(s) >= sms] or [max(SPLITS)]
            for sp in sweep:
                def run_pass():
                    for l in range(layers):
                        attn_call(lib, st, store, fm, q, o, part, b, ctx, layers, l, sp, k)
                run_pass()
                ts = [B.queued(lib, torch, run_pass, SPIN) / layers for _ in range(reps)]
                per[sp] = {"s": statistics.median(ts), "times_s": ts}
            sp = min(per, key=lambda s: per[s]["s"])
            row = {"splits": sp, "s": per[sp]["s"], "by_splits": per, "over_read": per[sp]["s"] / t_read,
                   "gbps": per_layer / per[sp]["s"] / 1e9}
            if fm in RECEIPT:
                lib.call("kv_rc_buffers", B.ptr(sk), None, None)

                def sketch_pass():
                    for l in range(layers):
                        attn_call(lib, st, store, fm, q, o, part, b, ctx, layers, l, sp, k)

                def tag_pass():
                    for _ in range(layers):
                        lib.call("kv_rc_tag", RECEIPT[fm], nvec, st)

                sketch_pass()
                ts = [B.queued(lib, torch, sketch_pass, SPIN) / layers for _ in range(reps)]
                lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
                tag_pass()
                tt = [B.queued(lib, torch, tag_pass, SPIN) / layers for _ in range(reps)]
                row.update({"vectors": nvec, "sketch_only_s": statistics.median(ts), "sketch_only_times_s": ts,
                            "tag_s": statistics.median(tt), "tag_times_s": tt,
                            "sketch_only_over_read": statistics.median(ts) / t_read,
                            "tag_over_read": statistics.median(tt) / t_read})
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: at {sp} splits, the sketch alone "
                    f"{row['sketch_only_s'] * 1e3:.3f} ms ({row['sketch_only_over_read']:.3f}x the read), the receipts' "
                    f"kernel alone {row['tag_s'] * 1e3:.3f} ms ({row['tag_over_read']:.3f}x), {nvec} vectors a layer")
                win = (ctypes.c_uint64 * 2)()
                lib.call("kv_rc_l2", nvec * 64, st, ctypes.cast(win, ctypes.c_void_p))
                run_pass()
                tl = [B.queued(lib, torch, run_pass, SPIN) / layers for _ in range(reps)]
                tt = [B.queued(lib, torch, tag_pass, SPIN) / layers for _ in range(reps)]
                row["l2"] = {"window_bytes": win[0], "persisting_bytes": win[1], "s": statistics.median(tl),
                             "times_s": tl, "over_read": statistics.median(tl) / t_read, "tag_s": statistics.median(tt),
                             "tag_times_s": tt, "tag_over_read": statistics.median(tt) / t_read}
                lib.call("kv_rc_l2", 0, st, ctypes.cast(win, ctypes.c_void_p))
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: with {row['l2']['persisting_bytes'] / 1e6:.0f}"
                    f" MB of the sketches' {nvec * 64 / 1e6:.0f} MB held in L2, a layer {row['l2']['s'] * 1e3:.3f} ms "
                    f"({row['l2']['over_read']:.3f}x the read), the receipts' kernel alone {row['l2']['tag_s'] * 1e3:.3f}"
                    f" ms ({row['l2']['tag_over_read']:.3f}x)")
                nblk = nvec // 16

                def tag16_pass():
                    for _ in range(layers):
                        lib.call("kv_rc_tag16", RECEIPT[fm], nblk, st)

                def blocks_equal(skb=sk, tgb=tg, grp=16) -> bool:
                    nr = nvec // grp
                    skn, tgn = skb.cpu().numpy(), tgb[:nr].cpu().numpy()
                    return all(R.receipt(RC_SEED, RECEIPT[fm], u, skn[grp * u: grp * u + grp])
                               == tgn[u].astype("<i4").tobytes() for u in list(range(8)) + list(range(nr - 8, nr)))

                tag16_pass()
                tt = [B.queued(lib, torch, tag16_pass, SPIN) / layers for _ in range(reps)]
                ok = blocks_equal()
                row["tag16"] = {"blocks": nblk, "s": statistics.median(tt), "times_s": tt,
                                "over_read": statistics.median(tt) / t_read, "receipts_equal": ok}
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: a receipt a block of 16 vectors, its "
                    f"kernel alone {row['tag16']['s'] * 1e3:.3f} ms ({row['tag16']['over_read']:.3f}x the read), "
                    f"{nblk} receipts a layer; the first and last 8 equal kv_receipt's: {ok}")
                if not ok:
                    raise SystemExit(f"{FORMS[fm]} k={k}: a block receipt differs from kv_receipt's")

                def block_pass(fn="kv_rc_tag16", nr=nblk):
                    for l in range(layers):
                        lib.call("kv_rc_buffers", B.ptr(sk), None, None)
                        attn_call(lib, st, store, fm, q, o, part, b, ctx, layers, l, sp, k)
                        lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
                        lib.call(fn, *(() if fn == "kv_rc_sha8" else (RECEIPT[fm],)), nr, st)

                for l2 in (False, True):
                    persisting = 0
                    if l2:
                        lib.call("kv_rc_l2", nvec * 64, st, ctypes.cast(win, ctypes.c_void_p))
                        persisting = win[1]
                    block_pass()
                    tb = [B.queued(lib, torch, block_pass, SPIN) / layers for _ in range(reps)]
                    lib.call("kv_rc_l2", 0, st, ctypes.cast(win, ctypes.c_void_p))
                    ok = blocks_equal()
                    key = "pass_l2" if l2 else "pass"
                    row["tag16"][key] = {"s": statistics.median(tb), "times_s": tb, "persisting_bytes": persisting,
                                         "over_read": statistics.median(tb) / t_read, "receipts_equal": ok}
                    log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: the sketch, then a receipt a block "
                        f"of 16 vectors, a layer {row['tag16'][key]['s'] * 1e3:.3f} ms "
                        f"({row['tag16'][key]['over_read']:.3f}x the read)"
                        + (f", the sketches held in L2 ({persisting / 1e6:.0f} MB persisting)" if l2 else "")
                        + f"; the first and last 8 equal kv_receipt's: {ok}")
                    if not ok:
                        raise SystemExit(f"{FORMS[fm]} k={k}: a block receipt after the sketch differs from kv_receipt's")

                def tag8_pass():
                    for _ in range(layers):
                        lib.call("kv_rc_tag8", RECEIPT[fm], 2 * nblk, st)

                lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
                tag8_pass()
                tt = [B.queued(lib, torch, tag8_pass, SPIN) / layers for _ in range(reps)]
                ok = blocks_equal(grp=8)
                row["tag8"] = {"receipts": 2 * nblk, "s": statistics.median(tt), "times_s": tt,
                               "over_read": statistics.median(tt) / t_read, "receipts_equal": ok}
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: two receipts a block, 8 vectors each, "
                    f"their kernel alone {row['tag8']['s'] * 1e3:.3f} ms ({row['tag8']['over_read']:.3f}x the read); "
                    f"the first and last 8 equal kv_receipt's: {ok}")
                if not ok:
                    raise SystemExit(f"{FORMS[fm]} k={k}: a half-block receipt differs from kv_receipt's")

                def half_block_pass():
                    block_pass("kv_rc_tag8", 2 * nblk)

                half_block_pass()
                tb = [B.queued(lib, torch, half_block_pass, SPIN) / layers for _ in range(reps)]
                ok = blocks_equal(grp=8)
                row["tag8"]["pass"] = {"s": statistics.median(tb), "times_s": tb,
                                       "over_read": statistics.median(tb) / t_read, "receipts_equal": ok}
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: the sketch, then two receipts a block of "
                    f"8 vectors each, a layer {row['tag8']['pass']['s'] * 1e3:.3f} ms "
                    f"({row['tag8']['pass']['over_read']:.3f}x the read); the first and last 8 equal kv_receipt's: {ok}")
                if not ok:
                    raise SystemExit(f"{FORMS[fm]} k={k}: a half-block receipt after the sketch differs from "
                                     "kv_receipt's")

                def sha_pass():
                    for _ in range(layers):
                        lib.call("kv_rc_sha8", 2 * nblk, st)

                def sha_block_pass():
                    block_pass("kv_rc_sha8", 2 * nblk)

                lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
                sha_pass()
                tt = [B.queued(lib, torch, sha_pass, SPIN) / layers for _ in range(reps)]
                ok = sha_equal(sk, tg, 2 * nblk, 8)
                sha_block_pass()
                tb = [B.queued(lib, torch, sha_block_pass, SPIN) / layers for _ in range(reps)]
                ok_pass = sha_equal(sk, tg, 2 * nblk, 8)
                row["sha8"] = {"receipts": 2 * nblk, "s": statistics.median(tt), "times_s": tt,
                               "over_read": statistics.median(tt) / t_read, "receipts_equal": ok,
                               "pass": {"s": statistics.median(tb), "times_s": tb,
                                        "over_read": statistics.median(tb) / t_read, "receipts_equal": ok_pass}}
                log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: two SHA-512 receipts a block, 8 vectors "
                    f"each, their kernel alone {row['sha8']['s'] * 1e3:.3f} ms ({row['sha8']['over_read']:.3f}x the "
                    f"read); the sketch, then them, a layer {row['sha8']['pass']['s'] * 1e3:.3f} ms "
                    f"({row['sha8']['pass']['over_read']:.3f}x the read); the first and last 8 equal hashlib's: "
                    f"{ok and ok_pass}")
                if not (ok and ok_pass):
                    raise SystemExit(f"{FORMS[fm]} k={k}: a SHA-512 half-block receipt differs from hashlib's")

                hi, lo = torch.cuda.Stream(priority=-1), torch.cuda.Stream(priority=0)
                bufs, tgs = (sk, sk2), (tg, tg[2 * nblk:])

                for grp, fn, nr in ((16, "kv_rc_tag16", nblk), (8, "kv_rc_tag8", 2 * nblk)):
                    def overlap_pass():
                        hashed = [None, None]
                        for l in range(layers):
                            if hashed[l % 2] is not None:
                                hi.wait_event(hashed[l % 2])
                            lib.call("kv_rc_buffers", B.ptr(bufs[l % 2]), None, None)
                            attn_call(lib, hi.cuda_stream, store, fm, q, o, part, b, ctx, layers, l, sp, k)
                            read = torch.cuda.Event()
                            read.record(hi)
                            lo.wait_event(read)
                            lib.call("kv_rc_buffers", B.ptr(bufs[l % 2]), B.ptr(tgs[l % 2]), None)
                            lib.call(fn, RECEIPT[fm], nr, lo.cuda_stream)
                            hashed[l % 2] = torch.cuda.Event()
                            hashed[l % 2].record(lo)
                        for e in hashed:
                            hi.wait_event(e)

                    with torch.cuda.stream(hi):
                        overlap_pass()
                        to = [B.queued(lib, torch, overlap_pass, SPIN) / layers for _ in range(reps)]
                    torch.cuda.synchronize()
                    last = (layers - 1) % 2
                    ok = blocks_equal(bufs[last], tgs[last], grp)
                    ov = row["tag16" if grp == 16 else "tag8"]["overlap"] = {
                        "s": statistics.median(to), "times_s": to, "over_read": statistics.median(to) / t_read,
                        "receipts_equal": ok}
                    log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k}: the sketch, with layer l's receipts "
                        f"of {grp} vectors on a second stream of lower priority beside layer l + 1's attention, a "
                        f"layer {ov['s'] * 1e3:.3f} ms ({ov['over_read']:.3f}x the read); the last layer's first and "
                        f"last 8 equal kv_receipt's: {ok}")
                    if not ok:
                        raise SystemExit(f"{FORMS[fm]} k={k}: an overlapped receipt of {grp} vectors differs from "
                                         "kv_receipt's")
            cell["forms"].setdefault(FORMS[fm], {})[k] = row
            log(f"exhaust {lib.name} B={b} ctx={ctx} {FORMS[fm]}" + ("" if fm in RTOL else f" k={k}")
                + f": a layer {row['s'] * 1e3:.3f} ms at {sp} splits, {row['gbps']:.0f} GB/s, {row['over_read']:.3f}x"
                " the read")
    for fm in forms_of(sh, only):
        if fm in RTOL:
            continue
        fe, fp = cell["forms"][FORMS[fm]], cell["forms"][FORMS[PLAIN_OF[fm]]][1]
        for k, row in fe.items():
            row["overhead_vs_read"] = row["over_read"] - 1.0
            row["overhead_vs_plain"] = (row["s"] - fp["s"]) / t_read
            if cell["floor"].get(k, {}).get("ok", True) and k in cell["floor"]:
                row["h"] = cell["floor"][k]["over_read"] - row["overhead_vs_read"]
        fe["fit_vs_read"] = fit(ks, [fe[k]["overhead_vs_read"] for k in ks])
        fe["fit_vs_plain"] = fit(ks, [fe[k]["overhead_vs_plain"] for k in ks])
    ks_f = [k for k, v in cell["floor"].items() if v.get("ok", True)]
    cell["ok"] = len(ks_f) == len(cell["floor"])
    cell["c"] = {k: cell["floor"][k]["over_read"] / k for k in ks_f}
    cell["floor_fit"] = fit(ks_f, [cell["floor"][k]["over_read"] - 1.0 for k in ks_f])
    log(f"exhaust {lib.name} B={b} ctx={ctx}: overhead a KV byte over the read, " + "; ".join(
        f"{FORMS[fm]} " + ", ".join(f"k={k} {cell['forms'][FORMS[fm]][k]['overhead_vs_read']:+.3f}" for k in ks)
        for fm in forms_of(sh, only) if fm not in RTOL) + "; floor " + ", ".join(
        f"k={k} {cell['floor'][k]['over_read'] - 1:+.3f}" for k in ks_f))
    del store, q, o, part
    torch.cuda.empty_cache()
    return cell


def weights(torch, m: dict):
    out = {}
    for name, (n, kd) in m["shapes"].items():
        g = torch.Generator(device="cuda").manual_seed(n + kd)
        w = (torch.randn(n, kd, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
        out[name] = [w.t()] + [w.clone().t() for _ in range(m["layers"] - 1)]
    return out


def write_side(lib: B.Lib, torch, m: dict, batches, ks, reps: int, sms: int, enc_rates: dict, log) -> dict:
    f = lib.field
    tw = m["layers"] * K.token_layer_words(f)
    one = torch.ones((), device="cuda")
    ws = weights(torch, m)
    w_bytes = sum(n * kd for n, kd in m["shapes"].values()) * m["layers"]
    hi = torch.cuda.Stream(priority=-1)
    lo = torch.cuda.Stream(priority=0)
    main = torch.cuda.current_stream()
    out = {"weight_bytes": w_bytes, "token_words": tw}
    for b in batches:
        xs = {name: (torch.randn(b, kd, device="cuda") * 0.5).to(torch.float8_e4m3fn)
              for name, (n, kd) in m["shapes"].items()}
        words = B.random_words(torch, tw * b, f, 11 + b)
        y = torch.empty_like(words)
        n = tw * b

        def gemm():
            for l in range(m["layers"]):
                for name in m["shapes"]:
                    torch._scaled_mm(xs[name], ws[name][l], one, one, out_dtype=torch.bfloat16)

        def both(do_gemm: bool, enc_grid: int | None, k: int) -> float:
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lib.call("kv_spin", SPIN, main.cuda_stream)
            a.record(main)
            ends = []
            if do_gemm:
                hi.wait_event(a)
                with torch.cuda.stream(hi):
                    gemm()
                ends.append(torch.cuda.Event())
                ends[-1].record(hi)
            if enc_grid is not None:
                lo.wait_event(a)
                lib.call("kv_encode", B.ptr(words), B.ptr(y), n, 0, k, enc_grid, lo.cuda_stream)
                ends.append(torch.cuda.Event())
                ends[-1].record(lo)
            for ev in ends:
                main.wait_event(ev)
            e.record(main)
            e.synchronize()
            return a.elapsed_time(e) / 1e3

        gemm()
        torch.cuda.synchronize()
        t_gemm = statistics.median([both(True, None, 1) for _ in range(reps)])
        row = {"gemm_s": t_gemm, "gemm_gbps": w_bytes / t_gemm / 1e9, "words": n, "bytes": n * f.word_bytes, "k": {}}
        for k in ks:
            B.set_key(lib, B.lib_key(lib, k))
            kr = {"sm_s": n / enc_rates[k]["per_s"], "grids": {}}
            for gr in WRITE_GRIDS:
                grid = gr * sms
                by = {}
                for cv in ENC_SMEM_CARVEOUTS:
                    lib.call("kv_encode_carveout", cv)
                    both(False, grid, k)
                    s_enc = [both(False, grid, k) for _ in range(reps)]
                    s_both = [both(True, grid, k) for _ in range(reps)]
                    t_enc, t_both = statistics.median(s_enc), statistics.median(s_both)
                    by[cv] = {"alone_s": t_enc, "with_gemm_s": t_both, "hidden_share": (t_gemm + t_enc - t_both) / t_enc,
                              "alone_samples_s": s_enc, "with_gemm_samples_s": s_both}
                cv = min(by, key=lambda c_: by[c_]["with_gemm_s"])
                t_enc, t_both = by[cv]["alone_s"], by[cv]["with_gemm_s"]
                kr["grids"][gr] = {"grid": grid, "alone_s": t_enc, "with_gemm_s": t_both, "smem_carveout": cv,
                                   "by_smem_carveout": by, "hidden_s": t_gemm + t_enc - t_both,
                                   "hidden_share": (t_gemm + t_enc - t_both) / t_enc}
            lib.call("kv_encode_carveout", 100)
            best = min(kr["grids"], key=lambda g_: kr["grids"][g_]["with_gemm_s"] - t_gemm)
            kr["best_grid"] = best
            kr["alone_s"] = min(v["alone_s"] for v in kr["grids"].values())
            kr["exposed_s"] = kr["grids"][best]["with_gemm_s"] - t_gemm
            kr["carveouts"] = {}
            for co in CARVEOUTS:
                torch._C._set_sm_carveout_experimental(co)
                try:
                    both(True, None, k)
                    t_gc = statistics.median([both(True, None, k) for _ in range(reps)])
                    cr = {"gemm_s": t_gc, "grids": {}}
                    for gr in CARVE_GRIDS:
                        t_both = statistics.median([both(True, co * gr, k) for _ in range(reps)])
                        cr["grids"][gr] = {"grid": co * gr, "with_gemm_s": t_both,
                                           "hidden_share": (t_gemm + kr["alone_s"] - t_both) / kr["alone_s"]}
                finally:
                    torch._C._set_sm_carveout_experimental(None)
                cr["best_grid"] = min(cr["grids"], key=lambda g_: cr["grids"][g_]["with_gemm_s"])
                cr["exposed_s"] = cr["grids"][cr["best_grid"]]["with_gemm_s"] - t_gemm
                kr["carveouts"][co] = cr
            lib.call("kv_encode_carveout", -1)
            kr["exposed_carved_s"] = min((v["exposed_s"] for v in kr["carveouts"].values()), default=math.nan)
            row["k"][k] = kr
            log(f"write {lib.name} B={b} k={k}: {n * f.word_bytes / 1e6:.1f} MB; encode alone {kr['alone_s'] * 1e3:.2f} ms "
                f"(compute-only rate {kr['sm_s'] * 1e3:.2f} ms); GEMM alone {t_gemm * 1e3:.2f} ms "
                f"({row['gemm_gbps']:.0f} GB/s); beside it " + ", ".join(
                    f"grid {g_}: +{(v['with_gemm_s'] - t_gemm) * 1e3:.2f} ms ({v['hidden_share']:.0%} hidden; "
                    + "/".join(f"{c_}: {w['hidden_share']:.0%}" for c_, w in v["by_smem_carveout"].items()) + ")"
                    for g_, v in kr["grids"].items()))
            log(f"write {lib.name} B={b} k={k}: with SMs carved out of the GEMM, " + "; ".join(
                f"{co} SMs: GEMM {cr['gemm_s'] * 1e3:.2f} ms, both {cr['grids'][cr['best_grid']]['with_gemm_s'] * 1e3:.2f}"
                f" ms at {cr['best_grid']} CTAs an SM (+{cr['exposed_s'] * 1e3:.2f} ms over the whole GEMM, "
                f"{cr['grids'][cr['best_grid']]['hidden_share']:.0%} hidden)" for co, cr in kr["carveouts"].items()))
        out[b] = row
        del xs, words, y
    del ws
    torch.cuda.empty_cache()
    return out


def model_check(res: dict, lib_name: str, m: dict, cells, tle: int) -> dict:
    """f_KV k x1 / ctx (the write side in reads of the step's bytes) against the measured encode, at each cell's
    batch and context: x1 is the compute-only encode a byte a round, in reads at R."""
    lr = res["libs"][lib_name]
    r = res["read"]["gbps"] * 1e9
    wb = K.field(lib_name.split("-")[0]).word_bytes
    enc = lr["rates"]["encode"]
    x1 = r / (enc[min(enc)]["per_s"] * min(enc) * wb)
    out = {"x1": x1, "rows": []}
    w = lr.get("write", {})
    for b, ctx in cells:
        kv_bytes = b * ctx * m["layers"] * tle
        step_read = (kv_bytes + w.get("weight_bytes", 0)) / r
        f_kv = kv_bytes / (kv_bytes + w.get("weight_bytes", 0))
        for k, kr in w.get(b, {}).get("k", {}).items():
            out["rows"].append({"batch": b, "context": ctx, "k": k, "f_kv": f_kv, "step_read_s": step_read,
                                "predicted": f_kv * k * x1 / ctx,
                                "measured_alone": kr["alone_s"] / step_read,
                                "measured_exposed": kr["exposed_s"] / step_read,
                                "measured_exposed_carved": kr["exposed_carved_s"] / step_read})
    return out


def write_result(res: dict, libs, out_dir: Path) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if not run_dir or not run_id:
        return
    ms = [{"name": "read", "value": res["read"]["gbps"], "unit": "GB/s"}] if "read" in res else []
    for lib in libs:
        for name, cell in res["libs"][lib.name].get("exhaust", {}).items():
            for form, rows in cell["forms"].items():
                for k, row in rows.items():
                    if isinstance(k, int):
                        ms.append({"name": f"{lib.name}_{name}_{form}_k{k}", "value": row["s"], "unit": "s"})
    # every process's files, where run.sh's KV_REPEAT runs this script more than once under one run
    names = ("kv-exhaust.json", "kv.log")
    files = sorted({os.path.relpath(p, run_dir) for p in (Path(run_dir) / "out").rglob("*") if p.name in names}
                   | {os.path.relpath(out_dir / n, run_dir) for n in names})
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {l.name: hashlib.sha256(Path(l.path).read_bytes()).hexdigest() for l in libs},
                                    "seed": B.SEED.hex()},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "kv codec and attention gates",
                          "detail": "device encode and decode = kv.py; each encoded attention form = its plain form bit "
                                    "for bit; plain forms within RTOL of float64", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def cells_of(s: str):
    return [tuple(int(v) for v in c.split("x")) for c in s.split(",") if c]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", action="append", required=True, help="NAME=PATH, NAME a field of kv.FIELDS (or FIELD-TAG)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="8b", choices=sorted(MODELS))
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--gate-cells", default="4x1000,2x4096,64x1024",
                    help="64 x 1024 has more of mma-dyn's work items than the SMs hold CTAs at every split")
    ap.add_argument("--cells", default="160x4096,80x8192,40x16384")
    ap.add_argument("--write-batches", default="", help="the write side's batches (default: the cells')")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--layers", type=int, default=0,
                    help="the cells' and the write side's layers (default: the model's); at 1 a layer's call reads "
                    "contiguous token-layers with the same grid")
    ap.add_argument("--forms", default="", help="the kv_attn forms the cells time (default all; every gate runs all), "
                    "each encoded form's plain form among them")
    ap.add_argument("--write-only", action="store_true",
                    help="time the write side only (the cells give the model check's step, not timed)")
    ap.add_argument("--sanitize", action="store_true",
                    help="two tiny gate cells of 2 layers (24 x 100 gives mma-dyn's CTAs several items), then stop")
    ap.add_argument("--ncu", action="store_true", help="launch each form of --forms once on one layer of the first "
                    "cell at the first k (plain forms at k = 1) and --ncu-splits, for Nsight Compute, then stop")
    ap.add_argument("--ncu-splits", type=int, default=8)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "kv.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    m = dict(MODELS[a.model], layers=a.layers or MODELS[a.model]["layers"])
    ks = [int(v) for v in a.ks.split(",")]
    only = {int(v) for v in a.forms.split(",") if v} or None
    if only and any(f in PLAIN_OF and PLAIN_OF[f] not in only for f in only):
        ap.error("--forms: every encoded form's plain form is timed too")
    libs = [B.Lib(*spec.split("=", 1)) for spec in a.lib]
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    res = {"device": props.name, "sms": sms, "total_memory": props.total_memory, "model": a.model, "ks": ks,
           "libs": {}, "ok": True}
    log(f"{props.name}: {sms} SMs, {props.total_memory / 1e9:.1f} GB; model {a.model}; libs "
        + ", ".join(f"{l.name} {shape(l)}" for l in libs))
    if a.sanitize:
        for lib in libs:
            g = attn_gate(lib, torch, m, [(2, 100), (24, 100)], ks[:1], log, layers_override=2)
            rg = receipt_gate(lib, torch, m, [(2, 100)], ks[:1], log, layers_override=2)
            res["ok"] &= all(c["ok"] for c in g.values()) and all(c["ok"] for c in rg.values())
            B.set_key(lib, B.lib_key(lib, ks[0]))
            bad = floor_gate(lib, torch, B.random_words(torch, 5000, lib.field, 1), 5000, ks[0], [3], B.stream(torch))
            log(f"floor gate {lib.name} k={ks[0]}: " + (f"FAIL at grids {bad}" if bad else "ok"))
            res["ok"] &= not bad and prim_gate(lib, torch, log)["ok"]
        log("sanitize smoke done" if res["ok"] else "sanitize smoke FAILED")
        return 0 if res["ok"] else 1
    if a.ncu:
        lib, (b, ctx) = libs[0], cells_of(a.cells)[0]
        sh, st = shape(lib), B.stream(torch)
        store = B.random_words(torch, b * ctx * sh["tlw"], lib.field, b * 7919 + ctx)
        q = (torch.randn(b, sh["heads"], K.HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16).contiguous()
        o = torch.empty(b, sh["heads"], K.HEAD_DIM, dtype=torch.float32, device="cuda")
        part = part_buffer(torch, sh, b)
        nvec = receipt_vectors(sh, rc_shape(lib), b, ctx)
        sk, tg = (torch.empty(nvec, w, dtype=torch.int32, device="cuda") for w in (16, 8))
        rc_setup(lib, RC_SEED)
        for fm in forms_of(sh, only):
            k = 1 if fm in RTOL else ks[0]
            B.set_key(lib, B.lib_key(lib, k))
            if fm in RECEIPT:
                lib.call("kv_rc_buffers", B.ptr(sk), B.ptr(tg), None)
            attn_call(lib, st, store, fm, q, o, part, b, ctx, 1, 0, a.ncu_splits, k)
            torch.cuda.synchronize()
            log(f"ncu {lib.name} B={b} ctx={ctx} {FORMS[fm]} k={k} at {a.ncu_splits} splits launched")
        return 0
    for lib in libs:
        occ = attn_occ(lib)
        log(f"occupancy {lib.name}: " + ", ".join(f"{n} {v['ctas_per_sm']} CTAs an SM, {v['decoder_warps_per_sm']} "
                                                  "decoder warps" for n, v in occ.items()))
        pg = prim_gate(lib, torch, log)
        g = B.gate(lib, torch, ks, 48, 1 << 16, log)
        ag = attn_gate(lib, torch, m, cells_of(a.gate_cells), ks, log)
        rg = receipt_gate(lib, torch, m, cells_of(a.gate_cells), ks, log)
        res["libs"][lib.name] = {"occupancy": occ, "prim_gate": pg, "gate": g, "attn_gate": ag, "receipt_gate": rg}
        res["ok"] &= pg["ok"] and all(v["ok"] for v in g.values()) and all(c["ok"] for c in ag.values()) and all(
            c["ok"] for c in rg.values())
    if not res["ok"]:
        log("a gate failed: nothing timed")
    else:
        rbuf = torch.empty(4 << 30, dtype=torch.uint8, device="cuda")
        rbuf.random_(0, 256)
        res["read"] = B.read_rate(libs[0], torch, rbuf, sms, a.reps)
        log(f"read: {res['read']['gbps']:.0f} GB/s over 4 GiB")
        del rbuf
        torch.cuda.empty_cache()
        for lib in libs:
            lr = res["libs"][lib.name]
            lr["rates"] = B.rates(lib, torch, ks, sms, a.reps, log)
            lr["exhaust"] = {}
            for b, ctx in [] if a.write_only else cells_of(a.cells):
                need = b * ctx * m["layers"] * shape(lib)["tle"]
                if need + (6 << 30) > props.total_memory:
                    log(f"exhaust {lib.name} B={b} ctx={ctx}: {need / 1e9:.1f} GB doesn't fit; skipped")
                    continue
                cell = lr["exhaust"][f"b{b}_ctx{ctx}"] = exhaust_cell(lib, torch, m, b, ctx, ks, a.reps, sms, log,
                                                                      only)
                res["ok"] &= cell["ok"]
            wb = [int(v) for v in a.write_batches.split(",") if v] or sorted({b for b, _ in cells_of(a.cells)})
            if wb:
                lr["write"] = write_side(lib, torch, m, wb, ks, a.reps, sms, lr["rates"]["encode"], log)
            cells = cells_of(a.cells) if a.write_only else [(c["batch"], c["context"]) for c in lr["exhaust"].values()]
            lr["model_check"] = model_check(res, lib.name, m, cells, shape(lib)["tle"])
            for row in lr["model_check"]["rows"]:
                log(f"model {lib.name} B={row['batch']} ctx={row['context']} k={row['k']}: f_KV k x1/ctx "
                    f"{row['predicted']:.4f} against the encode alone {row['measured_alone']:.4f} and exposed beside "
                    f"the GEMM {row['measured_exposed']:.4f}, {row['measured_exposed_carved']:.4f} with SMs carved out "
                    "(reads of the step's bytes)")
    (out_dir / "kv-exhaust.json").write_text(json.dumps(res, indent=1, default=str))
    write_result(res, libs, out_dir)
    log("done" if res["ok"] else "FAILED")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
