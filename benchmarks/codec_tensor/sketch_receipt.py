"""sketch-sha512 on the device: how the weight GEMM's sketch warps (wt.cu) and attention's form 13 (kv.cu) lay out the
remote bandwidth driver's signs over the store's chunks. The receipt, its signs, σ and y, is
`receipts.SketchReceipter`'s: `sign_bits` and `signs` are its signs as arrays, and `sketches` and `receipts_of` its σ
and its y over many chunks.

~~~text
E_j      the store's chunk j, its 4096 bytes 4096 j … in store order, whole codec words (the registration's layout,
         store-order/v0). A weight GEMM's store (wt.cu's fused decode GEMM, SB = 8192 bytes a task at p1024 and p512)
         is two chunks a task, task q's halves chunks 2 q and 2 q + 1 (`gemm_blocks`), so a launch receipts its own
         store's chunks j0 + v, whatever its groups and grid: the sketch warps (receipt_warp) take a chunk a B column
         of mma.m16n8k32, each byte against its own position's sign (`fragment_sketch`). The KV's chunk is a
         token-layer (kv.cu at 128-byte words: 32 words, 4 KiB, whole), and attention (kv_attn form 13) sketches it
         a KV head's 4 words at a time, a token a B column, A the signs at those words' positions
         (`kv_fragment_sketch`), the 8 heads' shares summed before the hash
~~~
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.core.protocols.remote_bandwidth import receipts as rb

ROWS, ROWS64 = rb.ROWS                    # k_s: the 16-row kernels' and the 64-row ones'
BLOCK = rb.BLOCK_BYTES
LANES = 32
SK_G = 4                                  # receipt_warp's tasks a group: 2 SK_G chunks, mma.m16n8k32's B columns
SIGN_WORDS = ROWS * BLOCK // 32           # a WT_RECEIPT pass's signs (`sign_words`)
FRAG_WORDS = 2 * SIGN_WORDS               # receipt_warp's selectors (`sign_frags`)
PLANE_WORDS = ROWS64 * BLOCK // 32        # receipt_warp's bit planes at 64 rows (`sign_planes`)
PLANE_MASK = 0x01010101
KV_HEADS, KV_WORD = 8, 128                # kv.cu form 13: a token-layer's KV heads and its codec words' bytes
KV_STEPS = KV_WORD // 8                   # its k-steps a tile: 8 bytes of each of a head's 4 words


def sign_bits(seed: bytes, rows: int = ROWS) -> np.ndarray:
    """`receipts.sign_bits` as rows × 512 bytes: bit u of row c (byte u // 8, LSB first) is 1 where S_i[c][u] = −1."""
    return np.array([np.frombuffer(r, np.uint8) for r in rb.sign_bits(seed, rows)])


def signs(seed: bytes, rows: int = ROWS) -> np.ndarray:
    """S_i, rows × 4096 int8 of ±1."""
    bits = np.unpackbits(sign_bits(seed, rows), axis=1, bitorder="little")
    return (1 - 2 * bits.astype(np.int8)).astype(np.int8)


def sketches(seed: bytes, blocks: np.ndarray, rows: int = ROWS) -> np.ndarray:
    """σ of already-masked chunks (n × 4096 bytes) under pass `seed`'s signs at k_s = `rows`, n × k_s int32:
    `receipts.SketchReceipter.sketch` of each."""
    r = rb.SketchReceipter(seed, rows)
    e = np.ascontiguousarray(blocks).view(np.uint8).reshape(-1, BLOCK)
    return np.frombuffer(b"".join(r.sketch(x.tobytes()) for x in e), "<i4").reshape(-1, rows)


def sign_words(bits: np.ndarray) -> np.ndarray:
    """S's bits (`sign_bits`) as a WT_RECEIPT pass writes them (wt_pass's signs): k_s 128 u32 (SIGN_WORDS at 16 rows),
    row c's 512 bytes little-endian from word 128 c."""
    b = np.asarray(bits, np.uint8)
    if b.shape not in ((ROWS, BLOCK // 8), (ROWS64, BLOCK // 8)):
        raise ValueError(f"S's bits are {ROWS} or {ROWS64} × {BLOCK // 8} bytes")
    return np.ascontiguousarray(b).reshape(-1).view("<u4").astype(np.uint32)


def sign_frags(bits: np.ndarray) -> np.ndarray:
    """receipt_warp's A selectors (wt_sign_frags): FRAG_WORDS u32, word 4 (32 pp + lane) + 2 x + w lane (g, t)'s
    selector word w at p = 2 pp + x. A register r of k-step 2 p + s, R = 4 s + r, is one prmt of word R // 6 (>> 16
    when (R mod 6) // 3 = 1) between `sketching.PRMT[R mod 3]`: its byte b is −1 (0xFF) where the selector's bit
    16 ((R mod 6) // 3) + 4 b + R mod 3 is set, S[g + 8 (r mod 2)]'s bit at position 64 p + 16 t + 8 s + 4 (r // 2) + b,
    and +1 where it is clear. A nibble's bit 3 is 0."""
    b = np.asarray(bits, np.uint8)
    if b.shape != (ROWS, BLOCK // 8):
        raise ValueError(f"S's bits are {ROWS} × {BLOCK // 8} bytes")
    u = np.unpackbits(b, axis=1, bitorder="little").astype(np.uint32)
    pp, lane = np.meshgrid(np.arange(BLOCK // 128), np.arange(LANES), indexing="ij")
    g, t = lane >> 2, lane & 3
    out = np.zeros((BLOCK // 128, LANES, 4), np.uint32)
    for x in range(2):
        for big_r in range(8):
            s, r = divmod(big_r, 4)
            for q in range(4):
                pos = 64 * (2 * pp + x) + 16 * t + 8 * s + 4 * (r >> 1) + q
                out[..., 2 * x + big_r // 6] |= u[g + 8 * (r & 1), pos] << np.uint32(16 * ((big_r % 6) // 3) + 4 * q
                                                                                     + big_r % 3)
    return out.reshape(-1)


def sign_planes(bits: np.ndarray) -> np.ndarray:
    """receipt_warp's A at 64 rows (wt_sign_frags): PLANE_WORDS u32, word 4 (32 p + lane) + 2 s + hh lane (g, t)'s
    plane of k-step 2 p + s, half hh. Its bit 8 b + ρ (ρ = 2 mt + rh) is S's bit (1 where −1) of row 16 mt + g + 8 rh
    at position 64 p + 16 t + 8 s + 4 hh + b, so m-tile mt's A registers are one AND each with PLANE_MASK << ρ: bytes
    2^ρ where the bit is set (−128 at ρ = 7), 0 where it is clear."""
    b = np.asarray(bits, np.uint8)
    if b.shape != (ROWS64, BLOCK // 8):
        raise ValueError(f"S's bits are {ROWS64} × {BLOCK // 8} bytes")
    u = np.unpackbits(b, axis=1, bitorder="little").astype(np.uint32)
    p, lane = np.meshgrid(np.arange(BLOCK // 64), np.arange(LANES), indexing="ij")
    g, t = lane >> 2, lane & 3
    out = np.zeros((BLOCK // 64, LANES, 4), np.uint32)
    for w in range(4):
        s, hh = divmod(w, 2)
        for q in range(4):
            pos = 64 * p + 16 * t + 8 * s + 4 * hh + q
            for rho in range(8):
                out[..., w] |= u[16 * (rho >> 1) + g + 8 * (rho & 1), pos] << np.uint32(8 * q + rho)
    return out.reshape(-1)


def _tasks(nbs: int, nchunks: int, groups: int):
    """Each format CTA's tasks as the fused GEMM runs them: CTA (g0, kc)'s task i is the store's
    q = (g0 + i groups) nchunks + kc."""
    if not 1 <= groups <= nbs:
        raise ValueError("a launch has 1 to nbs groups")
    for g0 in range(groups):
        for kc in range(nchunks):
            yield [(g0 + i * groups) * nchunks + kc for i in range(len(range(g0, nbs, groups)))]


def gemm_receipts(nbs: int, nchunks: int, sb: int, groups: int) -> np.ndarray:
    """The receipt indices v (receipt_warp's tag_base + v) of a launch's chunks in the order its CTAs take them: per
    CTA, group a of SK_G tasks, chunk n its task SK_G a + n // 2's half n mod 2, v = 2 q + n mod 2."""
    if sb != 2 * BLOCK:
        raise ValueError(f"a task is two {BLOCK}-byte blocks at SB = {2 * BLOCK}, not SB = {sb}")
    return np.array([2 * q + h for qs in _tasks(nbs, nchunks, groups) for q in qs for h in range(2)], np.int64)


def gemm_vectors(nbs: int, nchunks: int, sb: int, groups: int) -> np.ndarray:
    """A launch's map from its receipts' bytes to the store's (`receipts.plan_of`'s `vectors`): row v the store byte
    at each of chunk v's 4096 positions, from receipt_warp's loads, lane (n, t)'s uint4 at 128 pp + 64 e + 16 t of task
    q's half h the positions 64 (2 pp + e) + 16 t …; −1 where no chunk of the launch is receipt v."""
    out = np.full((nbs * nchunks * sb // BLOCK, BLOCK), -1, np.int64)
    pp, e, t, c = np.meshgrid(np.arange(BLOCK // 128), np.arange(2), np.arange(4), np.arange(16), indexing="ij")
    load = (128 * pp + 64 * e + 16 * t + c).reshape(-1)
    pos = (64 * (2 * pp + e) + 16 * t + c).reshape(-1)
    for v in gemm_receipts(nbs, nchunks, sb, groups):
        q, h = divmod(int(v), 2)
        out[v, pos] = q * sb + BLOCK * h + load
    return out


def gemm_blocks(store: np.ndarray, sb: int) -> np.ndarray:
    """A weight GEMM's store (`wt.cu`: SB = sb stored bytes a task) as its chunks, (chunks × 4096) uint8, chunk v the
    launch's receipt v (`gemm_vectors`)."""
    if sb % BLOCK:
        raise ValueError(f"a task of {sb} bytes is not whole {BLOCK}-byte blocks")
    return np.asarray(store, np.uint8).reshape(-1, BLOCK)


def _int8(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, np.int64) & 0xFF) ^ 0x80) - 0x80


def fragment_sketch(store: np.ndarray, nbs: int, nchunks: int, sb: int, groups: int, frags: np.ndarray) -> np.ndarray:
    """σ of a launch's chunks as receipt_warp computes them, row v receipt v: per CTA, group of SK_G tasks, and k-step
    (p, s), A (16 × 32) from each lane's prmt registers of its selectors (`sign_frags`), B (32 × 8) from each lane's
    uint4 of chunk g at 128 pp + 64 e + 16 t (x, y b0 and b1 of k-step 2 p, z, w of 2 p + 1, p = 2 pp + e), one
    mma.m16n8k32 (D[row][n] += Σ_k A[row][k] B[k][n]) at a time."""
    import sketching

    st = np.asarray(store, np.uint8).reshape(-1)
    fr = np.asarray(frags, np.uint32).reshape(BLOCK // 128, LANES, 4)
    a = np.zeros((BLOCK // 64, 2, ROWS, 32), np.int64)
    for pp in range(BLOCK // 128):
        for lane in range(LANES):
            g, t = divmod(lane, 4)
            for e in range(2):
                w0, w1 = int(fr[pp, lane, 2 * e]), int(fr[pp, lane, 2 * e + 1])
                sels = (w0, w0, w0, w0 >> 16, w0 >> 16, w0 >> 16, w1, w1)
                for big_r in range(8):
                    s, r = divmod(big_r, 4)
                    reg = sketching.prmt(*sketching.PRMT[big_r % 3], sels[big_r] & 0xFFFF)
                    regb = _int8([reg >> 8 * q for q in range(4)])
                    a[2 * pp + e, s, g + 8 * (r & 1), 16 * (r >> 1) + 4 * t:16 * (r >> 1) + 4 * t + 4] = regb
    return _group_sketches(st, nbs, nchunks, sb, groups, ROWS, lambda b: np.einsum("psrk,pskn->rn", a, b))


def _group_sketches(st, nbs, nchunks, sb, groups, rows, mma) -> np.ndarray:
    """Each sketch warp group's 2 SK_G chunks as B (BLOCK // 64 p, 2 s, 32 k, 2 SK_G n): lane (n, t)'s uint4 of chunk
    n at 64 p + 16 t, b0 of k-step 2 p + s its bytes 8 s …, b1 8 s + 4 …; `mma(b)` the rows × 2 SK_G sketches."""
    out = np.full((nbs * nchunks * sb // BLOCK, rows), np.iinfo(np.int64).min, np.int64)
    for qs in _tasks(nbs, nchunks, groups):
        for i0 in range(0, len(qs), SK_G):
            blk = [(q, h) for q in qs[i0:i0 + SK_G] for h in range(2)]
            b = np.zeros((BLOCK // 64, 2, 32, 2 * SK_G), np.int64)
            for n, (q, h) in enumerate(blk):
                d = st[q * sb + BLOCK * h:q * sb + BLOCK * (h + 1)].view(np.int8).astype(np.int64)
                # 16 bytes a lane: p, t, then (s, b0 or b1, byte); B's k is 16 (b1) + 4 t + byte
                b[..., n] = d.reshape(BLOCK // 64, 4, 2, 2, 4).transpose(0, 2, 3, 1, 4).reshape(BLOCK // 64, 2, 32)
            acc = mma(b)
            for n, (q, h) in enumerate(blk):
                if out[2 * q + h, 0] != np.iinfo(np.int64).min:
                    raise AssertionError(f"receipt {2 * q + h} twice")
                out[2 * q + h] = acc[:, n]
    return out


def _i32(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, np.int64) + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)


def fragment_sketch64(store: np.ndarray, nbs: int, nchunks: int, sb: int, groups: int, planes: np.ndarray) -> np.ndarray:
    """σ at 64 rows as receipt_warp computes it, row v receipt v. B as `fragment_sketch`'s; A, m-tile mt of k-step
    2 p + s, from lane (g, t)'s planes (`sign_planes`) W_hh of (p, s): a0 W_0 & PLANE_MASK << 2 mt, a1 the same at
    2 mt + 1, a2 and a3 W_1's, so m-tile mt's int32 accumulator holds row 16 mt + r's 2^ρ Σ_u bit_u e_u (−2^7 at
    ρ = 7, ρ = 2 mt + r // 8), and a fifth MMA of A all ones T = Σ_u e_u. Then σ = T − 2 (acc >> ρ), or
    T + 2 (acc >> 7) at ρ = 7: S = 1 − 2 bit. |acc| ≤ 2^26, so no int32 wraps."""
    st = np.asarray(store, np.uint8).reshape(-1)
    pl = np.asarray(planes, np.uint32).reshape(BLOCK // 64, LANES, 2, 2).astype(np.int64)
    a = np.zeros((BLOCK // 64, 2, ROWS64 // 16, 16, 32), np.int64)
    for lane in range(LANES):
        g, t = divmod(lane, 4)
        for mt in range(ROWS64 // 16):
            for rh in range(2):
                rho = 2 * mt + rh
                for hh in range(2):
                    reg = pl[:, lane, :, hh] & (PLANE_MASK << rho)
                    a[:, :, mt, g + 8 * rh, 16 * hh + 4 * t:16 * hh + 4 * t + 4] = _int8(
                        reg[..., None] >> (8 * np.arange(4)))
    rho = (2 * np.arange(ROWS64 // 16)[:, None] + (np.arange(16) >= 8)[None, :]).reshape(-1, 1)

    def mma(b):
        acc = _i32(np.einsum("psmrk,pskn->mrn", a, b).reshape(ROWS64, -1))
        tot = _i32(np.einsum("pskn->n", b))[None, :]
        return _i32(np.where(rho < 7, tot - 2 * (acc >> rho), tot + 2 * (acc >> 7)))

    return _group_sketches(st, nbs, nchunks, sb, groups, ROWS64, mma)


def kv_head_words(g: int) -> list[int]:
    """KV head g's words of a token-layer at 128-byte words, its K words then its V words (kv.cu's decoder lane j's
    wi)."""
    half = BLOCK // 2 // KV_WORD
    return [2 * g, 2 * g + 1, half + 2 * g, half + 2 * g + 1]


def kv_positions(g: int) -> np.ndarray:
    """The 512 positions of a token-layer's chunk that head g's words hold."""
    return np.concatenate([np.arange(KV_WORD * w, KV_WORD * (w + 1)) for w in kv_head_words(g)])


def kv_sign_frags(bits: np.ndarray) -> np.ndarray:
    """kv_rc_pass's A table (kv.cu's kRcA, rc_a_kernel) from S's bits (`sign_bits`): (KV_HEADS, KV_STEPS, 32, 4) u32,
    lane (g', t') of head g at k-step κ the signs of rows g' and g' + 8 at positions 128 wi + 8κ … + 3 (registers 0 and
    1) and + 4 … + 7 (2 and 3), wi = kv_head_words(g)[t'], as bytes 0x01 (+1) and 0xFF (−1), LSB first."""
    b = np.asarray(bits, np.uint8)
    if b.shape != (ROWS, BLOCK // 8):
        raise ValueError(f"S's bits are {ROWS} × {BLOCK // 8} bytes")
    u = np.unpackbits(b, axis=1, bitorder="little").astype(np.uint32)
    g, kk, lane, q = np.meshgrid(np.arange(KV_HEADS), np.arange(KV_STEPS), np.arange(LANES), np.arange(4),
                                 indexing="ij")
    wi = np.array([kv_head_words(h) for h in range(KV_HEADS)])[g, lane & 3]
    pos = KV_WORD * wi + 8 * kk + 4 * (q >> 1)
    row = (lane >> 2) + 8 * (q & 1)
    out = np.zeros(g.shape, np.uint32)
    for c in range(4):
        out |= np.where(u[row, pos + c] == 1, 0xFF, 0x01).astype(np.uint32) << np.uint32(8 * c)
    return out


def kv_fragment_sketch(blocks: np.ndarray, frags: np.ndarray) -> np.ndarray:
    """σ of token-layers (n × 4096 bytes) as kv_attn form 13 computes it, n × 16: per head g and k-step κ one
    mma.m16n8k32 of A (16 × 32, each lane's registers of `kv_sign_frags`) by B (32 × n, lane (g', t')'s b0 and b1 the
    bytes 8κ … + 3 and 8κ + 4 … + 7 of word kv_head_words(g)[t'] of token g'), the heads' shares summed."""
    blk = np.asarray(blocks, np.uint8).reshape(-1, BLOCK)
    n = blk.shape[0]
    by = _int8(np.asarray(frags, np.uint32).reshape(KV_HEADS, KV_STEPS, LANES, 4, 1).astype(np.int64)
               >> (8 * np.arange(4)))
    a = np.zeros((KV_HEADS, KV_STEPS, ROWS, 32), np.int64)
    b = np.zeros((KV_HEADS, KV_STEPS, 32, n), np.int64)
    words = blk.view(np.int8).astype(np.int64).reshape(n, BLOCK // KV_WORD, KV_STEPS, 2, 4)
    for lane in range(LANES):
        gid, t = divmod(lane, 4)
        for q in range(4):
            a[:, :, gid + 8 * (q & 1), 16 * (q >> 1) + 4 * t:16 * (q >> 1) + 4 * t + 4] = by[:, :, lane, q]
    for g in range(KV_HEADS):
        for t, wi in enumerate(kv_head_words(g)):
            for h in range(2):
                b[g, :, 16 * h + 4 * t:16 * h + 4 * t + 4] = words[:, wi, :, h].transpose(1, 2, 0)
    return np.einsum("gkrc,gkcn->nr", a, b)


def kv_vectors(slots: int, layers: int, w0: int = 0) -> np.ndarray:
    """kv_attn form 13's map from its receipts' bytes to the store's (`receipts.plan_of`'s `vectors`): row
    u layers + l, slot u's token-layer at layer l, whose byte 128 wi + 8κ + 4h + c (the position its sign in A is at)
    meets the store byte w0 + 4096 (u layers + l) + 128 wi + 8κ + 4h + c (kv.cu's decoder lane's raw word wi of head g,
    k-step κ, register h, byte c); the KV region starts at the store's byte w0."""
    g, kk, t, h, c = np.meshgrid(np.arange(KV_HEADS), np.arange(KV_STEPS), np.arange(4), np.arange(2), np.arange(4),
                                 indexing="ij")
    wi = np.array([kv_head_words(x) for x in range(KV_HEADS)])[g, t]
    pos = (KV_WORD * wi + 8 * kk + 4 * h + c).reshape(-1)
    out = np.full((slots * layers, BLOCK), -1, np.int64)
    out[:, pos] = w0 + BLOCK * np.arange(slots * layers)[:, None] + pos[None, :]
    return out


def receipts_of(seed: bytes, sk: np.ndarray, j0: int) -> list[bytes]:
    """y of chunks j0, j0 + 1, … from their sketches (n × k_s integers, a device's): `receipts.SketchReceipter.receipt`
    of each."""
    w = np.asarray(sk).astype("<i4")
    r = rb.SketchReceipter(seed, w.shape[1])
    return [r.receipt(j0 + v, w[v].tobytes()) for v in range(w.shape[0])]
