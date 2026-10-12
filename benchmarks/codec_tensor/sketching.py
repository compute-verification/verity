"""`ro-sketch-blake3/v0` over `wt.cu`'s store as its fused kernel computes it: exact INT8 sketches of the stored
(encoded) bytes, on the sketch warps' tensor cores before the decoders overwrite them, then BLAKE3 over the sketches.

~~~text
store    tasks q = G·chunks + kc (row block G, 512-column chunk kc), SB = TW·S bytes each
CTA b    = g0·chunks + kc of the launch's groups·chunks CTAs; its task i is row block g0 + i·groups at kc
step     512 stored bytes of a task, m = ⌈SB / 512⌉ of them, the last zero-padded; lane l = 4 g + tq holds the step's
         bytes 16 l .. 16 l + 15 as the int8 A operand of mma.m16n8k32 (registers a0 .. a3, byte q of a register)
tile j   T consecutive tasks of a CTA (i = T j .. T j + T − 1, zeros past its last task); TPC = ⌈⌈nbs / groups⌉ / T⌉ a
         CTA
vector   row r of a tile: position 32 (ii·m + s) + kk (task ii of the tile, step s) holds A[r][kk] of that step, the
         byte 4·reg + q of lane 4 (r mod 8) + (kk mod 16) / 4 with reg = 2 ⌊kk / 16⌋ + ⌊r / 8⌋, q = kk mod 4; so
         n = 32 m T. Vector v = 16 (b·TPC + j) + r
S        k × n signs: K_S = derive(σ, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n}), row c the first ⌈n / 8⌉ bytes of
         K_S.stream(("row", c)), bit p (byte p // 8, LSB first) 1 → −1, 0 → +1 (ra-sketch-tag's)
sketch   σ_v = S · e_v over the integers (|σ_v[c]| ≤ 128 n < 2^24), k int32 little-endian words
y_v      BLAKE3-keyed(K, u64be (jbase + v) ‖ σ_v), K = tags.Blake3Tag(name=NAME).key(σ); leaves (16 tags a leaf) and
         the RFC 6962 SHA-512 root as `tagging`
pass j   all of the above at σ_j = pass_seed(σ, j, root_(j−1)) (wt.cu's wt_pass, on the device after pass j − 1's root)
~~~

The partition follows the launch (groups), which the prover states with the window: a tile's sketch stays in its CTA
(its sketch warps' accumulators, summed in shared memory at the tile's end), so no two CTAs add into one vector, and
every stored byte is in exactly one vector.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import tagging as TG

NAME = "ro-sketch-blake3/v0"
SIGN_DOMAIN = "verity/bandwidth/sketch-signs/v0"
PASS_DOMAIN = b"verity/remote-bandwidth/seed/v0\0"
KS = (16, 32, 64)
STEP = 512
ROWS = 16
LANES = 32
WORD_PAD = {1: 1, 2: 2, 3: 4}     # sign words a lane a step, as loaded (u32, uint2, uint4)


def randomness():
    if str(TG.REPO) not in sys.path:
        sys.path.insert(0, str(TG.REPO))
    from verity.ml import randomness

    return randomness


def tag():
    return TG.tags_module().Blake3Tag(name=NAME)


def steps(sb: int) -> int:
    return -(-sb // STEP)


def vector_bytes(sb: int, t: int) -> int:
    return 32 * steps(sb) * t


def tiles_per_cta(nbs: int, groups: int, t: int) -> int:
    per = -(-nbs // groups)
    return -(-per // t)


def pass_seed(sigma: bytes, j: int, root: bytes) -> bytes:
    """Pass j's seed from the window's σ and the previous pass's root (red team F29): its receipt is this module's at
    that seed, so the signs are fresh every pass and exist only once root_(j−1) does. The first 32 bytes of
    `remote_bandwidth.receipts.seed` (ra-remote-driver's chain; a tag's seed is 32 bytes)."""
    if len(sigma) != 32 or len(root) != 64 or not 0 <= j < 1 << 32:
        raise ValueError("σ is 32 bytes, the root 64 and j a u32")
    return hashlib.sha512(PASS_DOMAIN + sigma + j.to_bytes(4, "big") + root).digest()[:32]


def sign_bits(seed: bytes, k: int, n: int) -> np.ndarray:
    """S's bits, k × n uint8 (1 is −1)."""
    if k not in KS or n < 1:
        raise ValueError(f"k is one of {KS} and n positive")
    key = randomness().derive(seed, SIGN_DOMAIN, {"tag": NAME, "k": k, "n": n})
    rows = [np.frombuffer(key.stream(("row", c), -(-n // 8)), dtype=np.uint8) for c in range(k)]
    return np.unpackbits(np.stack(rows), axis=1, bitorder="little")[:, :n]


def task_rows(store: np.ndarray, nbs: int, nchunks: int, sb: int) -> np.ndarray:
    """The store's tasks as their A rows: (row block, chunk, 16, 32 m) int8, position 32 s + kk."""
    m = steps(sb)
    a = np.asarray(store, dtype=np.uint8).reshape(nbs, nchunks, sb)
    a = np.concatenate([a, np.zeros((nbs, nchunks, STEP * m - sb), np.uint8)], axis=2)
    a = a.reshape(nbs, nchunks, m, 8, 4, 2, 2, 4)        # G, kc, s, g, tq, reg // 2, reg % 2, q
    a = a.transpose(0, 1, 6, 3, 2, 5, 4, 7)               # G, kc, row // 8, row % 8, s, kk // 16, tq, q
    return a.reshape(nbs, nchunks, ROWS, 32 * m).view(np.int8)


def vectors(store: np.ndarray, nbs: int, nchunks: int, sb: int, groups: int, t: int) -> np.ndarray:
    """Every vector, (groups·chunks·TPC·16) × n int8, vector 16 (b·TPC + j) + r."""
    rows = task_rows(store, nbs, nchunks, sb)
    tpc, w = tiles_per_cta(nbs, groups, t), rows.shape[3]
    out = np.zeros((groups, nchunks, tpc * ROWS, t * w), np.int8)
    for g0 in range(groups):
        r = rows[g0::groups]
        pad = np.zeros((tpc * t, nchunks, ROWS, w), np.int8)
        pad[:r.shape[0]] = r
        out[g0] = pad.reshape(tpc, t, nchunks, ROWS, w).transpose(2, 0, 3, 1, 4).reshape(nchunks, tpc * ROWS, t * w)
    return out.reshape(-1, t * w)


def vectors_torch(torch, store, nbs: int, nchunks: int, sb: int, groups: int, t: int):
    """`vectors` on a torch uint8 tensor (the device's store), as int8."""
    m = steps(sb)
    a = store.view(nbs, nchunks, sb)
    a = torch.cat([a, torch.zeros(nbs, nchunks, STEP * m - sb, dtype=torch.uint8, device=a.device)], dim=2)
    a = a.view(nbs, nchunks, m, 8, 4, 2, 2, 4).permute(0, 1, 6, 3, 2, 5, 4, 7).reshape(nbs, nchunks, ROWS, 32 * m)
    tpc, w = tiles_per_cta(nbs, groups, t), 32 * m
    out = torch.zeros(groups, nchunks, tpc * ROWS, t * w, dtype=torch.uint8, device=a.device)
    for g0 in range(groups):
        r = a[g0::groups]
        pad = torch.zeros(tpc * t, nchunks, ROWS, w, dtype=torch.uint8, device=a.device)
        pad[:r.shape[0]] = r
        out[g0] = pad.view(tpc, t, nchunks, ROWS, w).permute(2, 0, 3, 1, 4).reshape(nchunks, tpc * ROWS, t * w)
    return out.view(-1, t * w).view(torch.int8)


def sketch_torch(torch, vs, bits: np.ndarray, rows: int = 1 << 16):
    """S · e_v in float32 with TF32 off: exact, since every partial sum is an integer of magnitude at most 128 n < 2^24."""
    if 128 * bits.shape[1] >= 1 << 24:
        raise ValueError("float32 is exact only below n = 2^17")
    s = torch.from_numpy(1.0 - 2.0 * bits.astype(np.float32)).to(vs.device).t().contiguous()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        out = torch.empty(vs.shape[0], bits.shape[0], dtype=torch.int32, device=vs.device)
        for a in range(0, vs.shape[0], rows):
            out[a:a + rows] = (vs[a:a + rows].float() @ s).round().to(torch.int32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return out


def sketch(vs: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """S · e_v for every vector, exactly, through float64 BLAS (every partial sum an integer below 2^53)."""
    s = (1.0 - 2.0 * bits.astype(np.float64)).T
    out = np.empty((vs.shape[0], bits.shape[0]), np.int64)
    for a in range(0, vs.shape[0], 1 << 14):
        out[a:a + (1 << 14)] = np.rint(vs[a:a + (1 << 14)].astype(np.float64) @ s).astype(np.int64)
    return out


def sketch_words(sk: np.ndarray) -> np.ndarray:
    if np.abs(sk).max(initial=0) >= 1 << 31:
        raise ValueError("a sketch word is an int32")
    return sk.astype("<i4")


def sign_words(k: int) -> int:
    return WORD_PAD[-(-2 * (k // 8) // 6)]


def sign_fragments(bits: np.ndarray, m: int, t: int) -> np.ndarray:
    """S packed for the B operand, u32 words at ((ii·m + s)·32 + lane)·WP + w (WP = `sign_words`). Register
    r = 2 blk + h of a step (b0 at h = 0, b1 at h = 1, n-block blk) is word r // 6; its byte q is S[8 blk + g]
    [32 (ii·m + s) + 16 h + 4 tq + q], at bit 16 ⌊(r mod 6) / 3⌋ + 4 q + (r mod 3), and a nibble's bit 3 is 0. So a
    register is one prmt of the word (or the word >> 16) between (lo, hi) = (0xFF01FF01, 0xFF01FF01),
    (0xFFFF0101, 0xFFFF0101), (0x01010101, 0xFFFFFFFF) at e = r mod 3: a nibble's bit e picks 0xFF (−1)."""
    k, n = bits.shape
    kb, wp = k // 8, sign_words(k)
    if n != 32 * m * t:
        raise ValueError("S is k × 32 m T")
    b = bits.reshape(kb, 8, t, m, 2, 4, 4).transpose(2, 3, 1, 5, 0, 4, 6)    # ii, s, g, tq, blk, h, q
    b = b.reshape(t, m, LANES, 2 * kb, 4).astype(np.uint32)
    words = np.zeros((t, m, LANES, wp), np.uint32)
    for r in range(2 * kb):
        for q in range(4):
            words[..., r // 6] |= b[..., r, q] << np.uint32(16 * ((r % 6) // 3) + 4 * q + r % 3)
    return words.reshape(-1)


PRMT = ((0xFF01FF01, 0xFF01FF01), (0xFFFF0101, 0xFFFF0101), (0x01010101, 0xFFFFFFFF))


def prmt(lo: int, hi: int, sel: int) -> int:
    """PTX prmt.b32's default mode, on selector nibbles below 8."""
    src = lo | hi << 32
    return sum(((src >> 8 * ((sel >> 4 * j) & 7)) & 0xFF) << 8 * j for j in range(4))


def fragment_sketch(store: np.ndarray, nbs: int, nchunks: int, sb: int, groups: int, t: int, words: np.ndarray,
                    k: int) -> np.ndarray:
    """The sketch as the MMA warp computes it, from the lanes' A registers and the prmt-expanded sign registers, one
    mma.m16n8k32 (D[row][col] += Σ_kk A[row][kk] B[kk][col]) at a time; small stores only."""
    m, kb, wp = steps(sb), k // 8, sign_words(k)
    st = np.asarray(store, dtype=np.uint8).reshape(nbs, nchunks, sb)
    wd = words.reshape(t, m, LANES, wp)
    signs = np.zeros((t, m, kb, 32, 8), np.int64)                      # B[kk][g] of n-block blk
    for ii in range(t):
        for s in range(m):
            for lane in range(LANES):
                g, tq = divmod(lane, 4)
                for r in range(2 * kb):
                    w = int(wd[ii, s, lane, r // 6]) >> (16 if (r % 6) // 3 else 0)
                    v = prmt(*PRMT[r % 3], w & 0xFFFF)
                    blk, h = divmod(r, 2)
                    for q in range(4):
                        signs[ii, s, blk, 16 * h + 4 * tq + q, g] = 1 if (v >> 8 * q) & 0xFF == 1 else -1
    tpc = tiles_per_cta(nbs, groups, t)
    out = np.zeros((groups * nchunks * tpc, ROWS, k), np.int64)
    for g0 in range(groups):
        for kc in range(nchunks):
            b = g0 * nchunks + kc
            for i, G in enumerate(range(g0, nbs, groups)):
                j, ii = divmod(i, t)
                task = np.concatenate([st[G, kc], np.zeros(STEP * m - sb, np.uint8)]).view(np.int8).astype(np.int64)
                for s in range(m):
                    a = np.zeros((ROWS, 32), np.int64)
                    for lane in range(LANES):
                        g, tq = divmod(lane, 4)
                        for reg in range(4):
                            for q in range(4):
                                a[g + 8 * (reg & 1), 16 * (reg >> 1) + 4 * tq + q] = task[STEP * s + 16 * lane
                                                                                          + 4 * reg + q]
                    for blk in range(kb):
                        out[b * tpc + j, :, 8 * blk:8 * blk + 8] += a @ signs[ii, s, blk]
    return out.reshape(-1, k)


def tags_of(seed: bytes, sk: np.ndarray, jbase: int = 0) -> list[bytes]:
    """`tags.py`'s y_v (pure Python BLAKE3: small sketches only)."""
    keyed, w = tag().keyed(seed), sketch_words(sk)
    return [keyed(jbase + v, w[v].tobytes()) for v in range(w.shape[0])]


def tags_fast(seed: bytes, sk: np.ndarray, jbase: int = 0) -> list[bytes] | None:
    """Every y_v with the PyPI `blake3`, or None without it."""
    try:
        import blake3 as fast
    except ImportError:
        return None
    key, w = tag().key(seed), sketch_words(sk)
    return [fast.blake3((jbase + v).to_bytes(8, "big") + w[v].tobytes(), key=key).digest() for v in range(w.shape[0])]


def key(seed: bytes) -> bytes:
    return tag().key(seed)


def leaf_prefix(seed: bytes) -> bytes:
    T = TG.tags_module()
    return T.prefix(T.LEAF_DOMAIN, NAME, seed)


def leaves(seed: bytes, ys: list[bytes]) -> list[bytes]:
    return TG.tags_module().leaves(tag(), seed, ys)
