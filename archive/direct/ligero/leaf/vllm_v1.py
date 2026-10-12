"""``leaf = vllm-v1``: the operand rows hashed **in-circuit** as ``vllm-v1`` position leaves (lane b-ligero-vllm-v1):
``pos_leaf(row) = SHA-256("verity/pos-leaf/v0" || u64be(len row) || row_bytes)`` (``verity.ml.commitments.vllm_v1``, PROTOCOL.md
§2 and §8.1).  The digest the relation publishes IS the ``vllm-v1`` tree leaf; the verifier folds it into the port's
``vllm-v1`` tree and checks the port's step-root binding natively (``vllm_tree.py``).

**Framing.**  The 26-byte prefix is not block aligned, so no block is public and every compression of the leaf is proved:
``ceil((26 + n + 9) / 64)`` of them, 25 for a 1536-byte E4M3 row, 49 for a 3072-byte BF16 row.  The last block holds the
row's final 26 bytes, ``0x80``, zeros and the bit length ``8 (26 + n)``: private bytes, so unlike the frame-v3 row leaf
(``leaf/sha256.py``) the padding block is in the circuit too, and the published value is the finished digest.

**Columns.**  A column carries ``c`` = 128 (E4M3 x4) or 256 (BF16 x4) row bytes per operand, a multiple of 64, so the
message stream is ``hold (26) || own (c)`` in every column: ``c / 64`` data blocks compress from the carried chaining value,
and the column's last 26 bytes are carried to the next column as the new hold.  At a chain start the carried value 0 means
``H = IV`` and ``hold = prefix`` (the carry stores ``H - IV`` and ``hold - prefix``).  The final block is compressed in every
column (the chain compiler is column-uniform): only the last column's output is pinned, against the published digest; in
other columns it is a compression of honest but meaningless bytes.  Per operand per column: ``c / 64 + 1`` compressions
(3 for E4M3 x4, 5 for BF16 x4) against the frame-v3 SHA-256 leaf's ``c / 64``.

The gadget is ``leaf/sha256.py``'s (``compress_gadget``: one row per 2- / 3-input XOR bit, one per Ch bit, 16-bit-limb sums;
18,128 rows per compression).  Message bits are the operand's committed bit rows; the hold is re-derived into bits once per
column (208 rows), the chaining value likewise (256 rows).  Soundness of the arithmetic is ``leaf/sha256.py``'s.

Unsalted, deterministic: a low-entropy row is preimage-searchable from its public digest (as every scheme here).
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from verity.ml.commitments import vllm_v1 as core

from ..compile import Expr, LigeroCtx, P
from .blake3 import _bits_raw, _const, _prod, row_bytes
from .sha256 import (BLOCK_LEN, IV, K_VU, _W32, _byte_bits, _limb_expr, _limbs, _message_words, _operand_bits,
                     compress_gadget, compress_np)

__all__ = ["CARRY_ELEMS", "DIGEST_ELEMS", "PREFIX_LEN", "VllmV1Leaf", "census", "chain_states", "digest_of_limbs",
           "leaf_bytes", "message", "prefix"]

NAME = "vllm-v1"
ASSUMPTION = ("SHA-256 collision resistance (vllm-v1 position leaves, every compression in-circuit including the padding "
              "block; vllm-v1 node/lift trees and step-root bindings checked by the verifier)")
SCHEMA = "vllm-v1/pos-leaf/v0"
HASH_NAME = "sha256"
TREE = "vllm-v1"
PREFIX_LEN = len(core.POS_LEAF_TAG) + 8              # 26
HOLD_LIMBS = PREFIX_LEN // 2                          # 13
DIGEST_ELEMS = 16                                     # H_0 .. H_7 as 16-bit limbs, low first
CARRY_ELEMS = 16 + HOLD_LIMBS                         # [H - IV | hold - prefix]
MALFORMED_TAG = b"verity/vllm-v1-leaf/malformed-digest/v1\0"
ROLE_NAMES = {1: "x-row", 2: "w-col"}   # rowleaf.ROLE_X / ROLE_W: row names only, the leaf does not depend on the role
_IV_LIMBS = [v for x in IV for v in (x & 0xFFFF, x >> 16)]


def prefix(n_bytes: int) -> bytes:
    """``"verity/pos-leaf/v0" || u64be(n_bytes)``: the public 26 bytes before the row (``vllm_v1.pos_leaf_layout``)."""
    return core.pos_leaf_layout(n_bytes).prefix


def message(row: bytes) -> bytes:
    """The padded SHA-256 message of ``pos_leaf(row)``: prefix, row, ``0x80``, zeros, u64be bit length."""
    m = prefix(len(row)) + bytes(row)
    return m + b"\x80" + b"\0" * ((55 - len(m)) % 64) + (8 * len(m)).to_bytes(8, "big")


def params_sha256() -> str:
    return hashlib.sha256(b"vllm-v1-leaf|schema=" + SCHEMA.encode() + b"|leaf=pos_leaf(row_bytes)|prefix=" + core.POS_LEAF_TAG +
                          b"||u64be(len)|framing=digest-le16-limbs|padding=circuit|tree=vllm-v1 node/lift|root=step-root"
                          b"|malformed=" + MALFORMED_TAG).hexdigest()


def _prefix_limbs(n_bytes: int) -> list[int]:
    p = prefix(n_bytes)
    return [p[2 * q] | (p[2 * q + 1] << 8) for q in range(HOLD_LIMBS)]


def chain_states(rows: np.ndarray, word_bits: int) -> np.ndarray:
    """(R, n_blocks + 1, 8) uint32: IV, then the chaining value after every block of every row's padded message (the last
    is the ``pos_leaf`` digest as big-endian words)."""
    rb = row_bytes(rows, word_bits)
    R, n = rb.shape
    head = np.frombuffer(prefix(n), dtype=np.uint8)
    tail = np.frombuffer(message(b"\0" * n)[PREFIX_LEN + n:], dtype=np.uint8)
    msg = np.concatenate([np.broadcast_to(head, (R, PREFIX_LEN)), rb, np.broadcast_to(tail, (R, tail.size))], 1)
    blocks = np.ascontiguousarray(msg).view(">u4").reshape(R, -1, 16).astype(np.uint32)
    nb = blocks.shape[1]
    out = np.empty((R, nb + 1, 8), dtype=np.uint32)
    h = np.broadcast_to(np.asarray(IV, dtype=np.uint32)[:, None], (8, R)).copy()
    out[:, 0] = h.T
    for j in range(nb):
        h = compress_np(h, blocks[:, j, :].T)
        out[:, j + 1] = h.T
    return out


def digest_of_limbs(e) -> bytes:
    """16 limbs < 2^16 (word-major, low first) -> the 32-byte big-endian digest; ``ValueError`` otherwise."""
    e = np.asarray(e, dtype=np.int64).reshape(-1)
    if e.shape[0] != DIGEST_ELEMS or ((e < 0) | (e >= 1 << 16)).any():
        raise ValueError(f"a vllm-v1 leaf digest is {DIGEST_ELEMS} limbs < 2^16")
    return np.asarray(e[0::2] | (e[1::2] << 16), dtype=">u4").tobytes()


def leaf_bytes(digest_row) -> bytes:
    """The ``vllm-v1`` tree leaf of one published digest row: its 32 bytes.  Total on field-element rows of the right length:
    a row that is not 16 limbs < 2^16 (the gadget never pins one) maps to the tagged SHA-256 of its elements, which no
    position leaf equals except by collision.  Elements outside the field or a wrong length raise ``ValueError``."""
    e = np.asarray(digest_row, dtype=np.int64).reshape(-1)
    if e.shape[0] != DIGEST_ELEMS:
        raise ValueError(f"a vllm-v1 leaf digest is {DIGEST_ELEMS} elements")
    if ((e < 0) | (e >= P)).any():
        raise ValueError("digest elements are BabyBear field elements")
    try:
        return digest_of_limbs(e)
    except ValueError:
        return hashlib.sha256(MALFORMED_TAG + e.astype("<u4").tobytes()).digest()


def leaf_bytes_many(digests) -> list[bytes]:
    e = np.asarray(digests, dtype=np.int64)
    if e.ndim != 2 or e.shape[1] != DIGEST_ELEMS:
        raise ValueError(f"a vllm-v1 leaf digest is {DIGEST_ELEMS} elements")
    if e.shape[0] and not ((e < 0) | (e >= 1 << 16)).any():
        raw = np.ascontiguousarray((e[:, 0::2] | (e[:, 1::2] << 16)).astype(">u4")).view(np.uint8).reshape(-1, 32)
        return [bytes(r) for r in raw]
    return [leaf_bytes(r) for r in e]


def _column_shape(k: int, word_bits: int, row_words: int) -> tuple[int, int, int]:
    col_bytes = k * word_bits // 8
    n_bytes = row_words * word_bits // 8
    if not col_bytes or col_bytes % BLOCK_LEN or n_bytes % col_bytes:
        raise ValueError(f"vllm-v1 leaf: a column carries whole 64-byte blocks per operand and tiles the row, not {col_bytes} "
                         f"of {n_bytes} bytes")
    return col_bytes, n_bytes, col_bytes // BLOCK_LEN


_CHAIN_KERNEL = r"""
typedef unsigned char u8;

// row i (n bytes at rows + n i): the chaining value before every block of pos_leaf's padded message and after the last,
// as out[i][b][0..8); the message is pre (26 bytes) || row || 0x80 || zeros || u64be(8 (26 + n))
extern "C" __global__
void vllm_pos_chain(const u8* __restrict__ pre, const u8* __restrict__ rows, long long n, int nblocks, long long R,
                    u32* __restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const long long mlen = 26 + n;
    const u8* row = rows + i * n;
    u32 st[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u};
    u32 w[16];
    u32* o = out + i * (long long)(nblocks + 1) * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = st[j];
    for (int b = 0; b < nblocks; ++b) {
        for (int j = 0; j < 16; ++j) {
            u32 v = 0;
            for (int q = 0; q < 4; ++q) {
                long long p = 64LL * b + 4 * j + q;
                u32 x;
                if (p < 26) x = pre[p];
                else if (p < mlen) x = row[p - 26];
                else if (p == mlen) x = 0x80u;
                else if (p >= 64LL * nblocks - 8) x = (u32)(((unsigned long long)(8 * mlen)) >> (8 * (64LL * nblocks - 1 - p))) & 0xFFu;
                else x = 0;
                v = (v << 8) | x;
            }
            w[j] = v;
        }
        sha256_compress(st, w);
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[8 * (b + 1) + j] = st[j];
    }
}
"""
_chain_fn = None


def chain_states_gpu(rb, n_bytes: int):
    """:func:`chain_states` on the GPU: ``rb`` (R, n_bytes) uint8 cupy -> (R, n_blocks + 1, 8) uint32 cupy."""
    global _chain_fn
    from .. import frame_gpu
    cp = frame_gpu.cp
    if _chain_fn is None:
        src = frame_gpu.fv3.sha256_cuda.SRC.split('extern "C"')[0] + _CHAIN_KERNEL
        _chain_fn = cp.RawModule(code=src, options=("-std=c++17",)).get_function("vllm_pos_chain")
    rb = cp.ascontiguousarray(rb, dtype=cp.uint8)
    R = int(rb.shape[0])
    nb = len(message(b"\0" * n_bytes)) // BLOCK_LEN
    out = cp.empty((R, nb + 1, 8), dtype=cp.uint32)
    if R:
        pre = cp.asarray(np.frombuffer(prefix(n_bytes), dtype=np.uint8))
        _chain_fn((-(-R // 128),), (128,), (pre, rb, np.int64(n_bytes), np.int32(nb), np.int64(R), out))
    return out


class VllmV1Leaf:
    """The ``LeafScheme`` (``leaf/base.py``) of ``vllm-v1`` position leaves, and the scheme's tree rule (:attr:`tree`: the
    committed-object side builds ``vllm-v1`` trees bound by step roots instead of frame-v3 indexed trees, ``hashauth.py``).

    Carried state per operand (:data:`CARRY_ELEMS` rows, 0 at a chain start): ``[H - IV (16 limbs) | hold - prefix (13
    limbs)]``.  Digest (:data:`DIGEST_ELEMS` public pins per operand at the chain end): the ``pos_leaf`` digest's eight
    big-endian words as 16-bit limbs, low first; :meth:`leaf_bytes` is those 32 bytes."""

    name = NAME
    assumption = ASSUMPTION
    digest_elems = DIGEST_ELEMS
    carry_elems = CARRY_ELEMS
    schema = SCHEMA
    hash_name = HASH_NAME
    tree = TREE
    params_sha256 = params_sha256()
    test_only = False
    privacy_note = "deterministic, unsalted: hides only up to preimage search on a row"

    # --- committer / verifier side -----------------------------------------------------------------------------------
    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        """(R, DIGEST_ELEMS) int64: ``pos_leaf(row_bytes)`` as 16-bit limbs.  ``role`` does not enter the leaf (the port's
        tree and layout digest separate x rows from W columns)."""
        rows = np.asarray(rows, dtype=np.int64)
        return _limbs(chain_states(rows, word_bits)[:, -1])

    def leaf_bytes(self, digest_row) -> bytes:
        return leaf_bytes(digest_row)

    def leaf_bytes_many(self, digests) -> list[bytes]:
        return leaf_bytes_many(digests)

    def reference_digest(self, rows: np.ndarray, *, word_bits: int, role: int) -> list[bytes]:
        """``verity.ml.commitments.vllm_v1.pos_leaf`` per row: what :meth:`leaf_bytes` of :meth:`native` must equal."""
        return [core.pos_leaf(bytes(r)) for r in row_bytes(np.asarray(rows, dtype=np.int64), word_bits)]

    # --- the in-circuit hash -----------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int = K_VU,
               carry_in: Sequence[Expr], first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        """One column, one operand: ``hold || own`` -> ``c / 64`` data blocks into the carried chaining value, the last
        26 own bytes carried as the next hold, and the final (padding) block of ``hold_out`` compressed into the digest
        expressions (see the module doc).  ``role`` only names the rows; ``first`` / ``last`` are ignored."""
        if len(carry_in) != CARRY_ELEMS:
            raise ValueError(f"gadget: {len(carry_in)} carried rows, the scheme carries {CARRY_ELEMS}")
        col_bytes, n_bytes, nblk = _column_shape(len(words), word_bits, row_words)
        scope = f"vllm.{ROLE_NAMES.get(role, role)}"
        one = _const(ctx, 1)
        pre = _prefix_limbs(n_bytes)
        own = _byte_bits(_operand_bits(ctx, words, word_bits, scope), word_bits)
        h_in = [carry_in[x] + _IV_LIMBS[x] for x in range(16)]
        h = [_W32(_bits_raw(ctx, h_in[2 * w], 16, f"{scope}.h{w}.lo") + _bits_raw(ctx, h_in[2 * w + 1], 16, f"{scope}.h{w}.hi"))
             for w in range(8)]
        hold = []
        for q in range(HOLD_LIMBS):
            b = _bits_raw(ctx, carry_in[16 + q] + pre[q], 16, f"{scope}.hold{q}")
            hold += [b[:8], b[8:]]
        stream = hold + own
        for t in range(nblk):
            h = compress_gadget(ctx, h, _message_words(stream[BLOCK_LEN * t:BLOCK_LEN * (t + 1)]), f"{scope}.blk{t}")
        rem = stream[col_bytes:]
        out_limbs = [_prod(ctx, h[w].limbs[i], one, scope + ".hout") for w in range(8) for i in range(2)]
        hold_out = [_prod(ctx, _limb_expr(ctx, rem[2 * q], rem[2 * q + 1]), one, scope + ".holdout") for q in range(HOLD_LIMBS)]
        pad = message(b"\0" * n_bytes)[PREFIX_LEN + n_bytes:]
        pad = pad[len(pad) - (BLOCK_LEN - PREFIX_LEN):]
        final = rem + [[_const(ctx, (byte >> i) & 1) for i in range(8)] for byte in pad]
        d = compress_gadget(ctx, h, _message_words(final), f"{scope}.fin")
        digest = [d[w].limbs[i] for w in range(8) for i in range(2)]
        carry_out = [out_limbs[x] - _IV_LIMBS[x] for x in range(16)] + [hold_out[q] - pre[q] for q in range(HOLD_LIMBS)]
        return digest, carry_out

    def carry_values(self, rows: np.ndarray, *, word_bits: int, role: int, steps: int) -> np.ndarray:
        """The honest carried rows entering every column of every row: (R, steps, CARRY_ELEMS) int64 mod p."""
        rows = np.asarray(rows, dtype=np.int64)
        R, row_words = rows.shape
        col_bytes, n_bytes, nblk = _column_shape(row_words // steps, word_bits, row_words)
        if col_bytes * steps != n_bytes:
            raise ValueError("carry_values: steps columns do not tile the row")
        states = chain_states(rows, word_bits)
        car = np.zeros((R, steps, CARRY_ELEMS), dtype=np.int64)
        car[:, :, :16] = (_limbs(states[:, np.arange(steps) * nblk]) - np.asarray(_IV_LIMBS)) % P
        rb = row_bytes(rows, word_bits)
        msg = np.concatenate([np.broadcast_to(np.frombuffer(prefix(n_bytes), dtype=np.uint8), (R, PREFIX_LEN)), rb], 1)
        for c in range(steps):
            held = msg[:, col_bytes * c:col_bytes * c + PREFIX_LEN].astype(np.int64)
            car[:, c, 16:] = (held[:, 0::2] | (held[:, 1::2] << 8)) - np.asarray(_prefix_limbs(n_bytes))
        car[:, :, 16:] %= P
        return car

    def row_sponges(self, words, *, word_bits: int, role: int, row_words: int, n: int, steps: int):
        """The committer's chains of ``n`` whole rows on the GPU: ``words`` (n * steps, k) int64 CUDA torch -> ``(carry (n,
        steps, CARRY_ELEMS), digests (n, DIGEST_ELEMS))`` int64 torch = :meth:`carry_values` / :meth:`native`; ``None`` off
        CUDA or without the GPU committer."""
        from .. import frame_gpu
        if not words.is_cuda or not frame_gpu.available():
            return None
        import torch

        from .sha256 import _limbs_cp, _row_bytes_torch
        cp = frame_gpu.cp
        rows = words.reshape(n, row_words)
        if bool(((rows < 0) | (rows >= 1 << word_bits)).any()):
            raise ValueError("word out of range")
        col_bytes, n_bytes, nblk = _column_shape(row_words // steps, word_bits, row_words)
        with frame_gpu._torch_stream():
            rb = cp.asarray(_row_bytes_torch(rows, word_bits))
            st = chain_states_gpu(rb, n_bytes).astype(cp.int64)
            car = cp.zeros((n, steps, CARRY_ELEMS), dtype=cp.int64)
            car[:, :, :16] = (_limbs_cp(st[:, cp.arange(steps) * nblk]) - cp.asarray(_IV_LIMBS)) % P
            pre = cp.asarray(np.frombuffer(prefix(n_bytes), dtype=np.uint8))
            msg = cp.concatenate([cp.broadcast_to(pre, (n, PREFIX_LEN)), rb], 1)
            held = cp.stack([msg[:, col_bytes * c:col_bytes * c + PREFIX_LEN] for c in range(steps)], 1).astype(cp.int64)
            car[:, :, 16:] = ((held[:, :, 0::2] | (held[:, :, 1::2] << 8)) - cp.asarray(_prefix_limbs(n_bytes))) % P
            dg = _limbs_cp(st[:, -1])
            return torch.as_tensor(car, device=words.device), torch.as_tensor(dg, device=words.device)

    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """``words`` (l, k) int64 torch (VU v in columns [v steps, (v + 1) steps), pad columns 0) -> (carry (CARRY_ELEMS,
        l), digests (n_vus, DIGEST_ELEMS)) torch."""
        import torch
        w = words.detach().cpu().numpy().astype(np.int64)
        k = w.shape[1]
        if steps * k != row_words:
            raise ValueError("chain_witness: steps x k != row_words")
        rows = w[:n_vus * steps].reshape(n_vus, row_words)
        car = self.carry_values(rows, word_bits=word_bits, role=role, steps=steps)
        carry = np.zeros((CARRY_ELEMS, l), dtype=np.int64)
        carry[:, :n_vus * steps] = car.reshape(n_vus * steps, CARRY_ELEMS).T
        dg = self.native(rows, word_bits=word_bits, role=role)
        dev = words.device
        return torch.as_tensor(carry, device=dev), torch.as_tensor(dg, device=dev)

    def witness(self):
        return None

    def census(self) -> dict:
        from .sha256 import ROWS_PER_BLOCK
        return {"sha256_rows_per_block": ROWS_PER_BLOCK, "vllm_v1_framing": "pos_leaf, every block in-circuit (padding block "
                "per column, pinned at the chain end), digest published", "vllm_v1_schema": SCHEMA}


try:                                                          # the leaf registry (lane leaf-iface)
    from . import registry as _registry
    _registry.register(VllmV1Leaf)
except ImportError:                                           # pragma: no cover
    pass


def census(word_bits: int = 8, rel_name: str | None = None) -> dict:
    """Rows per column / VU of the composed relation (default the x4 fold of the word width's Ada / Hopper relation)."""
    from .. import hashchain
    from ..relations import relation
    rel = relation(rel_name or {8: "fp8-ada-x4", 16: "bf16-hopper-x4"}[word_bits])
    hr = hashchain.compose(rel, NAME)
    return {"word_bits": word_bits, "relation": rel.name, "steps": rel.steps, "rows_per_column": hr.census["rows_total"],
            "rows_base": hr.census["rows_base"], "rows_hash_per_column": hr.census["rows_hash"],
            "rows_per_vu": hr.census["rows_total"] * rel.steps}


if __name__ == "__main__":
    import json
    import sys as _sys
    print(json.dumps(census(int(_sys.argv[1]) if len(_sys.argv) > 1 else 8, _sys.argv[2] if len(_sys.argv) > 2 else None), indent=1))
