"""Batched SHA-256 over many equal-length messages as plain torch int32 tensor ops (EXPERIMENTAL, CMT-1).

Not a new hash kernel: the FIPS 180-4 compression function written as elementwise int32 ops over a lane dimension (one lane per
message), so it runs wherever torch runs (the GPU that holds the tensors, or the CPU) and can be fused by `torch.compile`.
Verified against `hashlib.sha256` (`test_torch_sha256.py`).  Words are int32 two's complement (adds wrap mod 2^32; logical
shifts are masked); byte order handled explicitly (SHA-256 is big-endian, tensors are little-endian).

    digests = sha256_messages(msgs)     # msgs: uint8 [N, L] with L % 64 == 0 after padding -> uint8 [N, 32]
    leaf = pos_leaf_digests(values)     # values: uint8 [N, 64] -> uint8 [N, 32]   (frame verity/pos-leaf/v0 || u64 len || value)
    node = node_digests(left, right)    # uint8 [N, 32] x 2 -> uint8 [N, 32]        (production node frame)
    lift = lift_digests(child)          # uint8 [N, 32] -> uint8 [N, 32]            (production lift frame)
"""
from __future__ import annotations

import hashlib
import struct

import torch

_K = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]
_H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]


def _s32(x: int) -> int:
    """Python int (0..2^32) -> int32 two's complement value."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


_K32 = [_s32(k) for k in _K]
_H032 = [_s32(h) for h in _H0]


def _shr(x: torch.Tensor, n: int) -> torch.Tensor:
    """logical shift right of int32 lanes"""
    return (x >> n) & ((1 << (32 - n)) - 1)


def _rotr(x: torch.Tensor, n: int) -> torch.Tensor:
    return (x << (32 - n)) | _shr(x, n)


def compress(state: list[torch.Tensor], w: list[torch.Tensor]) -> list[torch.Tensor]:
    """One SHA-256 compression of 16 big-endian words `w` (int32 lanes) into the 8-word `state`; pure elementwise ops."""
    a, b, c, d, e, f, g, h = state
    W = list(w)
    for t in range(64):
        if t >= 16:
            w15, w2 = W[(t - 15) % 16], W[(t - 2) % 16]
            s0 = _rotr(w15, 7) ^ _rotr(w15, 18) ^ _shr(w15, 3)
            s1 = _rotr(w2, 17) ^ _rotr(w2, 19) ^ _shr(w2, 10)
            W[t % 16] = W[t % 16] + s0 + W[(t - 7) % 16] + s1
        wt = W[t % 16]
        S1 = _rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)
        ch = (e & f) ^ (~e & g)
        t1 = h + S1 + ch + _K32[t] + wt
        S0 = _rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        t2 = S0 + maj
        h, g, f, e = g, f, e, d + t1
        d, c, b, a = c, b, a, t1 + t2
    return [state[0] + a, state[1] + b, state[2] + c, state[3] + d, state[4] + e, state[5] + f, state[6] + g, state[7] + h]


_compiled_compress = None


def compress_fn(compiled: bool):
    global _compiled_compress
    if not compiled:
        return compress
    if _compiled_compress is None:
        _compiled_compress = torch.compile(compress, dynamic=True, fullgraph=True)
    return _compiled_compress


def _bswap32(x: torch.Tensor) -> torch.Tensor:
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | (_shr(x, 8) & 0xFF00) | _shr(x, 24)


def bytes_to_words_be(msgs: torch.Tensor) -> torch.Tensor:
    """uint8 [N, L] (L % 4 == 0) -> int32 [N, L/4] big-endian words."""
    x = msgs.contiguous().view(torch.int32)
    return _bswap32(x)


def words_be_to_bytes(words: torch.Tensor) -> torch.Tensor:
    """int32 [N, k] big-endian words -> uint8 [N, 4k]."""
    return _bswap32(words.contiguous()).view(torch.uint8)


def pad_messages(msgs: torch.Tensor, msg_len: int) -> torch.Tensor:
    """uint8 [N, msg_len] -> padded uint8 [N, L] (L multiple of 64) per FIPS 180-4 (0x80, zeros, u64be bit length)."""
    n = msgs.shape[0]
    L = ((msg_len + 9 + 63) // 64) * 64
    out = torch.zeros((n, L), dtype=torch.uint8, device=msgs.device)
    out[:, :msg_len] = msgs
    out[:, msg_len] = 0x80
    tail = struct.pack(">Q", msg_len * 8)
    out[:, L - 8:] = torch.tensor(list(tail), dtype=torch.uint8, device=msgs.device)
    return out


def sha256_padded(padded: torch.Tensor, *, compiled: bool = False) -> torch.Tensor:
    """padded uint8 [N, L] (L % 64 == 0) -> digests uint8 [N, 32]."""
    n, L = padded.shape
    assert L % 64 == 0
    if compiled == "triton":                                # one Triton launch per batch (triton_sha256.py); same digests
        from .triton_sha256 import sha256_padded_triton
        return sha256_padded_triton(padded)
    words = bytes_to_words_be(padded)                       # [N, L/4]
    dev = padded.device
    state = [torch.full((n,), h, dtype=torch.int32, device=dev) for h in _H032]
    fn = compress_fn(compiled)
    for blk in range(L // 64):
        w = [words[:, blk * 16 + i].contiguous() for i in range(16)]
        state = fn(state, w)
    return words_be_to_bytes(torch.stack(state, dim=1))    # [N, 8] -> [N, 32]


def sha256_messages(msgs: torch.Tensor, *, compiled: bool = False) -> torch.Tensor:
    return sha256_padded(pad_messages(msgs, msgs.shape[1]), compiled=compiled)


# --- the production frames -----------------------------------------------------------------------------------------------

POS_LEAF_TAG = b"verity/pos-leaf/v0"
NODE_TAG, LIFT_TAG = b"verity-vllm/node/v1", b"verity-vllm/lift/v1"
_U64_32 = struct.pack(">Q", 32)
NODE_PREFIX = struct.pack(">I", len(NODE_TAG)) + NODE_TAG + _U64_32          # 31 B
LIFT_PREFIX = struct.pack(">I", len(LIFT_TAG)) + LIFT_TAG + _U64_32          # 31 B


def _const_rows(prefix: bytes, n: int, device) -> torch.Tensor:
    return torch.tensor(list(prefix), dtype=torch.uint8, device=device).unsqueeze(0).expand(n, -1)


_TEMPLATES: dict = {}
TRITON_DIRECT = True     # frame kernels (registers) instead of the frame-buffer path; both verified against the eager backend


def _template(msg_len: int, fills: tuple[tuple[int, bytes], ...], device) -> torch.Tensor:
    """Cached FIPS-padded row [L] with constant byte ranges pre-filled (prefixes, separators, 0x80, bit-length tail)."""
    key = (msg_len, fills, str(device))
    t = _TEMPLATES.get(key)
    if t is None:
        L = ((msg_len + 9 + 63) // 64) * 64
        row = bytearray(L)
        for off, b in fills:
            row[off: off + len(b)] = b
        row[msg_len] = 0x80
        row[L - 8:] = struct.pack(">Q", msg_len * 8)
        t = _TEMPLATES[key] = torch.tensor(list(row), dtype=torch.uint8, device=device)
    return t


def _frames(n: int, msg_len: int, fills, device) -> torch.Tensor:
    t = _template(msg_len, fills, device)
    return t.unsqueeze(0).expand(n, -1).clone()     # one kernel; variable ranges are then written by slice copies


def pos_leaf_digests(values: torch.Tensor, *, compiled: bool = False) -> torch.Tensor:
    """uint8 [N, V] -> pos_leaf digests: SHA-256(POS_LEAF_TAG || u64be(V) || value)."""
    n, V = values.shape
    hdr = POS_LEAF_TAG + struct.pack(">Q", V)
    if compiled == "triton" and TRITON_DIRECT and V % 4 == 0 and len(hdr) + V + 9 <= 128:
        from .triton_sha256 import leaf_digests_direct   # frame built in registers from the stream words: ONE launch, no frame buffer
        return leaf_digests_direct(values.contiguous().reshape(-1), V)
    if compiled == "triton":                        # padded frames built in place: 2 kernels + 1 launch, no host->device constants
        buf = _frames(n, len(hdr) + V, ((0, hdr),), values.device)
        buf[:, len(hdr): len(hdr) + V] = values
        return sha256_padded(buf, compiled=compiled)
    msgs = torch.cat([_const_rows(hdr, n, values.device), values], dim=1)
    return sha256_messages(msgs, compiled=compiled)


def node_digests(left: torch.Tensor, right: torch.Tensor, *, compiled: bool = False) -> torch.Tensor:
    n = left.shape[0]
    msgs = torch.cat([_const_rows(NODE_PREFIX, n, left.device), left, _const_rows(_U64_32, n, left.device), right], dim=1)   # 103 B
    return sha256_messages(msgs, compiled=compiled)


def lift_digests(child: torch.Tensor, *, compiled: bool = False) -> torch.Tensor:
    n = child.shape[0]
    msgs = torch.cat([_const_rows(LIFT_PREFIX, n, child.device), child], dim=1)                                              # 63 B
    return sha256_messages(msgs, compiled=compiled)


_NODE_LEN = len(NODE_PREFIX) + 32 + 8 + 32      # 103
_LIFT_LEN = len(LIFT_PREFIX) + 32               # 63
_NP = len(NODE_PREFIX)


def _fold_level_triton(level: torch.Tensor) -> torch.Tensor:
    """One level of the production pairing/lift rule as ONE Triton launch: rows 0..pairs-1 are node frames, an odd trailing node
    gets a lift frame in the last row (same 128-B padded length, padding baked into the frame)."""
    m = level.shape[0]
    pairs, odd = m // 2, m % 2
    buf = _frames(pairs + odd, _NODE_LEN, ((0, NODE_PREFIX), (_NP + 32, _U64_32)), level.device)
    if pairs:
        pv = level[: 2 * pairs].view(pairs, 64)
        buf[:pairs, _NP: _NP + 32] = pv[:, :32]
        buf[:pairs, _NP + 40: _NP + 72] = pv[:, 32:]
    if odd:
        buf[pairs: pairs + 1] = _template(_LIFT_LEN, ((0, LIFT_PREFIX),), level.device)
        buf[pairs: pairs + 1, len(LIFT_PREFIX): len(LIFT_PREFIX) + 32] = level[m - 1: m]
    return sha256_padded(buf, compiled="triton")


def fold_levels(leaves: torch.Tensor, *, compiled: bool = False, stop_at: int = 0) -> list[torch.Tensor]:
    """Production pairing / lift rule over leaf digests uint8 [N, 32] -> list of levels (last = [1, 32] root).
    `stop_at` > 0: stop as soon as a level has <= stop_at nodes (the caller finishes those levels on the host)."""
    levels = [leaves]
    level = leaves
    while level.shape[0] > 1 and not (stop_at > 0 and level.shape[0] <= stop_at):
        if compiled == "triton":
            if TRITON_DIRECT:
                from .triton_sha256 import fold_level_direct
                nxt = fold_level_direct(level)
            else:
                nxt = _fold_level_triton(level)
            levels.append(nxt)
            level = nxt
            continue
        m = level.shape[0]
        pairs = level[: m - (m % 2)].view(-1, 64)
        nxt = node_digests(pairs[:, :32], pairs[:, 32:], compiled=compiled) if pairs.shape[0] else level[:0]
        if m % 2 == 1:
            nxt = torch.cat([nxt, lift_digests(level[m - 1: m], compiled=compiled)], dim=0)
        levels.append(nxt)
        level = nxt
    return levels


def reference_digests(msgs_bytes: list[bytes]) -> list[bytes]:
    return [hashlib.sha256(m).digest() for m in msgs_bytes]
