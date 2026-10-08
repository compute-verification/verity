"""P3 on the GPU: the `reference` backend of `verity.core.protocols.pous.primitives` (the SHAKE256 instantiation), as batched torch code.

The operating point is `verity.core.protocols.pous.params.OPERATING_POINT`: a segment is `N` = 220 labels of `M` = 65,560 bytes (220 components
of 298 bytes), band graph B(220, 12), two layers joined by the transpose.  `verity.core.protocols.pous` is the oracle: the verifier's own
encoding with it must have the server's root, and the vLLM tree's `tests/engine/test_pous_codec.py` checks each piece against it.

This module is the GPU twin of `scheme.P3`; `dense.py` beside it is that of `dense.DENSE` and of the band (`band.BAND12`), and
holds the Feistel dispatch switches and `segment_tags` the band's kernels use.

    tag_s   = salt (24 B) || u64(s)
    Pi, Pi2 = Feistel-10 on M + 64 or 2M + 64 bytes, F_i(x) = SHAKE256(0x48 || u8(i) || x)[0 : width / 2]
    P_T     = Feistel-10 on M bytes,      F_i(T, x) = SHAKE256(0x50 || u8(i) || T || x)[0 : M / 2],  T = tag_s || u8(layer) || u16(v)
    H       = the reference's `KeyDerivation` (`key_mode`): the overwrite chain over Pi2 (Feistel-10 on 2M + 64 bytes, the
              default) or the XOR sponge over Pi, absorbing the header, the parents newest first (c_{v-1} .. c_{v-npar},
              npar = min(v, 12)) and a pad10*1 block
    c1_v    = P_(tag,0,v)(W_v ^ H(tag, 0, v, c1_par(v)))
    c2_v    = P_(tag,1,v)(transpose(c1)_v ^ H(tag, 1, v, c2_par(v)))

Feistel-10 is (L, R) -> (R, L ^ F_i(R)) for i = 1..10, computed in place on the two halves `A || B` by one launch of
`pous_native.feistel_rows` (one warp per row): odd rounds `A ^= F_i(B)`, even rounds `B ^= F_i(A)`, which leaves the output as
`A || B`; the inverse runs the rounds backwards.  Decode is two parallel passes per segment; encode is sequential in `v` within a
segment and parallel across segments.  Every tensor is `uint8` on one CUDA device.
"""
from __future__ import annotations

import torch
from verity.core.protocols.pous import primitives
from verity.core.protocols.pous.codec import Codec
from verity.core.protocols.pous.params import OPERATING_POINT, SALT_BITS, TAG_BITS
from verity.core.protocols.pous.scheme import P3

from . import native

N = OPERATING_POINT.n
WC = OPERATING_POINT.wc // 8
M = OPERATING_POINT.block_bytes
K = OPERATING_POINT.k
SEGMENT_BYTES = OPERATING_POINT.segment_bytes
PI_BYTES = M + primitives.CAPACITY_BITS // 8
SALT_BYTES = SALT_BITS // 8
TAG_BYTES = TAG_BITS // 8
H_HEADER = primitives.HEADER
PI_DOMAIN = primitives.PI_DOMAIN
P_DOMAIN = primitives.P_DOMAIN
NAME = "verity.core.protocols.pous.primitives.reference/gpu-v1"
assert OPERATING_POINT.wc % 8 == 0 and N * WC == M


def segment_tags(salt: bytes, segments: list[int], device) -> torch.Tensor:
    """`[len(segments), 32]` tags `salt || u64(s)`."""
    if len(salt) != SALT_BYTES:
        raise ValueError(f"the salt is {SALT_BYTES} bytes")
    rows = [salt + int(s).to_bytes(8, "little") for s in segments]
    return torch.tensor(list(b"".join(rows)), dtype=torch.uint8, device=device).view(len(segments), TAG_BYTES)


def _u8(values, device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.uint8, device=device)


THREAD_ROWS = [0]
"""From this many rows a Feistel launch runs one thread per row (throughput) instead of one warp (latency); 0 keeps the warp."""

WORD_IO = [False]
"""One-thread launches of untweaked permutations with whole-word halves use 64-bit word I/O (`feistel_words_kernel`)."""

FUSED = [False]
"""With `WORD_IO`, forward launches fuse each round's squeeze to the next round's absorb (`feistel_fused_kernel`)."""

PAIR = [False]
"""With `FUSED`, launches below `PAIR_MAX_ROWS` rows run each row on a lane pair (`feistel_pair_kernel`)."""

PAIR_MAX_ROWS = [1 << 62]
COALESCED = [False]
"""With `FUSED`, the other launches stage every chunk through shared memory (`feistel_coalesced_kernel`)."""


def permute(state: torch.Tensor) -> None:
    """The key derivation's permutation (`Pi` or `Pi2`, by the state's width) in place on `[R, width]` states."""
    native.ext().feistel_rows(state, state.new_empty((state.shape[0], 0)), PI_DOMAIN, False, THREAD_ROWS[0], WORD_IO[0], FUSED[0], PAIR[0],
                              PAIR_MAX_ROWS[0], COALESCED[0])


def _tweaks(tags: torch.Tensor, layer: int, vs: torch.Tensor) -> torch.Tensor:
    """`[R, 35]` tweaks `tag || u8(layer) || u16(v)`."""
    r = tags.shape[0]
    t = torch.empty(r, TAG_BYTES + 3, dtype=torch.uint8, device=tags.device)
    t[:, :TAG_BYTES] = tags
    t[:, TAG_BYTES] = layer
    t[:, TAG_BYTES + 1] = (vs & 0xFF).to(torch.uint8)
    t[:, TAG_BYTES + 2] = (vs >> 8).to(torch.uint8)
    return t


def p_t(x: torch.Tensor, tags: torch.Tensor, layer: int, vs: torch.Tensor, inverse: bool = False) -> None:
    """`P_T` (or its inverse) in place on `[R, M]` labels, row `r` under tweak `(tags[r], layer, vs[r])`."""
    native.ext().feistel_rows(x, _tweaks(tags, layer, vs), P_DOMAIN, inverse, THREAD_ROWS[0], WORD_IO[0], FUSED[0], PAIR[0],
                              PAIR_MAX_ROWS[0], COALESCED[0])


def key_mode(prims) -> str:
    """Which GPU twin computes the reference `Primitives` `prims`, read off their interfaces: `P_T` must be `FeistelP`, the key
    derivation's permutation `FeistelPi`, and its call schedule for `K` parents (`h.steps`) and width decide the mode.
    `overwrite`: the header in the chaining value, then `(r, h) <- Pi2(B || h)` for the parents newest first and the pad, `Pi2`
    of `2m + 512` bits, the key the pad call's `r` (`OverwriteChain`, the default).  `sponge`: a header call, the parents and
    the pad each XORed into the rate of `Pi` of `m + 512` bits (`XorSponge`, broken, kept to check the code on).  Anything
    else, feed-forward modes included, fails closed."""
    h = prims.h
    perm = getattr(h, "pi2", None) or getattr(h, "pi", None)
    newest = [(u,) for u in reversed(range(K))]
    steps = [tuple(x) for x in h.steps(range(K))]
    width = getattr(perm, "width", None)
    feistel = isinstance(prims.p, primitives.FeistelP) and isinstance(perm, primitives.FeistelPi)
    if feistel and not getattr(h, "feed_forward", False):
        if steps == [*newest, ()] and width == 16 * M + primitives.CAPACITY_BITS:
            return "overwrite"
        if steps == [(), *newest, ()] and width == 8 * M + primitives.CAPACITY_BITS:
            return "sponge"
    raise NotImplementedError(f"no GPU twin of {type(h).__name__} over {type(perm).__name__} with {type(prims.p).__name__}")


MODE = key_mode(primitives.reference(OPERATING_POINT))


def _header(tags: torch.Tensor, layer: int, vs: torch.Tensor, npar: torch.Tensor) -> torch.Tensor:
    """`[R, 48]`: "pous-p3-v1-H" || tag || u8(layer) || u16(v) || u8(npar)."""
    o = len(H_HEADER)
    hdr = torch.empty(tags.shape[0], o + TAG_BYTES + 4, dtype=torch.uint8, device=tags.device)
    hdr[:, :o] = _u8(H_HEADER, tags.device)
    hdr[:, o:o + TAG_BYTES] = tags
    hdr[:, o + TAG_BYTES] = layer
    hdr[:, o + TAG_BYTES + 1] = (vs & 0xFF).to(torch.uint8)
    hdr[:, o + TAG_BYTES + 2] = (vs >> 8).to(torch.uint8)
    hdr[:, o + TAG_BYTES + 3] = npar.to(torch.uint8)
    return hdr


def keys(tags: torch.Tensor, layer: int, vs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """`H(tags[r], layer, vs[r], parents)` for every row: `[R, M]`, the parents of row `r` being `labels[groups[r], vs[r] - j]`,
    `j = 1 .. min(vs[r], K)` (`labels` is `[G, N, M]`), in `MODE`."""
    r = tags.shape[0]
    npar = torch.clamp(vs, max=K)
    hdr = _header(tags, layer, vs, npar)
    sponge = MODE == "sponge"
    state = torch.zeros(r, PI_BYTES if sponge else M + PI_BYTES, dtype=torch.uint8, device=tags.device)
    if sponge:
        state[:, :hdr.shape[1]] = hdr
        permute(state)
    else:
        state[:, M:M + hdr.shape[1]] = hdr
    for j in range(1, K + 2):
        for rows, pad in (((npar >= j).nonzero().flatten(), False), ((npar == j - 1).nonzero().flatten(), True)):
            if not len(rows):
                continue
            sub = state[rows]
            if pad:
                block = torch.zeros(len(rows), M, dtype=torch.uint8, device=tags.device)
                block[:, 0], block[:, M - 1] = 0x01, 0x80
            else:
                block = labels[groups[rows], vs[rows] - j]
            if sponge:
                sub[:, :M] ^= block
            else:
                sub[:, :M] = block
            permute(sub)
            state[rows] = sub
    return state[:, :M].contiguous()


def transpose(labels: torch.Tensor) -> torch.Tensor:
    """`[G, N, M]` -> `[G, N, M]`: label `v` of the result is component `v` of every input label, in order."""
    g = labels.shape[0]
    return labels.view(g, N, N, WC).transpose(1, 2).contiguous().view(g, N, M)


def decode(c2: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """`W` of `G` segments: `c2` is `[G, N, M]` (the stored top labels), `tags` `[G, 32]`; returns `[G, N, M]`."""
    g = c2.shape[0]
    dev = c2.device
    groups = torch.arange(g, device=dev).repeat_interleave(N)
    vs = torch.arange(N, device=dev).repeat(g)
    rtags = tags[groups]
    x2 = c2.reshape(g * N, M).clone()
    p_t(x2, rtags, 1, vs, inverse=True)
    x2 ^= keys(rtags, 1, vs, c2, groups)
    c1 = transpose(x2.view(g, N, M))
    w = c1.reshape(g * N, M).clone()
    p_t(w, rtags, 0, vs, inverse=True)
    w ^= keys(rtags, 0, vs, c1, groups)
    return w.view(g, N, M)


def encode(w: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """`C` (the top labels) of `G` segments: `w` is `[G, N, M]`, `tags` `[G, 32]`; sequential in `v`, parallel over segments."""
    g = w.shape[0]
    dev = w.device
    groups = torch.arange(g, device=dev)
    lower = torch.zeros_like(w)
    for layer, inputs, out in ((0, w, lower), (1, None, torch.zeros_like(w))):
        if inputs is None:
            inputs = transpose(lower)
        for v in range(N):
            vs = torch.full((g,), v, dtype=torch.long, device=dev)
            x = inputs[:, v] ^ keys(tags, layer, vs, out, groups)
            p_t(x, tags, layer, vs)
            out[:, v] = x
        top = out
    return top


REFERENCE: Codec = P3
"""The reference codec the server encodes with and the verifier checks against, unless a run names another
(`twins.reference`)."""
