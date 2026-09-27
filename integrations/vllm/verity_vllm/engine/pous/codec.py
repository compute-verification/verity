"""P3 on the GPU: the `reference` backend of `verity_pous.primitives` (the SHAKE256 instantiation), as batched torch code.

The operating point is `verity_pous.params.OPERATING_POINT`: a segment is `N` = 220 labels of `M` = 65,560 bytes (220 components
of 298 bytes), band graph B(220, 12), two layers joined by the transpose.  `verity_pous` is the oracle: the verifier's own
encoding with it must have the server's root, and `tests/engine/test_pous_codec.py` checks each piece against it.

    tag_s   = salt (24 B) || u64(s)
    Pi      = Feistel-10 on M + 64 bytes, F_i(x) = SHAKE256(0x48 || u8(i) || x)[0 : (M + 64) / 2]
    P_T     = Feistel-10 on M bytes,      F_i(T, x) = SHAKE256(0x50 || u8(i) || T || x)[0 : M / 2],  T = tag_s || u8(layer) || u16(v)
    H       = sponge over Pi, rate M, capacity 64 B, IV 0: the header block "pous-p3-v1-H" || tag_s || u8(layer) || u16(v) || u8(npar)
              zero-filled to M, then the parents newest first (c_{v-1} .. c_{v-npar}, npar = min(v, 12)), then one pad10*1 block;
              key = S[0 : M]
    c1_v    = P_(tag,0,v)(W_v ^ H(tag, 0, v, c1_par(v)))
    c2_v    = P_(tag,1,v)(transpose(c1)_v ^ H(tag, 1, v, c2_par(v)))

Feistel-10 is (L, R) -> (R, L ^ F_i(R)) for i = 1..10, computed in place on the two halves `A || B`: odd rounds `A ^= F_i(B)`, even
rounds `B ^= F_i(A)`; ten rounds leave the output as `A || B`.  The inverse runs the rounds backwards.  Decode is two parallel
passes per segment; encode is sequential in `v` within a segment and parallel across segments.  Every tensor is `uint8` on one
CUDA device; every SHAKE256 call is one launch of `pous_native.shake256_rows` over all rows.
"""
from __future__ import annotations

import torch
from verity_pous import primitives
from verity_pous.params import OPERATING_POINT, SALT_BITS, TAG_BITS

from verity_vllm.engine.pous import native

N = OPERATING_POINT.n
WC = OPERATING_POINT.wc // 8
M = OPERATING_POINT.block_bytes
K = OPERATING_POINT.k
SEGMENT_BYTES = OPERATING_POINT.segment_bytes
PI_BYTES = M + primitives.CAPACITY_BITS // 8
SALT_BYTES = SALT_BITS // 8
TAG_BYTES = TAG_BITS // 8
H_HEADER = primitives.HEADER
ROUNDS = 10
PI_DOMAIN = primitives.PI_DOMAIN
P_DOMAIN = primitives.P_DOMAIN
NAME = "verity_pous.primitives.reference/gpu-v1"
assert OPERATING_POINT.wc % 8 == 0 and N * WC == M


def segment_tags(salt: bytes, segments: list[int], device) -> torch.Tensor:
    """`[len(segments), 32]` tags `salt || u64(s)`."""
    if len(salt) != SALT_BYTES:
        raise ValueError(f"the salt is {SALT_BYTES} bytes")
    rows = [salt + int(s).to_bytes(8, "little") for s in segments]
    return torch.tensor(list(b"".join(rows)), dtype=torch.uint8, device=device).view(len(segments), TAG_BYTES)


def _u8(values, device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.uint8, device=device)


def _feistel(x: torch.Tensor, prefix_of_round, inverse: bool) -> None:
    """In-place Feistel-10 on the rows of `x` (`[R, 2h]`); `prefix_of_round(i)` is the `[R or 1, p]` round-function prefix."""
    ext = native.ext()
    h = x.shape[1] // 2
    a, b = x[:, :h], x[:, h:]
    for i in (range(ROUNDS, 0, -1) if inverse else range(1, ROUNDS + 1)):
        src, dst = (b, a) if i % 2 == 1 else (a, b)
        ext.shake256_rows(prefix_of_round(i), src, dst, True)


def pi(state: torch.Tensor) -> None:
    """`Pi` in place on `[R, M + 64]` states."""
    dev = state.device
    prefixes = {i: _u8([PI_DOMAIN, i], dev).view(1, 2).expand(state.shape[0], 2) for i in range(1, ROUNDS + 1)}
    _feistel(state, lambda i: prefixes[i], inverse=False)


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
    tw = _tweaks(tags, layer, vs)
    pre = torch.empty(x.shape[0], 2 + tw.shape[1], dtype=torch.uint8, device=x.device)
    pre[:, 0] = P_DOMAIN
    pre[:, 2:] = tw

    def prefix(i: int) -> torch.Tensor:
        pre[:, 1] = i
        return pre

    _feistel(x, prefix, inverse)


def keys(tags: torch.Tensor, layer: int, vs: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """`H(tags[r], layer, vs[r], parents)` for every row: `[R, M]`, the parents of row `r` being `labels[groups[r], vs[r] - j]`,
    `j = 1 .. min(vs[r], K)` (`labels` is `[G, N, M]`)."""
    r = tags.shape[0]
    dev = tags.device
    npar = torch.clamp(vs, max=K)
    state = torch.zeros(r, PI_BYTES, dtype=torch.uint8, device=dev)
    state[:, :len(H_HEADER)] = _u8(H_HEADER, dev)
    o = len(H_HEADER)
    state[:, o:o + TAG_BYTES] = tags
    state[:, o + TAG_BYTES] = layer
    state[:, o + TAG_BYTES + 1] = (vs & 0xFF).to(torch.uint8)
    state[:, o + TAG_BYTES + 2] = (vs >> 8).to(torch.uint8)
    state[:, o + TAG_BYTES + 3] = npar.to(torch.uint8)
    pi(state)
    for j in range(1, K + 2):
        par = (npar >= j).nonzero().flatten()
        pad = (npar == j - 1).nonzero().flatten()
        if len(par):
            sub = state[par]
            sub[:, :M] ^= labels[groups[par], vs[par] - j]
            pi(sub)
            state[par] = sub
        if len(pad):
            sub = state[pad]
            sub[:, 0] ^= 0x01
            sub[:, M - 1] ^= 0x80
            pi(sub)
            state[pad] = sub
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
