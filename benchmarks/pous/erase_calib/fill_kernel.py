"""The Triton fill kernel for `fill.device_fill`, in its own module so `fill` imports without Triton for the CPU tests.

The helpers are module-level `@triton.jit` functions: the interpreter (`TRITON_INTERPRET=1`, which `driver.py validate
--device cpu` runs) re-executes a kernel's source against its module globals, so helpers nested in a closure are not
visible to it."""

import triton
import triton.language as tl


@triton.jit
def _rotl(x, n: tl.constexpr):
    return (x << n) | (x >> (32 - n))


@triton.jit
def _qr(a, b, c, d):
    a = a + b
    d = _rotl(d ^ a, 16)
    c = c + d
    b = _rotl(b ^ c, 12)
    a = a + b
    d = _rotl(d ^ a, 8)
    c = c + d
    b = _rotl(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _fmix(h):
    h = h ^ (h >> 16)
    h = h * 0x85EBCA6B
    h = h ^ (h >> 13)
    h = h * 0xC2B2AE35
    return h ^ (h >> 16)


@triton.jit
def _put(out_ptr, b, r, val, n_blocks, a, LAYOUT: tl.constexpr, W: tl.constexpr):
    # b: the logical block within this fill (int64 scalar); r: in-block words (int64 [LANES])
    if LAYOUT == 0:
        off = b * W + r
    elif LAYOUT == 1:
        off = ((b * a) % n_blocks) * W + r
    else:
        off = r * n_blocks + b
    tl.store(out_ptr + off, val.to(tl.int32, bitcast=True))


@triton.jit
def fill_kernel(out_ptr, key_ptr, n_blocks, a, block_base,
                ROUNDS: tl.constexpr, LAYOUT: tl.constexpr, L: tl.constexpr, NW: tl.constexpr):
    b = tl.program_id(0).to(tl.int64)
    gb = b + block_base  # the block's index in the whole stream (a chunked fill passes its offset)
    lane = tl.arange(0, L).to(tl.int64)
    z = tl.zeros([L], dtype=tl.uint32)
    k0 = tl.load(key_ptr + 0).to(tl.uint32, bitcast=True)
    k1 = tl.load(key_ptr + 1).to(tl.uint32, bitcast=True)
    if ROUNDS == 0:
        for s in tl.static_range(NW):
            r = s * L + lane
            g = gb * (NW * L) + r
            x = _fmix((g & 0xFFFFFFFF).to(tl.uint32) ^ k0)
            x = _fmix(x ^ (g >> 32).to(tl.uint32) ^ k1)
            _put(out_ptr, b, r, x, n_blocks, a, LAYOUT, NW * L)
    else:
        k2 = tl.load(key_ptr + 2).to(tl.uint32, bitcast=True)
        k3 = tl.load(key_ptr + 3).to(tl.uint32, bitcast=True)
        k4 = tl.load(key_ptr + 4).to(tl.uint32, bitcast=True)
        k5 = tl.load(key_ptr + 5).to(tl.uint32, bitcast=True)
        k6 = tl.load(key_ptr + 6).to(tl.uint32, bitcast=True)
        k7 = tl.load(key_ptr + 7).to(tl.uint32, bitcast=True)
        n0 = tl.load(key_ptr + 8).to(tl.uint32, bitcast=True)
        n1 = tl.load(key_ptr + 9).to(tl.uint32, bitcast=True)
        ctr = gb * L + lane
        c_lo = (ctr & 0xFFFFFFFF).to(tl.uint32)
        c_hi = (ctr >> 32).to(tl.uint32)
        x0 = z + 0x61707865
        x1 = z + 0x3320646E
        x2 = z + 0x79622D32
        x3 = z + 0x6B206574
        x4 = z + k0
        x5 = z + k1
        x6 = z + k2
        x7 = z + k3
        x8 = z + k4
        x9 = z + k5
        x10 = z + k6
        x11 = z + k7
        x12 = c_lo
        x13 = c_hi
        x14 = z + n0
        x15 = z + n1
        for _ in tl.static_range(ROUNDS // 2):
            x0, x4, x8, x12 = _qr(x0, x4, x8, x12)
            x1, x5, x9, x13 = _qr(x1, x5, x9, x13)
            x2, x6, x10, x14 = _qr(x2, x6, x10, x14)
            x3, x7, x11, x15 = _qr(x3, x7, x11, x15)
            x0, x5, x10, x15 = _qr(x0, x5, x10, x15)
            x1, x6, x11, x12 = _qr(x1, x6, x11, x12)
            x2, x7, x8, x13 = _qr(x2, x7, x8, x13)
            x3, x4, x9, x14 = _qr(x3, x4, x9, x14)
        W = NW * L
        _put(out_ptr, b, 0 * L + lane, x0 + 0x61707865, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 1 * L + lane, x1 + 0x3320646E, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 2 * L + lane, x2 + 0x79622D32, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 3 * L + lane, x3 + 0x6B206574, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 4 * L + lane, x4 + k0, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 5 * L + lane, x5 + k1, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 6 * L + lane, x6 + k2, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 7 * L + lane, x7 + k3, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 8 * L + lane, x8 + k4, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 9 * L + lane, x9 + k5, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 10 * L + lane, x10 + k6, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 11 * L + lane, x11 + k7, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 12 * L + lane, x12 + c_lo, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 13 * L + lane, x13 + c_hi, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 14 * L + lane, x14 + n0, n_blocks, a, LAYOUT, W)
        _put(out_ptr, b, 15 * L + lane, x15 + n1, n_blocks, a, LAYOUT, W)
