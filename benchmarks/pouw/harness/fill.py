"""The numpy twin of `native/harness.cu`'s `k_fill`: the bytes a fill of each format writes, from its seed. The harness checks
the device's fill against it once per run (the preflight), and the baselines' gates decode the ternary operands with it."""

from __future__ import annotations

import numpy as np

F32, BF16, E4M3, E2M1, INT8 = 0, 1, 2, 3, 4
UE4M3_ONE, UE8M0_ONE, F32_ONE, ZERO = 5, 6, 7, 8
F32_TERNARY, BF16_TERNARY, E4M3_TERNARY, E2M1_TERNARY, INT8_TERNARY = 12, 13, 14, 15, 16

TERNARY = {F32: F32_TERNARY, BF16: BF16_TERNARY, E4M3: E4M3_TERNARY, E2M1: E2M1_TERNARY, INT8: INT8_TERNARY}
UNIT_BYTES = {F32: 4, F32_ONE: 4, F32_TERNARY: 4, BF16: 2, BF16_TERNARY: 2}

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def unit_bytes(fmt: int) -> int:
    return UNIT_BYTES.get(fmt, 1)


def splitmix(seed: int, idx: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.uint64(seed & 0xFFFFFFFFFFFFFFFF) + (np.asarray(idx, np.uint64) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def _ternary(h: np.ndarray, shift: int) -> np.ndarray:
    return ((h >> np.uint64(shift)) % np.uint64(3)).astype(np.int64) - 1


def fill(fmt: int, units: int, seed: int, start: int = 0) -> bytes:
    """The bytes of units start .. start+units of a `fmt` fill from `seed` (a unit is one element; E2M1 packs two per byte)."""
    h = splitmix(seed, np.arange(start, start + units, dtype=np.uint64))
    u32 = lambda x: x.astype(np.uint32)  # noqa: E731
    if fmt == F32:
        sign, exp, mant = u32(h >> np.uint64(63)), u32(122 + (h >> np.uint64(40)) % np.uint64(8)), u32(h & np.uint64(0x7FFFFF))
        return ((sign << 31) | (exp << 23) | mant).astype("<u4").tobytes()
    if fmt == BF16:
        sign, exp = u32(h >> np.uint64(63)), u32(122 + (h >> np.uint64(40)) % np.uint64(8))
        mant = u32((h >> np.uint64(16)) & np.uint64(0x7F))
        return ((sign << 15) | (exp << 7) | mant).astype("<u2").tobytes()
    if fmt == E4M3:
        c = ((h >> np.uint64(24)) & np.uint64(0xFF)).astype(np.uint8)
        return np.where((c & 0x7F) == 0x7F, c - 1, c).astype(np.uint8).tobytes()
    if fmt == E2M1:
        return ((h >> np.uint64(24)) & np.uint64(0xFF)).astype(np.uint8).tobytes()
    if fmt == INT8:
        return ((h >> np.uint64(40)) & np.uint64(0xFF)).astype(np.uint8).tobytes()
    if fmt == UE4M3_ONE:
        return b"\x38" * units
    if fmt == UE8M0_ONE:
        return b"\x7f" * units
    if fmt == F32_ONE:
        return np.ones(units, "<f4").tobytes()
    if fmt == ZERO:
        return bytes(units)
    if fmt == F32_TERNARY:
        return _ternary(h, 32).astype("<f4").tobytes()
    if fmt == BF16_TERNARY:
        v = _ternary(h, 32)
        return np.where(v == 0, 0, np.where(v > 0, 0x3F80, 0xBF80)).astype("<u2").tobytes()
    if fmt == E4M3_TERNARY:
        v = _ternary(h, 32)
        return np.where(v == 0, 0, np.where(v > 0, 0x38, 0xB8)).astype(np.uint8).tobytes()
    if fmt == E2M1_TERNARY:
        code = lambda v: np.where(v == 0, 0, np.where(v > 0, 0x2, 0xA))  # noqa: E731
        return (code(_ternary(h, 32)) | (code(_ternary(h, 40)) << 4)).astype(np.uint8).tobytes()
    if fmt == INT8_TERNARY:
        return _ternary(h, 32).astype(np.int8).tobytes()
    raise ValueError(f"unknown fill format {fmt}")


# ---- decoding operands to exact values (float64) ---------------------------------------------------------------------------

E2M1_VALUES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float64)


def e4m3_values() -> np.ndarray:
    """The value of every E4M3 code (NaN at 0x7F and 0xFF)."""
    c = np.arange(256)
    sign = np.where(c & 0x80, -1.0, 1.0)
    e, m = (c >> 3) & 0xF, c & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((c & 0x7F) == 0x7F, np.nan, v)
    return sign * v


E4M3_VALUES = e4m3_values()


def decode(fmt: int, raw: bytes) -> np.ndarray:
    """The elements' values, as float64 (exact for every format here)."""
    b = np.frombuffer(raw, np.uint8)
    base = {F32_TERNARY: F32, BF16_TERNARY: BF16, E4M3_TERNARY: E4M3, E2M1_TERNARY: E2M1, INT8_TERNARY: INT8}.get(fmt, fmt)
    if base == F32:
        return np.frombuffer(raw, "<f4").astype(np.float64)
    if base == BF16:
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if base == E4M3:
        return E4M3_VALUES[b]
    if base == E2M1:
        return np.stack([E2M1_VALUES[b & 0xF], E2M1_VALUES[b >> 4]], axis=-1).reshape(-1)
    if base == INT8:
        return b.view(np.int8).astype(np.float64)
    raise ValueError(f"no decoding for format {fmt}")


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """float64 values (exact in FP32) to BF16 words, rounding to nearest even."""
    bits = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + np.uint64(0x7FFF) + ((bits >> np.uint64(16)) & np.uint64(1))) >> np.uint64(16)).astype(np.uint16)
