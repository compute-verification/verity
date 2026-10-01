"""``veritor.elem-bf16@1``: the elementwise BF16 gates of the vLLM activation and residual kernels.

The Python mirror of ``backends/sp1/common/src/elem.rs``, with the guest's manifest (so the digest the guest checks a
statement against) and its semantics:

* ``silu_mul_bf16`` -- vLLM ``act_and_mul_kernel<silu>``: ``bf16_rn(f32(s) * f32(u))`` with ``s`` the tabled
  ``bf16_rn(silu_f32(f32(g)))`` and the torch canonical NaN ``0x7FC0`` in both roundings;
* ``bf16_add_f2fp`` -- ``bf16_rn(f32(a) + f32(b))``, NaN -> ``0x7FFF``;
* ``bf16_to_f32`` -- the exact widening; ``in`` / ``weight`` -- 16-bit sources.

``silu_f32`` is the registered float64 stand-in for CUDA ``expf`` + division (RN32 of ``g / (1 + e^-g)``; NaN -> NaN,
``+inf`` -> ``+inf``, ``-inf`` -> ``-0``).  The guest embeds its 2^16-entry BF16 -> BF16 table; :func:`silu_table`
regenerates it from the same formula and :data:`SILU_TABLE_SHA256` pins it to the guest's bytes.  The FP32 add and
multiply are :mod:`verity.proofs.reduce_mufu`'s: they differ from ``elem.rs``'s only in NaN payloads, which every
gate here canonicalises.
"""

from __future__ import annotations

import hashlib
import math
import struct
from functools import cache

from .gateset import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet
from .reduce_mufu import BF16_NAN_TORCH, _f32_to_bf16_rn, bf16_add_f2fp, bf16_to_f32, f32_mul

__all__ = [
    "ELEM_BF16_DIGEST",
    "ELEM_BF16_NAME",
    "ELEM_BF16_VERSION",
    "SILU_TABLE_SHA256",
    "make_elem_bf16_gate_set",
    "silu_mul_bf16",
    "silu_table",
]

ELEM_BF16_NAME = "veritor.elem-bf16"
ELEM_BF16_VERSION = "1"
#: ``GateSet::resolve("veritor.elem-bf16@1", 32).digest()``.
ELEM_BF16_DIGEST = "061723740a994b6b47148b19a0b5c5a701edcd52398fd10e9a4a41d4575f5b9e"
#: sha256 of ``backends/sp1/common/src/elem_silu_table.bin`` (little-endian u16 per BF16 word), the guest's table.
SILU_TABLE_SHA256 = "bd9cb7bdc17bfb3a8ac9c415312053814258bc9aeb77dbdbd20720a6c8538f01"


def _f32_bits_of_double(x: float) -> int:
    """RN of a double to binary32 (overflow to infinity), as ``numpy.float32(x)``."""

    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F80_0000 if x > 0 else 0xFF80_0000


def _silu_f32_bits(u: int) -> int:
    e, m = (u >> 23) & 0xFF, u & 0x7F_FFFF
    if e == 0xFF:
        if m:
            return 0x7FC0_0000
        return 0x8000_0000 if u >> 31 else 0x7F80_0000
    if u & 0x7FFF_FFFF == 0:
        return 0  # the registered evaluator decodes -0 as the value 0: silu(-0) = +0
    g = struct.unpack("<f", struct.pack("<I", u))[0]
    if g < -80.0:
        return _f32_bits_of_double(g * math.exp(g))
    return _f32_bits_of_double(g / (1.0 + math.exp(-g)))


@cache
def silu_table() -> tuple[int, ...]:
    """``bf16_rn_torch(silu_f32(f32(g)))`` for every BF16 word ``g``; raises if it is not the guest's table."""

    table = tuple(_f32_to_bf16_rn(_silu_f32_bits(bf16_to_f32(g)), BF16_NAN_TORCH) for g in range(1 << 16))
    digest = hashlib.sha256(b"".join(w.to_bytes(2, "little") for w in table)).hexdigest()
    if digest != SILU_TABLE_SHA256:
        raise RuntimeError(f"the regenerated SiLU table {digest} is not the guest's {SILU_TABLE_SHA256}")
    return table


def silu_mul_bf16(g: int, u: int) -> int:
    return _f32_to_bf16_rn(f32_mul(silu_table()[g & 0xFFFF] << 16, bf16_to_f32(u)), BF16_NAN_TORCH)


def make_elem_bf16_gate_set() -> GateSet:
    """The gate set ``veritor.elem-bf16@1`` (statement width 32; 16-bit sources)."""

    return GateSet(
        (
            Gate("bf16_add_f2fp", 2, 16, replay_cost=1, proof_cost=1, evaluate=lambda a: bf16_add_f2fp(*a)),
            Gate("bf16_to_f32", 1, 32, replay_cost=1, proof_cost=1, evaluate=lambda a: bf16_to_f32(*a), arg_widths=(16,)),
            Gate("in", 0, 16, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("silu_mul_bf16", 2, 16, replay_cost=2, proof_cost=2, evaluate=lambda a: silu_mul_bf16(*a)),
            Gate("weight", 0, 16, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=ELEM_BF16_NAME,
        version=ELEM_BF16_VERSION,
    )
