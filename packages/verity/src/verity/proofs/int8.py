"""The ``veritor.int8-q7`` gate set: exact int8 products, int32 sums, Q7 requantization.

The shared exact-integer workload is a two-layer quantized network::

    S = X @ W1                          X int8 [m, d], W1 int8 [d, h]
    H = clip(floor(S / 128), 0, 127)    int8, values 0 .. 127
    T = H @ W2                          W2 int8 [h, d]
    Y = clip(floor(T / 128), -128, 127) int8

with ``floor`` toward ``-inf`` and ``clip`` saturating.  Every value is an
unsigned word of the declared width read as two's complement: an 8-bit word
holds an int8 (or an ``H`` value in ``0 .. 127``), a 32-bit word an int32.
The gates:

* ``mul8`` (8, 8 -> 32): the exact signed product of two int8 words as an
  int32 word;
* ``add32`` (32, 32 -> 32): addition modulo ``2**32``.  Within the declared
  domain -- partial sums of at most :data:`INT8_Q7_MAX_CONTRACTION` products,
  so ``|sum| <= 8192 * 2**14 = 2**27 < 2**31`` -- this *is* integer
  addition: no sum ever leaves the int32 range, so a dot product of
  ``k <= 8192`` terms accumulates exactly.  The bound on ``k`` is part of the
  statement a circuit over this set makes, not of the gate;
* ``requant_u7`` (32 -> 8): ``clip(floor(s / 128), 0, 127)`` of the int32
  word ``s``;
* ``requant_s8`` (32 -> 8): ``clip(floor(t / 128), -128, 127)`` of the int32
  word ``t``, as an int8 word;
* ``in`` and ``weight``: 8-bit sources.

The costs follow the toy ISA: a product costs ``2``, every other operation
``1``, a source ``0`` to replay and ``1`` to prove.  ``backends/sp1/common/src/
gateset.rs`` mirrors the manifest and the semantics bit for bit.

Promoted from veritor ``core/int8.py`` (eb865ce2), unchanged.  Carried so the Python column of the
``typed-obligation/v0`` conformance suite covers every vector, not because Verity builds new obligations over it.
"""

from __future__ import annotations

from typing import Final

from .gateset import INPUT_SOURCE, WEIGHT_SOURCE, Gate, GateSet

__all__ = [
    "INT8_Q7_MAX_CONTRACTION",
    "INT8_Q7_NAME",
    "INT8_Q7_VERSION",
    "make_int8_q7_gate_set",
    "requant_s8",
    "requant_u7",
    "signed",
    "word",
]

INT8_Q7_NAME: Final = "veritor.int8-q7"
INT8_Q7_VERSION: Final = "1"
INT8_Q7_MAX_CONTRACTION: Final = 8192
"""The longest dot product ``add32`` accumulates exactly: ``8192 * 2**14 = 2**27``."""

_MASK32 = (1 << 32) - 1


def signed(width: int, value: int) -> int:
    """The two's-complement reading of a ``width``-bit word."""

    return value - (1 << width) if value >> (width - 1) else value


def word(width: int, value: int) -> int:
    """The ``width``-bit two's-complement word of an integer in range."""

    return value & ((1 << width) - 1)


def requant_u7(s: int) -> int:
    """``clip(floor(s / 128), 0, 127)`` of an int32 ``s`` (``//`` is floor division)."""

    return min(max(s // 128, 0), 127)


def requant_s8(t: int) -> int:
    """``clip(floor(t / 128), -128, 127)`` of an int32 ``t``."""

    return min(max(t // 128, -128), 127)


def _mul8(args: tuple[int, ...]) -> int:
    return word(32, signed(8, args[0]) * signed(8, args[1]))


def _add32(args: tuple[int, ...]) -> int:
    return (args[0] + args[1]) & _MASK32


def _requant_u7(args: tuple[int, ...]) -> int:
    return requant_u7(signed(32, args[0]))


def _requant_s8(args: tuple[int, ...]) -> int:
    return word(8, requant_s8(signed(32, args[0])))


def make_int8_q7_gate_set() -> GateSet:
    """``veritor.int8-q7@1``: ``mul8``, ``add32``, ``requant_u7``, ``requant_s8`` and 8-bit sources.

    See the module docstring for the semantics and the domain bound that
    makes ``add32`` exact.  The set has two widths, so a statement over it
    carries ``32`` (its widest gate) and every opened value is decoded by
    its own leaf schema (``u8`` or ``u32``).
    """

    return GateSet(
        (
            Gate(
                "mul8",
                2,
                32,
                replay_cost=2,
                proof_cost=2,
                evaluate=_mul8,
                arg_widths=(8, 8),
            ),
            Gate("add32", 2, 32, replay_cost=1, proof_cost=1, evaluate=_add32),
            Gate(
                "requant_u7",
                1,
                8,
                replay_cost=1,
                proof_cost=1,
                evaluate=_requant_u7,
                arg_widths=(32,),
            ),
            Gate(
                "requant_s8",
                1,
                8,
                replay_cost=1,
                proof_cost=1,
                evaluate=_requant_s8,
                arg_widths=(32,),
            ),
            Gate("in", 0, 8, replay_cost=0, proof_cost=1, source=INPUT_SOURCE),
            Gate("weight", 0, 8, replay_cost=0, proof_cost=1, source=WEIGHT_SOURCE),
        ),
        name=INT8_Q7_NAME,
        version=INT8_Q7_VERSION,
    )
