"""circuit-check's bindings of `verity.circuits.ops.boolean.scalar`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_constant_roots() -> list[Any]:
    """The Boolean constants GLM-5.3 TP8's B1 request Calls, 1 and 0x11f (`const_word`: made on demand, as the word constant is,
    so no module binds one ahead)."""
    from verity.circuits.defs import bind
    from verity.circuits.ops.boolean import scalar as BS

    return [bind(BS.const_word(32, 1)), bind(BS.const_word(32, 0x11F))]


def _boolean_wiring_roots() -> list[Any]:
    """The wiring `WordBits{16}` and `BitsWord{32}` Call, on bits (`bit_of_bits`, `word_of_bits`: made on demand, as the word
    primitives are, so no module binds one ahead): the 16-bit word's first and last wires, and the 32 bits as a word."""
    from verity.circuits.defs import bind
    from verity.circuits.ops.boolean import scalar as BS

    return [bind(BS.bit_of_bits(16, 0)), bind(BS.bit_of_bits(16, 15)), bind(BS.word_of_bits(32))]
