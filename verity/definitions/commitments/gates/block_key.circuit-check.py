"""circuit-check's bindings of `verity.definitions.commitments.gates.block_key`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _block_key_roots() -> list[Any]:
    """BlockKey at the deployed binding, 64 tokens of 18 bits (a 208-byte message: two compressions, the second's padding
    constant), and at a toy one, 4 tokens of 16 bits (one compression); and KeyPrint, the print a key carried in from an
    earlier window gets."""
    from verity.circuits.defs import bind
    from verity.definitions.commitments.gates import block_key as BK

    return [bind(BK.BlockKey, B=64, TOK=18), bind(BK.BlockKey, B=4, TOK=16), bind(BK.KeyPrint)]
