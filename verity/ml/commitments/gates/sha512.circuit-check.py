"""circuit-check's bindings of `verity.ml.commitments.gates.sha512`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """`Sha512CompressFolded` with the chaining value SHA-512's IV and the block's 1,024 bits its parameters: a message's first
    compression. The hashing Definitions' bindings reach it at their own patterns, constants where a prefix's midstate or a
    padding is and XORs where an argument is a sum of bits."""
    from verity.ml.circuits.boolean.forms import const_bits
    from verity.ml.circuits.boolean.trace import encode_pattern
    from verity.ml.circuits.defs import bind
    from verity.ml.commitments.gates.sha512 import IV, Sha512CompressFolded

    iv = ["o" if b.c else "z" for v in IV for b in const_bits(v, 64)]
    return [bind(Sha512CompressFolded, P=encode_pattern(iv + [((i,), 0) for i in range(1024)]))]
