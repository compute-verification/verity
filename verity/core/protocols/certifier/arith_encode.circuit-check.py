"""circuit-check's bindings of `verity.core.protocols.certifier.arith_encode`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_arith_encode_roots() -> list[Any]:
    """Advice v1's encoder (`verity.core.protocols.certifier.arith_encode`). The counting variant, which the
    composition takes: the step at K = 18 (every symbol's scalings) and CW = 20 for decisions (BIN = 1) and for general
    symbols (BIN = 0), and at K = 3, CW = 4, its smallest that reaches a refusal for too many scalings and for an
    overflowing count; its finish at CW = 20; the book's model at the decisions' three contexts and at one (async's lag bits); and a window's charge at N = 2
    slots (its smallest with n_arrived's offset symbol) and at the compile slice's P·A = 14, each within the
    whole-Definition lowering budget (N = 64, the corpus difftest's, is 178,978 ANDs and isn't lowered whole). The
    bit-checking variant: the step at K = 18 for both kinds and at K = 3, and the finish. The KT model at plain
    contexts (C = 3, NW = 24) and at its smallest (C = 1, NW = 15)."""
    from verity.core.protocols.certifier.arith_encode import (book_kt_unit, charge_unit, count_finish_unit, count_unit,
                                                              finish_unit, kt_unit, step_unit)

    return [count_unit(18, 1, 20), count_unit(18, 0, 20), count_unit(3, 1, 4), count_finish_unit(20),
            book_kt_unit(3), book_kt_unit(1), charge_unit(2), charge_unit(14),
            step_unit(18, 1), step_unit(18, 0), step_unit(3, 1), finish_unit(), kt_unit(3, 24), kt_unit(1, 15)]
