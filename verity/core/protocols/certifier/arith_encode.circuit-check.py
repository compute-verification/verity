"""circuit-check's bindings of `verity.core.protocols.certifier.arith_encode`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _network_arith_encode_roots() -> list[Any]:
    """Advice v1's encoder (`verity.core.protocols.certifier.arith_encode`). The counting variant, which the
    composition takes: the step at K = 18 (every symbol's scalings) and CW = 20 for decisions (BIN = 1) and for general
    symbols (BIN = 0), and at K = 3, CW = 4, its smallest that reaches a refusal for too many scalings and for an
    overflowing count; its finish at CW = 20, without the header and with it at NB = 21 (v1's largest pool, 2^20 blocks);
    and the book's model at the decisions' three contexts and at one (async's lag bits). The bit-checking variant: the
    step at K = 18 for both kinds and at K = 3, and the finish. ENC-REPLAY's model at plain contexts (C = 3, NW = 24) and
    at its smallest (C = 1, NW = 15)."""
    from verity.core.protocols.certifier.arith_encode import (book_kt_unit, count_finish_unit, count_unit, finish_unit,
                                                              kt_unit, step_unit)

    return [count_unit(18, 1, 20), count_unit(18, 0, 20), count_unit(3, 1, 4), count_finish_unit(20, 0),
            count_finish_unit(20, 21), book_kt_unit(3), book_kt_unit(1), step_unit(18, 1), step_unit(18, 0),
            step_unit(3, 1), finish_unit(), kt_unit(3, 24), kt_unit(1, 15)]
