"""``Flock.Test.Tmpl``'s toy template, which the templated statement's tests and the table read's stager tests parse."""
from __future__ import annotations

from verity_flock import tmpl as T

#: ``Flock.Test.Tmpl``'s unit: two copies of a pair (each two copies of a two-input AND and their sum) at 128, bound to input
#: bits that step by one, its outputs at 256 and 257 and a row at 300
HEADER = "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 2 1 1 2"
AND2 = ["DEF 1 2 1 0", "0 1 2 1 1"]
PAIR = ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 2 3 0 1 0"]
ROWS = ["0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399", "257 1 133 1 399", "300 3 0 1 0 1 400"]


def toy_tmpl(*calls: str, rows: list[str] = ROWS) -> T.Tmpl:
    return T.parse("\n".join([HEADER, *AND2, *PAIR, f"DEF 400 0 {len(rows)} {len(calls)}", *rows, *calls]) + "\n")
