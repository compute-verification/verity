"""How `verity bench exfiltration` (`verity_bench.exfiltration`) writes a figure, a share, a plan and a list, which its
parts and its print share."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from verity_bench.exfiltration_common import PLAN_FILE


def _n(x: Any, none: str = "none") -> str:
    """A number to four significant figures, text as itself, and ``none`` for None."""

    if x is None:
        return none
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    if abs(x) >= 1e9 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.4g}"


def _p(x: float) -> str:
    """A probability, with its power of two where it is under 1."""

    return f"{_n(x)} (2^{math.log2(x):.4g})" if 0 < x < 1 else _n(x)


def _pct(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.4g}%"


def _part(x: float | None) -> str:
    return "none" if x is None else f"{100 * x:.3f}%"


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head), *("| " + " | ".join(r) + " |" for r in body)]


def _plan(priced: Mapping) -> str:
    """Which plan an account of the proving is: the rows' own at a pinned budget, or a plan of record at its own."""

    if priced["of"] == "rows":
        return f"`split`: the row's own plan at the pinned budget nearest, β = {_part(priced['beta'])}"
    return f"`{PLAN_FILE}`: the row's plan at β = {_part(priced['beta'])}"


def _listed(items: Sequence[str]) -> str:
    """Items in a sentence."""

    return " and ".join(items) if len(items) < 3 else ", ".join(items[:-1]) + f" and {items[-1]}"


def _names(names: Sequence[str]) -> str:
    """Rows' names in a sentence."""

    return _listed([f"`{name}`" for name in names])


def _on(stop: Mapping) -> str:
    """Where the sums that take the output count at 0 take it so, by the reader's answer as the document holds it
    (``stop``, `build`'s `attester`'s `stop`): the execution phases of a link it holds, said in the calculator's
    words, and nothing where it holds every one."""

    return "" if stop["below"] else f" on {_n(stop['held_phases'])} execution phase(s) of a link"
