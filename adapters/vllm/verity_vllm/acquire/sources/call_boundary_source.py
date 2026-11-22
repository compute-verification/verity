"""Call boundaries (`call_boundaries`, Q_word: a Value two Calls of one module body exchange) on the capture plan: which source claims
each `call_boundary` class.  A claimed class is that source's to acquire (a tap of a Value the runtime already holds); a class no source
claims has no acquisition source, and the Commit is refused naming it (`taps.attach`)."""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from verity_vllm.acquire.sources import int8_source
from verity_vllm.records.capture import CALL_BOUNDARY

FAMILY = "call_boundaries"

#: other sources' claims on the capture plan's `call_boundary` classes (plan class -> the claiming source's `short_name`, or None)
CLAIMS: list[Callable[[Mapping[str, Any]], str | None]] = [int8_source.claim]


def claimed(c: Mapping[str, Any]) -> str | None:
    """The source that claims plan class `c` (`CLAIMS`), or None: no source acquires it."""
    return next((s for f in CLAIMS if (s := f(c))), None)


def unattached(plan: Sequence[Mapping[str, Any]], attached: Collection[str]) -> list[str]:
    """The `call_boundary` classes of the capture plan (`capture_plan.of_build`) whose claimant is not among `attached` (the sources
    the Commit attaches), each named with it: no source acquires them."""
    return [f"{c['op_path']}/{c['member']}: claimed by {s}, a source the Commit does not attach" for c in plan
            if c["capture"] == CALL_BOUNDARY and (s := claimed(c)) is not None and s not in attached]


def sites(plan: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The `call_boundary` classes of the capture plan no source claims."""
    return [c for c in plan if c["capture"] == CALL_BOUNDARY and claimed(c) is None]


__all__ = ["FAMILY", "CLAIMS", "claimed", "unattached", "sites"]
