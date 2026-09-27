"""The draw: the verifier's own coins pick the units to prove, sent in the clear after registration and before the session.

    {"law": "subset", "population": N, "k": K, "units": [<ascending unit indices>]}
    {"law": "bernoulli", "population": N, "p": "NUM/DEN", "units": [...]}

This is the Lean verifier's `Flock.Draw` object and M0's `unit_draw`. The verifier samples it from the operating system's
randomness (Lean `flock-verify draw`, or M0's live `serve --draw`); nothing here samples, and no draw is derived from a
commitment. `check_draw` holds a draw to the registration and to the verifier's own law, and `check_proved` holds the
proved statement's units to the draw.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping, Sequence

from .verdict import Verdict, _Collect


def parse_law(text: str) -> dict[str, Any]:
    """`subset:K` or `bernoulli:NUM/DEN` (M0's `serve --draw` syntax) as a law object."""
    kind, _, arg = text.partition(":")
    if kind == "subset" and arg.isdigit():
        return {"law": "subset", "k": int(arg)}
    if kind == "bernoulli":
        num, _, den = arg.partition("/")
        if num.isdigit() and den.isdigit() and 0 < int(den) and int(num) <= int(den):
            return {"law": "bernoulli", "p": f"{int(num)}/{int(den)}"}
    raise ValueError(f"a law is subset:K or bernoulli:NUM/DEN, not {text!r}")


def law_of(draw: Mapping[str, Any]) -> dict[str, Any]:
    return {"law": "subset", "k": draw["k"]} if draw.get("law") == "subset" else {"law": "bernoulli", "p": draw.get("p")}


def check_draw(draw: Any, registration: Mapping[str, Any], law: Mapping[str, Any]) -> Verdict:
    """The draw is well formed, over the registered population, under the verifier's law."""
    c = _Collect()
    if not isinstance(draw, Mapping) or draw.get("law") not in ("subset", "bernoulli"):
        c.fail("not-a-draw", draw)
        return c.verdict()
    keys = {"law", "population", "units", "k" if draw["law"] == "subset" else "p"}
    if set(draw) != keys:
        c.fail("keys", sorted(set(draw) ^ keys))
        return c.verdict()
    n, units = draw["population"], draw["units"]
    if law_of(draw) != dict(law):
        c.fail("law", {"drawn": law_of(draw), "verifier": dict(law)})
    if n != registration["population"]:
        c.fail("population", {"drawn": n, "registered": registration["population"]})
    if not isinstance(units, list) or not all(isinstance(u, int) and not isinstance(u, bool) for u in units):
        c.fail("units", units)
        return c.verdict()
    if any(b <= a for a, b in zip(units, units[1:])):
        c.fail("units-order", units[:16])
    if units and not (0 <= units[0] and units[-1] < n):
        c.fail("units-range", [u for u in units if not 0 <= u < n][:16])
    if draw["law"] == "subset" and len(units) != draw["k"]:
        c.fail("units-count", {"units": len(units), "k": draw["k"]})
    if draw["law"] == "bernoulli":
        num, _, den = str(draw["p"]).partition("/")
        if not (num.isdigit() and den.isdigit() and 0 < int(den) and Fraction(int(num), int(den)) <= 1):
            c.fail("p", draw["p"])
    return c.verdict()


def check_proved(draw: Mapping[str, Any], proved_units: Sequence[int], instances: int) -> Verdict:
    """The proved statement's instances are exactly the drawn units, in draw order."""
    c = _Collect()
    if instances != len(draw["units"]):
        c.fail("instances", {"statement": instances, "drawn": len(draw["units"])})
    if list(proved_units) != list(draw["units"]):
        c.fail("units", {"statement": list(proved_units)[:16], "drawn": list(draw["units"])[:16]})
    return c.verdict()


__all__ = ["check_draw", "check_proved", "law_of", "parse_law"]
