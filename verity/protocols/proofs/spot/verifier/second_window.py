"""Round 2's reader over the joined record and its ranks' check, as the verifier's `Spot.SecondWindow` and `Spot.Ranks`
define them: a two-stage run's second-stage sessions held to the joined circuit's record (its ``ranks``, ``wires`` and
``boundary_reads``, `verity.commitments.windows.interface`) and to each rank's registration, with no interior
registration. The rule, the codes and why the interior registration's checks go are the Lean's (its module docstrings).

A rank's registration as round 2 reads it is ``{"rank", "digest", "circuit", "run", "members"}``, each member
``{"key", "base", "units", "outs"}`` with ``outs`` its output ports' leaves, ``[[port, leaves], ...]``. What the Lean has
not built yet refuses here too, with its code and `NOT_BUILT`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

#: how a rank's registered names are its own (`Spot.Ranks.PREFIX`)
PREFIX = "r<rank>/"
#: the detail of a check whose body is not built: it refuses (`Spot.Ranks.NOT_BUILT`)
NOT_BUILT = "not yet built: refused"
#: the ranks' checks' codes, in the order they run (`Spot.Ranks.CODES`)
RANK_CODES = ("ranks-shape", "ranks-window", "ranks-run", "ranks-members", "ranks-units", "ranks-wires", "ranks-reads")
#: `Spot.SecondStage`'s per-session codes this reader reuses, read against a rank member (`Spot.SecondWindow.SESSION_CODES`)
SESSION_CODES = ("second-header", "second-population", "second-roots", "second-sample", "second-read", "second-values",
                 "second-refs", "second-tables", "second-verified")
WINDOW_WHY = "a served rank record's digest is not its rank's window, or its registration binds another digest or other bytes"
RUN_WHY = "the ranks' registrations are not one serve's, one a rank at <root>/rank<r>, each in the log before the challenge"
MEMBERS_WHY = ("a rank's members of a template don't tile its units in key order, name no template of the rank's, or "
               "disagree on an output port's leaves a unit")
UNITS_WHY = "a template's units over the ranks are not the joined record's count of it"
#: the `two-stage-form` refusal's detail (`Spot.SecondStage.FORM_WHY`)
FORM_WHY = ("refused: a two-stage run, its second-stage sessions over an interior registration: a statement form the "
            "soundness proof does not cover")
_HEX512 = re.compile(r"[0-9a-f]{128}")
_DIGITS = re.compile(r"[0-9]+")


def _nat(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def ranks_of(field: Any) -> list[tuple[str, dict[str, int], int]] | None:
    """`Spot.Ranks.Ranks.ofJson?`: the ranks ``(window, units, wires)`` of the record's ``ranks`` field, None for one
    ``ranks-shape`` refuses."""
    if not isinstance(field, Mapping) or sorted(field) != ["of", "prefix"] or field["prefix"] != PREFIX:
        return None
    of = field["of"]
    if not isinstance(of, list) or not of:
        return None
    out = []
    for r in of:
        if (not isinstance(r, Mapping) or sorted(r) != ["units", "window", "wires"] or not isinstance(r["window"], str)
                or not _HEX512.fullmatch(r["window"]) or not isinstance(r["units"], Mapping)
                or not all(_nat(n) for n in r["units"].values()) or not _nat(r["wires"])):
            return None
        out.append((r["window"], dict(r["units"]), r["wires"]))
    return out


def segment(ranks: Sequence[tuple[str, Mapping[str, int], int]], template: str, r: int) -> tuple[int, int]:
    """`Spot.Ranks.Ranks.segment`: rank ``r``'s units of ``template`` in the joined circuit, ``[lo, hi)``."""
    lo = sum(q[1].get(template, 0) for q in ranks[:r])
    return lo, lo + (ranks[r][1].get(template, 0) if r < len(ranks) else 0)


def index_of(template: str, key: str) -> int | None:
    """`Spot.Ranks.indexOf`: ``j`` where ``key`` is ``<template>/c<j>``."""
    rest = key[len(template) + 2:] if key.startswith(template + "/c") else ""
    return int(rest) if _DIGITS.fullmatch(rest) else None


def members_of(reg: Mapping[str, Any], template: str) -> list[tuple[int, Mapping[str, Any]]]:
    """`Spot.Ranks.membersOf`: ``reg``'s members of ``template`` with their indices, in index order."""
    ms = [(j, h) for h in reg["members"] if (j := index_of(template, h["key"])) is not None]
    return sorted(ms, key=lambda x: x[0])


def _tiles(ms: Sequence[tuple[int, Mapping[str, Any]]], n: int) -> bool:
    nxt = pos = 0
    for j, h in ms:
        if j != nxt or h["base"] != pos or h["units"] < 1:
            return False
        nxt, pos = nxt + 1, pos + h["units"]
    return pos == n


def _leaves_agree(ms: Sequence[tuple[int, Mapping[str, Any]]]) -> bool:
    if not ms:
        return True
    h0 = ms[0][1]
    return all([p for p, _ in h["outs"]] == [p for p, _ in h0["outs"]]
               and all(a[1] * h0["units"] == b[1] * h["units"] for a, b in zip(h["outs"], h0["outs"], strict=False))
               for _j, h in ms)


def _reg(registrations: Sequence[Mapping[str, Any]], r: int) -> Mapping[str, Any] | None:
    return next((g for g in registrations if g["rank"] == r), None)


def member_of(ranks: Sequence[tuple[str, Mapping[str, int], int]], registrations: Sequence[Mapping[str, Any]],
              template: str, unit: int) -> tuple[int, str, int, int, int] | None:
    """`Spot.Ranks.memberOf`: the rank member ``(rank, template, j, base, units)`` of ``registrations`` that holds unit
    ``unit`` of ``template`` in the joined circuit."""
    r = next((r for r in range(len(ranks)) if segment(ranks, template, r)[0] <= unit < segment(ranks, template, r)[1]),
             None)
    reg = _reg(registrations, r) if r is not None else None
    if reg is None:
        return None
    v = unit - segment(ranks, template, r)[0]
    x = next(((j, h) for j, h in members_of(reg, template) if h["base"] <= v < h["base"] + h["units"]), None)
    return (r, template, x[0], x[1]["base"], x[1]["units"]) if x is not None else None


def rank_failures(record: Mapping[str, Any], records: Sequence[Mapping[str, Any]], served: Sequence[tuple[str, str]],
                  registrations: Sequence[Mapping[str, Any]], root: str,
                  logged: Callable[[str], bool]) -> list[tuple[str, Any]]:
    """`Spot.Ranks.failures`: ``ranks-shape`` first, then ``ranks-window``, ``ranks-run``, ``ranks-members``,
    ``ranks-units``, ``ranks-wires`` and ``ranks-reads`` of the joined ``record`` against its ranks' records, served
    records and registrations at the serve's ``root``."""
    ranks = ranks_of(record.get("ranks"))
    if ranks is None:
        return [("ranks-shape", NOT_BUILT)]
    n, out = len(ranks), []
    if not (len(served) == n and all((g := _reg(registrations, r)) is not None and served[r][0] == ranks[r][0]
                                     and g["digest"] == served[r][0] and g["circuit"] == served[r][1] for r in range(n))):
        out.append(("ranks-window", WINDOW_WHY))
    if not (len(registrations) == n and all(
            len(gs := [g for g in registrations if g["rank"] == r]) == 1 and gs[0]["run"] == f"{root}/rank{r}"
            and logged(gs[0]["circuit"]) for r in range(n))):
        out.append(("ranks-run", RUN_WHY))
    every = [list(ranks[r][1]) for r in range(n)]
    if not all((g := _reg(registrations, r)) is not None
               and all(any(index_of(t, h["key"]) is not None for t in every[r]) for h in g["members"])
               and all(_tiles(members_of(g, t), k)
                       and _leaves_agree([x for q in registrations for x in members_of(q, t)])
                       for t, k in ranks[r][1].items()) for r in range(n)):
        out.append(("ranks-members", MEMBERS_WHY))
    ts = record.get("templates")
    named = ([(x[0], x[1]) for x in ts if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and _nat(x[1])]
             if isinstance(ts, list) else None)
    if (named is None or len(named) != len(ts) or not all(t in dict(named) for q in ranks for t in q[1])
            or not all(sum(q[1].get(t, 0) for q in ranks) == k for t, k in named)):
        out.append(("ranks-units", UNITS_WHY))
    return out + [("ranks-wires", NOT_BUILT), ("ranks-reads", NOT_BUILT)]


def failures(allow: bool, record: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
             served: Sequence[tuple[str, str]], registrations: Sequence[Mapping[str, Any]], root: str,
             logged: Callable[[str], bool], sample: Sequence[int],
             statements: Sequence[Mapping[str, Any]]) -> list[tuple[str, Any]]:
    """`Spot.SecondWindow.failures`: the ranks' failures, then (with no reader of the sampled members built yet)
    ``ranks-members``; ``two-stage-form`` last while nothing else fails and the form is off."""
    found = rank_failures(record, records, served, registrations, root, logged) + [("ranks-members", NOT_BUILT)]
    return [("two-stage-form", FORM_WHY)] if not found and not allow else found
