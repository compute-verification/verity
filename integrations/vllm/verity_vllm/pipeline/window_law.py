"""The serving rows' window law: the partition file's law as one-stage's law object (`law_of`), a rate resolved over the
window's population by `serving_rows.population` (`service.params_law`), and the verifier's work table for a window under
the work law (`work_table_of`), from the registered compiler's transfer-gate templates (`pipeline.advice_windows`); and
the work table of a window whose reader templates hold a stated share of wrong units (`reader_floors`, `reader_table`)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from verity.core.protocols.profile import Stratified, Stratum
from verity.core.protocols.spot.one_stage import draw as DR


def law_of(law: str | Mapping[str, Any]) -> dict[str, Any]:
    """The partition file's law as one-stage's law object; a rate's is its kind and ``rate`` until `population` resolves it."""
    kind = (s := DR.parse_strata_rate(law)) and ("work" if s[0] else "stratified") or DR.parse_rate(law) and "subset"
    return {"law": kind, "rate": law} if kind else DR.parse_design(law) if isinstance(law, str) else dict(law)


def work_table_of(law: Mapping[str, Any], counts: Mapping[str, int], transfer_gates: Sequence[str] = ()) -> dict[str, Any] | None:
    """The verifier's work table (`draw.work_table`) for a window of ``counts`` (template -> its units, in the partition's order)
    under ``law``: under ``work:K``, each transfer-gate template (the units holding transfer gates, which the registered compiler
    lists) works 0 with its floor all its units, so every one of them is in the sample (`draw.work_k`: selection All, never
    sampled), and every other template works 1 with floor 1, its sample the stratified sampling design's share of K; None for any other law.
    Refused when the compiler lists transfer gates under another law, when one of its transfer-gate templates is not among the
    window's (its units would be left out of the proved set), or when every template is one (nothing is sampled: that window's
    selection is all)."""
    tg = list(dict.fromkeys(transfer_gates))
    if law.get("law") != "work":
        if tg:
            raise ValueError(f"the compiler lists transfer-gate templates {tg}: every unit of theirs is proved beside the drawn rest, "
                             f"which is the work law's draw (law work:K), not {dict(law)}")
        return None
    if missing := [t for t in tg if t not in counts]:
        raise ValueError(f"the compiler's transfer-gate template {missing[0]} is not among the partition's templates {list(counts)}: "
                         "its units would be left out of the proved set")
    if all(t in tg for t in counts):
        raise ValueError(f"every template of the window ({list(counts)}) holds transfer gates: nothing is sampled, so its selection is "
                         "all, not a work law")
    return {t: {"work": 0, "floor": int(n)} if t in tg else {"work": 1, "floor": 1} for t, n in counts.items()}


def reader_floors(units: Mapping[str, int], eps: Fraction, delta: Fraction) -> dict[str, int]:
    """A floor for each template of ``units`` (template -> its units) under which a sample leaves at most ``eps`` of their
    N units wrong except with probability ``delta``: template s's floor is ceil(k n_s / N), the same share of each, at the
    least k whose count bound over these templates alone (`profile.Stratified.count_bound`, the record's bound, Lean
    `Spot.Bound.countBound`) is at most eps N. The work sampling design samples each stratum at its floor or more
    (`profile.work_k`), each independently, and a larger sample of a stratum only lowers the bound, so the bound holds
    under ``work:K`` at any K and whatever the window's other templates sample. ``eps`` and ``delta`` are exact
    (`fractions.Fraction`), 0 < eps <= 1 and 0 < delta < 1; every floor is at least 1."""
    n = sum(units.values())
    if not units or min(units.values()) < 1:
        raise ValueError(f"each template has a unit to sample, not {dict(units)}")
    if not (isinstance(eps, Fraction) and isinstance(delta, Fraction) and 0 < eps <= 1 and 0 < delta < 1):
        raise ValueError(f"eps and delta are exact fractions with 0 < eps <= 1 and 0 < delta < 1, not {eps!r} and {delta!r}")

    def floors(k: int) -> dict[str, int]:
        return {t: -(-k * n_s // n) for t, n_s in units.items()}

    def held(k: int) -> bool:
        f = floors(k)
        return Stratified(tuple(Stratum(t, n_s, f[t]) for t, n_s in units.items())).count_bound(delta) <= eps * n

    lo, hi = 0, n  # at k = N every unit is sampled, and the bound is 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if held(mid) else (mid, hi)
    return floors(hi)


def reader_table(counts: Mapping[str, int], readers: Iterable[str], eps: Fraction,
                 delta: Fraction) -> dict[str, dict[str, int]]:
    """The verifier's work table (`draw.work_table`) for a window of ``counts`` (template -> its units) whose templates
    ``readers`` hold ``eps`` at ``delta``: every template works 1, each reader at its floor of `reader_floors` over the
    readers' units and every other template at floor 1. Refused where a reader is not among the window's templates, or
    there is none."""
    readers = list(dict.fromkeys(readers))
    if not readers or any(t not in counts for t in readers):
        raise ValueError(f"the reader templates {readers} are templates of the window, at least one: its templates are "
                         f"{list(counts)}")
    floors = reader_floors({t: int(counts[t]) for t in readers}, eps, delta)
    return {t: {"work": 1, "floor": floors.get(t, 1)} for t in counts}


__all__ = ["law_of", "reader_floors", "reader_table", "work_table_of"]
