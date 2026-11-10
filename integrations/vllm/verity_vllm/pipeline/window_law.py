"""The serving rows' window law: the partition file's law as one-stage's law object (`law_of`), a rate resolved over the
window's population by `serving_rows.population` (`service.params_law`), and the verifier's work table for a window under
the work law (`work_table_of`), from the registered compiler's transfer-gate templates (`pipeline.advice_windows`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

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
