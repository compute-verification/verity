"""The serving rows' window sampling design: the partition file's sampling design as one-stage's sampling design object (`sampling_design_of`), a rate resolved over the
window's population by `serving_rows.population` (`service.params_sampling_design`), and the verifier's work table for a window under
the work sampling design (`work_table_of`), from the registered compiler's transfer-gate templates (`pipeline.advice_windows`)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.protocols.proofs.spot.verifier.one_stage import draw as DR


def sampling_design_of(sampling_design: str | Mapping[str, Any]) -> dict[str, Any]:
    """The partition file's sampling design as one-stage's sampling design object; a rate's is its kind and ``rate`` until `population` resolves it."""
    kind = (s := DR.parse_strata_rate(sampling_design)) and ("work" if s[0] else "stratified") or DR.parse_rate(sampling_design) and "subset"
    return {"law": kind, "rate": sampling_design} if kind else DR.parse_design(sampling_design) if isinstance(sampling_design, str) else dict(sampling_design)


def work_table_of(sampling_design: Mapping[str, Any], counts: Mapping[str, int], transfer_gates: Sequence[str] = ()) -> dict[str, Any] | None:
    """The verifier's work table (`draw.work_table`) for a window of ``counts`` (template -> its units, in the partition's order)
    under ``sampling_design``: under ``work:K``, each transfer-gate template (the units holding transfer gates, which the registered compiler
    lists) works 0 with its floor all its units, so every one of them is in the sample (`draw.work_k`: selection All, never
    sampled), and every other template works 1 with floor 1, its sample the stratified sampling design's share of K; None for any other sampling design.
    Refused when the compiler lists transfer gates under another sampling design, when one of its transfer-gate templates is not among the
    window's (its units would be left out of the proved set), or when every template is one (nothing is sampled: that window's
    selection is all)."""
    tg = list(dict.fromkeys(transfer_gates))
    if sampling_design.get("law") != "work":
        if tg:
            raise ValueError(f"the compiler lists transfer-gate templates {tg}: every unit of theirs is proved beside the sampled rest, "
                             f"which is the work sampling design's draw (law work:K), not {dict(sampling_design)}")
        return None
    if missing := [t for t in tg if t not in counts]:
        raise ValueError(f"the compiler's transfer-gate template {missing[0]} is not among the partition's templates {list(counts)}: "
                         "its units would be left out of the proved set")
    if all(t in tg for t in counts):
        raise ValueError(f"every template of the window ({list(counts)}) holds transfer gates: nothing is sampled, so its selection is "
                         "all, not a work sampling design")
    return {t: {"work": 0, "floor": int(n)} if t in tg else {"work": 1, "floor": 1} for t, n in counts.items()}
