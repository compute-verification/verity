"""Where PoUS's bandwidth audit's time over plain goes, by part, from the spans its breakdown spec's arms record
(`llama31-8b-fp8-1gpu-bandwidth-ablation`).

Memory accounting's spans arms (`benchmarks/vllm_encoded/step_spans.py`) time each step by part, and each timed
rep's `extras` keeps a run's parts a step (`span_per_step.<kind>.<part>_ms`, `graph` for vLLM's replayed decode steps and
`eager` for prefill and mixed steps) and its steps. A part's time an output token is its time a step times the kind's
steps over the run's generated tokens. The top-up runs beside the step, so its cost is in no part: it shows only as the
other parts' stretch, which the spans arms with the top-up off (`verity_bench.vllm.bandwidth_ablation`) separate.

The audit's time over plain (the spans arms' time per output token, paired by rep) is split into rows that sum to it:

~~~text
top-up contention     (audit-spans − audit-spans-no-topup) − (plain-spans − plain-spans-no-topup), time per output token
decode <part>         a decode step's part, audit-spans-no-topup − plain-spans-no-topup, an output token
decode, the rest      every other decode part (model, pass, between, append_wait, kv_receipts, ...), the same
prefill steps         every part of the eager steps, the same
outside every span    (audit-spans-no-topup − plain-spans-no-topup) less every part: time before a run's first mark
~~~

The credited lines' own difference (`bandwidth-audit` − `bandwidth-plain`, spans off) is beside it: the spans' marks are
in the step's time, so it is the audit's cost, and the rows say where the spans arms' time went. No row is a ratio, and
nothing here is a gate: the harness's verdicts never read spans.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_bench.vllm.runner import summarize

SPEC = "llama31-8b-fp8-1gpu-bandwidth-ablation"
PLAIN, AUDIT = "bandwidth-plain", "bandwidth-audit"
PLAIN_SPANS, AUDIT_SPANS = "bandwidth-plain-spans", "bandwidth-audit-spans"
PLAIN_OFF, AUDIT_OFF = "bandwidth-plain-spans-no-topup", "bandwidth-audit-spans-no-topup"
ARMS = (PLAIN, AUDIT, PLAIN_SPANS, AUDIT_SPANS, PLAIN_OFF, AUDIT_OFF)
SPANS_ARMS = (PLAIN_SPANS, AUDIT_SPANS, PLAIN_OFF, AUDIT_OFF)
KINDS = ("graph", "eager")
PREFIX = "span_per_step."
#: a decode step's parts given rows of their own, with what each holds on the audit (the README's span list)
DECODE_ROWS = {
    "linear": "the weights' codec decode, FP8 MMA and sketch receipts (one fused launch)",
    "lm_head": "the lm_head's decode, BF16 MMA and receipt",
    "commit": "the last pass's receipts closed: KV slots' host read, compaction, step tree",
    "attention": "paged attention with the KV words decoded inside it",
}
NOT_PARTS = ("step", "append")


def ms_cell(summary: Mapping[str, Any]) -> str:
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}" if ci is None else f"{summary['mean']:.3f} ± {(ci[1] - ci[0]) / 2:.3f}"


def _reps(row: Mapping[str, Any], arm: str) -> list[Mapping[str, Any]]:
    return sorted(row["arms"][arm]["reps"], key=lambda rep: rep["rep"])


def _tpot_ms(rep: Mapping[str, Any]) -> float:
    return float(rep["metrics"]["time_per_output_token_s"]) * 1e3


def parts(rep: Mapping[str, Any], kind: str) -> dict[str, float]:
    """A rep's main-stream parts of `kind`, ms a step (empty when the run had no step of that kind)."""

    extras, head = rep.get("extras") or {}, f"{PREFIX}{kind}."
    return {key[len(head):-len("_ms")]: float(value) for key, value in extras.items()
            if key.startswith(head) and key.endswith("_ms") and key[len(head):-len("_ms")] not in NOT_PARTS}


def per_token(rep: Mapping[str, Any], kind: str) -> dict[str, float]:
    """A rep's parts of `kind`, ms an output token: ms a step × the kind's steps / the run's generated tokens."""

    steps = float((rep.get("extras") or {}).get(f"{PREFIX}{kind}.steps", 0.0))
    return {part: ms * steps / int(rep["generated_tokens"]) for part, ms in parts(rep, kind).items()}


def _diff(audit: Mapping[str, Mapping[str, float]], plain: Mapping[str, Mapping[str, float]], kind: str, part: str) -> float:
    return audit[kind].get(part, 0.0) - plain[kind].get(part, 0.0)


def _spans_recorded(row: Mapping[str, Any], arm: str) -> None:
    for rep in _reps(row, arm):
        if not any(parts(rep, kind) for kind in KINDS):
            raise ValueError(f"{arm} at {row['point']['id']} rep {rep['rep']} recorded no spans "
                             f"(no {PREFIX}* extras): it isn't a spans arm")


def attribution(row: Mapping[str, Any]) -> dict[str, Any]:
    """One point's audit-over-plain rows (ms an output token, each a summary over the paired reps) and their sum."""

    for arm in ARMS:
        if arm not in row["arms"]:
            raise ValueError(f"the point {row['point']['id']} has no timed {arm}")
    for arm in SPANS_ARMS:
        _spans_recorded(row, arm)
    reps = {arm: _reps(row, arm) for arm in ARMS}
    n = {len(value) for value in reps.values()}
    if len(n) != 1:
        raise ValueError(f"the arms at {row['point']['id']} have different rep counts: paired rows need the same reps")
    names = [f"decode {part}" for part in DECODE_ROWS] + ["decode, the rest", "prefill steps", "outside every span"]
    values: dict[str, list[float]] = {name: [] for name in ["credited", "total", "top-up contention", *names]}
    for i in range(n.pop()):
        tpot = {arm: _tpot_ms(reps[arm][i]) for arm in ARMS}
        values["credited"].append(tpot[AUDIT] - tpot[PLAIN])
        values["total"].append(tpot[AUDIT_SPANS] - tpot[PLAIN_SPANS])
        values["top-up contention"].append((tpot[AUDIT_SPANS] - tpot[AUDIT_OFF]) - (tpot[PLAIN_SPANS] - tpot[PLAIN_OFF]))
        audit = {kind: per_token(reps[AUDIT_OFF][i], kind) for kind in KINDS}
        plain = {kind: per_token(reps[PLAIN_OFF][i], kind) for kind in KINDS}
        decode = set(audit["graph"]) | set(plain["graph"])
        for part in DECODE_ROWS:
            values[f"decode {part}"].append(_diff(audit, plain, "graph", part))
        values["decode, the rest"].append(sum(_diff(audit, plain, "graph", part) for part in decode - set(DECODE_ROWS)))
        values["prefill steps"].append(sum(_diff(audit, plain, "eager", part)
                                           for part in set(audit["eager"]) | set(plain["eager"])))
        spanned = sum(values[name][-1] for name in names[:-1])
        values["outside every span"].append(tpot[AUDIT_OFF] - tpot[PLAIN_OFF] - spanned)
    total = summarize(values["total"])
    rows: list[dict[str, Any]] = []
    for name in ["top-up contention", *names]:
        summary = summarize(values[name])
        rows.append({"row": name, "ms_per_token": summary,
                     "share": summary["mean"] / total["mean"] if total["mean"] else None})
    return {"credited": summarize(values["credited"]), "total": total, "rows": rows,
            "sum": sum(row["ms_per_token"]["mean"] for row in rows)}


