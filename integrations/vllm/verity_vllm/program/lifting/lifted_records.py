"""What a record says about a lifted Program (`registry.lifted`): the post-EOS lag a run header ATTESTS and the lag a
Build DECLARED (read through GP-01's one rule, `program.workload.eos_lag_rows_declared`), and the (query, use) documents of a
lifted VU-partition use (`query.query_artifact`)."""

from __future__ import annotations

import re
from typing import Any

from verity.ml.definitions.registry.lifted import (LIFTED_RULES_VERSION, N_OUT_HISTOGRAM_MAX_SETS, USE_KIND_LIFTED, W_MAX, W_MAX_LIFTED,
                                                 W_MAX_LIFTED_STATUS, check_lag, width_summary_a44)
from verity_vllm.program.workload import eos_lag_rows_declared


LAG_FIELD = "RunHeader.profile.engine_kwargs.async_scheduling"  # the ONE attested field GP-01's eos_lag_rows and this LAG both read
LAG_DECLARED_FIELDS = ("lag.value", "stop_rule.lag", "stop_rule.eos_lag_rows")  # workload_program.json: the current names, then GP-01's older field
LAG_STATIC_FIELD = "workload_fn"  # the digest-bound source: the `lag=<n>` static of `Workload_v1{requests, shared, lag}` in the Program's own spec id
_LAG_STATIC_RE = re.compile(r"^Workload_v1\{.*,lag=(\d+)\}$", re.S)  # the LAST binding of the record, at the top level (the static
# lives in the spec-id string, not in a bindings dict of the Root wrapper)


def lag_from_header(header: dict) -> tuple[int, str]:
    """The record's ATTESTATION of the lag — the RESOLVED `SchedulerConfig.async_scheduling`
    in `profile.engine_kwargs` (True → 1, False → 0), read through the same function GP-01 / GM-01 read (`eos_lag_rows_declared`), from nowhere
    else; an absent or non-boolean attestation is refused by name (never a default of 0 or 1).  Returns `(lag, field)`; the field
    string goes into the report.  This is the attested side; the DECLARED side is `declared_lag_of_row` and the two must agree."""
    try:
        ek = header["profile"]["engine_kwargs"]
        if not isinstance(ek, dict) or "async_scheduling" not in ek:
            raise KeyError("async_scheduling")
    except (KeyError, TypeError):
        raise ValueError(f"{LAG_FIELD} is not attested in the run header: the post-EOS lag is a declared static and is never "
                         f"defaulted — refuse the record") from None
    lag = eos_lag_rows_declared(ek)
    if lag is None:
        raise ValueError(f"{LAG_FIELD} = {ek.get('async_scheduling')!r} is not the resolved boolean (the CLI default `None` is not an attestation)")
    return check_lag(lag), LAG_FIELD


def declared_lag_of_row(workload_program: dict | None) -> tuple[int | None, str | None]:
    """The lag the Build DECLARED as a static of the Program of record, read from the row's
    `workload_program.json` — `lag.value` / `stop_rule.lag`, else GP-01's `stop_rule.eos_lag_rows` (the field older
    Builds wrote from `applicability.engine.async_scheduling`).  Returns `(lag, field)`; `(None, None)` when the Build
    declared none — then the lifted tool never accepts the row (a Program with no lag declared is not a lifted Program of record).
    Nothing is read from a trace and nothing is defaulted.

    Sources, all of one artifact and all required to agree: the DIGEST-BOUND static `lag=<n>` in the Program's spec id (`workload_fn` =
    `Workload_v1{requests=…,shared=…,lag=n}`; the static lives in the spec-id string, the Root wrapper's bindings are `{}`),
    and the sidecar fields `lag.value` / `stop_rule.lag` / `stop_rule.eos_lag_rows` GP-01 writes from the same `wl.lag`.  A sidecar
    edited against the digest (or a digest against the sidecar) is "the declaration disagrees with itself", refused by name — the field
    reported first is the digest-bound one when present."""
    if not isinstance(workload_program, dict):
        return None, None
    found: list[tuple[int, str]] = []
    fn = workload_program.get(LAG_STATIC_FIELD)
    if isinstance(fn, str):
        m = _LAG_STATIC_RE.match(fn.strip())
        if m:
            found.append((check_lag(int(m.group(1))), LAG_STATIC_FIELD + ":lag="))
    for field in LAG_DECLARED_FIELDS:
        cur: Any = workload_program
        for part in field.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, bool):  # a bare boolean is the knob, not the lag: run it through the one rule
            cur = eos_lag_rows_declared({"async_scheduling": cur})
        if isinstance(cur, int) and not isinstance(cur, bool):
            found.append((check_lag(cur), field))
    if not found:
        return None, None
    if len({v for v, _ in found}) > 1:  # one artifact, two values: not a declaration (lag.value == stop_rule.lag == eos_lag_rows)
        raise ValueError("the Build's declaration disagrees with itself: " + ", ".join(f"{f}={v}" for v, f in found) + " — refused by name")
    return found[0]


def lifted_vu_use_document(query: Any, program: Any, verdict: Any, *, w_max_lifted: int = W_MAX_LIFTED, utc: str | None = None,
                           fam: Any = None, histogram_max_sets: int = N_OUT_HISTOGRAM_MAX_SETS) -> tuple[dict, dict]:
    """The (query document, use document) of a LIFTED VU-partition use — `query_artifact.vu_use_document` with the use
    configuration `{"kind": "vu-partition-lifted", "w_max": w_max_lifted, "rules_version": "lifted-v1", "w_max_active_reference": 32}`
    and a `width_summary` {w_max_lifted, max_seen (the validator's), n_out histogram (the boundary
    module's; `fam` = the evaluated family, else the query is evaluated here)}.  The query document is unchanged in shape (no kind, no
    w_max: it binds the query to Lift(P)'s digest).  `gates.gate_G8` refuses this document by name (`use.kind != "vu-partition"`): a
    lifted verdict never reads as an active-reference G8 pass."""
    from verity_vllm.query import query_artifact as QA
    from verity.ml.circuits.query_ast import evaluate
    qdoc, use = QA.vu_use_document(query, program, verdict, w_max=int(w_max_lifted), utc=utc)
    use["use"] = {"kind": USE_KIND_LIFTED, "w_max": int(w_max_lifted), "rules_version": LIFTED_RULES_VERSION,
                  "w_max_active_reference": W_MAX, "w_max_lifted_status": W_MAX_LIFTED_STATUS}
    fam = evaluate(program, query) if fam is None else fam
    use["width_summary"] = width_summary_a44(program, fam, getattr(verdict, "width", verdict), w_max_lifted=w_max_lifted, histogram_max_sets=histogram_max_sets)
    return qdoc, use
