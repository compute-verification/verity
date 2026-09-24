"""Table 1-4 rows (contract section 9) and the PASS derivation rule (section 10).

Field names are exactly the contract's.  Every row carries the five common fields
``experiment_id``, ``case``, ``profile_id``, ``worker``, ``adapter_revision``.  Rows are validated
against the specs below (missing, unknown or mistyped fields are errors) and written as JSONL
under ``fixtures/results/<experiment_id>/table{1,2,3,4}.jsonl``.

:func:`verdict` derives ``result`` and ``first_blocker`` of a Table 1 row from its check fields
(and, when given, the case's Table 2 rows) so that nobody hand-assigns ``PASS``; the renderer
recomputes it and flags disagreement.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RESULTS = ("PASS", "PARTIAL", "FAIL", "UNSUPPORTED", "BLOCKED", "NOT_RUN")
BLOCKER_KINDS = ("UNSUPPORTED", "BLOCKED", "NOT_RUN")
CHECKERS = ("same_runtime", "independent", "none")

COMMON_FIELDS = ("experiment_id", "case", "profile_id", "worker", "adapter_revision")

# type specs: "int" | "float" | "str" | "bool" | "list" | "any" | "T|None" | tuple(enum) | dict
Spec = Any

COUNT = {"exact": "int", "attempted": "int"}

TABLE1_SPEC: dict[str, Spec] = {
    "workload": {
        "requests": "int",
        "generated_positions": "int",
        "input_len_range": "list",
        "peak_concurrency_achieved": "int",
    },
    "production_changes": "str",
    "replay_restrictions": "list",
    "token_selection": COUNT,
    "boundary_noninterference": {
        "identical": "bool",
        "compared": "int",
        "mismatched_scalars": "int",
        "reference_method": "str",
    },
    "required_coverage": {"bound": "int", "required": "int"},
    "authentication": {"valid": "int", "attempted": "int"},
    "relation_checks": {"same_runtime": COUNT, "independent": COUNT},
    "structural": {"result": "str", "failing_invariant": "str|None"},
    "negative": {"rejected": "int", "attempted": "int"},
    "result": RESULTS,
    "first_blocker": "str|None",
}

# Optional Table 1 field (contract §15.8, witness of activation): what the recorded production
# pass actually exercised (prefix-cache hits per request, preemptions, prefill chunks, effective
# execution mode). Absent in rows written before it existed; validated when present.
TABLE1_OPTIONAL: dict[str, Spec] = {
    "workload_exercised": "any",
    "challenge": "any",
    # W1cF (audit-compiled-pass.md F4/F5/F7/F2/F3/F10): rows written before these existed lack them.
    "relation_errors": "any",  # {"count": int, "first": str|None}: checker calls that raised (a rejection)
    "negatives_not_run": "list",  # applicable negative ids with 0 attempted mutations (excluded from `negative`)
    "qualifications": "list",  # environment / identity differences that qualify a PASS without failing it
    "bindings": "any",  # structural bindings checked (kernel set, description digest, ...)
    # WB2 (2026-09-07, GUIDANCE §15.2 / IR §H.1): what kind of evidence the row's verification is; see
    # :func:`verification_evidence`.  ``{"label": str, "verifier": str|None, "report": str|None, "note": str|None}``.
    "verification_evidence": "any",
}

# ``verification_evidence.label`` values.  A standalone transcript proves correct derivation conditional
# on the verifier's interaction / order; an auditor's receipt establishes that interaction for the audited
# row; no timestamping / beacon infrastructure is required for the current POC (theory ruling on WA's
# finding 2, `docs/verity-ir-2026-09-07.md` §H.1, GUIDANCE §15.2).
EVIDENCE_TRANSCRIPT = "transcript-verified (security-mode staging)"  # the prover's own pod, its own seed and transcript
EVIDENCE_INDEPENDENT = "independently verified"  # an auditor re-verified from the archive with its own seed
EVIDENCE_DIAGNOSTIC = "diagnostic challenge (public seed)"  # not security evidence
EVIDENCE_DERIVED = "derived bundle (CPU-only structural checks; no challenge)"  # verity_vllm.check.poc_derive_logprobs
EVIDENCE_CPU_ONLY = "CPU-only structural checks (no challenge)"  # verity_vllm.check.poc_verify_cpu on a captured bundle
EVIDENCE_LABELS = (EVIDENCE_TRANSCRIPT, EVIDENCE_INDEPENDENT, EVIDENCE_DIAGNOSTIC, EVIDENCE_DERIVED, EVIDENCE_CPU_ONLY)
# ``challenge`` (GUIDANCE C1, ``verity_vllm.challenge``): how the challenge randomness was obtained.
# ``challenge.mode`` is ``"security"`` (verifier-owned staged randomness, unqualified verdict) or
# ``"diagnostic"`` (public seed; the verdict is rendered qualified, see :func:`result_label`).  Absent in
# rows written before the staged challenge existed (their seed was a raw ``--seed``: treat as diagnostic).
OPTIONAL: dict[int, dict[str, Spec]] = {1: TABLE1_OPTIONAL}

TABLE2_SPEC: dict[str, Spec] = {
    "op_id": "str",
    "phase": "str",
    "shape": "any",
    "dtype": "str",
    "layout_or_quant": "str|None",
    "runtime_impl": "str",
    "primitive_semantics": "str",
    "decomposition": "str",
    "interface": {"input_bytes": "int", "output_bits": "int", "range": "any|None"},
    "origin": ("captured", "derived"),
    "binding": {"present": "int", "unresolved": "int"},
    "checker": CHECKERS,
    "natural_checks": COUNT,
    "stress_checks": {"exact": "int", "attempted": "int", "domain": "str|None"},
    "mutations": {"rejected": "int", "attempted": "int"},
    "checker_work": {"seconds": "float", "flops_or_bytes": "float", "includes_full_op": "bool"},
    "status": "str",
    "limitation": "str|None",
}

TABLE3_SPEC: dict[str, Spec] = {
    "negative_id": "str",
    "attempted": "int",
    "rejected": "int",
    "unexpectedly_accepted": "int",
    "reproducer": "str|None",
}

TABLE4_SPEC: dict[str, Spec] = {
    "partition_id": "str",
    "instrumentation_mode": "str",
    "production_overhead_ratio": "float|None",
    "replay_commit_overhead_ratio": "float|None",
    "replay_vs_production_gpu_work": "float|None",
    "transparent_check_cost": {"cpu_s": "float|None", "gpu_s": "float|None"},
    "peak_replay_memory": {
        "device_gb": "float|None",
        "host_gb": "float|None",
        "delta_vs_reference": "float|None",
    },
    "committed_logical_bytes": "int|None",
    "processed_bytes": {
        "hashed": "int|None", "copied": "int|None", "retained": "int|None", "emitted": "int|None",
    },
    "description_size_bytes": "int|None",
    "constructor_inputs_bytes": "int|None",
    "logical_size": {"gates": "int|None", "vus": "int|None", "rus": "int|None"},
    "lookup_cost_s": "float|None",
    "validation_cost_s": "float|None",
    "moe_padding": {
        "inactive_units": "int|None",
        "inactive_checks": "int|None",
        "inactive_work_s": "float|None",
        "inactive_bytes": "int|None",
    },
    "restrictions": "list",
    "spend": {"gpu_hours": "float|None", "usd": "float|None"},
}  # fmt: skip

SPECS: dict[int, dict[str, Spec]] = {1: TABLE1_SPEC, 2: TABLE2_SPEC, 3: TABLE3_SPEC, 4: TABLE4_SPEC}


class RowError(ValueError):
    pass


SEVERITY = {"PASS": 0, "PARTIAL": 1, "FAIL": 2}


def overclaims(claimed: str, derived: str) -> bool:
    """True when ``claimed`` is a more favourable verdict than the rule-derived one.

    Only the three evidence verdicts compare; ``UNSUPPORTED``/``BLOCKED``/``NOT_RUN`` are never
    an over-claim (they claim nothing) and a rule that yields one of them constrains nothing.
    """
    if claimed not in SEVERITY or derived not in SEVERITY:
        return False
    return SEVERITY[claimed] < SEVERITY[derived]


def _check_type(path: str, value: Any, spec: Spec, problems: list[str]) -> None:
    if isinstance(spec, dict):
        if not isinstance(value, Mapping):
            problems.append(f"{path}: expected an object")
            return
        for key in spec:
            if key not in value:
                problems.append(f"{path}.{key}: missing")
        for key in value:
            if key not in spec:
                problems.append(f"{path}.{key}: unknown field")
        for key, sub in spec.items():
            if key in value:
                _check_type(f"{path}.{key}", value[key], sub, problems)
        return
    if isinstance(spec, tuple):
        if value not in spec:
            problems.append(f"{path}: {value!r} not in {list(spec)}")
        return
    nullable = spec.endswith("|None")
    base = spec.removesuffix("|None")
    if value is None:
        if not nullable:
            problems.append(f"{path}: must not be null")
        return
    ok = {
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        "str": lambda v: isinstance(v, str),
        "bool": lambda v: isinstance(v, bool),
        "list": lambda v: isinstance(v, list),
        "any": lambda v: True,
    }[base](value)
    if not ok:
        problems.append(f"{path}: expected {base}, got {type(value).__name__}")


def validate_row(table: int, row: Mapping[str, Any], *, check_verdict: bool = True) -> list[str]:
    """Problems with a row (empty when valid).  Unknown top-level fields are problems too.

    ``check_verdict=False`` skips the Table 1 section 10 comparison (the renderer does it itself
    with the experiment's Table 2 rows in hand).
    """
    spec = SPECS[table]
    problems: list[str] = []
    for key in COMMON_FIELDS:
        if key not in row:
            problems.append(f"{key}: missing")
        elif not isinstance(row[key], str) or not row[key]:
            problems.append(f"{key}: must be a non-empty string")
    optional = OPTIONAL.get(table, {})
    for key in row:
        if key not in spec and key not in COMMON_FIELDS and key not in optional:
            problems.append(f"{key}: unknown field for table {table}")
    for key, sub in spec.items():
        if key not in row:
            problems.append(f"{key}: missing")
        else:
            _check_type(key, row[key], sub, problems)
    for key, sub in optional.items():
        if key in row:
            _check_type(key, row[key], sub, problems)
    if table == 1 and check_verdict and not problems:
        # Table 2 rows (not available here) can only make the verdict worse, so the row may be more
        # severe than the standalone rule but never more favourable.
        v = verdict(row)
        if overclaims(row["result"], v.result):
            problems.append(
                f"result: row says {row['result']}, the section 10 rule says {v.result}"
            )
    if table == 3 and not problems:
        if row["rejected"] + row["unexpectedly_accepted"] > row["attempted"]:
            problems.append("rejected + unexpectedly_accepted > attempted")
    return problems


def require_valid(table: int, row: Mapping[str, Any]) -> dict[str, Any]:
    problems = validate_row(table, row)
    if problems:
        raise RowError(f"table {table} row invalid: " + "; ".join(problems))
    return dict(row)


# --- row builders --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Common:
    experiment_id: str
    case: str
    profile_id: str
    worker: str
    adapter_revision: str

    def fields(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "case": self.case,
            "profile_id": self.profile_id,
            "worker": self.worker,
            "adapter_revision": self.adapter_revision,
        }


def experiment_id(case: str, profile_id: str, utc: str) -> str:
    """``<case>-<profile_id>-<UTC yyyymmddThhmmZ>``."""
    return f"{case}-{profile_id}-{utc}"


@dataclass(slots=True)
class Verdict:
    result: str
    first_blocker: str | None
    reasons: list[str] = field(default_factory=list)


def _count(d: Mapping[str, Any], *keys: str) -> int:
    v: Any = d
    for k in keys:
        v = v.get(k, 0) if isinstance(v, Mapping) else 0
    return int(v or 0)


def verdict(
    row: Mapping[str, Any],
    table2_rows: Iterable[Mapping[str, Any]] | None = None,
    blocker: Mapping[str, str] | None = None,
) -> Verdict:
    """Contract section 10 applied to the Table 1 check fields.

    ``FAIL``: any attempted positive check disagreed (authentication, token selection, relation
    checks, boundary noninterference), a structural invariant failed, or an applicable negative
    test was accepted.  ``PARTIAL``: everything attempted agreed but something required is not
    covered, not executable or not exercised (coverage < required, no relation attempted, no
    negatives, no token-selection check, a Table 2 row with ``checker: none`` / unresolved
    binding / an unimplemented status, attention without an executable relation).  ``PASS``
    otherwise.  ``UNSUPPORTED`` / ``BLOCKED`` / ``NOT_RUN`` come from ``blocker`` and are honoured
    only when nothing was attempted; with evidence present the evidence decides.
    """
    reasons: list[str] = []
    fail: list[str] = []
    partial: list[str] = []

    auth = row.get("authentication", {})
    tok = row.get("token_selection", {})
    rel = row.get("relation_checks", {})
    bni = row.get("boundary_noninterference", {})
    cov = row.get("required_coverage", {})
    neg = row.get("negative", {})
    structural = row.get("structural", {})

    attempted_anything = (
        _count(auth, "attempted")
        + _count(tok, "attempted")
        + _count(rel, "same_runtime", "attempted")
        + _count(rel, "independent", "attempted")
        + _count(bni, "compared")
    ) > 0
    if blocker:
        if blocker.get("kind") not in BLOCKER_KINDS:
            raise RowError(
                f"blocker kind must be one of {list(BLOCKER_KINDS)}, got {blocker.get('kind')!r}"
            )
        if not attempted_anything:
            return Verdict(str(blocker["kind"]), blocker.get("reason") or str(blocker["kind"]), [])

    if _count(auth, "valid") < _count(auth, "attempted"):
        fail.append("authentication: an opening failed to authenticate")
    if _count(tok, "exact") < _count(tok, "attempted"):
        fail.append("token_selection: a sampler check disagreed")
    for klass in ("same_runtime", "independent"):
        if _count(rel, klass, "exact") < _count(rel, klass, "attempted"):
            fail.append(f"relation_checks.{klass}: a relation check disagreed")
    if _count(bni, "mismatched_scalars") > 0 or (
        _count(bni, "compared") > 0 and not bni.get("identical", False)
    ):
        fail.append("boundary_noninterference: replay outputs differ from production")
    if structural.get("result") not in ("ok", "pass", "PASS"):
        fail.append(
            f"structural: {structural.get('failing_invariant') or structural.get('result')}"
        )
    if _count(neg, "rejected") < _count(neg, "attempted"):
        fail.append("negative: an applicable negative test was not rejected")
    # W1cF F4: a relation checker that raised did not agree; it is a rejection, never "no
    # executable checker" (WA's M3b: a wrong-but-valid kernel source made the checker raise TypeError).
    errors = row.get("relation_errors")
    if isinstance(errors, Mapping) and _count(errors, "count") > 0:
        first = errors.get("first")
        fail.append(
            f"relation_checks.error: {_count(errors, 'count')} relation checker call(s) raised instead of "
            f"returning a prediction" + (f" ({first})" if first else "")
        )

    if _count(cov, "required") == 0:
        partial.append("required_coverage: nothing required (no frozen interface set)")
    elif _count(cov, "bound") < _count(cov, "required"):
        partial.append("required_coverage: a required interface is not bound")
    if _count(auth, "attempted") == 0:
        partial.append("authentication: no opening was checked")
    if _count(rel, "same_runtime", "attempted") + _count(rel, "independent", "attempted") == 0:
        partial.append("relation_checks: no executable relation was exercised")
    if _count(tok, "attempted") == 0:
        partial.append("token_selection: forced-token replay is not sampling evidence (15.3)")
    if _count(bni, "compared") == 0:
        partial.append("boundary_noninterference: no production output was compared")
    if _count(neg, "attempted") == 0:
        partial.append("negative: no targeted negative test was run")
    # W1cF F5: an applicable negative id with 0 attempted mutations is NOT_RUN, not passed.
    not_run = row.get("negatives_not_run")
    if isinstance(not_run, list) and not_run:
        partial.append(
            "negative: applicable negative test(s) with 0 attempted mutations (NOT_RUN): "
            + ", ".join(str(x) for x in not_run)
        )
    if table2_rows is not None:
        for t2 in table2_rows:
            if t2.get("case") not in (None, row.get("case")):
                continue
            label = f"{t2.get('op_id')}[{t2.get('decomposition')}]"
            if str(t2.get("status", "")).lower() == "error":
                # W1cF F4: the op's checker raised on at least one VU; that is a failed check.
                fail.append(f"table 2 {label}: relation checker raised (status error)")
            if t2.get("checker") == "none":
                partial.append(f"table 2 {label}: no executable checker")
            if _count(t2.get("binding", {}), "unresolved") > 0:
                partial.append(f"table 2 {label}: unresolved binding")
            if str(t2.get("status", "")).lower() in (
                "unimplemented",
                "unresolved",
                "unexecutable",
                "unknown_relation",
            ):
                partial.append(f"table 2 {label}: status {t2.get('status')}")
            if _count(t2.get("natural_checks", {}), "exact") < _count(
                t2.get("natural_checks", {}), "attempted"
            ):
                fail.append(f"table 2 {label}: natural checks disagreed")
            if _count(t2.get("stress_checks", {}), "exact") < _count(
                t2.get("stress_checks", {}), "attempted"
            ):
                fail.append(f"table 2 {label}: stress checks disagreed")
            if _count(t2.get("mutations", {}), "rejected") < _count(
                t2.get("mutations", {}), "attempted"
            ):
                fail.append(f"table 2 {label}: a mutation was accepted")

    reasons = fail + partial
    if fail:
        return Verdict("FAIL", fail[0], reasons)
    if partial:
        return Verdict("PARTIAL", partial[0], reasons)
    return Verdict("PASS", None, [])


def challenge_mode(row: Mapping[str, Any]) -> str:
    """``security`` or ``diagnostic``; a row without a ``challenge`` field predates the staged challenge
    (its sample came from a raw public ``--seed``) and is therefore diagnostic."""
    ch = row.get("challenge")
    if isinstance(ch, Mapping) and ch.get("mode") == "security":
        return "security"
    return "diagnostic"


def logprobs_label(row: Mapping[str, Any]) -> str:
    """``bound`` / ``unbound`` / ``mixed`` …: whether the row's committed output boundaries bind the chosen
    logprobs (``challenge.bindings.logprobs``, written by the WB2 verifier).  A row written before the field
    existed has boundary trees without ``output_logprob`` leaves: ``unbound`` -- never silently upgraded."""
    ch = row.get("challenge")
    if isinstance(ch, Mapping):
        b = ch.get("bindings")
        if isinstance(b, Mapping) and isinstance(b.get("logprobs"), str):
            return str(b["logprobs"])
    return "unbound"


def verification_evidence(row: Mapping[str, Any], audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The row's ``verification_evidence`` document.  Written by the verifier from the challenge mode
    (security -> :data:`EVIDENCE_TRANSCRIPT`, else :data:`EVIDENCE_DIAGNOSTIC`); an ``audit`` document
    (``results/<experiment>/audit.json``: an independent verifier's re-verification of the row from the
    archive with its own seed, citing its report) replaces it with :data:`EVIDENCE_INDEPENDENT`.  A row
    that already carries the field keeps it unless an audit is given."""
    if audit is not None:
        ev = audit.get("verification_evidence", audit)
        label = str(ev.get("label") or EVIDENCE_INDEPENDENT)
        if label not in EVIDENCE_LABELS:
            raise RowError(f"verification_evidence.label must be one of {list(EVIDENCE_LABELS)}, got {label!r}")
        return {"label": label, "verifier": ev.get("verifier"), "report": ev.get("report"), "note": ev.get("note")}
    have = row.get("verification_evidence")
    if isinstance(have, Mapping) and have.get("label") in EVIDENCE_LABELS:
        return dict(have)
    if challenge_mode(row) == "security":
        return {"label": EVIDENCE_TRANSCRIPT, "verifier": row.get("worker"), "report": None,
                "note": "the prover's own pod: the transcript proves correct derivation conditional on the verifier's interaction/order"}
    return {"label": EVIDENCE_DIAGNOSTIC, "verifier": row.get("worker"), "report": None, "note": "public seed; the security claim is not made"}


def result_label(row: Mapping[str, Any]) -> str:
    """The verdict as it must be displayed: unqualified only under security-mode challenge randomness.
    A ``FAIL`` is a fail under any challenge (a predictable sample that still found a fault is a fault)."""
    result = str(row.get("result"))
    if result == "FAIL" or result in BLOCKER_KINDS or challenge_mode(row) == "security":
        return result  # a blocker kind (NOT_RUN, ...) is not a verdict the challenge could qualify
    return f"{result} (diagnostic challenge)"


def table1_row(
    common: Common,
    *,
    workload: Mapping[str, Any],
    production_changes: str,
    replay_restrictions: Sequence[str],
    token_selection: Mapping[str, int],
    boundary_noninterference: Mapping[str, Any],
    required_coverage: Mapping[str, int],
    authentication: Mapping[str, int],
    relation_checks: Mapping[str, Mapping[str, int]],
    structural: Mapping[str, Any],
    negative: Mapping[str, int],
    table2_rows: Iterable[Mapping[str, Any]] | None = None,
    blocker: Mapping[str, str] | None = None,
    workload_exercised: Mapping[str, Any] | None = None,
    challenge: Mapping[str, Any] | None = None,
    relation_errors: Mapping[str, Any] | None = None,
    negatives_not_run: Sequence[str] | None = None,
    qualifications: Sequence[str] | None = None,
    bindings: Mapping[str, Any] | None = None,
    verification_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a Table 1 row; ``result``/``first_blocker`` come from :func:`verdict`.

    The optional W1cF fields (``relation_errors``, ``negatives_not_run``, ``qualifications``,
    ``bindings``) are written only when given, so rows from older verifiers validate unchanged.
    """
    row: dict[str, Any] = {
        **common.fields(),
        "workload": dict(workload),
        **({"workload_exercised": dict(workload_exercised)} if workload_exercised is not None else {}),
        **({"challenge": dict(challenge)} if challenge is not None else {}),
        **({"relation_errors": dict(relation_errors)} if relation_errors is not None else {}),
        **({"negatives_not_run": list(negatives_not_run)} if negatives_not_run is not None else {}),
        **({"qualifications": list(qualifications)} if qualifications is not None else {}),
        **({"bindings": dict(bindings)} if bindings is not None else {}),
        **({"verification_evidence": dict(verification_evidence)} if verification_evidence is not None else {}),
        "production_changes": production_changes,
        "replay_restrictions": list(replay_restrictions),
        "token_selection": dict(token_selection),
        "boundary_noninterference": dict(boundary_noninterference),
        "required_coverage": dict(required_coverage),
        "authentication": dict(authentication),
        "relation_checks": {k: dict(v) for k, v in relation_checks.items()},
        "structural": dict(structural),
        "negative": dict(negative),
    }
    v = verdict(row, table2_rows, blocker)
    row["result"] = v.result
    row["first_blocker"] = v.first_blocker
    return require_valid(1, row)


def table2_row(common: Common, **fields: Any) -> dict[str, Any]:
    return require_valid(2, {**common.fields(), **fields})


_T2_STATUS_ORDER = (
    "auth_failed",
    "structural",
    "schema",
    "missing",
    "error",
    "mismatch",
    "unknown_relation",
    "unresolved",
    "unimplemented",
    "exact",
)


def _status_rank(result: Any) -> int:
    """Lower is worse; unknown statuses rank worst."""
    return _T2_STATUS_ORDER.index(result.status) if result.status in _T2_STATUS_ORDER else -1


def table2_rows_from_report(
    common: Common,
    report: Any,
    description: Any,
    *,
    decomposition: str = "row",
    stress_checks: Mapping[str, Mapping[str, Any]] | None = None,
    mutations: Mapping[str, Mapping[str, int]] | None = None,
    runtime_impl: Mapping[str, str] | None = None,
    layout_or_quant: Mapping[str, str | None] | None = None,
    limitation: Mapping[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """One Table 2 row per (op, phase, checker class) seen in a :class:`relations.CheckReport`.

    Everything derivable is derived: ``shape``/``dtype``/``origin``/``primitive_semantics`` from the
    description node, ``natural_checks`` / ``binding`` / ``interface`` / ``checker_work`` from the
    VU results.  ``checker`` is ``none`` when nothing executable ran for the op; ``status`` is the
    worst VU status (``auth_failed`` < ... < ``mismatch`` < ``unimplemented`` < ``exact``).  Per-op
    overrides are keyed by op id (``stress_checks``, ``mutations``, ``runtime_impl``,
    ``layout_or_quant``, ``limitation``).
    """
    from verity_vllm.commit import identity  # local: rows.py stays importable without the rest of the package

    groups: dict[tuple[str, str, str], list[Any]] = {}
    for r in report.results:
        if not r.op:
            continue  # structural failure before the op was known; Table 1 carries it
        try:
            phase = identity.parse(r.canonical_id).phase or "?"
        except identity.IdentityError:
            phase = "?"
        groups.setdefault((r.op, phase, r.klass or "none"), []).append(r)

    rows: list[dict[str, Any]] = []
    for (op, phase, klass), results in sorted(groups.items()):
        node = description.node(op)
        exact = sum(1 for r in results if r.status == "exact")
        attempted = sum(1 for r in results if r.status in ("exact", "mismatch"))
        unresolved = sum(1 for r in results if r.status in ("missing", "structural", "schema"))
        worst = min(results, key=_status_rank).status
        input_bytes = max((r.input_bytes for r in results), default=0)
        output_bits = max((r.output_bytes for r in results), default=0) * 8
        impl = (runtime_impl or {}).get(op, node.runtime_impl or "unresolved")
        rows.append(
            table2_row(
                common,
                op_id=op,
                phase=phase,
                shape=list(node.shape),
                dtype=node.dtype,
                layout_or_quant=(layout_or_quant or {}).get(op),
                runtime_impl=impl,
                primitive_semantics=node.relation,
                decomposition=decomposition,
                interface={"input_bytes": input_bytes, "output_bits": output_bits, "range": None},
                origin=node.origin,
                binding={"present": len(results) - unresolved, "unresolved": unresolved},
                checker=klass if attempted else "none",
                natural_checks={"exact": exact, "attempted": attempted},
                stress_checks=dict(
                    (stress_checks or {}).get(op, {"exact": 0, "attempted": 0, "domain": None})
                ),
                mutations=dict((mutations or {}).get(op, {"rejected": 0, "attempted": 0})),
                checker_work={
                    "seconds": float(sum(r.checker_seconds for r in results)),
                    "flops_or_bytes": float(sum(r.input_bytes + r.output_bytes for r in results)),
                    "includes_full_op": any(r.includes_full_op for r in results),
                },
                status=worst,
                limitation=(limitation or {}).get(op),
            )
        )
    return rows


def table3_rows(common: Common, negative_results: Iterable[Any]) -> list[dict[str, Any]]:
    """From :class:`verity_vllm.negatives.NegativeResult` objects (or their ``to_json``)."""
    rows = []
    for r in negative_results:
        doc = r.to_json() if hasattr(r, "to_json") else dict(r)
        rows.append(require_valid(3, {**common.fields(), **doc}))
    return rows


def table4_row(common: Common, **fields: Any) -> dict[str, Any]:
    return require_valid(4, {**common.fields(), **fields})


def negative_summary(table3: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """The Table 1 ``negative`` field from Table 3 rows."""
    rejected = attempted = 0
    for r in table3:
        rejected += int(r.get("rejected", 0))
        attempted += int(r.get("attempted", 0))
    return {"rejected": rejected, "attempted": attempted}


# --- JSONL ---------------------------------------------------------------------------------------


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    return path


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RowError(f"{path}:{number}: {error}") from error
    return rows


def append_row(path: str | os.PathLike[str], table: int, row: Mapping[str, Any]) -> None:
    require_valid(table, row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
