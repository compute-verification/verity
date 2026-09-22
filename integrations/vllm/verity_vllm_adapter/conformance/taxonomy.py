"""The closed unsupported taxonomy (coverage wave Part 3 "make unsupported outcomes first-class", Part 4 item 2).

One place for the enum. Everything that names a support outcome — the per-port `unsupported` field of
`ladder.json`, `expectations.json` (`families.<f>.unsupported_reason`, `unsupported_functions[].reason`), a
port's `NotImplementedError("unsupported:… <function/value>")`, the matrix's "Main unsupported reason" column —
takes its value from `OUTCOMES` and is refused otherwise (`PortError`, fail closed).

    supported-exactly                                    the function/value is represented and every compared word is exact
    unsupported:primitive-semantics-missing              the kernel's exact numerics are not a registered primitive/composite
    unsupported:compact-representation-missing           the canonical program has no (compact) counterpart for the value/function
    unsupported:replay-cannot-obtain-canonical-value     replay/capture cannot materialise or bind the canonical value
    unsupported:commitment-layout-cannot-authenticate    the value exists canonically but the commitment layout cannot open it

Each `unsupported` outcome carries a free-text `detail` naming the function or value (Part 4: a clean
`unsupported:primitive-semantics-missing FA3 wgmma m64n8k16` is a *result*; inventing a representation locally
until PASS is not — a lane that needs a new IR construct reports it as a finding and stops at the rung it reached).

Mapping from `verity_vllm.correspondence.core`'s record classes (TA1/B1Q's file; not edited here — HP and TP have
in-flight changes to it): `CORRESPONDENCE_STATUS_TO_OUTCOME` / `classify_correspondence_record`. It is what the
harness applies when it derives an unsupported outcome from evidence rather than from a port's declaration:

    check.status == "exact"                               -> supported-exactly
    check.status == "mismatch", recorded on the binding   -> (no taxonomy outcome: a conformance downgrade, `same-runtime-only`,
                                                             typed-b1.md §3/§5; the rung is met with the mismatch listed)
    check.status == "mismatch", not recorded              -> (no taxonomy outcome: the rung is `not-met`)
    check.status == "missing-operand"                     -> unsupported:replay-cannot-obtain-canonical-value
    check.status == "shape-mismatch"                      -> unsupported:compact-representation-missing
    NotInProgram (capture id with no producing call),
      not classified by expectations.not_in_program       -> unsupported:compact-representation-missing
    weights leaf `not_in_program` (no consuming gate),
      not classified by expectations.not_in_program       -> unsupported:compact-representation-missing
    authentication_location is null on a located record  -> unsupported:commitment-layout-cannot-authenticate
"""

from __future__ import annotations

from typing import Any

SUPPORTED_EXACTLY = "supported-exactly"
PRIMITIVE_SEMANTICS_MISSING = "unsupported:primitive-semantics-missing"
COMPACT_REPRESENTATION_MISSING = "unsupported:compact-representation-missing"
REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE = "unsupported:replay-cannot-obtain-canonical-value"
COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE = "unsupported:commitment-layout-cannot-authenticate"

UNSUPPORTED_REASONS: tuple[str, ...] = (
    PRIMITIVE_SEMANTICS_MISSING,
    COMPACT_REPRESENTATION_MISSING,
    REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE,
    COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE,
)
OUTCOMES: tuple[str, ...] = (SUPPORTED_EXACTLY, *UNSUPPORTED_REASONS)

# `verity_vllm.correspondence.core` record `check.status` -> outcome (None: not a taxonomy outcome, see the module docstring)
CORRESPONDENCE_STATUS_TO_OUTCOME: dict[str, str | None] = {
    "exact": SUPPORTED_EXACTLY,
    "mismatch": None,
    "missing-operand": REPLAY_CANNOT_OBTAIN_CANONICAL_VALUE,
    "shape-mismatch": COMPACT_REPRESENTATION_MISSING,
}


class TaxonomyError(ValueError):
    """A support outcome outside the closed enum."""


def is_outcome(value: Any) -> bool:
    return isinstance(value, str) and value in OUTCOMES


def is_unsupported_reason(value: Any) -> bool:
    return isinstance(value, str) and value in UNSUPPORTED_REASONS


def check_unsupported_reason(value: Any, where: str) -> str:
    """Return `value` if it is one of the four unsupported reasons; raise `TaxonomyError` naming `where` otherwise."""
    if not is_unsupported_reason(value):
        raise TaxonomyError(f"{where}: must be one of {list(UNSUPPORTED_REASONS)}, got {value!r}")
    return value


def check_outcome(value: Any, where: str) -> str:
    if not is_outcome(value):
        raise TaxonomyError(f"{where}: must be one of {list(OUTCOMES)}, got {value!r}")
    return value


def parse(text: str | None) -> tuple[str, str] | None:
    """`"unsupported:primitive-semantics-missing FA3 wgmma m64n8k16"` -> `(reason, "FA3 wgmma m64n8k16")`.

    The form a port's `static_config.program()` / `bindings()` may raise as `NotImplementedError(...)` to classify why it
    does not claim C2 (Part 4 item 2). Returns None when the text does not start with an enum member — a free-text
    reason ("typed M1 program: lane CM") is *pending work*, not an unsupported outcome, and stays unclassified."""
    if not text:
        return None
    head, _, rest = text.strip().partition(" ")
    head = head.rstrip(":;,")
    if head in UNSUPPORTED_REASONS:
        return head, rest.strip()
    return None


def classify_correspondence_record(record: dict[str, Any]) -> str | None:
    """The outcome of one `verity-ir/correspondence/v0` record (see the module docstring); None for a mismatch."""
    st = (record.get("check") or {}).get("status")
    if st == "exact" and record.get("authentication_location") is None and record.get("value_identity") is not None:
        return COMMITMENT_LAYOUT_CANNOT_AUTHENTICATE
    return CORRESPONDENCE_STATUS_TO_OUTCOME.get(str(st))


def unsupported_record(rung: str, reason: str | None, detail: str | None, source: str) -> dict[str, Any]:
    """The `unsupported` object of `ladder.json`: `reason` is an enum member or None (unclassified: pending work)."""
    if reason is not None:
        check_unsupported_reason(reason, f"unsupported at {rung}")
    return {"rung": rung, "reason": reason, "detail": detail, "source": source, "classified": reason is not None}
