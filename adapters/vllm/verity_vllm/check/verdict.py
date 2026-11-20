"""A Match run's word over its gates (`match_verdict`)."""
from __future__ import annotations

from verity_vllm.check.result import FAIL, INSUFFICIENT, NOT_RUN, CheckResult

__all__ = ["match_verdict"]


def match_verdict(results: list[CheckResult], required: tuple[str, ...]) -> str:
    """A Match run's word over its gates: REJECTED when any gate FAILs; INCOMPLETE when a `required` gate is absent or did not run,
    or any gate is INSUFFICIENT_EVIDENCE; ACCEPTED otherwise.  A gate outside `required` that did not run does not weigh."""
    if any(r.outcome == FAIL for r in results):
        return "REJECTED"
    req = [r for r in results if r.name in required]
    if {r.name for r in req} != set(required) or any(r.outcome == NOT_RUN for r in req) or any(r.outcome == INSUFFICIENT for r in results):
        return "INCOMPLETE"
    return "ACCEPTED"


# ---------------------------------------------------------------------------------------------------------------------- aggregation


# ---------------------------------------------------------------------------------------------------------------------- building


