"""The network certifier's period evidence on the prover's side: each one-stage answer the period check
(`NetworkCheck.Period`) reads, with the inputs one-stage's compiled verifier gave it, which the check runs that verifier
on again and holds equal (`Properties.NetworkPeriod.AnswersVerified`)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

GOAL = "network/check-reads-verified-answers"


def verified_answer(answer: Mapping[str, Any], params: bytes, commitment: bytes, evidence: bytes) -> dict[str, Any]:
    """The period evidence's entry for one answer the check reads (a compute audit's, or a link-window's `answer`): the
    answer with the one-stage parameters, commitment and evidence it was given on, on which the period check runs
    `Spot.OneStage.protocol.answer` and refuses an answer it does not give back."""
    raise NotImplementedError(GOAL)
