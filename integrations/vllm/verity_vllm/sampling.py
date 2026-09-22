"""Sampling plans: what the verifier's challenge draws, recorded so `Bound` can be evaluated.

Two plans (W2c review, Q3):

``stratified`` (the harness default, :func:`verity_vllm.relations.draw_sample`)
    Round-robin over op families and (layer, phase) strata, without replacement, a fixed
    number ``n`` of VUs, plus the adjacent producer->consumer chains the harness forces in.
    This is a *testing* distribution: it maximises family coverage at a fixed budget.  Its
    inclusion probabilities are neither uniform nor independent, so ``Bound``'s acceptance
    law does not apply to it.

``bernoulli`` (this module, :func:`bernoulli_plan`)
    The two-stage independent selection ``Bound`` (``src/veritor/analysis/bound.py``) and the
    protocol's challenge (``src/veritor/protocol/challenge.py``) assume: every replay unit is
    selected independently with probability ``q``; inside a selected replay unit every
    interior VU is selected independently with probability ``s``.  Replay unit := one
    request (capture-format.md section 12: the whole request is the first-stage unit; the
    per-step trees are storage).  VU := one committed leaf (node instance at one position,
    one port).  ``q = 1`` is the one-stage local plan the planner selects on every fixture.

Selection is a deterministic function of a 32-byte seed the verifier derives after the roots
are known: ``u = HMAC-SHA256(seed, tag || id)[:8] / 2**64`` and the unit is selected iff
``u < rate``.  Per-unit thresholds are exactly independent Bernoulli draws, so the plan's
survival probability of an error set ``E`` is ``prod_r (1 - q + q (1-s)^{|E cap P_r|})``,
which is what ``Bound`` prices.  The protocol module reaches the same distribution by
``K ~ Binomial(N, rate)`` then a uniform ``K``-subset; the two are interchangeable for the
bound, and the adapter can switch to the protocol's derivation once it feeds the protocol.

The verifier writes ``sampling_plan.json`` describing both plans it used (rates, seed
derivation, population sizes, the selected units' counts); ``Bound`` needs exactly
``q``, ``s``, the RU/VU decomposition and the guarantee that the seed was fixed after the
roots -- the last is ``effective_seed``'s job in :mod:`verity_vllm.verify`.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from . import hashing
from .capture import CaptureReader
from .description import Description

PLAN_SCHEMA = "verity-vllm/sampling-plan/v1"
SEED_TAG = "verity-vllm/bernoulli-seed/v1"
RU_TAG = b"ru/"
VU_TAG = b"vu/"


def bernoulli_seed(seed: int, commitments: Mapping[str, Any]) -> bytes:
    """32 bytes: ``H("verity-vllm/bernoulli-seed/v1", be64(seed), json_digest(commitments))``.

    Domain-separated from the stratified plan's ``effective_seed`` so the two samples are
    independent functions of the same verifier seed, both fixed only after the roots."""
    return hashing.H(SEED_TAG, int(seed).to_bytes(8, "big"), hashing.json_digest(commitments))


def unit_uniform(seed: bytes, tag: bytes, unit_id: str) -> float:
    """``HMAC-SHA256(seed, tag || unit_id)[:8] / 2**64`` in ``[0, 1)``."""
    mac = hmac.new(seed, tag + unit_id.encode("utf-8"), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big") / 2.0**64


@dataclass
class BernoulliPlan:
    q: float
    s: float
    seed_hex: str
    requests: list[str]
    selected_requests: list[str]
    vus_per_request: dict[str, int]
    selected: list[str]
    expected_vus: float

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": "bernoulli",
            "replay_unit": "request (all of its steps; capture-format.md section 12)",
            "verification_unit": "one committed leaf: node instance x position x port",
            "q": self.q,
            "s": self.s,
            "seed": self.seed_hex,
            "seed_derivation": 'H("verity-vllm/bernoulli-seed/v1", be64(seed), json_digest(commitments.json))',
            "selection": "u = HMAC-SHA256(seed, tag || id)[:8] / 2**64 < rate, tag in {ru/, vu/}",
            "population": {"requests": len(self.requests), "vus": sum(self.vus_per_request.values())},
            "selected": {"requests": len(self.selected_requests), "vus": len(self.selected)},
            "expected_vus": self.expected_vus,
            "selected_requests": list(self.selected_requests),
            "acceptance_law": "sigma(E) = prod_r (1 - q + q * (1 - s) ** |E cap P_r|)",
        }


def bernoulli_plan(
    description: Description,
    reader: CaptureReader,
    *,
    q: float,
    s: float,
    seed: bytes,
    request_ids: Iterable[str] | None = None,
    op_types: Iterable[str] | None = None,
) -> BernoulliPlan:
    """Independent per-request selection at rate ``q``, then per-VU selection at rate ``s``."""
    if not (0.0 <= q <= 1.0 and 0.0 <= s <= 1.0):
        raise ValueError("q and s must be probabilities")
    wanted_types = set(op_types) if op_types is not None else None
    requests = sorted(request_ids) if request_ids is not None else sorted(reader.request_ids())
    selected_requests: list[str] = []
    vus_per_request: dict[str, int] = {}
    selected: list[str] = []
    for rid in requests:
        boundary = reader.boundary(rid)
        population = [
            cid
            for cid, inst in description.expected_ids(boundary, reader.profile_id).items()
            if (wanted_types is None or inst.node.op_type in wanted_types)
            and reader.locate(cid) is not None
        ]
        vus_per_request[rid] = len(population)
        if unit_uniform(seed, RU_TAG, rid) >= q:
            continue
        selected_requests.append(rid)
        for cid in sorted(population):
            if unit_uniform(seed, VU_TAG, cid) < s:
                selected.append(cid)
    expected = q * s * sum(vus_per_request.values())
    return BernoulliPlan(q, s, seed.hex(), requests, selected_requests, vus_per_request, selected, expected)


@dataclass
class StratifiedPlan:
    n: int
    effective_seed: int
    sample_size: int
    adjacent: int
    strata: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": "stratified",
            "distribution": "round-robin over op families, then (layer, phase) strata; without replacement; "
                            "plus forced adjacent producer->consumer chains",
            "requested": self.n,
            "effective_seed": self.effective_seed,
            "sample_size": self.sample_size,
            "adjacent_vus": self.adjacent,
            "strata_sampled": dict(self.strata),
            "bound_compatible": False,
            "note": "inclusion probabilities are neither uniform nor independent; use the bernoulli plan for Bound",
        }


def describe_stratified(
    description: Description, reader: CaptureReader, sample: Sequence[str], *, n: int,
    effective_seed: int, adjacent: int,
) -> StratifiedPlan:
    strata: dict[str, int] = {}
    for cid in sample:
        try:
            inst = description.instance_of(cid, reader.boundary(_request_of(cid)))
        except Exception:  # noqa: BLE001 - a foreign id is still counted
            strata["?"] = strata.get("?", 0) + 1
            continue
        key = f"{inst.node.op_type}/{inst.node.layer}/{inst.phase}"
        strata[key] = strata.get(key, 0) + 1
    return StratifiedPlan(n, effective_seed, len(sample), adjacent, strata)


def _request_of(cid: str) -> str:
    from . import identity

    parsed = identity.parse(cid)
    if parsed.request_id is None:
        raise ValueError(cid)
    return parsed.request_id


def plan_document(
    *, stratified: StratifiedPlan | None, bernoulli: BernoulliPlan | None, challenge_digests: Mapping[str, str | None]
) -> dict[str, Any]:
    doc: dict[str, Any] = {"schema": PLAN_SCHEMA, "bound_to": dict(challenge_digests), "plans": []}
    if stratified is not None:
        doc["plans"].append(stratified.to_json())
    if bernoulli is not None:
        doc["plans"].append(bernoulli.to_json())
    return doc


def as_dict(plan: BernoulliPlan | StratifiedPlan) -> dict[str, Any]:
    return asdict(plan)
