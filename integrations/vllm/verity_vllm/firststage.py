"""First-stage completeness: the whole request is the first-stage unit (W2c review, Q2).

Interpretation (capture-format.md section 12): a *request* is the unit whose external inputs and
outputs production binds; the per-step trees ``ru/<request_id>/<step>/`` are storage
organisation inside it.  Steps exchange state (KV rows read by later steps, the sampled token
that becomes the next input), so a step is not an independently verifiable unit until every
step of its request is committed.  The verifier may therefore draw a challenge only after

* every request in the capture has a committed boundary tree,
* every scheduled step recorded in the request's boundary has a committed replay-unit tree and
  no replay-unit tree is committed for a step the boundary does not list,
* the committed step records (phase, positions) agree with the boundary's, and the steps cover
  positions ``0 .. prompt_len + outputs - 2`` once, in order (so every cross-step read
  ``t' <= t`` resolves to a committed leaf, never to a fresh input).  Under capture-format v1.1
  (W10 state rows) a position served from another request's prefix cache (R3) is covered by a
  committed step of the *source* request instead, and a post-preemption recompute chunk (R4)
  may repeat positions an earlier step of the request committed.

:func:`first_stage_problems` checks these from the reader and ``commitments.json`` without
hashing anything (cheap, runs before the challenge; the Merkle audit is separate).
:func:`assert_sample_committed` is the guard the challenge derivation must pass: every sampled
VU lives in a committed tree.  Both are pure functions of the bundle and are used by
``verity_vllm.verify`` and by the review tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import identity
from .capture import CaptureReader, boundary_tree_id, ru_tree_id


class FirstStageError(ValueError):
    """The challenge references a tree that is not in the committed set."""


@dataclass
class FirstStageReport:
    requests: int = 0
    steps: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_json(self) -> dict[str, Any]:
        return {"requests": self.requests, "steps": self.steps, "ok": self.ok, "problems": list(self.problems)}


def first_stage_problems(reader: CaptureReader, commitments: Mapping[str, Any]) -> FirstStageReport:
    """Every request's boundary and every one of its steps are committed, consistently."""
    trees: Mapping[str, Any] = commitments.get("trees", {})
    report = FirstStageReport()
    for rid in reader.request_ids():
        report.requests += 1
        if not reader.has_boundary(rid):
            report.problems.append(f"{rid}: no boundary.json")
            continue
        if boundary_tree_id(rid) not in trees:
            report.problems.append(f"{rid}: boundary tree not committed")
        boundary = reader.boundary(rid)
        declared = {s.step: s for s in boundary.steps}
        committed_steps = {
            int(rec["step"]) for tid, rec in trees.items()
            if rec.get("kind") == "ru" and rec.get("request_id") == rid
        }
        if not boundary.steps:
            report.problems.append(f"{rid}: boundary lists no steps")
        for step, s in declared.items():
            report.steps += 1
            tid = ru_tree_id(rid, step)
            rec = trees.get(tid)
            if rec is None:
                report.problems.append(f"{tid}: step declared in the boundary, no committed root")
                continue
            if rec.get("phase") != s.phase or [int(t) for t in rec.get("positions", [])] != list(s.positions):
                report.problems.append(f"{tid}: committed (phase, positions) differ from the boundary's")
        for step in sorted(committed_steps - set(declared)):
            report.problems.append(f"{ru_tree_id(rid, step)}: committed root for a step the boundary does not list")
        on_disk = {step for r, step in reader.ru_keys(rid)}
        for step in sorted(on_disk - set(declared)):
            report.problems.append(f"{ru_tree_id(rid, step)}: on disk, not in the boundary's steps")
        for step in sorted(set(declared) - on_disk):
            report.problems.append(f"{ru_tree_id(rid, step)}: declared in the boundary, not on disk")
        # capture-format v1.1 (W10): positions served from another request's prefix cache (R3) are
        # never in this request's steps, and a post-preemption recompute chunk (R4) repeats positions
        # earlier steps committed; Boundary.validate checks both shapes, here only the coverage
        cached = sorted(boundary.cached_positions())
        covered = boundary.covered_positions()
        processed = boundary.prompt_len + max(len(boundary.output_token_ids) - 1, 0)
        if cached + covered != list(range(processed)):
            report.problems.append(
                f"{rid}: steps cover {len(covered)} positions"
                + (f" (+{len(cached)} served from a cached prefix)" if cached else "")
                + f", expected 0..{processed - 1} once in order (a cross-step read could hit an uncommitted position)"
            )
        # R3: a cached position is read from the SOURCE request's leaves, so the source's boundary
        # and the step that computed it must be in the committed set too (the token agreement of the
        # two requests is Boundary.cached_prefix_problems, run by the verifier's structural checks)
        for c in boundary.cached_prefix:
            src_id = c.source_request_id
            if not reader.has_boundary(src_id) or boundary_tree_id(src_id) not in trees:
                report.problems.append(f"{rid}: cached-prefix source {src_id} has no committed boundary")
                continue
            src = reader.boundary(src_id)
            for t in c.positions:
                s = src.step_of(t)
                if s is None or ru_tree_id(src_id, s.step) not in trees:
                    report.problems.append(
                        f"{rid}: cached position {t} is in no committed step of its source {src_id}"
                    )
                    break
    return report


def tree_of_sample_id(canonical_id: str, reader: CaptureReader) -> str:
    """The tree a sampled VU must be opened from (``ru/<request>/<step>`` for value ids)."""
    cid = identity.parse(canonical_id)
    if cid.kind != "value" or cid.request_id is None or cid.t is None:
        raise FirstStageError(f"{canonical_id}: not a request-local value id")
    if cid.speculative is not None:
        # speculative extension: an uncommitted value names the step that computed it (``x=<step>``)
        return ru_tree_id(cid.request_id, int(cid.speculative))
    boundary = reader.boundary(cid.request_id)
    step = boundary.step_of(int(cid.t))
    if step is None:
        raise FirstStageError(f"{canonical_id}: position {cid.t} is in no step of {cid.request_id}")
    return ru_tree_id(cid.request_id, step.step)


def assert_sample_committed(
    sample: Iterable[str], reader: CaptureReader, commitments: Mapping[str, Any]
) -> dict[str, int]:
    """Raise :class:`FirstStageError` if any sampled VU's tree has no committed root.

    Returns ``{"vus": n, "trees": k}`` for the record.  This is the ordering invariant the
    Q2 interpretation needs: a challenge may only name leaves of trees whose roots were fixed
    before the challenge existed."""
    trees = commitments.get("trees", {})
    seen: set[str] = set()
    n = 0
    for cid in sample:
        n += 1
        tid = tree_of_sample_id(cid, reader)
        if tid not in trees:
            raise FirstStageError(f"{cid}: its tree {tid} is not in the committed set")
        seen.add(tid)
    return {"vus": n, "trees": len(seen)}
