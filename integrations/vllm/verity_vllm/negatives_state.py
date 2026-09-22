"""State-row negatives (W10, contract §8 / §16.10): the cached-state and recompute variants of N05/N06.

Extends the harness in :mod:`verity_vllm.negatives` without changing it: each function appends
variants to an existing :class:`NegativeResult` (the caller finishes / writes the reproducer).

N06 -- substitute another request's cached state (R3):
    ``substitute_cached_state_keep_root``  the bytes of the source request's K/V value that a hit
        request reads at a cached position are replaced by a third request's value at that position
        (commitments unchanged) -> the source's opening fails.
    ``rebind_cached_source``  the hit request's boundary names a third request as the source of its
        cached prefix -> the cross-boundary token check fails (structural), and the relation over the
        rebound inputs mismatches.
N05 -- stale buffer across units (R4), both applied to the RECOMPUTE unit's copy only (a plain
    ``PatchedReader`` patches an id in every unit that holds it, i.e. both copies -- which is how the
    first run of this negative on the r4 row accepted its own perturbation, see state-identity.md §6):
    ``recompute_copy_differs``  the recompute unit commits a different value for a repeated identity
        (bytes and index leaf changed consistently) -> the verifier's structural check
        (``CaptureReader.recompute_problems``, index level, as run by verify.py) fails.
    ``recompute_copy_stale_bytes``  the recompute unit's stored bytes differ from its index leaf ->
        the byte-level audit (``recompute_problems(check_bytes=True)``) fails; the index-level check
        alone does not see it, and no verification unit binds to the recompute copy (``locate`` -> the
        first unit), so this is reported as its own variant, not folded into the first.

On a capture without cached prefixes / recompute units every variant is ``inconclusive`` (not
counted as attempted), so the B0/B1 rows are unaffected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_vllm import identity
from verity_vllm.capture import Boundary, CachedPrefix, CaptureReader, Entry, RUReader, Value
from verity_vllm.negatives import (
    NegativeHarness,
    NegativeResult,
    PatchedReader,
    PatchedRU,
    perturb_input_strongly,
    perturb_value,
)


class UnitPatchedReader(PatchedReader):
    """A ``PatchedReader`` whose value/entry patches apply to ONE replay unit ``(request_id, step)``:
    an identity a recompute unit repeats lives in two units, and the negative must change one copy."""

    def __init__(self, base: CaptureReader, *, unit: tuple[str, int], values: Mapping[str, bytes] | None = None,
                 entries: Mapping[str, Entry] | None = None) -> None:
        super().__init__(base)
        self._unit = unit
        self._unit_values = dict(values or {})
        self._unit_entries = dict(entries or {})

    def ru(self, request_id: str, step: int) -> RUReader:
        key = (request_id, step)
        if key not in self._rus:
            base = self._base.ru(request_id, step)
            if key == self._unit and (self._unit_values or self._unit_entries):
                self._rus[key] = PatchedRU(base, self._unit_values, self._unit_entries)
            else:
                self._rus[key] = base
        return self._rus[key]


def _hit_requests(h: NegativeHarness) -> list[Boundary]:
    out = []
    for rid in h.reader.request_ids():
        if h.reader.has_boundary(rid):
            b = h.reader.boundary(rid)
            if b.cached_prefix:
                out.append(b)
    return out


def _cross_request_vu(h: NegativeHarness, b: Boundary) -> tuple[str, str] | None:
    """(VU of the hit request, one of its inputs that is a source request's value)."""
    sources = {c.source_request_id for c in b.cached_prefix}
    candidates = [c for c in h.sample if identity.parse(c).request_id == b.request_id]
    if not candidates:
        candidates = [c for c, _ in h.description.expected_ids(b, h.reader.profile_id).items() if h.reader.locate(c) is not None]
    for cid in candidates:
        for _, ref in h._leaf_inputs(cid):
            p = identity.parse(ref)
            if p.kind == "value" and p.request_id in sources:
                return cid, ref
    return None


def _third_request_value(h: NegativeHarness, exclude: set[str], ref: str) -> tuple[str, bytes] | None:
    """Another request's committed value of the same op/port at the same position (different bytes)."""
    p = identity.parse(ref)
    assert p.t is not None and p.layer is not None and p.op is not None and p.port is not None
    original = h.reader.value(ref)
    for rid in h.reader.request_ids():
        if rid in exclude or not h.reader.has_boundary(rid):
            continue
        b = h.reader.boundary(rid)
        cid = str(identity.value_id(h.reader.profile_id, rid, p.t, b.phase_at(p.t), p.layer, p.op, p.port, shard=p.shard, expert=p.expert))
        if h.reader.locate(cid) is None:
            continue
        data = h.reader.value(cid).data
        if data != original.data:
            return rid, data
    return None


def extend_n06_cached_state(h: NegativeHarness, result: NegativeResult, limit: int = 4) -> NegativeResult:
    hits = _hit_requests(h)
    if not hits:
        result.record("no_cached_state", "inconclusive", reason="no request in this capture reads another request's cached state")
        return result
    done = 0
    for b in hits:
        if done >= limit:
            break
        found = _cross_request_vu(h, b)
        if found is None:
            result.record("no_cross_request_reader", "inconclusive", request_id=b.request_id, reason="no VU of the hit request reads a source value")
            continue
        vu, ref = found
        src = identity.parse(ref).request_id or ""
        third = _third_request_value(h, {b.request_id, src}, ref)
        # (a) another request's state substituted for the source's committed value, roots unchanged
        if third is None:
            substituted = perturb_input_strongly(h.reader.value(ref))
            third_id = None
        else:
            third_id, substituted = third
        patched = PatchedReader(h.reader, values={ref: substituted})
        r = h._check(patched, h.commitments, [vu]).results[0]
        result.record(
            "substitute_cached_state_keep_root",
            "rejected" if r.status == "auth_failed" else ("accepted" if r.status == "exact" else "inconclusive"),
            canonical_id=vu, cached_input=ref, substituted_from=third_id or "perturbed", status=r.status, detail=r.detail,
        )
        # (b) the hit request's boundary rebinds its cached prefix to a third request
        others = [rid for rid in h.reader.request_ids() if rid not in (b.request_id, src) and h.reader.has_boundary(rid)]
        rebound_to = None
        for rid in others:
            ob = h.reader.boundary(rid)
            if ob.prompt_len > max(t for c in b.cached_prefix for t in c.positions):
                rebound_to = ob
                break
        if rebound_to is None:
            result.record("rebind_cached_source", "inconclusive", request_id=b.request_id, reason="no third request long enough to rebind to")
        else:
            nb = Boundary.from_json(b.to_json())
            nb.cached_prefix = [CachedPrefix(rebound_to.request_id, rebound_to.prompt_len, c.positions) for c in b.cached_prefix]
            patched = PatchedReader(h.reader, boundaries={b.request_id: nb})
            problems = nb.cached_prefix_problems(lambda rid: patched.boundary(rid) if patched.has_boundary(rid) else None)
            try:
                r = h._check(patched, h.commitments, [vu]).results[0]
                status, detail = r.status, r.detail
            except Exception as error:  # noqa: BLE001 - a rebound input that does not exist is a rejection too
                status, detail = "unresolvable", repr(error)[:200]
            rejected = bool(problems) or status in ("mismatch", "auth_failed", "unresolvable")
            result.record(
                "rebind_cached_source",
                "rejected" if rejected else "accepted",
                canonical_id=vu, rebound_from=src, rebound_to=rebound_to.request_id, structural_problems=problems[:3], status=status, detail=detail,
            )
        done += 1
    return result


def extend_n05_recompute(h: NegativeHarness, result: NegativeResult, limit: int = 2) -> NegativeResult:
    done = 0
    for rid in h.reader.request_ids():
        if done >= limit or not h.reader.has_boundary(rid):
            continue
        b = h.reader.boundary(rid)
        rec = [s for s in b.steps if s.recompute]
        if not rec:
            continue
        baseline = h.reader.recompute_problems(rid)
        if baseline:
            result.record("recompute_unit_stale_bytes", "inconclusive", request_id=rid, reason=f"baseline already fails: {baseline[0]}")
            continue
        step = rec[0].step
        unit = h.reader.ru(rid, step)
        first_step: dict[int, int] = {}
        for s in b.steps:
            for t in s.positions:
                first_step.setdefault(t, s.step)
        target = next((e for e in unit.index.entries if first_step.get(identity.parse(e.canonical_id).t or -1, step) != step), None)
        if target is None:
            result.record("recompute_copy_differs", "inconclusive", request_id=rid, reason="recompute unit repeats no identity")
            continue
        cid = target.canonical_id
        value = unit.value(cid)
        new_bytes = perturb_value(value, h.rng)
        # the original unit's copy must be untouched (a plain PatchedReader would patch both copies)
        original_unit = first_step[identity.parse(cid).t or -1]
        # (a) a consistently committed different second value: bytes + index leaf of the recompute copy
        new_leaf = Value(cid, target.dtype, target.shape, new_bytes).leaf().hex()
        new_entry = Entry(cid, target.dtype, target.shape, target.offset, target.nbytes, target.origin, new_leaf)
        patched = UnitPatchedReader(h.reader, unit=(rid, step), values={cid: new_bytes}, entries={cid: new_entry})
        problems = patched.recompute_problems(rid)
        untouched = patched.ru(rid, original_unit).entry(cid).leaf == h.reader.ru(rid, original_unit).entry(cid).leaf
        result.record(
            "recompute_copy_differs", "rejected" if (problems and untouched) else "accepted",
            request_id=rid, recompute_step=step, original_step=original_unit, canonical_id=cid, original_copy_untouched=untouched,
            check="CaptureReader.recompute_problems (index leaves, the verifier's structural check)", problems=problems[:2],
        )
        # (b) stale bytes under an unchanged index: only the byte-level audit sees it
        patched = UnitPatchedReader(h.reader, unit=(rid, step), values={cid: new_bytes})
        problems_bytes = patched.recompute_problems(rid, check_bytes=True)
        problems_index = patched.recompute_problems(rid)
        result.record(
            "recompute_copy_stale_bytes", "rejected" if problems_bytes else "accepted",
            request_id=rid, recompute_step=step, canonical_id=cid, check="recompute_problems(check_bytes=True)",
            problems=problems_bytes[:2], index_level_check_sees_it=bool(problems_index),
            note="no verification unit binds to the recompute copy (locate -> first unit); the index-level check passes by design",
        )
        done += 1
    if done == 0 and not any(v["variant"].startswith("recompute_copy") for v in result.variants):
        result.record("recompute_copy_differs", "inconclusive", reason="no recompute unit in this capture")
    return result


def summary(results: list[NegativeResult]) -> dict[str, Any]:
    return {r.negative_id: {"attempted": r.attempted, "rejected": r.rejected} for r in results}
