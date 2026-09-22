"""Genuine selective preparation (research workstream C, decision 3; RC 2026-09-07).

The measured protocol, exactly:

    prescribed ingress / fixed serving state; checkpoint / shared static data;
    production boundary / protected outputs                       <- fixed before r_q   (T_0)
    r_q -> select first-stage replay/commitment units (requests)
        -> replay ONLY the selected requests -> construct + bind their interiors   (T_2)
    r_s -> select VUs inside those units -> check / prove the selected obligations

What this module adds to WB's staged challenge (:mod:`verity_vllm.challenge`), which assumed every
replay unit was committed *before* ``T_0`` (``statement.replay_units_declared``):

* :class:`SelectiveSession` -- the same phase digests and randomness derivation (``r_q =
  HMAC(seed, T_0 || purpose)``), with ``T_0`` declaring **no** interiors, and an optional
  *fixed-size* request selection so that ``q in {1, 2, 4, 8, N}`` is realised exactly (the campaign's
  Bernoulli selector draws ``K ~ Binomial(N, q)``, which cannot produce a table indexed by ``q``).
  With ``k=None`` the selection is WB's Bernoulli selector unchanged.
* :func:`fixed_size_subset` -- the exact-``k`` selector: Floyd's algorithm driven by an HMAC-SHA256
  stream keyed by ``r_q`` (the same keying discipline as the core's ``bernoulli_subset``; the
  distribution differs and is recorded in the transcript).  Every ``k``-subset is equally likely, so a
  prover who does not know ``r_q`` cannot bias which requests are inspected.
* :func:`selective_statement` -- ``T_0`` for a selective run: the profile, the description and
  required-set digests, the boundary roots of **all** N requests, the weights root, the binding
  outcomes, the policy and the seed commitment; ``replay_units_declared`` is empty by construction and
  ``first_stage`` records ``interiors_committed_before_r_q = 0``.
* :func:`first_stage_problems_for` -- WB's first-stage completeness (every step of a request
  committed, positions covered once in order) restricted to the selected requests: requests outside
  ``J`` have a committed boundary and, by design, no interior.
* :func:`rederive` -- a third party's re-derivation of a selective transcript (both selectors).
* the file handshake used on a pod where prover and verifier are separate processes
  (:func:`wait_for_file`, :func:`write_json_atomic`).

The interpretation caveat the report carries: the campaign's ``Bound`` is stated for independent
Bernoulli selection at rate ``q``; a fixed-size selection of ``k`` out of ``N`` is the equal-probability
sampling-without-replacement variant and is reported as such (``q = k/N`` is the marginal inclusion
probability, not the policy the bound was derived for).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import hashing
from .challenge import (
    CORE_STAGE_S,
    PURPOSE_Q,
    ChallengeError,
    ChallengeSource,
    Policy,
    RankedUnits,
    StagedSession,
    _HmacSource,
    core_bernoulli_subset,
    interior_phase,
    replay_phase,
    sample_phase,
    seed_commitment,
    statement_digest,
)

STAGE_Q_FIXED = b"q/replay-unit-fixed-k"
SELECTIVE_SCHEMA = "verity-vllm/selective-statement/v1"


# --- the fixed-size selector ---------------------------------------------------------------------


def _stream(key: bytes, stage: bytes, phase_digest: bytes):
    """Counter-mode HMAC-SHA256 stream keyed by ``key`` over ``(stage, phase_digest, counter)``."""
    counter = 0
    while True:
        block = hmac.new(key, stage + b"\x00" + phase_digest + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
        yield block


def _uniform_below(stream, bound: int) -> int:
    """An unbiased integer in ``[0, bound)`` by rejection over 256-bit draws."""
    if bound <= 0:
        raise ValueError("bound must be positive")
    limit = (1 << 256) - ((1 << 256) % bound)
    while True:
        x = int.from_bytes(next(stream), "big")
        if x < limit:
            return x % bound


def fixed_size_subset(r_q: bytes, phase_digest: bytes, count: int, k: int) -> tuple[int, ...]:
    """Exactly ``k`` distinct indices of ``range(count)``, each ``k``-subset equally likely.

    Floyd's algorithm: for ``j = count - k .. count - 1`` draw ``t`` uniform in ``[0, j]``; take ``t``
    unless already taken, else take ``j``.  Randomness: HMAC-SHA256(r_q, stage || T_0 || counter).
    """
    if not 0 <= k <= count:
        raise ChallengeError(f"k={k} outside [0, {count}]")
    if k == 0:
        return ()
    stream = _stream(r_q, STAGE_Q_FIXED, phase_digest)
    chosen: set[int] = set()
    for j in range(count - k, count):
        t = _uniform_below(stream, j + 1)
        chosen.add(j if t in chosen else t)
    assert len(chosen) == k
    return tuple(sorted(chosen))


# --- the statement (T_0) --------------------------------------------------------------------------


def selective_statement(
    *,
    profile_id: str,
    manifest_bytes: bytes,
    description_digest: str,
    required_digest: str | None,
    commitments_phase0: Mapping[str, Any],
    bindings_statement: Mapping[str, Any],
    policy: Policy,
    source_opening: Mapping[str, Any],
    request_ids: Sequence[str],
    k: int | None,
    weights_tree_id: str,
) -> dict[str, Any]:
    """Everything the prover is bound to before ``r_q`` exists, for a run advertised as selective.

    Same fields as WB's ``statement_of`` so that :func:`verity_vllm.challenge.statement_checks` applies,
    plus ``selective``; ``replay_units_declared`` must be empty (asserted: a run that committed any
    interior before ``r_q`` is not selective)."""
    trees = commitments_phase0.get("trees", {})
    declared = sorted(tid for tid, rec in trees.items() if rec.get("kind") == "ru")
    if declared:
        raise ChallengeError(f"{len(declared)} replay-unit tree(s) committed before r_q; this run is not selective")
    boundary_ids = sorted(rec["request_id"] for rec in trees.values() if rec.get("kind") == "boundary")
    if boundary_ids != sorted(request_ids):
        raise ChallengeError("the committed boundary set is not the production request set")
    return {
        "schema": SELECTIVE_SCHEMA,
        "profile_id": profile_id,
        "manifest_digest": hashlib.sha256(manifest_bytes).hexdigest(),
        "description_digest": description_digest,
        "required_digest": required_digest,
        "boundary_roots": {
            tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
            for tid, rec in sorted(trees.items()) if rec.get("kind") == "boundary"
        },
        "weights_root": (
            {k_: trees[weights_tree_id].get(k_) for k_ in ("root", "leaf_count", "ids_digest", "weights_sha256")}
            if weights_tree_id in trees else None
        ),
        "replay_units_declared": [],
        "first_stage": {"mode": "selective", "requests": len(request_ids), "interiors_committed_before_r_q": 0},
        "selective": {
            "population": "requests",
            "candidates": len(request_ids),
            "selection": {"kind": "fixed_size", "k": int(k)} if k is not None else {"kind": "bernoulli", "q": policy.to_json()["q"]},
        },
        "bindings": dict(bindings_statement),
        "policy": policy.to_json(),
        "source": dict(source_opening),
    }


# --- the session -----------------------------------------------------------------------------------


class SelectiveSession(StagedSession):
    """WB's staged session with the selective ``T_0`` and an optional exact-``k`` request selector."""

    def __init__(self, source: ChallengeSource, policy: Policy, *, k: int | None = None) -> None:
        super().__init__(source, policy)
        self.k = k
        self.transcript["selective"] = {
            "protocol": "decision-3 selective preparation: nothing per-request committed before r_q",
            "request_selector": (
                {"kind": "fixed_size", "k": int(k), "stage": STAGE_Q_FIXED.decode(),
                 "distribution": "uniform k-subset (Floyd); HMAC-SHA256 counter stream keyed by r_q"}
                if k is not None else
                {"kind": "bernoulli", "q": policy.to_json()["q"], "distribution": "core bernoulli_subset (K ~ Binomial(N, q))"}
            ),
        }

    def challenge_replay(self, request_ids: Sequence[str]) -> list[str]:
        if self.k is None:
            return super().challenge_replay(request_ids)
        self._expect("replay")
        candidates = sorted(request_ids)
        self.r_q = self.source.randomness(self.t0, PURPOSE_Q)
        picked = fixed_size_subset(self.r_q, self.t0, len(candidates), self.k)
        self.selected_requests = [candidates[i] for i in picked]
        msg = {"r_q": self.r_q.hex(), "purpose": PURPOSE_Q.decode(), "candidates": len(candidates),
               "selector": {"kind": "fixed_size", "k": int(self.k), "stage": STAGE_Q_FIXED.decode()},
               "selected_indices": list(picked), "selected": list(self.selected_requests)}
        self.t1 = replay_phase(self.t0, msg)
        self.transcript["replay_challenge"] = msg
        self.transcript["phases"].append({"phase": "replay", "digest": self.t1.hex()})
        self._phase = "interiors"
        return list(self.selected_requests)


def rederive(transcript: Mapping[str, Any], *, statement: Mapping[str, Any], interiors: Mapping[str, Any],
             request_ids: Sequence[str], units: RankedUnits, resolve_ids: Any, coverage: Any | None) -> dict[str, Any]:
    """Re-derive every phase digest and both selections of a selective transcript from the documents
    it claims to be bound to (either request selector).  Mirrors :func:`verity_vllm.challenge.reverify`."""
    problems: list[str] = []
    phases = {p["phase"]: bytes.fromhex(p["digest"]) for p in transcript.get("phases", [])}
    t0 = statement_digest(statement)
    if phases.get("statement") != t0:
        problems.append("T_0 does not re-derive from the statement")
    reveal = transcript.get("reveal") or {}
    seed_hex = reveal.get("seed")
    if not seed_hex:
        problems.append("transcript reveals no seed")
        return {"ok": False, "problems": problems}
    seed = bytes.fromhex(seed_hex)
    src = transcript.get("source", {})
    if src.get("seed_commitment") != seed_commitment(seed):
        problems.append("revealed seed does not open the seed commitment")
    rc = transcript["replay_challenge"]
    r_q = _HmacSource.derive(seed, t0, PURPOSE_Q)
    if r_q.hex() != rc["r_q"]:
        problems.append("r_q does not re-derive from the seed and T_0")
    candidates = sorted(request_ids)
    sel = rc.get("selector") or {}
    if sel.get("kind") == "fixed_size":
        picked = fixed_size_subset(r_q, t0, len(candidates), int(sel["k"]))
    else:
        from .challenge import CORE_STAGE_Q

        policy = Policy.parse(transcript["policy"]["q"], transcript["policy"]["s"])
        picked = core_bernoulli_subset(r_q, CORE_STAGE_Q, t0, len(candidates), policy.q)
    if [candidates[i] for i in picked] != list(rc["selected"]):
        problems.append("the selected requests do not re-derive from r_q")
    t1 = replay_phase(t0, rc)
    if phases.get("replay") != t1:
        problems.append("T_1 does not re-derive")
    t2 = interior_phase(t1, interiors)
    if phases.get("interior") != t2:
        problems.append("T_2 does not re-derive from the committed interiors")
    sc = transcript["sample_challenge"]
    from .challenge import PURPOSE_S, coverage_seed

    r_s = _HmacSource.derive(seed, t2, PURPOSE_S)
    if r_s.hex() != sc["r_s"]:
        problems.append("r_s does not re-derive from the seed and T_2")
    policy = Policy.parse(transcript["policy"]["q"], transcript["policy"]["s"])
    ranks = core_bernoulli_subset(r_s, CORE_STAGE_S, t2, units.count, policy.s)
    if list(ranks) != list(sc["selected_ranks"]):
        problems.append("the certified ranks do not re-derive from r_s")
    by_tree: dict[str, list[int]] = {}
    for rank in ranks:
        tid, idx = units.resolve(rank)
        by_tree.setdefault(tid, []).append(idx)
    certified: list[str] = []
    for tid, idxs in by_tree.items():
        certified.extend(resolve_ids(tid, idxs))
    if certified != list(sc["certified_sample"]):
        problems.append("the certified sample does not re-derive from the ranks")
    if coverage is not None and list(coverage(coverage_seed(r_s))) != list(sc["coverage_sample"]):
        problems.append("the coverage sample does not re-derive from r_s")
    t3 = sample_phase(t2, sc)
    if phases.get("sample") != t3:
        problems.append("T_3 does not re-derive")
    return {"ok": not problems, "problems": problems}


def statement_checks(statement: Mapping[str, Any], *, reader: Any, description: Any, required_doc: Mapping[str, Any] | None,
                     commitments: Mapping[str, Any], production_sha256: str | None, selected: Sequence[str],
                     commitments_phase0: Mapping[str, Any] | None = None) -> list[str]:
    """:func:`verity_vllm.challenge.statement_checks` for a *selective* statement (INT3, additive).

    WB's check requires ``statement.replay_units_declared`` to equal the bundle's committed replay-unit
    trees; a selective statement declares none by construction and the final ``commitments.json`` holds
    J's.  So: every WB field except that one, then the selective invariants -- ``replay_units_declared`` is
    empty, every committed replay-unit tree belongs to a selected request, every selected request has at
    least one, and (when ``commitments_phase0.json`` travels with the bundle) every phase-0 record is
    unchanged in the final commitments and no phase-0 record is a replay unit."""
    from . import required as vrequired
    from .capture import WEIGHTS_TREE_ID

    problems: list[str] = []
    trees: Mapping[str, Any] = commitments.get("trees", {})
    if statement.get("schema") != SELECTIVE_SCHEMA:
        problems.append(f"statement.schema is not {SELECTIVE_SCHEMA}")
    if statement.get("profile_id") != reader.profile_id:
        problems.append("statement.profile_id is not the bundle's profile")
    if statement.get("manifest_digest") != hashlib.sha256((reader.run_dir / "manifest.json").read_bytes()).hexdigest():
        problems.append("statement.manifest_digest is not sha256(manifest.json)")
    if statement.get("description_digest") != description.digest().hex():
        problems.append("statement.description_digest is not the bundle's description")
    want_req = vrequired.required_digest(required_doc).hex() if required_doc else None
    if statement.get("required_digest") != want_req:
        problems.append("statement.required_digest is not the bundle's required set")
    boundary = {tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
                for tid, rec in sorted(trees.items()) if rec.get("kind") == "boundary"}
    if statement.get("boundary_roots") != boundary:
        problems.append("statement.boundary_roots differ from commitments.json")
    weights = {k: trees[WEIGHTS_TREE_ID].get(k) for k in ("root", "leaf_count", "ids_digest", "weights_sha256")} if WEIGHTS_TREE_ID in trees else None
    if statement.get("weights_root") != weights:
        problems.append("statement.weights_root differs from commitments.json")
    if statement.get("replay_units_declared") != []:
        problems.append("statement.replay_units_declared is not empty: the run was not selective")
    if ((statement.get("first_stage") or {}).get("interiors_committed_before_r_q") or 0) != 0:
        problems.append("statement.first_stage claims interiors committed before r_q")
    if (statement.get("selective") or {}).get("candidates") != len(boundary):
        problems.append("statement.selective.candidates is not the number of committed boundaries")
    chosen = set(selected)
    ru_requests = {rec.get("request_id") for rec in trees.values() if rec.get("kind") == "ru"}
    outside = sorted(r for r in ru_requests if r not in chosen)
    if outside:
        problems.append(f"{len(outside)} request(s) outside J have committed replay units, e.g. {outside[0]}")
    missing = sorted(chosen - ru_requests)
    if missing:
        problems.append(f"{len(missing)} selected request(s) have no committed replay unit, e.g. {missing[0]}")
    rec_sha = ((statement.get("bindings") or {}).get("output") or {}).get("production_record_sha256")
    if production_sha256 is not None and rec_sha is not None and rec_sha != production_sha256:
        problems.append("statement binds a different production record (sha256)")
    if commitments_phase0 is not None:
        trees0: Mapping[str, Any] = commitments_phase0.get("trees", {})
        kinds0 = {rec.get("kind") for rec in trees0.values()}
        if not kinds0 <= {"boundary", "weights"}:
            problems.append(f"phase-0 commitments carry {sorted(k for k in kinds0 if k not in ('boundary', 'weights'))}")
        changed = [tid for tid, rec in trees0.items() if trees.get(tid) != rec]
        if changed:
            problems.append(f"{len(changed)} phase-0 record(s) differ in the final commitments, e.g. {changed[0]}")
        phase0_boundary = {tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
                           for tid, rec in sorted(trees0.items()) if rec.get("kind") == "boundary"}
        if statement.get("boundary_roots") != phase0_boundary:
            problems.append("statement.boundary_roots differ from commitments_phase0.json")
    return problems


# --- first-stage completeness restricted to J -----------------------------------------------------


def first_stage_problems_for(reader: Any, commitments: Mapping[str, Any], selected: Sequence[str]) -> dict[str, Any]:
    """WB's first-stage completeness (:func:`verity_vllm.firststage.first_stage_problems`) for the selected
    requests only, plus the selective invariant: no request outside ``J`` has any committed interior."""
    from .capture import boundary_tree_id, ru_tree_id

    trees: Mapping[str, Any] = commitments.get("trees", {})
    problems: list[str] = []
    steps_total = 0
    chosen = set(selected)
    for rid in selected:
        if not reader.has_boundary(rid):
            problems.append(f"{rid}: no boundary.json")
            continue
        if boundary_tree_id(rid) not in trees:
            problems.append(f"{rid}: boundary tree not committed")
        boundary = reader.boundary(rid)
        declared = {s.step: s for s in boundary.steps}
        if not declared:
            problems.append(f"{rid}: selected but its boundary lists no steps")
        committed = {int(rec["step"]) for rec in trees.values() if rec.get("kind") == "ru" and rec.get("request_id") == rid}
        for step, s in declared.items():
            steps_total += 1
            rec = trees.get(ru_tree_id(rid, step))
            if rec is None:
                problems.append(f"{ru_tree_id(rid, step)}: declared, no committed root")
            elif rec.get("phase") != s.phase or [int(t) for t in rec.get("positions", [])] != list(s.positions):
                problems.append(f"{ru_tree_id(rid, step)}: committed (phase, positions) differ from the boundary's")
        for step in sorted(committed - set(declared)):
            problems.append(f"{ru_tree_id(rid, step)}: committed root for a step the boundary does not list")
        on_disk = {step for _, step in reader.ru_keys(rid)}
        if on_disk != set(declared):
            problems.append(f"{rid}: RUs on disk {sorted(on_disk)} != boundary steps {sorted(declared)}")
        covered = [t for s in boundary.steps for t in s.positions]
        processed = boundary.prompt_len + max(len(boundary.output_token_ids) - 1, 0)
        if covered != list(range(processed)):
            problems.append(f"{rid}: steps cover {len(covered)} positions, expected 0..{processed - 1} once in order")
    outside = sorted({rec["request_id"] for rec in trees.values() if rec.get("kind") == "ru"} - chosen)
    if outside:
        problems.append(f"{len(outside)} request(s) outside J have committed interiors, e.g. {outside[0]}")
    on_disk_outside = sorted({rid for rid, _ in reader.ru_keys()} - chosen)
    if on_disk_outside:
        problems.append(f"{len(on_disk_outside)} request(s) outside J have RUs on disk, e.g. {on_disk_outside[0]}")
    return {"requests": len(selected), "steps": steps_total, "ok": not problems, "problems": problems}


def replayed_positions(reader: Any, selected: Sequence[str]) -> dict[str, int]:
    """Token accounting of the selected requests: prompt tokens, generated tokens, positions processed."""
    prompt = generated = positions = 0
    for rid in selected:
        b = reader.boundary(rid)
        prompt += b.prompt_len
        generated += len(b.output_token_ids)
        positions += b.prompt_len + max(len(b.output_token_ids) - 1, 0)
    return {"prompt_tokens": prompt, "generated_tokens": generated, "positions_processed": positions}


# --- file handshake ---------------------------------------------------------------------------------


def write_json_atomic(path: str | os.PathLike[str], doc: Mapping[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, p)
    return p


def wait_for_file(path: str | os.PathLike[str], *, timeout_s: float, poll_s: float = 1.0, log: Any = None) -> float:
    """Block until ``path`` exists; returns the seconds waited.  Raises ``TimeoutError``."""
    p = Path(path)
    t0 = time.perf_counter()
    last = 0.0
    while not p.exists():
        waited = time.perf_counter() - t0
        if waited > timeout_s:
            raise TimeoutError(f"{p} did not appear within {timeout_s:.0f}s")
        if log is not None and waited - last >= 60:
            log(f"waiting for {p.name} ({waited:.0f}s)")
            last = waited
        time.sleep(poll_s)
    return time.perf_counter() - t0


def fraction_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"
