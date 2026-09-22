"""Fresh-process bundle checker (contract §15.1) with statement binding and a staged, verifier-owned
challenge (GUIDANCE C1/C2/C3; ``docs/vllm-poc/binding.md``)::

    python -m verity_vllm_adapter.verify <run_dir> --production <dir> --checkpoints manifests/checkpoints.json
        [--challenge security|diagnostic] [--seed N (diagnostic only)] [--q 1] [--s 1/4096] [--coverage-size 1000]
        [--transcript <challenge_transcript.json>]   # re-verify a recorded challenge instead of drawing one

Loads a finished replay+commit run directory (W2 capture format) with **no access to the replay
process**, then, in this order:

1. reads ``description.json``, ``required.json``, ``commitments.json`` (roots fixed on disk) and
   checks first-stage completeness (every request's boundary and every step committed);
2. **binds the statement** (C3, :mod:`verity_vllm_adapter.binding`): the committed weights are
   recomputed from the pinned checkpoint (model), ``description.json``/``required.json`` are
   regenerated from ``config.json`` + the semantic profile (architecture), every boundary root is
   recomputed from ``production.jsonl`` (output);
3. fixes the statement digest ``T_0`` over the profile, the production record, the boundary and
   weights roots, the description/required digests, the binding outcomes and the policy ``(q, s)``,
   and only then obtains ``r_q`` from the :class:`~verity_vllm.challenge.ChallengeSource` -- the
   verifier's own secret seed in **security mode** (default), a public ``--seed`` in **diagnostic
   mode** -- selecting the replay units ``J``; accepts their committed interiors into ``T_2``; obtains
   ``r_s`` and selects the **certified** sample ``T`` (each leaf of ``J`` independently with
   probability ``s``, the core protocol's exact Bernoulli sampler) and, from the same ``r_s``, the
   **coverage** sample (stratified, plus the adjacent chains
   ``qkv_proj.gemm -> rotary_emb.rope -> attn.attn -> o_proj.gemm`` and ``mlp.act_fn.act_mul ->
   mlp.down_proj.gemm``); the whole exchange is written as ``challenge_transcript.json`` and
   re-derived once before use (C1);
4. authenticates every opening it uses and re-executes the registered runtime operator for every
   selected VU (both samples) on the GPU from the opened inputs only
   (:mod:`verity_vllm_adapter.relations_gpu`; W3's sampler relation for ``sampler.select``),
   reporting the two samples separately in ``sampling_plan.json`` (C2);
5. checks required-interface coverage, protected-output linkage (N14 first line), the §15.4
   after-warm-up / changed-contents predicates on the capture;
6. runs the negative tests N01 N02 N03 N04 N06 N07 N13 N14 (W2's harness, adjacent VUs first),
   N15 N16 N17 (a mutated weight / description / production record must fail its binding) and
   N18 (a tampered transcript must fail re-derivation);
7. writes ``verify.json``, ``challenge.json``, ``challenge_transcript.json``, ``sampling_plan.json`` and
   Table 1-4 rows (``table{1,2,3,4}.jsonl``) with contract §9 field names via W2's ``rows`` module;
   ``result`` is derived by W2's verdict rule, never assigned; in diagnostic mode every artifact
   carries ``challenge.mode = "diagnostic"`` and the verdict is rendered ``PASS (diagnostic challenge)``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np

import hashlib
from fractions import Fraction

from verity_vllm import challenge as vchallenge
from verity_vllm import commit as vcommit
from verity_vllm import dtypes, firststage, identity, negatives_state, relations, required as vrequired, rows, sampling
from verity_vllm.capture import CaptureError, CaptureReader, RUReader, WEIGHTS_TREE_ID, boundary_tree_id, ru_tree_id
from verity_vllm.commit import TreeCache
from verity_vllm.description import Description, DescriptionError
from verity_vllm.negatives import NegativeHarness, NegativeResult, PatchedReader, perturb_value
from verity_vllm_adapter import binding as vbinding
from verity_vllm_adapter import profile as prof
from verity_vllm_adapter import verify_bindings as VB  # W1cF: audit fix-ups F2/F3/F4/F5/F7/F9/F10 (one call site each)

# default row stamp (the common verifier's lane); a lane running it passes --worker or sets VERITY_WORKER
WORKER = os.environ.get("VERITY_WORKER", "W1")
FAMILY_CHAIN = [
    ("self_attn.qkv_proj.gemm", 0),
    ("self_attn.rotary_emb.rope", 0),
    ("self_attn.rotary_emb.rope", 1),
    ("self_attn.attn.attn", 0),
    ("self_attn.o_proj.gemm", 0),
    ("mlp.act_fn.act_mul", 0),
    ("mlp.down_proj.gemm", 0),
]
# MoE negatives (W9, contract §8 N10-N12): run on captures with slotted nodes, inapplicable otherwise
MOE_NEGATIVE_IDS = ("N10", "N11", "N12")
# MoE block (W9): router -> ids/weights -> one active expert slot -> combination -> shared add
MOE_CHAIN = [
    ("post_attention_layernorm.rmsnorm", 0),
    ("mlp.gate.gemm", 0),
    ("mlp.experts.cast", 0),
    ("mlp.experts.topk", 0),
    ("mlp.experts.topk_weights", 0),
    ("mlp.experts.expert_out", 0),  # e= filled from the run's router result (first active slot)
    ("mlp.experts.combine", 0),
    ("mlp.experts.add", 0),
]


def production_changes_of(manifest: dict[str, Any]) -> str:
    """Contract §9 ``production_changes`` (``none`` | ``flags:[..]`` | ``source:[..]``) derived from the
    production profile manifest. Stock compiled production is ``none`` (the declared env flags are
    listed). A production engine flag that changes execution (``enforce_eager=True`` -> eager-only
    production; ``compilation_config.custom_ops``) is a **separately declared profile row** (§2: "eager-only
    or graph-off production is a separate row, never a substitute"), so the row says so explicitly
    instead of calling itself stock."""
    env = ", ".join(f"{k}={v}" for k, v in (manifest.get("engine_flags") or {}).items())
    ek = manifest.get("engine_kwargs") or {}
    comp = manifest.get("compilation") or {}
    flags: list[str] = []
    labels: list[str] = []
    if ek.get("enforce_eager") or comp.get("enforce_eager"):
        flags.append("enforce_eager=True")
        labels.append("DECLARED eager-only production profile (compiled execution and CUDA graphs off in production)")
    custom = (ek.get("compilation_config") or {}).get("custom_ops")
    if custom:
        flags.append(f"compilation_config.custom_ops={custom}")
        labels.append(f"DECLARED custom_ops={custom} production profile")
    if not flags:
        return f"none (stock vLLM; declared flags: {env})"
    return (
        f"flags:[{', '.join(flags)}] ({'; '.join(labels)}; NOT stock compiled production; effective compilation mode "
        f"{comp.get('mode')}, cudagraph_mode {comp.get('cudagraph_mode')}, custom_ops {comp.get('custom_ops')}; declared env flags: {env})"
    )


class FastNegativeHarness(NegativeHarness):
    """W2's harness, rebuilding only the trees a mutation touches (roots are independent per tree)."""

    # state rows (W10): the cached-state (R3) and recompute (R4) variants of N06 / N05; both are
    # inconclusive (not attempted) on captures without cached prefixes / recompute units
    def n06(self, limit: int = 6) -> Any:
        result = super().n06(limit)
        negatives_state.extend_n06_cached_state(self, result)
        return self._finish(result)

    def n05(self) -> Any:
        result = super().n05()
        negatives_state.extend_n05_recompute(self, result)
        return self._finish(result)

    def _rebuild(self, patched: CaptureReader) -> dict[str, Any]:
        if not isinstance(patched, PatchedReader):
            return super()._rebuild(patched)
        doc = json.loads(json.dumps(self.commitments))
        touched: set[str] = set()
        for cid in list(patched._patch_values) + list(patched._patch_entries):
            tid = self.reader.tree_of(cid)
            if tid:
                touched.add(tid)
        if patched._patch_weights:
            touched.add(WEIGHTS_TREE_ID)
        for rid in patched._patch_boundaries:
            touched.add(boundary_tree_id(rid))
        cache = TreeCache(patched, recompute_leaves=True)
        for tid in touched:
            old = doc["trees"].get(tid)
            if old is None:
                continue
            tree = cache.tree(tid)
            meta = {k: v for k, v in old.items() if k not in ("kind", "root", "leaf_count", "ids_digest")}
            doc["trees"][tid] = tree.record(old["kind"], **meta)
        return doc


def adjacent_chain(description: Description, reader: CaptureReader, profile_id: str, layer: str, prefer_phase: str) -> list[str]:
    """Value ids of the adjacent chain at one (request, position) that is present in the capture."""
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        for step in b.steps:
            if step.phase != prefer_phase:
                continue
            t = step.positions[-1]
            ids = []
            ok = True
            for suffix, port in FAMILY_CHAIN:
                op = f"layers.{layer}.{suffix}"
                if op not in description:
                    ok = False
                    break
                cid = str(identity.value_id(profile_id, rid, t, step.phase, layer, op, port))
                if reader.locate(cid) is None:
                    ok = False
                    break
                ids.append(cid)
            if ok:
                return ids
    return []


def moe_chain(description: Description, reader: CaptureReader, profile_id: str, layer: str, prefer_phase: str) -> list[str]:
    """The MoE adjacent chain at one (request, position) present in the capture; the expert slot is
    the first slot the run's router activated there (read from the capture's family record)."""
    if f"layers.{layer}.mlp.experts.expert_out" not in description:
        return []
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        for step in b.steps:
            if step.phase != prefer_phase:
                continue
            t = step.positions[-1]
            ids: list[str] = []
            ok = True
            for suffix, port in MOE_CHAIN:
                op = f"layers.{layer}.{suffix}"
                if op not in description:
                    ok = False
                    break
                cid = str(identity.value_id(profile_id, rid, t, step.phase, layer, op, port))
                if op.endswith(".expert_out"):
                    ru = reader.ru(rid, step.step)
                    fam = next((f for f in ru.index.families if f["family"] == cid), None)
                    if fam is None or not fam["active"]:
                        ok = False
                        break
                    cid = f"{cid}/e={int(fam['active'][0])}"
                if reader.locate(cid) is None:
                    ok = False
                    break
                ids.append(cid)
            if ok:
                return ids
    return []


def slot_kind(reader: CaptureReader, cid: str) -> str | None:
    """'active' | 'inactive' for a per-slot value id (from the capture's entry origin), else None."""
    try:
        if identity.parse(cid).expert is None:
            return None
        loc = reader.locate(cid)
        if loc is None:
            return None
        return "inactive" if reader.ru(*loc).entry(cid).derived is not None else "active"
    except Exception:  # noqa: BLE001
        return None


def capture_predicates(description: Description, reader: CaptureReader) -> dict[str, Any]:
    """§15.4: expected capture count per request, identities, changed contents across requests."""
    profile_id = reader.profile_id
    out: dict[str, Any] = {"requests": 0, "expected_leaves": 0, "present_leaves": 0, "missing_examples": [], "steps_expected_equal": True}
    out["unexpected_leaves"] = 0
    out["derived_leaves_present"] = 0
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        expected_by_step: dict[int, set[str]] = {}
        for node in description.nodes:  # node by node: a slotted node's population is large
            for inst in description.instances(b, ops=[node.op]):
                step = b.step_of(inst.t)
                if step is not None:
                    expected_by_step.setdefault(step.step, set()).add(inst.output_id(profile_id))
        out["requests"] += 1
        committed_steps = {s for (_, s) in reader.ru_keys(rid)}
        if committed_steps != {s.step for s in b.steps}:
            out["steps_expected_equal"] = False
        for (_, step) in reader.ru_keys(rid):
            ru = reader.ru(rid, step)
            present = {e.canonical_id for e in ru.index.entries}
            out["derived_leaves_present"] += sum(1 for e in ru.index.entries if e.derived is not None)
            expected = expected_by_step.get(step, set())
            out["expected_leaves"] += len(expected)
            out["present_leaves"] += len(expected & present)
            out["unexpected_leaves"] += len(present - expected)
            for cid in sorted(expected - present)[:5]:
                if len(out["missing_examples"]) < 5:
                    out["missing_examples"].append(cid)
    # changed contents where predicted: the same node (repeated shape) at the same position in two
    # requests must differ at the first position where their prompt tokens differ; where the two
    # prompts share a prefix the contents must be equal there (causal model, batch-invariant profile).
    rids = reader.request_ids()
    op = "layers.0.self_attn.qkv_proj.gemm"
    out["changed_contents_between_requests"] = None
    out["equal_contents_on_shared_prefix"] = None
    out["changed_contents_detail"] = None
    if len(rids) >= 2 and op in description:
        def row(rid: str, t: int) -> bytes | None:
            b = reader.boundary(rid)
            step = b.step_of(t)
            if step is None:
                return None
            cid = str(identity.value_id(profile_id, rid, t, step.phase, "0", op, 0))
            return reader.value(cid).data if reader.locate(cid) is not None else None

        def common_prefix(x: str, y: str) -> int:
            px, py = reader.boundary(x).prompt_token_ids, reader.boundary(y).prompt_token_ids
            return next((i for i, (u, v) in enumerate(zip(px, py)) if u != v), min(len(px), len(py)))

        # the pair with the longest shared prefix exercises both predicates
        a, b = max(((x, y) for i, x in enumerate(rids) for y in rids[i + 1:]), key=lambda xy: common_prefix(*xy))
        pa, pb = reader.boundary(a).prompt_token_ids, reader.boundary(b).prompt_token_ids
        k = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), min(len(pa), len(pb)))
        detail = {"requests": [a, b], "first_differing_position": k, "op": op}
        va, vb = row(a, k), row(b, k)
        if va is not None and vb is not None:
            out["changed_contents_between_requests"] = va != vb
        if k > 0:
            ua, ub = row(a, k - 1), row(b, k - 1)
            if ua is not None and ub is not None:
                out["equal_contents_on_shared_prefix"] = ua == ub
                detail["shared_prefix_position"] = k - 1
        out["changed_contents_detail"] = detail
    return out


# --- staged challenge helpers --------------------------------------------------------------------


def statement_of(reader: CaptureReader, description: Description, required_doc: dict[str, Any] | None,
                 commitments: dict[str, Any], bindings: vbinding.Bindings, policy: vchallenge.Policy,
                 source_opening: dict[str, Any], first_stage: dict[str, Any]) -> dict[str, Any]:
    """Everything the prover is bound to before ``r_q`` exists (``T_0``): the registered profile, the
    production record, the boundary and weights roots, the description/required digests, the binding
    outcomes, the policy and the source's seed commitment.  The replay-unit (interior) roots are
    deliberately *not* here; they enter ``T_2``."""
    trees = commitments.get("trees", {})
    return {
        "profile_id": reader.profile_id,
        "manifest_digest": hashlib.sha256((reader.run_dir / "manifest.json").read_bytes()).hexdigest(),
        "description_digest": description.digest().hex(),
        "required_digest": vrequired.required_digest(required_doc).hex() if required_doc else None,
        "boundary_roots": {tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
                           for tid, rec in sorted(trees.items()) if rec.get("kind") == "boundary"},
        "weights_root": {k: trees[WEIGHTS_TREE_ID].get(k) for k in ("root", "leaf_count", "ids_digest", "weights_sha256")} if WEIGHTS_TREE_ID in trees else None,
        "replay_units_declared": sorted(tid for tid, rec in trees.items() if rec.get("kind") == "ru"),
        "first_stage": {"ok": first_stage["ok"], "requests": first_stage["requests"], "steps": first_stage["steps"]},
        "bindings": bindings.statement(),
        "policy": policy.to_json(),
        "source": source_opening,
    }


# the population helpers live in verity_vllm.challenge so that `python -m verity_vllm.bundle rederive`
# (CPU only, no adapter, no weights) rebuilds exactly what the verifier sampled over
interiors_of = vchallenge.interiors_of
make_resolver = vchallenge.make_resolver
make_coverage = vchallenge.make_coverage


def plan_stats(report: relations.CheckReport, ids: list[str]) -> dict[str, Any]:
    wanted = set(ids)
    statuses: dict[str, int] = {}
    classes: dict[str, dict[str, int]] = {}
    for r in report.results:
        if r.canonical_id not in wanted:
            continue
        statuses[r.status] = statuses.get(r.status, 0) + 1
        k = classes.setdefault(r.klass or "none", {"exact": 0, "attempted": 0})
        if r.status in ("exact", "mismatch"):
            k["attempted"] += 1
        if r.status == "exact":
            k["exact"] += 1
    return {"vus": len(ids), "statuses": statuses, "relation_checks": classes}


def transcript_negatives(transcript: dict[str, Any], *, statement: dict[str, Any], interiors: dict[str, Any], request_ids: list[str],
                         units: vchallenge.RankedUnits, resolve_ids, coverage, out_dir: Path) -> dict[str, Any]:
    """N18: a transcript whose selections are not what its randomness yields must fail re-derivation."""
    variants = []

    def attempt(label: str, mutate) -> None:
        t = json.loads(json.dumps(transcript))
        mutate(t)
        r = vchallenge.reverify(t, statement=statement, interiors=interiors, request_ids=request_ids, units=units, resolve_ids=resolve_ids, coverage=coverage)
        variants.append({"variant": label, "outcome": "rejected" if not r["ok"] else "accepted", "detail": "; ".join(r["problems"])[:300]})

    sc = transcript["sample_challenge"]
    attempt("replace the certified sample by the prover's choice (drop its first VU)", lambda t: t["sample_challenge"]["certified_sample"].pop(0))
    attempt("swap one selected rank for a neighbour", lambda t: t["sample_challenge"]["selected_ranks"].__setitem__(0, t["sample_challenge"]["selected_ranks"][0] + 1))
    attempt("substitute another r_s (prover-chosen randomness)", lambda t: t["sample_challenge"].__setitem__("r_s", "00" * 32))
    attempt("substitute another r_q", lambda t: t["replay_challenge"].__setitem__("r_q", "11" * 32))
    attempt("reveal a different seed", lambda t: t["reveal"].__setitem__("seed", "22" * 32))
    attempt("drop one VU from the coverage sample", lambda t: t["sample_challenge"]["coverage_sample"].pop(-1))
    attempt("claim a different statement digest (T_0)", lambda t: t["phases"][0].__setitem__("digest", "33" * 32))
    if len(request_ids) > 1 and transcript["replay_challenge"]["selected"]:
        attempt("drop one replay unit from J", lambda t: (t["replay_challenge"]["selected"].pop(0), t["replay_challenge"]["selected_indices"].pop(0)))
    if transcript.get("reveal", {}).get("draws"):
        # admissibility (3), uniqueness: a second draw for the same purpose, or a draw bound to another phase
        attempt("record a second r_s draw for the same purpose (keep the favourable one)",
                lambda t: t["reveal"]["draws"].append({**t["reveal"]["draws"][-1], "r": "55" * 32}))
        attempt("rebind the recorded r_s draw to another phase digest",
                lambda t: t["reveal"]["draws"][-1].__setitem__("phase_digest", "66" * 32))
    del sc
    rejected = sum(1 for v in variants if v["outcome"] == "rejected")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "N18.json"
    path.write_text(json.dumps({"negative_id": "N18", "variants": variants, "attempted": len(variants), "rejected": rejected,
                                "unexpectedly_accepted": len(variants) - rejected}, indent=1) + "\n")
    return {"negative_id": "N18", "attempted": len(variants), "rejected": rejected, "unexpectedly_accepted": len(variants) - rejected,
            "reproducer": str(path), "variants": variants, "inconclusive": 0}


def load_description(reader: CaptureReader, known_relations: Iterable[str] | None = None, *,
                     description: Description | None = None) -> Description:
    """The bundle's ``description.json``, validated: its structure (symbols, node references, ports) and,
    unless ``known_relations=()``, its relation ids against the registry (default: every registered
    relation plus ``unresolved``).  Pass ``description=`` to re-validate an already loaded one (the
    relation ids, once the runtime relations are registered).  A description that does not validate --
    WA's M8b, a dropped layer leaves ``norm.rmsnorm`` referring to a node that is gone -- is the structural
    check ``description validation`` failing: :class:`VB.StructuralFailure`, which :func:`run` writes as
    ``FAIL structural: description validation: …`` (WB2, audit finding 4)."""
    known: Iterable[str] | None
    if known_relations is None:
        known = set(relations.ids()) | {"unresolved"}
    elif not tuple(known_relations):
        known = None  # structure only
    else:
        known = known_relations
    try:
        if description is None:
            description = Description(reader.description)
        description.validate(known_relations=known)
    except DescriptionError as e:
        raise VB.StructuralFailure("description validation", str(e)) from e
    except (CaptureError, KeyError, TypeError, ValueError) as e:
        raise VB.StructuralFailure("description validation", f"{type(e).__name__}: {e}") from e
    return description


def checkpoint_reference(case: str, checkpoints_path: str, weights_from_model: str | None) -> dict[str, Any]:
    """The pinned checkpoint record (`manifests/checkpoints.json`) with its files located: at the manifest's
    ``local_path`` (the recording pod's layout) or, for a verifier on another machine, at ``--weights-from-model``.
    Either way the files' sha256 are checked against the frozen ones by model binding, so the location is a
    convenience and the *reference* stays the manifest's hashes."""
    if weights_from_model is None:
        return prof.checkpoint_for(case, checkpoints_path)
    repo = prof.CASES[case]["repo"]
    cp = dict(prof.load_checkpoints(checkpoints_path)[repo])
    local = Path(weights_from_model)
    missing = [f for f in list(cp["safetensors_sha256"]) + ["config.json"] if not (local / f).exists()]
    if missing:
        raise SystemExit(f"--weights-from-model {local}: missing {missing}")
    cp["local_path"] = str(local)
    cp["downloaded"] = True
    return cp


# --- MoE negatives N10-N12 (contract §8), through W2's overlay harness ---------------------------


class TrimmedRU(RUReader):
    """A replay unit whose derived (inactive-slot) entries are dropped: the population as an
    adversary would like to present it, active slots only (N12)."""

    def __init__(self, base: RUReader) -> None:
        self.dir = base.dir
        self.request_id = base.request_id
        self.step = base.step
        kept = [e for e in base.index.entries if e.derived is None]
        self.index = type(base.index)(
            request_id=base.index.request_id, step=base.index.step, phase=base.index.phase, positions=base.index.positions,
            entries=kept, profile_id=base.index.profile_id, conflicts=base.index.conflicts, tainted=base.index.tainted,
            values_nbytes=base.index.values_nbytes, families=[],
        )
        self._entries = {e.canonical_id: e for e in kept}
        self._values_path = base._values_path
        self._mm = None


class TrimmedReader(PatchedReader):
    def __init__(self, base: CaptureReader, key: tuple[str, int]) -> None:
        super().__init__(base)
        self._trim_key = key

    def ru(self, request_id: str, step: int) -> RUReader:
        key = (request_id, step)
        if key == self._trim_key:
            if key not in self._rus:
                self._rus[key] = TrimmedRU(self._base.ru(request_id, step))
            return self._rus[key]
        return super().ru(request_id, step)


def moe_negatives(harness: FastNegativeHarness, description: Description, reader: CaptureReader, report: relations.CheckReport,
                  sample: list[str], limit: int, which: set[str]) -> list[NegativeResult]:
    """N10 activity decision corrupted without a consistent router result; N11 non-canonical value at
    an inactive slot; N12 population presented with active slots only."""
    out: list[NegativeResult] = []
    slotted = [n for n in description.nodes if n.experts]
    if not slotted:
        return out
    active = [c for c in sample if slot_kind(reader, c) == "active"][:limit]
    inactive = [c for c in sample if slot_kind(reader, c) == "inactive"][:limit]

    def outcome(r: relations.VUResult, rep: relations.CheckReport) -> str:
        auth_ok = rep.openings_verified == rep.openings_attempted > 0
        if r.status == "exact":
            return "accepted"
        if r.status in ("mismatch", "missing", "structural", "schema") and (auth_ok or r.status != "mismatch"):
            return "rejected"
        return "inconclusive"

    if "N10" in which:
        res = NegativeResult("N10")
        for cid in active:
            c = identity.parse(cid)
            loc = reader.locate(cid)
            assert loc is not None
            ru = reader.ru(*loc)
            fam_id = cid.rsplit("/e=", 1)[0]
            fam = next((f for f in ru.index.families if f["family"] == fam_id), None)
            if fam is None:
                res.record("move_active_output_to_inactive_slot", "inconclusive", canonical_id=cid, reason="no family record")
                continue
            inactive_slots = [e for e in range(int(fam["slots"])) if e not in set(fam["active"])]
            target = f"{fam_id}/e={inactive_slots[len(res.variants) % len(inactive_slots)]}"
            value = reader.value(cid)
            zeros = bytes(len(value.data))
            # (a) the captured expert output appears at a slot the router did not select
            patched = PatchedReader(reader, values={target: value.data})
            commitments = harness._rebuild(patched)
            rep = harness._check(patched, commitments, [target])
            r = rep.results[0]
            res.record("move_active_output_to_inactive_slot", outcome(r, rep), canonical_id=target, source=cid,
                       relation=r.relation_id, status=r.status, detail=r.detail)
            # (b) the selected slot is presented as inactive (zero) while the router still names it
            patched = PatchedReader(reader, values={cid: zeros})
            commitments = harness._rebuild(patched)
            rep = harness._check(patched, commitments, [cid])
            r = rep.results[0]
            res.record("zero_active_slot_router_unchanged", outcome(r, rep), canonical_id=cid,
                       relation=r.relation_id, status=r.status, detail=r.detail)
            # (c) the router's ids changed to name the moved slot, logits unchanged: the router
            # relation rejects the ids (and the combination/expert checks would follow)
            ids_cid = str(identity.value_id(c.profile_id or "", c.request_id or "", c.t or 0, c.phase or "", c.layer or "",
                                            fam_id.split("/op=")[1].split("/")[0].rsplit(".", 1)[0] + ".topk", 0))
            if reader.locate(ids_cid) is not None:
                ids_val = reader.value(ids_cid)
                arr = ids_val.array().reshape(-1).copy()
                k = int(np.flatnonzero(arr == (c.expert or 0))[0]) if (arr == (c.expert or 0)).any() else 0
                arr[k] = int(target.rsplit("e=", 1)[1])
                patched = PatchedReader(reader, values={ids_cid: dtypes.array_to_bytes(arr.reshape(ids_val.shape))})
                commitments = harness._rebuild(patched)
                rep = harness._check(patched, commitments, [ids_cid])
                r = rep.results[0]
                res.record("router_ids_changed_logits_unchanged", outcome(r, rep), canonical_id=ids_cid,
                           relation=r.relation_id, status=r.status, detail=r.detail)
        out.append(harness._finish(res))

    if "N11" in which:
        res = NegativeResult("N11")
        for i, cid in enumerate(inactive):
            value = reader.value(cid)
            n = len(value.data) // 2
            if i % 2 == 0:  # negative zero: a non-canonical encoding of the zero contribution
                bad = np.full(n, 0x8000, dtype=np.uint16).tobytes()
                variant = "negative_zero_at_inactive_slot"
            else:  # a small non-zero contribution
                bad = perturb_value(value, harness.rng)
                variant = "nonzero_at_inactive_slot"
            patched = PatchedReader(reader, values={cid: bad})
            commitments = harness._rebuild(patched)
            rep = harness._check(patched, commitments, [cid])
            r = rep.results[0]
            res.record(variant, outcome(r, rep), canonical_id=cid, relation=r.relation_id, status=r.status, detail=r.detail)
        out.append(harness._finish(res))

    if "N12" in which:
        res = NegativeResult("N12")
        for cid in inactive[: max(1, limit // 2)]:
            loc = reader.locate(cid)
            assert loc is not None
            trimmed = TrimmedReader(reader, loc)
            tid = ru_tree_id(loc[0], int(loc[1]))
            # a valid commitment over the trimmed population: rebuild only that unit's tree
            doc = json.loads(json.dumps(harness.commitments))
            tree = TreeCache(trimmed, recompute_leaves=True).tree(tid)
            old = doc["trees"][tid]
            doc["trees"][tid] = tree.record(old["kind"], **{k: v for k, v in old.items() if k not in ("kind", "root", "leaf_count", "ids_digest")})
            commitments = doc
            # (a) the inactive slot is simply not there: the check reports it missing
            rep = harness._check(trimmed, commitments, [cid])
            r = rep.results[0]
            res.record("inactive_slot_absent_from_population", outcome(r, rep), canonical_id=cid, status=r.status, detail=r.detail)
            # (b) the structural population check: expected slots of the description vs present
            cov = vrequired.coverage(reader.required, description, trimmed, request_ids=[loc[0]]) if reader.has("required.json") else None
            missing_slots = sum(m["count"] for m in (cov.missing if cov else []) if "expert_out" in m["interface"][0])
            res.record("population_of_active_slots_only_vs_description", "rejected" if missing_slots > 0 else "accepted",
                       replay_unit=tid, missing_slot_leaves=missing_slots,
                       bound=cov.bound if cov else None, required=cov.required if cov else None)
        out.append(harness._finish(res))
    return out


def moe_table2_rows(common: rows.Common, report: relations.CheckReport, description: Description, reader: CaptureReader,
                    runtime_impl: dict[str, str]) -> tuple[list[dict[str, Any]], list[relations.VUResult], dict[str, Any]]:
    """Table 2 rows for slotted (expert) nodes with active and inactive slots kept apart; returns the
    rows, the results they consumed, and the Table 4 moe_padding figures."""
    slotted = {n.op for n in description.nodes if n.experts}
    if not slotted:
        return [], [], {"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None}
    consumed: list[relations.VUResult] = []
    groups: dict[tuple[str, str, str], list[relations.VUResult]] = {}
    for r in report.results:
        if r.op in slotted:
            consumed.append(r)
            kind = slot_kind(reader, r.canonical_id) or "unknown"
            groups.setdefault((r.op, identity.parse(r.canonical_id).phase or "?", kind), []).append(r)
    out = []
    inactive_checks = inactive_work = inactive_bytes = 0
    for (op, phase, kind), results in sorted(groups.items()):
        node = description.node(op)
        exact = sum(1 for r in results if r.status == "exact")
        attempted = sum(1 for r in results if r.status in ("exact", "mismatch"))
        unresolved = sum(1 for r in results if r.status in ("missing", "structural", "schema"))
        worst = min(results, key=rows._status_rank).status
        klass = results[0].klass or "none"
        checker = klass if attempted else "none"
        if kind == "inactive":
            checker = "independent" if attempted else "none"  # the zero prescription needs no runtime
            inactive_checks += attempted
            inactive_work += sum(r.checker_seconds for r in results)
            inactive_bytes += sum(r.output_bytes for r in results)
        out.append(rows.table2_row(
            common, op_id=f"{op}[{kind}]", phase=phase, shape=list(node.shape), dtype=node.dtype, layout_or_quant=None,
            runtime_impl=runtime_impl.get(op, node.runtime_impl or "unresolved") if kind == "active" else "none (derived zero contribution)",
            primitive_semantics=node.relation, decomposition="row",
            interface={"input_bytes": max((r.input_bytes for r in results), default=0), "output_bits": max((r.output_bytes for r in results), default=0) * 8, "range": "slot"},
            origin="captured" if kind == "active" else "derived",
            binding={"present": len(results) - unresolved, "unresolved": unresolved},
            checker=checker, natural_checks={"exact": exact, "attempted": attempted},
            stress_checks={"exact": 0, "attempted": 0, "domain": None}, mutations={"rejected": 0, "attempted": 0},
            checker_work={"seconds": float(sum(r.checker_seconds for r in results)), "flops_or_bytes": float(sum(r.input_bytes + r.output_bytes for r in results)),
                          "includes_full_op": any(r.includes_full_op for r in results)},
            status=worst,
            limitation=None if kind == "active" else "activity decided from the checked router ids; derived leaf, no bytes captured",
        ))
    summary = json.loads((Path(reader.run_dir) / "capture_summary.json").read_text()).get("moe") or {} if (Path(reader.run_dir) / "capture_summary.json").exists() else {}
    padding = {
        "inactive_units": int(summary.get("inactive_slots") or 0) or None,
        "inactive_checks": inactive_checks or None,
        "inactive_work_s": float(inactive_work) if inactive_checks else None,
        "inactive_bytes": int(inactive_bytes) if inactive_checks else None,
    }
    return out, consumed, padding


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--challenge", choices=list(vchallenge.MODES), default="security",
                    help="security: the verifier's own secret seed (default); diagnostic: a public --seed, verdict qualified")
    ap.add_argument("--seed", type=int, default=None, help="public seed; diagnostic mode only")
    ap.add_argument("--q", default="1", help="replay-unit (request) selection probability, a fraction")
    ap.add_argument("--s", default="1/4096", help="verification-unit selection probability inside J, a fraction")
    ap.add_argument("--coverage-size", type=int, default=1000, help="stratified coverage sample size (plus adjacent chains)")
    ap.add_argument("--sample-size", type=int, default=None, help="alias of --coverage-size (older run_case.sh)")
    ap.add_argument("--transcript", default=None, help="re-verify: re-derive and reuse this recorded challenge_transcript.json")
    # N10-N12 (W9) run only on captures with slotted (MoE) nodes; a dense capture skips them
    ap.add_argument("--negatives", default="N01,N02,N03,N04,N06,N07,N13,N14,N15,N16,N17,N18,N19,N10,N11,N12")
    ap.add_argument("--negative-limit", type=int, default=4)
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--out", default=None, help="verify output dir (default <run_dir>/verify)")
    ap.add_argument("--production", required=True, help="production dir (production.jsonl; timings/steptrace when present)")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json", help="the pinned checkpoints (the verifier's reference)")
    ap.add_argument("--weights-from-model", default=None,
                    help="directory holding the pinned checkpoint's files (config.json, *.safetensors) when it is not at "
                         "the checkpoints manifest's local_path; its file shas must equal the frozen ones")
    ap.add_argument("--semantic-profile", default="auto",
                    help="path, or 'auto' = the one shipped profile covering the checkpoint's architecture (binding.select_semantic_profile)")
    ap.add_argument("--noninterference", default=None, help="compare.json from verity_vllm_adapter.compare")
    ap.add_argument("--adapter-revision", default="unknown")
    ap.add_argument("--case", default=None)
    ap.add_argument("--worker", default=os.environ.get("VERITY_WORKER", WORKER), help="worker stamped on the rows (contract section 9 common fields)")
    ap.add_argument("--weights-resolver", choices=["none", "checkpoint"], default="none",
                    help="checkpoint: the bundle stores weight hashes only; values are re-derived from the checkpoint through vLLM's loader "
                         "and re-hashed against the index (W1d compiled_fx.weights_resolver; needs --checkpoints)")
    args = ap.parse_args(argv)
    if args.sample_size is not None:
        args.coverage_size = args.sample_size

    t_start = time.perf_counter()
    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else run_dir / "verify"
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    # W1cF F10: Triton writes one <fn name>.ptx per compiled variant into TRITON_CACHE_DIR (read at each compile);
    # give it a fresh, known dir unless the caller pinned one, so the PTX this process compiles is collectable.
    triton_cache_dir = os.environ.setdefault("TRITON_CACHE_DIR", str(out / "triton-cache"))
    # Likewise Inductor's cache (its autotune cache would otherwise hand the fused-kernel loader another process's
    # best_config as the only config). A caller who sets either keeps it; what was already there is recorded.
    inductor_cache_dir = os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(out / "inductor-cache"))
    triton_cache_before = VB.ptx_paths(triton_cache_dir)

    # The checker re-executes the registered runtime operators *under the profile*: the bundle
    # manifest's declared engine flags are applied to this process before vLLM is imported. Found
    # the hard way (B1/p1, experiment B1-c37c3f0b8363-20260907T1036Z): without VLLM_BATCH_INVARIANT=1
    # the CUDA launcher of _C.fused_add_rms_norm picks the vectorised kernel (different variance
    # summation order) and one rmsnorm row in 910 came back 1 ulp off in 3 of 1536 elements
    # (verity_vllm_adapter.repro_rmsnorm). A conflicting value already in the environment aborts:
    # the row would not be a same_runtime check.
    engine_flags = dict((json.loads((run_dir / "manifest.json").read_text()).get("engine_flags") or {}))
    for k, v in engine_flags.items():
        cur = os.environ.get(k)
        if cur is not None and cur != str(v):
            raise SystemExit(f"environment {k}={cur!r} conflicts with the profile's declared {k}={v!r}; refusing to check under a different runtime configuration")
        os.environ[k] = str(v)

    # WB2 (audit finding 4): the bundle's documents are loaded and structurally validated *before* the GPU
    # relations are imported, so a bundle whose description does not validate (WA's M8b: a dropped layer)
    # or whose commitments are malformed is a `FAIL structural: <check>: …` row from any environment, and
    # never a traceback.  The relation ids are checked once the runtime relations are registered (below).
    try:
        reader = CaptureReader(run_dir)
    except CaptureError as e:
        raise VB.StructuralFailure("capture format", str(e)) from e
    description = load_description(reader, known_relations=())  # structure only; relation ids after install()
    try:
        commitments = vcommit.load_commitments(reader)
    except vcommit.CommitError as e:
        raise VB.StructuralFailure("commitments", str(e)) from e

    import verity_vllm_adapter.relations_gpu as gpu_rel

    installed = gpu_rel.install()
    # A bundle may name an additional relation module (W1d compiled-graph relations launch the bundle's
    # kernels/<sha>.py sources); it is imported here and told where the bundle lives.
    extra_module = reader.description.get("relations_module") if isinstance(reader.description, dict) else None
    if extra_module:
        import importlib

        os.environ.setdefault("VERITY_RUN_DIR", str(run_dir))
        installed += importlib.import_module(extra_module).install()
    load_description(reader, description=description)  # now the relation ids too
    # §16.1: a compiled-profile bundle ships the generated kernel sources it launched (kernels/<sha256>.py);
    # the fused-region relation launches them standalone. Files must hash to their names.
    kernel_sources = None
    if (run_dir / "kernels").is_dir():
        import verity_vllm_adapter.relations_inductor as ind_rel

        kernel_sources = ind_rel.load_sources(run_dir / "kernels")
        if kernel_sources.get("bad_hash"):
            # W1d compiled-fx bundles (`compiled_fx.capture.write_kernels`) key `kernels/<sha256[:16]>.py` by the
            # *name-normalised* source and say so in kernels/index.json ("hash"); W1c's rule above (full sha256 of the
            # file content) refused every such bundle once the two lanes were merged (CF2, 2026-09-08: the same refusal on
            # the H100 bundle and on W1d's archived A100 bundle that had PASSed on W1d's branch). Re-check under the
            # bundle's declared contract; a file that hashes under neither is still refused.
            idx = run_dir / "kernels" / "index.json"
            contract = json.loads(idx.read_text()).get("hash", "") if idx.exists() else ""
            if "name-normalised" in contract:
                from verity_vllm_adapter.compiled_fx.kernels import normalise_kernel

                still_bad = [n for n in kernel_sources["bad_hash"]
                             if hashlib.sha256(normalise_kernel((run_dir / "kernels" / n).read_text()).encode()).hexdigest()[:16] != Path(n).stem]
                kernel_sources["loaded"] += len(kernel_sources["bad_hash"]) - len(still_bad)
                kernel_sources["bad_hash"] = still_bad
                kernel_sources["contract"] = contract
        if kernel_sources.get("bad_hash"):
            raise SystemExit(f"bundle kernels/ files whose content does not hash to their name: {kernel_sources['bad_hash']}")
        print(f"kernel sources loaded: {kernel_sources['loaded']} from {run_dir / 'kernels'}", flush=True)

    weights_source: dict[str, Any] | None = None
    weights_resolver = None  # also installed on the negative harness' own reader (its PatchedWeights inherit it)
    if args.weights_resolver == "checkpoint":  # W1d: hash-only weights re-derived through vLLM's loader
        from verity_vllm_adapter import profile as prof
        from verity_vllm_adapter.compiled_fx.weights_resolver import CheckpointWeightResolver

        case_for_cp = args.case or reader.manifest.get("case", "B0")
        weights_resolver = CheckpointWeightResolver(prof.checkpoint_for(case_for_cp, args.checkpoints), reader.weights.index.source_sha256)
        reader.weights.resolver = weights_resolver
        weights_resolver._load()  # before the timed phases: checks the checkpoint files' sha256, loads the model once
        weights_source = weights_resolver.to_json()
    if reader.has_weights():
        widx = reader.weights.index
        referenced = [e for e in widx.entries if e.offset is None]
        if referenced and not args.weights_from_model:
            raise SystemExit(f"{len(referenced)} weights are referenced by checkpoint sha, not stored: pass --weights-from-model <checkpoint dir>")
        if referenced:  # §16.8 bundle (W1c): entries referenced by checkpoint sha, rebuilt by weight_sources
            from verity_vllm_adapter import weight_sources

            expected = None
            if args.checkpoints and args.case:
                import verity_vllm_adapter.profile as prof

                expected = prof.checkpoint_for(args.case, args.checkpoints).get("safetensors_sha256")
            weights_resolver = weight_sources.make_resolver(args.weights_from_model, expected_file_sha256=expected)
            reader.weights.resolver = weights_resolver
            weights_source = {"model_dir": args.weights_from_model, "referenced": len(referenced), "stored": len(widx.entries) - len(referenced),
                              "file_sha256_recorded": widx.source_sha256, "manifest_sha256": expected}
            print(f"weights: {len(referenced)} referenced to checkpoint (rebuilt from {args.weights_from_model}), {len(widx.entries) - len(referenced)} stored", flush=True)
    required_doc = reader.required if reader.has("required.json") else None
    profile_id = reader.profile_id
    manifest = reader.manifest
    case = args.case or manifest.get("case", "B?")
    prod_dir = Path(args.production)
    production_path = prod_dir / "production.jsonl"
    request_ids = reader.request_ids()
    structural_problems: list[str] = []
    # W1cF F2/F3: structural bindings of the bundle to production's record (mismatch = FAIL structural)
    prod_kernels_doc = None
    if args.production and (Path(args.production) / "kernels.json").exists():
        prod_kernels_doc = json.loads((Path(args.production) / "kernels.json").read_text())
    w1cf_bindings = {"kernel_set": VB.kernel_set_binding(run_dir, manifest, prod_kernels_doc),
                     "description_digest": VB.description_binding(required_doc, description)}
    for bname, b in w1cf_bindings.items():
        if b is not None:
            print(f"binding {bname}: {'ok' if b['ok'] else 'MISMATCH'} {b.get('problem') or ''}", flush=True)

    # --- 1. first-stage completeness (every boundary and every step committed) before anything is drawn ---
    first_stage = firststage.first_stage_problems(reader, commitments)
    structural_problems += [f"first-stage: {p}" for p in first_stage.problems[:3]]

    # --- 2. statement binding (C3): the verifier's references, not the prover's claims ---
    t0 = time.perf_counter()
    cp = checkpoint_reference(case, args.checkpoints, args.weights_from_model)
    if args.semantic_profile == "auto":  # resolved once, from the pinned checkpoint's config, before any binding step
        args.semantic_profile = str(vbinding.select_semantic_profile(json.loads((Path(cp["local_path"]) / "config.json").read_text()), reader.manifest))
        print(f"semantic profile: {Path(args.semantic_profile).name} (selected by architecture + manifest requirements)", flush=True)
    # Weight values may be absent from a self-contained bundle (`weights/values.bin` omitted): the reader
    # then rebuilds each committed weight from the pinned checkpoint through the profile's weight map and
    # checks its sha256 against the index (binding.checkpoint_resolver). Stored values are read as before.
    if weights_resolver is None:  # not a §16.8 bundle: WB's resolver (stored values read as before; omitted values.bin rebuilt)
        weights_resolver = vbinding.checkpoint_resolver(
            cp=cp, config=json.loads((Path(cp["local_path"]) / "config.json").read_text()),
            profile=vbinding.load_semantic_profile(args.semantic_profile))
        reader.weights.resolver = weights_resolver
    weights_values_stored = (run_dir / "weights" / "values.bin").exists()
    print(f"weights: values {'stored in the bundle' if weights_values_stored else 'NOT in the bundle; rebuilt from the pinned checkpoint on demand'}", flush=True)
    bindings = vbinding.bind(reader, commitments, cp=cp, production_path=production_path, semantic_profile_path=args.semantic_profile)
    timing["bindings_s"] = time.perf_counter() - t0
    structural_problems += bindings.problems
    for r in (bindings.model, bindings.architecture, bindings.output):
        print(f"binding {r.name:12s} {'ok ' if r.ok else 'FAILED'} {r.seconds:6.1f}s  {r.detail[:220]}", flush=True)

    # --- 3. the staged challenge (C1): T_0 -> r_q -> J -> T_2 -> r_s -> (certified, coverage) ---
    t0 = time.perf_counter()
    policy = vchallenge.Policy.parse(args.q, args.s)
    chain_prefill = adjacent_chain(description, reader, profile_id, "0", "prefill")
    chain_decode = adjacent_chain(description, reader, profile_id, "0", "decode")
    chain_mid = adjacent_chain(description, reader, profile_id, str(max(0, len(description.nodes_of_type("attention")) // 2)), "decode")
    # MoE block (W9): router -> ids/weights -> one active expert slot -> combination -> shared add
    moe_prefill = moe_chain(description, reader, profile_id, "0", "prefill")
    moe_decode = moe_chain(description, reader, profile_id, str(max(0, len(description.nodes_of_type("attention")) // 2)), "decode")
    adjacent_all = list(dict.fromkeys(chain_prefill + chain_decode + chain_mid + moe_prefill + moe_decode))
    resolve_ids = make_resolver(reader)
    recorded: dict[str, Any] | None = json.loads(Path(args.transcript).read_text()) if args.transcript else None
    if recorded is not None:
        mode = recorded["source"]["mode"]
        policy = vchallenge.Policy.parse(recorded["policy"]["q"], recorded["policy"]["s"])
        source_opening = dict(recorded["source"])
    else:
        source = vchallenge.make_source(args.challenge, seed=args.seed)
        mode = source.mode
        source_opening = source.opening()
    statement = statement_of(reader, description, required_doc, commitments, bindings, policy, source_opening, first_stage.to_json())
    if recorded is None:
        session = vchallenge.StagedSession(source, policy)
        session.accept_statement(statement)
        selected = session.challenge_replay(request_ids)
        interiors, units = interiors_of(commitments, reader, selected)
        session.accept_interiors(interiors)
        adjacent = [c for c in adjacent_all if identity.parse(c).request_id in set(selected)]
        coverage = make_coverage(description, reader, adjacent, args.coverage_size, selected)
        certified, coverage_sample = session.challenge_sample(units, resolve_ids, coverage=coverage)
        transcript = session.finish()
    else:
        transcript = recorded
        selected = list(recorded["replay_challenge"]["selected"])
        interiors, units = interiors_of(commitments, reader, selected)
        adjacent = [c for c in adjacent_all if identity.parse(c).request_id in set(selected)]
        coverage = make_coverage(description, reader, adjacent, args.coverage_size, selected)
        certified = list(recorded["sample_challenge"]["certified_sample"])
        coverage_sample = list(recorded["sample_challenge"]["coverage_sample"])
    # the transcript must re-derive from the documents before it is used (a fresh run checks itself
    # the way a third party would; a recorded transcript is only accepted if it re-derives)
    rederived = vchallenge.reverify(transcript, statement=statement, interiors=interiors, request_ids=request_ids, units=units,
                                    resolve_ids=resolve_ids, coverage=coverage)
    if not rederived["ok"]:
        structural_problems.append("challenge transcript does not re-derive: " + "; ".join(rederived["problems"])[:200])
    sample = list(dict.fromkeys(coverage_sample + certified))
    try:
        firststage.assert_sample_committed(sample, reader, commitments)
    except firststage.FirstStageError as e:
        structural_problems.append(f"first-stage: challenge references an uncommitted tree: {e}")
    challenge = relations.make_challenge(commitments, description, sample, required_doc)
    digests = {p["phase"]: p["digest"] for p in transcript["phases"]}
    challenge_doc = {
        **challenge.to_json(),
        "challenge": {"mode": mode, "source": source_opening.get("source"), "policy": policy.to_json(),
                      "phase_digests": digests, "r_q": transcript["replay_challenge"]["r_q"], "r_s": transcript["sample_challenge"]["r_s"],
                      "transcript": "challenge_transcript.json", "rederived": rederived["ok"]},
        "selected_replay_units": selected,
        "certified_sample": certified,
        "coverage_sample": coverage_sample,
        "coverage_size": args.coverage_size,
        "adjacent_vus": adjacent,
        **({"seed": args.seed} if mode == "diagnostic" else {}),
    }
    (out / "challenge.json").write_text(json.dumps(challenge_doc, indent=1) + "\n")
    (out / "challenge_transcript.json").write_text(json.dumps(transcript, indent=1) + "\n")
    timing["challenge_s"] = time.perf_counter() - t0
    print(f"challenge mode={mode} q={policy.to_json()['q']} s={policy.to_json()['s']}: J={len(selected)}/{len(request_ids)} requests, "
          f"population {units.count} VUs, certified {len(certified)}, coverage {len(coverage_sample)} (adjacent {len(adjacent)}), union {len(sample)}; "
          f"T_0 {digests['statement'][:12]} T_1 {digests['replay'][:12]} T_2 {digests['interior'][:12]} T_3 {digests['sample'][:12]}; "
          f"rederived={rederived['ok']} ({timing['challenge_s']:.1f}s)", flush=True)

    # --- 4. authenticated openings + relation checks over both samples ---
    from verity_vllm_adapter.memo_cache import MemoCache  # RC: weight tensors authenticated once, reused across VUs

    cache = MemoCache(reader)
    t0 = time.perf_counter()
    report = relations.check(description, reader, commitments, sample, challenge=challenge, required_doc=required_doc, cache=cache)
    check_s = time.perf_counter() - t0
    coverage_sample_cov = relations.verify_sample_coverage(sample, report)
    fam = report.families()
    print(f"check: {len(sample)} VUs in {check_s:.1f}s; statuses {report.statuses()}", flush=True)
    for op_type, by_class in sorted(fam.items()):
        for klass, fr in by_class.items():
            print(f"  {op_type:12s} {klass:12s} exact={fr.exact} attempted={fr.attempted} mismatched={fr.mismatched} unimpl={fr.unimplemented} errors={fr.errors} auth_fail={fr.auth_failures} gpu_s={fr.seconds:.2f}", flush=True)
    for r in report.rejected()[:10]:
        print(f"  REJECTED {r.status} {r.canonical_id} {r.relation_id}: {r.detail} {r.first_divergence}", flush=True)
    adjacent_results = [r.to_json() for r in report.results if r.canonical_id in set(adjacent)]
    per_plan = {"certified": plan_stats(report, certified), "coverage": plan_stats(report, coverage_sample)}
    print(f"  certified (Bound-compatible, q={policy.to_json()['q']} s={policy.to_json()['s']}): {per_plan['certified']}", flush=True)
    print(f"  coverage  (stratified + adjacent, not Bound-compatible): {per_plan['coverage']}", flush=True)
    relation_errors = VB.relation_errors_of(report)  # W1cF F4: checker exceptions are rejections
    # W1cF F7/F10: the verifier's own environment vs the manifest's, and the PTX it compiles from the bundled
    # sources vs the production PTX shipped in the bundle (qualifications, printed in the verdict line)
    verifier_env, qualifications = VB.verifier_environment(manifest)
    ptx = None
    fused_launch_configs = None
    if kernel_sources is not None:
        import verity_vllm_adapter.relations_inductor as ind_rel

        ptx = VB.ptx_identity(run_dir, sources=ind_rel.SOURCES, triton_cache_dir=triton_cache_dir, manifest=manifest, env=verifier_env,
                              compile_fn=ind_rel.compile_source, preexisting=triton_cache_before)
        qualifications += ptx.get("qualifications", [])
        fused_launch_configs = dict(ind_rel.LAUNCH_CONFIG_USED)
        print(f"ptx identity: {ptx.get('summary')} (compiled {len(ind_rel.SOURCES)} bundled sources in {ptx.get('compile_seconds', 0):.1f}s)", flush=True)
    for q in qualifications:
        print(f"  QUALIFICATION {q}", flush=True)

    # --- sampling_plan.json (C2): two samples, two purposes, both bound to the transcript ---
    strat = sampling.describe_stratified(description, reader, coverage_sample, n=args.coverage_size,
                                         effective_seed=int(transcript["sample_challenge"]["coverage_seed"]), adjacent=len(adjacent))
    plan_doc = sampling.plan_document(stratified=strat, bernoulli=None, challenge_digests={
        "commitments_digest": challenge.commitments_digest, "description_digest": challenge.description_digest,
        "required_digest": challenge.required_digest, "challenge": challenge.digest().hex(), **{f"T_{i}": digests[k] for i, k in enumerate(("statement", "replay", "interior", "sample"))}})
    plan_doc["plans"][0]["purpose"] = "coverage: exercise every op family and (layer, phase) stratum at a fixed budget; a testing distribution, NOT the certified sample"
    plan_doc["plans"][0]["seed_derivation"] = transcript["sample_challenge"]["coverage_seed_derivation"]
    plan_doc["plans"][0]["results"] = per_plan["coverage"]
    plan_doc["plans"].append({
        "mode": "bernoulli", "purpose": "certified: the sample Bound's acceptance law prices; each replay unit (request) independently with q, each of its committed leaves independently with s",
        "replay_unit": "request (all of its steps; capture-format.md section 12)", "verification_unit": "one committed leaf: node instance x position x port",
        "q": policy.to_json()["q"], "s": policy.to_json()["s"], "sampler": transcript["core_sampler"],
        "randomness": {"r_q": transcript["replay_challenge"]["r_q"], "r_s": transcript["sample_challenge"]["r_s"], "source": source_opening.get("source"), "mode": mode},
        "population": {"requests": len(request_ids), "selected_requests": len(selected), "vus_in_selected_requests": units.count},
        "selected": {"requests": len(selected), "vus": len(certified)}, "expected_vus": float(policy.q * policy.s) * sum(int(rec["leaf_count"]) for rec in commitments["trees"].values() if rec.get("kind") == "ru"),
        "acceptance_law": "sigma(E) = prod_r (1 - q + q * (1 - s) ** |E cap P_r|)", "bound_compatible": True, "results": per_plan["certified"],
    })
    plan_doc["challenge"] = {"mode": mode, "note": "security: the verifier's secret seed, unpredictable to the prover before each phase closed; "
                                                  "diagnostic: a public seed, predictable, the verdict is qualified"}
    (out / "sampling_plan.json").write_text(json.dumps(plan_doc, indent=1) + "\n")

    # --- 5. coverage, linkage, capture predicates ---
    cov = vrequired.coverage(required_doc, description, reader) if required_doc else None
    linkage = []
    for rid in request_ids:
        try:
            linkage.append(relations.check_output_linkage(description, reader, commitments, rid, cache=cache).to_json())
        except Exception as e:  # noqa: BLE001
            linkage.append({"request_id": rid, "problems": [f"{type(e).__name__}: {e}"]})
    linkage_ok = all(not lr.get("problems") for lr in linkage)
    boundary_problems = {rid: reader.boundary(rid).validate() for rid in request_ids}
    boundary_problems = {k: v for k, v in boundary_problems.items() if v}
    predicates = capture_predicates(description, reader)

    # --- 6. negatives (W2 harness; adjacent VUs first so N02 challenges them) + binding/transcript negatives ---
    negatives: list[dict[str, Any]] = []
    neg_s = 0.0
    if not args.no_negatives:
        t0 = time.perf_counter()
        neg_sample = list(dict.fromkeys(adjacent + [c for c in coverage_sample if c not in set(adjacent)][:24]))
        harness = FastNegativeHarness(run_dir, out_dir=out / "negatives", sample=neg_sample, seed=int(transcript["sample_challenge"]["coverage_seed"]) % (2**31))
        harness.reader.weights.resolver = weights_resolver  # the harness' own reader (PatchedWeights inherit it); §16.8 or checkpoint resolver
        ids = [x for x in args.negatives.split(",") if x]
        # N10-N12 (W9) mutate slotted (MoE) leaves and run after the loop, on the check report; on a capture
        # without a slotted node they are inapplicable (recorded, neither counted nor NOT_RUN, like N19).
        moe_ids = {x for x in ids if x in MOE_NEGATIVE_IDS}
        for nid in ids:
            if nid in moe_ids:
                continue
            t1 = time.perf_counter()
            try:
                if nid in vbinding.BINDING_NEGATIVE_IDS:
                    if not any(n["negative_id"] in vbinding.BINDING_NEGATIVE_IDS for n in negatives):
                        for d in vbinding.binding_negatives(reader, commitments, cp=cp, production_path=production_path, out_dir=out / "negatives",
                                                            semantic_profile_path=args.semantic_profile, ids=[x for x in ids if x in vbinding.BINDING_NEGATIVE_IDS]):
                            if d["negative_id"] in ids:
                                d["seconds"] = time.perf_counter() - t1
                                print(f"  {d['negative_id']}: attempted={d['attempted']} rejected={d['rejected']} accepted={d['unexpectedly_accepted']}"
                                      + (f" inapplicable: {d['inapplicable']}" if d.get("inapplicable") else ""), flush=True)
                                negatives.append(d)
                    continue
                if nid == "N18":
                    d = transcript_negatives(transcript, statement=statement, interiors=interiors, request_ids=request_ids, units=units,
                                             resolve_ids=resolve_ids, coverage=coverage, out_dir=out / "negatives")
                else:
                    fn = {"N01": harness.n01, "N02": harness.n02, "N03": harness.n03, "N04": harness.n04, "N05": harness.n05,
                          "N06": harness.n06, "N07": harness.n07, "N13": harness.n13, "N14": harness.n14}[nid]
                    r = fn(args.negative_limit) if nid in ("N01", "N02", "N03", "N04", "N06") else fn()
                    d = r.to_json()
                    d["inconclusive"] = r.inconclusive
                    d["variants"] = r.variants
            except Exception as e:  # noqa: BLE001
                d = {"negative_id": nid, "attempted": 0, "rejected": 0, "unexpectedly_accepted": 0, "reproducer": None,
                     "error": f"{type(e).__name__}: {e}"}
            d["seconds"] = time.perf_counter() - t1
            print(f"  {nid}: attempted={d['attempted']} rejected={d['rejected']} accepted={d['unexpectedly_accepted']} ({d['seconds']:.1f}s)" + (f" error={d['error']}" if d.get("error") else ""), flush=True)
            negatives.append(d)
        if moe_ids and any(n.experts for n in description.nodes):
            t1 = time.perf_counter()
            try:
                for r in moe_negatives(harness, description, reader, report, sample, args.negative_limit, moe_ids):
                    d = r.to_json()
                    d["inconclusive"] = r.inconclusive
                    d["variants"] = r.variants
                    d["seconds"] = time.perf_counter() - t1
                    t1 = time.perf_counter()
                    print(f"  {r.negative_id}: attempted={d['attempted']} rejected={d['rejected']} accepted={d['unexpectedly_accepted']} ({d['seconds']:.1f}s)", flush=True)
                    negatives.append(d)
            except Exception as e:  # noqa: BLE001
                for nid in sorted(moe_ids):
                    negatives.append({"negative_id": nid, "attempted": 0, "rejected": 0, "unexpectedly_accepted": 0, "reproducer": None,
                                      "error": f"{type(e).__name__}: {e}", "seconds": 0.0})
                print(f"  MoE negatives failed: {type(e).__name__}: {e}", flush=True)
        elif moe_ids:
            for nid in sorted(moe_ids):
                negatives.append({"negative_id": nid, "attempted": 0, "rejected": 0, "unexpectedly_accepted": 0, "reproducer": None,
                                  "variants": [], "inconclusive": 0, "seconds": 0.0,
                                  "inapplicable": "no slotted (MoE) node in the description: no derived slot to mutate"})
        neg_s = time.perf_counter() - t0
    # A requested negative that ran zero variants is not evidence and must not hide behind the others'
    # rejections (audit-compiled-pass.md finding 5; the verdict rule counts attempted vs rejected only).
    # A harness *error* is a structural problem of the row (the verifier did not do what the row claims);
    # zero variants without an error (mutation target absent in this capture) is recorded as NOT_RUN (W1cF F5).
    # A negative the bundle has no target for (N19 on boundaries without output_logprob leaves) is *inapplicable*:
    # recorded on the row, neither counted nor NOT_RUN (WB2).
    inapplicable = {d["negative_id"]: d["inapplicable"] for d in negatives if d.get("inapplicable")}
    not_attempted = [d for d in negatives if int(d.get("attempted", 0)) == 0 and d["negative_id"] not in inapplicable]
    for d in not_attempted:
        if d.get("error"):
            structural_problems.append(f"negative {d['negative_id']}: harness error, 0 variants attempted ({d['error']})")
    if not_attempted:
        print(f"WARNING negatives with 0 attempts: {[d['negative_id'] for d in not_attempted]}", flush=True)
    requested_ids = [] if args.no_negatives else [x for x in args.negatives.split(",") if x and x not in inapplicable]
    not_run = VB.negatives_not_run(negatives, requested_ids)
    VB.mark_not_run(negatives, not_run)
    for d in negatives:
        if d["negative_id"] in inapplicable:
            d["status"] = "INAPPLICABLE"
    if not_run:
        print(f"  negatives NOT_RUN (0 attempted): {not_run}", flush=True)
    if inapplicable:
        print(f"  negatives INAPPLICABLE to this bundle: {inapplicable}", flush=True)

    # --- 7. rows ---
    utc = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    exp_id = rows.experiment_id(case, profile_id, utc)
    common = rows.Common(exp_id, case, profile_id, args.worker, args.adapter_revision)
    prod_timings = json.loads((prod_dir / "timings.json").read_text()) if (prod_dir / "timings.json").exists() else {}
    prod_steps = json.loads((prod_dir / "steptrace.json").read_text()).get("summary", {}) if (prod_dir / "steptrace.json").exists() else {}
    replay_doc = json.loads((run_dir / "replay.json").read_text()) if (run_dir / "replay.json").exists() else {}
    noninter = json.loads(Path(args.noninterference).read_text()) if args.noninterference else None

    boundaries = [reader.boundary(rid) for rid in request_ids]
    workload = {
        "requests": len(boundaries),
        "generated_positions": sum(len(b.output_token_ids) for b in boundaries),
        "input_len_range": [min(b.prompt_len for b in boundaries), max(b.prompt_len for b in boundaries)],
        "peak_concurrency_achieved": int(prod_steps.get("peak_concurrency_achieved", 0) or 0),
    }
    # §15.8 witness of activation for the recorded production pass
    workload_exercised: dict[str, Any] | None = None
    wx_path = prod_dir / "workload_exercised.json"
    if wx_path.exists():
        workload_exercised = json.loads(wx_path.read_text())
    else:
        prod_rows = [json.loads(l) for l in production_path.read_text().splitlines() if l.strip()]
        cached = {r["request_id"]: r.get("num_cached_tokens") for r in prod_rows}
        workload_exercised = {
            "recorded_pass": "unknown (recorder before workload_exercised.json)",
            "enable_prefix_caching": (manifest.get("cache") or {}).get("enable_prefix_caching", "stock default (not recorded)"),
            "num_cached_tokens_per_request": cached,
            "requests_with_cached_tokens": sum(1 for v in cached.values() if v),
            "prefix_cache_hits_at_admission": int(prod_steps.get("prefix_cache_hits_at_admission", 0) or 0),
            "peak_concurrency_achieved": int(prod_steps.get("peak_concurrency_achieved", 0) or 0),
            "engine_steps": int(prod_steps.get("engine_steps", 0) or 0),
            "preemptions": int(prod_steps.get("preemptions", 0) or 0),
            "prefill_chunks_per_request": prod_steps.get("prefill_chunks_per_request"),
            "max_tokens_in_a_step": int(prod_steps.get("max_tokens_in_a_step", 0) or 0),
            "effective_execution": manifest.get("compilation"),
        }
    cp_pass = (prod_timings.get("cached_prefix_pass") or {})
    if cp_pass:
        workload_exercised["cached_prefix_pass_tokens_identical_to_recorded"] = f"{cp_pass.get('tokens_identical_to_recorded')}/{cp_pass.get('requests')}"
    svs = prod_timings.get("stock_vs_stock_token_identical_runs")
    if svs:
        workload_exercised["stock_vs_stock_token_identical_runs"] = f"{svs.get('identical')}/{svs.get('compared')}"
    sampler_fam = fam.get("sampler", {})
    token_sel = {"exact": sum(f.exact for f in sampler_fam.values()), "attempted": sum(f.attempted for f in sampler_fam.values())}
    rel_checks = report.relation_checks()
    for k in ("same_runtime", "independent"):
        rel_checks.setdefault(k, {"exact": 0, "attempted": 0})
    if noninter:
        bni = {"identical": bool(noninter["identical"]), "compared": int(noninter["compared_scalars"]),
               "mismatched_scalars": int(noninter["mismatched_scalars"]), "reference_method": str(noninter["reference_method"])}
    else:
        bni = {"identical": False, "compared": 0, "mismatched_scalars": 0, "reference_method": "not run"}
    if boundary_problems:
        structural_problems.append(f"boundary: {list(boundary_problems.items())[0]}")
    if not linkage_ok:
        structural_problems.append("output linkage: " + str([lr for lr in linkage if lr.get("problems")][0]["problems"][0]))
    if not coverage_sample_cov.ok:
        structural_problems.append("sample coverage")
    if not predicates["steps_expected_equal"]:
        structural_problems.append("committed replay units != boundary steps")
    if predicates["changed_contents_between_requests"] is False:
        structural_problems.append("identical contents at the same node/position in two requests with different tokens")
    if predicates["equal_contents_on_shared_prefix"] is False:
        structural_problems.append("different contents at the same node/position in two requests sharing that prompt prefix")
    if predicates["present_leaves"] != predicates["expected_leaves"]:
        structural_problems.append(f"capture has {predicates['present_leaves']}/{predicates['expected_leaves']} expected leaves")
    for bname, b in w1cf_bindings.items():  # W1cF F2/F3 (RC: `bindings` is WB's Bindings object since the rename above; this loop wants W1cF's dict)
        if b is not None and b.get("problem"):
            structural_problems.append(f"{bname}: {b['problem']}")
    if ptx is not None and ptx.get("problem"):  # W1cF F10: shipped PTX not hashing to its names / not the manifest's set
        structural_problems.append(f"ptx: {ptx['problem']}")
    structural = {"result": "ok" if not structural_problems else "failed", "failing_invariant": structural_problems[0] if structural_problems else None}
    t3 = rows.table3_rows(common, [{k: d[k] for k in ("negative_id", "attempted", "rejected", "unexpectedly_accepted", "reproducer")} for d in negatives])
    negative = rows.negative_summary(t3)
    restrictions = [f"{k}={v}" for k, v in (replay_doc.get("replay_restrictions") or {}).items() if not isinstance(v, dict)]
    restrictions += [f"trace_replay.{k}: {v}" for k, v in (replay_doc.get("replay_restrictions", {}).get("trace_replay_api") or {}).items()]
    # state rows (W10): engine overrides the replay ran with -- production's declared ones re-applied
    # (chunk size, KV budget, prefix cache) and replay-only ones (a chunk restriction on a small pod)
    replay_only = replay_doc.get("replay_restrictions", {}).get("engine_args") or {}
    for k, v in ((replay_doc.get("state_rows") or {}).get("engine_overrides_applied") or {}).items():
        restrictions.append(f"engine.{k}={v} ({'replay-side restriction' if k in replay_only else 'declared in production, re-applied'})")
    runtime_impl = {n.op: (n.runtime_impl or "unresolved") for n in description.nodes}
    limitation = {n.op: VB.fused_limitation(n, fused_launch_configs) for n in description.nodes}  # W1cF F8/F9
    if "batch.num_tokens" in description:  # W9 revision-1 MoE captures
        limitation["batch.num_tokens"] = "batch composition witness (scheduler decision reproduced by trace replay), not a computed value; consumed by the expert relation for the fused-MoE kernel config"
    # W9: slotted (expert) nodes get their own Table 2 rows, active and inactive slots kept apart
    moe_t2, moe_consumed, moe_padding = moe_table2_rows(common, report, description, reader, runtime_impl)
    dense_report = relations.CheckReport(report.challenge_digest, [r for r in report.results if r not in moe_consumed],
                                         report.openings_verified, report.openings_attempted)
    t2 = rows.table2_rows_from_report(common, dense_report, description, runtime_impl=runtime_impl, limitation=limitation) + moe_t2
    challenge_field = {
        "mode": mode, "source": source_opening.get("source"), "policy": policy.to_json(),
        "phase_digests": digests, "selected_replay_units": len(selected), "certified_vus": len(certified), "coverage_vus": len(coverage_sample),
        "certified": per_plan["certified"]["relation_checks"], "coverage": per_plan["coverage"]["relation_checks"],
        # ``logprobs``: whether the committed boundary trees carry the chosen logprobs (WB2; ``unbound`` = the bundle
        # predates the output_logprob leaves and was verified as before; a row without the key predates WB2 = unbound)
        "bindings": {"model": bindings.model.ok, "architecture": bindings.architecture.ok, "output": bindings.output.ok,
                     "logprobs": bindings.output.data.get("logprobs", "unbound")},
        "transcript_rederived": rederived["ok"],
        **({"negatives_inapplicable": inapplicable} if inapplicable else {}),
    }
    t1 = rows.table1_row(
        common,
        workload=workload,
        production_changes=production_changes_of(manifest),
        replay_restrictions=restrictions,
        token_selection=token_sel,
        boundary_noninterference=bni,
        required_coverage={"bound": cov.bound, "required": cov.required} if cov else {"bound": 0, "required": 0},
        authentication={"valid": report.openings_verified, "attempted": report.openings_attempted},
        relation_checks=rel_checks,
        structural=structural,
        negative=negative,
        table2_rows=t2,
        workload_exercised=workload_exercised,
        challenge=challenge_field,
        relation_errors=relation_errors,  # W1cF F4
        negatives_not_run=not_run,  # W1cF F5
        qualifications=qualifications,  # W1cF F7/F10
        bindings=w1cf_bindings,  # W1cF F2/F3 (kernel set, description digest); WB's C3 bindings are in `challenge.bindings` and verify.json `bindings`
        # WB2 (GUIDANCE §15.2): the verifier's own run is at most "transcript-verified (security-mode staging)";
        # "independently verified" is an auditor's label, attached via results/<experiment>/audit.json at render time.
        verification_evidence=rows.verification_evidence({"challenge": challenge_field, "worker": common.worker}),
    )
    cap_timing = (replay_doc.get("timing") or {})
    prod_over = prod_timings.get("production_overhead_ratio")
    replay_run_s = cap_timing.get("run_s")
    ref_run_s = (noninter or {}).get("reference_run_s")
    t4 = rows.table4_row(
        common,
        partition_id=(required_doc or {}).get("partition_id", "row-default"),
        instrumentation_mode=str(replay_doc.get("instrumentation_mode") or (
            "compiled-launch (CachingAutotuner.run wrapped inside Inductor pieces + TorchDispatchMode outside; compiled, CUDA graphs off)"
            if replay_doc.get("replay_exec") == "compiled-nograph" else "dispatch-mode (TorchDispatchMode + custom-op wrappers; eager-only)")),
        production_overhead_ratio=float(prod_over) if prod_over is not None else None,
        replay_commit_overhead_ratio=(float(replay_run_s) / float(ref_run_s)) if replay_run_s and ref_run_s else None,
        replay_vs_production_gpu_work=None,
        transparent_check_cost={"cpu_s": float(check_s), "gpu_s": float(sum(r.checker_seconds for r in report.results))},
        peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
        committed_logical_bytes=int((replay_doc.get("capture") or {}).get("bytes_captured") or 0) or None,
        processed_bytes={"hashed": int((replay_doc.get("capture") or {}).get("bytes_captured") or 0) or None,
                         "copied": int((replay_doc.get("capture") or {}).get("bytes_captured") or 0) or None,
                         "retained": int((replay_doc.get("capture") or {}).get("bytes_captured") or 0) or None, "emitted": None},
        description_size_bytes=(run_dir / "description.json").stat().st_size,
        constructor_inputs_bytes=None,
        logical_size={"gates": len(description.nodes), "vus": int(predicates["expected_leaves"]), "rus": len(reader.ru_keys())},
        lookup_cost_s=float(timing["challenge_s"]),
        validation_cost_s=float(check_s + neg_s + timing["bindings_s"]),
        moe_padding=moe_padding,
        restrictions=restrictions,
        spend={"gpu_hours": None, "usd": None},
    )
    for n, rws in ((1, [t1]), (2, t2), (3, t3), (4, [t4])):
        rows.write_jsonl(out / f"table{n}.jsonl", rws)

    doc = {
        "run_dir": str(run_dir),
        "experiment_id": exp_id,
        "case": case,
        "profile_id": profile_id,
        "challenge": {**challenge_field, "r_q": transcript["replay_challenge"]["r_q"], "r_s": transcript["sample_challenge"]["r_s"],
                      "transcript": str(out / "challenge_transcript.json"), "reverified_from": args.transcript},
        "verdict_label": vchallenge.verdict_label(t1["result"], mode),
        **({"seed": args.seed} if mode == "diagnostic" else {}),
        "engine_flags_applied": engine_flags,
        "relations_installed": installed,
        "kernel_sources": kernel_sources,
        "weights_source": weights_source,
        "bindings": bindings.to_json(),
        "production_record_sha256": bindings.output.data.get("production_record_sha256"),
        "first_stage": first_stage.to_json(),
        "sample_size": len(sample),
        "sampling_plans": per_plan,
        "adjacent_vus": adjacent,
        "adjacent_results": adjacent_results,
        "challenge_digest": challenge.digest().hex(),
        "report": report.to_json(),
        "families": {ot: {k: v.to_json() for k, v in bc.items()} for ot, bc in fam.items()},
        "sample_coverage": coverage_sample_cov.to_json(),
        "required_coverage": {"bound": cov.bound, "required": cov.required, "missing": cov.missing[:10], "unexpected_ids": cov.unexpected_ids[:10]} if cov else None,
        "output_linkage": linkage,
        "boundary_problems": boundary_problems,
        "capture_predicates": predicates,
        "moe": {
            "witness_of_activation": (json.loads((run_dir / "capture_summary.json").read_text()).get("moe") if (run_dir / "capture_summary.json").exists() else None),
            "table4_padding": moe_padding,
            "slot_results": {k: sum(1 for r in moe_consumed if slot_kind(reader, r.canonical_id) == k and r.status == "exact") for k in ("active", "inactive")},
            "slot_attempted": {k: sum(1 for r in moe_consumed if slot_kind(reader, r.canonical_id) == k and r.status in ("exact", "mismatch")) for k in ("active", "inactive")},
        } if moe_consumed else None,
        "negatives": negatives,
        "negatives_with_zero_attempts": [d["negative_id"] for d in not_attempted],
        "negatives_not_run": not_run,
        "negatives_inapplicable": inapplicable,
        "relation_errors": relation_errors,
        "structural_bindings": w1cf_bindings,
        "verifier_environment": verifier_env,
        "verifier_caches": {"triton": triton_cache_dir, "inductor": inductor_cache_dir, "triton_ptx_preexisting": len(triton_cache_before)},
        "qualifications": qualifications,
        "ptx_identity": ptx,
        "fused_launch_configs": fused_launch_configs,
        "table1": t1,
        "timing": {"bindings_s": timing["bindings_s"], "binding_detail_s": {r.name: r.seconds for r in (bindings.model, bindings.architecture, bindings.output)},
                   "challenge_s": timing["challenge_s"], "check_s": check_s, "negatives_s": neg_s, "total_s": time.perf_counter() - t_start},
        "utc": utc,
    }
    (out / "verify.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"RESULT {doc['verdict_label']} first_blocker={t1['first_blocker']} auth={t1['authentication']} relation_checks={t1['relation_checks']} "
          f"coverage={t1['required_coverage']} negative={t1['negative']} structural={t1['structural']} challenge_mode={mode} "
          f"relation_errors={relation_errors['count']} negatives_not_run={not_run} qualifications={qualifications}", flush=True)
    return 0


def _row_stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size)


def run(argv: list[str] | None = None) -> int:
    """W1cF F9 + WB2 (audit finding 4): whatever stops the pipeline yields a Table 1 row and a RESULT line,
    never a traceback alone.

    * a refusal (``SystemExit`` with a message: kernels/ files not hashing to their names, a conflicting
      environment, weights referenced without ``--weights-from-model``) -> ``FAIL structural: bundle refused: …``
      with ``structural.result = refused``;
    * a structural check that did not pass (``VB.StructuralFailure``: description validation, commitments, capture
      format; WA's M8b) -> ``FAIL structural: <check>: <detail>``, ``structural.result = failed``;
    * any other exception (a ``CommitError`` escaping an opening, a missing document) -> ``FAIL structural: <check
      named from the exception type>: …`` (``VB.structural_reason``), ``structural.result = failed``.

    Every such row claims no relation counts (all zero).  The traceback goes to stderr and into
    ``verify.json.exception``.  Exit code 1 for all of them (the existing convention: 0 = the pipeline completed
    and the row's verdict, PASS / PARTIAL / FAIL, stands; 1 = the pipeline did not complete and the row says why).
    A verdict row ``main`` wrote *in this run* before failing is left alone; a stale ``table1.jsonl`` from an
    earlier run in the same ``--out`` is replaced (previously the traceback was the only output then)."""
    import traceback

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--case", default=None)
    ap.add_argument("--adapter-revision", default="unknown")
    ap.add_argument("--worker", default=os.environ.get("VERITY_WORKER", WORKER))
    a, _ = ap.parse_known_args(argv)
    run_dir = Path(a.run_dir)
    out = Path(a.out) if a.out else run_dir / "verify"
    before = _row_stamp(out / "table1.jsonl")

    try:
        return main(argv)
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit) and isinstance(e.code, str) and e.code:
            reason, result, tb = f"bundle refused: {e.code}", "refused", traceback.format_exc()
        else:
            raise
    except VB.StructuralFailure as e:
        reason, result, tb = f"{e.check}: {e.detail}", "failed", traceback.format_exc()
    except Exception as e:  # noqa: BLE001
        reason, result, tb = VB.structural_reason(e), "failed", traceback.format_exc()

    print(tb, file=sys.stderr, flush=True)
    if (out / "table1.jsonl").exists() and _row_stamp(out / "table1.jsonl") != before:
        print(f"REFUSED after a verdict row was written in this run (row kept): {reason}", file=sys.stderr, flush=True)
        return 1
    profile_id, case = "unknown", a.case or "B?"
    try:
        m = json.loads((run_dir / "manifest.json").read_text())
        import verity_vllm_adapter.profile as prof

        profile_id = prof.profile_id_of(m)
        case = a.case or m.get("case", case)
    except Exception:  # noqa: BLE001
        pass
    print(f"REFUSED: {reason}", file=sys.stderr, flush=True)
    VB.refusal_row(out, case=case, profile_id=profile_id, worker=a.worker, adapter_revision=a.adapter_revision,
                   reason=reason[:600], exception=tb, result=result)
    return 1


if __name__ == "__main__":
    sys.exit(run())
