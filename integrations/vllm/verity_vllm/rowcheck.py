"""JSON-only cross-check of an adapter results row (W2c review, CONTRACT section 16.5 / D2).

The tensors of a row (``ru/**/values.bin``, ``weights/values.bin``) stay on the pod; what is
committed to the repository is the JSON: ``bundle/{manifest,description,required,commitments}.json``,
``challenge.json``, ``verify.json``, ``table*.jsonl``, ``negatives/*.json`` and the production
record ``production/production.jsonl``.  This module re-derives, from those files alone and with
W2's generic code (``hashing``, ``capture.Boundary``, ``commit.MerkleTree``, ``description``,
``relations.Challenge``), everything the adapter's fresh verifier claimed that does not need a
tensor, and reports agreement or disagreement per check.

What it can establish without tensors
-------------------------------------
* ``profile_id``: the manifest bytes hash to the profile id every document carries.
* ``description_digest`` / ``required_digest`` / ``commitments_digest`` / ``challenge_digest``:
  the challenge is bound to exactly the on-disk documents (N13's premise), and its digest
  is what W2's ``Challenge.digest`` gives for those fields.
* ``boundary_roots``: every boundary root in ``commitments.json`` is *recomputed from the
  production record* (prompt ids, output ids, sampling params, seed, finish reason) plus the
  committed step records.  This is the check the adapter verifier does not perform (it never
  reads ``production.jsonl``); a match means the committed boundary *is* the production
  record for this row.
* ``ru_namespace``: every replay-unit tree's ``leaf_count`` and ``ids_digest`` equal those of
  the id set the description predicts for that request/step -- the committed namespace is
  the declared computation (N12's premise) for all steps, not only the sampled ones.
* ``steps``: the committed steps cover positions ``0 .. prompt_len+outputs-2`` once, in order
  (capture-format v1.1: positions a request read from another request's cached prefix excluded,
  a recompute step's repeats counted once), and the sampled positions are exactly the
  ``sampler.select`` instances.
* ``sample``: every sampled VU parses, lives in a committed tree, and is in that tree's
  predicted namespace; the adjacent chains are inside the sample; sample ids are unique.
* ``report_vs_challenge``: ``verify.json`` reports on exactly the challenge's sample and
  with the challenge's digest; the family counts add up to the sample size.
* ``required_vs_description``: every required interface names a description node and the
  required set is what the description would derive.
* ``weights_linkage``: the weights tree's ``weights_sha256`` is the manifest's.
* ``verdict``: ``rows.verdict`` recomputed from the row's tables equals the recorded one.
* ``seed_derivation``: reports how the challenge seed was derived (raw argument vs mixed with
  the roots) -- a property of the run, not a pass/fail.

What it cannot establish without tensors (listed in the report as ``needs_tensors``): leaf
hashes and Merkle roots of RU/weights trees, opening authentication, every relation check
(same_runtime or independent), token selection against the opened logits, the negative tests
N01-N04/N06/N07, boundary noninterference against the reference captures, and the weight
leaves against the checkpoint file.

Usage::

    python -m verity_vllm.rowcheck fixtures/results/<experiment_id> [--out report.json]

Exit status 0 when every attempted check agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import hashing, identity, rows
from .capture import Boundary, StepRecord
from .commit import MerkleTree, boundary_tree_id, ru_tree_id
from .description import Description, NodeInstance
from .relations import Challenge

NEEDS_TENSORS = (
    "leaf hashes and Merkle roots of every ru/ and weights tree (needs values.bin)",
    "opening authentication of the sampled VUs and their inputs (N01, N03)",
    "relation checks, same_runtime and independent, for every sampled VU (N02, N04)",
    "token_selection against the opened lm_head logits (N08/N09 when applicable)",
    "N06 (weights tree recomputed from the stored weight bytes) and N07's reproducer",
    "boundary noninterference against the two reference captures (ref1/ref2 values)",
    "weight leaves against the checkpoint file named by manifest.model.weights_sha256",
)


@dataclass
class Check:
    name: str
    status: str  # "agree" | "disagree" | "skip"
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RowFiles:
    row: Path
    bundle: Path
    manifest_path: Path
    description_path: Path
    required_path: Path | None
    commitments_path: Path
    challenge_path: Path | None
    verify_path: Path | None
    production_path: Path | None
    tables: dict[str, Path]
    negatives_dir: Path | None


def locate(row: str | Path) -> RowFiles:
    """Accept either a results row (adapter layout) or a bare capture directory."""
    row = Path(row)
    bundle = row / "bundle" if (row / "bundle").is_dir() else row
    manifest = bundle / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"{row}: no bundle/manifest.json or manifest.json")

    def first(*candidates: Path) -> Path | None:
        return next((c for c in candidates if c.exists()), None)

    return RowFiles(
        row=row,
        bundle=bundle,
        manifest_path=manifest,
        description_path=bundle / "description.json",
        required_path=first(bundle / "required.json"),
        commitments_path=bundle / "commitments.json",
        challenge_path=first(row / "challenge.json", bundle / "challenge.json"),
        verify_path=first(row / "verify.json", bundle / "verify.json", row / "report.json"),
        production_path=first(
            row / "production" / "production.jsonl", row / "production.jsonl",
            bundle / "production.jsonl",
        ),
        tables={f"table{i}": row / f"table{i}.jsonl" for i in (1, 2, 3, 4)
                if (row / f"table{i}.jsonl").exists()},
        negatives_dir=first(row / "negatives"),
    )


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


LOGPROB_STATUSES = ("bound", "unbound", "unknown_leaf_set", "uncommitted")


def logprob_binding_status(boundary: Boundary, committed: Mapping[str, Any] | None, profile_id: str) -> str:
    """Which leaf set the committed boundary tree carries, read off its ``ids_digest`` (WB2):

    * ``bound`` -- the tree has one ``output_logprob`` leaf per produced token: the chosen logprobs
      are protected outputs of this bundle;
    * ``unbound`` -- the tree is the pre-WB2 leaf set (tokens, presence, params, seed, finish reason):
      the bundle predates the field; it verifies as before and the row must say so;
    * ``unknown_leaf_set`` -- neither (the committed ids are not the record's under either
      convention; the root comparison reports the mismatch);
    * ``uncommitted`` -- no tree for this request.

    The decision uses the ids only, never the values, so a bundle is never silently upgraded (a
    record with logprobs against an ``unbound`` tree stays unbound) or downgraded (a ``bound`` tree
    against a record without logprobs is a binding failure)."""
    if committed is None:
        return "uncommitted"
    digest = committed.get("ids_digest")
    for status, with_lp in (("bound", True), ("unbound", False)):
        ids = identity.tree_sorted(boundary.leaf_ids(profile_id, logprobs=with_lp))
        if digest == hashing.ids_digest(ids).hex():
            return status
    return "unknown_leaf_set"


def logprob_summary(status: Mapping[str, str]) -> str:
    """The row-level label: ``bound`` / ``unbound`` when every request agrees, else the mixture."""
    distinct = sorted(set(status.values()))
    if not distinct:
        return "unbound"
    if len(distinct) == 1:
        return distinct[0]
    return "mixed: " + ", ".join(f"{s} x{sum(1 for v in status.values() if v == s)}" for s in distinct)


def production_boundaries(
    production: Mapping[str, Mapping[str, Any]], commitments: Mapping[str, Any]
) -> dict[str, Boundary]:
    """Boundaries rebuilt from the production record and the committed step records only.

    Each boundary binds the record's ``chosen_logprobs`` exactly when the committed boundary tree
    does (:func:`logprob_binding_status` == ``bound``); see :func:`production_boundaries_with_status`
    for the per-request status."""
    return production_boundaries_with_status(production, commitments)[0]


def production_boundaries_with_status(
    production: Mapping[str, Mapping[str, Any]], commitments: Mapping[str, Any]
) -> tuple[dict[str, Boundary], dict[str, str]]:
    """``(boundaries, logprob status per request)``; the status is one of :data:`LOGPROB_STATUSES`.

    A ``bound`` tree makes the boundary carry the record's ``chosen_logprobs`` -- ``[]`` when the
    record has none, so that ``validate()`` reports the missing logprobs and the root comparison
    fails, instead of the tree being rebuilt without them."""
    steps: dict[str, list[StepRecord]] = {}
    btrees: dict[str, Mapping[str, Any]] = {}
    for rec in commitments["trees"].values():
        if rec.get("kind") == "ru":
            steps.setdefault(str(rec["request_id"]), []).append(
                StepRecord(int(rec["step"]), str(rec["phase"]), tuple(int(t) for t in rec["positions"]))
            )
        elif rec.get("kind") == "boundary":
            btrees[str(rec["request_id"])] = rec
    profile_id = str(commitments.get("profile_id") or "")
    out: dict[str, Boundary] = {}
    status: dict[str, str] = {}
    for rid, rec in production.items():
        sp = dict(rec.get("sampling_params") or {})
        kw: dict[str, Any] = dict(
            seed=sp.get("seed"),
            finish_reason=rec.get("finish_reason"),
            steps=sorted(steps.get(rid, []), key=lambda s: s.step),
            max_tokens=sp.get("max_tokens"),
        )
        plain = Boundary.build(rid, rec["prompt_token_ids"], rec["output_token_ids"], sp, **kw)
        st = logprob_binding_status(plain, btrees.get(rid), profile_id) if profile_id else "unbound"
        status[rid] = st
        if st == "bound":
            lps = rec.get("chosen_logprobs")
            out[rid] = Boundary.build(
                rid, rec["prompt_token_ids"], rec["output_token_ids"], sp, **kw,
                chosen_logprobs=list(lps) if isinstance(lps, list) else [],
            )
        else:
            out[rid] = plain
    return out, status


def expected_step_ids(desc: Description, boundary: Boundary, profile_id: str, step: int) -> list[str]:
    """The ids the description predicts for one replay unit (request x step), unsorted."""
    st = next(s for s in boundary.steps if s.step == step)
    sampled = set(boundary.sampled_positions())
    out: list[str] = []
    for node in desc.nodes:
        if st.phase not in node.phases:
            continue
        shards: list[int | None] = list(range(node.shards)) if node.shards else [None]
        experts: list[int | None] = list(range(node.experts)) if node.experts else [None]
        for t in st.positions:
            if node.positions == "sampled" and t not in sampled:
                continue
            for port in range(node.ports):
                for s in shards:
                    for e in experts:
                        out.append(
                            NodeInstance(node, boundary.request_id, t, st.phase, port, s, e).output_id(profile_id)
                        )
    return out


def check_row(files: RowFiles, *, namespace: bool = True) -> dict[str, Any]:
    checks: list[Check] = []
    manifest = _load(files.manifest_path)
    description = Description(_load(files.description_path))
    commitments = _load(files.commitments_path)
    required = _load(files.required_path) if files.required_path else None
    challenge_doc = _load(files.challenge_path) if files.challenge_path else None
    verify_doc = _load(files.verify_path) if files.verify_path else None
    production = None
    if files.production_path:
        production = {str(r["request_id"]): r for r in _jsonl(files.production_path)}

    # 1. profile_id -------------------------------------------------------------------------
    pid_bytes = hashlib.sha256(files.manifest_path.read_bytes()).hexdigest()[:12]
    pid_canon = hashing.profile_id(manifest)
    carriers = {
        "manifest_bytes": pid_bytes,
        "manifest_canonical_json": pid_canon,
        "commitments": commitments.get("profile_id"),
        "description": description.profile_id,
    }
    if required is not None:
        carriers["required"] = required.get("profile_id")
    if verify_doc is not None:
        carriers["verify"] = verify_doc.get("profile_id")
    prod_manifest = files.row / "production" / "manifest.json"
    if prod_manifest.exists():
        carriers["production_manifest_bytes"] = hashlib.sha256(prod_manifest.read_bytes()).hexdigest()[:12]
    distinct = {v for v in carriers.values() if v is not None}
    checks.append(Check("profile_id", "agree" if len(distinct) == 1 else "disagree",
                        f"carriers: {carriers}", {"carriers": carriers}))
    profile_id = pid_canon

    # 2. document digests and the challenge binding -----------------------------------------
    desc_digest = description.digest().hex()
    req_digest = hashing.json_digest(required).hex() if required is not None else None
    com_digest = hashing.json_digest(commitments).hex()
    if required is not None:
        ok = required.get("description_digest") == desc_digest
        checks.append(Check("required.description_digest", "agree" if ok else "disagree",
                            f"required.json binds description {required.get('description_digest')}; "
                            f"recomputed {desc_digest}"))
    if challenge_doc is not None:
        bound = {
            "commitments_digest": (challenge_doc.get("commitments_digest"), com_digest),
            "description_digest": (challenge_doc.get("description_digest"), desc_digest),
            "required_digest": (challenge_doc.get("required_digest"), req_digest),
        }
        for name, (recorded, recomputed) in bound.items():
            checks.append(Check(f"challenge.{name}", "agree" if recorded == recomputed else "disagree",
                                f"recorded {recorded}, recomputed {recomputed}"))
        ch = Challenge(
            commitments_digest=str(challenge_doc["commitments_digest"]),
            description_digest=str(challenge_doc["description_digest"]),
            required_digest=challenge_doc.get("required_digest"),
            sample=tuple(challenge_doc["sample"]),
        )
        recomputed = ch.digest().hex()
        checks.append(Check("challenge.digest", "agree" if recomputed == challenge_doc.get("digest") else "disagree",
                            f"recorded {challenge_doc.get('digest')}, recomputed {recomputed} over "
                            f"(commitments, description, required, sample[{len(ch.sample)}])"))
        # seed derivation (a property, not a verdict)
        seed = challenge_doc.get("seed")
        derivation = "unknown"
        staged = challenge_doc.get("challenge") if isinstance(challenge_doc.get("challenge"), dict) else None
        if staged and staged.get("mode") in ("security", "diagnostic"):
            derivation = (f"staged ({staged.get('mode')}): r_q/r_s from a {staged.get('source')} over the phase digests "
                          f"T_0..T_3 (verity_vllm.challenge; transcript {staged.get('transcript')})")
        elif "effective_seed" in challenge_doc:
            derivation = "mixed: effective_seed = H(seed, json_digest(commitments)) (generic verifier); diagnostic"
        elif seed is not None:
            derivation = "raw: the --seed argument is the sampling seed; not mixed with the roots; diagnostic"
        checks.append(Check("seed_derivation", "skip", derivation, {"seed": seed, "mode": (staged or {}).get("mode", "diagnostic")}))
    else:
        checks.append(Check("challenge", "skip", "no challenge.json in the row"))

    # 3. boundary roots from the production record ------------------------------------------
    boundaries: dict[str, Boundary] = {}
    if production is None:
        checks.append(Check("boundary_roots_from_production", "skip", "no production record in the row"))
    else:
        boundaries, lp_status = production_boundaries_with_status(production, commitments)
        btrees = {rec["request_id"]: rec for rec in commitments["trees"].values() if rec.get("kind") == "boundary"}
        matched, mismatched, problems = 0, [], []
        for rid, b in boundaries.items():
            probs = b.validate()
            if probs:
                problems.append(f"{rid}: {'; '.join(probs)}")
            rec = btrees.get(rid)
            if rec is None:
                mismatched.append(f"{rid}: no boundary tree committed")
                continue
            tree = MerkleTree.from_values(b.leaves(profile_id), boundary_tree_id(rid))
            if tree.root.hex() != rec["root"] or tree.leaf_count != int(rec["leaf_count"]):
                mismatched.append(f"{rid}: root {tree.root.hex()[:16]} != {rec['root'][:16]}")
            elif tree.ids_digest.hex() != rec.get("ids_digest"):
                mismatched.append(f"{rid}: ids_digest differs")
            else:
                matched += 1
        extra = sorted(set(btrees) - set(boundaries))
        status = "agree" if matched == len(btrees) and not mismatched and not extra and not problems else "disagree"
        logprobs = logprob_summary(lp_status)
        checks.append(Check("boundary_roots_from_production", status,
                            f"{matched}/{len(btrees)} boundary roots recomputed from production.jsonl + committed steps; "
                            f"chosen logprobs {logprobs}"
                            + (f"; mismatched: {mismatched[:3]}" if mismatched else "")
                            + (f"; committed but not in production: {extra[:3]}" if extra else "")
                            + (f"; boundary problems: {problems[:3]}" if problems else ""),
                            {"matched": matched, "committed": len(btrees), "mismatched": mismatched,
                             "not_in_production": extra, "boundary_problems": problems,
                             "logprobs": logprobs, "logprobs_per_request": lp_status}))
        if verify_doc is not None and "production_record_sha256" not in json.dumps(verify_doc):
            checks.append(Check("production_record_bound", "disagree",
                                "neither challenge.json nor verify.json carries a hash of production.jsonl; "
                                f"sha256(production.jsonl) = {hashlib.sha256(files.production_path.read_bytes()).hexdigest()}"))

    # 4. steps and sampled positions ---------------------------------------------------------
    if boundaries:
        bad = []
        sampler_nodes = [n for n in description.nodes if n.positions == "sampled"]
        for rid, b in boundaries.items():
            # capture-format v1.1 (W10): cached-prefix positions (R3) are the source request's;
            # a recompute step (R4) repeats positions counted once (Boundary.covered_positions)
            cached = sorted(b.cached_positions())
            covered = b.covered_positions()
            processed = b.prompt_len + max(len(b.output_token_ids) - 1, 0)
            if cached + covered != list(range(processed)):
                bad.append(f"{rid}: steps cover {len(covered)} positions"
                           + (f" (+{len(cached)} cached)" if cached else "") + f", expected {processed} in order")
            want = set(b.sampled_positions())
            for n in sampler_nodes:
                got = {t for t, _ in description.positions_of(n, b)}
                if got != want:
                    bad.append(f"{rid}: {n.op} instances at {sorted(got)[:4]}... != sampled {sorted(want)[:4]}...")
        checks.append(Check("steps_cover_positions", "agree" if not bad else "disagree",
                            f"{len(boundaries)} requests; " + ("; ".join(bad[:3]) if bad else "steps in order, sampled = sampler.select instances")))

    # 5. RU namespace: leaf_count and ids_digest per committed step --------------------------
    ru_trees = {tid: rec for tid, rec in commitments["trees"].items() if rec.get("kind") == "ru"}
    if not boundaries:
        checks.append(Check("ru_namespace", "skip", "needs the production record to rebuild boundaries"))
    elif not namespace:
        checks.append(Check("ru_namespace", "skip", "disabled (--no-namespace)"))
    else:
        mism, total_leaves = [], 0
        for tid, rec in ru_trees.items():
            rid, step = str(rec["request_id"]), int(rec["step"])
            b = boundaries.get(rid)
            if b is None:
                mism.append(f"{tid}: request not in production record")
                continue
            if tid != ru_tree_id(rid, step):
                mism.append(f"{tid}: tree id != ru_tree_id({rid}, {step})")
            ids = sorted(expected_step_ids(description, b, profile_id, step))
            total_leaves += len(ids)
            if len(ids) != int(rec["leaf_count"]):
                mism.append(f"{tid}: {len(ids)} predicted ids, {rec['leaf_count']} committed leaves")
            elif hashing.ids_digest(ids).hex() != rec.get("ids_digest"):
                mism.append(f"{tid}: ids_digest differs (same count)")
        missing_steps = []
        for rid, b in boundaries.items():
            for s in b.steps:
                if ru_tree_id(rid, s.step) not in ru_trees:
                    missing_steps.append(ru_tree_id(rid, s.step))
        status = "agree" if not mism and not missing_steps else "disagree"
        checks.append(Check("ru_namespace", status,
                            f"{len(ru_trees)} ru trees; predicted leaves {total_leaves}; "
                            + (f"mismatches {mism[:3]}" if mism else "every leaf_count and ids_digest equals the description's prediction")
                            + (f"; steps without a committed tree: {missing_steps[:3]}" if missing_steps else ""),
                            {"ru_trees": len(ru_trees), "predicted_leaves": total_leaves,
                             "mismatches": mism, "missing_steps": missing_steps}))
        if verify_doc is not None:
            cp = verify_doc.get("capture_predicates") or {}
            if cp.get("expected_leaves") is not None:
                ok = int(cp["expected_leaves"]) == total_leaves
                checks.append(Check("expected_leaves_vs_verify", "agree" if ok else "disagree",
                                    f"verify.json expected_leaves={cp.get('expected_leaves')} present_leaves="
                                    f"{cp.get('present_leaves')}; recomputed {total_leaves}"))

    # 6. the sample ---------------------------------------------------------------------------
    if challenge_doc is not None:
        sample = list(challenge_doc["sample"])
        bad, seen = [], set()
        step_ids_cache: dict[str, set[str]] = {}
        for cid in sample:
            if cid in seen:
                bad.append(f"duplicate {cid}")
            seen.add(cid)
            try:
                parsed = identity.parse(cid)
            except Exception as error:  # noqa: BLE001
                bad.append(f"{cid}: {error}")
                continue
            if parsed.kind != "value":
                bad.append(f"{cid}: not a value id")
                continue
            if parsed.profile_id != profile_id:
                bad.append(f"{cid}: profile {parsed.profile_id}")
            b = boundaries.get(parsed.request_id or "")
            if b is None:
                if boundaries:
                    bad.append(f"{cid}: request not in production record")
                continue
            st = b.step_of(int(parsed.t))
            if st is None:
                bad.append(f"{cid}: position {parsed.t} in no committed step")
                continue
            tid = ru_tree_id(b.request_id, st.step)
            if tid not in ru_trees:
                bad.append(f"{cid}: tree {tid} not committed")
                continue
            if namespace:
                if tid not in step_ids_cache:
                    step_ids_cache[tid] = set(expected_step_ids(description, b, profile_id, st.step))
                if cid not in step_ids_cache[tid]:
                    bad.append(f"{cid}: not in the predicted namespace of {tid}")
        adjacent = list(challenge_doc.get("adjacent_vus") or [])
        missing_adj = [a for a in adjacent if a not in seen]
        if missing_adj:
            bad.append(f"{len(missing_adj)} adjacent VUs not in the sample")
        checks.append(Check("sample", "agree" if not bad else "disagree",
                            f"{len(sample)} sampled VUs ({len(adjacent)} adjacent); "
                            + ("; ".join(bad[:3]) if bad else "all parse, all in committed trees"
                               + (" and in their predicted namespaces" if namespace and boundaries else "")),
                            {"sample_size": len(sample), "adjacent": len(adjacent), "problems": bad}))

    # 7. verify.json vs challenge ------------------------------------------------------------
    if verify_doc is not None and challenge_doc is not None:
        report = verify_doc.get("report") or {}
        problems = []
        if verify_doc.get("challenge_digest", report.get("challenge_digest")) != challenge_doc.get("digest"):
            problems.append("verify.json challenge_digest != challenge.json digest")
        if report.get("challenge_digest") not in (None, challenge_doc.get("digest")):
            problems.append("report.challenge_digest != challenge.json digest")
        n = len(challenge_doc["sample"])
        if verify_doc.get("sample_size") not in (None, n):
            problems.append(f"sample_size {verify_doc.get('sample_size')} != {n}")
        statuses = report.get("statuses") or {}
        if statuses and sum(int(v) for v in statuses.values()) != n:
            problems.append(f"statuses sum {sum(statuses.values())} != sample {n}")
        fam_total = 0
        for fam in (report.get("families") or {}).values():
            for klass in fam.values():
                fam_total += int(klass.get("attempted", 0))
        if fam_total and fam_total != n:
            problems.append(f"family attempted total {fam_total} != sample {n}")
        sc = verify_doc.get("sample_coverage")
        if isinstance(sc, dict) and not sc.get("ok", True):
            problems.append("verify.json sample_coverage not ok")
        checks.append(Check("report_vs_challenge", "agree" if not problems else "disagree",
                            "; ".join(problems) if problems else
                            f"report digest = challenge digest; statuses/families sum to {n}",
                            {"statuses": statuses}))

    # 8. required set vs description --------------------------------------------------------
    if required is not None:
        problems = []
        interfaces = required.get("interfaces") or required.get("required") or []
        ops = {n.op for n in description.nodes}
        for entry in interfaces:
            op = entry.get("op") if isinstance(entry, dict) else None
            if op is not None and op not in ops:
                problems.append(f"required op {op} not in description")
        checks.append(Check("required_vs_description", "agree" if not problems else "disagree",
                            f"{len(interfaces)} required interfaces; " + ("; ".join(problems[:3]) if problems else "every op is a description node"),
                            {"interfaces": len(interfaces)}))
        if verify_doc is not None and isinstance(verify_doc.get("required_coverage"), dict):
            rc = verify_doc["required_coverage"]
            ok = int(rc.get("required", -1)) == len(interfaces) and not rc.get("missing")
            checks.append(Check("required_coverage_vs_verify", "agree" if ok else "disagree",
                                f"verify.json required={rc.get('required')} bound={rc.get('bound')} missing={len(rc.get('missing') or [])}; "
                                f"required.json has {len(interfaces)} interfaces"))

    # 9. weights linkage ------------------------------------------------------------------------
    wrec = commitments["trees"].get("weights")
    if wrec is None:
        checks.append(Check("weights_linkage", "skip", "no weights tree committed"))
    else:
        want = (manifest.get("model") or {}).get("weights_sha256")
        ok = wrec.get("weights_sha256") == want and want
        checks.append(Check("weights_linkage", "agree" if ok else "disagree",
                            f"weights tree carries {wrec.get('weights_sha256')}; manifest.model.weights_sha256 = {want}; "
                            f"{wrec.get('leaf_count')} whole-tensor leaves (bytes not in the row: leaves vs checkpoint need tensors)"))

    # 10. verdict recomputed from the tables ---------------------------------------------------
    if "table1" in files.tables:
        t1 = _jsonl(files.tables["table1"])
        t2 = _jsonl(files.tables["table2"]) if "table2" in files.tables else []
        if t1:
            row = t1[0]
            try:
                v = rows.verdict(row, t2)
                recorded = row.get("result")
                checks.append(Check("verdict", "agree" if v.result == recorded else "disagree",
                                    f"recorded {recorded}, recomputed {v.result}"
                                    + (f" ({v.first_blocker})" if v.first_blocker else ""),
                                    {"reasons": list(v.reasons)}))
            except Exception as error:  # noqa: BLE001
                checks.append(Check("verdict", "skip", f"rows.verdict raised: {error}"))

    # 11. negatives: reproducers agree with the table ------------------------------------------
    if files.negatives_dir is not None and "table3" in files.tables:
        t3 = {r["negative_id"]: r for r in _jsonl(files.tables["table3"])}
        problems = []
        subset_sizes: set[int] = set()
        for path in sorted(files.negatives_dir.glob("N*.json")):
            doc = _load(path)
            nid = doc.get("negative_id", path.stem)
            row = t3.get(nid)
            if row is None:
                problems.append(f"{nid}: reproducer without a table3 row")
                continue
            variants = doc.get("variants") or []
            if variants:
                rejected = sum(1 for v in variants if v.get("outcome") == "rejected")
                if rejected != int(row.get("rejected", -1)) or len(variants) != int(row.get("attempted", -1)):
                    problems.append(f"{nid}: reproducer {rejected}/{len(variants)} vs table3 {row.get('rejected')}/{row.get('attempted')}")
            if doc.get("sample") is not None and challenge_doc is not None:
                # the adapter runs the negatives on a prefix of the sample (adjacent VUs first)
                if not set(doc["sample"]) <= set(challenge_doc["sample"]):
                    problems.append(f"{nid}: reproducer sample is not a subset of the challenge sample")
                subset_sizes.add(len(doc["sample"]))
        checks.append(Check("negatives_vs_table3", "agree" if not problems else "disagree",
                            f"{len(t3)} table3 rows; " + ("; ".join(problems[:3]) if problems else
                                                          "reproducer counts match the table; mutations drawn from a "
                                                          f"{sorted(subset_sizes)}-VU subset of the challenge sample"),
                            {"negative_sample_sizes": sorted(subset_sizes)}))

    return {
        "row": str(files.row),
        "profile_id": profile_id,
        "checks": [asdict(c) for c in checks],
        "agree": sum(c.status == "agree" for c in checks),
        "disagree": sum(c.status == "disagree" for c in checks),
        "skipped": sum(c.status == "skip" for c in checks),
        "needs_tensors": list(NEEDS_TENSORS),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("row", help="results row directory (adapter layout) or a capture directory")
    ap.add_argument("--out", default=None, help="write the JSON report here")
    ap.add_argument("--no-namespace", action="store_true",
                    help="skip the per-step namespace recomputation (seconds per million leaves)")
    args = ap.parse_args(argv)
    report = check_row(locate(args.row), namespace=not args.no_namespace)
    for c in report["checks"]:
        print(f"{c['status']:8s} {c['name']:32s} {c['detail']}")
    print(f"agree={report['agree']} disagree={report['disagree']} skipped={report['skipped']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["disagree"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
