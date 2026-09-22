"""Verifier side of genuine selective preparation (workstream C, decision 3; RC 2026-09-07).

    PYTHONPATH=. python -m verity_vllm_adapter.selective_verifier --run <run dir> --handshake <dir> \
        --production <dir> --checkpoints manifests/checkpoints.json --case B1 --semantic-profile <json> \
        --k 4 --s 1/4096 --coverage-size 200 --out <dir>

Runs against :mod:`verity_vllm_adapter.selective_prover` through the file handshake:

1. ``READY_PHASE0`` -> read the pre-``r_q`` material (manifest, description, required set, every
   boundary, the weights, ``commitments_phase0.json``); assert that **no replay unit exists**; bind
   the statement to the verifier's references (C3a model, C3b architecture, C3c output) exactly as
   WB's verifier does, over the phase-0 commitments.
2. ``T_0`` = the selective statement (:func:`verity_vllm.selective.selective_statement`); the source
   draws its seed *after* ``T_0`` is fixed; ``r_q`` -> ``J`` (exactly ``k`` requests, or Bernoulli
   ``q``); write ``replay_challenge.json`` for the prover.
3. ``READY_PHASE2`` -> the final ``commitments.json``: phase-0 records unchanged, first-stage
   completeness of ``J`` (and no interior outside ``J``), the observed description equal to the
   registered one; ``T_2`` over ``J``'s interiors; ``r_s`` -> the certified sample (core Bernoulli at
   rate ``s`` over ``J``'s leaves) and the labelled coverage sample; re-derive the transcript.
4. Authenticated openings + the registered relation checkers over the sample (GPU, under the
   profile's engine flags), output linkage for every request of ``J``, the expected-leaf predicate
   per replay unit of ``J``.  Witness bytes (opened values + Merkle paths) are counted exactly.

Writes ``selective_verify.json``, ``challenge_transcript.json``, ``challenge.json``, ``results.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from verity_vllm import challenge as vchallenge
from verity_vllm import commit as vcommit
from verity_vllm import identity, relations, selective
from verity_vllm import required as vrequired
from verity_vllm.capture import WEIGHTS_TREE_ID, CaptureReader, ru_tree_id
from verity_vllm.commit import Opening, RangeOpening, TreeCache
from verity_vllm.description import Description
from verity_vllm_adapter import binding as vbinding
from verity_vllm_adapter import profile as prof
from verity_vllm_adapter.memo_cache import MemoCache
from verity_vllm_adapter.selective_prover import ABORT, CHALLENGE_FILE, READY0, READY2


def wait(hs: Path, name: str, timeout_s: float, log: Any) -> float:
    t0 = time.perf_counter()
    last = 0.0
    while not (hs / name).exists():
        if (hs / ABORT).exists():
            raise SystemExit(f"prover aborted: {(hs / ABORT).read_text()[:300]}")
        waited = time.perf_counter() - t0
        if waited > timeout_s:
            raise SystemExit(f"timeout waiting for {name}")
        if waited - last >= 60:
            log(f"waiting for {name} ({waited:.0f}s)")
            last = waited
        time.sleep(0.5)
    return time.perf_counter() - t0


def path_bytes(o: Opening | RangeOpening) -> int:
    if isinstance(o, RangeOpening):
        return o.path_bytes()
    return sum(32 for p in o.path if p is not None)


def witness_accounting(description: Description, reader: CaptureReader, cache: TreeCache, items: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Exact witness bytes of the checked VUs ``items = [(canonical_id, status)]``: every opened value (output
    and inputs) and its Merkle path, counted per use (``openings``) and once per distinct opened value
    (``unique_*``: a verifier authenticates a value once and reuses it, which is what the weight tensors need)."""
    value_bytes = path_b = openings = 0
    weights_value_bytes = weights_openings = 0
    seen: dict[str, tuple[int, int]] = {}
    profile_id = reader.profile_id
    for canonical_id, status in items:
        if status in ("structural", "unknown_relation", "unresolved"):
            continue
        try:
            cid = identity.parse(canonical_id)
            boundary = reader.boundary(cid.request_id or "")
            inst = description.instance_of(canonical_id, boundary)
            resolved = description.resolve_args(inst, boundary, profile_id)
        except Exception:  # noqa: BLE001
            continue
        ids = [canonical_id]
        for ref in resolved.values():
            if isinstance(ref, dict) and "const" in ref:
                continue
            ids.extend(ref if isinstance(ref, list) else [ref])
        for oid in ids:
            if oid in seen:
                vb, pb = seen[oid]
            else:
                try:
                    o = cache.open(oid)
                except Exception:  # noqa: BLE001
                    continue
                vb, pb = len(o.value), path_bytes(o)
                seen[oid] = (vb, pb)
            openings += 1
            value_bytes += vb
            path_b += pb
            if oid.startswith("w="):
                weights_value_bytes += vb
                weights_openings += 1
    uniq_w = sum(vb for oid, (vb, _) in seen.items() if oid.startswith("w="))
    uniq_v = sum(vb for vb, _ in seen.values())
    uniq_p = sum(pb for _, pb in seen.values())
    return {"openings": openings, "value_bytes": value_bytes, "path_bytes": path_b, "witness_bytes": value_bytes + path_b,
            "weights_value_bytes": weights_value_bytes, "weights_openings": weights_openings,
            "unique_openings": len(seen), "unique_value_bytes": uniq_v, "unique_path_bytes": uniq_p, "unique_witness_bytes": uniq_v + uniq_p,
            "unique_weights_value_bytes": uniq_w, "unique_weights_openings": sum(1 for oid in seen if oid.startswith("w=")),
            "unique_activation_value_bytes": uniq_v - uniq_w,
            "note": "every opening a checked VU used (output + inputs), value bytes + 32 B per Merkle path node; weight tensors are opened whole "
                    "(v2: a RangeOpening over all row blocks) by the same-runtime relations and dominate the per-use total; unique_* counts each "
                    "distinct opened value once"}


def leaf_predicate(description: Description, reader: CaptureReader, selected: list[str]) -> dict[str, Any]:
    """Expected leaves (from the description over the boundary) vs present leaves per replay unit of J."""
    profile_id = reader.profile_id
    out: dict[str, Any] = {"requests": 0, "replay_units": 0, "expected_leaves": 0, "present_leaves": 0, "unexpected_leaves": 0, "missing_examples": []}
    for rid in selected:
        b = reader.boundary(rid)
        expected_by_step: dict[int, set[str]] = {}
        for inst in description.instances(b):
            step = b.step_of(inst.t)
            if step is not None:
                expected_by_step.setdefault(step.step, set()).add(inst.output_id(profile_id))
        out["requests"] += 1
        for _, step in reader.ru_keys(rid):
            out["replay_units"] += 1
            present = {e.canonical_id for e in reader.ru(rid, step).index.entries}
            expected = expected_by_step.get(step, set())
            out["expected_leaves"] += len(expected)
            out["present_leaves"] += len(expected & present)
            out["unexpected_leaves"] += len(present - expected)
            for cid in sorted(expected - present)[:3]:
                if len(out["missing_examples"]) < 5:
                    out["missing_examples"].append(cid)
    out["ok"] = out["expected_leaves"] == out["present_leaves"] and out["unexpected_leaves"] == 0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--handshake", required=True)
    ap.add_argument("--production", required=True)
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", required=True)
    ap.add_argument("--semantic-profile", required=True)
    ap.add_argument("--k", type=int, default=None, help="exactly k requests (fixed-size selector); omit for Bernoulli --q")
    ap.add_argument("--q", default=None, help="Bernoulli request selection probability (when --k is not given)")
    ap.add_argument("--s", default="1/4096")
    ap.add_argument("--coverage-size", type=int, default=200)
    ap.add_argument("--challenge", choices=list(vchallenge.MODES), default="security")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait-timeout-s", type=float, default=7200.0)
    ap.add_argument("--no-checks", action="store_true", help="stop after the sample is drawn (no GPU relation checks)")
    ap.add_argument("--adapter-revision", default="unknown")
    args = ap.parse_args(argv)

    run = Path(args.run)
    hs = Path(args.handshake)
    hs.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "selective_verifier.log"

    def log(*a: Any) -> None:
        msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    def abort(reason: str) -> int:
        (hs / ABORT).write_text(reason + "\n")
        log("ABORT", reason)
        return 4

    T: dict[str, float] = {}
    t_all = time.perf_counter()
    T["wait_phase0_s"] = wait(hs, READY0, args.wait_timeout_s, log)
    t_work0 = time.perf_counter()
    reader = CaptureReader(run)
    if reader.ru_keys():
        return abort(f"{len(reader.ru_keys())} replay unit(s) exist before r_q: the run is not selective")
    commitments0 = json.loads((run / "commitments_phase0.json").read_text())
    c0_sha = hashlib.sha256((run / "commitments_phase0.json").read_bytes()).hexdigest()
    if json.loads((hs / READY0).read_text()).get("commitments_phase0_sha256") != c0_sha:
        return abort("READY_PHASE0 names a different commitments_phase0.json")
    kinds = {rec.get("kind") for rec in commitments0["trees"].values()}
    if not kinds <= {"boundary", "weights"}:
        return abort(f"phase-0 commitments carry {kinds}")
    manifest = reader.manifest
    # the checkers re-execute the registered runtime operators *under the profile*: the declared engine
    # flags are applied before the GPU relation registry is imported (as verify.py does), and the registry
    # is needed now because the description names its relations (embedding.same_runtime.v1, ...)
    for k_, v in (manifest.get("engine_flags") or {}).items():
        cur = os.environ.get(k_)
        if cur is not None and cur != str(v):
            raise SystemExit(f"environment {k_}={cur!r} conflicts with the profile's {k_}={v!r}")
        os.environ[k_] = str(v)
    t0 = time.perf_counter()
    import verity_vllm_adapter.relations_gpu as gpu_rel

    installed = gpu_rel.install()
    T["relations_install_s"] = time.perf_counter() - t0
    description = Description(reader.description)
    description.validate(known_relations=set(relations.ids()) | {"unresolved"})
    required_doc = reader.required if reader.has("required.json") else None
    request_ids = reader.request_ids()
    profile_id = reader.profile_id
    production_path = Path(args.production) / "production.jsonl"
    log(f"phase 0 received: {len(request_ids)} boundaries, weights {commitments0['trees'][WEIGHTS_TREE_ID]['leaf_count']} leaves, no replay unit")

    # --- bindings over the phase-0 commitments (the verifier's references) ---
    t0 = time.perf_counter()
    cp = prof.checkpoint_for(args.case, args.checkpoints)
    bindings = vbinding.bind(reader, commitments0, cp=cp, production_path=production_path, semantic_profile_path=args.semantic_profile)
    T["bindings_s"] = time.perf_counter() - t0
    for r in (bindings.model, bindings.architecture, bindings.output):
        log(f"binding {r.name:12s} {'ok ' if r.ok else 'FAILED'} {r.seconds:6.1f}s  {r.detail[:200]}")
    if not bindings.ok:
        return abort("bindings failed: " + "; ".join(bindings.problems)[:300])

    # --- T_0 and r_q ---
    t0 = time.perf_counter()
    n = len(request_ids)
    if args.k is not None:
        policy = vchallenge.Policy(Fraction(args.k, n), Fraction(args.s))
    else:
        policy = vchallenge.Policy.parse(args.q or "1", args.s)
    source = vchallenge.make_source(args.challenge, seed=args.seed)
    statement = selective.selective_statement(
        profile_id=profile_id, manifest_bytes=reader.manifest_bytes(), description_digest=description.digest().hex(),
        required_digest=vrequired.required_digest(required_doc).hex() if required_doc else None,
        commitments_phase0=commitments0, bindings_statement=bindings.statement(), policy=policy, source_opening=source.opening(),
        request_ids=request_ids, k=args.k, weights_tree_id=WEIGHTS_TREE_ID)
    session = selective.SelectiveSession(source, policy, k=args.k)
    t0_digest = session.accept_statement(statement)
    selected = session.challenge_replay(request_ids)
    T["statement_and_r_q_s"] = time.perf_counter() - t0
    (out / "statement.json").write_text(json.dumps(statement, indent=1) + "\n")
    selective.write_json_atomic(hs / CHALLENGE_FILE, {
        "T_0": t0_digest.hex(), "r_q": session.r_q.hex(), "selected": selected, "k": len(selected), "candidates": n,
        "selector": session.transcript["replay_challenge"].get("selector", {"kind": "bernoulli", "q": policy.to_json()["q"]}),
        "commitments_phase0_sha256": c0_sha, "utc": prof.utc_stamp()})
    T["work_before_r_q_s"] = time.perf_counter() - t_work0
    log(f"T_0 {t0_digest.hex()[:12]}; r_q -> J = {len(selected)}/{n}: {[s[:8] for s in selected]}")

    # --- phase 2: the interiors of J ---
    T["wait_phase2_s"] = wait(hs, READY2, args.wait_timeout_s, log)
    t_work2 = time.perf_counter()
    reader = CaptureReader(run)  # re-open: RUs and boundaries of J were written since
    commitments = vcommit.load_commitments(reader)
    changed = [tid for tid, rec in commitments0["trees"].items() if commitments["trees"].get(tid) != rec]
    if changed:
        return abort(f"phase-0 records changed: {changed[:3]}")
    fs = selective.first_stage_problems_for(reader, commitments, selected)
    if not fs["ok"]:
        return abort("first-stage: " + "; ".join(fs["problems"][:3]))
    observed = json.loads((run / "description_observed.json").read_text())
    ignored = set(vbinding.load_semantic_profile(args.semantic_profile).get("ignored_description_fields", []))
    exp_nodes = {nd["op"]: nd for nd in reader.description["nodes"]}
    obs_nodes = {nd["op"]: nd for nd in observed["nodes"]}
    arch_diff = sorted(set(exp_nodes) ^ set(obs_nodes))
    arch_diff += [op for op in sorted(set(exp_nodes) & set(obs_nodes))
                  if {k: v for k, v in exp_nodes[op].items() if k not in ignored} != {k: v for k, v in obs_nodes[op].items() if k not in ignored}]
    violations = sum(len(nd.get("arg_violations") or []) for nd in obs_nodes.values())
    if arch_diff or violations:
        return abort(f"observed description differs from the registered one on {len(arch_diff)} node(s) / {violations} arg violation(s): {arch_diff[:3]}")
    t0 = time.perf_counter()
    interiors, units = vchallenge.interiors_of(commitments, reader, selected)
    session.accept_interiors(interiors)
    from verity_vllm_adapter.verify import adjacent_chain  # CPU helpers only

    chain = adjacent_chain(description, reader, profile_id, "0", "prefill") + adjacent_chain(description, reader, profile_id, "0", "decode") \
        + adjacent_chain(description, reader, profile_id, str(max(0, len(description.nodes_of_type("attention")) // 2)), "decode")
    adjacent = [c for c in chain if identity.parse(c).request_id in set(selected)]
    resolve_ids = vchallenge.make_resolver(reader)
    coverage = vchallenge.make_coverage(description, reader, adjacent, args.coverage_size, selected)
    certified, coverage_sample = session.challenge_sample(units, resolve_ids, coverage=coverage)
    transcript = session.finish()
    T["interiors_and_r_s_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rederived = selective.rederive(transcript, statement=statement, interiors=interiors, request_ids=request_ids, units=units,
                                   resolve_ids=resolve_ids, coverage=coverage)
    T["rederive_s"] = time.perf_counter() - t0
    sample = list(dict.fromkeys(coverage_sample + certified))
    challenge = relations.make_challenge(commitments, description, sample, required_doc)
    digests = {p["phase"]: p["digest"] for p in transcript["phases"]}
    (out / "challenge_transcript.json").write_text(json.dumps(transcript, indent=1) + "\n")
    (out / "challenge.json").write_text(json.dumps({**challenge.to_json(), "selected_replay_units": selected, "certified_sample": certified,
                                                     "coverage_sample": coverage_sample, "phase_digests": digests,
                                                     "r_q": transcript["replay_challenge"]["r_q"], "r_s": transcript["sample_challenge"]["r_s"],
                                                     # the same context keys verify.py records (INT3, additive): a third party's re-derivation of the
                                                     # coverage sample (bundle rederive) and the sample membership of a chain need them
                                                     "challenge": {"mode": source.mode, "source": source.opening().get("source"), "policy": policy.to_json(),
                                                                   "selector": statement["selective"]["selection"], "protocol": "selective"},
                                                     "coverage_size": args.coverage_size, "adjacent_vus": adjacent}, indent=1) + "\n")
    log(f"T_2 {digests['interior'][:12]}: population {units.count} VUs in {len(interiors)} RUs; r_s -> certified {len(certified)}, "
        f"coverage {len(coverage_sample)} (adjacent {len(adjacent)}); union {len(sample)}; rederived={rederived['ok']}")
    if not rederived["ok"]:
        return abort("transcript does not re-derive: " + "; ".join(rederived["problems"]))

    tokens = selective.replayed_positions(reader, selected)
    pred = leaf_predicate(description, reader, selected)
    log(f"leaf predicate over J: expected {pred['expected_leaves']} present {pred['present_leaves']} unexpected {pred['unexpected_leaves']} ok={pred['ok']}")

    # the GUIDANCE §32.5 unconditional output type check (lane OUT16): every present output_token
    # boundary leaf of EVERY request — not only J — opened against the phase-0-bound roots and
    # required to sit inside the declared token domain (int32 container, declared 16 -> high
    # 16-bit word == 0).  Runs on every audit, before and independent of the relation sampling
    # (also under --no-checks); a domain check gated on r_q would leave out-of-domain outputs in
    # unselected requests surviving with 1 - q (the §31.3 two-stage trap).  On a description
    # without a declared domain this checks nothing and old records verify identically.
    t0 = time.perf_counter()
    domains = [relations.check_output_domains(description, reader, commitments, rid).to_json() for rid in request_ids]
    T["output_domains_s"] = time.perf_counter() - t0
    domains_ok = all(not d["problems"] for d in domains)
    log(f"output domains: {sum(d['checked'] for d in domains)} token leaves over {len(domains)} requests "
        f"in {T['output_domains_s']:.3f}s ok={domains_ok}")

    report_doc: dict[str, Any] | None = None
    witness: dict[str, Any] | None = None
    linkage: list[dict[str, Any]] = []
    if not args.no_checks:
        cache = MemoCache(reader)  # weight tensors authenticated once, reused across VUs (RC)
        t0 = time.perf_counter()
        report = relations.check(description, reader, commitments, sample, challenge=challenge, required_doc=required_doc, cache=cache)
        T["check_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        witness = witness_accounting(description, reader, cache, [(r.canonical_id, r.status) for r in report.results])
        T["witness_accounting_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        for rid in selected:
            linkage.append(relations.check_output_linkage(description, reader, commitments, rid, cache=cache).to_json())
        T["linkage_s"] = time.perf_counter() - t0
        report_doc = report.to_json()
        with (out / "results.jsonl").open("w") as f:
            for r in report.results:
                f.write(json.dumps(r.to_json()) + "\n")
        log(f"check: {len(sample)} VUs in {T['check_s']:.1f}s; statuses {report.statuses()}; authentication {report.authentication()}; "
            f"witness {witness['witness_bytes'] / 1e6:.1f} MB over {witness['openings']} openings; linkage ok for "
            f"{sum(1 for l in linkage if not l['problems'])}/{len(linkage)} requests")
    T["work_after_phase2_s"] = time.perf_counter() - t_work2
    T["total_s"] = time.perf_counter() - t_all
    T["work_total_s"] = T["work_before_r_q_s"] + T["work_after_phase2_s"]

    statuses = report_doc["statuses"] if report_doc else {}
    all_exact = bool(report_doc) and set(statuses) <= {"exact"}
    verdict = "PASS" if (all_exact and pred["ok"] and domains_ok and all(not l["problems"] for l in linkage)) else (
        "NOT_CHECKED" if args.no_checks and domains_ok else "FAIL")
    doc = {
        "schema": "verity-vllm/selective-verify/v1",
        "run_dir": str(run), "case": args.case, "profile_id": profile_id, "adapter_revision": args.adapter_revision,
        "challenge": {"mode": source.mode, "policy": policy.to_json(), "selector": statement["selective"]["selection"],
                      "phase_digests": digests, "r_q": transcript["replay_challenge"]["r_q"], "r_s": transcript["sample_challenge"]["r_s"], "rederived": rederived},
        "requests_total": n, "selected": selected, "k": len(selected),
        "replayed_tokens": tokens,
        "population": {"vus": units.count, "replay_units": len(interiors)},
        "certified_sample": len(certified), "coverage_sample": len(coverage_sample), "adjacent": len(adjacent), "sample_union": len(sample),
        "bindings": bindings.to_json(),
        "first_stage": fs,
        "leaf_predicate": pred,
        "architecture_observed_equals_registered": True,
        "relations_installed": installed,
        "check": {"statuses": statuses, "authentication": report_doc.get("authentication") if report_doc else None,
                  "families": report_doc.get("families") if report_doc else None} if report_doc else None,
        "witness": witness,
        "output_linkage": linkage,
        "output_domains": domains,
        "obligations": {
            "vus_checked": len(sample) if report_doc else 0, "vus_certified": len(certified), "vus_coverage": len(coverage_sample),
            "output_records_linked": sum(l["checked"] for l in linkage), "binding_checks": 3, "boundary_roots_bound": len(request_ids),
            "leaf_predicate_leaves": pred["expected_leaves"],
            "output_domain_leaves": sum(d["checked"] for d in domains),
        },
        "timing": T, "verdict": verdict, "utc": prof.utc_stamp(),
    }
    (out / "selective_verify.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    log(f"RESULT {verdict} k={len(selected)} certified={len(certified)} coverage={len(coverage_sample)} total={T['total_s']:.1f}s work={T['work_total_s']:.1f}s")
    return 0 if verdict in ("PASS", "NOT_CHECKED") else 1


if __name__ == "__main__":
    sys.exit(main())
