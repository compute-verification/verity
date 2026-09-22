"""Fresh-process verifier (contract 15.1)::

    python -m verity_vllm.verify <bundle_dir> --seed N [--sample K | --exhaustive] [--out DIR]
        [--case CASE] [--worker W] [--adapter-revision REV] [--experiment-id ID]
        [--production-record production.jsonl] [--negatives N01,N02,...|none]
        [--torch auto|never|require] [--plugin module:function ...] [--no-audit]
        [--request-id RID ...] [--op-type T ...] [--results-root DIR]
        [--bound-s S [--bound-q Q]]

The verifier is a separate process that reads only the **bundle** (a capture run directory as in
``docs/vllm-poc/capture-format.md`` with ``commitments.json`` present: ``manifest.json``,
``description.json``, ``required.json``, ``ru/**``, ``weights/``).  It never imports or talks to
the process that produced it.  In order:

1. load the bundle; validate structure; (unless ``--no-audit``) recompute every leaf and root
   against ``commitments.json`` and check the weights' source linkage (N06);
2. install checkers: the built-in ``independent`` numpy relations, the torch ``same_runtime``
   relations when torch is importable (``--torch``), and any ``--plugin module:function``
   (``verity_vllm_sampler.registry:install`` is tried by default when importable);
3. derive the challenge **after** reading the roots: ``effective_seed =
   H("verity-vllm/challenge/v1", "seed", seed_be64, json_digest(commitments))[:8]`` seeds the
   stratified sample; the sample is
   extended with an adjacent producer/consumer pair when it has none (15.1); the challenge binds
   commitments, description, required set and sample (N13);
4. authenticate every opening used and run the registered relation of every selected VU
   (``relations.check``); check sample coverage (N07), required-interface coverage, output
   linkage (N14) and the adjacent-pair invariant;
5. run the applicable negatives on the loaded bundle (``negatives.NegativeHarness``);
6. write ``challenge.json``, ``report.json``, ``table1.jsonl``, ``table2.jsonl``,
   ``table3.jsonl`` and ``summary.txt`` to ``--out`` (default ``<bundle>/verify``), and, with
   ``--results-root``, copy the tables into ``<root>/<experiment_id>/`` for the renderer.

Exit status: 0 when every attempted check agreed (verdict ``PASS`` or ``PARTIAL``), 1 when
something was rejected (``FAIL``), 2 when the bundle could not be verified at all.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import commit, firststage, hashing, relations, required, rows, sampling
from .capture import CaptureError, CaptureReader
from .description import Description, DescriptionError
from .negatives import NEGATIVE_IDS, NegativeHarness

DEFAULT_PLUGINS = ("verity_vllm_sampler.registry:install",)
DEFAULT_SAMPLE = 64


class BundleError(Exception):
    pass


@dataclass(slots=True)
class Bundle:
    path: Path
    reader: CaptureReader
    description: Description
    required: dict[str, Any] | None
    commitments: dict[str, Any]

    @property
    def manifest(self) -> dict[str, Any]:
        return self.reader.manifest

    @property
    def profile_id(self) -> str:
        return self.reader.profile_id


def load_bundle(path: str | os.PathLike[str]) -> Bundle:
    path = Path(path)
    if not path.is_dir():
        raise BundleError(f"{path} is not a directory")
    try:
        reader = CaptureReader(path)
        _ = reader.manifest
    except (CaptureError, OSError, ValueError) as error:
        raise BundleError(f"{path}: cannot read the capture: {error}") from error
    if not reader.has("description.json"):
        raise BundleError(f"{path}: description.json is missing")
    if reader.commitments is None:
        raise BundleError(f"{path}: commitments.json is missing (not a bundle, just a capture)")
    try:
        description = Description(reader.description)
    except DescriptionError as error:
        raise BundleError(f"{path}: description.json is invalid: {error}") from error
    try:
        commitments = commit.load_commitments(reader)
    except commit.CommitError as error:
        raise BundleError(f"{path}: {error}") from error
    if commitments.get("profile_id") not in (None, reader.profile_id):
        raise BundleError(
            f"{path}: commitments.json is for profile {commitments.get('profile_id')}, "
            f"manifest hashes to {reader.profile_id}"
        )
    required_doc = reader.required if reader.has("required.json") else None
    return Bundle(path, reader, description, required_doc, commitments)


# --- challenge -----------------------------------------------------------------------------------


def effective_seed(seed: int, commitments: Mapping[str, Any]) -> tuple[int, str]:
    """``(effective_seed, roots_digest_hex)``: the sample seed is a function of the caller's seed
    *and* of the committed roots, so it cannot be chosen before the roots exist."""
    roots = hashing.json_digest(dict(commitments))
    mixed = hashing.H(hashing.CHALLENGE_TAG, b"seed", int(seed).to_bytes(8, "big"), roots)
    return int.from_bytes(mixed[:8], "big"), roots.hex()


# --- checkers ------------------------------------------------------------------------------------


def apply_engine_flags(manifest: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    """Set the bundle manifest's ``engine_flags`` (the profile's declared environment, e.g.
    ``VLLM_BATCH_INVARIANT=1``) in this process so the runtime relations re-execute under the
    profile.  A value already in the environment that differs from the declared one is a
    conflict: with ``strict`` it raises :class:`BundleError`; otherwise it is recorded and the
    environment is left alone.  Returns ``{"applied": {...}, "conflicts": {...}}``."""
    applied: dict[str, str] = {}
    conflicts: dict[str, dict[str, str]] = {}
    for key, value in (manifest.get("engine_flags") or {}).items():
        wanted = str(value)
        current = os.environ.get(str(key))
        if current is not None and current != wanted:
            conflicts[str(key)] = {"environment": current, "declared": wanted}
            continue
        os.environ[str(key)] = wanted
        applied[str(key)] = wanted
    if conflicts and strict:
        detail = ", ".join(
            f"{k}={v['environment']!r} (declared {v['declared']!r})" for k, v in conflicts.items()
        )
        raise BundleError(
            f"environment conflicts with the profile's declared engine_flags: {detail}"
        )
    return {"applied": applied, "conflicts": conflicts}


def install_checkers(
    torch_mode: str = "auto", plugins: Iterable[str] = DEFAULT_PLUGINS
) -> dict[str, Any]:
    info: dict[str, Any] = {"torch": None, "torch_relations": [], "plugins": {}}
    if torch_mode not in ("auto", "never", "require"):
        raise ValueError("--torch must be auto|never|require")
    if torch_mode != "never":
        try:
            import torch  # noqa: F401

            from . import relations_torch

            info["torch"] = torch.__version__
            info["torch_relations"] = relations_torch.install(replace=True)
        except ImportError as error:
            if torch_mode == "require":
                raise BundleError(f"--torch require: {error}") from error
            info["torch"] = None
    for spec in plugins:
        module_name, _, func_name = spec.partition(":")
        func_name = func_name or "install"
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            info["plugins"][spec] = {"installed": False, "reason": f"not importable: {error}"}
            continue
        try:
            registered = getattr(module, func_name)()
        except Exception as error:  # a broken plugin is reported, never fatal
            info["plugins"][spec] = {
                "installed": False,
                "reason": f"{type(error).__name__}: {error}",
            }
            continue
        info["plugins"][spec] = {"installed": True, "relations": list(registered or [])}
    return info


# --- production record (boundary noninterference) -----------------------------------------------


def load_production_record(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """W1's ``production.jsonl`` (one record per line) or a JSON list / dict keyed by request."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        doc = json.loads(text)
        if isinstance(doc, dict) and "request_id" not in doc:
            records = [{"request_id": k, **v} for k, v in doc.items()]
        elif isinstance(doc, dict):
            records = [doc]
        else:
            records = list(doc)
    return {str(r["request_id"]): r for r in records}


def compare_production(
    reader: CaptureReader, record: Mapping[str, Mapping[str, Any]], method: str
) -> dict[str, Any]:
    compared = mismatched = 0
    details: list[str] = []
    for rid in reader.request_ids():
        prod = record.get(rid)
        if prod is None:
            details.append(f"{rid}: no production record")
            continue
        boundary = reader.boundary(rid)
        for name, ours in (
            ("prompt_token_ids", boundary.prompt_token_ids),
            ("output_token_ids", boundary.output_token_ids),
        ):
            theirs = [int(x) for x in prod.get(name, [])]
            n = max(len(ours), len(theirs))
            compared += n
            bad = sum(
                1
                for i in range(n)
                if i >= len(ours) or i >= len(theirs) or int(ours[i]) != theirs[i]
            )
            mismatched += bad
            if bad:
                details.append(f"{rid}: {name} differs in {bad} of {n} scalars")
        if "finish_reason" in prod:
            compared += 1
            if str(prod["finish_reason"]) != str(boundary.finish_reason):
                mismatched += 1
                details.append(
                    f"{rid}: finish_reason {boundary.finish_reason!r} != {prod['finish_reason']!r}"
                )
    return {
        "identical": compared > 0 and mismatched == 0 and not details,
        "compared": compared,
        "mismatched_scalars": mismatched,
        "reference_method": method,
        "_details": details,
    }


# --- the run -------------------------------------------------------------------------------------


@dataclass(slots=True)
class VerifyResult:
    bundle: Path
    out_dir: Path
    experiment_id: str
    seed: int
    effective_seed: int
    roots_digest: str
    challenge: relations.Challenge
    sample: list[str]
    report: relations.CheckReport
    sample_coverage: relations.SampleCoverage
    required_coverage: required.Coverage | None
    linkage: list[relations.LinkageReport]
    adjacent: dict[str, Any]
    structural_problems: list[str]
    negatives: list[Any]
    checkers: dict[str, Any]
    table1: dict[str, Any]
    table2: list[dict[str, Any]]
    table3: list[dict[str, Any]]
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return str(self.table1["result"])

    @property
    def exit_code(self) -> int:
        return 0 if self.verdict in ("PASS", "PARTIAL") else 1


def _production_changes(manifest: Mapping[str, Any]) -> str:
    flags = manifest.get("engine_flags") or {}
    if not flags:
        return "none"
    return "flags:[" + ", ".join(f"{k}={v}" for k, v in sorted(flags.items())) + "]"


def _replay_restrictions(manifest: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    comp = manifest.get("compilation") or {}
    if comp.get("enforce_eager"):
        out.append("eager-only")
    interception = (manifest.get("adapter") or {}).get("interception")
    if interception:
        out.append(f"interception:{interception}")
    return out


def verify(
    bundle_dir: str | os.PathLike[str],
    *,
    seed: int,
    sample_size: int | None = DEFAULT_SAMPLE,
    exhaustive: bool = False,
    out_dir: str | os.PathLike[str] | None = None,
    case: str | None = None,
    worker: str = "verify",
    adapter_revision: str | None = None,
    experiment_id: str | None = None,
    utc: str | None = None,
    production_record: str | os.PathLike[str] | None = None,
    negatives: Sequence[str] | None = NEGATIVE_IDS,
    audit: bool = True,
    torch_mode: str = "auto",
    plugins: Iterable[str] = DEFAULT_PLUGINS,
    request_ids: Sequence[str] | None = None,
    op_types: Sequence[str] | None = None,
    production_changes: str | None = None,
    replay_restrictions: Sequence[str] | None = None,
    results_root: str | os.PathLike[str] | None = None,
    registry: Mapping[str, relations.Relation] | None = None,
    bound_q: float | None = None,
    bound_s: float | None = None,
) -> VerifyResult:
    """Verify ``bundle_dir``; see the module docstring for the steps.

    ``bound_s`` (and optionally ``bound_q``, default 1.0) additionally draws the
    ``Bound``-compatible independent per-request / per-VU sample (:mod:`verity_vllm.sampling`),
    checks it, and records it in ``bound_challenge.json`` / ``bound_report.json`` and under
    ``report.json["bound_sample"]``; the stratified sample and the tables are unchanged.
    ``sampling_plan.json`` always records the plan(s) actually used.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    bundle = load_bundle(bundle_dir)
    reader, description, commitments = bundle.reader, bundle.description, bundle.commitments
    out = Path(out_dir) if out_dir is not None else bundle.path / "verify"
    out.mkdir(parents=True, exist_ok=True)
    manifest = bundle.manifest
    case = case or str(manifest.get("case") or bundle.path.name)
    adapter_revision = adapter_revision or str(
        (manifest.get("adapter") or {}).get("revision") or "unknown"
    )
    utc = utc or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    experiment_id = experiment_id or rows.experiment_id(case, bundle.profile_id, utc)
    common = rows.Common(experiment_id, case, bundle.profile_id, worker, adapter_revision)
    timings["load"] = time.perf_counter() - t0

    # 1. structure and (optionally) the full audit
    t0 = time.perf_counter()
    structural: list[str] = []
    structural += [f"capture: {p}" for p in reader.validate(check_bytes=audit)]
    for rid in reader.request_ids():
        if reader.has_boundary(rid):
            structural += [f"boundary {rid}: {p}" for p in reader.boundary(rid).validate()]
            # prefix-cache bindings (R3): same tokens on 0..t between the hit and its source;
            # recompute units (R4): repeated identities carry the first unit's bytes
            structural += [
                f"boundary {rid}: {p}"
                for p in reader.boundary(rid).cached_prefix_problems(
                    lambda src: reader.boundary(src) if reader.has_boundary(src) else None
                )
            ]
            structural += [f"recompute {rid}: {p}" for p in reader.recompute_problems(rid)]
        else:
            structural.append(f"boundary {rid}: missing")
    if audit:
        structural += [f"commitments: {p}" for p in commit.verify_commitments(reader, commitments)]
        structural += [f"weights: {p}" for p in commit.verify_source_linkage(reader, commitments)]
    # review fix (W2c, Q2): the whole request is the first-stage unit -- every request's boundary
    # and every one of its steps must have a committed root before a challenge is drawn.
    first_stage = firststage.first_stage_problems(reader, commitments)
    structural += [f"first-stage: {p}" for p in first_stage.problems]
    timings["audit"] = time.perf_counter() - t0

    # 2. checkers, under the profile: the manifest's declared engine flags are applied to this
    # process before any runtime relation is imported (W9 §8.1 / W1 B1-p1: without
    # VLLM_BATCH_INVARIANT=1 a same-runtime rmsnorm check launches another kernel variant and
    # is 1 ulp off; the adapter verifier does the same). A conflicting value already in the
    # environment is refused when runtime relations may run; with --torch never (numpy-only
    # independent relations) it is recorded, not fatal.
    engine_flags = apply_engine_flags(manifest, strict=torch_mode != "never")
    checkers = install_checkers(torch_mode, plugins)
    checkers["engine_flags"] = engine_flags
    registry = relations.REGISTRY if registry is None else registry
    checkers["registry"] = relations.registry_json(registry)

    # 3. challenge, after the roots are known
    t0 = time.perf_counter()
    eff_seed, roots_digest = effective_seed(seed, commitments)
    if exhaustive:
        sample: list[str] = []
        for rid in request_ids or reader.request_ids():
            for cid, inst in description.expected_ids(
                reader.boundary(rid), reader.profile_id
            ).items():
                if op_types and inst.node.op_type not in op_types:
                    continue
                if reader.locate(cid) is not None:
                    sample.append(cid)
    else:
        sample = relations.draw_sample(
            description,
            reader,
            sample_size or DEFAULT_SAMPLE,
            seed=eff_seed,
            request_ids=request_ids,
            op_types=op_types,
        )
    sample, pairs = relations.ensure_adjacent_pair(description, reader, sample, seed=eff_seed)
    # review fix (W2c, Q2): a challenge may only name leaves of committed trees
    try:
        firststage.assert_sample_committed(sample, reader, commitments)
    except firststage.FirstStageError as error:
        structural.append(f"first-stage: challenge references an uncommitted tree: {error}")
    challenge = relations.make_challenge(commitments, description, sample, bundle.required)
    _write_json(
        out / "challenge.json",
        {
            "bundle": str(bundle.path),
            "seed": seed,
            "roots_digest": roots_digest,
            "effective_seed": eff_seed,
            "derivation": (
                'effective_seed = be64(H("verity-vllm/challenge/v1", "seed", be64(seed), '
                "json_digest(commitments.json))[:8])"
            ),
            "exhaustive": exhaustive,
            "adjacent_pairs": [p.to_json() for p in pairs],
            "challenge": challenge.to_json(),
        },
    )
    # Q3: the Bound-compatible plan, drawn alongside (opt-in), never replacing the stratified one
    bound_plan: sampling.BernoulliPlan | None = None
    bound_challenge: relations.Challenge | None = None
    if bound_s is not None:
        bound_plan = sampling.bernoulli_plan(
            description,
            reader,
            q=1.0 if bound_q is None else bound_q,
            s=bound_s,
            seed=sampling.bernoulli_seed(seed, commitments),
            request_ids=request_ids,
            op_types=op_types,
        )
        bound_challenge = relations.make_challenge(
            commitments, description, bound_plan.selected, bundle.required
        )
        _write_json(
            out / "bound_challenge.json",
            {"bundle": str(bundle.path), "plan": bound_plan.to_json(), "challenge": bound_challenge.to_json()},
        )
    timings["challenge"] = time.perf_counter() - t0

    # 4. openings + relations, coverage, linkage, adjacency
    t0 = time.perf_counter()
    report = relations.check(
        description,
        reader,
        commitments,
        sample,
        challenge=challenge,
        required_doc=bundle.required,
        registry=registry,
    )
    sample_cov = relations.verify_sample_coverage(sample, report)
    req_cov = required.coverage(bundle.required, description, reader) if bundle.required else None
    linkage = [
        relations.check_output_linkage(description, reader, commitments, rid)
        for rid in reader.request_ids()
    ]
    adjacent = relations.adjacent_check(pairs, report)
    if not sample_cov.ok:
        structural.append(
            f"sample coverage: missing={sample_cov.missing[:3]} "
            f"duplicated={sample_cov.duplicated[:3]} extra={sample_cov.extra[:3]}"
        )
    for lr in linkage:
        structural += [f"output linkage {lr.request_id}: {p}" for p in lr.problems]
    if req_cov is not None and req_cov.unexpected_ids:
        structural.append(f"namespace: {len(req_cov.unexpected_ids)} committed ids not predicted")
    if not pairs:
        structural.append("adjacent_vus: the capture has no producer->consumer pair to check")
    elif not adjacent["ok"]:
        # review fix (W2c): a pair whose VUs could not be executed (`missing` weights referenced
        # by sha, an unimplemented relation) is a coverage gap -- the table 2 rows already make
        # the verdict PARTIAL -- not a structural failure; only a definite disagreement is.
        definite = {"mismatch", "auth_failed", "structural", "schema"}
        disagreed = any(
            row["producer_status"] in definite
            or row["consumer_status"] in definite
            or (row["producer_status"] == row["consumer_status"] == "exact")  # exact, no input opened
            for row in adjacent["pairs"]
        )
        if disagreed:
            structural.append("adjacent_vus: no producer->consumer pair came out exact end to end")
        else:
            adjacent["reason"] = "no pair executable (missing inputs or unimplemented relations)"
    timings["check"] = time.perf_counter() - t0

    # 4b. the Bound-compatible sample, checked and recorded separately (rows unchanged)
    bound_report: relations.CheckReport | None = None
    if bound_plan is not None and bound_challenge is not None:
        t0 = time.perf_counter()
        bound_report = relations.check(
            description,
            reader,
            commitments,
            bound_plan.selected,
            challenge=bound_challenge,
            required_doc=bundle.required,
            registry=registry,
        )
        _write_json(
            out / "bound_report.json",
            {
                "plan": bound_plan.to_json(),
                "challenge_digest": bound_challenge.digest().hex(),
                "check": bound_report.to_json(),
                "sample_coverage": relations.verify_sample_coverage(
                    bound_plan.selected, bound_report
                ).to_json(),
            },
        )
        timings["bound_check"] = time.perf_counter() - t0
    _write_json(
        out / "sampling_plan.json",
        sampling.plan_document(
            stratified=sampling.describe_stratified(
                description, reader, sample, n=sample_size or DEFAULT_SAMPLE,
                effective_seed=eff_seed, adjacent=sum(1 for _ in pairs),
            ),
            bernoulli=bound_plan,
            challenge_digests={
                "commitments_digest": challenge.commitments_digest,
                "description_digest": challenge.description_digest,
                "required_digest": challenge.required_digest,
                "stratified_challenge": challenge.digest().hex(),
                "bernoulli_challenge": bound_challenge.digest().hex() if bound_challenge else None,
            },
        ),
    )

    # 5. negatives on the loaded bundle
    t0 = time.perf_counter()
    neg_results: list[Any] = []
    if negatives:
        harness = NegativeHarness(
            bundle.path, out_dir=out / "negatives", sample=sample, seed=eff_seed, registry=registry
        )
        neg_results = harness.run(list(negatives))
    timings["negatives"] = time.perf_counter() - t0

    # 6. tables
    if production_record is not None:
        record = load_production_record(production_record)
        bni = compare_production(
            reader, record, f"production record {Path(production_record).name}"
        )
        bni_details = bni.pop("_details")
    else:
        bni = {
            "identical": False,
            "compared": 0,
            "mismatched_scalars": 0,
            "reference_method": "none: no production record supplied to the verifier",
        }
        bni_details = []
    boundaries = [reader.boundary(rid) for rid in reader.request_ids()]
    prompt_lens = [len(b.prompt_token_ids) for b in boundaries] or [0]
    workload = {
        "requests": len(boundaries),
        "generated_positions": sum(len(b.output_token_ids) for b in boundaries),
        "input_len_range": [min(prompt_lens), max(prompt_lens)],
        "peak_concurrency_achieved": int(
            (manifest.get("workload") or {}).get("peak_concurrency_achieved", 0)
        ),
    }
    fams = report.families()
    tok = {"exact": 0, "attempted": 0}
    for klass_map in (fams.get("sampler", {}), fams.get("argmax", {})):
        for fam in klass_map.values():
            tok["exact"] += fam.exact
            tok["attempted"] += fam.attempted
    table3 = rows.table3_rows(common, neg_results)
    table2 = rows.table2_rows_from_report(common, report, description)
    table1 = rows.table1_row(
        common,
        workload=workload,
        production_changes=production_changes or _production_changes(manifest),
        replay_restrictions=list(replay_restrictions)
        if replay_restrictions is not None
        else _replay_restrictions(manifest),
        token_selection=tok,
        boundary_noninterference=bni,
        required_coverage=req_cov.to_json() if req_cov else {"bound": 0, "required": 0},
        authentication=report.authentication(),
        relation_checks=report.relation_checks(),
        structural={
            "result": "ok" if not structural else "fail",
            "failing_invariant": structural[0] if structural else None,
        },
        negative=rows.negative_summary(table3),
        table2_rows=table2,
    )
    rows.write_jsonl(out / "table1.jsonl", [table1])
    rows.write_jsonl(out / "table2.jsonl", table2)
    rows.write_jsonl(out / "table3.jsonl", table3)

    result = VerifyResult(
        bundle.path, out, experiment_id, seed, eff_seed, roots_digest, challenge, sample, report,
        sample_cov, req_cov, linkage, adjacent, structural, neg_results, checkers,
        table1, table2, table3, timings,
    )  # fmt: skip
    _write_json(
        out / "report.json",
        {
            "bundle": str(bundle.path),
            "experiment_id": experiment_id,
            "profile_id": bundle.profile_id,
            "verdict": {"result": table1["result"], "first_blocker": table1["first_blocker"]},
            "challenge": {
                "seed": seed,
                "effective_seed": eff_seed,
                "roots_digest": roots_digest,
                "digest": challenge.digest().hex(),
                "sample_size": len(sample),
            },
            "check": report.to_json(),
            "sample_coverage": sample_cov.to_json(),
            "required_coverage": (
                {**req_cov.to_json(), "missing": req_cov.missing[:20]} if req_cov else None
            ),
            # 15.3, recorded separately: did the committed trajectory follow the boundary records
            # (output linkage), and did the *unforced* sampler relation agree (token_selection)
            "trajectory_followed": all(lr.ok for lr in linkage) and bool(linkage),
            "output_linkage": [lr.to_json() for lr in linkage],
            "unforced_sampler_agrees": tok,
            "adjacent_vus": adjacent,
            "structural": structural,
            "first_stage": first_stage.to_json(),
            "bound_sample": (
                {
                    "plan": bound_plan.to_json(),
                    "challenge_digest": bound_challenge.digest().hex(),
                    "authentication": bound_report.authentication(),
                    "relation_checks": bound_report.relation_checks(),
                    "statuses": bound_report.to_json().get("statuses"),
                }
                if bound_plan is not None and bound_challenge is not None and bound_report is not None
                else None
            ),
            "boundary_noninterference": {**bni, "details": bni_details[:20]},
            "negatives": [r.to_json() for r in neg_results],
            "checkers": checkers,
            "seconds": timings,
        },
    )
    (out / "summary.txt").write_text(summary_text(result), encoding="utf-8")
    if results_root is not None:
        dest = Path(results_root) / experiment_id
        dest.mkdir(parents=True, exist_ok=True)
        for name in (
            "table1.jsonl",
            "table2.jsonl",
            "table3.jsonl",
            "challenge.json",
            "report.json",
        ):
            shutil.copyfile(out / name, dest / name)
        for name in ("manifest.json", "commitments.json"):
            if (bundle.path / name).exists():
                shutil.copyfile(bundle.path / name, dest / name)
        desc_path = bundle.path / "description.json"
        if desc_path.exists() and desc_path.stat().st_size <= 5_000_000:
            shutil.copyfile(desc_path, dest / "description.json")
        elif desc_path.exists():
            (dest / "description.sha256").write_text(
                hashing.sha256_hex(desc_path.read_bytes()) + f"  {desc_path}\n"
            )
        shutil.copyfile(out / "summary.txt", dest / "log.txt")
    return result


def summary_text(r: VerifyResult) -> str:
    t1 = r.table1
    lines = [
        f"verity-vllm verify: {r.bundle}",
        f"experiment_id {r.experiment_id}  profile {t1['profile_id']}  case {t1['case']}",
        f"verdict {t1['result']}"
        + (f"  first blocker: {t1['first_blocker']}" if t1["first_blocker"] else ""),
        "",
        f"seed {r.seed} -> effective seed {r.effective_seed} "
        f"(roots digest {r.roots_digest[:16]}...)",
        f"challenge {r.challenge.digest().hex()[:16]}...  sample {len(r.sample)} VUs",
        f"authentication {t1['authentication']['valid']}/{t1['authentication']['attempted']} "
        "openings valid",
        "relation checks: "
        + ", ".join(f"{k} {v['exact']}/{v['attempted']}" for k, v in t1["relation_checks"].items()),
        f"token selection {t1['token_selection']['exact']}/{t1['token_selection']['attempted']}",
        f"required coverage {t1['required_coverage']['bound']}"
        f"/{t1['required_coverage']['required']}",
        f"negatives {t1['negative']['rejected']}/{t1['negative']['attempted']} rejected",
        f"structural {t1['structural']['result']}"
        + (
            f": {t1['structural']['failing_invariant']}"
            if t1["structural"]["failing_invariant"]
            else ""
        ),
        f"boundary noninterference: {t1['boundary_noninterference']}",
        "",
        "per op family (class): exact/attempted [other statuses]",
    ]
    for op_type, fams in sorted(r.report.families().items()):
        for klass, fam in sorted(fams.items()):
            others = {
                k: v
                for k, v in (
                    ("mismatched", fam.mismatched),
                    ("unimplemented", fam.unimplemented),
                    ("auth_failed", fam.auth_failures),
                    ("errors", fam.errors),
                )
                if v
            }
            lines.append(
                f"  {op_type} ({klass or 'none'}): {fam.exact}/{fam.attempted}"
                + (f" {others}" if others else "")
                + (" includes_full_op" if fam.includes_full_op else "")
            )
    statuses = r.report.statuses()
    lines.append(f"VU statuses: {statuses}")
    rejected = r.report.rejected()
    if rejected:
        lines.append("first non-exact VUs:")
        for vu in rejected[:5]:
            lines.append(f"  {vu.status} {vu.canonical_id} ({vu.relation_id}): {vu.detail}")
            if vu.first_divergence:
                lines.append(f"    first divergence: {vu.first_divergence}")
    lines.append("")
    lines.append(
        f"adjacent VUs: {'ok' if r.adjacent['ok'] else 'NOT ok'} {r.adjacent['pairs'][:2]}"
    )
    lines.append("negatives:")
    for n in r.negatives:
        flag = "" if n.rejected == n.attempted and n.attempted else "  <-- LOOK"
        lines.append(
            f"  {n.negative_id}: {n.rejected}/{n.attempted} rejected, "
            f"{n.unexpectedly_accepted} accepted, {n.inconclusive} inconclusive{flag}"
        )
    if r.structural_problems:
        lines.append("structural problems:")
        lines += [f"  {p}" for p in r.structural_problems[:10]]
    lines.append("")
    torch_v = r.checkers.get("torch")
    lines.append(
        f"torch: {torch_v or 'not importable'}; "
        f"torch relations: {r.checkers.get('torch_relations')}"
    )
    for spec, info in r.checkers.get("plugins", {}).items():
        lines.append(f"plugin {spec}: {info}")
    unimpl = [x["relation_id"] for x in r.checkers.get("registry", []) if not x.get("implemented")]
    lines.append(f"unimplemented relation slots: {unimpl}")
    lines.append(f"seconds: {{{', '.join(f'{k}: {v:.2f}' for k, v in r.seconds.items())}}}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# --- CLI -----------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m verity_vllm.verify",
        description="Fresh-process verifier for a verity-vllm bundle (contract 15.1).",
    )
    p.add_argument("bundle", type=Path)
    p.add_argument("--seed", type=int, required=True, help="challenge seed (mixed with the roots)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, help="VUs to check (stratified)")
    g.add_argument("--exhaustive", action="store_true", help="check every committed VU")
    p.add_argument(
        "--out", type=Path, default=None, help="output directory (default <bundle>/verify)"
    )
    p.add_argument(
        "--case", default=None, help="case id for the rows (default manifest.case or dir name)"
    )
    p.add_argument("--worker", default=os.environ.get("VERITY_WORKER", "verify"))
    p.add_argument("--adapter-revision", default=None)
    p.add_argument("--experiment-id", default=None)
    p.add_argument(
        "--production-record",
        type=Path,
        default=None,
        help="production.jsonl to compare boundaries against",
    )
    p.add_argument(
        "--negatives", default=",".join(NEGATIVE_IDS), help="comma-separated ids, or 'none'"
    )
    p.add_argument("--torch", default="auto", choices=("auto", "never", "require"))
    p.add_argument(
        "--plugin", action="append", default=None, help="module[:function] registering checkers"
    )
    p.add_argument("--no-default-plugins", action="store_true")
    p.add_argument("--no-audit", action="store_true", help="skip recomputing every leaf/root")
    p.add_argument("--request-id", action="append", default=None)
    p.add_argument("--op-type", action="append", default=None)
    p.add_argument("--production-changes", default=None)
    p.add_argument("--replay-restriction", action="append", default=None)
    p.add_argument(
        "--bound-s",
        type=float,
        default=None,
        help="also draw the Bound-compatible plan: independent per-VU selection at this rate",
    )
    p.add_argument(
        "--bound-q",
        type=float,
        default=None,
        help="first-stage rate for --bound-s: independent per-request selection (default 1.0)",
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="also write fixtures/results/<experiment_id>/",
    )
    args = p.parse_args(argv)

    plugins = list(args.plugin or [])
    if not args.no_default_plugins:
        plugins = list(DEFAULT_PLUGINS) + [x for x in plugins if x not in DEFAULT_PLUGINS]
    negatives = (
        None
        if args.negatives.lower() == "none"
        else [x.strip() for x in args.negatives.split(",") if x.strip()]
    )
    try:
        result = verify(
            args.bundle,
            seed=args.seed,
            sample_size=args.sample,
            exhaustive=args.exhaustive,
            out_dir=args.out,
            case=args.case,
            worker=args.worker,
            adapter_revision=args.adapter_revision,
            experiment_id=args.experiment_id,
            production_record=args.production_record,
            negatives=negatives,
            audit=not args.no_audit,
            torch_mode=args.torch,
            plugins=plugins,
            request_ids=args.request_id,
            op_types=args.op_type,
            production_changes=args.production_changes,
            replay_restrictions=args.replay_restriction,
            results_root=args.results_root,
            bound_q=args.bound_q,
            bound_s=args.bound_s,
        )
    except (BundleError, relations.ChallengeBindingError, ValueError) as error:
        print(f"verify: cannot verify: {error}", file=sys.stderr)
        return 2
    sys.stdout.write((result.out_dir / "summary.txt").read_text(encoding="utf-8"))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
