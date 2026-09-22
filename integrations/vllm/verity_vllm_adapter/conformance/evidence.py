"""Evidence discovery and the Level 1/3/4 checks over existing artefacts (`--from-results`).

Two on-disk layouts are understood, both produced by the existing modules:

* pod layout (`pod/run_case.sh`): `<R>/production/`, `<R>/ref1/`, `<R>/ref2/`, `<R>/commit/` (the bundle),
  `<R>/compare.json`, `<R>/verify-<challenge>/verify.json` (+ `table1.jsonl` there);
* collected layout (`pod/collect.sh`, what `fixtures/results/<experiment>/` holds): `production/`, `ref1/`,
  `ref2/`, `bundle/` (+ `bundle.json` digests), `compare.json`, `verify.json`, `table1.jsonl`, optional
  `audit.json`, optional `production-r2/`.

A commit store alone (TA1's `results/TA1-typed-b1-20260907/B1/`: `manifest.json`, `replay.json`,
`commitments.json`, `correspondence/*.json`) is also accepted; it carries Level 2 evidence and the replay half
of Level 1 only.

Every check here is a *pointer check plus the recomputation the CPU can do*: manifests are matched to the port
field by field, profile ids are recomputed from the bundle manifest (`profile.profile_id_of`), bundle file
digests are re-hashed against `bundle.json`, Table 1 verdicts are recomputed with the §10 rule
(`verity_vllm.rows.verdict`), proof records are matched to the obligation's statement digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .port import Port, repo_root

PID_RE = re.compile(r"(?:^|/)p=([0-9a-f]{12})/")


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | None) -> str | None:
    """Repository-relative pointer when possible (evidence pointers must survive a checkout move)."""
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------------------------------------


@dataclass
class ResultsDir:
    path: Path
    layout: str  # "collected" | "pod" | "commit-store"
    commit: Path | None = None  # the bundle / commit store directory
    manifest: dict | None = None
    profile_id: str | None = None  # recomputed from the bundle manifest
    productions: list[Path] = field(default_factory=list)
    references: list[Path] = field(default_factory=list)
    rank: int | None = None  # per-rank capture of a TP port (`commit/rank<r>/`, `--rank-tag`)
    ranks: list[int] | None = None  # a joined per-rank store (lane TP: `commit/` = tp_merge of `commit/rank<r>/`)
    compare: Path | None = None
    compare_link: str | None = None  # a dangling `compare.json` symlink's target (a re-verification row collected without its pod's compare)
    verify: Path | None = None
    table1: Path | None = None
    table2: Path | None = None
    bundle_json: Path | None = None
    audit: Path | None = None
    correspondence: list[Path] = field(default_factory=list)
    manifest_violations: list[str] = field(default_factory=list)

    @property
    def matches_port(self) -> bool:
        return self.manifest is not None and not self.manifest_violations

    def as_json(self) -> dict[str, Any]:
        return {"path": rel(self.path), "layout": self.layout, "rank": self.rank, "ranks": self.ranks, "commit": rel(self.commit), "profile_id": self.profile_id,
                "productions": [rel(p) for p in self.productions], "references": [rel(p) for p in self.references],
                "compare": rel(self.compare), "verify": rel(self.verify), "table1": rel(self.table1),
                "bundle_json": rel(self.bundle_json), "audit": rel(self.audit),
                "correspondence": [rel(p) for p in self.correspondence],
                "matches_port": self.matches_port, "manifest_violations": self.manifest_violations}


def discover_all(path: Path, port: Port) -> list[ResultsDir]:
    """`discover`, expanded: a pod row whose replay ran once per rank (`run_case.sh --rank-tag r`: `commit/rank<r>/`,
    `ref1/rank<r>/`, `compare.rank<r>.json`, `verify-*/rank<r>/`) yields one ResultsDir per rank, sharing the
    production runs; anything else yields the single directory."""
    path = Path(path)
    ranks = sorted(path.glob("commit/rank*/manifest.json"))
    if not ranks:
        return [discover(path, port)]
    if (path / "commit" / "commitments.json").exists():
        # lane TP on hardware: the model lives in vLLM's worker processes, so the capture runs inside every worker
        # (tp_worker.py) and the ranks' streams are joined into ONE store (tp_merge: `commit/` with one
        # commitments.json over both ranks' `s=<r>` leaves; `commit/rank<r>/` are the raw per-rank streams it was
        # joined from). One ResultsDir; the ranks are recorded on it.
        rd = discover(path, port)
        rd.ranks = [int(mf.parent.name[len("rank"):]) for mf in ranks]
        return [rd]
    out: list[ResultsDir] = []
    for mf in ranks:
        rdir = mf.parent
        r = int(rdir.name[len("rank"):])
        rd = discover(path, port)  # production runs, audit, bundle.json of the row
        rd.layout, rd.rank, rd.commit = "pod", r, rdir
        rd.manifest = _json(mf)
        m = dict(rd.manifest)
        from verity_vllm_adapter.profile import profile_id_of  # noqa: PLC0415

        rd.profile_id = m.pop("profile_id", None) or profile_id_of(m)
        rd.manifest_violations = port.check_manifest(rd.manifest)
        rd.references = [p for p in (path / "ref1" / rdir.name, path / "ref2" / rdir.name, path / "ref3" / rdir.name) if (p / "replay.json").exists()]
        rd.compare = path / f"compare.{rdir.name}.json" if (path / f"compare.{rdir.name}.json").exists() else None
        rd.compare_link = None
        rd.verify = None
        for v in sorted(path.glob(f"verify-*/{rdir.name}/verify.json")):
            rd.verify = v
        rd.table1 = rd.verify.parent / "table1.jsonl" if rd.verify is not None and (rd.verify.parent / "table1.jsonl").exists() else None
        rd.table2 = rd.verify.parent / "table2.jsonl" if rd.verify is not None and (rd.verify.parent / "table2.jsonl").exists() else None
        rd.correspondence = sorted(rdir.glob("correspondence/*.json"))
        out.append(rd)
    return out


def discover(path: Path, port: Port) -> ResultsDir:
    """Classify one results directory / bundle / commit store and match its profile manifest to the port."""
    from verity_vllm_adapter.profile import profile_id_of  # noqa: PLC0415

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a directory")
    rd = ResultsDir(path, "collected")
    if (path / "bundle").is_dir():
        rd.commit = path / "bundle"
    elif (path / "commit").is_dir():
        rd.commit, rd.layout = path / "commit", "pod"
    elif (path / "commitments.json").exists() and (path / "manifest.json").exists():
        rd.commit, rd.layout = path, "commit-store"
    if rd.commit is not None and (rd.commit / "manifest.json").exists():
        rd.manifest = _json(rd.commit / "manifest.json")
    elif (path / "manifest.json").exists():
        rd.manifest = _json(path / "manifest.json")
    elif (path / "production" / "manifest.json").exists():
        rd.manifest = _json(path / "production" / "manifest.json")
    if rd.manifest is not None:
        m = dict(rd.manifest)
        rd.profile_id = m.pop("profile_id", None) or profile_id_of(m)
        rd.manifest_violations = port.check_manifest(rd.manifest)
    for name in ("production", "production-r2", "production-r3"):
        if (path / name / "production.jsonl").exists():
            rd.productions.append(path / name)
    for name in ("ref1", "ref2", "ref3"):
        if (path / name / "replay.json").exists():
            rd.references.append(path / name)
    if (path / "compare.json").exists():
        rd.compare = path / "compare.json"
    elif (path / "compare.json").is_symlink():
        rd.compare_link = os.readlink(path / "compare.json")
    if (path / "verify.json").exists():
        rd.verify = path / "verify.json"
    else:
        for v in sorted(path.glob("verify-*/verify.json")):
            rd.verify = v
    if (path / "table1.jsonl").exists():
        rd.table1 = path / "table1.jsonl"
    elif rd.verify is not None and (rd.verify.parent / "table1.jsonl").exists():
        rd.table1 = rd.verify.parent / "table1.jsonl"
    if (path / "table2.jsonl").exists():
        rd.table2 = path / "table2.jsonl"
    elif rd.table1 is not None and (rd.table1.parent / "table2.jsonl").exists():
        rd.table2 = rd.table1.parent / "table2.jsonl"
    if (path / "bundle.json").exists():
        rd.bundle_json = path / "bundle.json"
    if (path / "audit.json").exists():
        rd.audit = path / "audit.json"
    for c in sorted(path.glob("correspondence/*.json")) + sorted((rd.commit or path).glob("correspondence/*.json")):
        if c not in rd.correspondence:
            rd.correspondence.append(c)
    return rd


# ---------------------------------------------------------------------------------------------------------
# Level 1: production reproducibility and replay reproducibility
# ---------------------------------------------------------------------------------------------------------


def _production_rows(pdir: Path) -> dict[str, dict]:
    rows = {}
    for r in _jsonl(pdir / "production.jsonl"):
        rows[r["request_id"]] = r
    return rows


def production_repro(dirs: list[ResultsDir]) -> dict[str, Any]:
    """Compare every pair of production runs of the port (tokens and chosen logprobs per request), and record
    each run's own in-process second pass (`run.json.stock_vs_stock_identical`)."""
    runs: list[dict[str, Any]] = []
    for rd in dirs:
        if not rd.matches_port:
            continue
        for p in rd.productions:
            run = _json(p / "run.json") if (p / "run.json").exists() else {}
            rows = _production_rows(p)
            runs.append({"path": rel(p), "profile_id": run.get("profile_id"), "requests": len(rows),
                         "generated_positions": sum(len(r["output_token_ids"]) for r in rows.values()),
                         "in_process_second_pass_identical": run.get("stock_vs_stock_identical"),
                         "cached_prefix_pass_identical": run.get("cached_prefix_pass_identical"), "_rows": rows})
    pairs = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            a, b = runs[i]["_rows"], runs[j]["_rows"]
            shared = sorted(set(a) & set(b))
            tok = sum(1 for r in shared if a[r]["output_token_ids"] == b[r]["output_token_ids"])
            lp = sum(1 for r in shared if a[r].get("chosen_logprobs") == b[r].get("chosen_logprobs"))
            pairs.append({"a": runs[i]["path"], "b": runs[j]["path"], "requests_shared": len(shared),
                          "tokens_identical": tok, "logprobs_identical": lp,
                          "identical": bool(shared) and tok == len(shared) and lp == len(shared)})
    for r in runs:
        r.pop("_rows")
    cross = [p for p in pairs if p["requests_shared"]]
    if cross:
        status = "reproducible" if all(p["identical"] for p in cross) else "not-reproducible"
        method = "independent production processes compared per request (tokens and chosen logprobs)"
    elif runs and all(r["in_process_second_pass_identical"] for r in runs):
        status = "reproducible"
        method = "one production process; its second stock pass over the workload is token-identical (run.json.stock_vs_stock_identical)"
    elif runs:
        status = "not-reproducible" if any(r["in_process_second_pass_identical"] is False for r in runs) else "insufficient"
        method = "run.json.stock_vs_stock_identical"
    else:
        status, method = "absent", None
    return {"status": status, "method": method, "runs": runs, "pairs": pairs}


def _commit_replay(rd: ResultsDir) -> dict | None:
    if rd.commit is not None and (rd.commit / "replay.json").exists():
        return _json(rd.commit / "replay.json")
    return None


def _replay_exec(rp: dict) -> str:
    """The replay's execution mode as `replay.py --replay-exec` names it: recorded directly, or derived — W1d's
    `compiled_fx` module records its `instrumentation_mode` instead, and older eager replays only the effective
    execution."""
    if rp.get("replay_exec"):
        return str(rp["replay_exec"])
    if str(rp.get("instrumentation_mode", "")).startswith("compiled-fx") or rp.get("adapter_module") == "verity_vllm_adapter.compiled_fx":
        return "compiled-fx"
    eff = rp.get("effective_execution") or {}
    if eff.get("enforce_eager", True):
        return "eager"
    return "compiled-nograph" if str(eff.get("cudagraph_mode", "")).upper() == "NONE" else "compiled"


def replay_repro(dirs: list[ResultsDir]) -> dict[str, Any]:
    """Replay followed the recorded trajectory, reproduced the production logprobs, agreed with the unforced
    sampler, and the commit capture is bit-identical to the fresh-process references (`compare.json`)."""
    replays = []
    for rd in dirs:
        if not rd.matches_port:
            continue
        rp = _commit_replay(rd)
        if rp is None:
            continue
        problems: list[str] = []
        tf = rp.get("trajectory_followed") or {}
        if tf.get("followed") != tf.get("requests"):
            problems.append(f"trajectory_followed {tf}")
        lv = rp.get("logprobs_vs_production") or {}
        if lv and lv.get("equal") != lv.get("requests_compared"):
            problems.append(f"logprobs_vs_production {lv}")
        ua = rp.get("unforced_sampler_agreement") or {}
        if ua and ua.get("agree") != ua.get("positions"):
            problems.append(f"unforced_sampler_agreement {ua}")
        if rp.get("boundary_problems"):
            problems.append(f"boundary_problems {len(rp['boundary_problems'])}")
        missing: list[str] = []  # evidence absent from this row (not evidence against it)
        cmp_doc = _json(rd.compare) if rd.compare else None
        cmp_summary = None
        if cmp_doc is not None:
            cmp_summary = {"identical": cmp_doc.get("identical"), "compared_scalars": cmp_doc.get("compared_scalars"),
                           "mismatched_scalars": cmp_doc.get("mismatched_scalars"),
                           "references": len(cmp_doc.get("pairs", [])),
                           "reference_vs_reference_identical": (cmp_doc.get("reference_vs_reference") or {}).get("identical")}
            if not cmp_doc.get("identical"):
                problems.append("compare.json: commit capture differs from the references")
        elif rd.compare_link:
            missing.append(f"compare.json is a dangling link to {rd.compare_link} (the pod's comparison was not collected with this row)")
        elif rd.layout != "commit-store":
            missing.append("compare.json missing (reference-vs-commit noninterference not shown by this row)")
        cap = rp.get("capture") or {}
        for k in ("unbound_inputs", "module_outputs_not_recorded", "kv_reads_unresolved"):
            if cap.get(k):
                problems.append(f"capture.{k}={cap[k]}")
        replays.append({"results": rel(rd.path), "commit": rel(rd.commit), "profile_id": rd.profile_id,
                        "replay_exec": _replay_exec(rp), "instrumentation": rp.get("instrumentation_mode") or (rp.get("replay_restrictions") or {}).get("interception"),
                        "effective_execution": rp.get("effective_execution"), "requests": rp.get("requests"),
                        "trajectory_followed": tf, "logprobs_vs_production": lv, "unforced_sampler_agreement": ua,
                        "capture": {k: cap.get(k) for k in ("steps", "replay_units", "leaves", "bytes_captured", "unbound_inputs",
                                                            "module_outputs_not_recorded", "kv_reads_unresolved")},
                        "references": [rel(r) for r in rd.references], "compare": cmp_summary,
                        "replay_restrictions": rp.get("replay_restrictions"), "problems": problems, "missing": missing,
                        "status": "not-reproducible" if problems else ("incomplete" if missing else "reproducible")})
    if not replays:
        return {"status": "absent", "replays": []}
    # the port's replay half of Level 1: at least one row shows it completely and no row contradicts it;
    # rows with absent pieces are recorded as incomplete (their absence is named), not counted as evidence either way
    if any(r["status"] == "not-reproducible" for r in replays):
        status = "not-reproducible"
    elif any(r["status"] == "reproducible" for r in replays):
        status = "reproducible"
    else:
        status = "incomplete"
    return {"status": status, "replays": replays,
            "rows": {k: sum(r["status"] == k for r in replays) for k in ("reproducible", "incomplete", "not-reproducible")}}


# ---------------------------------------------------------------------------------------------------------
# Level 3: protocol bundle pointer check
# ---------------------------------------------------------------------------------------------------------


def bundle_integrity(rd: ResultsDir) -> dict[str, Any] | None:
    """Re-hash the bundle files `bundle.json` lists and that are present locally."""
    if rd.bundle_json is None or rd.commit is None:
        return None
    doc = _json(rd.bundle_json)
    checked, ok, missing, bad = 0, 0, [], []
    for name, rec in (doc.get("files") or {}).items():
        local = None
        if name.startswith("commit/"):
            local = rd.commit / name[len("commit/"):]
        elif name.startswith(("ref1/", "ref2/")):
            local = rd.path / name
        elif "/" not in name:
            local = rd.path / name
        if local is None or not local.exists():
            missing.append(name)
            continue
        checked += 1
        if sha256_file(local) == rec.get("sha256"):
            ok += 1
        else:
            bad.append(name)
    return {"bundle_json": rel(rd.bundle_json), "profile_id": doc.get("profile_id"), "experiment_id": doc.get("experiment_id"),
            "files_listed": len(doc.get("files") or {}), "files_checked": checked, "files_ok": ok, "files_bad": bad,
            "files_not_local": missing, "ok": checked > 0 and not bad}


def table1_rows(rd: ResultsDir) -> list[dict[str, Any]]:
    """Table-1 rows of this results dir with the §10 verdict recomputed (`verity_vllm.rows.verdict`)."""
    if rd.table1 is None:
        return []
    from verity_vllm import rows as R  # noqa: PLC0415

    t2 = _jsonl(rd.table2) if rd.table2 else []
    out = []
    for row in _jsonl(rd.table1):
        rec = {"table1": rel(rd.table1), "experiment_id": row.get("experiment_id"), "case": row.get("case"),
               "profile_id": row.get("profile_id"), "result": row.get("result"), "first_blocker": row.get("first_blocker"),
               "challenge_mode": (row.get("challenge") or {}).get("mode") if isinstance(row.get("challenge"), dict) else None,
               "verification_evidence": row.get("verification_evidence")}
        problems = R.validate_row(1, row, check_verdict=False)
        rec["row_problems"] = problems
        if not problems:
            try:
                v = R.verdict(row, t2)
                rec["recomputed_result"] = v.result
                rec["recomputed_first_blocker"] = v.first_blocker
                rec["overclaims"] = R.overclaims(row.get("result"), v.result)
            except R.RowError as e:
                rec["row_problems"] = [str(e)]
        out.append(rec)
    return out


def verify_summary(rd: ResultsDir) -> dict[str, Any] | None:
    if rd.verify is None:
        return None
    v = _json(rd.verify)
    rep = v.get("report") or {}
    statuses = rep.get("statuses") or {}
    negs = v.get("negatives") or []
    n_att = sum(int(n.get("attempted", 0)) for n in negs)
    n_rej = sum(int(n.get("rejected", 0)) for n in negs)
    auth = rep.get("authentication") or {}
    return {"verify": rel(rd.verify), "profile_id": v.get("profile_id"), "experiment_id": v.get("experiment_id"),
            "challenge_digest": v.get("challenge_digest"), "sample_size": v.get("sample_size"),
            "authentication": auth, "statuses": statuses, "relation_checks": rep.get("relation_checks"),
            "negatives": {"attempted": n_att, "rejected": n_rej},
            "all_exact": bool(statuses) and set(statuses) == {"exact"},
            "authentication_complete": bool(auth) and auth.get("valid") == auth.get("attempted"),
            "negatives_all_rejected": n_att > 0 and n_att == n_rej}


def level3_evidence(dirs: list[ResultsDir], port: Port) -> dict[str, Any]:
    """A Table-1 row whose recomputed §10 verdict is PASS, the verifier's report over the same bundle (same
    profile id and challenge digest), and the bundle's file digests re-hashed — all for a manifest that matches
    the port."""
    bundles = []
    for rd in dirs:
        if not rd.matches_port or rd.commit is None:
            continue
        rows = table1_rows(rd)
        ver = verify_summary(rd)
        integ = bundle_integrity(rd)
        problems: list[str] = []
        pass_rows = [r for r in rows if r.get("recomputed_result") == "PASS" and not r.get("overclaims") and not r["row_problems"]]
        if not rows:
            problems.append("no table1.jsonl row")
        elif not pass_rows:
            problems.append("no Table-1 row whose recomputed §10 verdict is PASS: " + "; ".join(
                f"{r['result']} (rule: {r.get('recomputed_result')}, {r.get('recomputed_first_blocker')})" for r in rows))
        for r in rows:
            if r["profile_id"] != rd.profile_id:
                problems.append(f"table1 profile_id {r['profile_id']} != bundle manifest {rd.profile_id}")
        if ver is None:
            problems.append("no verify.json (the verifier's report over the bundle)")
        else:
            if ver["profile_id"] != rd.profile_id:
                problems.append(f"verify.json profile_id {ver['profile_id']} != bundle manifest {rd.profile_id}")
            if not ver["all_exact"]:
                problems.append(f"verify.json statuses {ver['statuses']}")
            if not ver["authentication_complete"]:
                problems.append(f"verify.json authentication {ver['authentication']}")
            if not ver["negatives_all_rejected"]:
                problems.append(f"verify.json negatives {ver['negatives']}")
            for r in pass_rows:
                if ver.get("challenge_digest") and _row_challenge_digest(rd, r) not in (None, ver["challenge_digest"]):
                    problems.append("table1 challenge digest != verify.json challenge digest")
        if integ is not None and not integ["ok"]:
            problems.append(f"bundle.json digests: bad {integ['files_bad']}")
        audit = _json(rd.audit) if rd.audit else None
        bundles.append({"results": rel(rd.path), "commit": rel(rd.commit), "profile_id": rd.profile_id,
                        "profile_id_in_port": rd.profile_id in port.config.known_profile_ids,
                        "table1_rows": rows, "verify": ver, "bundle_integrity": integ,
                        "audit": {"path": rel(rd.audit), "result": audit.get("result"), "worker": audit.get("worker")} if audit else None,
                        "problems": problems, "status": "verified" if not problems else "not-verified"})
    if not bundles:
        return {"status": "absent", "bundles": []}
    return {"status": "verified" if any(b["status"] == "verified" for b in bundles) else "not-verified", "bundles": bundles}


def _row_challenge_digest(rd: ResultsDir, row: dict) -> str | None:
    if rd.table1 is None:
        return None
    for r in _jsonl(rd.table1):
        if r.get("experiment_id") == row.get("experiment_id"):
            ch = r.get("challenge")
            if isinstance(ch, dict):
                return ch.get("digest")
    return None


# ---------------------------------------------------------------------------------------------------------
# Level 4: proof-backed obligation pointer check
# ---------------------------------------------------------------------------------------------------------


def level4_evidence(obligations: list[Path], proof_records: list[Path], dirs: list[ResultsDir]) -> dict[str, Any]:
    """An exported typed obligation whose provenance names a capture (`p=<profile id>`) of a bundle that
    matches the port, and a proof record whose host verification accepted a statement with the obligation's
    statement digest."""
    by_pid = {rd.profile_id: rd for rd in dirs if rd.matches_port and rd.profile_id}
    records = []
    for ob_path in obligations:
        ob = _json(ob_path)
        prov = ob.get("provenance") or {}
        cid = prov.get("canonical_id") or ""
        m = PID_RE.search(cid)
        pid = m.group(1) if m else None
        problems: list[str] = []
        if pid is None:
            problems.append("obligation provenance has no capture canonical id")
        elif pid not in by_pid:
            problems.append(f"obligation derives from profile {pid}, which is not a bundle of this port among the supplied results")
        if ob.get("verdict") != "accept" or ob.get("witness_verdict") not in (None, "accept"):
            problems.append(f"obligation verdict {ob.get('verdict')!r} / witness {ob.get('witness_verdict')!r}")
        fixture_files = {}
        fx = prov.get("fixture", "")
        fx_dir = repo_root() / fx.split(" ")[0] if fx else None
        for name, digest in (prov.get("files_sha256") or {}).items():
            local = fx_dir / name if fx_dir else None
            if local is not None and local.exists():
                fixture_files[name] = "ok" if sha256_file(local) == digest else "sha256-mismatch"
            else:
                fixture_files[name] = "not-local"
        if any(v == "sha256-mismatch" for v in fixture_files.values()):
            problems.append("fixture files differ from provenance.files_sha256")
        proofs = []
        for pr in proof_records:
            for vf in sorted(Path(pr).glob("verify*.json")):
                v = _json(vf)
                match = (v.get("statement_digest") == ob.get("statement_digest") and bool(v.get("ok")) and bool(v.get("verdict"))
                         and v.get("statement_match") is not False and not v.get("unsound"))
                proofs.append({"record": rel(vf), "backend": v.get("backend"), "vk_hash": v.get("vk_hash"),
                               "statement_digest": v.get("statement_digest"), "typed_object_digest": v.get("typed_object_digest"),
                               "ok": v.get("ok"), "verdict": v.get("verdict"), "statement_match": v.get("statement_match"),
                               "matches_obligation": match})
        good = [p for p in proofs if p["matches_obligation"]]
        if not good:
            problems.append("no proof record whose accepted statement digest equals the obligation's")
        if ob.get("digest") and good and any(p["typed_object_digest"] not in (None, ob["digest"]) for p in good):
            problems.append("proof record typed_object_digest != obligation digest")
        records.append({"obligation": rel(ob_path), "name": ob.get("name"), "canonical_id": cid, "profile_id": pid,
                        "results": rel(by_pid[pid].path) if pid in by_pid else None, "statement_digest": ob.get("statement_digest"),
                        "object_digest": ob.get("digest"), "verdict": ob.get("verdict"), "witness_verdict": ob.get("witness_verdict"),
                        "fixture": fx, "fixture_files": fixture_files, "proofs": proofs, "problems": problems,
                        "status": "proved" if not problems else "not-proved"})
    if not records:
        return {"status": "absent", "records": []}
    return {"status": "proved" if any(r["status"] == "proved" for r in records) else "not-proved", "records": records}
