"""(T2) The Program digest of record: the chain of digests the row's documents carry must be ONE digest, and a rebuild must reproduce it.

CPU part (this check, ms): `build_summary.json` (request/step/per-shape `program_digest`; `workload.workload_digest` at B>=2) ==
`build_request*/artifact.json.program_digest` (the Build's own artifacts) == `build_workload/workload_program.json.workload_digest` ==
`commit/manifest_verify.json.used_program_digest` (what the Commit keyed its binding map by).  `source_identity.json` records the SOURCE
of record (the git sha of the tree that ran each stage) beside them -- lifted, not asserted: the Build of #23 ran at 933cb05, its Commit
at 89af272 (the `v1-reference` commit); a v2 reproduces the DIGESTS, not the sha.

Rebuild part (NOT run here; tier T2, 1-4 h and 18-181 GB RAM per row on the meta device, no GPU needed): on a big-RAM pod,
`STAGES=build bash verity_capture/sweep/row_pod.sh <row> <role> <hf repo> <rev> build` with `--target` = the recorded
`build_request/artifact.json.target_profile`, then compare `build_summary.json` digests to this expected document.  In candidate mode the
v2's `build_summary.json` is read instead and compared to the same expected.
"""
from __future__ import annotations

import glob
import json
import os

from .. import resolver

NAME = "program_digest"
TIER = "T2"
STRENGTH = "strong"
INPUTS = ("build_summary.json", "build_request*/artifact.json", "build_workload/workload_program.json", "commit/manifest_verify.json", "source_identity.json")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("build_summary.json") else "no build_summary.json on this row"


def _digests_of(bs: dict) -> dict:
    out = {"request": ((bs.get("request") or {}).get("program_digest")), "step": ((bs.get("step") or {}).get("program_digest")),
           "workload": ((bs.get("workload") or {}).get("workload_digest")) if isinstance(bs.get("workload"), dict) else None,
           "shapes": {k: v.get("program_digest") for k, v in sorted(bs.items()) if k.startswith("build_request_LP") and isinstance(v, dict)}}
    return out


def expected(row: resolver.Row) -> dict:
    bs = row.json("build_summary.json")
    out = {"build_summary": _digests_of(bs)}
    b = len(json.load(open(row.workload())).get("requests") or []) if row.has("build_summary.json") and (resolver.VLLM_ROOT / "workloads" / f"{row.key}.json").is_file() else None
    of_record = out["build_summary"]["workload"] if (b or 0) > 1 else out["build_summary"]["request"]
    out["program_digest_of_record"] = of_record
    out["batch"] = b
    if row.has("commit/manifest_verify.json"):
        out["commit_keyed_by"] = row.json("commit/manifest_verify.json").get("used_program_digest")
    if row.has("build_workload/workload_program.json"):
        out["workload_program_digest"] = row.json("build_workload/workload_program.json").get("workload_digest")
    return out


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("build_summary.json")
    if cand is not None:
        bs = json.load(open(cand))
        d = _digests_of(bs)
        b = len(json.load(open(row.workload())).get("requests") or [])
        return {"build_summary": d, "program_digest_of_record": d["workload"] if b > 1 else d["request"], "batch": b, "source": "candidate"}
    # self-check: the chain read from the Build's own per-shape artifacts, independent of build_summary.json's copy
    exp = expected(row)
    arts: dict[str, str | None] = {}
    for rel in ("build_request",) + tuple(sorted(k for k in exp["build_summary"]["shapes"])):
        if row.has(f"{rel}/artifact.json"):
            arts[rel] = row.json(f"{rel}/artifact.json").get("program_digest")
    out = dict(exp, source="record-chain")
    out["artifact_json"] = arts
    out["chain_consistent"] = all(exp["build_summary"]["shapes"].get(k) == v for k, v in arts.items() if k != "build_request") and \
        (arts.get("build_request") in (None, exp["build_summary"]["request"])) and \
        (exp.get("commit_keyed_by") in (None, exp["program_digest_of_record"])) and \
        (exp.get("workload_program_digest") in (None, exp["build_summary"]["workload"]))
    if row.has("source_identity.json"):
        si = row.json("source_identity.json")
        out["source_of_record"] = {"git_sha": (si.get("requested") or {}).get("git_sha"), "stage": si.get("stage")}
    return out
