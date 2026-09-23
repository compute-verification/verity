"""(g) The verdict of record as typed checks (`verdict.json`, vllm-verdict/v1; `verity_capture.commit.verdict`) states the fixture's
class: `outcome == PASS` iff `class = "GREEN"`, `compat.commit_pass` == the v1 `commit/verdict.json.commit_pass`, and every check
GROUP whose v1 facts the record carries is graded the way those facts say -- lifted here INDEPENDENTLY of verdict.py, straight from
runs.jsonl / commit/verdict.json / manifest_verify.json (`program_source_identity` from `extra.identity_check.ok`, `checkpoint_binding`
from `weights_pin.result`, `boundary_linkage` from `boundary_linkage.result`, `local_replay` from `c2_sampled_replay.ok`,
`required_value_coverage` from `binding.coverage.ok` / `coverage.ok` / `manifest_verify.ok`, `runtime_match` from `tokens_equal_ref`).
On #57 (the negative) the X-03 family must FAIL with the named difference `fa2_layers_bad` (`program_source_identity/identity_layout`).

`actual` READS `<row>/verdict.json` when the row (or the v2 candidate) has one and otherwise REBUILDS it in memory from the row's
Commit evidence (`verdict.from_record`, never written into the row dir).  CPU, ms per row; the v1 reference rows carry no verdict.json,
so on them this check IS the retroactive gate of Phase 5b (old facts -> new record with no loss of decision).
"""
from __future__ import annotations

import json

from .. import resolver

NAME = "verdict"
TIER = "T0"
STRENGTH = "strong"
INPUTS = ("verdict.json | commit/verdict.json + commit/runs.jsonl + commit_summary.json + commit/manifest_verify.json", "fixtures.toml class")

PASS, FAIL = "PASS", "FAIL"


def applies(row: resolver.Row) -> str | None:
    if row.candidate("verdict.json") is not None or row.has("verdict.json") or row.has("commit/verdict.json"):
        return None
    if not (row.has("stages.txt") or row.has("build_summary.json") or row.has("commit_summary.json")):
        return "row not reachable here (no rows root / live dir / store replica)"
    return "no Commit verdict on this row (Commit never closed): nothing to grade"


def _treatment(row: resolver.Row) -> list[dict]:
    if not row.has("commit/runs.jsonl"):
        return []
    return [r for r in row.jsonl("commit/runs.jsonl") if r.get("arm") != "control" and isinstance(r.get("commit"), dict)]


def _tri(vals: list) -> str | None:
    """FAIL if any recorded False, PASS if all recorded True, None when the record does not carry the fact."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return FAIL if any(v is False for v in vals) else PASS


def expected(row: resolver.Row) -> dict:
    exp: dict = {"class": row.cls, "pass": row.cls == "GREEN"}
    v1 = row.json("commit/verdict.json") if row.has("commit/verdict.json") else None
    if v1 is not None and v1.get("commit_pass") is not None:
        exp["commit_pass"] = bool(v1["commit_pass"])
    trt = _treatment(row)
    if trt:
        cm = [r["commit"] for r in trt]
        ex = [(c.get("extra") or {}) for c in cm]
        groups = {
            "program_source_identity": _tri([(e.get("identity_check") or {}).get("ok") for e in ex]),
            "checkpoint_binding": _tri([(c.get("weights_pin") or {}).get("result") for c in cm]),
            "boundary_linkage": _tri([(c.get("boundary_linkage") or {}).get("result") for c in cm]),
            "local_replay": _tri([(c.get("c2_sampled_replay") or {}).get("ok") for c in cm]),
            "runtime_match": _tri([r.get("tokens_equal_ref", r.get("tokens_equal")) for r in trt]),
        }
        cov = [((c.get("binding") or {}).get("coverage") or {}).get("ok") for c in cm] + [(e.get("coverage") or c.get("coverage") or {}).get("ok") for e, c in zip(ex, cm)]
        if row.has("commit/manifest_verify.json"):
            cov.append(row.json("commit/manifest_verify.json").get("ok"))
        groups["required_value_coverage"] = _tri(cov)
        exp["groups"] = {k: v for k, v in groups.items() if v is not None}
        # the violation kinds the X-03 identity check recorded (native_host.identity_check: a step's dict of per-kind findings; `sampler_ids`
        # is the OBSERVED count and is a difference only when it is not the manifest's; `*_n` are counters)
        bad: set[str] = set()
        for e in ex:
            idc = e.get("identity_check") or {}
            want_sampler = (idc.get("manifest") or {}).get("sampler_ids")
            for v in idc.get("violations") or []:
                if not isinstance(v, dict):
                    continue
                bad |= {k for k, x in v.items() if k not in ("step", "sampler_ids") and not k.endswith("_n") and x}
                if want_sampler is not None and v.get("sampler_ids") is not None and v.get("sampler_ids") != want_sampler:
                    bad.add("sampler_ids")
        if bad:
            exp["identity_layout_differences"] = sorted(bad)          # #57: ["fa2_layers_bad"]
    return exp


def _lift(v: dict, cls: str | None) -> dict:
    checks = v.get("checks") or []
    groups = {c["name"]: c["outcome"] for c in checks if c.get("rule") is None and "/" not in c["name"]}
    il = next((c for c in checks if c.get("name") == "program_source_identity/identity_layout"), None)
    out = {"class": cls, "pass": v.get("outcome") == PASS, "outcome": v.get("outcome"), "schema": v.get("schema"),
           "commit_pass": (v.get("compat") or {}).get("commit_pass"), "groups": groups,
           "failing": sorted(c["name"] for c in checks if c.get("rule") is not None and c["outcome"] == FAIL),
           "insufficient": sorted(c["name"] for c in checks if c.get("rule") is not None and c["outcome"] == "INSUFFICIENT_EVIDENCE"),
           "program_digest": v.get("program_digest"), "manifest_digest": v.get("manifest_digest"), "query_id": v.get("query_id")}
    if il is not None:
        out["identity_layout_differences"] = sorted({d.get("name") for d in (il.get("differences") or []) if d.get("name")})
    return out


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("verdict.json")
    if cand is not None:
        return dict(_lift(json.load(open(cand)), row.cls), source="candidate")
    if row.has("verdict.json"):
        return dict(_lift(row.json("verdict.json"), row.cls), source="recorded")
    from verity_capture.commit.verdict import from_record
    row_dir = row.path("commit/verdict.json", verify=False).parent.parent
    return dict(_lift(from_record(row_dir).to_dict(), row.cls), source="reconstructed")
