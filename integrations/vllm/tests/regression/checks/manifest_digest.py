"""(a) The manifest of record reproduces: rebuild `manifest.json` from the row's request Programs x declared workload with the CURRENT
code, by the exact path `row_pod.sh` uses (`verity-vllm manifest build-global --fixture WL --programs-root ROW --workload-digest WD`
at B>=2; `verity-vllm manifest build --program ROW/build_request --workload WL` at B=1) and compare:

    manifest_digest     sha256 over the sorted (family + IDENTITY_KEY) rows     -- `manifest_format.manifest_digest`
    program_digest      the Program digest the identities are keyed by          -- GP-01's workload digest (B>=2) / the request Program's (B=1)
    identities          the identity count
    identity_rows_sha256  sha256 over the sorted canonical IDENTITY_KEY tuples (family-blind): byte-for-byte equality of the identity SET
                        with the recorded manifest.json when that file is resolvable (222 MB for #23: pod only)

Expected is lifted from `commit/manifest_verify.json` (used_digest / used_program_digest / used_identities; the v1 Commit's own
rebuild said `ok`) and, when the recorded manifest.json is reachable, from the file itself.  STRONG: a single identity row moved,
dropped or renamed changes the digest.  TP rows (#70, #75: rank Programs under build/rank*/) are not built by this path -> not applicable.
The builder is the integration's (the query path `verity-vllm manifest`, Q_module_body_v1 -- the same build / build-global flags
row_pod.sh uses).

`actual` also states what the built manifest PROMOTED (interior rows whose output crosses a module
boundary, flagged `promoted: true`) -- the facts a labelled decision on the fixture may name (`resolver.sanction`; #57's decision):

    promoted            the count of promoted identity rows (0 on a v1-shaped manifest)
    promoted_member     the one `op_path/output_member` those rows are named as (a sorted list if several, None if none)
    promoted_member_class  the same with the layer index folded to `<k>` (`model.layers.<k>.mlp.experts/out`): one string when the
                        promoted rows are one member per layer (#67 / #68), else the sorted list / None -- what a decision names
    promoted_class_counts  on a query manifest (`query.engine` q-word or v2-query), the rows beyond v1 per member class
    only_new            how many identities the manifest has BEYOND the v1 manifest of record with none of v1's removed -- the
                        IDENTITY_KEY set difference against the row's `manifest.json` when it resolves here (None when a v1 identity is
                        missing: not an additive difference), else the manifest's own account (= promoted); `only_new_basis` says which
"""
from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from .. import resolver

NAME = "manifest_digest"
TIER = "T0"          # inputs: Programs 50-300 MB + workload; the rebuild itself is CPU-minutes (run it on a pod)
STRENGTH = "strong"
REBUILD = True       # 94% of the T0 wall (2-26 min a row): skipped unless VERITY_REGRESSION_REBUILD=1, the post-train job's switch
INPUTS = ("manifest.json", "commit/manifest_verify.json", "build_summary.json", "build_request*/instances.json.gz", "workloads/<row>.json")


def applies(row: resolver.Row) -> str | None:
    if row.spec.get("tp", 1) > 1:
        return "TP row: rank Programs (build/rank*/) are merged by tp_stage.sh, not by build/build-global"
    if not row.has("commit/manifest_verify.json") and not row.has("manifest.json"):
        return "no manifest of record on this row (Commit never built one)"
    return None


def _identity_key(r: dict) -> str:
    from verity_vllm.query.manifest.format import IDENTITY_KEY
    return json.dumps([r[k] if k != "element_range" else list(r[k]) for k in IDENTITY_KEY], separators=(",", ":"), sort_keys=True)


def identity_rows_sha256(identities: list[dict]) -> str:
    rows = sorted(_identity_key(r).encode() for r in identities)
    h = hashlib.sha256()
    for b in rows:
        h.update(b)
        h.update(b"\n")
    return h.hexdigest()


def _summary(man: dict, *, with_rows: bool) -> dict:
    out = {"manifest_digest": man.get("manifest_digest"), "program_digest": man.get("program_digest"),
           "identities": (man.get("populations") or {}).get("identities", len(man.get("identities") or [])),
           "complete": man.get("complete"), "schema": man.get("schema"), "unmodelled": man.get("unmodelled") or {}}
    if with_rows:
        out["identity_rows_sha256"] = identity_rows_sha256(man.get("identities") or [])
    return out


def _member_class(member: str) -> str:
    """`op_path/output_member` with the layer index folded to `<k>`."""
    op, _, mem = member.rpartition("/")
    return ".".join("<k>" if p.isdigit() else p for p in op.split(".")) + "/" + mem


#: `query.engine` of a manifest the query path built (`verity_vllm.query.required`): Q_word v1 since S1, Q_module_body_v1 before
QUERY_ENGINES = ("q-word", "v2-query")


def _one_or_list(members: list[str]):
    return members[0] if len(members) == 1 else (members or None)


def _promotion(row: resolver.Row, man: dict) -> dict:
    """The promotion facts of a built manifest (see the module docstring): `promoted`, `promoted_member`,
    `promoted_member_class`, `only_new` (+ `only_new_basis`, and the set arithmetic against v1 when its manifest.json resolves)."""
    rows = man.get("identities") or []
    v2 = (man.get("query") or {}).get("engine") in QUERY_ENGINES
    promoted = [r for r in rows if r.get("promoted")]
    members = sorted({f"{r.get('op_path')}/{r.get('output_member')}" for r in promoted})
    out: dict = {"promoted": len(promoted), "promoted_member": _one_or_list(members),
                 "promoted_member_class": _one_or_list(sorted({_member_class(m) for m in members}))}
    if row.has("manifest.json"):                            # the v1 manifest of record (never the candidate's: `row.path` is the reference)
        old = {_identity_key(r) for r in row.json("manifest.json").get("identities") or []}
        new_rows = {_identity_key(r): r for r in rows}
        new = set(new_rows)
        added, removed = len(new - old), len(old - new)
        out.update(only_new=added if not removed else None, only_new_basis="IDENTITY_KEY set difference vs the v1 manifest.json",
                   v1_identities=len(old), added_n=added, removed_n=removed)
        if v2:
            # a v2 (query) manifest promotes nothing: what v1 promoted is a boundary Value here.  Its "promoted" facts are the rows it
            # names beyond v1 -- the same set difference, so a decision labelled on v1's promotion (#57: `model/out`) reads the same
            members = sorted({f"{new_rows[k].get('op_path')}/{new_rows[k].get('output_member')}" for k in new - old})
            out.update(promoted=added, promoted_member=_one_or_list(members),
                       promoted_member_class=_one_or_list(sorted({_member_class(m) for m in members})), promoted_basis="v2-query: rows beyond v1")
            out["promoted_class_counts"] = dict(sorted(collections.Counter(_member_class(f"{new_rows[k].get('op_path')}/{new_rows[k].get('output_member')}")
                                                                           for k in new - old).items()))
    else:
        out.update(only_new=len(promoted), only_new_basis="promoted rows (v1 manifest.json not resolvable here)")
    return out


def expected(row: resolver.Row) -> dict:
    out: dict = {}
    if row.has("commit/manifest_verify.json"):
        mv = row.json("commit/manifest_verify.json")
        out.update({"manifest_digest": mv.get("used_digest"), "program_digest": mv.get("used_program_digest"), "identities": mv.get("used_identities"),
                    "complete": True, "v1_verify_ok": mv.get("ok")})
    if row.has("manifest.json"):
        man = row.json("manifest.json")
        out.update(_summary(man, with_rows=True))
    out.pop("v1_verify_ok", None)
    return out


BUILDER = "verity_vllm.pipeline.cli"         # Q_module_body_v1 over the Program + correspondence (same build / build-global flags as row_pod.sh)
POLICIES_OFF = ["--no-norm-scales", "--no-router-softmax", "--no-vocab-range"]    # the manifests of record predate them


def _flags(row: resolver.Row) -> list[str]:
    """The builder flags the row's manifest of record was built with: `manifest_flags` of its fixtures.toml table (#73's epoch record:
    `[]`, every policy on), else POLICIES_OFF."""
    return list(row.spec.get("manifest_flags", POLICIES_OFF))


def _rebuild_argv(row: resolver.Row, programs_root: Path, out: Path) -> list[str]:
    wl = row.workload()
    fx = json.load(open(wl))
    b = len(fx.get("requests") or [])
    base = [sys.executable, "-m", BUILDER, "manifest"]
    if b > 1:
        bs = row.json("build_summary.json")
        wd = ((bs.get("workload") or {}).get("workload_digest")) or None
        argv = base + ["build-global", "--fixture", str(wl), "--programs-root", str(programs_root), "--out", str(out), *_flags(row)]
        if wd:
            argv += ["--workload-digest", wd]
        return argv
    return base + ["build", "--program", str(programs_root / "build_request"), "--workload", str(wl), "--out", str(out), *_flags(row)]


def actual(row: resolver.Row) -> dict:
    cand = row.candidate("manifest.json")
    if cand is not None:                                   # candidate mode: the v2 implementation's manifest.json IS the thing under test
        with open(cand) as f:
            man = json.load(f)
        return dict(_summary(man, with_rows=True), **_promotion(row, man), source="candidate")
    root = row.programs_root()
    if root is None:
        raise resolver.NotResolvable(f"{row.key}: no request Programs (build_request*/instances.json.gz) reachable here")
    scratch = resolver.scratch(row)
    out = scratch / "manifest.rebuilt.json"
    argv = _rebuild_argv(row, root, out)
    p = subprocess.run(argv, cwd=str(resolver.VLLM_ROOT), capture_output=True, text=True)
    (scratch / "manifest.rebuilt.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
    if p.returncode != 0 or not out.is_file():
        return {"rebuild_rc": p.returncode, "rebuild_argv": argv, "rebuild_tail": (p.stdout + p.stderr).strip()[-1500:], "source": "rebuild"}
    man = json.load(open(out))
    res = dict(_summary(man, with_rows=True), **_promotion(row, man), source="rebuild", rebuild_rc=0,
               rebuild_engine=(man.get("engine") or man.get("populations", {}).get("engine") or (man.get("query") or {}).get("engine") or resolver.ENGINE))
    return res
