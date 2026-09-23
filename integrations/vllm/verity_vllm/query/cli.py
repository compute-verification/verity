"""`python -m verity_vllm.query.cli` -- the v2 (Q_module_body_v1) required-value manifest, beside v1's `required_manifest`.

    build         --program <derive dir> [--workload wl.json] [--request-id r0] --out manifest.json
    build-global  --fixture wl.json --programs-root <row dir> [--workload-digest WD] --out manifest.json
    compare       --v1 manifest.json --v2 manifest.json [--out report.json]

`build` / `build-global` take the same inputs as v1's (a derive_step output dir holds `instances.json.gz` + `result.json` [+ `artifact.json`];
a row dir holds one such dir per request shape, `*LP<prompt_len>_T<max_tokens-1>`), and write a manifest of v1's schema whose identities
are the bridge's (`v1_bridge.request_manifest` / `compose_global`).  `--no-return-slots` builds the pure-dataflow population (no
RETURN-SLOT policy) for the record.  `compare` classifies every identity difference (`compare.summarize`).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any

from .compare import compare, summarize
from .correspondence import Correspondence
from .program_view import from_instances
from .v1_bridge import compose_global, request_manifest

__all__ = ["build", "build_global", "main"]


def _load_json(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def build(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, rank: int = 0, ranks: int = 1,
          profile: str | None = None, return_slots: bool = True, log=None) -> dict[str, Any]:
    t0 = time.time()
    P = from_instances(os.path.join(program_dir, "instances.json.gz"))
    res = _load_json(os.path.join(program_dir, "result.json"))
    art_p = os.path.join(program_dir, "artifact.json")
    art = _load_json(art_p) if os.path.exists(art_p) else None
    corr = Correspondence.of(P, program_dir, implementation_paths=Correspondence.implementation_paths_of(res))   # the record when the Build carries it, else v1 annotations
    r = request_manifest(P, corr, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, rank=rank, ranks=ranks,
                         profile=profile, return_slots=return_slots)
    if log:
        pop = r.manifest["populations"]
        log(f"  {os.path.basename(program_dir)}: {len(P.calls)} calls -> {pop['identities']} identities ({pop['steps']} steps, {pop['units']} units, "
            f"{pop['boundary_values']} boundary values) complete={r.manifest['complete']} correspondence={corr.source} in {time.time() - t0:.1f}s")
    return r.manifest


def _program_dirs_for(fx: dict[str, Any], programs_root: str) -> dict[str, str]:
    """v1's lookup (`required_manifest._program_dirs_for`): `*LP<prompt_len>_T<max_tokens-1>` under the root, else a `build_request*`
    whose result.json declares exactly (LP, T)."""
    progs: dict[str, str] = {}
    for r in fx["requests"]:
        lp, t = int(r["prompt_len"]), int(r["max_tokens"]) - 1
        hits = sorted(glob.glob(os.path.join(programs_root, f"*LP{lp}_T{t}")))
        if not hits:
            for cand in sorted(glob.glob(os.path.join(programs_root, "build_request*"))):
                rj = os.path.join(cand, "result.json")
                if os.path.isdir(cand) and os.path.exists(rj):
                    try:
                        res = _load_json(rj)
                    except Exception:  # noqa: BLE001
                        continue
                    if res.get("ok") and int(res.get("LP", -1)) == lp and int(res.get("T", -1)) == t:
                        hits = [cand]
                        break
        if not hits:
            raise SystemExit(f"no request Program for {r['request_id']} (LP={lp} T={t}) under {programs_root}")
        progs[str(r["request_id"])] = hits[0]
    return progs


def build_global(fixture: dict[str, Any], programs_root: str, *, workload_digest: str | None = None, rank: int = 0, ranks: int = 1,
                 profile: str | None = None, return_slots: bool = True, log=None) -> dict[str, Any]:
    declared = [dict(r, index=int(r.get("index", i))) for i, r in enumerate(fixture["requests"])]
    if len({r["index"] for r in declared}) != len(declared):
        raise ValueError("workload declaration: duplicate request index")
    reqs = []
    for r in sorted(declared, key=lambda r: (int(r.get("arrive_step", 0)), r["index"])):
        reqs.append(dict(r, workload_request_id=r.get("request_id"), request_id=f"r{r['index']}"))
    dirs = _program_dirs_for(fixture, programs_root)
    parts: dict[str, dict[str, Any]] = {}
    for r in reqs:
        rid = r["request_id"]
        d = dirs[str(r["workload_request_id"])]
        wl = {"sweep": {"arrive_steps": [int(r.get("arrive_step", 0))], "row_id": fixture.get("fixture"), "batch": len(reqs)}}
        parts[rid] = build(d, workload=wl, request_id=rid, rank=rank, ranks=ranks, profile=profile, return_slots=return_slots, log=log)
        if len(parts[rid]["steps"]) != int(r["max_tokens"]):
            raise ValueError(f"request {rid}: Program has {len(parts[rid]['steps'])} steps, the declaration caps {r['max_tokens']} tokens: wrong component")
    return compose_global(fixture, parts, reqs, workload_digest=workload_digest, rank=rank, ranks=ranks)


def check_correspondence(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, manifest: bool = True, log=None) -> dict[str, Any]:
    """`Q_module_body_v1` under the record vs under the v1-annotation fallback on ONE Build dir that carries the record: the owning
    module per Call, the Call names the bridge reads, the partition (unit ids), the boundary Values and -- `manifest` -- the bridge's
    manifest digest, each equal or listed.  A Build without the record has nothing to check against (`applicable: False`)."""
    from verity.verification.query import boundary_values

    from .module_body import Q_module_body_v1
    t0 = time.time()
    P = from_instances(os.path.join(program_dir, "instances.json.gz"))
    res = _load_json(os.path.join(program_dir, "result.json"))
    impl = Correspondence.implementation_paths_of(res)
    canon = Correspondence.of(P, program_dir, implementation_paths=impl)
    out: dict[str, Any] = {"schema": "vllm-v2/query-correspondence-check/v1", "program_dir": program_dir, "wrapper": res.get("wrapper"), "calls": len(P.calls),
                           "program_digest": (res.get("program") or {}).get("digest"), "correspondence_digest": canon.digest, "source": canon.source, "problem": canon.problem}
    if canon.source != "correspondence":
        out.update(applicable=False, why="the Build carries no runtime-correspondence/v1 record: only the fallback reads it")
        return out
    fallback = Correspondence(P, implementation_paths=impl)
    module_diffs = [{"call": c.id, "record": canon.module_of(c.id), "v1_annotations": fallback.module_of(c.id), "name": c.name} for c in P.calls if canon.module_of(c.id) != fallback.module_of(c.id)]
    name_diffs = [{"call": c.id, "record": str(canon.name(c.id)), "v1_annotations": str(fallback.name(c.id)), "name": c.name} for c in P.calls if canon.name(c.id) != fallback.name(c.id)]
    Qa, Qb = Q_module_body_v1(P, canon), Q_module_body_v1(P, fallback)
    units_a = {u.id: u.calls for u in Qa(P)}
    units_b = {u.id: u.calls for u in Qb(P)}
    sa, sb = set(boundary_values(P, Qa)), set(boundary_values(P, Qb))
    out.update(applicable=True, module_paths={"record": len(canon.modules), "v1_annotations": len(fallback.modules)}, implementation_prefix={"record": canon.prefix, "v1_annotations": fallback.prefix},
               module_diffs_n=len(module_diffs), module_diffs=module_diffs[:50], name_diffs_n=len(name_diffs), name_diffs=name_diffs[:50],
               partition_equal=(units_a == units_b), units={"record": len(units_a), "v1_annotations": len(units_b)},
               boundary_values={"record": len(sa), "v1_annotations": len(sb), "equal": sa == sb, "only_record": len(sa - sb), "only_v1_annotations": len(sb - sa)})
    if manifest:
        try:
            art_p = os.path.join(program_dir, "artifact.json")
            art = _load_json(art_p) if os.path.exists(art_p) else None
            ma = request_manifest(P, canon, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir).manifest
            mb = request_manifest(P, fallback, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir).manifest
            out["manifest"] = {"record": {"digest": ma["manifest_digest"], "identities": ma["populations"]["identities"], "complete": ma["complete"], "correspondence": ma["correspondence"]},
                               "v1_annotations": {"digest": mb["manifest_digest"], "identities": mb["populations"]["identities"], "complete": mb["complete"], "correspondence": mb["correspondence"]},
                               "digest_equal": ma["manifest_digest"] == mb["manifest_digest"], "wrapper_prefix": [ma["wrapper_prefix"], mb["wrapper_prefix"]]}
        except Exception as e:  # noqa: BLE001  -- a step wrapper is not a request Program; the partition comparison above still stands
            out["manifest"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    out["identical"] = not module_diffs and not name_diffs and out["partition_equal"] and out["boundary_values"]["equal"] and (out.get("manifest", {}).get("digest_equal", True) is True)
    out["seconds"] = round(time.time() - t0, 1)
    if log:
        log(f"  {os.path.basename(program_dir)} ({res.get('wrapper')}): {len(P.calls)} calls, module diffs {len(module_diffs)}, name diffs {len(name_diffs)}, partition_equal {out['partition_equal']}, "
            f"boundary equal {out['boundary_values']['equal']}, manifest {out.get('manifest', {}).get('digest_equal', out.get('manifest'))} in {out['seconds']}s")
    return out


def _write(man: dict[str, Any], out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(man, f, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--program", required=True)
    b.add_argument("--workload", default=None)
    b.add_argument("--request-id", default=None)
    b.add_argument("--profile", default=None)
    b.add_argument("--rank", type=int, default=0)
    b.add_argument("--ranks", type=int, default=1)
    b.add_argument("--no-return-slots", action="store_true")
    b.add_argument("--out", required=True)
    g = sub.add_parser("build-global")
    g.add_argument("--fixture", required=True)
    g.add_argument("--programs-root", required=True)
    g.add_argument("--workload-digest", default=None)
    g.add_argument("--profile", default=None)
    g.add_argument("--rank", type=int, default=0)
    g.add_argument("--ranks", type=int, default=1)
    g.add_argument("--no-return-slots", action="store_true")
    g.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--v1", required=True)
    c.add_argument("--v2", required=True)
    c.add_argument("--ignore-program-digest", action="store_true")
    c.add_argument("--out", default=None)
    k = sub.add_parser("check-correspondence", help="Q_module_body_v1 under the runtime-correspondence/v1 record vs under the v1-annotation fallback on one Build dir")
    k.add_argument("--program", required=True)
    k.add_argument("--workload", default=None)
    k.add_argument("--request-id", default=None)
    k.add_argument("--no-manifest", action="store_true")
    k.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    if a.cmd == "check-correspondence":
        wl = _load_json(a.workload) if a.workload else None
        r = check_correspondence(a.program, workload=wl, request_id=a.request_id, manifest=not a.no_manifest, log=log)
        if a.out:
            _write(r, a.out)
        print(json.dumps({kk: r.get(kk) for kk in ("wrapper", "calls", "source", "applicable", "module_diffs_n", "name_diffs_n", "partition_equal", "boundary_values", "identical", "correspondence_digest")}
                         | {"manifest_digest_equal": (r.get("manifest") or {}).get("digest_equal"), "manifest_digests": [(r.get("manifest") or {}).get(s, {}).get("digest") for s in ("record", "v1_annotations")]}))
        return 0 if (r.get("identical") or not r.get("applicable")) else 1
    if a.cmd == "build":
        wl = _load_json(a.workload) if a.workload else None
        man = build(a.program, workload=wl, request_id=a.request_id, rank=a.rank, ranks=a.ranks, profile=a.profile, return_slots=not a.no_return_slots, log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "complete": man["complete"], "out": a.out}))
        return 0
    if a.cmd == "build-global":
        fx = _load_json(a.fixture)
        man = build_global(fx, a.programs_root, workload_digest=a.workload_digest, rank=a.rank, ranks=a.ranks, profile=a.profile,
                           return_slots=not a.no_return_slots, log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "program_digest": man["program_digest"],
                          "complete": man["complete"], "by_family": man["populations"]["by_family"], "out": a.out}))
        return 0
    if a.cmd == "compare":
        m1, m2 = _load_json(a.v1), _load_json(a.v2)
        cmp = compare(m1["identities"], m2["identities"], ignore_program_digest=a.ignore_program_digest)
        s = summarize(cmp)
        s["v1_manifest_digest"], s["v2_manifest_digest"] = m1.get("manifest_digest"), m2.get("manifest_digest")
        s["v1_program_digest"], s["v2_program_digest"] = m1.get("program_digest"), m2.get("program_digest")
        if a.out:
            _write(s, a.out)
        print(json.dumps({k: s[k] for k in ("v1_identities", "v2_identities", "common", "v1_rows", "v2_rows", "multiplicity_diff_n", "only_v1_n", "only_v2_n")}))
        for k in ("member_rename", "range", "only_v1", "only_v2"):
            for line in s[k]:
                print(f"  {k:14s} n={line['n']:6d} {line.get('op_path')}/{line.get('member')} spec={line.get('spec')} :: {line['reason'][:120]}")
        for m in s["multiplicity"][:10]:
            print(f"  multiplicity   v1_rows={m['v1_rows']} v2_rows={m['v2_rows']} {m['key'][:160]}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
