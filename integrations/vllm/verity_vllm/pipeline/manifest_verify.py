"""Is `manifest.json` THE manifest of record of this Program x workload?  Rebuild it by the integration's query path (`verity_vllm.pipeline.manifest`:
`Q_module_body_v1` over the Build's Programs + correspondence, the same `build` / `build-global` inputs row_pod.sh built it with) and
compare digests (`verity_vllm.query.manifest.verify.verify`); write the `verity-sweep/manifest-verify/v1` record beside the Commit
(`commit/manifest_verify.json`).

    verity-vllm manifest-verify --manifest D/manifest.json --out D/commit/manifest_verify.json \\
        --program D/build_request --workload workloads/<row>.json                                 # B=1
    verity-vllm manifest-verify --manifest D/manifest.json --out D/commit/manifest_verify.json \\
        --fixture workloads/<row>.json --programs-root D [--workload-digest WD]                     # B>=2

Exit 0 iff the record's `ok`.  A verify FAIL is a Commit FAIL (`required_value_coverage/manifest_of_record_verified` in
verdict.json), never a footnote.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import option, workload_arg
from verity_vllm.query.manifest.verify import verify


def _load(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def rebuild(*, program: str | None = None, workload: str | None = None, fixture: str | None = None, programs_root: str | None = None,
            workload_digest: str | None = None, rank: int = 0, ranks: int = 1, profile: str | None = None, norm_scales: bool = False,
            guarded_max: bool = False, taps: tuple[str, ...] = (), query: str | None = None, log=None) -> dict[str, Any]:
    """The manifest of this Program x workload by the query path (`manifest.build` at B=1, `manifest.build_global` at B>=2), under the
    norm-scale / guarded-max policies and the tap policies when the caller's row runs them (`norm_scales`, `guarded_max`, `taps`; the policies
    are the caller's, never read off the manifest checked; so is `query`, the query of record unless the caller names another)."""
    from verity_vllm.pipeline import manifest
    from verity_vllm.query.required import QUERY_OF_RECORD
    query = query or QUERY_OF_RECORD
    if program:
        wl = _load(workload) if workload else None
        return manifest.build(program, workload=wl, rank=rank, ranks=ranks, profile=profile, norm_scales=norm_scales, guarded_max=guarded_max,
                              taps=taps, query=query, log=log)
    if fixture and programs_root:
        return manifest.build_global(_load(fixture), programs_root, workload_digest=workload_digest, rank=rank, ranks=ranks, profile=profile,
                                     norm_scales=norm_scales, guarded_max=guarded_max, taps=taps, query=query, log=log)
    raise SystemExit("manifest_verify: give --program [--workload] (B=1) or --fixture --programs-root [--workload-digest] (B>=2)")


@dataclass(frozen=True, kw_only=True)
class Options:
    manifest: str = option("--manifest", help="the manifest the Commit used (D/manifest.json)")
    program: str | None = option("--program", default=None, help="B=1: the Build's derive_step dir (build_request)")
    workload: str | None = option("--workload", type=workload_arg, default=None, help="B=1: the sweep workload manifest the build used")
    fixture: str | None = option("--fixture", type=workload_arg, default=None, help="B>=2: the workload manifest / fixture (as build-global)")
    programs_root: str | None = option("--programs-root", default=None, help="B>=2: the Build dir holding the request Programs")
    workload_digest: str | None = option("--workload-digest", default=None,
                                         help="B>=2: GP-01's FULL workload digest (as build-global)")
    profile: str | None = option("--profile", default=None)
    rank: int = option("--rank", type=int, default=0)
    ranks: int = option("--ranks", type=int, default=1)
    norm_scales: bool = option("--norm-scales", default=True, action="boolean_optional",
                               help="rebuild under the norm-scale policy (on by default; --no-norm-scales when the row ran NORM_TAP=0, for A/B only)")
    guarded_max: bool = option("--guarded-max", default=True, action="boolean_optional",
                               help="rebuild under the guarded-max policy (on by default; --no-guarded-max when the row ran GUARDED_MAX_TAP=0, for A/B only)")
    router_softmax: bool = option("--router-softmax", default=True, action="boolean_optional",
                                  help="rebuild under the router-softmax policy (on by default; --no-router-softmax when the row ran ROUTER_TAP=0, for A/B only)")
    vocab_range: bool = option("--vocab-range", default=True, action="boolean_optional",
                               help="rebuild under the vocabulary-range policy (on by default; --no-vocab-range when the row ran VOCAB_TAP=0, for A/B only)")
    query: str | None = option("--query", default=None, help="the query to rebuild under (default: the query of record; Q_module_body_v1 for A/B only)")
    out: str | None = option("--out", default=None, help="write the verification record (json)")


def main(a: Options) -> int:
    def log(s):
        print(s, file=sys.stderr, flush=True)

    used = _load(a.manifest)
    from verity_vllm.pipeline.manifest import taps_of
    rebuilt = rebuild(program=a.program, workload=a.workload, fixture=a.fixture, programs_root=a.programs_root, workload_digest=a.workload_digest,
                      rank=a.rank, ranks=a.ranks, profile=a.profile, norm_scales=a.norm_scales, guarded_max=a.guarded_max,
                      taps=taps_of(a.router_softmax, a.vocab_range), query=a.query, log=log)
    rec = verify(used, rebuilt, manifest_path=a.manifest)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print(f"[manifest] VERIFY {'OK: the manifest of record' if rec['ok'] else 'FAIL: ' + rec['why']}")
    return 0 if rec["ok"] else 1
