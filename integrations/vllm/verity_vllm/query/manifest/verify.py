"""Is `manifest.json` THE manifest of record of this Program x workload?  Rebuild it by the integration's query path (`verity_vllm.pipeline.manifest`:
`Q_module_body_v1` over the Build's Programs + correspondence, the same `build` / `build-global` inputs row_pod.sh built it with) and
compare digests; write the `verity-sweep/manifest-verify/v1` record beside the Commit (`commit/manifest_verify.json`).

    verity-vllm manifest-verify --manifest D/manifest.json --out D/commit/manifest_verify.json \\
        --program D/build_request --workload workloads/<row>.json                                 # B=1
    verity-vllm manifest-verify --manifest D/manifest.json --out D/commit/manifest_verify.json \\
        --fixture workloads/<row>.json --programs-root D [--workload-digest WD]                     # B>=2

Exit 0 iff `ok`: the used manifest's `manifest_digest` == the digest of its own rows == the rebuilt digest, the program digests agree and
the rebuilt manifest is complete.  [rev-cb F-cb-11] the manifest of record is a FUNCTION of the Program x declared workload: a weakened
manifest in the evidence dir has the wrong digest here even when its own `manifest_digest` is honest.  A verify FAIL is a Commit FAIL
(EXEC-05: `required_value_coverage/manifest_of_record_verified` in verdict.json), never a footnote.  Replaces `required_manifest verify`
(the v1 builders' rebuild) on the row path since the v2 flip; the record's fields are unchanged so the regression harness lifts
`used_digest` / `used_program_digest` / `used_identities` as before.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from verity_vllm.config import option
from verity_vllm.query.manifest.format import manifest_digest

SCHEMA = "verity-sweep/manifest-verify/v1"
ENGINE = "v2-query"


def _load(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def rebuild(*, program: str | None = None, workload: str | None = None, fixture: str | None = None, programs_root: str | None = None,
            workload_digest: str | None = None, rank: int = 0, ranks: int = 1, profile: str | None = None, norm_scales: bool = False,
            taps: tuple[str, ...] = (), log=None) -> dict[str, Any]:
    """The manifest of this Program x workload by the query path (`manifest.build` at B=1, `manifest.build_global` at B>=2), under the
    norm-scale policy and the tap policies when the caller's row runs them (`norm_scales`, `taps`; the policies are the caller's, never read
    off the manifest checked)."""
    from verity_vllm.pipeline import manifest
    if program:
        wl = _load(workload) if workload else None
        return manifest.build(program, workload=wl, rank=rank, ranks=ranks, profile=profile, norm_scales=norm_scales, taps=taps, log=log)
    if fixture and programs_root:
        return manifest.build_global(_load(fixture), programs_root, workload_digest=workload_digest, rank=rank, ranks=ranks, profile=profile,
                                     norm_scales=norm_scales, taps=taps, log=log)
    raise SystemExit("manifest_verify: give --program [--workload] (B=1) or --fixture --programs-root [--workload-digest] (B>=2)")


def verify(used: dict[str, Any], rebuilt: dict[str, Any], *, manifest_path: str | None = None) -> dict[str, Any]:
    rows_digest = manifest_digest(used.get("identities") or [])
    rec = {"schema": SCHEMA, "manifest": manifest_path, "used_digest": used.get("manifest_digest"), "used_rows_digest": rows_digest,
           "rebuilt_digest": rebuilt["manifest_digest"], "used_program_digest": used.get("program_digest"), "rebuilt_program_digest": rebuilt["program_digest"],
           "used_identities": len(used.get("identities") or []), "rebuilt_identities": rebuilt["populations"]["identities"], "rebuilt_complete": rebuilt.get("complete"),
           "used_query": (used.get("query") or {}).get("engine") if isinstance(used.get("query"), dict) else None,
           "rebuilt_engine": ENGINE}
    rec["ok"] = bool(rec["used_digest"] == rows_digest == rec["rebuilt_digest"] and rec["used_program_digest"] == rec["rebuilt_program_digest"] and rebuilt.get("complete") is True)
    if not rec["ok"]:
        why = []
        if rec["used_digest"] != rows_digest:
            why.append(f"manifest_digest {str(rec['used_digest'])[:16]} != digest of its rows {rows_digest[:16]} (edited after digesting)")
        if rows_digest != rec["rebuilt_digest"]:
            why.append(f"rows digest {rows_digest[:16]} != the Program x workload's manifest {rec['rebuilt_digest'][:16]} ({rec['used_identities']} vs {rec['rebuilt_identities']} identities): not the manifest of record")
        if rec["used_program_digest"] != rec["rebuilt_program_digest"]:
            u, b = str(rec["used_program_digest"]), str(rec["rebuilt_program_digest"])
            if u[:16] == b[:16] or len(u) != len(b):                          # [exec M-0545] a prefix passed as --workload-digest: say so, full width
                why.append(f"program_digest {u} ({len(u)} hex) != the rebuilt {b} ({len(b)} hex): pass the FULL digest (workload_program.json), not a prefix")
            else:
                why.append(f"program_digest {u[:16]} != the Build's {b[:16]}")
        if rebuilt.get("complete") is not True:
            why.append(f"the Program x workload manifest is incomplete: {rebuilt.get('unmodelled')}")
        rec["why"] = "; ".join(why)
    return rec


@dataclass(frozen=True, kw_only=True)
class Options:
    manifest: str = option("--manifest", help="the manifest the Commit used (D/manifest.json)")
    program: str | None = option("--program", default=None, help="B=1: the Build's derive_step dir (build_request)")
    workload: str | None = option("--workload", default=None, help="B=1: the sweep workload manifest the build used")
    fixture: str | None = option("--fixture", default=None, help="B>=2: the workload manifest / fixture (as build-global)")
    programs_root: str | None = option("--programs-root", default=None, help="B>=2: the Build dir holding the request Programs")
    workload_digest: str | None = option("--workload-digest", default=None,
                                         help="B>=2: GP-01's FULL workload digest (as build-global)")
    profile: str | None = option("--profile", default=None)
    rank: int = option("--rank", type=int, default=0)
    ranks: int = option("--ranks", type=int, default=1)
    norm_scales: bool = option("--norm-scales", default=False, action="store_true",
                               help="rebuild under the norm-scale policy (the row runs NORM_TAP=1; opt-in, never the record)")
    router_softmax: bool = option("--router-softmax", default=False, action="store_true",
                                  help="rebuild under the router-softmax policy (the row runs ROUTER_TAP=1; opt-in, never the record)")
    vocab_range: bool = option("--vocab-range", default=False, action="store_true",
                               help="rebuild under the vocabulary-range policy (the row runs VOCAB_TAP=1; opt-in, never the record)")
    out: str | None = option("--out", default=None, help="write the verification record (json)")


def main(a: Options) -> int:
    log = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    used = _load(a.manifest)
    from verity_vllm.pipeline.manifest import taps_of
    rebuilt = rebuild(program=a.program, workload=a.workload, fixture=a.fixture, programs_root=a.programs_root, workload_digest=a.workload_digest,
                      rank=a.rank, ranks=a.ranks, profile=a.profile, norm_scales=a.norm_scales,
                      taps=taps_of(a.router_softmax, a.vocab_range), log=log)
    rec = verify(used, rebuilt, manifest_path=a.manifest)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
    print(f"[manifest] VERIFY {'OK: the manifest of record' if rec['ok'] else 'FAIL: ' + rec['why']}")
    return 0 if rec["ok"] else 1
