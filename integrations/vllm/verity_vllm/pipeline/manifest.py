"""`verity-vllm manifest` -- the v2 (Q_module_body_v1) required-value manifest, beside v1's `required_manifest`.

    build         --program <derive dir> [--workload wl.json] [--request-id r0] --out manifest.json
    build-global  --fixture wl.json --programs-root <row dir> [--workload-digest WD] --out manifest.json
                  TP row: --programs-root <rank r dir> --rank r [--workload-digest WD_r], once per rank, --ranks W
    compare       --v1 manifest.json --v2 manifest.json [--out report.json]

`build` / `build-global` take the same inputs as v1's (a derive_step output dir holds `instances.json.gz` + `result.json` [+ `artifact.json`];
a row dir holds one such dir per request shape, `*LP<prompt_len>_T<max_tokens-1>`), and write a manifest of v1's schema whose identities
are the bridge's (`required.request_manifest` / `compose_global`; a TP row's rank manifests merged by `merge_ranks`).  `--no-return-slots`
builds the pure-dataflow population (no RETURN-SLOT policy) for the record.  `--norm-scales` (opt-in, never the record) adds the norm-scale
policy (`query.norm_scales`), `--guarded-max` (likewise) the guarded-max policy (`query.guarded_max`: the query header states it, and
every attention hidden-stream identity gains the MS class); `--router-softmax` and `--vocab-range` (opt-in, never the record) add the
router-softmax and vocabulary-range policies (`query.router_softmax`, `query.vocab_range`); the scale-products policy
(`query.scale_products`) applies to every manifest and adds nothing to a Program without a shared-scale FP8 GEMM.  `--cross-call-check`
(opt-in) fails the Build when a value is computed in two Calls (`query.cross_call`).  `compare` classifies every identity difference
(`compare.summarize`).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from verity_vllm.config import option
from verity_vllm.query.compare import compare, summarize
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import from_instances
from verity_vllm.query.required import compose_global, merge_ranks, request_manifest
from verity_vllm.query import cross_call
from verity_vllm.query import guarded_max as GM
from verity_vllm.query import norm_scales as NS
from verity_vllm.query import router_softmax as RS
from verity_vllm.query import scale_products as SP
from verity_vllm.query import vocab_range as VR
from verity_vllm.query import word as word_query

__all__ = ["TAPS", "DEFAULT_TAPS", "taps_of", "build", "build_global", "build_global_ranks", "cross_call_check", "main"]

#: the tap policies whose identities are Call-interior values, by name (`request_manifest(taps=...)`)
TAPS = {RS.FAMILY: RS.POLICY, VR.FAMILY: VR.POLICY}
#: the tap policies every manifest is built under (each adds nothing to a Program without its Calls)
DEFAULT_TAPS = (SP.POLICY,)


def taps_of(router_softmax: bool = False, vocab_range: bool = False) -> tuple[str, ...]:
    """The tap policy names a manifest is built under, in `TAPS` order."""
    return tuple(name for name, on in ((RS.FAMILY, router_softmax), (VR.FAMILY, vocab_range)) if on)


def _load_json(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


@dataclass(frozen=True)
class _Committed:
    """The committed values of one request Program under the query of record, as `word.check_query` reads them."""
    required: Any
    literals: frozenset[int]
    program_path: Any


def word_check(P: Any, corr: Correspondence, required: Any, word: str, *, strict: bool = True, acquired: dict[str, int] | None = None,
               log=None) -> dict[str, Any]:
    """`Q_word_v1{X,W}` (`word`: "16/32[+extras]") over `P`, committed = the manifest's required values and the interior words a tap
    acquires (`acquired`, `word.check_calls`); `strict` raises `word.QueryRuleViolation` naming every violation.  The manifest is not
    touched."""
    from verity_vllm.query.module_body import literal_calls
    X, W, extras = word_query.parse(word)
    r = word_query.check_query(P, _Committed(required, literal_calls(P), corr.module_of), X, W, extras, strict=strict, acquired=acquired)
    if log:
        log(f"  {r['query']}: {r['calls']} calls -> {r['units']} units over {r['gates']} gates, {r['committed_interior_words']} committed interior words"
            + (f" ({r['acquired_interior_words']} acquired by a tap)" if acquired is not None else ""))
    return r


def cross_call_check(program_dir: str, *, strict: bool = True, log=None) -> dict[str, Any]:
    """Cross-Call recomputes (`cross_call.check_program`) over the Build's instance sequence; `strict` raises
    `cross_call.RecomputeViolation` naming every one.  The manifest is not touched."""
    return cross_call.check_program(os.path.join(program_dir, "instances.json.gz"), strict=strict, log=log)


def build(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, rank: int = 0, ranks: int = 1,
          profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False, guarded_max: bool = False,
          taps: tuple[str, ...] = (), cross_call: bool = False, log=None) -> dict[str, Any]:
    t0 = time.time()
    P = from_instances(os.path.join(program_dir, "instances.json.gz"))
    res = _load_json(os.path.join(program_dir, "result.json"))
    art_p = os.path.join(program_dir, "artifact.json")
    art = _load_json(art_p) if os.path.exists(art_p) else None
    corr = Correspondence.of(P, program_dir, implementation_paths=Correspondence.implementation_paths_of(res))   # the record when the Build carries it, else v1 annotations
    r = request_manifest(P, corr, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, rank=rank, ranks=ranks,
                         profile=profile, return_slots=return_slots, norm_scales=NS.POLICY if norm_scales else None,
                         taps=tuple(TAPS[t] for t in taps) + DEFAULT_TAPS)
    if word:
        sp = SP.policy_of(r.manifest)
        acquired = {**(NS.TAP_WORDS if norm_scales else {}), **{f: w for t in taps for f, w in TAPS[t].tap_words.items()}, **(SP.TAP_WORDS if sp else {})}
        word_check(P, corr, r.required, word, acquired=acquired if (norm_scales or taps or sp) else None, log=log)
    if guarded_max:
        r.manifest["query"] = GM.header(r.manifest["query"], GM.attention_words(P), GM.max_scaled_words(P))
        r.manifest.update(GM.with_ms(r.manifest))
    if cross_call:
        cross_call_check(program_dir, log=log)
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
                 profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False,
                 guarded_max: bool = False, taps: tuple[str, ...] = (), cross_call: bool = False, log=None) -> dict[str, Any]:
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
        parts[rid] = build(d, workload=wl, request_id=rid, rank=rank, ranks=ranks, profile=profile, return_slots=return_slots, word=word,
                           norm_scales=norm_scales, guarded_max=guarded_max, taps=taps, cross_call=cross_call, log=log)
        if len(parts[rid]["steps"]) != int(r["max_tokens"]):
            raise ValueError(f"request {rid}: Program has {len(parts[rid]['steps'])} steps, the declaration caps {r['max_tokens']} tokens: wrong component")
    man = compose_global(fixture, parts, reqs, workload_digest=workload_digest, rank=rank, ranks=ranks)
    if guarded_max:
        man = GM.with_ms(dict(man, query=GM.merged(man["query"], [p["query"] for p in parts.values()])))
    return man


def build_global_ranks(fixture: dict[str, Any], roots: list[str], ranks: list[int], world: int, *, workload_digests: list[str | None] | None = None,
                       profile: str | None = None, return_slots: bool = True, word: str | None = None, norm_scales: bool = False,
                       guarded_max: bool = False, taps: tuple[str, ...] = (), cross_call: bool = False, log=None) -> dict[str, Any]:
    """One global manifest per rank Program set (`roots[i]` is rank `ranks[i]`'s), merged by `merge_ranks` when the row is TP."""
    wds = list(workload_digests or [None] * len(roots))
    if len(ranks) != len(roots) or len(wds) != len(roots):
        raise ValueError(f"build-global: {len(roots)} --programs-root, {len(ranks)} --rank, {len(wds)} --workload-digest: give one per rank")
    if world > 1 and sorted(ranks) != list(range(world)):
        raise ValueError(f"build-global: TP world {world} needs one --programs-root per rank 0..{world - 1} (got ranks {ranks})")
    parts = [build_global(fixture, root, workload_digest=wd, rank=r, ranks=world, profile=profile, return_slots=return_slots, word=word,
                          norm_scales=norm_scales, guarded_max=guarded_max, taps=taps, cross_call=cross_call, log=log)
             for root, r, wd in zip(roots, ranks, wds)]
    return merge_ranks(parts) if world > 1 else parts[0]


def check_correspondence(program_dir: str, *, workload: dict[str, Any] | None = None, request_id: str | None = None, manifest: bool = True, log=None) -> dict[str, Any]:
    """`Q_module_body_v1` under the record vs under the v1-annotation fallback on ONE Build dir that carries the record: the owning
    module per Call, the Call names the bridge reads, the partition (unit ids), the boundary Values and -- `manifest` -- the bridge's
    manifest digest, each equal or listed.  A Build without the record has nothing to check against (`applicable: False`)."""
    from verity_vllm.query.module_body import Q_module_body_v1, literal_calls, module_body_boundary
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
    literals = literal_calls(P)
    sa, sb = set(module_body_boundary(P, canon, literals).values), set(module_body_boundary(P, fallback, literals).values)
    out.update(applicable=True, module_paths={"record": len(canon.modules), "v1_annotations": len(fallback.modules)}, implementation_prefix={"record": canon.prefix, "v1_annotations": fallback.prefix},
               module_diffs_n=len(module_diffs), module_diffs=module_diffs[:50], name_diffs_n=len(name_diffs), name_diffs=name_diffs[:50],
               partition_equal=(units_a == units_b), units={"record": len(units_a), "v1_annotations": len(units_b)},
               boundary_values={"record": len(sa), "v1_annotations": len(sb), "equal": sa == sb, "only_record": len(sa - sb), "only_v1_annotations": len(sb - sa)})
    if manifest:
        try:
            art_p = os.path.join(program_dir, "artifact.json")
            art = _load_json(art_p) if os.path.exists(art_p) else None
            tp = res.get("tp") if isinstance(res.get("tp"), dict) else {}
            rk = dict(rank=int(tp.get("rank") or 0), ranks=int(tp.get("world") or 1))          # a rank Program's manifest is its rank's
            ma = request_manifest(P, canon, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, **rk).manifest
            mb = request_manifest(P, fallback, res, request_id=request_id, workload=workload, artifact=art, program_dir=program_dir, **rk).manifest
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


ROUTER_SOFTMAX_HELP = ("opt-in, never the record: the router-softmax policy (verity_vllm.query.router_softmax) -- one router_softmax identity "
                       "per (step, MoE layer) of the selection-order router, `moe/L<k>/router_softmax`; the Commit then needs --router-tap-so")
VOCAB_RANGE_HELP = ("opt-in, never the record: the vocabulary-range policy (verity_vllm.query.vocab_range) -- one vocab_range identity per "
                    "(step, TP embedding site), `<embedding module>/vocab_range`; the Commit then needs --vocab-range")
NORM_SCALES_HELP = ("opt-in, never the record: the norm-scale policy (verity_vllm.query.norm_scales) -- one norm_scales identity per norm "
                    "Call row, `<norm module>/norm_scale`; the Commit then needs --norm-tap-so")
CROSS_CALL_HELP = "check that no value is computed in two Calls (verity_vllm.query.cross_call); fail the Build naming every recompute"
GUARDED_MAX_HELP = ("opt-in, never the record: the guarded-max policy (verity_vllm.query.guarded_max) -- the attention stream's ROW word 3 "
                    "carries GuardNegInfZero(m) after a row's first key block and its MS class every block's max * scale; the Commit then "
                    "needs the guarded-max tap build")


@dataclass(frozen=True, kw_only=True)
class BuildOptions:
    program: str = option("--program")
    workload: str | None = option("--workload", default=None)
    request_id: str | None = option("--request-id", default=None)
    profile: str | None = option("--profile", default=None)
    rank: int = option("--rank", type=int, default=0)
    ranks: int = option("--ranks", type=int, default=1)
    no_return_slots: bool = option("--no-return-slots", default=False, action="store_true")
    word_check: str | None = option("--word-check", env="VERITY_WORD_CHECK", default=None,
                                    help="check Q_word_v1{X,W} (\"16/32[+extras]\") over the Program; fail the Build naming every violation")
    norm_scales: bool = option("--norm-scales", default=False, action="store_true", help=NORM_SCALES_HELP)
    guarded_max: bool = option("--guarded-max", default=False, action="store_true", help=GUARDED_MAX_HELP)
    router_softmax: bool = option("--router-softmax", default=False, action="store_true", help=ROUTER_SOFTMAX_HELP)
    vocab_range: bool = option("--vocab-range", default=False, action="store_true", help=VOCAB_RANGE_HELP)
    cross_call_check: bool = option("--cross-call-check", default=False, action="store_true", help=CROSS_CALL_HELP)
    out: str = option("--out")
    cmd: str = field(default="build", init=False)


@dataclass(frozen=True, kw_only=True)
class BuildGlobalOptions:
    fixture: str = option("--fixture")
    programs_root: list[str] = option("--programs-root", action="append",
                                      help="TP row: repeat per rank, with --rank r at the same position")
    workload_digest: list[str] | None = option("--workload-digest", action="append", default=None,
                                               help="TP row: repeat per rank")
    profile: str | None = option("--profile", default=None)
    rank: list[int] | None = option("--rank", type=int, action="append", default=None,
                                    help="the TP rank of the --programs-root at the same position (default 0)")
    ranks: int = option("--ranks", type=int, default=1)
    no_return_slots: bool = option("--no-return-slots", default=False, action="store_true")
    word_check: str | None = option("--word-check", env="VERITY_WORD_CHECK", default=None,
                                    help="check Q_word_v1{X,W} (\"16/32[+extras]\") over the Program; fail the Build naming every violation")
    norm_scales: bool = option("--norm-scales", default=False, action="store_true", help=NORM_SCALES_HELP)
    guarded_max: bool = option("--guarded-max", default=False, action="store_true", help=GUARDED_MAX_HELP)
    router_softmax: bool = option("--router-softmax", default=False, action="store_true", help=ROUTER_SOFTMAX_HELP)
    vocab_range: bool = option("--vocab-range", default=False, action="store_true", help=VOCAB_RANGE_HELP)
    cross_call_check: bool = option("--cross-call-check", default=False, action="store_true", help=CROSS_CALL_HELP)
    out: str = option("--out")
    cmd: str = field(default="build-global", init=False)


@dataclass(frozen=True, kw_only=True)
class CompareOptions:
    v1: str = option("--v1")
    v2: str = option("--v2")
    ignore_program_digest: bool = option("--ignore-program-digest", default=False, action="store_true")
    out: str | None = option("--out", default=None)
    cmd: str = field(default="compare", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckCorrespondenceOptions:
    """Q_module_body_v1 under the runtime-correspondence/v1 record vs under the v1-annotation fallback on one Build dir"""

    program: str = option("--program")
    workload: str | None = option("--workload", default=None)
    request_id: str | None = option("--request-id", default=None)
    no_manifest: bool = option("--no-manifest", default=False, action="store_true")
    out: str | None = option("--out", default=None)
    cmd: str = field(default="check-correspondence", init=False)


COMMANDS = {"build": BuildOptions, "build-global": BuildGlobalOptions, "compare": CompareOptions, "check-correspondence": CheckCorrespondenceOptions}


def main(a: Any) -> int:
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
        man = build(a.program, workload=wl, request_id=a.request_id, rank=a.rank, ranks=a.ranks, profile=a.profile, return_slots=not a.no_return_slots,
                    word=a.word_check, norm_scales=a.norm_scales, guarded_max=a.guarded_max, taps=taps_of(a.router_softmax, a.vocab_range),
                    cross_call=a.cross_call_check, log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "complete": man["complete"], "out": a.out}))
        return 0
    if a.cmd == "build-global":
        fx = _load_json(a.fixture)
        man = build_global_ranks(fx, a.programs_root, a.rank or [0] * len(a.programs_root), a.ranks, workload_digests=a.workload_digest,
                                 profile=a.profile, return_slots=not a.no_return_slots, word=a.word_check, norm_scales=a.norm_scales,
                                 guarded_max=a.guarded_max, taps=taps_of(a.router_softmax, a.vocab_range),
                                 cross_call=a.cross_call_check, log=log)
        _write(man, a.out)
        print(json.dumps({"identities": man["populations"]["identities"], "manifest_digest": man["manifest_digest"], "program_digest": man["program_digest"],
                          "complete": man["complete"], "by_family": man["populations"]["by_family"],
                          "tp_peer_binding_n_unbound": (man.get("tp_peer_binding") or {}).get("n_unbound"), "out": a.out}))
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
