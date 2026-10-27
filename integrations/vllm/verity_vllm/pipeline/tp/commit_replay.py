"""The C2 replay of a deferred TP Commit (`tp-commit --replay-deferred`), on CPU, from the bundle each rank sealed.

    verity-vllm tp-commit-replay COMMIT_OUT --checkpoint-dir CK [--checkpoints-manifest M --model REPO --revision REV] [--ranks 0,3]

The TP Commit runs every check but the ranks' sampled replay, boundary linkage and population reconciliation, and each rank writes
one bundle per instrumented run, `<out>/replay_bundle_p<pair>_rank<r>/` (`pipeline.replay_bundle`: the rank's dumped store, its
committed weights tree and the tensors no checkpoint holds, its binding map, `context.json`: the replay's configuration, the rank's
run root, the required manifest's sha256, the engine's facts of record).  The Commit's verdict is PENDING (`summary.json` `pass`
None, every rank's `commit_pass` None) and `replay_deferred.json` names the bundles by manifest sha256 and keeps what the fold reads
back: the value correspondence before the C2's components, the weights pin, cross-rank and fold-Match components, and the
cross-rank pass of every pair.

Per rank (`replay_rank`), the bundle is opened against its manifest and against the record: the manifest sha256 the record names, the
rank's run root in `runs.jsonl` and its combination into the pair's TP run root (under the form the record's ``root_tag`` names,
`tp.run_root.combine`: SHA-512 over rows-sha512's 64-byte roots, or an older record's SHA-256 v1), the dumped store's step roots folding to it, the
binding map's digest, the required manifest and the rank's Programs of record.  The rank's weight SHARDS are composed from the
checkpoint (`weights_of_record.compose` at the rank's (world, rank)) and opened against the committed weights root
(`cpu_replay.open_weights`).  Then the same C2 as the rank's process runs in place (`rank_replay.evaluate`, the same seed), and its
records go to `sampled_replay_p<pair>_rank<r>.json`, so a rank replayed is never replayed again and the ranks may replay on
different hosts.  Once every rank's record is there, `fold` decides the Commit as the in-process Commit would have: each rank's
gates after its pre-replay verdict (`c2.replay_gates`), the row's verdict, `finalize_pair<k>.json`, coverage, `sampled_replay.json`,
`summary.json` and `runs.jsonl`.  Exit 0 PASS, 3 FAIL, 4 ranks remain, 2 refused (a bundle that is not this record's, a pick too
large for this host's memory).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import resource
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import option
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline.tp import c2 as C2

RESULT_SCHEMA = "verity-vllm/tp-rank-replay/v1"


@dataclass(frozen=True, kw_only=True)
class Options:
    commit_out: str = option("commit_out", help="the deferred TP Commit's --out (its replay_deferred.json names the ranks' sealed bundles)")
    checkpoint_dir: str = option("--checkpoint-dir", help="the checkpoint of record (safetensors shards + config.json)")
    checkpoints_manifest: str | None = option("--checkpoints-manifest", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    model: str | None = option("--model", default=None, help="the checkpoint's repo (its checkpoints.json entry)")
    revision: str | None = option("--revision", default=None, help="the checkpoint's revision")
    ranks: str | None = option("--ranks", default=None, help="comma-separated ranks to replay here (default: every rank without a replay on the record)")
    build_rank: list[str] = option("--build-rank", action="append", default=[],
                                   help="rank r's Build dir on this host, repeated in rank order (default: the Commit's; digests are checked either way)")
    required_manifest: str | None = option("--required-manifest", default=None, help="the Commit's required manifest (default: its recorded path)")
    threads: int = option("--threads", type=int, default=8, help="threads opening the weights")
    replay_workers: int | None = option("--replay-workers", type=int, default=None,
                                        help="the forked pool's workers (default: the Commit's own; the draw does not depend on it)")
    worker_peak_gb: float | None = option("--worker-peak-gb", type=float, default=None,
                                          help="a forked worker's peak memory beyond what the row evaluator states for its pick, GiB (`cpu_replay.worker_peak`)")
    fold_only: bool = option("--fold-only", default=False, action="store_true", help="replay nothing: decide the record from the ranks' replays on it")


class Refused(Exception):
    """A bundle that is not the Commit record's, or a record with nothing deferred."""


def _rows(out: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in (out / "runs.jsonl").read_text().splitlines() if ln.strip()]


def result_path(out: str | os.PathLike, pair: int, rank: int) -> Path:
    return Path(out) / f"sampled_replay_p{pair}_rank{rank}.json"


# ------------------------------------------------------------------------------------------------ one rank, on CPU

def _of_record(out: Path, dd: dict[str, Any], b: dict[str, Any], pair: int, rank: int, *,
               awaiting: bool = True) -> tuple[dict, dict, Any, Any, dict]:
    """(the run's row, the rank's commit block, store, run, binding map) once the bundle is shown to be the record's; else Refused.
    ``awaiting=False`` (a reader of the committed values, `tp.serving_rows.from_bundle`): the rank's replay was deferred to this
    bundle and is pending or passed; a rank whose replay failed is refused."""
    from verity_vllm.commit import store_dump as SD
    from verity_vllm.pipeline.tp import run_root as RT

    ctx, bdir = b["context"], Path(b["path"])
    ent = next((x for x in dd.get("bundles") or [] if (int(x["pair"]), int(x["rank"])) == (pair, rank)), None)
    if ent is None or ent.get("manifest_sha256") != b["manifest_sha256"]:
        raise Refused(f"{C2.DEFERRED} names no bundle of pair {pair} rank {rank} with manifest sha256 {b['manifest_sha256'][:16]}")
    if (int(ctx["pair"]), int(ctx["tp"]["rank"]), int(ctx["tp"]["world"])) != (pair, rank, int(dd["world"])):
        raise Refused(f"{bdir}: its context is pair {ctx['pair']} tp {ctx['tp']}, not pair {pair} rank {rank} of world {dd['world']}")
    row = next((r for r in _rows(out) if r["pair"] == pair and r["arm"] != "control"), None)
    c = ((row or {}).get("commit") or {}).get("per_rank", {}).get(str(rank))
    if c is None or "replay_deferred" not in c or c.get("commit_pass") is not None and (awaiting or c["commit_pass"] is not True):
        raise Refused(f"pair {pair} rank {rank}: no rank " + ("awaiting its deferred replay" if awaiting else "whose deferred replay is pending or passed")
                      + f" in {out / 'runs.jsonl'}")
    vec = {int(v["rank"]): bytes.fromhex(v["run_root"]) for v in row["commit"].get("commitment_vector") or []}
    try:
        combined = RT.combine(json.loads((out / "versions.json").read_text()).get("root_tag"), int(dd["world"]), vec).hex()
    except ValueError as e:
        raise Refused(f"pair {pair}: the commitment vector does not combine into a TP run root: {e}") from e
    if len(vec) != int(dd["world"]) or combined != row["commit"].get("tp_run_root") or ctx["tp_run_root"] != row["commit"]["tp_run_root"]:
        raise Refused(f"pair {pair}: the commitment vector does not combine into the TP run root of record {str(row['commit'].get('tp_run_root'))[:16]}")
    store, run = SD.load(bdir / "store")
    folded = store._run_root(run)
    for what, got in (("the bundle's run root", ctx["run_root"]), ("the store's run root", run.run_root.hex()),
                      ("the store's step roots folded", folded.hex() if folded else None), ("the commitment vector's", vec[rank].hex())):
        if got != c.get("run_root"):
            raise Refused(f"rank {rank}: {what} {str(got)[:16]} is not the rank's run root of record {str(c.get('run_root'))[:16]}")
    bmap = json.loads((bdir / "binding_map.json").read_text())
    canon = json.dumps(bmap.get("entries"), sort_keys=True, separators=(",", ":"), default=str).encode()     # `binding.build_binding_map`'s digest
    want = (c.get("binding") or {}).get("map_digest")
    if not want or {bmap.get("digest"), hashlib.sha256(canon).hexdigest()} != {want} or (bmap.get("run_root"), int(bmap.get("rank") or 0)) != (c.get("run_root"), rank):
        raise Refused(f"rank {rank}: the bundle's binding map {str(bmap.get('digest'))[:16]} is not the rank's map of record {str(want)[:16]}")
    run.weight_registration = c.get("weight_registration")
    return row, c, store, run, bmap


def _manifest(o: Options, ctx: dict[str, Any]) -> dict | None:
    want = ctx.get("required_manifest_sha256")
    if not want:
        return None
    p = o.required_manifest or ctx.get("required_manifest")
    if not p or not os.path.isfile(p) or C2.sha256_file(Path(p)) != want:
        raise Refused(f"{p} is not the Commit's required manifest (sha256 {want[:16]})")
    return json.loads(Path(p).read_text())


def checkpoint_index(o: Options) -> Any:
    from verity_vllm.check import weights_of_record as WR

    return WR.CheckpointIndex(o.checkpoint_dir, checkpoints_manifest=o.checkpoints_manifest, model=o.model, revision=o.revision)


def registered_roles(p: Any) -> dict[str, dict[str, Any]]:
    """The rank's Build `weight_roles` by registered name (`cpu_replay.registered_roles` of its request Programs' dirs, less its index's prefix)."""
    from verity_vllm.pipeline import cpu_replay as CPU

    return CPU.registered_roles((v.get("dir") for v in p.loaded.values()), getattr(p.index, "prefix", "") or "")


def _peak_rss() -> int:
    return max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024


def replay_rank(o: Options, out: Path, dd: dict[str, Any], pair: int, rank: int, ck: Callable[[], Any] | None = None) -> dict[str, Any]:
    """The rank's C2 from its bundle (module docstring), its record written to `sampled_replay_p<pair>_rank<r>.json`.  ``ck``: the
    checkpoint's index, shared by the ranks one process replays, since building it reads every shard whole for its sha256."""
    import traceback

    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.check.replay.driver import PoolDoesNotFit
    from verity_vllm.pipeline import cpu_replay as CPU
    from verity_vllm.pipeline.tp import rank_replay as RR

    t0 = time.perf_counter()
    try:
        b = RB.open_bundle(RB.path(out, pair, rank=rank))
    except (OSError, ValueError) as e:
        raise Refused(f"pair {pair} rank {rank}: {e}") from e
    ctx, bdir = b["context"], Path(b["path"])
    _row, c, store, run, bmap = _of_record(out, dd, b, pair, rank)
    man = _manifest(o, ctx)
    cfg = dict(ctx["replay"])
    if o.build_rank:
        cfg["programs"] = {str(r): d for r, d in enumerate(o.build_rank)}
    if o.replay_workers is not None:
        cfg["workers"] = o.replay_workers
    tree = json.loads((bdir / "weights" / "tree.json").read_text())
    wc, attest = None, None
    try:
        p = RR.load(cfg, rank, int(dd["world"]))
        index = ck() if ck else checkpoint_index(o)
        words, wc = CPU.open_weights(bdir, tree, (c.get("weight_registration") or {}).get("root"), index, threads=o.threads,
                                     tp={"world": int(dd["world"]), "rank": rank}, roles=registered_roles(p))
        # the bytes the relations consume are the checkpoint's, opened against the run's committed weights root: self-consistency
        # with the run as in the rank's process, from the checkpoint's bytes; the pin of record stays the Commit's weights pin
        wc.update(self_consistent=wc["equal"], pin_of_record=None, pinned=None)
        print(f"[tp-replay] WEIGHTS (pair {pair}, rank {rank}): {'OPENED' if wc['equal'] else 'FAIL'} {wc['opened']}/{wc['servable']} registered shards "
              f"against the committed weights root {str(wc['committed_root'])[:16]} ({wc['opened_by_source']}; unopened {wc['unopened_n']}) in {wc['wall_s']} s"
              + (f" -- {wc['why']}" if wc["why"] else ""), flush=True)
        consumed: set[str] = set()
        sr, lk = RR.evaluate(p, cfg, bmap=bmap, committed=OC.OpenedReader(store, run), weights_of=lambda prefix: CPU.provider(words, prefix, consumed),
                             weights_check=wc, run_root=run.run_root, manifest_doc=man, worker_peak=CPU.worker_peak(o))
        sr["engine_of_replay"], sr["engine_of_record"] = sr.get("engine_of_record"), ctx.get("engine_of_record")
        attest = CPU.weights_attest(c, wc, consumed, words, tree.get("extra") or {}, b)
    except (Refused, PoolDoesNotFit):
        raise
    except Exception as e:  # noqa: BLE001  -- fail closed, as the rank's process: an unrunnable replay is 'not evaluated', never PASS
        sr, lk = RR.not_run(rank, e, traceback.format_exc())
    doc = {"schema": RESULT_SCHEMA, "pair": pair, "rank": rank, "bundle": b["path"], "manifest_sha256": b["manifest_sha256"], "where": "cpu",
           "host": socket.gethostname(), "sampled_replay": sr, "boundary_linkage": lk, "weights_check": wc, "weights_attest": attest,
           "wall_s": round(time.perf_counter() - t0, 3), "peak_rss_bytes": _peak_rss(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    C2.write_atomic(result_path(out, pair, rank), json.dumps(doc, indent=1, default=str) + "\n")
    smp = sr.get("sample") or {}
    print(f"[tp-replay] SAMPLED REPLAY (pair {pair}, rank {rank}): result {sr.get('result')} grade {sr.get('grade')} picked {smp.get('picked')} "
          f"evaluated {smp.get('evaluated')} equal {smp.get('equal')} mismatch {smp.get('mismatch_n')}; boundary linkage {lk.get('result')} "
          f"{lk.get('linked')}/{lk.get('pairs')}; {doc['wall_s']} s, peak RSS {doc['peak_rss_bytes'] >> 20} MiB; {sr.get('why') or ''}", flush=True)
    return doc


# ------------------------------------------------------------------------------------------------ the record, decided

def fold(out: Path) -> bool | None:
    """The Commit decided from every rank's replay on the record (None, nothing written, while a rank's is missing): each rank's
    gates after its pre-replay verdict, then everything the in-process Commit derives from them."""
    from verity_vllm.pipeline.tp import config_run as TPC

    dd = json.loads((out / C2.DEFERRED).read_text())
    results: dict[tuple[int, int], dict] = {}
    for b in dd["bundles"]:
        rp = result_path(out, b["pair"], b["rank"])
        if not rp.is_file():
            return None
        res = json.loads(rp.read_text())
        if res.get("manifest_sha256") != b["manifest_sha256"]:
            raise Refused(f"{rp} is the replay of bundle {str(res.get('manifest_sha256'))[:16]}, not {b['manifest_sha256'][:16]}")
        results[(int(b["pair"]), int(b["rank"]))] = res
    rows, versions = _rows(out), json.loads((out / "versions.json").read_text())
    required = dd.get("required_manifest") is not None
    for row in rows:
        if row["arm"] == "control" or row["pair"] not in dd["pending_pairs"] or row["commit"].get("commit_pass") is not None:
            continue
        pair, per_rank = int(row["pair"]), row["commit"]["per_rank"]
        finp = out / f"finalize_pair{pair}.json"
        fin = json.loads(finp.read_text())
        xr = dd["xrank"].get(str(pair)) or {}
        reasons = []
        for rk in sorted(per_rank, key=int):
            c = per_rank[rk]
            if c.get("commit_pass") is None and "replay_deferred" in c:
                res = results[(pair, int(rk))]
                sr, lk = res["sampled_replay"], res["boundary_linkage"]
                C2.annotate_unreplayable([sr], xr, pair, set(dd["xrank_families"]))
                ok, why = c["replay_deferred"]["pre"]
                ok, why = C2.replay_gates(c, sr, lk, ok, why, rank=int(rk), replay_on=True, required=required,
                                          xrank_families=set(dd["xrank_families"]) if xr.get("result") is not None else None)
                c["commit_pass"], c["commit_fail_reason"] = ok, why
                c["replay_deferred"]["replayed"] = {k: res.get(k) for k in ("where", "host", "manifest_sha256", "wall_s", "peak_rss_bytes", "utc", "weights_attest")}
                f_ = next(x for x in fin if int(x["rank"]) == int(rk))
                f_["sampled_replay"] = dict(f_.get("sampled_replay") or {}, sampled_replay=sr, boundary_linkage=lk, wall_s=res["wall_s"], where="cpu")
            if c.get("commit_pass") is False:
                reasons.append(f"rank {rk}: {c.get('commit_fail_reason')}")
        row["commit"]["commit_pass"] = not reasons
        row["commit"]["commit_fail_reason"] = "PASS" if not reasons else "; ".join(reasons)
        C2.write_atomic(finp, json.dumps(fin, indent=1, default=str) + "\n")
        covp = out / ("coverage.json" if pair == 0 else f"coverage_pair{pair}.json")
        if (reasons or pair == 0) and covp.is_file():
            cj = json.loads(covp.read_text())
            cj.pop("replay_pending_ranks", None)
            if reasons:
                cj["verdict"] = cj["verdict"] if cj.get("verdict") == "NOT_RUN" else "FAIL"
                cj["identity_coverage"] = None if cj.get("identity_coverage") is None else False
                cj["reason"] = "; ".join(reasons)
            C2.write_atomic(covp, json.dumps(cj, indent=1, default=str) + "\n")
        elif pair > 0:
            covp.unlink(missing_ok=True)
        if reasons:
            print(f"[tp2commit] COMMIT FAIL (pair {pair}): {row['commit']['commit_fail_reason']}", flush=True)
    sr_comp, lk_comp = C2.sampled_replay_components(rows, versions)
    qp_comp = C2.query_population_component(rows, True)
    covp = out / "coverage.json"
    if covp.is_file():
        cj = json.loads(covp.read_text())
        cm = dd["components"]
        cj["value_correspondence"] = C2.annotate_value_correspondence(json.loads(json.dumps(dd.get("vc_base"))), sr_comp, lk_comp, qp_comp, cm["weights_pin"],
                                                                   cm["cross_rank_collectives"], cm["fold_match_binding"], replay_on=True)
        C2.write_atomic(covp, json.dumps(cj, indent=1, default=str) + "\n")
    srp = out / "sampled_replay.json"
    if srp.is_file():
        sj = json.loads(srp.read_text())
        sj["sampled_replay"], sj["boundary_linkage"] = sr_comp, lk_comp
        C2.write_atomic(srp, json.dumps(sj, indent=1, default=str) + "\n")
    trt = [r for r in rows if r["arm"] != "control"]
    summ = json.loads((out / "summary.json").read_text())
    summ["commit_pass"] = [(r.get("commit") or {}).get("commit_pass") for r in trt]
    summ["sampled_replay"] = {k: sr_comp.get(k) for k in C2.SR_SUMMARY_KEYS}
    summ["boundary_linkage"], summ["query_population"] = lk_comp, qp_comp
    all_pass = bool(dd["all_pass"]) and all(c is True for c in summ["commit_pass"]) and qp_comp.get("result") is not False
    summ["pass"] = all_pass and all(r["tokens_equal_ref"] for r in rows)
    if summ.get("only_arm"):
        summ.update(TPC.only_arm_summary(summ["only_arm"], rows, trt, summ["commit_pass"]))
    replayed = [{k: res.get(k) for k in ("pair", "rank", "bundle", "manifest_sha256", "where", "host", "wall_s", "peak_rss_bytes", "utc")}
                for _k, res in sorted(results.items())]
    summ.setdefault("replay_deferred", {}).update(pending_pairs=[], replayed=replayed)
    C2.write_atomic(out / "runs.jsonl", "".join(json.dumps(r, default=str) + "\n" for r in rows))
    C2.write_atomic(out / "summary.json", json.dumps(summ, indent=1, default=str) + "\n")
    dd.update(pending_pairs=[], replayed=replayed, decided={"pass": summ["pass"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    C2.write_atomic(out / C2.DEFERRED, json.dumps(dd, indent=1, default=str) + "\n")
    print(f"[tp2commit] {'PASS' if summ['pass'] else 'FAIL'} (decided on CPU from {len(results)} rank replays): sampled replay {sr_comp.get('result')}, "
          f"boundary linkage {lk_comp.get('result')}, C2 population {qp_comp.get('result')}; {sr_comp.get('why') or ''}", flush=True)
    return summ["pass"]


def main(o: Options) -> int:
    from verity_vllm.check.replay.driver import PoolDoesNotFit
    from verity_vllm.program.compiled_relations import warm_vml

    out = Path(o.commit_out)
    try:
        if not (out / C2.DEFERRED).is_file():
            raise Refused(f"{out / C2.DEFERRED}: not a deferred TP Commit (tp-commit --replay-deferred) or decided in the Commit")
        dd = json.loads((out / C2.DEFERRED).read_text())
        if dd.get("decided"):
            print(f"[tp-replay] already decided: pass {dd['decided'].get('pass')} ({dd['decided'].get('utc')})", flush=True)
            return 0 if dd["decided"].get("pass") else 3
        want = {int(x) for x in o.ranks.split(",")} if o.ranks else None
        todo = [(int(b["pair"]), int(b["rank"])) for b in dd["bundles"]
                if (want is None or int(b["rank"]) in want) and not result_path(out, b["pair"], b["rank"]).is_file()]
        if todo and not o.fold_only:
            warm_vml()
            ck = functools.cache(lambda: checkpoint_index(o))
            for pair, rank in todo:
                replay_rank(o, out, dd, pair, rank, ck)
        ok = fold(out)
    except (Refused, PoolDoesNotFit) as e:
        print(f"[tp-replay] REFUSED: {e}", flush=True)
        return 2
    sys.stdout.flush()
    if ok is None:
        left = [(int(b["pair"]), int(b["rank"])) for b in dd["bundles"] if not result_path(out, b["pair"], b["rank"]).is_file()]
        print(f"[tp-replay] PENDING: (pair, rank) {left} have no replay on the record yet", flush=True)
        return 4
    return 0 if ok else 3
