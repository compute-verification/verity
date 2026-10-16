"""[R17 tp D90, COORD R17-4] The PER-RANK FOLD MATCH of a tensor-parallel row: each rank's observer log folded with the rank profile (the
received collective value = an input of a rank-local component, `TpCollective`), compared to THAT rank's Program exactly as `row_pod.sh`
compares a TP1 row (cb_a.compare at B1 / batch_decomp X-09 at B>1, GM-01 global_match over the rank's GP-01 workload Program), then the
CROSS-RANK leg (`tp.rank_match`) over both folds + the collective recorder rows of the SAME engine run + the rank Programs' `tp_links.json`
+ tp2_match's stored-step provenance.  CPU only.  The rank-local legs of the ranks run CONCURRENTLY (one subprocess chain per rank,
--rank-workers, default all): they are independent until the cross-rank leg; the record is assembled in rank order.

    verity-vllm tp-fold-match --capture <tp2_capture --record-collectives out> --profile derived_LLAMA32_1B_tp2 \\
        --build-rank0 <build_request_r0> --build-rank1 <build_request_r1> [--gprog-rank0 <build_workload_r0/workload_program.json> --gprog-rank1 ...] \\
        [--derived-rank0 DIR ... --derived-rank1 DIR ...] --workload WL [--tp2-match <run>/tp2_match.json] --out <D/fold_match> [--python PY]

Verdict (`<out>/tp2_fold_match.json`): PASS iff BOTH ranks fold with 0 errors / 0 unresolved, BOTH rank compares PASS (the TP1 rule:
canonical-equal, same steps, no differing operand; X-09 verdict at B>1), GM-01 PASS on EVERY rank (the leg is required: no --gprog = FAIL, rev F-r17-04), and the cross-rank leg PASSes.  A
missing partner rank, a missing recorder or a missing provenance record is a named FAIL, never a skipped leg (rev M-0026 R2): with the peers
parameter as a free input a rank-local Match alone would be trivially satisfiable.
"""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.check.match import rank_match as RM
from verity_vllm.config import UsageError, option

SCHEMA = "verity/tp2/fold-match/v1"


def _run(cmd: list[str], log: Path, timeout: int | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _load_links(p: Path) -> dict | None:
    """[lane vllm-tp-n] a rank's tp_links.json through `read_tp_links` (v1 -> v2 in memory; every link carries `peers` and `module`, the
    recorder-vocabulary module path from the derive's recorded model root)."""
    try:
        from verity_vllm.pipeline.build import read_tp_links
        return read_tp_links(str(p))
    except Exception:  # noqa: BLE001
        return None


def _link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def fold_rank(py: str, log: Path, profile: str, out: Path, timeout: int | None, reuse: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = reuse and all((out / f).exists() for f in ("fold_summary.json", "program.json", "instances.jsonl"))
    rc = 0 if done else _run([py, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", str(log), "--profile", profile,
                              "--out", str(out / "fold_summary.json"), "--persist-dir", str(out)], out / "fold.log", timeout)
    fs = _load(out / "fold_summary.json") or {}
    ok = rc == 0 and (out / "program.json").exists() and (out / "instances.jsonl").exists() and int(fs.get("errors") or 0) == 0 and \
        int(fs.get("unresolved") or 0) == 0
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "ok": ok, "errors": fs.get("errors"), "unresolved": fs.get("unresolved"),
            "unsupported": fs.get("unsupported"), "n_instances": fs.get("n_instances"), "n_steps": fs.get("n_steps"), "digest": fs.get("digest"),
            "tp": (fs.get("fold") or {}).get("tp"), "collectives": len((fs.get("fold") or {}).get("collectives") or []),
            "collectives_by_kind": (fs.get("fold") or {}).get("collectives_by_kind"), "error_examples": (fs.get("error_examples") or [])[:3],
            "max_rss_gb": fs.get("max_rss_gb")}


def compare_rank(py: str, rank_dir: Path, build: Path, derived: list[str], within: str, timeout: int | None) -> dict:
    """The TP1 verdict rule of row_pod.sh: B1 cb_a.compare (canonical-equal, same steps, no differing operand); B>1 batch_decomp X-09."""
    t0 = time.time()
    if derived:
        cmd = [py, "-m", "verity_vllm.pipeline.cli", "batch-decomp", "--oracle", str(rank_dir), "--out", str(rank_dir / "match_decomp.json"),
               "--within-step", within]
        for d in [str(build)] + list(derived):
            cmd += ["--derived", d]
        rc = _run(cmd, rank_dir / "match_compare.log", timeout)
        dec = _load(rank_dir / "match_decomp.json") or {}
        reqs = dec.get("requests") or []
        verdict = "PASS" if dec.get("verdict") == "PASS" and rc == 0 else f"FAIL batch_decomp {dec.get('verdict')} rc={rc}: {'; '.join(dec.get('reasons') or [])[:300]}"
        return {"comparator": "cb_a.batch_decomp (X-09)", "rc": rc, "verdict": verdict, "seconds": round(time.time() - t0, 1),
                "requests": len(reqs), "canonical_equal_all": all(r.get("canonical_equal") is True for r in reqs) and bool(reqs),
                "operands_different": sum(int((r.get("totals") or {}).get("operands_different", 0) or 0) for r in reqs),
                "unattributed": (dec.get("attribution") or {}).get("unattributed"), "xreq": dec.get("xreq_total"), "missing_derived": dec.get("missing_derived"),
                "per_request": [{"request_id": r.get("request_id"), "LP": r.get("LP"), "T": r.get("T"), "derived": r.get("derived"),
                                 "canonical_equal": r.get("canonical_equal"), "steps": r.get("steps")} for r in reqs]}
    rc = _run([py, "-m", "verity_vllm.pipeline.cli", "program-compare", "--a", str(build), "--b", str(rank_dir), "--out", str(rank_dir / "match_compare.json")],
              rank_dir / "match_compare.log", timeout)
    doc = _load(rank_dir / "match_compare.json") or {}
    cmp = doc.get("compare") or {}            # cb-a/compare/v1: the step comparison sits under `compare` (row_pod reads the same field)
    tot = cmp.get("totals") or {}
    sa, sb = (cmp.get("a") or {}).get("steps"), (cmp.get("b") or {}).get("steps")
    canon = (cmp.get("canonical_sha256") or {}).get("equal")
    ok = rc == 0 and canon is True and sa == sb and int(tot.get("operands_different", 0) or 0) == 0
    return {"comparator": "cb_a.compare", "rc": rc, "seconds": round(time.time() - t0, 1),
            "verdict": "PASS" if ok else f"FAIL canonical_equal={canon} steps={sa}/{sb} operands_different={tot.get('operands_different')} rc={rc}",
            "canonical_equal": canon, "steps": [sa, sb], "totals": tot, "histogram_diff_T_normalised": len(doc.get("histogram_diff_T_normalised") or {}), "first_diff": (cmp.get("first_differences") or cmp.get("diffs") or [None])[0]}


def global_match_rank(py: str, rank_dir: Path, gprog: Path, workload: Path, within: str, timeout: int | None) -> dict:
    t0 = time.time()
    rc = _run([py, "-m", "verity_vllm.pipeline.cli", "global-match", "--gprog", str(gprog), "--workload", str(workload), "--oracle", str(rank_dir),
               "--out", str(rank_dir / "global_match.json"), "--within-step", within], rank_dir / "global_match.log", timeout)
    G = _load(rank_dir / "global_match.json") or {}
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "verdict": G.get("verdict") if G else {124: "TIMEOUT"}.get(rc, f"rc={rc}"),
            "ok": G.get("verdict") == "PASS" and rc == 0, "failed_checks": G.get("failed_checks"),
            "checks": {c.get("id"): c.get("status") for c in (G.get("checks") or [])}, "achieved_concurrency": G.get("achieved_concurrency"),
            "generated_lengths": {e.get("request_id"): e.get("generated_length") for e in (G.get("requests") or [])},
            "eos_truncated_requests": G.get("eos_truncated_requests"), "within_step_criterion": G.get("within_step_criterion")}


def links_for_rank(rank: int, build: Path, compare: dict, derived: list[str]) -> list[dict] | dict[str, list[dict]] | None:
    """B1: the rank Program's tp_links.json `links`.  B>1: {request_id: links of the request's Program} from the batch decomposition's
    per-request derived dir (each derived dir of a rank carries its own tp_links.json)."""
    if not derived:
        doc = _load_links(build / "tp_links.json")
        return list(doc["links"]) if doc and isinstance(doc.get("links"), list) else None
    out: dict[str, list[dict]] = {}
    for r in compare.get("per_request") or []:
        d = r.get("derived")
        if not d:
            return None
        doc = _load_links(Path(d) / "tp_links.json")
        if not doc:
            return None
        out[str(r["request_id"])] = list(doc["links"])
    return out or None


@dataclass(kw_only=True)
class Options:
    capture: str = option("--capture",
                          help="tp2_capture --record-collectives run dir: log.rank<r>.jsonl.gz, tokens.json, coll.rank<r>.jsonl")
    profile: str = option("--profile",
                          help="rank profile module (verity_vllm.engine.profiles.<name>), e.g. derived_LLAMA32_1B_tp2 (profiles.generic, world = W)")
    build_rank0: str | None = option("--build-rank0", default=None)
    build_rank1: str | None = option("--build-rank1", default=None)
    derived_rank0: list[str] = option("--derived-rank0", action="append", default=[],
                                      help="B>1: further request Programs of rank 0 (per served (LP, T))")
    derived_rank1: list[str] = option("--derived-rank1", action="append", default=[])
    gprog_rank0: str | None = option("--gprog-rank0", default=None, help="rank 0's GP-01 workload_program.json (GM-01 leg)")
    gprog_rank1: str | None = option("--gprog-rank1", default=None)
    build_rank: list[str] = option("--build-rank", action="append", default=[],
                                   help="rank r's Build dir (first request Program), repeated in rank order; world = count")
    derived_rank: list[str] = option("--derived-rank", action="append", default=[],
                                     help="R=DIR: a further request Program of rank R (B>1)")
    gprog_rank: list[str] = option("--gprog-rank", action="append", default=[],
                                   help="R=PATH: rank R's GP-01 workload_program.json")
    workload: str | None = option("--workload", default=None)
    tp2_match: str | None = option("--tp2-match", default=None,
                                   help="tp2_match.json of the same run (its `provenance` = the stored-step value leg); REQUIRED for PASS")
    coll_dir: str | None = option("--coll-dir", default=None, help="dir of coll.rank<r>.jsonl (default: --capture)")
    out: str = option("--out")
    python: str = option("--python", default=sys.executable,
                         help="interpreter for the fold / compare subprocesses (row_pod's PY312)")
    within_step: str = option("--within-step", default="dag")
    timeout: int | None = option("--timeout", type=int, default=None, help="per subprocess, seconds")
    world: int = option("--world", type=int, default=2)
    rank_workers: int = option("--rank-workers", type=int, default=0,
                               help="ranks whose fold/compare/GM-01 legs run concurrently (0 = all ranks; each leg is its own subprocess: B8 1024/128 fold 26 GB + GM-01 14 GB per rank)")
    reuse_fold: bool = option("--reuse-fold", action="store_true", default=False,
                              help="keep a rank's persisted fold (fold_summary.json, program.json, instances.jsonl under --out) and run only its compare and GM-01 legs")


def main(a: Options) -> int:
    cap, out = Path(a.capture), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    coll_dir = Path(a.coll_dir) if a.coll_dir else cap
    if a.build_rank:
        if a.build_rank0 or a.build_rank1:
            raise UsageError("give either --build-rank DIR (repeated, rank order) or --build-rank0/--build-rank1, not both")
        a.world = len(a.build_rank)
        builds = {r: Path(d) for r, d in enumerate(a.build_rank)}
        derived = {r: [] for r in builds}
        gprog: dict[int, str | None] = {r: None for r in builds}
        for spec in a.derived_rank:
            r, _, d = spec.partition("=")
            derived[int(r)].append(d)
        for spec in a.gprog_rank:
            r, _, g = spec.partition("=")
            gprog[int(r)] = g
    else:
        if not (a.build_rank0 and a.build_rank1):
            raise UsageError("--build-rank DIR (repeated, rank order) or --build-rank0 and --build-rank1 required")
        builds = {0: Path(a.build_rank0), 1: Path(a.build_rank1)}
        derived = {0: a.derived_rank0, 1: a.derived_rank1}
        gprog = {0: a.gprog_rank0, 1: a.gprog_rank1}
    if a.world < 2 or a.world != len(builds):
        raise UsageError(f"world {a.world} vs {len(builds)} rank Builds")
    t_start = time.time()
    res: dict[str, Any] = {"schema": SCHEMA, "capture": str(cap), "profile": a.profile, "world": a.world, "per_rank": {}, "problems": []}

    ranks = list(range(a.world))

    def run_rank(r: int) -> tuple[dict[str, Any], list[str]]:
        """The rank-local legs of ONE rank (fold -> compare -> GM-01), each a subprocess; returns (per_rank record, named problems).
        Ranks are independent until the cross-rank leg, so they run concurrently (--rank-workers); the record and the problem list
        are assembled in rank order afterwards, so the verdict document does not depend on which rank finished first."""
        log = cap / f"log.rank{r}.jsonl.gz"
        rd = out / f"rank{r}"
        pr: dict[str, Any] = {"log": str(log)}
        problems: list[str] = []
        if not log.exists():
            pr["fold"] = {"ok": False, "rc": None, "why": "partner rank missing: no observer log for this rank"}
            problems.append(f"rank {r}: partner rank missing (no {log.name})")
            return pr, problems
        rd.mkdir(parents=True, exist_ok=True)
        _link(rd / "capture" / "log.jsonl.gz", log)        # the TP1 Match dir layout the comparators read (<oracle>/capture/log.jsonl.gz, tokens.json)
        if (cap / "tokens.json").exists():
            _link(rd / "capture" / "tokens.json", cap / "tokens.json")
        pr["fold"] = fold_rank(a.python, log, a.profile, rd, a.timeout, a.reuse_fold)
        print(f"[tp2foldmatch] rank {r} fold: ok={pr['fold']['ok']} rc={pr['fold']['rc']} instances={pr['fold']['n_instances']} steps={pr['fold']['n_steps']} "
              f"errors={pr['fold']['errors']} unresolved={pr['fold']['unresolved']} unsupported={pr['fold']['unsupported']} collectives={pr['fold']['collectives']} "
              f"{pr['fold']['collectives_by_kind']} tp={pr['fold']['tp']} {pr['fold']['seconds']}s", flush=True)
        if not pr["fold"]["ok"]:
            problems.append(f"rank {r}: fold not ok (rc {pr['fold']['rc']}, errors {pr['fold']['errors']}, unresolved {pr['fold']['unresolved']}): "
                            f"{json.dumps(pr['fold']['error_examples'])[:300]}")
            return pr, problems
        tp = pr["fold"].get("tp") or {}
        if int(tp.get("rank", -1)) != r or int(tp.get("world", 0)) != a.world:
            problems.append(f"rank {r}: the log's RunHeader.notes.tp2 states rank {tp.get('rank')} / world {tp.get('world')} (swapped or mislabelled rank)")
        pr["compare"] = compare_rank(a.python, rd, builds[r], derived[r], a.within_step, a.timeout)
        print(f"[tp2foldmatch] rank {r} compare ({pr['compare']['comparator']}): {pr['compare']['verdict']} {pr['compare']['seconds']}s", flush=True)
        if pr["compare"]["verdict"] != "PASS":
            problems.append(f"rank {r}: compare {pr['compare']['verdict']}")
        if not gprog[r] or not a.workload:
            # [rev F-r17-04] the GM-01 leg is REQUIRED: a row without --gprog-rank<r> + --workload is not a PASS (no skipped leg, R2)
            problems.append(f"rank {r}: GM-01 leg not run (--gprog-rank{r} / --workload absent): the global Match over the rank's GP-01 workload Program is required")
        if gprog[r]:
            if not a.workload:
                problems.append("--gprog given without --workload")
            else:
                pr["global_match"] = global_match_rank(a.python, rd, Path(gprog[r]), Path(a.workload), a.within_step, a.timeout)
                print(f"[tp2foldmatch] rank {r} GM-01: {pr['global_match']['verdict']} failed={pr['global_match']['failed_checks']} "
                      f"concurrency={pr['global_match']['achieved_concurrency']} {pr['global_match']['seconds']}s", flush=True)
                if not pr["global_match"]["ok"]:
                    problems.append(f"rank {r}: GM-01 {pr['global_match']['verdict']} {pr['global_match']['failed_checks']}")
        return pr, problems

    workers = max(1, min(a.rank_workers or len(ranks), len(ranks)))
    res["rank_workers"] = workers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        per_rank = list(ex.map(run_rank, ranks))          # rank order, whatever the completion order
    for r, (pr, problems) in zip(ranks, per_rank):
        res["per_rank"][str(r)] = pr
        res["problems"].extend(problems)

    # ---- cross-rank leg
    folded = [r for r in ranks if (res["per_rank"][str(r)].get("fold") or {}).get("ok")]
    xr: dict[str, Any]
    if folded != ranks:
        xr = {"verdict": "FAIL", "failed": ["partner_rank"], "why": f"ranks folded {folded} of {ranks}: the cross-rank leg needs every rank's fold"}
    else:
        colls, producers, ferr, rec, links, rec_sha = {}, {}, {}, {}, {}, {}
        for r in ranks:
            rd = out / f"rank{r}"
            fs = _load(rd / "fold_summary.json") or {}
            colls[r] = list((fs.get("fold") or {}).get("collectives") or [])
            producers[r] = RM.own_producers_from_record(rd / "instances.jsonl")
            ferr[r] = int(fs.get("errors") or 0)
            p = coll_dir / f"coll.rank{r}.jsonl"
            if p.exists():
                rec[r] = RM.load_recorder(p)
                rec_sha[r] = hashlib.sha256(p.read_bytes()).hexdigest()      # [rev F-r17-03] binds the provenance record to THESE rows
            else:
                res["problems"].append(f"rank {r}: no collective recorder rows ({p}); the value leg cannot run")
            lk = links_for_rank(r, builds[r], res["per_rank"][str(r)].get("compare") or {}, derived[r])
            if lk is None:
                res["problems"].append(f"rank {r}: no tp_links.json for the rank Program(s); the link leg cannot run")
            else:
                links[r] = lk
        prov = None
        if a.tp2_match:
            tm = _load(Path(a.tp2_match))
            prov = (tm or {}).get("provenance")
            # [rev S-r17-07 n1] the consumed record, by digest: tp2_fold_match.json names WHICH tp2_match.json its provenance leg read
            try:
                res["tp2_match"] = {"path": str(a.tp2_match), "sha256": hashlib.sha256(Path(a.tp2_match).read_bytes()).hexdigest(),
                                    "recorder_sha256": (prov or {}).get("recorder_sha256")}
            except OSError as e:
                res["problems"].append(f"{a.tp2_match}: unreadable ({e})")
            if prov is None:
                res["problems"].append(f"{a.tp2_match}: no provenance record")
        else:
            res["problems"].append("no --tp2-match: the stored-step provenance leg (received == Def(part_rank0 .. part_rank{world-1})) is REQUIRED")
        xr = RM.cross_rank_check(colls, producers=producers, fold_errors=ferr, recorder=rec if len(rec) == len(ranks) else None,
                                 links=links if len(links) == len(ranks) else None, provenance=prov,
                                 recorder_sha256=rec_sha if len(rec_sha) == len(ranks) else None, require_all=True)
        if len(rec) != len(ranks):
            xr["failed"] = list(xr.get("failed") or []) + ["recorder_missing"]
            xr["verdict"] = "FAIL"
        if len(links) != len(ranks):
            xr["failed"] = list(xr.get("failed") or []) + ["links_missing"]
            xr["verdict"] = "FAIL"
        if prov is None:
            xr["failed"] = list(xr.get("failed") or []) + ["provenance_missing"]
            xr["verdict"] = "FAIL"
        (out / "tp2_rank_match.json").write_text(json.dumps(xr, indent=1, default=str) + "\n")
        print(f"[tp2foldmatch] cross-rank leg: {xr['verdict']} failed={xr.get('failed')} collectives={xr.get('n_collectives')} "
              f"checks={[(c['name'], c['pass']) for c in xr.get('checks') or []]}", flush=True)
    res["cross_rank"] = {k: v for k, v in xr.items() if k not in ("peer_binding",)}
    res["cross_rank"]["peer_binding_n"] = len(xr.get("peer_binding") or {})
    res["cross_rank"]["artifact"] = str(out / "tp2_rank_match.json")

    ok_ranks = all((res["per_rank"][str(r)].get("fold") or {}).get("ok") and (res["per_rank"][str(r)].get("compare") or {}).get("verdict") == "PASS"
                   and ((res["per_rank"][str(r)].get("global_match") or {}).get("ok") is True) for r in ranks)     # [F-r17-04] GM-01 None = not PASS
    res["pass"] = bool(ok_ranks and xr.get("verdict") == "PASS" and not res["problems"])
    res["seconds"] = round(time.time() - t_start, 1)
    res["legs"] = {"fold_per_rank": {str(r): (res["per_rank"][str(r)].get("fold") or {}).get("ok") for r in ranks},
                   "compare_per_rank": {str(r): (res["per_rank"][str(r)].get("compare") or {}).get("verdict") for r in ranks},
                   "global_match_per_rank": {str(r): (res["per_rank"][str(r)].get("global_match") or {}).get("verdict") for r in ranks},
                   "cross_rank": xr.get("verdict")}
    (out / "tp2_fold_match.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(f"[tp2foldmatch] {'PASS' if res['pass'] else 'FAIL'} legs={res['legs']} problems={res['problems'][:4]} {res['seconds']}s -> {out / 'tp2_fold_match.json'}", flush=True)
    return 0 if res["pass"] else 1
