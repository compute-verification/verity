"""The PER-RANK FOLD MATCH of a tensor-parallel row: each rank's observer log folded with the rank profile (the
received collective value = an input of a rank-local component, `TpCollective`), compared to THAT rank's Program exactly as `row_pod.sh`
compares a TP1 row (cb_a.compare at B1 / batch_decomp X-09 at B>1, under --shared-x09 the X-09 GM-01 computes; GM-01 global_match over
the rank's GP-01 workload Program), then the CROSS-RANK check (`tp.rank_match`) over both folds + the collective recorder rows of the SAME
engine run + the rank Programs' `tp_links.json`
+ tp2_match's stored-step provenance.  CPU only.  The rank-local stages of the ranks run CONCURRENTLY (one subprocess chain per rank,
--fold-workers ranks folding and --rank-workers in their compare and GM-01 stages at once, default all): they are independent until the
cross-rank check; the record is assembled in rank order.

    verity-vllm tp-fold-match --capture <tp2_capture --record-collectives out> --profile derived_LLAMA32_1B_tp2 \\
        --build-rank0 <build_request_r0> --build-rank1 <build_request_r1> [--gprog-rank0 <build_workload_r0/workload_program.json> --gprog-rank1 ...] \\
        --workload WL [--tp2-match <run>/tp2_match.json] --out <D/fold_match> [--python PY]

Verdict (`<out>/tp2_fold_match.json`): PASS iff BOTH ranks fold with 0 errors / 0 unresolved, BOTH rank compares PASS (the TP1 rule:
canonical-equal, same steps, no differing operand; X-09 verdict at B>1), GM-01 PASS on EVERY rank (required: no --gprog = FAIL), and the cross-rank check PASSes.  A
missing partner rank, a missing recorder or a missing provenance record is a named FAIL, never a skipped stage: with the peers
parameter as a free input a rank-local Match alone would be trivially satisfiable.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.check.match import rank_match as RM
from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.log import init_logger
from verity_vllm.pipeline.tp.links import read_tp_links
from verity_vllm.records.schema_ids import TP_FOLD_MATCH_SCHEMA

__all__ = ["Options", "main"]

logger = init_logger(__name__)

SCHEMA = TP_FOLD_MATCH_SCHEMA


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


def _run_unless(reuse: bool, out: Path, cmd: list[str], log: Path, timeout: int | None) -> int:
    """`_run`, except under --reuse-legs a leg whose document `out` is already written is kept (rc 0); one a killed leg cut short,
    which does not load, runs again."""
    return 0 if reuse and _load(out) is not None else _run(cmd, log, timeout)


def _load(p: Path) -> dict | None:
    """The JSON document at ``p``; None when the file is missing or does not parse (one nested past the parser's depth does not)."""
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError, RecursionError):
        return None


def _block(doc: dict, name: str) -> dict:
    """The block ``name`` of a document: a stage's own (a fold summary, a compare, a decomposition), one of its entries, a rank's
    record or the cross-rank check's.  Empty for one that holds none: a stage that wrote no document (it was killed, timed out or
    failed first) loads as an empty one, a fold summary may hold no `fold` statistics, and a rank whose fold failed has no `compare`
    and no `global_match`.  Each field read off the missing block is then None."""
    block = doc.get(name)
    return block if block else {}


def _listed(doc: dict, name: str) -> list:
    """The list ``name`` of a document; empty for one that lists none, and for a stage that wrote no document."""
    listed = doc.get(name)
    return listed if listed else []


def _load_links(p: Path) -> dict | None:
    """a rank's tp_links.json through `read_tp_links` (v1 -> v2 in memory; every link carries `peers` and `module`, the
    recorder-vocabulary module path from the derive's recorded model root); None when the file is missing or is not one (a document
    nested past the parser's depth, or one that counts a world, a rank or a collective with an infinite number, is not)."""
    try:
        return read_tp_links(str(p))
    except (OSError, ValueError, TypeError, AttributeError, RecursionError, OverflowError):
        return None


def _link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def _folded_under(fs: dict, construction: list[str]) -> bool:
    """Whether the fold `fs` (a fold_summary.json) was folded under the worker flags `construction` (`match.fold_construction_args`)."""
    def flag(f: str) -> str | None:
        return construction[construction.index(f) + 1] if f in construction else None
    return (fs.get("moe_construction"), fs.get("sampler_construction"), bool(fs.get("registered_values"))) == \
        (flag("--moe-construction"), flag("--sampler-construction"), "--registered-values" in construction)


def fold_rank(py: str, log: Path, profile: str, out: Path, timeout: int | None, reuse: bool = False, construction: Sequence[str] = ()) -> dict:
    """The rank's fold under `construction`, the fold worker's flags for the rank's Build (`match.fold_construction_args`); --reuse-fold
    keeps a persisted fold only if it was folded under the same flags."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    construction = list(construction)
    done = reuse and all((out / f).exists() for f in ("fold_summary.json", "program.json", "instances.jsonl")) and \
        _folded_under(_load(out / "fold_summary.json") or {}, construction)
    rc = 0 if done else _run([py, "-m", "verity_vllm.pipeline.cli", "match", "--worker", "fold", "--log", str(log), "--profile", profile,
                              "--out", str(out / "fold_summary.json"), "--persist-dir", str(out), *construction], out / "fold.log", timeout)
    fs = _load(out / "fold_summary.json") or {}
    stats = _block(fs, "fold")
    ok = rc == 0 and (out / "program.json").exists() and (out / "instances.jsonl").exists() and int(fs.get("errors") or 0) == 0 and \
        int(fs.get("unresolved") or 0) == 0
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "ok": ok, "kept": done, "errors": fs.get("errors"), "unresolved": fs.get("unresolved"),
            "unsupported": fs.get("unsupported"), "n_instances": fs.get("n_instances"), "n_steps": fs.get("n_steps"), "digest": fs.get("digest"),
            "tp": stats.get("tp"), "collectives": len(_listed(stats, "collectives")),
            "collectives_by_kind": stats.get("collectives_by_kind"), "error_examples": _listed(fs, "error_examples")[:3],
            "max_rss_gb": fs.get("max_rss_gb"),
            "construction": {k: fs.get(k) for k in ("sampler_construction", "moe_construction", "registered_values")}}


def compare_rank(py: str, rank_dir: Path, build: Path, derived: list[str], within: str, timeout: int | None, reuse: bool = False,
                 shared: bool = False) -> dict:
    """The TP1 verdict rule of row_pod.sh: B1 cb_a.compare (canonical-equal, same steps, no differing operand); B>1 batch_decomp X-09,
    under `shared` the document GM-01 wrote (`global_match_rank`'s --decomp-out) read as row_pod's shared pipeline reads it."""
    t0 = time.time()
    if derived:
        if shared:
            # Read here, not at import: only a row whose ranks share one Build reads the row's records, and this module imports without them.
            from verity_vllm.pipeline.row_records import decomp_rc
            rc = decomp_rc(str(rank_dir))
        else:
            cmd = [py, "-m", "verity_vllm.pipeline.cli", "batch-decomp", "--oracle", str(rank_dir), "--out", str(rank_dir / "match_decomp.json"),
                   "--within-step", within]
            for d in [str(build)] + list(derived):
                cmd += ["--derived", d]
            rc = _run_unless(reuse, rank_dir / "match_decomp.json", cmd, rank_dir / "match_compare.log", timeout)
        dec = _load(rank_dir / "match_decomp.json") or {}
        reqs = _listed(dec, "requests")
        verdict = "PASS" if dec.get("verdict") == "PASS" and rc == 0 else f"FAIL batch_decomp {dec.get('verdict')} rc={rc}: {'; '.join(_listed(dec, 'reasons'))[:300]}"
        return {"comparator": "cb_a.global_match --decomp-out (X-09)" if shared else "cb_a.batch_decomp (X-09)", "rc": rc, "verdict": verdict,
                "seconds": round(time.time() - t0, 1),
                "requests": len(reqs), "canonical_equal_all": all(r.get("canonical_equal") is True for r in reqs) and bool(reqs),
                "operands_different": sum(int(_block(r, "totals").get("operands_different", 0) or 0) for r in reqs),
                "unattributed": _block(dec, "attribution").get("unattributed"), "xreq": dec.get("xreq_total"), "missing_derived": dec.get("missing_derived"),
                "per_request": [{"request_id": r.get("request_id"), "LP": r.get("LP"), "T": r.get("T"), "derived": r.get("derived"),
                                 "canonical_equal": r.get("canonical_equal"), "steps": r.get("steps")} for r in reqs]}
    rc = _run_unless(reuse, rank_dir / "match_compare.json", [py, "-m", "verity_vllm.pipeline.cli", "program-compare", "--a", str(build), "--b", str(rank_dir),
                     "--out", str(rank_dir / "match_compare.json"), "--operands", "progressions"], rank_dir / "match_compare.log", timeout)
    doc = _load(rank_dir / "match_compare.json") or {}
    cmp = _block(doc, "compare")              # cb-a/compare/v1: the step comparison sits under `compare` (row_pod reads the same field)
    tot = _block(cmp, "totals")
    sa, sb = _block(cmp, "a").get("steps"), _block(cmp, "b").get("steps")
    canon = _block(cmp, "canonical_sha256").get("equal")
    dag_doc = None
    if canon is not True and within == "dag" and rc == 0:     # the within-step criterion of record: a step order may differ
        _run_unless(reuse, rank_dir / "match_dag.json", [py, "-m", "verity_vllm.pipeline.cli", "program-dag", "--a", str(build), "--b", str(rank_dir),
                    "--out", str(rank_dir / "match_dag.json")], rank_dir / "match_dag.log", timeout)
        dag_doc = _load(rank_dir / "match_dag.json")
    dag = (dag_doc or {}).get("equal") is True
    ok = rc == 0 and (canon is True or dag) and sa == sb and int(tot.get("operands_different", 0) or 0) == 0
    return {"comparator": "cb_a.compare", "rc": rc, "seconds": round(time.time() - t0, 1),
            "verdict": "PASS" if ok else f"FAIL canonical_equal={canon} dag_multiset_equal={dag} steps={sa}/{sb} operands_different={tot.get('operands_different')} rc={rc}",
            "criterion": "sequence" if canon is True else "dag-multiset over the live cone (within-step criterion of record)",
            "dag_multiset": dag_doc, "canonical_equal": canon, "steps": [sa, sb], "totals": tot, "histogram_diff_T_normalised": len(_block(doc, "histogram_diff_T_normalised")), "first_diff": (cmp.get("first_differences") or cmp.get("diffs") or [None])[0]}


def global_match_rank(py: str, rank_dir: Path, gprog: Path, workload: Path, within: str, timeout: int | None, reuse: bool = False,
                      x09: list[str] | None = None) -> dict:
    """GM-01 over the rank's GP-01 Program; with `x09` (the rank's request Program dirs) it also writes the X-09 document it computes
    (--decomp-out match_decomp.json, those dirs as --derived), so a B>1 rank decomposes its fold once (row_pod's shared pipeline)."""
    t0 = time.time()
    shared = [] if x09 is None else ["--decomp-out", str(rank_dir / "match_decomp.json"), *(a for d in x09 for a in ("--derived", d))]
    keep = reuse and (x09 is None or _load(rank_dir / "match_decomp.json") is not None)
    rc = _run_unless(keep, rank_dir / "global_match.json", [py, "-m", "verity_vllm.pipeline.cli", "global-match", "--gprog", str(gprog), "--workload", str(workload), "--oracle", str(rank_dir),
               "--out", str(rank_dir / "global_match.json"), "--within-step", within, *shared], rank_dir / "global_match.log", timeout)
    G = _load(rank_dir / "global_match.json") or {}
    return {"rc": rc, "seconds": round(time.time() - t0, 1), "verdict": G.get("verdict") if G else {124: "TIMEOUT"}.get(rc, f"rc={rc}"),
            "ok": G.get("verdict") == "PASS" and rc == 0, "failed_checks": G.get("failed_checks"),
            "checks": {c.get("id"): c.get("status") for c in _listed(G, "checks")}, "achieved_concurrency": G.get("achieved_concurrency"),
            "generated_lengths": {e.get("request_id"): e.get("generated_length") for e in _listed(G, "requests")},
            "eos_truncated_requests": G.get("eos_truncated_requests"), "within_step_criterion": G.get("within_step_criterion")}


def links_for_rank(rank: int, build: Path, compare: dict, derived: list[str]) -> list[dict] | dict[str, list[dict]] | None:
    """B1: the rank Program's tp_links.json `links`.  B>1: {request_id: links of the request's Program} from the batch decomposition's
    per-request derived dir (each derived dir of a rank carries its own tp_links.json), read off the rank's compare record
    (`compare_rank`'s `per_request`; a record that lists none has no links)."""
    if not derived:
        doc = _load_links(build / "tp_links.json")
        return list(doc["links"]) if doc and isinstance(doc.get("links"), list) else None
    out: dict[str, list[dict]] = {}
    for r in _listed(compare, "per_request"):
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
                          help="rank profile module (verity_vllm.observe.fold.profiles.<name>), e.g. derived_<ROLE>_tp<W> (profiles.generic, world = W)")
    build_rank0: str | None = option("--build-rank0", default=None)
    build_rank1: str | None = option("--build-rank1", default=None)
    gprog_rank0: str | None = option("--gprog-rank0", default=None, help="rank 0's GP-01 workload_program.json (GM-01 stage)")
    gprog_rank1: str | None = option("--gprog-rank1", default=None)
    build_rank: list[str] = option("--build-rank", action="append", default=[],
                                   help="rank r's Build dir (first request Program), repeated in rank order; world = count")
    derived_rank: list[str] = option("--derived-rank", action="append", default=[],
                                     help="R=DIR: a further request Program of rank R (B>1)")
    gprog_rank: list[str] = option("--gprog-rank", action="append", default=[],
                                   help="R=PATH: rank R's GP-01 workload_program.json")
    workload: str | None = option("--workload", type=workload_arg, default=None)
    tp2_match: str | None = option("--tp2-match", default=None,
                                   help="tp2_match.json of the same run (its `provenance` = the stored-step value check); REQUIRED for PASS")
    out: str = option("--out")
    python: str = option("--python", default=sys.executable,
                         help="interpreter for the fold / compare subprocesses (row_pod's PY312)")
    within_step: str = option("--within-step", default="dag")
    timeout: int | None = option("--timeout", type=int, default=None, help="per subprocess, seconds")
    world: int = option("--world", type=int, default=2)
    rank_workers: int = option("--rank-workers", type=int, default=0,
                               help="ranks whose fold/compare/GM-01 stages run concurrently (0 = all ranks; each stage is its own subprocess: B8 1024/128 fold 26 GB + GM-01 14 GB per rank)")
    fold_workers: int = option("--fold-workers", type=int, default=0,
                               help="ranks whose fold stage runs concurrently (0 = as --rank-workers, which then bounds only the compare and GM-01 stages: "
                                    "a GLM-5.3 B8 I256 rank's fold peaks at 40 GB, its GM-01 with the shared X-09 near 400 GB)")
    reuse_fold: bool = option("--reuse-fold", action="store_true", default=False,
                              help="keep a rank's persisted fold (fold_summary.json, program.json, instances.jsonl under --out), if it was folded under the rank Build's "
                                   "construction, and run only its compare and GM-01 stages")
    reuse_legs: bool = option("--reuse-legs", action="store_true", default=False,
                              help="also keep a rank's written compare (X-09 at B>1) / DAG / GM-01 documents under --out, where its fold was kept, so only the "
                                   "missing stages and the cross-rank check run")
    shared_x09: bool = option("--shared-x09", action="store_true", default=False,
                              help="B>1: a rank's X-09 is the one its GM-01 computes and writes (global-match --decomp-out, row_pod's MATCH_PIPELINE=shared), "
                                   "not a second batch-decomp process over the same fold under the audit checker's operations (past 800 GB on a "
                                   "GLM-5.3 B8 rank)")


def _stage(per_rank: dict[str, dict], r: int, name: str) -> dict:
    """Rank r's record of its `fold`, its `compare` or its `global_match`; empty for a rank whose fold failed, which ran neither of
    the last two, and for the `global_match` of a rank given no GP-01 Program or no workload."""
    return _block(per_rank[str(r)], name)


def _fails_too(xr: dict[str, Any], name: str) -> None:
    """Fails the cross-rank check ``xr`` by ``name``, an input it was not given, beside the checks it failed itself (it lists none
    when it stopped at the ranks or at a rank's duplicate collective)."""
    xr["failed"] = list(_listed(xr, "failed")) + [name]
    xr["verdict"] = "FAIL"


def _rank_legs(a: Options, r: int, rd: Path, build: Path, derived: list[str], gprog: str | None, pr: dict[str, Any], problems: list[str],
               reuse_legs: bool = False) -> None:
    """The checks of folded rank `r` after its fold, into its record `pr` and `problems`: the compare (X-09 at B>1) and GM-01.  `reuse_legs`
    keeps their written documents (--reuse-legs), only for a rank whose fold was kept: a refolded rank's checks read the new fold."""
    def gm01(x09: list[str] | None = None) -> dict:
        assert gprog is not None and a.workload is not None
        g = global_match_rank(a.python, rd, Path(gprog), Path(a.workload), a.within_step, a.timeout, reuse_legs, x09)
        logger.info("[tp2foldmatch] rank %s GM-01: %s failed=%s concurrency=%s %ss", r, g["verdict"], g["failed_checks"], g["achieved_concurrency"],
                    g["seconds"])
        return g

    shared = a.shared_x09 and bool(derived) and bool(gprog) and bool(a.workload)
    if shared:     # GM-01 first: its X-09 document is the compare's
        pr["global_match"] = gm01([str(build), *derived])
    pr["compare"] = compare_rank(a.python, rd, build, derived, a.within_step, a.timeout, reuse_legs, shared)
    logger.info("[tp2foldmatch] rank %s compare (%s): %s %ss", r, pr["compare"]["comparator"], pr["compare"]["verdict"], pr["compare"]["seconds"])
    if pr["compare"]["verdict"] != "PASS":
        problems.append(f"rank {r}: compare {pr['compare']['verdict']}")
    if not gprog or not a.workload:
        # the GM-01 stage is REQUIRED: a row without --gprog-rank<r> + --workload is not a PASS (no skipped stage)
        problems.append(f"rank {r}: GM-01 stage not run (--gprog-rank{r} / --workload absent): the global Match over the rank's GP-01 workload Program is required")
    if gprog:
        if not a.workload:
            problems.append("--gprog given without --workload")
        else:
            if not shared:
                pr["global_match"] = gm01()
            if not pr["global_match"]["ok"]:
                problems.append(f"rank {r}: GM-01 {pr['global_match']['verdict']} {pr['global_match']['failed_checks']}")


def main(a: Options) -> int:
    cap, out = Path(a.capture), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    derived: dict[int, list[str]]
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
        derived = {0: [], 1: []}
        gprog = {0: a.gprog_rank0, 1: a.gprog_rank1}
    if a.world < 2 or a.world != len(builds):
        raise UsageError(f"world {a.world} vs {len(builds)} rank Builds")
    t_start = time.time()
    res: dict[str, Any] = {"schema": SCHEMA, "capture": str(cap), "profile": a.profile, "world": a.world, "per_rank": {}, "problems": []}

    ranks = list(range(a.world))
    # Read here, not at import: `pipeline.match` loads the core `verity`, and this module imports without numpy, torch or the core.
    from verity_vllm.pipeline.match import fold_construction_args
    construction = {r: fold_construction_args(a.workload, [str(builds[r]), *derived[r]]) for r in ranks}

    def run_rank(r: int) -> tuple[dict[str, Any], list[str]]:
        """The rank-local stages of ONE rank (fold -> compare -> GM-01), each a subprocess; returns (per_rank record, named problems).
        Ranks are independent until the cross-rank check, so they run concurrently (--fold-workers in the fold, --rank-workers in the
        stages after it); the record and the problem list are assembled in rank order afterwards, so the verdict document does not depend
        on which rank finished first."""
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
        with fold_slots:
            pr["fold"] = fold_rank(a.python, log, a.profile, rd, a.timeout, a.reuse_fold, construction=construction[r])
        fold = pr["fold"]
        logger.info("[tp2foldmatch] rank %s fold: ok=%s rc=%s instances=%s steps=%s errors=%s unresolved=%s unsupported=%s collectives=%s %s tp=%s %ss",
                    r, fold["ok"], fold["rc"], fold["n_instances"], fold["n_steps"], fold["errors"], fold["unresolved"], fold["unsupported"],
                    fold["collectives"], fold["collectives_by_kind"], fold["tp"], fold["seconds"])
        if not pr["fold"]["ok"]:
            problems.append(f"rank {r}: fold not ok (rc {pr['fold']['rc']}, errors {pr['fold']['errors']}, unresolved {pr['fold']['unresolved']}): "
                            f"{json.dumps(pr['fold']['error_examples'])[:300]}")
            return pr, problems
        tp = _block(pr["fold"], "tp")       # what the log's run header states of its rank; a log that states none is no rank's
        if int(tp.get("rank", -1)) != r or int(tp.get("world", 0)) != a.world:
            problems.append(f"rank {r}: the log's RunHeader.notes.tp2 states rank {tp.get('rank')} / world {tp.get('world')} (swapped or mislabelled rank)")
        with leg_slots:
            _rank_legs(a, r, rd, builds[r], derived[r], gprog[r], pr, problems, reuse_legs=a.reuse_legs and bool(pr["fold"].get("kept")))
        return pr, problems

    workers = max(1, min(a.rank_workers or len(ranks), len(ranks)))
    folds = max(1, min(a.fold_workers or workers, len(ranks)))
    fold_slots, leg_slots = threading.BoundedSemaphore(folds), threading.BoundedSemaphore(workers)
    res["rank_workers"], res["fold_workers"] = workers, folds
    with ThreadPoolExecutor(max_workers=max(workers, folds)) as ex:
        per_rank = list(ex.map(run_rank, ranks))          # rank order, whatever the completion order
    for rank, (pr, problems) in zip(ranks, per_rank):
        res["per_rank"][str(rank)] = pr
        res["problems"].extend(problems)

    # ---- cross-rank check
    folded = [r for r in ranks if _stage(res["per_rank"], r, "fold").get("ok")]
    xr: dict[str, Any]
    if folded != ranks:
        xr = {"verdict": "FAIL", "failed": ["partner_rank"], "why": f"ranks folded {folded} of {ranks}: the cross-rank check needs every rank's fold"}
    else:
        colls, producers, ferr, rec, links, rec_sha = {}, {}, {}, {}, {}, {}
        for rank in ranks:
            rd = out / f"rank{rank}"
            fs = _load(rd / "fold_summary.json") or {}
            colls[rank] = list(_listed(_block(fs, "fold"), "collectives"))
            producers[rank] = RM.own_producers_from_record(rd / "instances.jsonl")
            ferr[rank] = int(fs.get("errors") or 0)
            p = cap / f"coll.rank{rank}.jsonl"
            if p.exists():
                rec[rank] = RM.load_recorder(p)
                rec_sha[rank] = hashlib.sha256(p.read_bytes()).hexdigest()      # binds the provenance record to THESE rows
            else:
                res["problems"].append(f"rank {rank}: no collective recorder rows ({p}); the value check cannot run")
            lk = links_for_rank(rank, builds[rank], _stage(res["per_rank"], rank, "compare"), derived[rank])
            if lk is None:
                res["problems"].append(f"rank {rank}: no tp_links.json for the rank Program(s); the link check cannot run")
            else:
                links[rank] = lk
        prov = None
        if a.tp2_match:
            tm = _load(Path(a.tp2_match))
            prov = (tm or {}).get("provenance")
            # the consumed record, by digest: tp2_fold_match.json names WHICH tp2_match.json its provenance check read
            try:
                res["tp2_match"] = {"path": str(a.tp2_match), "sha256": hashlib.sha256(Path(a.tp2_match).read_bytes()).hexdigest(),
                                    "recorder_sha256": (prov or {}).get("recorder_sha256")}
            except OSError as e:
                res["problems"].append(f"{a.tp2_match}: unreadable ({e})")
            if prov is None:
                res["problems"].append(f"{a.tp2_match}: no provenance record")
        else:
            res["problems"].append("no --tp2-match: the stored-step provenance check (received == Def(part_rank0 .. part_rank{world-1})) is REQUIRED")
        xr = RM.cross_rank_check(colls, producers=producers, fold_errors=ferr, recorder=rec if len(rec) == len(ranks) else None,
                                 links=links if len(links) == len(ranks) else None, provenance=prov,
                                 recorder_sha256=rec_sha if len(rec_sha) == len(ranks) else None, require_all=True)
        if len(rec) != len(ranks):
            _fails_too(xr, "recorder_missing")
        if len(links) != len(ranks):
            _fails_too(xr, "links_missing")
        if prov is None:
            _fails_too(xr, "provenance_missing")
        (out / "tp2_rank_match.json").write_text(json.dumps(xr, indent=1, default=str) + "\n")
        logger.info("[tp2foldmatch] cross-rank check: %s failed=%s collectives=%s checks=%s", xr["verdict"], xr.get("failed"), xr.get("n_collectives"),
                    [(c["name"], c["pass"]) for c in _listed(xr, "checks")])
    res["cross_rank"] = {k: v for k, v in xr.items() if k not in ("peer_binding",)}
    res["cross_rank"]["peer_binding_n"] = len(_block(xr, "peer_binding"))        # a check that did not reach its peer binding bound none
    res["cross_rank"]["artifact"] = str(out / "tp2_rank_match.json")

    ok_ranks = all(_stage(res["per_rank"], r, "fold").get("ok") and _stage(res["per_rank"], r, "compare").get("verdict") == "PASS"
                   and (_stage(res["per_rank"], r, "global_match").get("ok") is True) for r in ranks)     # GM-01 None = not PASS
    res["pass"] = bool(ok_ranks and xr.get("verdict") == "PASS" and not res["problems"])
    res["seconds"] = round(time.time() - t_start, 1)
    res["legs"] = {"fold_per_rank": {str(r): _stage(res["per_rank"], r, "fold").get("ok") for r in ranks},
                   "compare_per_rank": {str(r): _stage(res["per_rank"], r, "compare").get("verdict") for r in ranks},
                   "global_match_per_rank": {str(r): _stage(res["per_rank"], r, "global_match").get("verdict") for r in ranks},
                   "cross_rank": xr.get("verdict")}
    (out / "tp2_fold_match.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    logger.info("[tp2foldmatch] %s legs=%s problems=%s %ss -> %s", "PASS" if res["pass"] else "FAIL", res["legs"], res["problems"][:4], res["seconds"],
                out / "tp2_fold_match.json")
    return 0 if res["pass"] else 1
