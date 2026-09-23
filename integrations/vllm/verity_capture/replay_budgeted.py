"""verity_capture/replay_budgeted.py — a BUDGETED conformance experiment over the same evidence tier A uses.

**Separately versioned (`BUDGETED_VERSION`), not a replacement for `replay.tier_a` and never a gate input.**  Tier A draws `k`
instances per (specialization, regime) stratum; a specialization is the full label, so the per-position attention family
`Attention_v3{T=...}` — one specialization per visible KV length — costs `k` evaluations per distinct T (M9: 271 of 283
specializations and 82 % of tier-A seconds on SmolLM2 B8-static C256; O(T^2) at C1024).  This module asks a different, weaker
question with an explicit budget: *for each family, do B evidence-backed cases drawn across the family's T range (plus its boundary
cases) reproduce bit-exactly?*  It reports exactly which cases were evaluated and which specializations were NOT, so nobody can read
it as tier-A coverage.

    python -m verity_capture.replay_budgeted --log L --snapshots S --profile P --out R.json [--budget 24] [--buckets 6] [--k 3]
        [--seed 0] [--max-arg-leaves 4000000] [--steps 0-5] [--family-budget Attention_v3=32] [--budget-s 1800] [--safetensors G] [--cos-sin F]
    python -m verity_capture.replay_budgeted --run RUN_DIR --out R.json ...      # paths / profile / checkpoint from RUN_DIR/run.json

Sampling.  Instances at snapshot steps are grouped by *family* (spec id before `{`).
- A family whose spec ids carry `T=` (attention) is POOLED: its distinct T values are split into `--buckets` contiguous buckets of
  (near) equal specialization count; strata are (bucket x regime).  Boundary cases come first — the smallest T, the largest T, an
  instance at the first snapshot step and one at the last — then the budget `B` is filled round-robin over the strata, each stratum
  visiting distinct specializations before repeating one and distinct steps within a specialization (`replay._distinct_step_order`).
- Every other family keeps tier A's own rule, `k` per (specialization, regime) stratum, so the side-by-side with `replay_a.json`
  changes only the pooled families.
- The evaluation of one (specialization, instance) IS `replay.Replayer.evaluate_instance` (or `evaluate_child` along a
  `plan_child` path for an oversize instance): imported, not copied, so a case's result equals tier A's for that instance.
- A case whose expected output (or an operand's producer output) was never snapshotted is reported as **`unrecorded`** and counts
  against nothing: an interior value that was never recorded is not testable and is never replaced by an evaluator-generated value.

Report (`verity-perf/replay_budgeted/v1`): per family the population (specializations, instances, T range, per-bucket counts), the
selected cases with T / step / bucket / regime / why, every evaluation entry, `evaluations`, `seconds`, `max_rss_gb`, the
tier-A-equivalent evaluation count (k per stratum), and `limitations`: specializations not evaluated (with their T), empty strata,
unrecorded cases, an early stop under `--budget-s`.  `status`: "pass" = every evaluated case passed and nothing was unrecorded;
"partial" = every evaluated case passed but some were unrecorded / the wall budget stopped the run; "fail" = a failing or erroring
case; "skipped" = nothing evaluated.  The verdict is about the evaluated cases only.

Memory preflight (v0.1.3).  Before folding, the run's own `fold_summary.json` peak plus `INDEX_TRANSIENT_PER_BYTE` x the size of
`snapshots.json` is compared with the cgroup limit (`memory.max`, v1 `memory.limit_in_bytes`, or `COV_MEM_LIMIT_GB`); when the
projection exceeds a known limit the tool writes a `skipped` report with the projection and exits 2 instead of being SIGKILLed after
the fold (measured: `B0-cov_b0_b8-mixed_c1024` folds to 52.7 GB, its 3.02 GB index needs ~10 GB more -> killed at 1 145 s in 61 GB).
`--force` runs anyway; with no limit known the projection is only printed and recorded under `preflight`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .compare import SnapshotStore, WeightSource, _jsonable, _max_rss_gb, fold_log, load_profile
from .fold import FoldResult
from .replay import Replayer, Skipped, _distinct_step_order, _regime

BUDGETED_VERSION = "0.1.3"
SCHEMA = "verity-perf/replay_budgeted/v1"
_T_RE = re.compile(r"\bT=(\d+)")


def family_of(spec: str) -> str:
    return spec.split("{")[0]


def T_of(spec: str) -> int | None:
    m = _T_RE.search(spec)
    return int(m.group(1)) if m else None


def parse_steps(s: str | None) -> list[int] | None:
    if s is None or str(s).strip().lower() in ("", "all"):
        return None
    out: set[int] = set()
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return sorted(out)


def t_buckets(Ts: list[int], n: int) -> list[tuple[int, int]]:
    """`n` contiguous [T_lo, T_hi] buckets over the sorted distinct T values, near-equal counts (fewer when there are fewer T)."""
    Ts = sorted(set(Ts))
    if not Ts:
        return []
    n = max(1, min(n, len(Ts)))
    out = []
    for i in range(n):
        lo, hi = (i * len(Ts)) // n, ((i + 1) * len(Ts)) // n
        if hi > lo:
            out.append((Ts[lo], Ts[hi - 1]))
    return out


def bucket_of(T: int, buckets: list[tuple[int, int]]) -> int:
    for i, (lo, hi) in enumerate(buckets):
        if lo <= T <= hi:
            return i
    raise ValueError(f"T={T} in no bucket {buckets}")


# ---------------------------------------------------------------------------------------------------------
# selection (pure: no evidence touched)
# ---------------------------------------------------------------------------------------------------------


def select_cases(result: FoldResult, snap_steps: list[int], *, budget: int, buckets: int, k: int, seed: int,
                 family_budget: dict[str, int] | None = None, pooled: Iterable[str] | None = None) -> dict:
    """The plan: per family its population and the ordered list of cases to evaluate (instance metadata + why).

    `pooled`: families to pool (default: every family whose spec ids carry `T=`).  `family_budget[f]` overrides `budget` for f."""
    rng = random.Random(seed)
    fam_budget = dict(family_budget or {})
    metas = [m for m in result.instances if int(m["step"]) in set(snap_steps)]
    all_by_fam: dict[str, list[dict]] = defaultdict(list)
    for m in result.instances:
        all_by_fam[family_of(m["spec"])].append(m)
    by_fam: dict[str, list[dict]] = defaultdict(list)
    for m in metas:
        by_fam[family_of(m["spec"])].append(m)
    if pooled is None:
        pooled_set = {f for f, ms in all_by_fam.items() if any(T_of(m["spec"]) is not None for m in ms)}
    else:
        pooled_set = set(pooled)
    plan: dict[str, Any] = {"seed": seed, "budget": budget, "buckets": buckets, "k": k, "snapshot_steps": list(snap_steps),
                            "family_budget": fam_budget, "pooled_families": sorted(pooled_set), "families": {}}
    first_step, last_step = (min(snap_steps), max(snap_steps)) if snap_steps else (None, None)
    for fam in sorted(all_by_fam):
        ms = by_fam.get(fam, [])
        specs_all = sorted({m["spec"] for m in all_by_fam[fam]})
        specs_ev = sorted({m["spec"] for m in ms})
        rec: dict[str, Any] = {"pooled": fam in pooled_set, "specs": len(specs_all), "specs_at_snapshot_steps": len(specs_ev),
                               "instances": len(all_by_fam[fam]), "instances_at_snapshot_steps": len(ms), "cases": [],
                               "tier_a_equivalent_evaluations": 0}
        # tier A's own requirement for the comparison column: k per (spec, regime) stratum
        strata_ta: Counter = Counter((m["spec"], _regime(m)) for m in ms)
        rec["tier_a_equivalent_evaluations"] = sum(min(k, n) for n in strata_ta.values())
        rec["tier_a_strata"] = len(strata_ta)
        if not ms:
            rec["limitation"] = "no instance at a snapshot step"
            plan["families"][fam] = rec
            continue
        if fam not in pooled_set:
            # tier A's rule, tier A's order
            regimes: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for m in ms:
                regimes[(m["spec"], _regime(m))].append(m)
            for (spec, rg) in sorted(regimes):
                for m in _distinct_step_order(regimes[(spec, rg)], rng)[:k]:
                    rec["cases"].append(_case(m, why="k_per_spec_regime", regime=rg))
            rec["strata"] = {"kind": "spec x regime", "count": len(regimes)}
            plan["families"][fam] = rec
            continue
        # pooled family: T buckets x regime
        B = int(fam_budget.get(fam, budget))
        Ts_all = sorted({T_of(m["spec"]) for m in all_by_fam[fam] if T_of(m["spec"]) is not None})
        Ts_ev = sorted({T_of(m["spec"]) for m in ms if T_of(m["spec"]) is not None})
        bks = t_buckets(Ts_ev, buckets)
        rec["T"] = {"min": Ts_all[0] if Ts_all else None, "max": Ts_all[-1] if Ts_all else None, "distinct": len(Ts_all),
                    "distinct_at_snapshot_steps": len(Ts_ev), "buckets": [{"T_lo": lo, "T_hi": hi} for lo, hi in bks]}
        rec["budget"] = B
        strata: dict[tuple[int, str], list[dict]] = defaultdict(list)
        for m in ms:
            T = T_of(m["spec"])
            if T is None:
                continue
            strata[(bucket_of(T, bks), _regime(m))].append(m)
        for (b, rg), lst in strata.items():
            bk = rec["T"]["buckets"][b]
            bk.setdefault("strata", {})[rg] = {"specs": len({m["spec"] for m in lst}), "instances": len(lst)}
        rec["strata"] = {"kind": "T-bucket x regime", "count": len(strata)}
        chosen: list[dict] = []
        used: set[int] = set()

        def take(m: dict, why: str) -> None:
            if m["node"] in used or len(chosen) >= B:
                return
            used.add(m["node"])
            T = T_of(m["spec"])
            chosen.append(_case(m, why=why, regime=_regime(m), T=T, bucket=bucket_of(T, bks) if T is not None else None))

        # boundary cases first: smallest T, largest T, first snapshot step, last snapshot step
        with_T = [m for m in ms if T_of(m["spec"]) is not None]
        if with_T:
            take(min(with_T, key=lambda m: (T_of(m["spec"]), m["step"], m["node"])), "boundary:min_T")
            take(max(with_T, key=lambda m: (T_of(m["spec"]), m["step"], m["node"])), "boundary:max_T")
            fs = [m for m in with_T if m["step"] == first_step]
            if fs:
                take(rng.choice(sorted(fs, key=lambda m: m["node"])), "boundary:first_step")
            ls = [m for m in with_T if m["step"] == last_step]
            if ls:
                take(rng.choice(sorted(ls, key=lambda m: m["node"])), "boundary:last_step")
        # stratified fill: round-robin over strata; within a stratum distinct specs first (seeded), distinct steps within a spec
        orders: dict[tuple[int, str], list[dict]] = {}
        for key, lst in strata.items():
            by_spec: dict[str, list[dict]] = defaultdict(list)
            for m in lst:
                by_spec[m["spec"]].append(m)
            spec_order = sorted(by_spec)
            rng.shuffle(spec_order)
            per_spec = {s: _distinct_step_order(by_spec[s], rng) for s in spec_order}
            seq: list[dict] = []
            i = 0
            while len(seq) < len(lst):
                for s in spec_order:
                    if i < len(per_spec[s]):
                        seq.append(per_spec[s][i])
                i += 1
            orders[key] = seq
        keys = sorted(orders)
        rng.shuffle(keys)
        specs_drawn = {c["spec"] for c in chosen}

        def fill(new_only: bool) -> None:
            # v0.1.2: two passes.  Pass 1 draws only specializations nobody has drawn yet (boundary cases included), round-robin
            # over the strata, so a thin stratum (e.g. a single decode T) cannot spend the budget on repeats while other strata
            # still hold unvisited specializations.  Pass 2 (only if budget is left) allows repeats: distinct steps within a
            # specialization first (`_distinct_step_order`), still round-robin over the strata.
            pos = {key: 0 for key in keys}
            progressed = True
            while len(chosen) < B and progressed:
                progressed = False
                for key in keys:
                    seq = orders[key]
                    while pos[key] < len(seq) and (seq[pos[key]]["node"] in used or (new_only and seq[pos[key]]["spec"] in specs_drawn)):
                        pos[key] += 1
                    if pos[key] < len(seq):
                        m = seq[pos[key]]
                        take(m, f"stratified:bucket{key[0]}:{key[1]}")
                        specs_drawn.add(m["spec"])
                        pos[key] += 1
                        progressed = True
                    if len(chosen) >= B:
                        break

        fill(True)
        fill(False)
        rec["cases"] = chosen
        sampled_specs = {c["spec"] for c in chosen}
        not_sampled = sorted(set(specs_ev) - sampled_specs, key=lambda s: (T_of(s) or -1, s))
        rec["specs_not_evaluated"] = {"count": len(not_sampled) + (len(specs_all) - len(specs_ev)),
                                      "at_snapshot_steps": len(not_sampled), "no_snapshot_step": len(specs_all) - len(specs_ev),
                                      "T_values_not_evaluated": sorted({T_of(s) for s in not_sampled if T_of(s) is not None})[:400]}
        empty = [f"bucket{b}:{rg}" for b in range(len(bks)) for rg in ("prefill", "decode") if (b, rg) not in strata]
        rec["empty_strata"] = empty
        plan["families"][fam] = rec
    plan["n_cases"] = sum(len(f["cases"]) for f in plan["families"].values())
    plan["tier_a_equivalent_evaluations"] = sum(f["tier_a_equivalent_evaluations"] for f in plan["families"].values())
    return plan


def _case(m: dict, *, why: str, regime: str, T: int | None = None, bucket: int | None = None) -> dict:
    return {"node": m["node"], "spec": m["spec"], "step": int(m["step"]), "name": m["name"], "event": m.get("event"), "regime": regime,
            "T": T if T is not None else T_of(m["spec"]), "bucket": bucket, "why": why}


# ---------------------------------------------------------------------------------------------------------
# evaluation (replay.py's, imported)
# ---------------------------------------------------------------------------------------------------------


def run_budgeted(result: FoldResult, store: SnapshotStore, weights: WeightSource | None, *, budget: int = 24, buckets: int = 6, k: int = 3,
                 seed: int = 0, max_arg_leaves: int = 4_000_000, steps: Iterable[int] | None = None, family_budget: dict[str, int] | None = None,
                 budget_s: float | None = None, pooled: Iterable[str] | None = None) -> dict:
    """Select (pure) then evaluate every case with `Replayer.evaluate_instance` / `evaluate_child`; see the module docstring."""
    t0 = time.time()
    snap_steps = sorted(set(store.steps) if steps is None else set(store.steps) & set(int(s) for s in steps))
    report: dict[str, Any] = {"schema": SCHEMA, "version": BUDGETED_VERSION, "tier": "budgeted", "status": "pass", "budget": budget, "buckets": buckets,
                              "k": k, "seed": seed, "max_arg_leaves": max_arg_leaves, "snapshot_steps": snap_steps, "budget_s": budget_s,
                              "not_a_gate": "budgeted experiment: the verdict covers the evaluated cases only (see limitations)"}
    plan = select_cases(result, snap_steps, budget=budget, buckets=buckets, k=k, seed=seed, family_budget=family_budget, pooled=pooled)
    report["plan"] = {kk: v for kk, v in plan.items() if kk != "families"}
    report["families"] = {}
    try:
        rep = Replayer(result, store, weights, chain=False, snapshots=True)
    except ValueError as e:
        report.update(status="skipped", reason=str(e), seconds=round(time.time() - t0, 3), max_rss_gb=round(_max_rss_gb(), 3))
        return report
    rng = random.Random(seed + 1)          # the child-member draws (tier A draws them from its own rng too)
    paths: dict[str, list[tuple[int, int]] | None] = {}
    entries: list[dict] = []
    stopped = None
    for fam, frec in plan["families"].items():
        out_f: dict[str, Any] = {kk: v for kk, v in frec.items() if kk != "cases"}
        out_f["cases"] = []
        out_f["counts"] = Counter()
        out_f["seconds"] = 0.0
        for c in frec["cases"]:
            if budget_s is not None and time.time() - t0 > budget_s:
                stopped = f"wall budget {budget_s:.0f} s reached before case node {c['node']} ({fam})"
                break
            m = rep.by_node[c["node"]]
            spec = m["spec"]
            t1 = time.time()
            try:
                if spec not in paths:
                    leaves = rep.arg_leaves(m["node"])
                    paths[spec] = None if leaves <= max_arg_leaves else rep.plan_child(m["node"], max_arg_leaves, rng)
                    if leaves > max_arg_leaves and paths[spec] is None:
                        raise Skipped(f"{leaves} argument leaves exceed max_arg_leaves={max_arg_leaves} and no child unit fits")
                e = rep.evaluate_child(m, paths[spec], rng) if paths[spec] is not None else rep.evaluate_instance(m)
            except Skipped as ex:
                # missing evidence: an unrecorded expected value / operand.  Reported, never substituted.
                kind = "unrecorded" if ("no snapshot" in ex.reason or "not snapshotted" in ex.reason) else "skipped"
                e = {"node": m["node"], "name": m["name"], "spec": spec, "step": m["step"], "status": kind, "reason": ex.reason,
                     "granularity": "child" if paths.get(spec) else "instance"}
            except Exception as ex:  # noqa: BLE001 - evaluator problems are reported per case
                e = {"node": m["node"], "name": m["name"], "spec": spec, "step": m["step"], "status": "error", "reason": f"{type(ex).__name__}: {ex}"}
            e["seconds"] = round(time.time() - t1, 3)
            e["case"] = {kk: c[kk] for kk in ("T", "bucket", "regime", "why")}
            e["family"] = fam
            out_f["cases"].append(e)
            out_f["counts"][e["status"]] += 1
            out_f["seconds"] += e["seconds"]
            entries.append(e)
        out_f["evaluations"] = sum(v for s, v in out_f["counts"].items() if s in ("pass", "fail", "error"))
        out_f["counts"] = dict(out_f["counts"])
        out_f["seconds"] = round(out_f["seconds"], 3)
        if out_f["cases"]:
            Ts = [e["case"]["T"] for e in out_f["cases"] if e["case"]["T"] is not None]
            out_f["evaluated_T"] = sorted(set(Ts))
        report["families"][fam] = out_f
        if stopped:
            break
    st = Counter(e["status"] for e in entries)
    report["summary"] = dict(st)
    report["evaluations"] = sum(st.get(s, 0) for s in ("pass", "fail", "error"))
    report["unrecorded"] = [{"node": e["node"], "spec": e["spec"], "step": e["step"], "reason": e["reason"][:160]} for e in entries if e["status"] == "unrecorded"]
    report["limitations"] = {
        "specs_not_evaluated": {f: v.get("specs_not_evaluated") for f, v in report["families"].items() if v.get("pooled")},
        "families_not_reached": [f for f in plan["families"] if f not in report["families"]],
        "empty_strata": {f: v.get("empty_strata") for f, v in report["families"].items() if v.get("empty_strata")},
        "unrecorded_cases": len(report["unrecorded"]), "stopped_early": stopped,
        "note": "a specialization not listed under an evaluated case has NO evidence from this experiment; unrecorded values were never replaced"}
    if st.get("fail") or st.get("error"):
        report["status"] = "fail"
        report["status_reason"] = f"{st.get('fail', 0)} failing and {st.get('error', 0)} erroring cases"
    elif not st.get("pass"):
        report["status"] = "skipped"
        report["status_reason"] = "nothing evaluated"
    elif st.get("unrecorded") or st.get("skipped") or stopped:
        report["status"] = "partial"
        report["status_reason"] = f"{st.get('unrecorded', 0)} unrecorded, {st.get('skipped', 0)} skipped" + (f"; {stopped}" if stopped else "")
    report["tier_a_equivalent_evaluations"] = plan["tier_a_equivalent_evaluations"]
    report["seconds"] = round(time.time() - t0, 3)
    report["max_rss_gb"] = round(_max_rss_gb(), 3)
    return report


def summary_line(rep: dict) -> str:
    fams = rep.get("families", {})
    pooled = {f: (v.get("evaluations"), v.get("budget"), v.get("specs"), round(v.get("seconds", 0), 1)) for f, v in fams.items() if v.get("pooled")}
    return (f"budgeted v{rep.get('version')} {rep.get('status'):7s} evaluations={rep.get('evaluations')} (tier-A-equivalent {rep.get('tier_a_equivalent_evaluations')}) "
            f"summary={rep.get('summary')} pooled(evals,budget,specs,s)={pooled} unrecorded={len(rep.get('unrecorded', []))} "
            f"{rep.get('seconds')}s {rep.get('max_rss_gb')}GB")


def resolve_run(run_dir: str | Path) -> dict[str, str | None]:
    """log / snapshots dir / profile / safetensors glob / cos_sin of a campaign run directory (`run.json` + `capture/`),
    the same resolution `bench/run_config.py` applies when it builds the replay command."""
    run = Path(run_dir)
    man = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    cap = run / "capture"
    arts = man.get("artifacts") or {}
    snaps = Path(arts["snapshots"]).parent if arts.get("snapshots") else cap
    ck = man.get("checkpoint_dir")
    safetensors = None
    if ck:
        safetensors = ck if any(ch in ck for ch in "*?[") else str(Path(ck) / "model*.safetensors")
    cs = snaps / "cos_sin_cache.npy"
    return {"log": str(arts.get("log") or (cap / "log.jsonl.gz")), "snapshots": str(snaps), "profile": man.get("profile"),
            "safetensors": safetensors, "cos_sin": str(cs) if cs.exists() else None}


INDEX_TRANSIENT_PER_BYTE = 3.3   # measured on B0-cov_b0_b8-static_c256: 0.256 GB snapshots.json -> +0.83 GB while parsing (compare.SnapshotStore)


def cgroup_memory_limit_gb() -> float | None:
    """The memory limit this process runs under (cgroup v2 `memory.max`, v1 `memory.limit_in_bytes`, or the campaign's
    `COV_MEM_LIMIT_GB`), None when unknown.  Pods here do not expose the files; the runner passes the env."""
    env = os.environ.get("COV_MEM_LIMIT_GB")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            s = Path(p).read_text().strip()
        except OSError:
            continue
        if s.isdigit() and int(s) < 1 << 60:
            return int(s) / 1e9
    return None


def preflight_memory(run_dir: str | Path | None, snapshots_dir: str | Path, limit_gb: float | None) -> dict:
    """Projected peak of fold + snapshot index for this run, from the run's own records (no folding): the legacy
    `fold_summary.json` peak (the fold's live set) plus INDEX_TRANSIENT_PER_BYTE x `snapshots.json` size.  Measured motivation:
    `B0-cov_b0_b8-mixed_c1024` folds to 52.7 GB and its 3.02 GB index needs ~10 GB more while parsing -> every fold+index process
    is SIGKILLed in a 61 GB cgroup after the ~19 min fold (compare / replay_chain / replay_a in the legacy run, and this tool)."""
    fold_gb = None
    if run_dir:
        fs = Path(run_dir) / "fold_summary.json"
        if fs.exists():
            try:
                fold_gb = float(json.loads(fs.read_text()).get("max_rss_gb"))
            except (ValueError, TypeError, json.JSONDecodeError):
                fold_gb = None
    idx = Path(snapshots_dir) / "snapshots.json"
    idx_gb = idx.stat().st_size / 1e9 if idx.exists() else None
    proj = (fold_gb or 0.0) + (idx_gb or 0.0) * INDEX_TRANSIENT_PER_BYTE if (fold_gb is not None or idx_gb is not None) else None
    out = {"fold_peak_gb_legacy": fold_gb, "snapshots_json_gb": idx_gb, "projected_peak_gb": round(proj, 2) if proj is not None else None,
           "limit_gb": limit_gb, "verdict": "unknown"}
    if proj is not None and limit_gb is not None:
        out["verdict"] = "over" if proj > limit_gb else "fits"
    elif proj is not None:
        out["verdict"] = "no limit known"
    return out


def filter_snapshot_index(src: str | Path, steps: set[int], dst: str | Path, chunk: int = 1 << 24) -> dict:
    """Write `dst/snapshots.json` holding only the index entries whose `step` is in `steps` (entries without a step are kept),
    streaming the source index in `chunk`-sized pieces (`json.JSONDecoder.raw_decode` per entry, O(chunk) memory instead of the
    3.3x-file-size object tree), and symlink every other file/dir of `src` (values/, cos_sin_cache.npy, ...) into `dst` so an
    unchanged `compare.SnapshotStore(dst)` serves exactly the kept entries.  Supported layout: a top-level object with an
    `entries` array (`verity-capture/snapshots/v1`); anything else raises ValueError (caller falls back to the full index).
    Replay needs every producer of an operand to be present, so `steps` should be contiguous from 0 (the capture's own rule)."""
    src, dst = Path(src), Path(dst)
    t0 = time.time()
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == "snapshots.json":
            continue
        link = dst / child.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(child.resolve())
    dec = json.JSONDecoder()
    kept = total = 0
    with open(src / "snapshots.json", "r", encoding="utf-8") as fh, open(dst / "snapshots.json", "w", encoding="utf-8") as out:
        buf = fh.read(chunk)
        m = re.search(r'"entries"\s*:\s*\[', buf)
        if not buf.lstrip().startswith("{") or not m:
            raise ValueError("snapshots.json is not an object with an `entries` array; streaming filter unsupported")
        out.write(buf[:m.start()] + '"entries": [')
        pos = m.end()
        first = True
        while True:
            while True:                                  # position on the next entry or the closing bracket
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                more = fh.read(chunk)
                if not more:
                    raise ValueError("truncated snapshots.json")
                buf, pos = buf[pos:] + more, 0
            if buf[pos] == "]":
                pos += 1
                break
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                more = fh.read(chunk)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                continue
            total += 1
            st = obj.get("step") if isinstance(obj, dict) else None
            if st is None or int(st) in steps:
                out.write(("" if first else ",\n") + json.dumps(obj, separators=(",", ":")))
                first = False
                kept += 1
            pos = end
        out.write("]" + buf[pos:] + fh.read())
    return {"steps": sorted(steps), "entries_total": total, "entries_kept": kept, "dir": str(dst), "seconds": round(time.time() - t0, 3),
            "src_bytes": (src / "snapshots.json").stat().st_size, "dst_bytes": (dst / "snapshots.json").stat().st_size}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default=None, help="campaign run dir: resolves --log/--snapshots/--profile/--safetensors/--cos-sin from run.json")
    ap.add_argument("--log", default=None)
    ap.add_argument("--snapshots", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=24, help="cases per pooled (T-indexed) family")
    ap.add_argument("--buckets", type=int, default=6, help="T buckets per pooled family")
    ap.add_argument("--k", type=int, default=3, help="tier A's k for the non-pooled families")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-arg-leaves", type=int, default=4_000_000)
    ap.add_argument("--steps", default=None, help="snapshot steps to sample from (e.g. 0-5, 0,1,8; default all)")
    ap.add_argument("--family-budget", action="append", default=[], help="FAMILY=B override, repeatable")
    ap.add_argument("--budget-s", type=float, default=None, help="stop evaluating after this many seconds (reported as partial)")
    ap.add_argument("--safetensors", default=None)
    ap.add_argument("--cos-sin", default=None)
    ap.add_argument("--force", action="store_true", help="run even when the memory preflight projects fold + index over the cgroup limit")
    ap.add_argument("--index-steps", default=None,
                    help="memory workaround: stream-filter snapshots.json to these steps (contiguous from 0, e.g. 0-9) into <out dir>/index_<steps>/ "
                         "and open that with the unchanged SnapshotStore; sampling is then restricted to those steps and the report says so")
    a = ap.parse_args(argv)
    if a.run:
        rr = resolve_run(a.run)
        for key in ("log", "snapshots", "profile", "safetensors", "cos_sin"):
            if getattr(a, key) is None:
                setattr(a, key, rr[key])
    for key in ("log", "snapshots", "profile"):
        if getattr(a, key) is None:
            ap.error(f"--{key} is required (or --run with a run.json that provides it)")
    index_filter = None
    if a.index_steps:
        want = set(parse_steps(a.index_steps) or [])
        dst = Path(a.out).parent / f"index_{a.index_steps.replace(',', '_')}"
        try:
            index_filter = filter_snapshot_index(a.snapshots, want, dst)
            print(f"index filter: kept {index_filter['entries_kept']} of {index_filter['entries_total']} entries for steps {a.index_steps} "
                  f"({index_filter['dst_bytes'] / 1e6:.1f} MB of {index_filter['src_bytes'] / 1e9:.3f} GB) in {index_filter['seconds']} s -> {dst}", file=sys.stderr)
            a.snapshots = str(dst)
        except ValueError as e:
            print(f"index filter unsupported ({e}); using the full index", file=sys.stderr)
    pre = preflight_memory(a.run, a.snapshots, cgroup_memory_limit_gb())
    print(f"preflight: fold peak (legacy fold_summary) {pre['fold_peak_gb_legacy']} GB + snapshot index {pre['snapshots_json_gb']} GB x "
          f"{INDEX_TRANSIENT_PER_BYTE} ≈ {pre['projected_peak_gb']} GB; limit {pre['limit_gb']} GB -> {pre['verdict']}", file=sys.stderr)
    if pre["verdict"] == "over" and not a.force:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rep = {"schema": SCHEMA, "version": BUDGETED_VERSION, "tier": "budgeted", "status": "skipped", "log": a.log, "profile": a.profile,
               "reason": f"memory preflight: projected fold + index peak {pre['projected_peak_gb']} GB exceeds the {pre['limit_gb']} GB limit "
                         "(pass --force to try anyway)", "preflight": pre, "evaluations": 0, "seconds": 0.0}
        out.write_text(json.dumps(rep, indent=1))
        print(summary_line(rep), file=sys.stderr)
        print(str(out))
        return 2
    t0 = time.time()
    profile = load_profile(a.profile)
    repo = (profile.config or {}).get("model")
    result = fold_log(a.log, profile, with_program=True)
    fold_s = round(time.time() - t0, 3)
    snaps = Path(a.snapshots)
    store = SnapshotStore(snaps)
    cos_sin = Path(a.cos_sin) if a.cos_sin else ((snaps / "cos_sin_cache.npy") if (snaps / "cos_sin_cache.npy").exists() else None)
    weights = WeightSource(result.params[-1][1], [Path(p) for p in sorted(glob.glob(a.safetensors))] if a.safetensors else None, cos_sin, repo=repo,
                           config=dict(profile.config or {}))
    fb = {}
    for it in a.family_budget:
        f, _, b = it.partition("=")
        fb[f] = int(b)
    rep = run_budgeted(result, store, weights, budget=a.budget, buckets=a.buckets, k=a.k, seed=a.seed, max_arg_leaves=a.max_arg_leaves,
                       steps=parse_steps(a.steps), family_budget=fb or None, budget_s=a.budget_s)
    rep = {"log": a.log, "profile": a.profile, "model": repo, "digest": result.digest, "fold_seconds": fold_s, "weights_available": weights.available(),
           "preflight": pre, "index_filter": index_filter, **rep}
    if index_filter and "limitations" in rep:
        rep["limitations"]["index_filtered_to_steps"] = index_filter["steps"]
    rep["total_seconds"] = round(time.time() - t0, 3)
    rep["max_rss_gb"] = round(_max_rss_gb(), 3)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(rep), indent=1))
    print(summary_line(rep), file=sys.stderr)
    print(str(out))
    return 0 if rep["status"] in ("pass", "partial") else 1


if __name__ == "__main__":
    sys.exit(main())
