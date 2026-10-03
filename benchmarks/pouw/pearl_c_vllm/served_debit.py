"""The honest debit of a served pass at its scheme's cap ρ: a seeded, stratified sample of a retained pass's tiles (the
transcript `verify_run.py` reads), each opened from the pass's files and held to `Verifier.check_tile` and the tile rule
(`pearl_c_work.check_opened`) as `verify_run.check` holds it, with the replayed debit and the cap that rule compares it to
(`pearl_c_work._debit_cap`: ρ times the credited cells' credit) recorded.  CPU only.

  python benchmarks/pouw/pearl_c_vllm/served_debit.py RUN_DIR/prefill/manifest.json RUN_DIR/decode/manifest.json \\
      --tiles 128 64 [--seed S] [--jobs J] [--out DIR]

- **Strata** are a pass's matmul shapes (m, k, n).  Each gets its `--tiles` N: its matmuls in a seeded random order,
  cycled, and a uniform tile of each, so a stratum of 32 matmuls at N = 128 gives each matmul 4 tiles and one of 2,048 gives
  64 of its matmuls 1 each.
- **Per tile:** the verdict at the scheme's cap; the debit and the cap of its credited cells and their ratio (over 1 is a
  rejection); the debit's share of the cells' credit, the cap over ρ (`tile_numbers`), which is already rev1's
  (`creditDevRev1`, the first promotion add of every word out); and the debit's two counts, its flagged atoms and its
  later promotions into a +0 total (``atoms`` and ``later_zero_promotions``: `pearl_c_debit.chain_debit`, rev1's
  `laterZeroPromos`), whose adds the debit already holds.
- **The estimates** weight each stratum by its share of the pass's tiles and of its credited MACs (credited rows × credited
  columns × kept k, from the records), with the share over each cap within the stratum taken from its sample.

**Per input** (``--per-input N`` instead of ``--tiles``), the form a completeness citation needs (lean's ruling on #923):
N seeded inputs of each (k, n), an input being one matmul of the pass, each measured in full up to ``--census-max`` tiles
and otherwise by ``--draws`` work-weighted draws, never pooled.  ``<out>/inputs.json`` holds each input's failures, ``W``,
``w_max`` and its upper limit at 95% simultaneous over the m inputs (`input_bound`), and the cited maximum.

Writes ``<out>/<label>.tiles.jsonl`` (a line a tile, as each finishes; the label is the manifest's directory) and
``<out>/summary.json``.  Its gate is that every sampled tile's openings and checked values hold (`Verifier.check_tile`):
the measurement is of the committed pass.

**On a served pass.** `window.sh` keeps each served window's verify pass outside its run directory, under
``$WORK/passes/<run id>/{prefill,decode}/manifest.json`` on node 2 (``WORK=/workspace/pouw/mvp-e2e``, about 45 GB a run,
so a pass lasts only until it is cleaned).  Run on the machine that holds it, through the queue's CPU cores, with one BLAS
thread a worker (``--jobs`` defaults to the CPUs this process may use)::

  research run --on vy-nebius-2 --queue --phase cpu --cpus 44 --mem-gb 64 --preemptible --custody-r2 --custody-ttl 8h \\
      --project verity --campaign pouw --question Q --declared-output 'out/*' -- bash -c \\
      'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
         PYTHONPATH=$PWD/packages/verity/src:$PWD/protocols/pouw:$PWD/integrations/vllm:$PWD/benchmarks/pouw
       P=/workspace/pouw/mvp-e2e/passes/RUN_ID
       exec nice -n 19 /workspace/pouw/mvp-e2e/venv312/bin/python -u benchmarks/pouw/pearl_c_vllm/served_debit.py \\
         $P/decode/manifest.json $P/prefill/manifest.json --tiles 64 128 --out $RESEARCH_RUN_DIR/out'

A tile takes about 100 s at k = 4,096 and up to 11 min at k = 14,336 on one of node 2's queue cores (Llama-3.1-8B's
1,023 sampled tiles: 98 min on 44, r20261002-173355-4eef).  Add no ``taskset`` of your own: the queue pins the job's cores.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_result  # noqa: E402
import verify_run as VR  # noqa: E402
from verity.commitments import Commitment  # noqa: E402
from verity_pouw.audit import Epoch, Transcript, Verifier, Workload  # noqa: E402
from verity_pouw.protocol import Context, Matmul, Shape  # noqa: E402
from verity_pouw.schemes import pearl_c_work as PW  # noqa: E402
from verity_vllm.protocol_options import pouw_pearl_c_device as PD  # noqa: E402

CAPS = {"1/400": Fraction(1, 400), "1/600": Fraction(1, 600), "1/1000": Fraction(1, 1000)}
SEED = 20261002


class _Kept(PW._TileRuleView):
    """`verify_run`'s view of the scheme, keeping the operands `check_tile`'s encode made so the tile rule reads the same
    ones without encoding twice."""

    kept = None

    def encode(self, ctx, noise, a, b, tile):
        ops = self._scheme.encode(ctx, noise, a, b, tile)
        self.kept = (ctx, tuple(a), tuple(b), tile, ops)
        return ops


_SEEN: list = []
_DEBIT_CAP = PW._debit_cap


def _recorded(*args, **kwargs):
    out = _DEBIT_CAP(*args, **kwargs)
    _SEEN.append(out)
    return out


_CHAINS: list = []
_CHAIN_DEBIT = PW.D.chain_debit


def _chain_recorded(*args, **kwargs):
    out = _CHAIN_DEBIT(*args, **kwargs)
    _CHAINS.append(out)
    return out


def load(path: Path) -> dict:
    """The pass as `verify_run.verify` reads it: the epoch from the beacon and the weights' roots, the records held to their
    published root, the transcript's roots and the verifier's tile enumeration."""
    raw = path.read_bytes()
    man = json.loads(raw)
    if man.get("schema") != PD.MANIFEST_SCHEMA:
        raise ValueError(f"schema {man.get('schema')!r}, not {PD.MANIFEST_SCHEMA}")
    scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    roots = {int(w): bytes.fromhex(v["root"]) for w, v in man["weights"].items()}
    epoch = Epoch.unverified(Workload(matmuls), roots, bytes.fromhex(man["beacon"]))
    if epoch.salt.hex() != man["salt"]:
        raise ValueError("the manifest's salt is not the beacon's and the weights'")
    records = [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]),
                             tuple(man["weights"][str(mm.weight)]["excluded_b"])) for mm, u in zip(matmuls, man["matmuls"])]
    splits = {w: PW.Split(man["weights"][str(w)]["k"]) for w in epoch.workload.weights()}
    reason = PW.check_records(epoch, splits, records)
    if reason is None and PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex() != man["records_root"]:
        reason = "the records are not the ones whose root the pass published"
    if reason is not None:
        raise ValueError(reason)
    c = man["commitment"]
    transcript = Transcript(tuple(bytes.fromhex(u["root_a"]) for u in man["matmuls"]), Commitment(bytes.fromhex(c["root"]), c["count"]), ())
    view = _Kept(scheme)
    verifier = Verifier(view, epoch)
    if len(verifier.refs) != c["count"]:
        raise ValueError(f"the commitment counts {c['count']} tiles; the workload has {len(verifier.refs)}")
    return {"man": man, "sha256": hashlib.sha256(raw).hexdigest(), "scheme": scheme, "view": view, "epoch": epoch,
            "records": records, "transcript": transcript, "verifier": verifier,
            "prover": VR.Opened(man, VR.Files(path.parent), verifier.refs)}


def credited_macs(record: PW.UnitRecord) -> int:
    real = record.real_rows - sum(1 for i in set(record.excluded_a) if i < record.real_rows)
    return real * (record.matmul.shape.n - len(set(record.excluded_b))) * record.k_kept


def sample(job: dict, per_stratum: int, seed: int) -> tuple[dict, list[int]]:
    """The strata (by matmul shape) with their tile and credited-MAC totals, and the sampled tile ids."""
    refs, records = job["verifier"].refs, job["records"]
    first: dict[int, int] = {}
    count: dict[int, int] = {}
    for ref in refs:
        first.setdefault(ref.matmul, ref.id)
        count[ref.matmul] = count.get(ref.matmul, 0) + 1
    strata: dict[str, dict] = {}
    for pos, rec in enumerate(records):
        s = rec.matmul.shape
        st = strata.setdefault(f"m{s.m}-k{s.k}-n{s.n}", {"m": s.m, "k": s.k, "n": s.n, "matmuls": [], "tiles": 0, "credited_macs": 0})
        st["matmuls"].append(pos)
        st["tiles"] += count[pos]
        st["credited_macs"] += credited_macs(rec)
    rng = np.random.default_rng(seed)
    picks = []
    for name in sorted(strata):
        st = strata[name]
        order = [st["matmuls"][i] for i in rng.permutation(len(st["matmuls"]))]
        for t in range(per_stratum):
            pos = order[t % len(order)]
            picks.append(first[pos] + int(rng.integers(count[pos])))
    return strata, sorted(set(picks))


_JOB: dict = {}


def one(tid: int) -> dict:
    j = _JOB
    scheme, view, epoch, transcript, verifier, records = (j[x] for x in ("scheme", "view", "epoch", "transcript", "verifier", "records"))
    t0 = time.perf_counter()
    ref = verifier.refs[tid]
    rec, u = records[ref.matmul], epoch.workload.matmuls[ref.matmul]
    s = u.shape
    out = {"tile": tid, "matmul": ref.matmul, "m": s.m, "k": s.k, "n": s.n, "row0": ref.tile.rows[0], "rows": len(ref.tile.rows),
           "col0": ref.tile.cols[0], "cols": len(ref.tile.cols)}
    opened = j["prover"].open(epoch, tid)
    view.kept = None
    v = verifier.check_tile(transcript, opened)
    out["openings_ok"] = v.ok
    if not v.ok:
        out.update(ok=False, reason=v.reason, seconds=round(time.perf_counter() - t0, 1))
        return out
    ctx = Context(epoch.salt, u, transcript.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
    a, b = [o.value for o in opened.a], [o.value for o in opened.b]
    kept = view.kept
    ops = kept[4] if kept is not None and kept[:4] == (ctx, tuple(a), tuple(b), ref.tile) else scheme.encode(ctx, scheme.noise(ctx), a, b, ref.tile)
    _SEEN.clear()
    _CHAINS.clear()
    reason = PW.check_opened(scheme, rec, ref.tile, a, b, ops)
    rows, cols = rec.credited(ref.tile)
    out.update(ok=reason is None, reason=reason, credited_rows=len(rows), credited_cols=len(cols), macs=len(rows) * len(cols) * rec.k_kept)
    if rows and cols:
        out.update(tile_numbers(scheme, *(_SEEN[-1] if _SEEN else _DEBIT_CAP(scheme, rec, ref.tile, ops, rows, cols))))
        if _CHAINS:
            out["atoms"], out["later_zero_promotions"] = _CHAINS[-1]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def tile_numbers(scheme, debit: Fraction, cap: Fraction) -> dict:
    """The tile rule's (debit, cap) as a tile's line records them: their ratio, and the debit's share of the credited cells'
    credit, the cap over ρ.  That credit is `pearl_c_work.tile_cap`'s `creditDevRev1` share, so the first promotion add is
    out of it once; taking it out again would overstate every share."""
    credit = cap / scheme.cap
    return {"debit": float(debit), "cap": float(cap), "ratio": float(debit / cap), "share": float(debit / credit),
            "debit_exact": str(debit), "credit_exact": str(credit)}


def summarize(label: str, job: dict, strata: dict, tiles: list[dict]) -> dict:
    scheme = job["scheme"]
    measured = [t for t in tiles if "share" in t]
    by: dict[str, list[dict]] = {}
    for t in measured:
        by.setdefault(f"m{t['m']}-k{t['k']}-n{t['n']}", []).append(t)
    total_tiles = sum(st["tiles"] for st in strata.values())
    total_macs = sum(st["credited_macs"] for st in strata.values())
    out_strata, est_tiles, est_macs = {}, {c: 0.0 for c in CAPS}, {c: 0.0 for c in CAPS}
    for name, st in sorted(strata.items()):
        got = by.get(name, [])
        row = {k: st[k] for k in ("m", "k", "n", "tiles", "credited_macs")} | {"matmuls": len(st["matmuls"]), "sampled": len(got)}
        if got:
            sh = [t["share"] for t in got]
            macs = sum(t["macs"] for t in got)
            row.update(share_max=max(sh), share_median=statistics.median(sh), worst_tile=max(got, key=lambda t: t["share"])["tile"],
                       over={c: sum(t["share"] > float(r) for t in got) for c, r in CAPS.items()},
                       over_macs={c: sum(t["macs"] for t in got if t["share"] > float(r)) / macs for c, r in CAPS.items()})
            for c in CAPS:
                est_tiles[c] += st["tiles"] / total_tiles * row["over"][c] / len(got)
                est_macs[c] += st["credited_macs"] / total_macs * row["over_macs"][c]
        out_strata[name] = row
    worst = max(measured, key=lambda t: t["share"]) if measured else None
    n = len(measured)
    return {"label": label, "run": job["man"].get("run"), "scheme": scheme.name, "cap": str(scheme.cap),
            "manifest_sha256": job["sha256"], "tiles_total": total_tiles, "credited_macs_total": total_macs,
            "sampled": len(tiles), "measured": n, "openings_failed": sum(not t["openings_ok"] for t in tiles),
            "rejected_at_scheme_cap": sum(not t["ok"] for t in tiles if t["openings_ok"]),
            "reasons": sorted({t["reason"] for t in tiles if t.get("reason")}),
            "over": {c: sum(t["share"] > float(r) for t in measured) for c, r in CAPS.items()},
            "tile_share_over": est_tiles, "mac_share_over": est_macs,
            "upper95_if_none_over": 1 - 0.05 ** (1 / n) if n else None,
            "share_max": worst["share"] if worst else None, "ratio_max": worst["ratio"] if worst else None,
            "worst": {k: worst[k] for k in ("tile", "matmul", "m", "k", "n", "row0", "col0", "share", "ratio")} if worst else None,
            "later_zero_promotions": sum(t.get("later_zero_promotions", 0) for t in measured),
            "tiles_with_later_zero_promotions": sum(t.get("later_zero_promotions", 0) > 0 for t in measured),
            "share_median": statistics.median(t["share"] for t in measured) if measured else None,
            "share_p99": float(np.quantile([t["share"] for t in measured], 0.99)) if measured else None,
            "cpu_seconds": round(sum(t["seconds"] for t in tiles), 1), "strata": out_strata}


def measure(path: Path, per: int, seed: int, jobs: int, out_dir: Path) -> dict:
    """One pass: its sample measured over `jobs` forked workers, a line a tile as each finishes, then its summary."""
    label, t0 = path.parent.name, time.perf_counter()
    job = load(path)
    strata, picks = sample(job, per, seed)
    refs = job["verifier"].refs
    picks.sort(key=lambda t: -job["records"][refs[t].matmul].matmul.shape.k)
    print(f"{label}: {job['man'].get('run')} {job['scheme'].name} cap {job['scheme'].cap}: {len(picks)} tiles of {len(refs)} "
          f"in {len(strata)} strata (loaded in {time.perf_counter() - t0:.0f} s)", flush=True)
    _JOB.clear()
    _JOB.update(job)
    gc.collect()
    # the workers share the parent's tile enumeration copy-on-write; frozen, no collection in them touches its pages
    gc.freeze()
    tiles: list[dict] = []
    try:
        with open(out_dir / f"{label}.tiles.jsonl", "w") as fh, \
                multiprocessing.get_context("fork").Pool(max(1, min(jobs, len(picks)))) as pool:
            for t in pool.imap_unordered(one, picks, chunksize=1):
                tiles.append(t)
                fh.write(json.dumps(t) + "\n")
                fh.flush()
                if len(tiles) % 32 == 0 or len(tiles) == len(picks):
                    worst = max((x["share"] for x in tiles if "share" in x), default=None)
                    print(f"{label}: {len(tiles)}/{len(picks)} tiles, worst share {worst}", flush=True)
    finally:
        gc.unfreeze()
    out = summarize(label, job, strata, tiles)
    print(json.dumps({k: v for k, v in out.items() if k != "strata"}), flush=True)
    return out


def pick_inputs(job: dict, per_linear: int, census_max: int, draws: int, seed: int) -> list[dict]:
    """Per-input mode: `per_linear` seeded inputs (one matmul each, one linear's GEMM in the pass) of each (k, n), each with
    its tiles' weights.  An input of at most `census_max` tiles is measured in full; a larger one by `draws` work-weighted
    draws with replacement, as the audit draws.  A tile's weight is its area (rows × cols): within one input every tile has
    the same k, so its share of the input's `W_ref` is its share of the cells."""
    refs = job["verifier"].refs
    tiles: dict[int, list[int]] = {}
    for ref in refs:
        tiles.setdefault(ref.matmul, []).append(ref.id)
    groups: dict[tuple[int, int], list[int]] = {}
    for pos, rec in enumerate(job["records"]):
        groups.setdefault((rec.matmul.shape.k, rec.matmul.shape.n), []).append(pos)
    rng = np.random.default_rng(seed)
    out = []
    for kn in sorted(groups):
        for pos in sorted(rng.choice(groups[kn], size=min(per_linear, len(groups[kn])), replace=False).tolist()):
            ids = tiles[pos]
            area = [len(refs[t].tile.rows) * len(refs[t].tile.cols) for t in ids]
            if len(ids) <= census_max:
                method, drawn = "census", list(ids)
            else:
                method = "sampled"
                drawn = [ids[i] for i in rng.choice(len(ids), size=draws, replace=True, p=np.array(area) / sum(area))]
            mm = job["records"][pos].matmul
            out.append({"matmul": pos, "weight": mm.weight, "m": mm.shape.m, "k": mm.shape.k, "n": mm.shape.n, "NT": len(ids),
                        "W": sum(area), "w_max": max(area), "area": dict(zip(ids, area)), "method": method, "drawn": drawn})
    return out


def _bernstein_upper(x: float, w_max: float, total: float, delta: float) -> float:
    """An upper limit on the mean work-weighted rate μ/W of independent tiles from their observed weighted failures x:
    Bernstein's lower tail, P[X <= μ - t] <= exp(-t² / (2·w_max·μ + 2·w_max·t/3)), solved for the largest μ it allows."""
    L = math.log(1 / delta)
    ok = lambda mu: mu <= x or (mu - x) ** 2 <= L * (2 * w_max * mu + 2 * w_max * (mu - x) / 3)
    lo, hi = x, total
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return min(1.0, hi / total)


def input_bound(row: dict, delta: float) -> tuple[float, str]:
    """Lean's ruling on #923: each input's rate p = E[X] (X its work-weighted failing fraction under the tiles' noise) at a
    simultaneous upper confidence limit (δ = 0.05/m per input), assuming the input's tiles fail independently (each tile's
    noise is the oracle at points of its own).  With a = w_max/W:
    - a census with no failure: ln(1/δ)·a (P[none fail] = Π(1 - p_g) <= exp(-p/a));
    - n work-weighted draws with no failure: ln(1/δ)·a / (1 - exp(-n·a)) (E[(1 - X)^n] <= E[exp(-nX)] <= exp(-p(1 - exp(-n·a))/a));
      Clopper–Pearson on the draws would bound the realized X, not p;
    - a census with failures: Bernstein on the weighted sum;
    - draws with failures: no bound here (1): such an input must be measured in full."""
    a = row["w_max"] / row["W"]
    if row["failures"] == 0:
        if row["method"] == "census":
            return min(1.0, math.log(1 / delta) * a), "census, rule of three (work-weighted)"
        return min(1.0, math.log(1 / delta) * a / -math.expm1(-row["drawn"] * a)), "sampled, rule of three (work-weighted draws)"
    if row["method"] == "census":
        return _bernstein_upper(row["failures_weighted"], row["w_max"], row["W"], delta), "census, Bernstein"
    return 1.0, "sampled with failures: measure this input in full"


def measure_inputs(path: Path, per_linear: int, census_max: int, draws: int, seed: int, jobs: int, out_dir: Path) -> list[dict]:
    """Per-input mode on one pass: every chosen input's tiles measured over `jobs` forked workers, a line a tile, then a row
    an input with its failures at the scheme's cap (a tile whose openings hold and the tile rule rejects), `W` and
    `w_max`.  The bound is added once every pass's inputs are known (`m`)."""
    label = path.parent.name
    job = load(path)
    inputs = pick_inputs(job, per_linear, census_max, draws, seed)
    picks = sorted({t for row in inputs for t in row["drawn"]}, key=lambda t: -job["records"][job["verifier"].refs[t].matmul].matmul.shape.k)
    print(f"{label}: {job['man'].get('run')} {job['scheme'].name}: {len(inputs)} inputs, {len(picks)} tiles", flush=True)
    _JOB.clear()
    _JOB.update(job)
    gc.collect()
    gc.freeze()
    got: dict[int, dict] = {}
    try:
        with open(out_dir / f"{label}.input-tiles.jsonl", "w") as fh, \
                multiprocessing.get_context("fork").Pool(max(1, min(jobs, len(picks)))) as pool:
            for t in pool.imap_unordered(one, picks, chunksize=1):
                got[t["tile"]] = t
                fh.write(json.dumps(t) + "\n")
                fh.flush()
                if len(got) % 32 == 0 or len(got) == len(picks):
                    print(f"{label}: {len(got)}/{len(picks)} tiles, {sum(not x['ok'] for x in got.values())} rejected", flush=True)
    finally:
        gc.unfreeze()
    rows = []
    for row in inputs:
        seen = [got[t] for t in row["drawn"]]
        fail = [t for t in seen if t["openings_ok"] and not t["ok"]]
        rows.append({k: v for k, v in row.items() if k not in ("area", "drawn")} | {
            "pass": label, "run": job["man"].get("run"), "manifest_sha256": job["sha256"], "drawn": len(row["drawn"]),
            "openings_failed": sum(not t["openings_ok"] for t in seen), "failures": len(fail),
            "failures_weighted": sum(row["area"][t["tile"]] for t in {t["tile"]: t for t in fail}.values()),
            "share_max": max((t["share"] for t in seen if "share" in t), default=None),
            "reasons": sorted({t["reason"] for t in fail if t.get("reason")})})
    return rows


def write_inputs(rows: list[dict], seed: int, census_max: int, draws: int, out_dir: Path) -> int:
    """Each input's bound at δ = 0.05/m, the cited maximum and the run's result, into ``<out>/inputs.json``."""
    delta = 0.05 / len(rows)
    for r in rows:
        r["upper"], r["bound_method"] = input_bound(r, delta)
    worst = max(rows, key=lambda r: r["upper"])
    doc = {"tool": "pouw_pearl_c_served_debit/per-input", "seed": seed, "confidence": 0.95, "m": len(rows),
           "delta_per_input": delta, "census_max": census_max, "draws": draws,
           "cited_upper": worst["upper"], "cited_input": {k: worst[k] for k in ("pass", "matmul", "weight", "m", "k", "n")},
           "independence": "every bound assumes an input's tiles fail independently (each tile's noise is the oracle at "
                           "points of its own)", "inputs": rows}
    path = out_dir / "inputs.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    ok = all(r["openings_failed"] == 0 for r in rows)
    run_result.write(path, ok, "pearl-c served debit, per input: every measured tile's openings and checked values hold",
                     f"{len(rows)} inputs, {sum(r['failures'] for r in rows)} rejected tiles; the worst input's 95% "
                     f"simultaneous upper limit {worst['upper']:.4g} ({worst['bound_method']})",
                     [("inputs", len(rows), "count"), ("failures", sum(r["failures"] for r in rows), "count"),
                      ("cited_upper", worst["upper"], "share")])
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("manifests", type=Path, nargs="*")
    p.add_argument("--tiles", type=int, nargs="+", default=None, help="tiles per stratum, one per manifest")
    p.add_argument("--per-input", type=int, default=None, metavar="N",
                   help="per-input mode: N seeded inputs of each (k, n) per manifest, each bounded on its own (inputs.json)")
    p.add_argument("--census-max", type=int, default=128, help="per-input mode: an input of at most this many tiles is measured in full")
    p.add_argument("--draws", type=int, default=64, help="per-input mode: work-weighted draws of a larger input")
    p.add_argument("--rebound", type=Path, default=None, metavar="INPUTS_JSON",
                   help="recompute an inputs.json's bounds from its recorded counts (no tile is measured; manifests unused)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.rebound is not None:
        doc = json.loads(a.rebound.read_text())
        out_dir = a.out or a.rebound.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return write_inputs(doc["inputs"], doc["seed"], doc["census_max"], doc["draws"], out_dir)
    if not a.manifests:
        p.error("give one or more manifests")
    if (a.tiles is None) == (a.per_input is None):
        p.error("give --tiles (pooled strata) or --per-input (each input bounded)")
    if a.tiles is not None and len(a.tiles) != len(a.manifests):
        p.error("one --tiles per manifest")
    out_dir = a.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.per_input is not None:
        PW._debit_cap, PW.D.chain_debit = _recorded, _chain_recorded
        try:
            rows = [r for path in a.manifests for r in measure_inputs(path, a.per_input, a.census_max, a.draws, a.seed, a.jobs, out_dir)]
        finally:
            PW._debit_cap, PW.D.chain_debit = _DEBIT_CAP, _CHAIN_DEBIT
        return write_inputs(rows, a.seed, a.census_max, a.draws, out_dir)
    PW._debit_cap, PW.D.chain_debit = _recorded, _chain_recorded
    try:
        passes = [measure(path, per, a.seed, a.jobs, out_dir) for path, per in zip(a.manifests, a.tiles)]
    finally:
        PW._debit_cap, PW.D.chain_debit = _DEBIT_CAP, _CHAIN_DEBIT
    summary = {"tool": "pouw_pearl_c_served_debit", "seed": a.seed, "caps": {c: str(r) for c, r in CAPS.items()}, "passes": passes}
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    ok = all(x["measured"] > 0 and x["openings_failed"] == 0 for x in passes)
    run_result.write(path, ok, "pearl-c served debit: every sampled tile's openings and checked values hold",
                     "; ".join(f"{x['label']} {x['run']} {x['scheme']}: {x['over']['1/1000']} of {x['measured']} tiles over 1/1000, "
                               f"worst share {x['share_max']}" for x in passes),
                     [m for x in passes for m in ((f"{x['label']}.measured", x["measured"], "count"),
                                                  (f"{x['label']}.over_1_1000", x["over"]["1/1000"], "count"),
                                                  (f"{x['label']}.share_max", x["share_max"] or 0.0, "share"))])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
