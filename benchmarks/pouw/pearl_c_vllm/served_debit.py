"""The honest debit of a served pass at its scheme's cap ρ: a seeded, stratified sample of a retained pass's tiles (the
transcript `verify_run.py` reads), each opened from the pass's files and held to `Verifier.check_tile` and the tile rule
(`clear_work.check_opened`) as `verify_run.check` holds it, with the replayed debit and the cap that rule compares it to
(`clear_work._debit_cap`: ρ times the credited cells' credit) recorded.  CPU only.

  python benchmarks/pouw/pearl_c_vllm/served_debit.py RUN_DIR/prefill/manifest.json RUN_DIR/decode/manifest.json \\
      --tiles 128 64 [--seed S] [--jobs J] [--out DIR]

- **Strata** are a pass's matmul shapes (m, k, n).  Each gets its `--tiles` N: its matmuls in a seeded random order,
  cycled, and a uniform tile of each, so a stratum of 32 matmuls at N = 128 gives each matmul 4 tiles and one of 2,048 gives
  64 of its matmuls 1 each.
- **Per tile:** the verdict at the scheme's cap; the debit and the cap of its credited cells and their ratio (over 1 is a
  rejection); the debit's share of the cells' credit, the cap over ρ (`tile_numbers`), which is already rev1's
  (`creditDevRev1`, the first promotion add of every word out); and the debit's two counts, its flagged atoms and its
  later promotions into a +0 total (``atoms`` and ``later_zero_promotions``: `pearl_c_debit.chain_debit`, rev1's
  `laterZeroPromos`), whose adds the debit already holds.  Under a format with the row cap (-h3) the rule also caps each
  credited A row (`clear_work.row_cap`, 4ρ at Pearl-C's census ks), and ``row_ratio_max`` is the largest row's debit over
  its cap (the tile rule checked as far as it ran: the rows up to the first one over).
- **The estimates** weight each stratum by its share of the pass's tiles and of its credited MACs (credited rows × credited
  columns × kept k, from the records), with the share over each cap within the stratum taken from its sample.

**Per input** (``--per-input N`` instead of ``--tiles``), the form a completeness citation needs:
N seeded inputs of each (k, n), an input being one matmul of the pass, each measured in full up to ``--census-max`` tiles
and otherwise by ``--draws`` work-weighted draws, never pooled.  ``<out>/inputs.json`` holds each input's failures, ``W``,
``w_max`` and its upper limit at 95% simultaneous over the m inputs (`input_bound`), and the cited maximum.  With
``--fresh-draws N`` the same inputs are measured instead by N trials each: a work-weighted tile under a salt of its own,
derived from ``--fresh-source`` (recorded OS randomness drawn after the pass, fixed in the command), so that the trials
are independent draws of the input's rate itself (`HonestTileCapRate`'s oracle, salt and draw, the activations fixed) and
Clopper–Pearson bounds it, 1 − δ^(1/N) at no failure whatever the input's tile count (`HonestCap.lean`'s
fourth case; lines in ``<label>.fresh-tiles.jsonl``).  The cited limit uses every trial run for an input.

A tile is replayed on `fast_tile`'s batch forms (``--scalar``: the scalar reference only), every ``--cross-check``-th tile
id on both, and ``--resume`` reuses an earlier run's lines on the same pass.

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
         PYTHONPATH=$PWD:$PWD/tools/config:$PWD/integrations/vllm:$PWD/benchmarks/pouw
       P=/workspace/pouw/mvp-e2e/passes/RUN_ID
       exec nice -n 19 /workspace/pouw/mvp-e2e/venv312/bin/python -u benchmarks/pouw/pearl_c_vllm/served_debit.py \\
         $P/decode/manifest.json $P/prefill/manifest.json --tiles 64 128 --out $RESEARCH_RUN_DIR/out'

A tile takes about 100 s at k = 4,096 and up to 11 min at k = 14,336 on one of node 2's queue cores (Llama-3.1-8B's
1,023 sampled tiles: 98 min on 44).  Add no ``taskset`` of your own: the queue pins the job's cores.
"""
from __future__ import annotations

import argparse
import bisect
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
from itertools import accumulate
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "clear"))
import fast_tile as FT  # noqa: E402
import run_result  # noqa: E402
import verify_run as VR  # noqa: E402
from verity.ml.commitments import Commitment  # noqa: E402
from verity.ml.commitments import verify_opening  # noqa: E402
from verity.ml.randomness import derive  # noqa: E402
from verity.core.protocols.pouw.audit import (ROW_SCHEMA, activation_binding, opened_value, rows_domain,  # noqa: E402
                                              weight_binding)
from clear_audit import Publication, Verifier, commitment_hash  # noqa: E402
from verity.core.protocols.pouw.coins import CoinCommitment  # noqa: E402
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c_work as PW  # noqa: E402
import clear_work as CW  # noqa: E402
from verity_vllm.protocols.pouw import pearl_c as PD  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
CAPS = {"1/400": Fraction(1, 400), "1/600": Fraction(1, 600), "1/1000": Fraction(1, 1000)}
SEED = 20261002
FRESH = "verity/pouw/fp8-served-table-fresh-trials/v1"


class _Kept(CW._TileRuleView):
    """`verify_run`'s view of the scheme, keeping the operands `check_tile`'s encode made so the tile rule reads the same
    ones without encoding twice."""

    kept = None

    def encode(self, ctx, noise, a, b, tile):
        ops = self._scheme.encode(ctx, noise, a, b, tile)
        self.kept = (ctx, tuple(a), tuple(b), tile, ops)
        return ops


_SEEN: list = []
_DEBIT_CAP = CW._debit_cap


def _recorded(*args, **kwargs):
    out = _DEBIT_CAP(*args, **kwargs)
    _SEEN.append((out, len(_CHAINS)))
    return out


_CHAINS: list = []
_CHAIN_DEBIT = CW.D.chain_debit
#: the debit's implementation `_chain_recorded` calls: the scalar reference, or `fast_tile`'s batch form
_IMPL = {"chain_debit": _CHAIN_DEBIT}


def _chain_recorded(*args, **kwargs):
    out = _IMPL["chain_debit"](*args, **kwargs)
    _CHAINS.append(out)
    return out


class TileIndex:
    """The workload's tile enumeration as `audit.tile_refs` numbers it (each matmul's `scheme.tiles` in order), without a
    `TileRef` a tile: (matmul, tile) of an id, and each matmul's ids and areas."""

    def __init__(self, scheme, records):
        grids: dict[Shape, list] = {}
        self.grids, self.first, n = [], [], 0
        for r in records:
            self.grids.append(grids.setdefault(r.matmul.shape, scheme.tiles(r.matmul.shape)))
            self.first.append(n)
            n += len(self.grids[-1])
        self.count = n

    def __call__(self, tid: int):
        pos = bisect.bisect_right(self.first, tid) - 1
        return pos, self.grids[pos][tid - self.first[pos]]

    def of(self, pos: int) -> tuple[list[int], list[int]]:
        grid = self.grids[pos]
        return list(range(self.first[pos], self.first[pos] + len(grid))), [len(t.rows) * len(t.cols) for t in grid]


def load(path: Path, light: bool = False, coins: CoinCommitment | None = None) -> dict:
    """The pass as `verify_run.verify` reads it: the epoch from its salt coin under `coins` (the epoch's published coin
    commitment), or a recorded pass's beacon (`pouw_pearl_c_device.manifest_epoch`), and the weights' roots, the records
    held to their published root, the transcript's roots and the verifier's tile enumeration.  `light`: the manifest alone (no verifier,
    no file of the pass is opened), for reading recorded lines back (``--resumed-only``)."""
    raw = path.read_bytes()
    man = json.loads(raw)
    PD.manifest_schema(man, recorded=True)
    scheme = PD.scheme_of(man["scheme"], **{k: tuple(v) if isinstance(v, list) else v for k, v in man.get("instance", {}).items()})
    matmuls = tuple(Matmul(u["index"], Shape(u["m"], u["k"], u["n"]), u["weight"]) for u in man["matmuls"])
    epoch, _ = PD.manifest_epoch(man, coins, recorded=True)
    records = [PW.UnitRecord(mm, u["real_rows"], u["k_kept"], tuple(u["excluded_a"]),
                             tuple(man["weights"][str(mm.weight)]["excluded_b"])) for mm, u in zip(matmuls, man["matmuls"])]
    splits = {w: PW.Split(man["weights"][str(w)]["k"]) for w in epoch.workload.weights()}
    reason = CW.check_records(epoch, splits, records)
    if reason is None and PW.records_root(epoch.salt, epoch.weights_root, splits, records).hex() != man["records_root"]:
        reason = "the records are not the ones whose root the pass published"
    if reason is not None:
        raise ValueError(reason)
    c = man["commitment"]
    index = TileIndex(scheme, records)
    if index.count != c["count"]:
        raise ValueError(f"the commitment counts {c['count']} tiles; the workload has {index.count}")
    sha = hashlib.sha256(raw).hexdigest()
    if light:
        return {"man": man, "sha256": sha, "scheme": scheme, "epoch": epoch, "records": records, "tile": index}
    transcript = Publication(tuple(bytes.fromhex(u["root_a"]) for u in man["matmuls"]), Commitment(bytes.fromhex(c["root"]), c["count"]), ())
    view = _Kept(scheme)
    verifier = Verifier(view, epoch)
    if len(verifier.refs) != c["count"]:
        raise ValueError(f"the commitment counts {c['count']} tiles; the workload has {len(verifier.refs)}")
    return {"man": man, "sha256": sha, "scheme": scheme, "view": view, "epoch": epoch, "tile": index,
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


def one(item) -> dict:
    """One tile's line (`item` a tile id, or a fresh trial's (tile id, matmul, draw, salt)), on the batch forms when they are
    installed; every `cross_check`-th tile id (a fresh trial: draw) is replayed again on the scalar reference, and any field
    that differs is recorded (``cross_check_failed``), which fails the run."""
    out = _measure(item)
    every = _JOB.get("cross_check", 0)
    if _JOB.get("undo") is not None and every and (item if isinstance(item, int) else item[2]) % every == 0:
        _JOB["undo"]()
        _IMPL["chain_debit"] = _CHAIN_DEBIT
        try:
            ref = _measure(item)
        finally:
            _JOB["undo"] = FT.install(_JOB["scheme"], debit=False)
            _IMPL["chain_debit"] = FT.chain_debit
        diff = {k: [out.get(k), ref.get(k)] for k in set(out) | set(ref) if k not in ("seconds", "replay") and out.get(k) != ref.get(k)}
        out["cross_checked"] = True
        if diff:
            out["cross_check_failed"] = diff
    return out


def _openings_ok(job: dict, opened) -> bool:
    """`Verifier.check_tile`'s A and B openings alone: the committed rows, without the tile's digest under the pass's salt."""
    epoch, ref = job["epoch"], opened.ref
    u = epoch.workload.matmuls[ref.matmul]
    h = commitment_hash(job["scheme"])
    da = rows_domain(activation_binding(epoch.salt, u.index), u.index, u.shape.m, h)
    db = rows_domain(weight_binding(u.weight), -2, u.shape.n, h)
    limits = job["verifier"].limits
    return ([o.position for o in opened.a] == list(ref.tile.rows) and [o.position for o in opened.b] == list(ref.tile.cols)
            and all(verify_opening(da, Commitment(job["transcript"].activation_roots[ref.matmul], u.shape.m), o, ROW_SCHEMA, limits)
                    for o in opened.a)
            and all(verify_opening(db, Commitment(epoch.weight_roots[u.weight], u.shape.n), o, ROW_SCHEMA, limits) for o in opened.b))


def _measure(item) -> dict:
    """A tile id: the tile as the audit checks it under the pass's salt.  A fresh trial (tile id, matmul, draw, salt): the
    tile's committed rows (their openings held to the roots) under its own salt, as `HonestTileCapRate` draws one: the noise,
    the encoding and the tile rule (admissibility is the work law's, `_TileRuleView`'s), with no digest (the prover committed
    to none under this salt)."""
    j = _JOB
    scheme, view, epoch, transcript, verifier, records = (j[x] for x in ("scheme", "view", "epoch", "transcript", "verifier", "records"))
    t0 = time.perf_counter()
    tid, fresh = (item, None) if isinstance(item, int) else (item[0], item[1:])
    ref = verifier.refs[tid]
    rec, u = records[ref.matmul], epoch.workload.matmuls[ref.matmul]
    s = u.shape
    out = {"tile": tid, "matmul": ref.matmul, "m": s.m, "k": s.k, "n": s.n, "row0": ref.tile.rows[0], "rows": len(ref.tile.rows),
           "col0": ref.tile.cols[0], "cols": len(ref.tile.cols), "manifest_sha256": j["sha256"],
           "replay": "scalar" if j.get("undo") is None or _IMPL["chain_debit"] is _CHAIN_DEBIT else "batch"}
    opened, h = j["prover"].open(epoch, tid), commitment_hash(scheme)
    a, b = [opened_value(h, o.value) for o in opened.a], [opened_value(h, o.value) for o in opened.b]
    if fresh is not None:
        _, draw, salt = fresh
        out.update(draw=draw, salt=salt.hex(), openings_ok=_openings_ok(j, opened))
        if not out["openings_ok"]:
            out.update(ok=False, reason="openings", seconds=round(time.perf_counter() - t0, 1))
            return out
        ctx = Context(salt, u, transcript.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
        ops = scheme.encode(ctx, scheme.noise(ctx), a, b, ref.tile)
        if ops is None or not view.admissible(ctx, ops):
            out.update(ok=False, reason="inadmissible inputs", seconds=round(time.perf_counter() - t0, 1))
            return out
    else:
        view.kept = None
        v = verifier.check_tile(transcript, opened)
        out["openings_ok"] = v.ok
        if not v.ok:
            out.update(ok=False, reason=v.reason, seconds=round(time.perf_counter() - t0, 1))
            return out
        ctx = Context(epoch.salt, u, transcript.activation_roots[ref.matmul], epoch.weight_roots[u.weight])
        kept = view.kept
        ops = kept[4] if kept is not None and kept[:4] == (ctx, tuple(a), tuple(b), ref.tile) else scheme.encode(ctx, scheme.noise(ctx), a, b, ref.tile)
    _SEEN.clear()
    _CHAINS.clear()
    reason = CW.check_opened(scheme, rec, ref.tile, a, b, ops)
    rows, cols = rec.credited(ref.tile)
    out.update(ok=reason is None, reason=reason, credited_rows=len(rows), credited_cols=len(cols), macs=len(rows) * len(cols) * rec.k_kept)
    if rows and cols:
        # the tile rule's first (debit, cap) is the tile's; under the row cap (-h3) one per credited A row follows it
        tile, chains = _SEEN[0] if _SEEN else (_DEBIT_CAP(scheme, rec, ref.tile, ops, rows, cols), len(_CHAINS))
        out.update(tile_numbers(scheme, *tile))
        if chains:
            out["atoms"], out["later_zero_promotions"] = _CHAINS[chains - 1]
        if len(_SEEN) > 1:
            widen = CW.row_cap_multiple(scheme, u.shape.k)
            out["row_ratio_max"] = max(float(d / (c * widen)) for (d, c), _ in _SEEN[1:])
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def tile_numbers(scheme, debit: Fraction, cap: Fraction) -> dict:
    """The tile rule's (debit, cap) as a tile's line records them: their ratio, and the debit's share of the credited cells'
    credit, the cap over ρ.  That credit is `clear_work.tile_cap`'s `creditDevRev1` share, so the first promotion add is
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


def measure(path: Path, per: int, seed: int, jobs: int, out_dir: Path, opts: dict, label: str | None = None) -> dict:
    """One pass: its sample measured over `jobs` forked workers, a line a tile as each finishes, then its summary."""
    label, t0 = label or path.parent.name, time.perf_counter()
    job = load(path, coins=opts.get("coins"))
    strata, picks = sample(job, per, seed)
    print(f"{label}: {job['man'].get('run')} {job['scheme'].name} cap {job['scheme'].cap}: {len(picks)} tiles of "
          f"{len(job['verifier'].refs)} in {len(strata)} strata (loaded in {time.perf_counter() - t0:.0f} s)", flush=True)
    tiles = list(_run_tiles(label, job, picks, jobs, out_dir / f"{label}.tiles.jsonl", opts).values())
    out = summarize(label, job, strata, tiles)
    print(json.dumps({k: v for k, v in out.items() if k != "strata"}), flush=True)
    return out


def line_key(t: dict):
    """A line's key: its tile id, or a fresh trial's (matmul, draw)."""
    return t["tile"] if "draw" not in t else (t["matmul"], t["draw"])


def _item_key(item):
    return item if isinstance(item, int) else (item[1], item[2])


def resumed(paths: list[str], sha256: str) -> dict:
    """The tile lines of earlier runs on this pass, by tile id, from ``--resume FILE[=MANIFEST_SHA256]``: a line counts when it
    names this manifest (``manifest_sha256``), or names none and its file was given with this manifest's digest (lines
    written before tile lines named their pass), and its openings were checked.  A line whose cross-check failed is never
    reused."""
    out: dict[int, dict] = {}
    for spec in paths:
        name, _, given = spec.partition("=")
        for line in Path(name).read_text().splitlines():
            t = json.loads(line)
            if t.get("manifest_sha256", given) == sha256 and "openings_ok" in t and "cross_check_failed" not in t:
                out[line_key(t)] = t | {"manifest_sha256": sha256, "resumed_from": t.get("resumed_from", name)}
    return out


def _run_tiles(label: str, job: dict, picks, jobs: int, path: Path, opts: dict) -> dict:
    """Every picked item's line (a tile id, or a fresh trial: `_measure`), by `line_key`: a resumed one as recorded (its
    geometry held to this pass's enumeration, a fresh trial's salt to its own), the others measured over `jobs` forked
    workers, a matmul's tiles in runs of a few so that a worker's per-weight noise caches hit.  Every line is written to
    `path` as it is known.  Raises if a cross-checked tile's batch and scalar replays differ."""
    tile, records = job["tile"], job["records"]
    prior = resumed(opts.get("resume") or [], job["sha256"])
    def tile_of(item):
        return item if isinstance(item, int) else item[0]
    got: dict = {}
    for item in picks:
        t = prior.get(_item_key(item))
        if t is not None:
            tid = tile_of(item)
            pos, tl = tile(tid)
            if not isinstance(item, int) and t["salt"] != item[3].hex():
                raise ValueError(f"{label}: resumed trial {line_key(t)} has another salt (another --fresh-source)")
            if (t["tile"], t["matmul"], t["row0"], t["rows"], t["col0"], t["cols"]) != (
                    tid, pos, tl.rows[0], len(tl.rows), tl.cols[0], len(tl.cols)):
                raise ValueError(f"{label}: resumed line {line_key(t)} is not this pass's tile {tid}")
            got[_item_key(item)] = t
    todo = sorted((x for x in picks if _item_key(x) not in got),
                  key=lambda x: (-records[tile(tile_of(x))[0]].matmul.shape.k, tile(tile_of(x))[0], _item_key(x)))
    if opts.get("resumed_only"):
        print(f"{label}: {len(got)} of {len(picks)} resumed; {len(todo)} unmeasured, left out (--resumed-only)", flush=True)
        path.write_text("".join(json.dumps(t) + "\n" for t in got.values()))
        return got
    print(f"{label}: {len(got)} of {len(picks)} tiles resumed, {len(todo)} to measure "
          f"({'scalar' if opts.get('scalar') else 'batch'} replay, cross-check every {opts.get('cross_check') or 'no'} tile id)", flush=True)
    _JOB.clear()
    _JOB.update(job, cross_check=opts.get("cross_check", 0), undo=None)
    if not opts.get("scalar"):
        _JOB["undo"] = FT.install(job["scheme"], debit=False)
        _IMPL["chain_debit"] = FT.chain_debit
    gc.collect()
    # the workers share the parent's tile enumeration copy-on-write; frozen, no collection in them touches its pages
    gc.freeze()
    failed = []
    try:
        with open(path, "w") as fh:
            for t in got.values():
                fh.write(json.dumps(t) + "\n")
            fh.flush()
            if todo:
                with multiprocessing.get_context("fork").Pool(max(1, min(jobs, len(todo)))) as pool:
                    for n, t in enumerate(pool.imap_unordered(one, todo, chunksize=max(1, min(4, len(todo) // max(1, jobs)))), 1):
                        got[line_key(t)] = t
                        fh.write(json.dumps(t) + "\n")
                        fh.flush()
                        if "cross_check_failed" in t:
                            failed.append(line_key(t))
                        if n % 32 == 0 or n == len(todo):
                            print(f"{label}: {n}/{len(todo)} measured, {sum(not x['ok'] for x in got.values())} rejected, "
                                  f"{sum(bool(x.get('cross_checked')) for x in got.values())} cross-checked, "
                                  f"worst share {max((x['share'] for x in got.values() if 'share' in x), default=None)}", flush=True)
    finally:
        gc.unfreeze()
        if _JOB.get("undo") is not None:
            _JOB["undo"]()
        _IMPL["chain_debit"] = _CHAIN_DEBIT
    if failed:
        raise RuntimeError(f"{label}: the batch replay differs from the scalar reference on tiles {failed}")
    return got


def pick_inputs(job: dict, per_linear: int, census_max: int, draws: int, seed: int) -> list[dict]:
    """Per-input mode: `per_linear` seeded inputs (one matmul each, one linear's GEMM in the pass) of each (k, n), each with
    its tiles' weights.  An input of at most `census_max` tiles is measured in full; a larger one by `draws` work-weighted
    draws with replacement, as the audit draws.  A tile's weight is its area (rows × cols): within one input every tile has
    the same k, so its share of the input's `W_ref` is its share of the cells."""
    groups: dict[tuple[int, int], list[int]] = {}
    for pos, rec in enumerate(job["records"]):
        groups.setdefault((rec.matmul.shape.k, rec.matmul.shape.n), []).append(pos)
    rng = np.random.default_rng(seed)
    out = []
    for kn in sorted(groups):
        for pos in sorted(rng.choice(groups[kn], size=min(per_linear, len(groups[kn])), replace=False).tolist()):
            ids, area = job["tile"].of(pos)
            if len(ids) <= census_max:
                method, drawn = "census", list(ids)
            else:
                method = "sampled"
                drawn = [ids[i] for i in rng.choice(len(ids), size=draws, replace=True, p=np.array(area) / sum(area))]
            mm = job["records"][pos].matmul
            out.append({"matmul": pos, "weight": mm.weight, "m": mm.shape.m, "k": mm.shape.k, "n": mm.shape.n, "NT": len(ids),
                        "W": sum(area), "w_max": max(area), "area": dict(zip(ids, area)), "method": method, "drawn": drawn})
    return out


def fresh_trials(job: dict, row: dict, draws: int, source: bytes) -> list[tuple[int, int, int, bytes]]:
    """An input's fresh trials, (tile id, matmul, draw, salt): `draws` work-weighted draws of its tiles with replacement,
    each under a salt of its own, both derived from `source` (recorded OS randomness drawn after the pass, so independent
    of its activations) under a domain naming the table and a context naming the input.  Draw d is `key.uniform(W, d)`
    over the tiles' cumulative credited cells, as `clear_work.draw` maps it: within one input every cell weighs the
    same `W_ref`, so the draw is exactly proportional to `TR.Wref`, as `HonestTileCapRate`'s tickets are."""
    ctx = {"manifest": job["sha256"], "matmul": row["matmul"]}
    ids = list(row["area"])
    cumulative = list(accumulate(row["area"][t] for t in ids))
    key = derive(source, FRESH, ctx | {"use": "tiles"})
    picks = [bisect.bisect_right(cumulative, key.uniform(cumulative[-1], d)) for d in range(draws)]
    return [(ids[i], row["matmul"], d, derive(source, FRESH, ctx | {"use": "salt", "draw": d}).value) for d, i in enumerate(picks)]


def clopper_pearson_upper(x: int, n: int, delta: float) -> float:
    """The exact one-sided upper limit on a Bernoulli rate from x successes in n independent trials: the p at which
    P[Bin(n, p) <= x] = δ (1 - δ^(1/n) at x = 0)."""
    if x >= n:
        return 1.0
    if x == 0:
        return -math.expm1(math.log(delta) / n)

    def cdf(p: float) -> float:
        logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)
                for i in range(x + 1)]
        top = max(logs)
        return math.exp(top) * sum(math.exp(v - top) for v in logs)

    lo, hi = x / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > delta else (lo, mid)
    return hi


def _bernstein_upper(x: float, w_max: float, total: float, delta: float) -> float:
    """An upper limit on the mean work-weighted rate μ/W of independent tiles from their observed weighted failures x:
    Bernstein's lower tail, P[X <= μ - t] <= exp(-t² / (2·w_max·μ + 2·w_max·t/3)), solved for the largest μ it allows."""
    L = math.log(1 / delta)
    def ok(mu):
        return mu <= x or (mu - x) ** 2 <= L * (2 * w_max * mu + 2 * w_max * (mu - x) / 3)
    lo, hi = x, total
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return min(1.0, hi / total)


def input_bound(row: dict, delta: float) -> tuple[float, str]:
    """The completeness citation's bound: each input's rate p = E[X] (X its work-weighted failing fraction under the
    tiles' noise) at a simultaneous upper confidence limit (δ = 0.05/m per input), assuming the input's tiles fail
    independently (each tile's noise is the oracle at points of its own).  With a = w_max/W:
    - a census with no failure: ln(1/δ)·a (P[none fail] = Π(1 - p_g) <= exp(-p/a));
    - n work-weighted draws with no failure: ln(1/δ)·a / (1 - exp(-n·a)) (E[(1 - X)^n] <= E[exp(-nX)] <= exp(-p(1 - exp(-n·a))/a));
      Clopper–Pearson on the draws would bound the realized X, not p;
    - a census with failures: Bernstein on the weighted sum;
    - draws with failures: no bound here (1): such an input must be measured in full;
    - fresh trials (``--fresh-draws``: each a work-weighted draw under a salt of its own, so independent Bernoulli trials of
      p itself): Clopper–Pearson, with or without failures."""
    if row["method"] == "fresh":
        return clopper_pearson_upper(row["failures"], row["drawn"], delta), "fresh-salt draws, Clopper–Pearson"
    a = row["w_max"] / row["W"]
    if row["failures"] == 0:
        if row["method"] == "census":
            return min(1.0, math.log(1 / delta) * a), "census, rule of three (work-weighted)"
        return min(1.0, math.log(1 / delta) * a / -math.expm1(-row["drawn"] * a)), "sampled, rule of three (work-weighted draws)"
    if row["method"] == "census":
        return _bernstein_upper(row["failures_weighted"], row["w_max"], row["W"], delta), "census, Bernstein"
    return 1.0, "sampled with failures: measure this input in full"


def measure_inputs(path: Path, per_linear: int, census_max: int, draws: int, seed: int, jobs: int, out_dir: Path, opts: dict,
                   label: str | None = None) -> list[dict]:
    """Per-input mode on one pass: every chosen input's tiles measured over `jobs` forked workers, a line a tile, then a row
    an input with its failures at the scheme's cap (a tile whose openings hold and the tile rule rejects), `W` and
    `w_max`.  The bound is added once every pass's inputs are known (`m`)."""
    label = label or path.parent.name
    job = load(path, light=bool(opts.get("resumed_only")), coins=opts.get("coins"))
    inputs = pick_inputs(job, per_linear, census_max, draws, seed)
    fresh = opts.get("fresh_draws")
    if fresh:
        for row in inputs:
            row["method"], row["drawn"] = "fresh", fresh_trials(job, row, fresh, opts["fresh_source"])
        picks = [x for row in inputs for x in row["drawn"]]
    else:
        picks = sorted({t for row in inputs for t in row["drawn"]})
    print(f"{label}: {job['man'].get('run')} {job['scheme'].name}: {len(inputs)} inputs, {len(picks)} "
          f"{'fresh trials' if fresh else 'tiles'}", flush=True)
    got = _run_tiles(label, job, picks, jobs, out_dir / f"{label}.{'fresh' if fresh else 'input'}-tiles.jsonl", opts)
    rows = []
    for row in inputs:
        seen = [got[_item_key(x)] for x in row["drawn"] if _item_key(x) in got]
        if row["method"] == "census" and len(seen) < len(row["drawn"]):
            raise ValueError(f"{label}: census input {row['matmul']} has {len(row['drawn']) - len(seen)} tiles unmeasured")
        fail = [t for t in seen if t["openings_ok"] and not t["ok"]]
        rows.append({k: v for k, v in row.items() if k not in ("area", "drawn")} | {
            "pass": label, "run": job["man"].get("run"), "manifest_sha256": job["sha256"], "drawn": len(seen),
            "unmeasured_draws": len(row["drawn"]) - len(seen),
            "openings_failed": sum(not t["openings_ok"] for t in seen), "failures": len(fail),
            "failures_weighted": sum(row["area"][t["tile"]] for t in (fail if fresh else {t["tile"]: t for t in fail}.values())),
            "share_max": max((t["share"] for t in seen if "share" in t), default=None),
            "reasons": sorted({t["reason"] for t in fail if t.get("reason")})})
    return rows


def write_inputs(rows: list[dict], seed: int, census_max: int, draws: int, out_dir: Path, fresh: dict | None = None) -> int:
    """Each input's bound at δ = 0.05/m, the cited maximum and the run's result, into ``<out>/inputs.json``; `fresh`
    (``fresh_draws``, ``fresh_source``) when its inputs were measured by fresh trials."""
    delta = 0.05 / len(rows)
    for r in rows:
        r["upper"], r["bound_method"] = input_bound(r, delta)
    worst = max(rows, key=lambda r: r["upper"])
    doc = {"tool": "pouw_pearl_c_served_debit/per-input", "seed": seed, "confidence": 0.95, "m": len(rows),
           "delta_per_input": delta, "census_max": census_max, "draws": draws,
           "cited_upper": worst["upper"], "cited_input": {k: worst[k] for k in ("pass", "matmul", "weight", "m", "k", "n")},
           "independence": "every bound assumes an input's tiles fail independently (each tile's noise is the oracle at "
                           "points of its own)", "inputs": rows}
    if fresh:
        doc |= fresh | {"independence": "each trial is a work-weighted draw of the input's tiles under a salt of its own, both "
                                        "derived from fresh_source after the pass (verity.ml.randomness.derive, " + FRESH + "): "
                                        "independent Bernoulli trials of the input's rate, the activations fixed"}
    path = out_dir / "inputs.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    ok = all(r["openings_failed"] == 0 for r in rows)
    run_result.write(path, ok, "pearl-c served debit, per input: every measured tile's openings and checked values hold",
                     f"{len(rows)} inputs, {sum(r['failures'] for r in rows)} rejected tiles; the worst input's 95% "
                     f"simultaneous upper limit {worst['upper']:.4g} ({worst['bound_method']})",
                     [("inputs", len(rows), "count"), ("failures", sum(r["failures"] for r in rows), "count"),
                      ("cited_upper", worst["upper"], "share")])
    return 0 if ok else 1


def labels_of(manifests: list[Path]) -> list[str]:
    """Each manifest's label, its directory's name (``decode``, ``prefill``), qualified by the run's directory when two
    manifests share one."""
    names = [m.parent.name for m in manifests]
    return [f"{m.parent.parent.name}-{n}" if names.count(n) > 1 else n for m, n in zip(manifests, names)]


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
    p.add_argument("--resume", nargs="+", default=[], metavar="FILE[=MANIFEST_SHA256]",
                   help="reuse an earlier run's tile lines on the same pass (a *.tiles.jsonl or *.input-tiles.jsonl); "
                        "=SHA256 names the manifest of lines that do not name one")
    p.add_argument("--resumed-only", action="store_true",
                   help="per-input mode: measure nothing; read the inputs' tiles from --resume alone (the manifests, not the "
                        "pass's files), leaving a sampled input's unmeasured draws out (they were not chosen by their outcome) "
                        "and refusing a census input with any tile unmeasured")
    p.add_argument("--scalar", action="store_true", help="replay every tile on the scalar reference only (no batch forms)")
    p.add_argument("--cross-check", type=int, default=64, metavar="N",
                   help="replay every tile whose id is a multiple of N on the scalar reference too, and fail on any difference "
                        "(0: none)")
    p.add_argument("--fresh-draws", type=int, default=0, metavar="N",
                   help="per-input mode: measure each input by N work-weighted draws, each under a fresh salt of its own "
                        "(independent trials of its rate, bounded by Clopper–Pearson) instead of its census or draws")
    p.add_argument("--fresh-source", default=None, metavar="HEX",
                   help="the 32-byte source the fresh draws and salts derive from: recorded OS randomness drawn after the "
                        "pass, fixed in the command before the run (required with --fresh-draws)")
    p.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--coins", type=Path, help="the epoch's coin commitment as its challenger published it (a pass with a salt coin)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if a.rebound is not None:
        doc = json.loads(a.rebound.read_text())
        out_dir = a.out or a.rebound.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return write_inputs(doc["inputs"], doc["seed"], doc["census_max"], doc["draws"], out_dir,
                            {k: doc[k] for k in ("fresh_draws", "fresh_source") if k in doc} or None)
    if not a.manifests:
        p.error("give one or more manifests")
    if (a.tiles is None) == (a.per_input is None):
        p.error("give --tiles (pooled strata) or --per-input (each input bounded)")
    if a.tiles is not None and len(a.tiles) != len(a.manifests):
        p.error("one --tiles per manifest")
    out_dir = a.out or Path(settings.RESEARCH_RUN_DIR or ".") / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"resume": a.resume, "scalar": a.scalar, "cross_check": a.cross_check, "resumed_only": a.resumed_only,
            "coins": CoinCommitment.from_json(json.loads(a.coins.read_text())) if a.coins else None}
    if a.resumed_only and (a.per_input is None or not a.resume):
        p.error("--resumed-only takes --per-input and --resume")
    fresh = None
    if a.fresh_draws:
        if a.per_input is None:
            p.error("--fresh-draws takes --per-input")
        if a.fresh_source is None:
            p.error("--fresh-draws takes --fresh-source HEX, fixed before the run (lean's condition on the fresh-trial ε)")
        source = bytes.fromhex(a.fresh_source)
        if len(source) != 32:
            p.error("--fresh-source is 32 bytes")
        fresh = {"fresh_draws": a.fresh_draws, "fresh_source": source.hex()}
        opts |= {"fresh_draws": a.fresh_draws, "fresh_source": source}
        print(f"fresh draws: {a.fresh_draws} an input from source {source.hex()}", flush=True)
    labels = labels_of(a.manifests)
    if a.per_input is not None:
        CW._debit_cap, CW.D.chain_debit = _recorded, _chain_recorded
        try:
            rows = [r for path, label in zip(a.manifests, labels)
                    for r in measure_inputs(path, a.per_input, a.census_max, a.draws, a.seed, a.jobs, out_dir, opts, label)]
        finally:
            CW._debit_cap, CW.D.chain_debit = _DEBIT_CAP, _CHAIN_DEBIT
        return write_inputs(rows, a.seed, a.census_max, a.draws, out_dir, fresh)
    CW._debit_cap, CW.D.chain_debit = _recorded, _chain_recorded
    try:
        passes = [measure(path, per, a.seed, a.jobs, out_dir, opts, label) for path, per, label in zip(a.manifests, a.tiles, labels)]
    finally:
        CW._debit_cap, CW.D.chain_debit = _DEBIT_CAP, _CHAIN_DEBIT
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
