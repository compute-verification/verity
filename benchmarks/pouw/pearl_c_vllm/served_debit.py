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
  rejection); the debit's share of the cells' credit, and that share at Lean's rev1 credit (`creditDevRev1`: each credited
  cell's credit less its first promotion add, one FADD at the scheme's prices).
- **The estimates** weight each stratum by its share of the pass's tiles and of its credited MACs (credited rows × credited
  columns × kept k, from the records), with the share over each cap within the stratum taken from its sample.

Writes ``<out>/<label>.tiles.jsonl`` (a line a tile, as each finishes; the label is the manifest's directory) and
``<out>/summary.json``.  Its gate is that every sampled tile's openings and checked values hold (`Verifier.check_tile`):
the measurement is of the committed pass.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    reason = PW.check_opened(scheme, rec, ref.tile, a, b, ops)
    rows, cols = rec.credited(ref.tile)
    out.update(ok=reason is None, reason=reason, credited_rows=len(rows), credited_cols=len(cols), macs=len(rows) * len(cols) * rec.k_kept)
    if rows and cols:
        debit, cap = _SEEN[-1] if _SEEN else _DEBIT_CAP(scheme, rec, ref.tile, ops, rows, cols)
        credit = cap / scheme.cap
        credit_rev1 = credit - scheme.prices.fadd * len(rows) * len(cols)
        out.update(debit=float(debit), cap=float(cap), ratio=float(debit / cap), share=float(debit / credit),
                   share_rev1=float(debit / credit_rev1), debit_exact=str(debit), credit_exact=str(credit))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


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
            row.update(share_max=max(sh), share_median=statistics.median(sh), share_rev1_max=max(t["share_rev1"] for t in got),
                       worst_tile=max(got, key=lambda t: t["share"])["tile"],
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
            "share_max": worst["share"] if worst else None, "share_rev1_max": max((t["share_rev1"] for t in measured), default=None),
            "ratio_max": worst["ratio"] if worst else None,
            "worst": {k: worst[k] for k in ("tile", "matmul", "m", "k", "n", "row0", "col0", "share", "share_rev1", "ratio")} if worst else None,
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("manifests", type=Path, nargs="+")
    p.add_argument("--tiles", type=int, nargs="+", required=True, help="tiles per stratum, one per manifest")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    if len(a.tiles) != len(a.manifests):
        p.error("one --tiles per manifest")
    out_dir = a.out or Path(os.environ.get("RESEARCH_RUN_DIR", ".")) / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    PW._debit_cap = _recorded
    try:
        passes = [measure(path, per, a.seed, a.jobs, out_dir) for path, per in zip(a.manifests, a.tiles)]
    finally:
        PW._debit_cap = _DEBIT_CAP
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
