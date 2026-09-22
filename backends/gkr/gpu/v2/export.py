"""The checker-v2 circuits and VU witnesses of ``gkr_export.v2`` exported over BabyBear for the GPU prover.

``gkr_export.v2`` is field-generic (every coefficient and value is reduced mod ``modulus``); its ``FieldApi`` refuses
BabyBear only because the v2 *epilogue* (``Pack.check`` -> ``Epilogue.check`` on the 32-bit word) is SAT as stated
on 31-bit fields (``checker_min.md`` 3.2: the F4 word class; the state-chained VU body is UNSAT over 31-bit fields by
wrap-free transfer).  This module exports it anyway, unchanged, so that the GPU prover can be measured at the v2
shape; every number produced from it is labelled ``INCOMPLETE_CHECKER_DIAGNOSTIC`` naming the epilogue's word
arithmetic as the unverified rows until the limb-split epilogue (``lane/b-air-v2``, ``V2_AIR.md`` 4.5) is adopted.

Files (``bb_export`` formats, ``<QQ`` header then u32 LE): ``units.bin`` (264 columns), ``epilogue.bin`` (16),
``public.bin``; ``circuit.txt`` / ``epilogue.txt`` with ``itable`` lines for the four implicit tables; ``chain.txt``
(``link <in|-> <out> <epi> <init mod p> <zero_nonlast>``); ``manifest.json`` with ``variant: v2``.

    python -m gpu.v2.export export --root fixtures/bench-instances/v1 --tier vu-k1536 --hi 4096 --out DIR --procs 16
    python -m gpu.v2.export export --root ... --tier vu-k1536-neg --out DIR          # DIR/neg/<id>/ per instance
    python -m gpu.v2.export synth --out DIR --vus 2 --steps 3                      # + DIR/negatives (4 chain cases)
    python -m gpu.v2.export negatives --out DIR                                    # a-v2's 30 unit + 4 chain cases
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import struct
import sys
from pathlib import Path

import numpy as np
from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.gkr_export import v2 as V2

P = 2**31 - 2**27 + 1
STEP = 16


class BabyBearV2Api:
    """The ``FieldApi`` surface ``gkr_export.v2`` needs (``synth_v2``, ``unit_negatives``), over BabyBear."""

    name = "babybear"
    variant = "v2"
    modulus = P

    def build_unit(self):
        return V2.build_unit_circuit_v2()

    def build_epilogue(self):
        return V2.build_epilogue_circuit_v2()

    def export_circuit(self, ctx, path: Path) -> None:
        V2.export_circuit_v2(ctx, path, modulus=P)

    def export_extra(self, unit, epi, out: Path) -> None:
        V2.write_chain(V2.chain_spec(unit, epi), out / "chain.txt", P)

    def vu_rows(self, unit, epi, a, b, y16, accs=None, p: Params = REAL, cross_check: bool = True):
        return V2.vu_rows_v2(unit, epi, a, b, y16, accs, p, cross_check, P)


API = BabyBearV2Api()


def write_rows(rows: list[list[int]], path: Path, ncols: int | None = None) -> None:
    ncols = len(rows[0]) if rows else (ncols or 0)
    arr = np.array([[v % P for v in r] for r in rows], dtype="<u4").reshape(len(rows), ncols)
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(rows), ncols))
        f.write(arr.tobytes())


def manifest_for(uctx, ectx, out: Path, **extra) -> dict:
    man = {"field": "babybear", "variant": "v2", "modulus": P, "unit_columns": len(uctx.cols),
           "epilogue_columns": len(ectx.cols), "chain_file": "chain.txt",
           "chain_sha256": hashlib.sha256((out / "chain.txt").read_bytes()).hexdigest(), **extra}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    return man


# ---------------------------------------------------------------------------------------------------------------------
# bench-instances export (multiprocess, as bb_export.export)
# ---------------------------------------------------------------------------------------------------------------------

_UNIT = None
_EPI = None


def _init_worker() -> None:
    global _UNIT, _EPI
    _UNIT = API.build_unit()
    _EPI = API.build_epilogue()


def _one(task):
    vid, a, b, y16, accs, cross_check = task
    try:
        rows, erow = API.vu_rows(_UNIT, _EPI, a, b, y16, accs, cross_check=cross_check)
    except Reject as e:
        return vid, None, None, str(e)
    ub = np.array([[v % P for v in r] for r in rows], dtype="<u4").tobytes()
    eb = np.array([v % P for v in erow], dtype="<u4").tobytes()
    return vid, ub, eb, None


def _write_batch(out: Path, results, y16s, steps, ucols, ecols) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "units.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results) * steps, ucols))
        f.writelines(ub for _, ub, _ in results)
    with open(out / "epilogue.bin", "wb") as f:
        f.write(struct.pack("<QQ", len(results), ecols))
        f.writelines(eb for _, _, eb in results)
    write_rows([[y] for y in y16s], out / "public.bin", 1)


def export(root: Path, tier: str, lo: int, hi: int | None, out: Path, procs: int) -> dict:
    from verity_numerical.bench import instances as I

    tierdata = I.load(tier, root)
    hi = tierdata.n if hi is None else min(hi, tierdata.n)
    uctx, _ = API.build_unit()
    ectx, _ = API.build_epilogue()
    out.mkdir(parents=True, exist_ok=True)
    API.export_circuit(uctx, out / "circuit.txt")
    API.export_circuit(ectx, out / "epilogue.txt")
    API.export_extra(uctx, ectx, out)
    steps = tierdata.K // STEP
    # negatives: no integer cross-check at witness generation, so that a wrong claimed word reaches the prover as an
    # in-domain-looking witness (the chain is the checker's own; the epilogue row carries the claimed word) and the
    # *proof system* has to reject it; a non-finite operand (no T_OP row) or a saturated intermediate still fails here
    cross_check = not tier.endswith("-neg")
    tasks = []
    for vid in range(lo, hi):
        x, w, y = tierdata.instance(vid)
        accs = tierdata.accs(vid) if tierdata.acc is not None else None
        tasks.append((vid, x, w, y, accs, cross_check))
    results = {}
    if procs > 1:
        with mp.get_context("fork").Pool(procs, initializer=_init_worker) as pool:
            for vid, ub, eb, err in pool.imap_unordered(_one, tasks, chunksize=4):
                results[vid] = (ub, eb, err)
    else:
        _init_worker()
        for t in tasks:
            vid, ub, eb, err = _one(t)
            results[vid] = (ub, eb, err)
    extra = {"tier": tier, "range": [lo, hi], "steps": steps, "root": str(root), "cross_check": cross_check,
             "root_manifest_sha256": I.manifest_sha256(root), "rejected_at_witness": {}, "written": []}
    if tier.endswith("-neg"):
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                extra["rejected_at_witness"][vid] = err
                continue
            _write_batch(out / "neg" / str(vid), [(vid, ub, eb)], [int(tierdata.y[vid])], steps, len(uctx.cols), len(ectx.cols))
            extra["written"].append(vid)
        rows = {r[0]: r for r in tierdata.index}
        extra["index"] = {vid: dict(zip(tierdata.columns, rows[vid])) for vid in range(lo, hi)}
    else:
        ok = []
        for vid in range(lo, hi):
            ub, eb, err = results[vid]
            if err is not None:
                extra["rejected_at_witness"][vid] = err
            else:
                ok.append((vid, ub, eb))
        if extra["rejected_at_witness"]:
            print(f"{len(extra['rejected_at_witness'])} positives rejected at witness generation", file=sys.stderr)
        _write_batch(out, ok, [int(tierdata.y[vid]) for vid, _, _ in ok], steps, len(uctx.cols), len(ectx.cols))
        extra["written"] = [vid for vid, _, _ in ok]
    return manifest_for(uctx, ectx, out, **extra)


# ---------------------------------------------------------------------------------------------------------------------
# synthetic VUs + the a-v2 negatives
# ---------------------------------------------------------------------------------------------------------------------

def synth(out: Path, vus: int, steps: int, seed: int) -> dict:
    """``gkr_export.v2.synth_v2`` over BabyBear (honest VUs + the 4 chain-level negatives under ``negatives/``),
    plus ``manifest.json`` and the chain negatives as ``neg/<name>/`` directories the runner can load."""
    import shutil

    info = V2.synth_v2(out, vus, steps, seed, REAL, API)
    uctx, _ = API.build_unit()
    ectx, _ = API.build_epilogue()
    negdir = out / "neg"
    negdir.mkdir(exist_ok=True)
    cases = {"public_wrong_word": ("units.bin", "epilogue.bin", "negatives/public_wrong_word.bin"),
             "false_overflow_flag": tuple(f"negatives/false_overflow_flag/{f}" for f in ("units.bin", "epilogue.bin", "public.bin"))}
    if steps >= 2:
        cases["broken_chain"] = ("negatives/units_broken_chain.bin", "epilogue.bin", "public.bin")
        cases["saturated_intermediate"] = tuple(f"negatives/saturated_intermediate/{f}" for f in ("units.bin", "epilogue.bin", "public.bin"))
    for name, files in cases.items():
        d = negdir / name
        d.mkdir(exist_ok=True)
        for src, dst in zip(files, ("units.bin", "epilogue.bin", "public.bin")):
            shutil.copy(out / src, d / dst)
    return manifest_for(uctx, ectx, out, tier="synth", vus=vus, steps=steps, seed=seed, y16=info["y16"],
                        chain_negatives=sorted(cases))


def negatives(out: Path, seed: int) -> dict:
    """a-v2's transition-unit negatives (``DIR/negatives/<name>.bin``, one mutated unit row each, ``DIR/circuit.txt``)
    and the synth VU with its chain negatives (``DIR/vu/neg/<name>/``), over BabyBear."""
    summary = V2.unit_negatives(out, seed, REAL, P, API)
    uctx, _ = API.build_unit()
    ectx, _ = API.build_epilogue()
    manifest_for(uctx, ectx, out / "vu", tier="synth", vus=2, steps=3, seed=seed, chain_negatives=[d.name for d in sorted((out / "vu" / "neg").iterdir())])
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpu.v2.export")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--root", type=Path, required=True)
    e.add_argument("--tier", default="vu-k1536")
    e.add_argument("--lo", type=int, default=0)
    e.add_argument("--hi", type=int, default=None)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--procs", type=int, default=1)
    s = sub.add_parser("synth")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--vus", type=int, default=2)
    s.add_argument("--steps", type=int, default=3)
    s.add_argument("--seed", type=int, default=0)
    n = sub.add_parser("negatives")
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--seed", type=int, default=20260922)
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        m = export(args.root, args.tier, args.lo, args.hi, args.out, args.procs)
        print(json.dumps({k: v for k, v in m.items() if k not in ("index", "written")}, default=str))
    elif args.cmd == "synth":
        print(json.dumps(synth(args.out, args.vus, args.steps, args.seed), default=str))
    elif args.cmd == "negatives":
        print(json.dumps(negatives(args.out, args.seed), default=str))
    else:
        for name, (ctx, _) in (("unit", API.build_unit()), ("epilogue", API.build_epilogue())):
            nq: dict[str, int] = {}
            for t, _l in ctx.queries:
                nq[t] = nq.get(t, 0) + 1
            print(name, "cols", len(ctx.cols), "wires", len(ctx.wires), "depth", max(ctx.depth), "asserts", len(ctx.asserts), "queries", nq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
