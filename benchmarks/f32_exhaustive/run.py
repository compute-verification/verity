"""Every binary32 input of a correctly rounded reference primitive (`verity.ml.definitions.registry.ref_prims`'s
`f32_log_rn_bits`, `f32_exp_rn_bits`) against MPFR's correctly rounded result at 24 bits (gmpy2, emin -148, emax 128,
subnormalized, nearest even), NaN canonical 0x7FC00000.

    uv run --with gmpy2 benchmarks/f32_exhaustive/run.py log --out DIR [--procs N] [--lo U --hi U]

Writes DIR/exhaustive-<fn>.json: the inputs checked, the mismatches (up to 1000, each input, evaluator and MPFR word) and their
count. A pass is a mismatch count of 0 over [0, 2^32).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time

CHUNK = 1 << 20
NAN = 0x7FC00000


def _ctx():
    import gmpy2

    return gmpy2.context(precision=24, emin=-148, emax=128, subnormalize=True, round=gmpy2.RoundToNearest)


def _mpfr_word(fn: str, u: int) -> int:
    import gmpy2

    x = struct.unpack("<f", struct.pack("<I", u))[0]
    with _ctx():
        r = getattr(gmpy2, fn)(gmpy2.mpfr(x))
        if gmpy2.is_nan(r):
            return NAN
        return struct.unpack("<I", struct.pack("<f", float(r)))[0]


def _evaluator(fn: str):
    from verity.ml.definitions.registry import ref_prims as RP

    return {"log": RP.f32_log_rn_bits, "exp": RP.f32_exp_rn_bits}[fn].__wrapped__


def _chunk(args: tuple[str, int, int]) -> tuple[int, int, list[tuple[int, int, int]]]:
    fn, lo, hi = args
    ev = _evaluator(fn)
    bad = []
    for u in range(lo, hi):
        a, b = ev(u), _mpfr_word(fn, u)
        if a != b:
            bad.append((u, a, b))
    return lo, hi - lo, bad


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fn", choices=("log", "exp"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--lo", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--hi", type=lambda s: int(s, 0), default=1 << 32)
    a = ap.parse_args(argv)
    jobs = [(a.fn, lo, min(lo + CHUNK, a.hi)) for lo in range(a.lo, a.hi, CHUNK)]
    checked, bad, t0 = 0, [], time.monotonic()
    print(f"{a.fn}: [{a.lo:#x}, {a.hi:#x}) in {len(jobs)} chunks on {a.procs} processes", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, (lo, n, b) in enumerate(pool.imap_unordered(_chunk, jobs), 1):
            checked += n
            bad += b
            if b or i % 64 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chunks, {checked} inputs, {len(bad)} mismatches, {time.monotonic() - t0:.0f} s"
                      + (f"; first in chunk {lo:#x}: {b[0]}" if b else ""), flush=True)
    os.makedirs(a.out, exist_ok=True)
    rec = {"fn": a.fn, "lo": a.lo, "hi": a.hi, "checked": checked, "mismatches": len(bad),
           "first_mismatches": [{"x": f"{u:#010x}", "evaluator": f"{e:#010x}", "mpfr": f"{m:#010x}"}
                                for u, e, m in sorted(bad)[:1000]]}
    with open(os.path.join(a.out, f"exhaustive-{a.fn}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("fn", "checked", "mismatches")}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
