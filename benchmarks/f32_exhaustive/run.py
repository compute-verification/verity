"""Every binary32 input of a reference primitive of `verity.definitions.registry.ref_prims`, in one of two modes.

    uv run --with gmpy2 benchmarks/f32_exhaustive/run.py log --out DIR [--procs N] [--lo U --hi U]
    uv run benchmarks/f32_exhaustive/run.py boolean F32Sqrt --out DIR [--procs N] [--lo U --hi U]

``log`` and ``exp`` run a correctly rounded evaluator (`f32_log_rn_bits`, `f32_exp_rn_bits`) against MPFR's correctly rounded
result at 24 bits (gmpy2, emin -148, emax 128, subnormalized, nearest even), NaN canonical 0x7FC00000. ``boolean WORD`` runs the
Boolean version of `ref_prims.WORD` (`verity.definitions.boolean.boolean_version`), bit-sliced
(`verity.circuits.evaluation.bits.evaluate_words`), against the word primitive's own evaluator; WORD takes one 32-bit word
and gives one.

Writes DIR/exhaustive-<fn>.json (``exhaustive-boolean-WORD.json``): the inputs checked, the mismatches (up to 1000, each the
input and the two words) and their count. A pass is a mismatch count of 0 over [0, 2^32).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import struct
import sys
import time
from functools import cache

CHUNK = 1 << 18
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
    from verity.definitions.registry import ref_prims as RP

    return {"log": RP.f32_log_rn_bits, "exp": RP.f32_exp_rn_bits}[fn].__wrapped__


@cache
def _boolean(name: str):
    """(`ref_prims`' word primitive ``name``, its Boolean version), refused unless it maps one 32-bit word to one."""
    from verity.definitions.boolean import boolean_version
    from verity.definitions.registry import ref_prims as RP

    word = getattr(RP, name)
    params, ret = word.signature()
    if [(t.leaves, t.homogeneous_width()) for _, t in params] != [(1, 32)] or (ret.leaves, ret.homogeneous_width()) != (1, 32):
        raise SystemExit(f"{word.id}: not one 32-bit word to one, so not this script's 2^32 inputs")
    return word, boolean_version(word)


def _chunk(args: tuple[str, str | None, int, int]) -> tuple[int, int, list[tuple[int, int, int]]]:
    fn, name, lo, hi = args
    if fn == "boolean":
        import numpy as np

        from verity.circuits.evaluation.bits import evaluate_words

        word, bits = _boolean(name)
        got = evaluate_words(bits, np.arange(lo, hi, dtype=np.uint64).reshape(-1, 1)).reshape(-1).tolist()
        want = [word.evaluate(u) for u in range(lo, hi)]
    else:
        ev = _evaluator(fn)
        got = [ev(u) for u in range(lo, hi)]
        want = [_mpfr_word(fn, u) for u in range(lo, hi)]
    return lo, hi - lo, [(u, g, w) for u, g, w in zip(range(lo, hi), got, want) if g != w]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("fn", choices=("log", "exp", "boolean"))
    ap.add_argument("word", nargs="?", help="boolean: the ref_prims word primitive whose Boolean version is checked (F32Sqrt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--lo", type=lambda s: int(s, 0), default=0)
    ap.add_argument("--hi", type=lambda s: int(s, 0), default=1 << 32)
    a = ap.parse_args(argv)
    if (a.fn == "boolean") != (a.word is not None):
        ap.error("boolean takes a word primitive's name, and only boolean does")
    rec: dict = {"fn": a.fn}
    labels = ("evaluator", "mpfr")
    if a.fn == "boolean":
        word, bits = _boolean(a.word)
        rec.update(word=word.id, boolean=bits.id)
        labels = ("boolean", "word")
    jobs = [(a.fn, a.word, lo, min(lo + CHUNK, a.hi)) for lo in range(a.lo, a.hi, CHUNK)]
    every = max(1, len(jobs) // 64)
    checked, bad, t0 = 0, [], time.monotonic()
    print(f"{a.fn} {a.word or ''}: [{a.lo:#x}, {a.hi:#x}) in {len(jobs)} chunks on {a.procs} processes", flush=True)
    with mp.Pool(a.procs) as pool:
        for i, (lo, n, b) in enumerate(pool.imap_unordered(_chunk, jobs), 1):
            checked += n
            bad += b
            if b or i % every == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} chunks, {checked} inputs, {len(bad)} mismatches, {time.monotonic() - t0:.0f} s"
                      + (f"; first in chunk {lo:#x}: {b[0]}" if b else ""), flush=True)
    os.makedirs(a.out, exist_ok=True)
    rec.update(lo=a.lo, hi=a.hi, checked=checked, mismatches=len(bad),
               first_mismatches=[{"x": f"{u:#010x}", labels[0]: f"{g:#010x}", labels[1]: f"{w:#010x}"}
                                 for u, g, w in sorted(bad)[:1000]])
    name = f"exhaustive-{a.fn}" + (f"-{a.word}" if a.word else "")
    with open(os.path.join(a.out, f"{name}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({k: rec[k] for k in ("fn", "word", "boolean", "checked", "mismatches") if k in rec}), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
