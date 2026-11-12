"""The words' inert-work counts of a Pearl-C call on the H100 or H200: `fast_tile.word_debits` on every word of A′'s
rows by B̃'s, from debit_sm90.cubin (debit_sm90.cu; build.sh builds it into its tree's debit/) through the driver API
(run.py's `Dev`).

  Debit(cubin_bytes).words(a, b) -> (atoms, zeros, bad)
  Debit(cubin_bytes).resident(a, b) -> body, words()'s launches alone on operands put once (for `run.Dev.timed`)
  python benchmarks/pouw/pearl_c_sm90/debit.py --cubin debit_sm90.cubin [--shape R:N:K ...] [--timed R:N:K ...]
      [--resident R:N:K ...] [--reps N] [--out F]

``a`` (rows x k) and ``b`` (n x k) are E4M3 codes (uint8), k a multiple of 128; each output is rows x n: the flagged
atoms and the later +0 promotions of word (i, j) (uint16), and whether it left the finite range (bool), where the counts
are not the reference's and the caller takes the scalar reference's for the word's tile.  The kernels take whole 64 x 32
blocks of words, so A′ and B̃ run padded with zero rows, whose words are dropped; the rows run in chunks whose scratch
(three 32-bit words a window of every word) fits ``budget`` bytes of device memory.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW, BLOCK_ROWS, BLOCK_COLS = 128, 64, 32


def _driver():
    """This directory's run.py, loaded from its file as `pearl_c_sm90_run` once a process: other directories hold a
    run.py too, so its name alone does not say which."""
    name = "pearl_c_sm90_run"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, "run.py"))
        module = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules[name]


run = _driver()


class Debit:
    def __init__(self, cubin: bytes, budget: int = 16 << 30):
        self.dev = run.Dev(cubin)
        self.dev.open()
        self.budget = budget
        self.live: dict[str, tuple[int, int]] = {}

    def _buf(self, name: str, n: int) -> int:
        """Device buffer `name` of at least n bytes, kept for the next chunk."""
        p, size = self.live.get(name, (0, 0))
        if size < n:
            if p:
                self.dev.free(p)
            p, size = self.dev.alloc(n), n
            self.live[name] = (p, size)
        return p

    def _put(self, name: str, arr: np.ndarray) -> int:
        arr = np.ascontiguousarray(arr)
        p = self._buf(name, arr.nbytes)
        self.dev.ck("cuMemcpyHtoD_v2", run.u64(p), C.c_void_p(arr.ctypes.data), arr.nbytes)
        return p

    def _get(self, p: int, shape: tuple[int, int], dtype) -> np.ndarray:
        out = np.empty(shape, dtype)
        self.dev.ck("cuMemcpyDtoH_v2", C.c_void_p(out.ctypes.data), run.u64(p), out.nbytes)
        return out

    def chunk_rows(self, n: int, k: int) -> int:
        return max(BLOCK_ROWS, self.budget // (12 * n * (k // WINDOW)) // BLOCK_ROWS * BLOCK_ROWS)

    @staticmethod
    def _checked(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
        """B̃ padded with zero rows to whole blocks of columns, and its real row count."""
        k, nb = a.shape[1], b.shape[0]
        if a.dtype != np.uint8 or b.dtype != np.uint8 or b.shape[1] != k or k % WINDOW:
            raise ValueError(f"A′ {a.shape} {a.dtype} by B̃ {b.shape} {b.dtype}: uint8 codes, k a multiple of {WINDOW}")
        n = -(-nb // BLOCK_COLS) * BLOCK_COLS
        return (np.concatenate([b, np.zeros((n - nb, k), np.uint8)]) if n != nb else b), nb

    def _chunk(self, pa: int, pb: int, rows: int, n: int, k: int) -> tuple[int, int, int]:
        """The three launches over `rows` (a whole number of blocks) of A′'s rows at pa by B̃'s n rows at pb; the outputs'
        buffers (atoms, zeros, bad)."""
        slots = rows * n
        s0, lo, hi = (self._buf(x, 4 * (k // WINDOW) * slots) for x in ("s0", "lo", "hi"))
        z, bd = self._buf("z", 2 * slots), self._buf("bad", slots)
        oa, oz, ob = self._buf("atoms", 2 * slots), self._buf("zeros", 2 * slots), self._buf("bad_out", slots)
        grid = (n // BLOCK_COLS, rows // BLOCK_ROWS)
        self.dev.launch("debit_fwd", grid, 128, p=(pa, pb, s0), i=(rows, n, k))
        self.dev.launch("debit_bounds", -(-slots // 256), 256, p=(0, 0, s0, lo, hi, z, bd), i=(rows, n, k))
        self.dev.launch("debit_flags", grid, 128, p=(pa, pb, 0, lo, hi, z, bd, oa, oz, ob), i=(rows, n, k))
        return oa, oz, ob

    def words(self, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b, nb = self._checked(a, b)
        (ra, k), n = a.shape, b.shape[0]
        atoms, zeros, bad = np.zeros((ra, nb), np.uint16), np.zeros((ra, nb), np.uint16), np.zeros((ra, nb), bool)
        if not ra or not nb:
            return atoms, zeros, bad
        step = self.chunk_rows(n, k)
        pb = self._put("b", b)
        for r in range(0, ra, step):
            part = a[r:r + step]
            rows = -(-len(part) // BLOCK_ROWS) * BLOCK_ROWS
            if rows != len(part):
                part = np.concatenate([part, np.zeros((rows - len(part), k), np.uint8)])
            oa, oz, ob = self._chunk(self._put("a", part), pb, rows, n, k)
            self.dev.sync()
            got = len(a[r:r + step])
            atoms[r:r + got] = self._get(oa, (rows, n), np.uint16)[:got, :nb]
            zeros[r:r + got] = self._get(oz, (rows, n), np.uint16)[:got, :nb]
            bad[r:r + got] = self._get(ob, (rows, n), np.uint8)[:got, :nb] != 0
        return atoms, zeros, bad

    def resident(self, a: np.ndarray, b: np.ndarray):
        """words(a, b)'s launches alone, as a body for `run.Dev.timed`: A′ (padded to whole blocks of rows) and B̃ put on
        the device once, here, and each chunk's three launches at its rows' offset in A′, with nothing copied either way
        and no synchronization, as a served call's count would run on operands the call already holds."""
        b, nb = self._checked(a, b)
        (ra, k), n = a.shape, b.shape[0]
        if not ra or not nb:
            raise ValueError("resident: A′ and B̃ each need a row")
        rows = -(-ra // BLOCK_ROWS) * BLOCK_ROWS
        if rows != ra:
            a = np.concatenate([a, np.zeros((rows - ra, k), np.uint8)])
        pa, pb, step = self._put("a_all", a), self._put("b", b), self.chunk_rows(n, k)

        def body() -> None:
            for r in range(0, rows, step):
                self._chunk(pa + r * k, pb, min(step, rows - r), n, k)
        return body


def codes(rng: np.random.Generator, rows: int, k: int) -> np.ndarray:
    """E4M3 code rows whose words flag atoms and promote into +0: small codes with some 32-code slices large (the small
    atoms then round away), some promotion windows zero, both signs, and no NaN code."""
    c = rng.integers(1, 0x10, (rows, k))
    for t in range(0, k, 32):
        if rng.random() < 0.3:
            c[:, t:t + 32] = rng.integers(0x50, 0x7F, (rows, 32))
    for t in range(0, k, WINDOW):
        if rng.random() < 0.15:
            c[:, t:t + WINDOW] = 0
    return (c | rng.integers(0, 2, (rows, k)) << 7).astype(np.uint8)


def check(cubin: bytes, shapes: list[tuple[int, int, int]], timed: list[tuple[int, int, int]], seed: int,
          resident: list[tuple[int, int, int]] = (), reps: int = 5) -> dict:
    """The device's counts against `fast_tile.word_debits` on `codes` at each (rows, n, k) of `shapes`, its seconds at
    each of `timed` (a words() call on the host's clock, copies included), and at each of `resident` the milliseconds of
    `reps` bodies of `Debit.resident` between events (the launches alone)."""
    import time

    sys.path.insert(0, os.path.join(HERE, "..", "pearl_c_vllm"))
    import fast_tile as FT
    from verity.protocols.pouw.prover.schemes import pearl_c_debit as D

    d, rng, out = Debit(cubin), np.random.default_rng(seed), {"shapes": [], "timed": [], "resident": []}
    for rows, n, k in shapes:
        a, b = codes(rng, rows, k), codes(rng, n, k)
        atoms, zeros, bad = d.words(a, b)
        want = FT.word_debits(a.tolist(), b.tolist(), D.H100)
        same = (atoms == want[0]) & (zeros == want[1])
        out["shapes"].append({"rows": rows, "n": n, "k": k, "words": int(same.size), "differ": int((~same).sum()),
                              "bad": int(bad.sum()), "atoms": int(want[0].sum()), "zeros": int(want[1].sum())})
        print(json.dumps(out["shapes"][-1]), flush=True)
    for rows, n, k in timed:
        a, b = codes(rng, rows, k), codes(rng, n, k)
        t = time.perf_counter()
        atoms, _, bad = d.words(a, b)
        out["timed"].append({"rows": rows, "n": n, "k": k, "s": time.perf_counter() - t, "bad": int(bad.sum()),
                             "atoms": int(atoms.astype(np.int64).sum())})
        print(json.dumps(out["timed"][-1]), flush=True)
    for rows, n, k in resident:
        ms = d.dev.timed(d.resident(codes(rng, rows, k), codes(rng, n, k)), reps)
        out["resident"].append({"rows": rows, "n": n, "k": k, "ms": float(np.median(ms)), "all_ms": ms})
        print(json.dumps(out["resident"][-1]), flush=True)
    out["ok"] = bool(out["shapes"]) and all(s["differ"] == 0 and s["bad"] == 0 for s in out["shapes"])
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="the device's word counts against fast_tile.word_debits, and its seconds")
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--shape", nargs="*", default=["64:32:128", "192:96:2048", "128:64:8192"], help="ROWS:N:K, checked")
    ap.add_argument("--timed", nargs="*", default=["8192:16384:2048", "8192:2048:8192"], help="ROWS:N:K, timed only")
    ap.add_argument("--resident", nargs="*", default=[], help="ROWS:N:K, the launches alone on operands put once, timed")
    ap.add_argument("--reps", type=int, default=5, help="--resident: bodies timed at each shape")
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    def parse(xs):
        return [tuple(int(v) for v in x.split(":")) for x in xs]

    out = check(open(a.cubin, "rb").read(), parse(a.shape), parse(a.timed), a.seed, parse(a.resident), a.reps)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(json.dumps(out, indent=1) + "\n")
    print(f"debit check {'PASS' if out['ok'] else 'FAIL'}", flush=True)
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
