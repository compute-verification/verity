"""The per-call binding of v31.cu's GPU prover for Dither's no-fold call (CMP-172, circuit/dither.py on #4974): ctypes over
libv31.so (v31.cu built with -DV31_LIB; `build_lib` below, run.sh's MODE=fill), one `Prover` a shape. v31.py's `Fill` is the
reference it is checked against and the verifier.

    p = Prover(m, n, k, lp=256, L=4096, row=1024, lib=PATH, device=0)    # its CGP1 params (dither's domain, a fresh Ks)
    p.set_weights(B)                     # n x k int8: a numpy array, or a device pointer (int) with on_device=True
    seeds = call_seeds(salt, coin)       # the call's seeds from the challenger's live coin (dither.call_seeds)
    a_ptr = p.load_a(A)                  # optional: A (m x k int8, host or on_device) into the Prover's buffer
    c = p.prove(seeds, a_ptr)            # the call's index, at once; a_ptr: A (m x k int8) on the device, None: the pad
    root = p.root(c)                     # 64 bytes, once the call's commit is done
    ans = p.answer(c, i, j, s)           # proof block (i, j, s)'s opened rows: {"rows": {pos: (row, salt, path)}, "B": B[j]}
    rows = p.open(c, positions)          # [(row, salt, path)] for any leaves

Calls run in issue order on the device, each commit beside the next call's compute; `prove` returns before the call is
done, and a call's values can be opened until the call two after it is issued (`prove` waits for that call's commit
first). Several Provers share a process (a domain slot each, up to 8; one ROW, one device). W' is formed again only when
the weight-side seeds (dB, U, P) or the weights change, so v3.2's period coins form it once a period. Leaf p of call c is
salted prf_salt(call_key(Ks, c), p), Ks the Prover's own.

    python3 v31lib.py gate --lib LIB [--q 64]   the GPU's words, pad and every tree level against the reference and the
                                                verifier's verdicts on the GPU's own openings, honest and cheating calls

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Self

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v31

SEED_BYTES = 64 * len(v31.FILL_STREAMS)


def build_lib(out: str, gencode: str = "arch=compute_120a,code=sm_120a", nvcc: str = "nvcc") -> str:
    """libv31.so from v31.cu (no CUTLASS; 8 domain slots)."""
    cmd = [nvcc, "-O3", "-std=c++17", "-gencode", gencode, "-DSHA_ADD=0", "-DSHA_ROT=0", "-DSHA_ROLL=1", "-DV31_LIB",
           "-DCOMMIT_GPU_DOMS=8", "-Xcompiler", "-fPIC", "-shared", "-o", out, str(HERE / "v31.cu"), "-lcuda", "-lcublasLt"]
    subprocess.run(cmd, check=True)
    return out


class Lib:
    """libv31.so's functions with their ctypes signatures."""

    def __init__(self, path: str):
        d = ctypes.CDLL(path)
        vp, u8p, u64p, i, u64 = ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_uint64
        sig = {
            "v31_error": ([], ctypes.c_char_p),
            "v31_create": ([i, i, i, i, i, i, ctypes.c_char_p, i], vp),
            "v31_info": ([vp, u64p], i),
            "v31_set_b": ([vp, vp, i], i),
            "v31_load_a": ([vp, vp, i, ctypes.POINTER(vp)], i),
            "v31_prove": ([vp, u8p, u8p, vp, ctypes.c_char_p, u64p], i),
            "v31_root": ([vp, u64, vp], i),
            "v31_open": ([vp, u64, u64p, i, vp, vp], i),
            "v31_words": ([vp, u64, vp], i),
            "v31_levels": ([vp, u64, vp], i),
            "v31_pad": ([vp, vp], i),
            "v31_sync": ([vp], i),
            "v31_destroy": ([vp], None),
        }
        for name, (args, res) in sig.items():
            fn = getattr(d, name)
            fn.argtypes, fn.restype = args, res
            setattr(self, name[4:], fn)

    def check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"libv31: {self.error().decode()}")


class Prover:
    """One shape's GPU prover: m x n output cells at contraction k, noise period lp, segments of L, leaf rows of `row`
    bytes. `params_dir` keeps its CGP1 file (default a temporary one)."""

    def __init__(self, m: int, n: int, k: int, lp: int = 256, L: int = 4096, row: int = 1024, *, lib: str | Lib,
                 device: int = 0, ks: bytes | None = None, params_dir: str | None = None):
        self.f = v31.Fill(m, n, k, lp, L, row)
        self.lib = lib if isinstance(lib, Lib) else Lib(lib)
        self.ks = os.urandom(64) if ks is None else ks
        d = Path(params_dir or os.environ.get("TMPDIR", "/tmp"))
        d.mkdir(parents=True, exist_ok=True)
        self.params_path = d / f"v31lib-{os.getpid()}-{id(self):x}-{v31.fill_params_name(m, n, k, lp, L, row)}"
        self.params_path.write_bytes(v31.fill_params(self.f, self.ks))
        os.chmod(self.params_path, 0o600)
        self.h = self.lib.create(m, n, k, lp, L, row, str(self.params_path).encode(), device)
        if not self.h:
            raise RuntimeError(f"libv31: {self.lib.error().decode()}")
        self.B: np.ndarray | None = None
        self.calls: dict[int, dict] = {}

    def close(self) -> None:
        if self.h:
            self.lib.destroy(self.h)
            self.h = None
            self.params_path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def info(self) -> dict:
        out = (ctypes.c_uint64 * 12)()
        self.lib.check(self.lib.info(self.h, out))
        return dict(zip(("m", "n", "k", "lp", "L", "row", "S", "tiles", "lpt", "leaves", "depth", "calls"), map(int, out)))

    def set_weights(self, B, on_device: bool = False) -> None:
        """The weights: an n x k int8 array (kept for the answers' B rows), or a device pointer with on_device."""
        if on_device:
            self.lib.check(self.lib.set_b(self.h, ctypes.c_void_p(int(B)), 1))
            self.B = None
            return
        B = np.ascontiguousarray(B, dtype=np.int8)
        if B.shape != (self.f.n, self.f.k):
            raise ValueError(f"B is n x k = {self.f.n} x {self.f.k}, not {B.shape}")
        self.lib.check(self.lib.set_b(self.h, B.ctypes.data_as(ctypes.c_void_p), 0))
        self.B = B

    def load_a(self, A, on_device: bool = False) -> int:
        """A (m x k int8: a numpy array, or a device pointer with on_device) into the Prover's own operand buffer, after
        every issued call's form; returns the buffer's device pointer, for `prove`. The pad overwrites the same buffer."""
        if on_device:
            src = ctypes.c_void_p(int(A))
        else:
            A = np.ascontiguousarray(A, dtype=np.int8)
            if A.shape != (self.f.m, self.f.k):
                raise ValueError(f"A is m x k = {self.f.m} x {self.f.k}, not {A.shape}")
            src = A.ctypes.data_as(ctypes.c_void_p)
        dev = ctypes.c_void_p()
        self.lib.check(self.lib.load_a(self.h, src, int(on_device), ctypes.byref(dev)))
        return int(dev.value)

    def prove(self, seeds: dict[str, bytes], a_ptr: int | None = None, cheat: str = "none") -> int:
        """Issue one call on `seeds` (dither's six streams), A at device pointer a_ptr (m x k int8, written before this
        call by the caller's stream, which the caller has synchronized) or the pad. Returns the call's index."""
        sb = v31.fill_seeds_bytes(seeds)
        if len(sb) != SEED_BYTES:
            raise ValueError("six 64-byte seeds")
        c = self.info()["calls"]
        key = v31.call_key(self.ks, c)
        out = ctypes.c_uint64()
        self.lib.check(self.lib.prove(self.h, sb, key, None if a_ptr is None else ctypes.c_void_p(a_ptr), cheat.encode(),
                                      ctypes.byref(out)))
        assert out.value == c
        self.calls[c] = {"seeds": dict(seeds), "key": key, "pad": a_ptr is None}
        self.calls.pop(c - 3, None)
        return c

    def root(self, c: int) -> bytes:
        buf = ctypes.create_string_buffer(64)
        self.lib.check(self.lib.root(self.h, c, buf))
        return buf.raw

    def open(self, c: int, positions) -> list[tuple[bytes, bytes, tuple[bytes, ...]]]:
        """Leaves `positions` of call c: each one's row, salt and path (the sibling at each level, the leaf's first)."""
        pos = [int(p) for p in positions]
        if not pos:
            return []
        depth, row = self.info()["depth"], self.f.row
        parr = (ctypes.c_uint64 * len(pos))(*pos)
        rows, paths = ctypes.create_string_buffer(len(pos) * row), ctypes.create_string_buffer(len(pos) * depth * 64)
        self.lib.check(self.lib.open(self.h, c, parr, len(pos), rows, paths))
        key = self.calls[c]["key"]
        r, pa = rows.raw, paths.raw
        return [(r[x * row:(x + 1) * row], v31.cref.prf_salt(key, p),
                 tuple(pa[64 * (x * depth + d):64 * (x * depth + d + 1)] for d in range(depth))) for x, p in enumerate(pos)]

    def answer(self, c: int, i: int, j: int, s: int) -> dict:
        """Proof block (i, j, s)'s opened rows (T_s's and T_{s-1}'s leaves) and B's row j (when the weights are kept)."""
        pos = v31.fill_cells_positions(self.f, i, j, s)
        return {"rows": dict(zip(pos, self.open(c, pos))), "B": None if self.B is None else self.B[j]}

    def words(self, c: int) -> bytes:
        buf = ctypes.create_string_buffer(self.f.tiles * self.f.S * 32768)
        self.lib.check(self.lib.words(self.h, c, buf))
        return buf.raw

    def levels(self, c: int) -> bytes:
        depth = self.info()["depth"]
        buf = ctypes.create_string_buffer((2 * (1 << depth) - 1) * 64)
        self.lib.check(self.lib.levels(self.h, c, buf))
        return buf.raw

    def pad(self) -> bytes:
        buf = ctypes.create_string_buffer(self.f.m * self.f.k)
        self.lib.check(self.lib.pad(self.h, buf))
        return buf.raw

    def sync(self) -> None:
        self.lib.check(self.lib.sync(self.h))


call_seeds = v31.call_seeds


# ------------------------------------------------------------------------------------------------------------- gate
# NAME M N K LP L ROW A CHEAT: A pad (the filler) or llm/uniform (a served operand, through the context's device copy)
GATES = """
f-pad-k 256 256 4096 256 4096 1024 pad none
f-pad-768 256 384 3072 256 768 1024 pad none
f-llm-4096 256 256 8192 256 4096 1024 llm none
f-uniform-1280 256 512 2560 256 1280 1024 uniform none
f-lp512 256 256 4096 512 1024 2048 llm none
f-row8k 128 256 8192 256 8192 8192 pad none
f-big 512 1024 8192 256 4096 1024 llm none
c-skip 256 256 4096 256 1024 1024 pad skip:5
c-forge 256 256 4096 256 1024 1024 llm forge:2:0.25
"""


def gate_case(lib: Lib, name: str, m: int, n: int, k: int, lp: int, L: int, row: int, a_kind: str, cheat: str, q: int,
              rng: np.random.Generator) -> dict:
    """One gate: a call on fresh coins (a 64-byte coin drawn after the operands are fixed, the stand-in for the challenger's
    opening), the GPU's words, pad and tree against the reference, then q proof blocks from a check coin drawn after the
    root, answered from the GPU's own openings."""
    f = v31.Fill(m, n, k, lp, L, row)
    A, B = v31.operands(f, rng, "llm" if a_kind == "pad" else a_kind)
    rep: dict = {"name": name, "shape": f.label(), "S": f.S, "A": a_kind, "cheat": cheat}
    with Prover(m, n, k, lp, L, row, lib=lib) as p:
        p.set_weights(B)
        salt, coin = os.urandom(64), os.urandom(64)
        seeds = call_seeds(salt, coin)
        a_ptr = None
        if a_kind != "pad":
            a_ptr = p.load_a(A)
        t0 = time.perf_counter()
        c = p.prove(seeds, a_ptr, cheat)
        root = p.root(c)
        rep["prove_root_ms"] = 1e3 * (time.perf_counter() - t0)
        if a_kind == "pad":
            A = v31.pad_rows(seeds, range(m), k)
            rep["pad_equal"] = p.pad() == A.tobytes()
        co = v31.Coins.expand(f, seeds)
        skip = int(cheat.split(":")[1]) if cheat.startswith("skip:") else None
        T = v31.fill_words(f, A, B, co, skip)
        g_words, g_lv = p.words(c), p.levels(c)
        words = v31.words_buffer(f, T)
        if cheat.startswith("forge:"):   # forge_kernel's T_S0 of tiles 0, stride, ..: the GPU's own words there
            parts = cheat.split(":")
            s0, frac = int(parts[1]), float(parts[2]) if len(parts) > 2 else 1.0
            w, changed = bytearray(words), 0
            for t in range(0, f.tiles, max(1, int(1 / frac + 0.5))):
                a = (t * f.S + s0 - 1) * 32768
                changed += g_words[a:a + 32768] != w[a:a + 32768]
                w[a:a + 32768] = g_words[a:a + 32768]
            words, rep["forged_tiles_changed"] = bytes(w), changed
        ref = v31.fill_commit(f, words, v31.call_key(p.ks, c))
        rep["words_equal"] = g_words == ref.words
        if not rep["words_equal"]:
            gw, rw = np.frombuffer(g_words, "<u2"), np.frombuffer(ref.words, "<u2")
            rep["words_bad"] = int((gw != rw).sum())
        rep["levels_equal"] = g_lv == v31.levels_bytes(ref.levels)
        rep["root_gpu"], rep["root_ref"] = root.hex()[:32], ref.root.hex()[:32]
        check = os.urandom(64)
        picks = v31.fill_picks(f, check, q)
        Ar = None if a_kind == "pad" else A

        def answer(i: int, j: int, s: int) -> dict:
            ans = p.answer(c, i, j, s)
            ans["A"] = None if Ar is None else Ar[i]
            return ans

        ok, fails, line = v31.fill_verdict(f, seeds, root, picks, answer, f"{name} call {c}")
        rep.update(accepted=ok, verdict=line, failures=fails[:8])
    rep["exact"] = (rep["words_equal"] and rep["levels_equal"] and rep.get("pad_equal", True)
                    and rep.get("forged_tiles_changed", 1) > 0)
    rep["good"] = (rep["exact"] and ok) if cheat == "none" else (not ok and rep["exact"])
    return rep


def gate_pipeline(lib: Lib, q: int, rng: np.random.Generator) -> dict:
    """Two Provers of different shapes in one process, their calls interleaved back to back (four each, each commit beside
    the next call's compute): every call's root against the reference, a stale call's refusal, and a verdict on the last."""
    shapes = [(256, 384, 4096, 256, 4096, 1024), (384, 256, 2048, 256, 1024, 1024)]
    out: dict = {"name": "pipeline", "calls": []}
    ps = [Prover(*s, lib=lib) for s in shapes]
    try:
        Bs = []
        for p in ps:
            _, B = v31.operands(p.f, rng, "llm")
            p.set_weights(B)
            Bs.append(B)
        issued = []
        for r in range(4):
            for x, p in enumerate(ps):
                seeds = call_seeds(os.urandom(64), os.urandom(64))
                issued.append((x, p.prove(seeds), seeds))
            if r >= 1:   # the previous round's roots, read while this round's calls run
                for x, c, seeds in issued[-4:-2]:
                    out["calls"].append(_pipeline_check(ps[x], Bs[x], c, seeds))
        for x, c, seeds in issued[-2:]:
            out["calls"].append(_pipeline_check(ps[x], Bs[x], c, seeds))
        try:
            ps[0].root(0)
            out["stale_refused"] = False
        except RuntimeError as e:
            out["stale_refused"] = "isn't one of the last two calls" in str(e)
        x, c, seeds = issued[-1]
        f = ps[x].f
        picks = v31.fill_picks(f, os.urandom(64), q)
        ok, _, line = v31.fill_verdict(f, seeds, ps[x].root(c), picks,
                                           lambda i, j, s: {**ps[x].answer(c, i, j, s), "A": None}, f"pipeline call {c}")
        out.update(accepted=ok, verdict=line)
    finally:
        for p in ps:
            p.close()
    out["good"] = out["accepted"] and out["stale_refused"] and all(r["root_equal"] for r in out["calls"])
    return out


def _pipeline_check(p: Prover, B: np.ndarray, c: int, seeds: dict[str, bytes]) -> dict:
    f = p.f
    T = v31.fill_words(f, v31.pad_rows(seeds, range(f.m), f.k), B, v31.Coins.expand(f, seeds))
    ref = v31.fill_commit(f, v31.words_buffer(f, T), v31.call_key(p.ks, c))
    return {"shape": f"{f.m}x{f.n}x{f.k} L {f.L}", "call": c, "root_equal": p.root(c) == ref.root}


def cmd_gate(a: argparse.Namespace) -> int:
    lib = Lib(a.lib)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    good = True
    for line in GATES.strip().splitlines():
        name, m, n, k, lp, L, row, ak, cheat = line.split()
        rep = gate_case(lib, name, int(m), int(n), int(k), int(lp), int(L), int(row), ak, cheat, a.q, rng)
        (out / f"fill-{name}.json").write_text(json.dumps(rep, indent=1))
        print(f"{name}: exact {rep['exact']} ({'pad ' + str(rep['pad_equal']) + ', ' if 'pad_equal' in rep else ''}words "
              f"{rep['words_equal']}, levels {rep['levels_equal']}); {rep['verdict']}", flush=True)
        good &= rep["good"]
    rep = gate_pipeline(lib, a.q, rng)
    (out / "fill-pipeline.json").write_text(json.dumps(rep, indent=1))
    print(f"pipeline: roots {sum(r['root_equal'] for r in rep['calls'])}/{len(rep['calls'])} equal, stale call refused "
          f"{rep['stale_refused']}; {rep['verdict']}", flush=True)
    good &= rep["good"]
    return 0 if good else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gate")
    p.add_argument("--lib", required=True)
    p.add_argument("--out", default=".")
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv[1:])
    return {"gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
