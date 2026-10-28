"""The per-call binding of v31.cu's GPU prover for Dither's no-fold call (verity/core/protocols/pouw/circuit/dither.py):
ctypes over libv31.so (v31.cu built with -DV31_LIB; `build_lib` below, run.sh's MODE=fill), one `Prover` a shape. v31.py's
`Fill` is the reference it is checked against and the verifier.

    p = Prover(m, n, k, lp=128, L=1024, row=16384, served=True, lib=PATH, device=0)   # its CGP1 params (dither's domain,
                                                                                      # a fresh Ks unless ks= is given)
    p.set_weights(B)                     # n x k int8, a numpy array (kept for the answers' B rows)
    p.point_weights(ptr)                 # or the weights at a device pointer, no copy (None: set_weights' again)
    seeds = served_seeds(coin, receipt, call)   # dither.seeds: the raw coin, the operand-root receipt, the call's index
    a_ptr = p.load_a(A)                  # optional: A (m x k int8, a numpy array) into the Prover's buffer
    c = p.prove(seeds, a_ptr)            # the call's index, at once; a_ptr: A (m x k int8) on the device (None: the
                                         # filler's pad, served=False only)
    root = p.root(c)                     # 64 bytes, once the call's commit is done
    ans = p.answer(c, i, j, s)           # proof block (i, j, s)'s opened rows: {"rows": {pos: (row, salt, path)}, "B": B[j]}
    rows = p.open(c, positions)          # [(row, salt, path)] for any leaves
    p.levels_async(c, dst)               # every tree level into page-locked dst behind the commit, whole after p.sync()

A served call (served=True, dither.Shape(o=True)) commits per tile T_1 .. T_{S-1} (u16) and then O = (A + d_A)(B +
d_B)^T mod 2^32 (u32; T_S is implied by O); a filler's commits T_1 .. T_S of its pad. Calls run in issue order on the
device, each commit beside the next call's compute; `prove` returns before the call is done, and a call's values can be
opened until the call two after it is issued (`prove` waits for that call's commit first). Several Provers share a process
(a domain slot each, up to 8; one ROW, one device). The weight side (b', W') is formed again only when the weight-side
seeds (dB, U, P) or the weights change: per-call weight coins form it every call, v3.2's period coins once a period. The
c-th call of a Prover salts its leaf pos prf_salt(Ks, c leaves + pos) (dither.salts(Ks, off), off = c leaves), Ks the
Prover's own. A served call whose A + d_A (or B + d_B) has a row holding both -128 and 128 has no root: O's int8
decomposition (a' = A + d_A - sigma) can't carry it, and `root` raises. v31.cu's knobs come from the environment when a
Prover is made: V31_LTW=1 has a filler call at S = 1 (L >= K) take cuBLASLt's int32 X W'^T in place of the protocol GEMM,
its leaves the same bytes (the low halves); V31_PF, V31_CCTA, V31_SCHED and V31_CPRIO tune the commit beside the next
call's compute. One thread drives every Prover, the one that made the first live one (any other is refused).
`Lib.lt_create`, `lt_run`, `lt_algo` and `lt_destroy` are the autotuned cuBLASLt GEMM a V31_LTW call makes, on the
caller's device pointers: a plain baseline.

    python3 v31lib.py gate --lib LIB [--q 64]   the GPU's values, pad and every tree level against the reference and the
                                                verifier's verdicts on the GPU's own openings, honest and cheating calls,
                                                filler and served; point_weights, levels_async and the one-thread rule

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Self

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v31

HERE = Path(__file__).resolve().parent

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
            "v31_create": ([i, i, i, i, i, i, i, ctypes.c_char_p, i], vp),
            "v31_info": ([vp, u64p], i),
            "v31_set_b": ([vp, vp], i),
            "v31_point_b": ([vp, vp], i),
            "v31_load_a": ([vp, vp, ctypes.POINTER(vp)], i),
            "v31_prove": ([vp, u8p, u64, vp, ctypes.c_char_p, u64p], i),
            "v31_root": ([vp, u64, vp], i),
            "v31_open": ([vp, u64, u64p, i, vp, vp], i),
            "v31_values": ([vp, u64, vp], i),
            "v31_levels_async": ([vp, u64, vp], i),
            "v31_pad": ([vp, vp], i),
            "v31_sync": ([vp], i),
            "v31_stream": ([vp, i], vp),
            "v31_destroy": ([vp], None),
            "v31_lt_create": ([i, i, i, vp, vp, vp, vp], vp),
            "v31_lt_run": ([vp] * 5, None),
            "v31_lt_algo": ([vp, vp], None),
            "v31_lt_destroy": ([vp], None),
        }
        for name, (args, res) in sig.items():
            fn = getattr(d, name)
            fn.argtypes, fn.restype = args, res
            setattr(self, name[4:], fn)

    def check(self, rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"libv31: {self.error().decode()}")


class Prover:
    """One shape's GPU prover: m x n output cells at contraction k, noise tiles of lp, segments of L, leaf rows of `row`
    bytes, served calls or a filler's. `params_dir` keeps its CGP1 file (default a temporary one)."""

    def __init__(self, m: int, n: int, k: int, lp: int = 256, L: int = 4096, row: int = 1024, *, served: bool = False,
                 lib: str | Lib, device: int = 0, ks: bytes | None = None, params_dir: str | None = None):
        self.f = v31.Fill(m, n, k, lp, L, row, served)
        self.lib = lib if isinstance(lib, Lib) else Lib(lib)
        self.ks = os.urandom(64) if ks is None else ks
        d = Path(params_dir or tempfile.gettempdir())
        d.mkdir(parents=True, exist_ok=True)
        self.params_path = d / f"v31lib-{os.getpid()}-{id(self):x}-{v31.fill_params_name(m, n, k, lp, L, row, served)}"
        self.params_path.write_bytes(v31.fill_params(self.f, self.ks))
        os.chmod(self.params_path, 0o600)
        self.h = self.lib.create(m, n, k, lp, L, row, int(served), str(self.params_path).encode(), device)
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
        out = (ctypes.c_uint64 * 14)()
        self.lib.check(self.lib.info(self.h, out))
        return dict(zip(("m", "n", "k", "lp", "L", "row", "S", "tiles", "lpt", "leaves", "depth", "calls", "W", "o"),
                        map(int, out)))

    def set_weights(self, B) -> None:
        """The weights: an n x k int8 array, kept for the answers' B rows (weights on the device: `point_weights`)."""
        B = np.ascontiguousarray(B, dtype=np.int8)
        if B.shape != (self.f.n, self.f.k):
            raise ValueError(f"B is n x k = {self.f.n} x {self.f.k}, not {B.shape}")
        self.lib.check(self.lib.set_b(self.h, B.ctypes.data_as(ctypes.c_void_p)))
        self.B = B

    def point_weights(self, ptr: int | None) -> None:
        """The weights where they are, n x k int8 at device pointer ptr, no copy and no wait (unchanged until the calls
        issued after have formed: `sync`), or None for the Prover's own buffer (`set_weights`'s); the answers' B is None."""
        self.lib.check(self.lib.point_b(self.h, None if ptr is None else ctypes.c_void_p(int(ptr))))
        self.B = None

    def load_a(self, A) -> int:
        """A (an m x k int8 array) into the Prover's own operand buffer, after every issued call's form; returns the
        buffer's device pointer, for `prove`. The pad overwrites the same buffer."""
        A = np.ascontiguousarray(A, dtype=np.int8)
        if A.shape != (self.f.m, self.f.k):
            raise ValueError(f"A is m x k = {self.f.m} x {self.f.k}, not {A.shape}")
        dev = ctypes.c_void_p()
        self.lib.check(self.lib.load_a(self.h, A.ctypes.data_as(ctypes.c_void_p), ctypes.byref(dev)))
        return int(dev.value)

    def prove(self, seeds: dict[str, bytes], a_ptr: int | None = None, cheat: str = "none") -> int:
        """Issue one call on `seeds` (dither's streams: a served call's five, `served_seeds`; a filler's six, with the
        pad), A at device pointer a_ptr (m x k int8, written before this call by the caller's stream, which the caller has
        synchronized) or the filler's pad. The c-th call salts from c leaves. Returns the call's index."""
        if self.f.o and a_ptr is None:
            raise ValueError("a served call takes the caller's A")
        need = set(v31.STREAMS) | (set() if self.f.o else {"pad"})
        if not need <= set(seeds):
            raise ValueError(f"seeds for {sorted(need)}")
        sb = v31.fill_seeds_bytes(seeds)
        if len(sb) != SEED_BYTES:
            raise ValueError("64-byte seeds")
        c = self.info()["calls"]
        off = c * self.f.leaves
        out = ctypes.c_uint64()
        self.lib.check(self.lib.prove(self.h, sb, off, None if a_ptr is None else ctypes.c_void_p(a_ptr), cheat.encode(),
                                      ctypes.byref(out)))
        assert out.value == c
        self.calls[c] = {"seeds": dict(seeds), "off": off, "pad": a_ptr is None}
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
        off = self.calls[c]["off"]
        r, pa = rows.raw, paths.raw
        return [(r[x * row:(x + 1) * row], v31.cref.prf_salt(self.ks, off + p),
                 tuple(pa[64 * (x * depth + d):64 * (x * depth + d + 1)] for d in range(depth))) for x, p in enumerate(pos)]

    def answer(self, c: int, i: int, j: int, s: int) -> dict:
        """Proof block (i, j, s)'s opened rows (`Fill.opened`: T_s's, T_{s-1}'s, and a served call's O at s = S) and B's
        row j (when the weights are kept)."""
        pos = self.f.opened(i, j, s)
        return {"rows": dict(zip(pos, self.open(c, pos))), "B": None if self.B is None else self.B[j]}

    def values(self, c: int) -> bytes:
        """Call c's committed values (dither.values_buffer: leaves x row bytes)."""
        buf = ctypes.create_string_buffer(self.f.leaves * self.f.row)
        self.lib.check(self.lib.values(self.h, c, buf))
        return buf.raw

    def levels_async(self, c: int, dst: int) -> None:
        """Call c's every tree level (leaves first, each node 8 native u64 words) copied behind its commit to dst, a host
        pointer to (2 width - 1) x 64 bytes (page-locked, or the copy waits for the commit); whole after `sync`."""
        self.lib.check(self.lib.levels_async(self.h, c, ctypes.c_void_p(int(dst))))

    def levels(self, c: int) -> bytes:
        buf = np.zeros(8 * (2 << self.info()["depth"]) - 8, np.uint64)
        self.levels_async(c, buf.ctypes.data)
        self.sync()
        return buf.byteswap().tobytes()

    def pad(self) -> bytes:
        buf = ctypes.create_string_buffer(self.f.m * self.f.k)
        self.lib.check(self.lib.pad(self.h, buf))
        return buf.raw

    def sync(self) -> None:
        self.lib.check(self.lib.sync(self.h))

    def stream(self, which: int) -> int:
        """The shared compute (0) or commit (1) stream's handle, for timing events: a call's compute is on the first,
        and its commit and `levels_async`'s copies follow on the second."""
        s = self.lib.stream(self.h, which)
        if not s:
            raise RuntimeError("v31_stream: called from a thread other than the binding's")
        return int(s)


call_seeds = v31.call_seeds
served_seeds = v31.served_seeds


# ------------------------------------------------------------------------------------------------------------- gate
# NAME KIND M N K LP L ROW A CHEAT. KIND fill (A pad: the filler's pad; llm/uniform: a served operand through the filler's
# path) or served (A llm/uniform, through the context's device copy). CHEAT none, a cheat the verifier must refuse, or
# rootless: an honest served call the prover must refuse at its root (a row of A + d_A spans -128 .. 128)
GATES = """
f-pad-k fill 256 256 4096 512 4096 1024 pad none
f-pad-768 fill 256 384 3072 256 768 1024 pad none
f-llm-4096 fill 256 256 8192 512 4096 1024 llm none
f-uniform-1280 fill 256 512 2560 256 1280 1024 uniform none
f-lp512 fill 256 256 4096 512 1024 2048 llm none
f-lp1024 fill 256 256 8192 1024 4096 1024 llm none
c-skip-lp1024 fill 256 256 4096 1024 4096 1024 pad skip:1
f-lp2048 fill 256 256 6144 2048 8192 1024 llm none
f-row8k fill 128 256 8192 256 2048 8192 pad none
f-big fill 512 1024 8192 512 4096 1024 llm none
c-skip fill 256 256 4096 256 1024 1024 pad skip:5
c-forge fill 256 256 4096 256 1024 1024 llm forge:2:0.25
f-k2048 fill 256 384 2048 256 2048 2048 llm none
c-skip-s1 fill 256 256 2048 256 2048 1024 pad skip:3
c-forge-s1 fill 256 256 2048 256 2048 1024 llm forge:1:0.25
s-k served 256 256 4096 512 4096 16384 llm none
s-s4 served 256 256 4096 128 1024 16384 llm none
s-partial served 256 384 3072 128 896 8192 llm none
s-lp256 served 256 256 4096 256 2048 16384 llm none
s-lp1024 served 256 256 8192 1024 4096 16384 llm none
s-row1k served 256 512 2048 128 1024 1024 llm none
s-big served 512 1024 8192 512 4096 16384 llm none
s-uniform served 256 256 2048 256 2048 16384 uniform rootless
cs-forgeO served 256 256 4096 512 4096 16384 llm forgeO:0.25
cs-forgeO-s4 served 256 256 4096 128 1024 16384 llm forge:0:0.25
cs-forge-s4 served 256 256 4096 128 1024 16384 llm forge:2:0.25
cs-skip-s4 served 256 256 4096 128 1024 16384 llm skip:5
cs-nodither served 256 256 4096 512 4096 16384 llm nodither
cs-nodither-s4 served 256 256 4096 128 1024 16384 llm nodither
"""


def _forged(f: v31.Fill, cheat: str) -> tuple[int, int, int] | None:
    """A forge cheat's (offset in a tile's values, length, tile stride): forge_kernel's T_S0 (S0 >= 1) or O (forgeO,
    forge:0) of tiles 0, stride, .."""
    if not cheat.startswith("forge"):
        return None
    parts = cheat.split(":")
    s0 = 0 if parts[0] == "forgeO" else int(parts[1])
    frac = float(parts[-1]) if len(parts) > (1 if parts[0] == "forgeO" else 2) else 1.0
    stride = max(1, int(1 / frac + 0.5))
    return ((s0 - 1) * 32768, 32768, stride) if s0 else (f.W * 32768, 65536, stride)


def gate_case(lib: Lib, name: str, served: bool, m: int, n: int, k: int, lp: int, L: int, row: int, a_kind: str,
              cheat: str, q: int, rng: np.random.Generator) -> dict:
    """One gate: a call on fresh coins (64-byte coins drawn after the operands are fixed, the stand-in for the challenger's
    opening), the GPU's values, pad and tree against the reference, then q proof blocks from a check coin drawn after the
    root, answered from the GPU's own openings."""
    f = v31.Fill(m, n, k, lp, L, row, served)
    A, B = v31.operands(f, rng, "llm" if a_kind == "pad" else a_kind)
    rep: dict = {"name": name, "o": served, "shape": f.label(), "S": f.S, "W": f.W, "A": a_kind, "cheat": cheat}
    with Prover(m, n, k, lp, L, row, served=served, lib=lib) as p:
        p.set_weights(B)
        seeds = served_seeds(os.urandom(64), os.urandom(64), 0) if served else call_seeds(os.urandom(64), os.urandom(64))
        a_ptr = None
        if a_kind != "pad":
            a_ptr = p.load_a(A)
        t0 = time.perf_counter()
        c = p.prove(seeds, a_ptr, "none" if cheat == "rootless" else cheat)
        try:
            root = p.root(c)
        except RuntimeError as e:
            rep.update(root_refused=str(e), accepted=False, verdict=f"REFUSE {name} call {c}: no root: {e}", exact=True)
            rep["good"] = cheat == "rootless" and "spans -128 .. 128" in str(e)
            return rep
        rep["prove_root_ms"] = 1e3 * (time.perf_counter() - t0)
        if cheat == "rootless":
            rep.update(accepted=True, verdict=f"ACCEPT {name} call {c}: a root, where the prover must refuse", exact=False,
                       good=False)
            return rep
        if a_kind == "pad":
            A = v31.pad_rows(seeds, range(m), k)
            rep["pad_equal"] = p.pad() == A.tobytes()
        co = v31.Coins.expand(f, seeds)
        if cheat == "nodither":
            co = dataclasses.replace(co, dA=np.zeros_like(co.dA), dB=np.zeros_like(co.dB))
        skip = int(cheat.split(":")[1]) if cheat.startswith("skip:") else None
        values = v31.fill_buffer(f, *v31.fill_values(f, A, B, co, skip))
        g_values, g_lv = p.values(c), p.levels(c)
        fg = _forged(f, cheat)
        if fg:   # the forged regions: the GPU's own values there
            at, ln, stride = fg
            w, changed = bytearray(values), 0
            for t in range(0, f.tiles, stride):
                a = t * f.lpt * f.row + at
                changed += g_values[a:a + ln] != w[a:a + ln]
                w[a:a + ln] = g_values[a:a + ln]
            values, rep["forged_tiles_changed"] = bytes(w), changed
        ref = v31.fill_commit(f, values, p.ks, p.calls[c]["off"])
        rep["values_equal"] = g_values == ref.values
        if not rep["values_equal"]:
            rep["values_bad_bytes"] = int((np.frombuffer(g_values, np.uint8) != np.frombuffer(ref.values, np.uint8)).sum())
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
    rep["exact"] = (rep["values_equal"] and rep["levels_equal"] and rep.get("pad_equal", True)
                    and rep.get("forged_tiles_changed", 1) > 0)
    rep["good"] = (rep["exact"] and ok) if cheat == "none" else (not ok and rep["exact"])
    return rep


#: gate_pipeline's two Provers: ((M, N, K, LP, L, ROW), served)
PIPELINE = [((256, 384, 4096, 512, 4096, 1024), False), ((384, 256, 2048, 128, 1024, 1024), True)]


def gate_pipeline(lib: Lib, q: int, rng: np.random.Generator) -> dict:
    """A filler's Prover and a served Prover of different shapes in one process, their calls interleaved back to back (four
    each, each commit beside the next call's compute, the weight coins every call): every call's root against the
    reference (the c-th call salted from c leaves), a stale call's refusal, and a verdict on each Prover's last call."""
    out: dict = {"name": "pipeline", "calls": [], "verdicts": []}
    ps = [Prover(*s, served=o, lib=lib) for s, o in PIPELINE]
    try:
        ops = []
        for p in ps:
            A, B = v31.operands(p.f, rng, "llm")
            p.set_weights(B)
            ops.append((A if p.f.o else None, B, p.load_a(A) if p.f.o else None))
        issued = []
        for r in range(4):
            for x, p in enumerate(ps):
                c = p.info()["calls"]
                seeds = (served_seeds(os.urandom(64), os.urandom(64), c) if p.f.o
                         else call_seeds(os.urandom(64), os.urandom(64)))
                issued.append((x, p.prove(seeds, ops[x][2]), seeds))
            if r >= 1:   # the previous round's roots, read while this round's calls run
                for x, c, seeds in issued[-4:-2]:
                    out["calls"].append(_pipeline_check(ps[x], ops[x], c, seeds))
        for x, c, seeds in issued[-2:]:
            out["calls"].append(_pipeline_check(ps[x], ops[x], c, seeds))
        try:
            ps[0].root(0)
            out["stale_refused"] = False
        except RuntimeError as e:
            out["stale_refused"] = "isn't one of the last two calls" in str(e)
        for x, c, seeds in issued[-2:]:
            f, Ax = ps[x].f, ops[x][0]
            picks = v31.fill_picks(f, os.urandom(64), q)
            ok, _, line = v31.fill_verdict(
                f, seeds, ps[x].root(c), picks,
                lambda i, j, s, x=x, c=c, Ax=Ax: {**ps[x].answer(c, i, j, s), "A": None if Ax is None else Ax[i]},
                f"pipeline call {c}")
            out["verdicts"].append({"ok": ok, "line": line})
        out.update(accepted=all(v["ok"] for v in out["verdicts"]), verdict=" | ".join(v["line"] for v in out["verdicts"]))
    finally:
        for p in ps:
            p.close()
    out["good"] = out["accepted"] and out["stale_refused"] and all(r["root_equal"] for r in out["calls"])
    return out


def gate_point_levels(lib: Lib, rng: np.random.Generator) -> dict:
    """`point_weights`, `levels_async` and the one-thread rule: a filler's three calls back to back, the first two on
    weights a served Prover's A buffer holds, the third on its own after `set_weights` (which leaves the holder's buffer
    as it was), each call's levels copied behind its commit against the reference's, and a second thread refused."""
    m, n, k, lp, L, row = 256, 256, 2048, 256, 2048, 1024
    with Prover(n, 128, k, lp, L, row, served=True, lib=lib) as holder, Prover(m, n, k, lp, L, row, lib=lib) as p:
        f, B = p.f, [v31.operands(p.f, rng, "llm")[1] for _ in range(2)]
        p.point_weights(holder.load_a(B[0]))
        bufs, seeds, equal = [np.zeros(8 * (2 << p.info()["depth"]) - 8, np.uint64) for _ in range(3)], [], []
        for x, buf in enumerate(bufs):   # call x, back to back
            if x == 2:
                p.set_weights(B[1])
            seeds.append(call_seeds(os.urandom(64), os.urandom(64)))
            p.levels_async(p.prove(seeds[x]), buf.ctypes.data)
        p.sync()
        for x, (sd, buf) in enumerate(zip(seeds, bufs)):
            values = v31.fill_buffer(f, *v31.fill_values(f, v31.pad_rows(sd, range(m), k), B[x // 2], v31.Coins.expand(f, sd)))
            equal.append(buf.byteswap().tobytes() == v31.levels_bytes(v31.fill_commit(f, values, p.ks, x * f.leaves).levels))
        kept = holder.pad() == B[0].tobytes()
        with ThreadPoolExecutor(1) as other:
            rc = other.submit(lib.info, p.h, (ctypes.c_uint64 * 14)()).result()
    return {"name": "point-levels", "levels_equal": equal, "holder_kept": kept, "thread_refused": rc == -1,
            "good": all(equal) and kept and rc == -1}


def _pipeline_check(p: Prover, ops: tuple, c: int, seeds: dict[str, bytes]) -> dict:
    f, (A, B, _) = p.f, ops
    if A is None:
        A = v31.pad_rows(seeds, range(f.m), f.k)
    values = v31.fill_buffer(f, *v31.fill_values(f, A, B, v31.Coins.expand(f, seeds)))
    ref = v31.fill_commit(f, values, p.ks, c * f.leaves)
    return {"shape": f"{'served' if f.o else 'filler'} {f.m}x{f.n}x{f.k} L {f.L}", "call": c, "root_equal": p.root(c) == ref.root}


def cmd_gate(a: argparse.Namespace) -> int:
    lib = Lib(a.lib)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    good = True
    only = set(a.only.split(",")) if a.only else None
    for line in GATES.strip().splitlines():
        name, kind, m, n, k, lp, L, row, ak, cheat = line.split()
        if only and kind not in only:
            continue
        rep = gate_case(lib, name, kind == "served", int(m), int(n), int(k), int(lp), int(L), int(row), ak, cheat, a.q, rng)
        (out / f"fill-{name}.json").write_text(json.dumps(rep, indent=1))
        detail = ("root refused" if "root_refused" in rep else
                  f"{'pad ' + str(rep['pad_equal']) + ', ' if 'pad_equal' in rep else ''}values {rep['values_equal']}, "
                  f"levels {rep['levels_equal']}")
        print(f"{name}: {'good' if rep['good'] else 'BAD'} exact {rep['exact']} ({detail}); {rep['verdict']}", flush=True)
        good &= rep["good"]
    rep = gate_pipeline(lib, a.q, rng)
    (out / "fill-pipeline.json").write_text(json.dumps(rep, indent=1))
    print(f"pipeline: {'good' if rep['good'] else 'BAD'} roots {sum(r['root_equal'] for r in rep['calls'])}/"
          f"{len(rep['calls'])} equal, stale call refused {rep['stale_refused']}; {rep['verdict']}", flush=True)
    good &= rep["good"]
    rep = gate_point_levels(lib, rng)
    (out / "fill-point-levels.json").write_text(json.dumps(rep, indent=1))
    print(f"point-levels: {'good' if rep['good'] else 'BAD'} levels {sum(rep['levels_equal'])}/3 equal, holder kept "
          f"{rep['holder_kept']}, second thread refused {rep['thread_refused']}", flush=True)
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
    p.add_argument("--only", default="", help="the gates of these kinds only (fill, served; comma-separated)")
    a = ap.parse_args(argv[1:])
    return {"gate": cmd_gate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
