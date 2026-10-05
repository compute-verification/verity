"""The fp4-nvf4 VU witness on the device: the rows of :mod:`gpu.nvf4.circuit`'s unit (24 per VU, VU-major) and epilogue
for every VU at once.

    gen = NVF4Generator(statement(), device)      # preprocessing: the compiled unit's program, dense lookup tables
    rows = gen.run(A, B, Y)                       # A, B: (N, 24, 68) operand words, Y: (N,) claimed FP32 words -> Rows

* the chain: B-Ligero's device hint generator (``archive/direct/ligero/fp4/hints_device.py``: ``public_rows_torch`` +
  ``unit_hints_torch``, bit-identical to its numpy reference) step by step over the ``N`` VUs, ``c`` of step ``s + 1`` the
  output word of step ``s`` from ``c_0 = 0``;
* the operand hints of :func:`gpu.nvf4.circuit._decode` (pair codes, scale pair codes, ``any`` / ``part`` / ``G`` and its
  selector) from the operand words and the decode;
* every derived row (``prod``, ``aux``, ``dig``, ``lk``, ``inv``, ``invz``) by the unit's ``System.program`` as one array
  program over all ``24 N`` units, in exact int64 (every row but the inverses has a known integer interval inside
  ``(-2^62, 2^62)``), reduced mod p into the committed columns.

A VU off the domain, whose hints do not close, or whose chain does not reach its claimed word is marked in ``Rows.bad``.
``check_rows`` evaluates every assertion and quadratic of the unit on the produced columns (the self-check).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from archive.direct.ligero.compile import Expr
from archive.direct.ligero.fp4 import hints_device as HD
from archive.direct.ligero.fp4 import relation as R

from ..field import P
from .circuit import NVF4Statement

__all__ = ["NVF4Generator", "GraphedNVF4Generator", "Rows", "check_rows", "public_words"]

K, GROUP, GROUPS, STEPS, WORDS = R.K, R.GROUP, R.GROUPS, R.STEPS, R.WORDS
FIELD_KINDS = ("inv", "invz")


@dataclass
class Rows:
    units: torch.Tensor      # (N * 24, unit columns) int64 mod P, VU-major
    epilogue: torch.Tensor   # (N, epilogue columns) int64 mod P
    public: torch.Tensor     # (N, 3) int64: sign bit, exponent field, fraction of the final word
    bad: torch.Tensor        # (N,) bool
    seconds: dict


def public_words(y: torch.Tensor) -> torch.Tensor:
    """FP32 words (N,) -> (N, 3) ``(s, t, f)``: the three public components (``relation.py`` ``y_end``)."""
    return torch.stack([(y >> 31) & 1, (y >> 23) & 0xFF, y & (2**23 - 1)], dim=1)


def _field_pow(x: torch.Tensor, e: int) -> torch.Tensor:
    out = torch.ones_like(x)
    base = x % P
    while e:
        if e & 1:
            out = (out * base) % P
        base = (base * base) % P
        e >>= 1
    return out


class NVF4Generator:
    """Preprocessing (the unit's program, dense lookup tables, column layout) once; ``run`` per batch."""

    def __init__(self, st: NVF4Statement, device) -> None:
        self.dev = dev = torch.device(device)
        self.st = st
        sys_ = st.unit.sys
        self.rows = sys_.rows
        self.hint_rows = {n: i for i, n in sys_.hints}
        self.bl_names = [n for _, n in sys_.hints if not n.startswith("x.")]
        self.bl_sys = SimpleNamespace(hints=[(None, n) for n in self.bl_names])
        self.ops = [op for op in sys_.program if op[0] != "hint"]
        field_rows = {op[1] for op in self.ops if op[0] in FIELD_KINDS}
        for op in self.ops:
            if op[0] in FIELD_KINDS:
                continue
            exprs = [a for a in op[2:] if isinstance(a, Expr)]
            if op[0] == "lk":
                exprs = [op[2].key]
            if any(j in field_rows for e in exprs for j in e.lin):
                raise ValueError(f"row {self.rows[op[1]].name}: an integer op reads a field inverse")
        self.dense: dict[tuple[int, int], tuple[int, torch.Tensor]] = {}
        for op in self.ops:
            if op[0] == "lk":
                rec, ci = op[2], op[3]
                lo = min(rec.keys)
                t = torch.zeros(max(rec.keys) - lo + 1, dtype=torch.int64)
                t[torch.tensor(rec.keys) - lo] = torch.tensor(rec.cols[ci][0], dtype=torch.int64)
                self.dense[(id(rec), ci)] = (lo, t.to(dev))
        self.col_rows = st.ue.col_rows[1:]
        self.ecols = st.ee.col_names
        if self.ecols != ["one", "e.s", "e.t", "e.f"] and self.ecols != ["one", "s", "t", "f"]:
            raise ValueError(f"epilogue columns {self.ecols}")
        self.num = torch.tensor(R.NUMERATOR, dtype=torch.int64, device=dev)
        self.pos_w = torch.ones(R.POSITIONS, dtype=torch.int64, device=dev) << (
            R.RADIX * torch.arange(R.POSITIONS, dtype=torch.int64, device=dev))
        # one step of the chain is ~400 elementwise ops on N lanes, 24 times in sequence: launch-bound unless fused
        # (graphed witness 28.7 -> 12.1 ms at 4096 VUs on the 5090, same rows; VERITY_NVF4_COMPILE=0: eager)
        self._step = HD.unit_hints_torch
        if dev.type == "cuda" and os.environ.get("VERITY_NVF4_COMPILE", "1") == "1":
            self._step = torch.compile(HD.unit_hints_torch, dynamic=False)

    # --- the chain (B-Ligero's hints) + the decode hints ------------------------------------------------------------------
    def _chain(self, A: torch.Tensor, B: torch.Tensor, Y: torch.Tensor, variant: str | None = None):
        """``A``/``B`` (N, 24, 68) int64, ``Y`` (N,) -> hint values {name: (N * 24,)}, bad (N,), final words (N,).
        ``variant``: the hints of a prover following the WRONG rule ``fp4/witness.VARIANTS[..]`` (numpy, host; negatives),
        its accumulator fed forward; the operand decode stays honest."""
        N = A.shape[0]
        c = torch.zeros(N, dtype=torch.int64, device=self.dev)
        bad = torch.zeros(N, dtype=torch.bool, device=self.dev)
        # the decode does not read the accumulator: every step's in one call (step-major lanes)
        pub_all = HD.public_rows_torch(A.transpose(0, 1).reshape(STEPS * N, WORDS), B.transpose(0, 1).reshape(STEPS * N, WORDS),
                                       self.num).view(HD.N_PINS, STEPS, N)
        hs = []
        for s in range(STEPS):
            a, b = A[:, s], B[:, s]
            pub = pub_all[:, s]
            if variant is None:
                h, bd, c = self._step(self.bl_sys, c, a, b, pub, self.num, self.pos_w)
                bad |= bd
            else:
                from archive.direct.ligero.fp4.witness import unit_hints
                H, yv = unit_hints(c.cpu().numpy(), a.cpu().numpy(), b.cpu().numpy(), variant=variant)
                h = torch.as_tensor(np.stack([H[n] for n in self.bl_names]), dtype=torch.int64, device=self.dev)
                c = torch.as_tensor(yv, dtype=torch.int64, device=self.dev)
            hs.append(h)
        bad |= c != Y
        H = torch.stack(hs, dim=2).reshape(len(self.bl_names), N * STEPS)        # (n, N, 24) -> VU-major lanes
        pub = pub_all.transpose(1, 2).reshape(HD.N_PINS, N * STEPS)
        vals = {self.hint_rows[n]: H[i] for i, n in enumerate(self.bl_names)}
        a, b = A.reshape(N * STEPS, WORDS), B.reshape(N * STEPS, WORDS)
        hr = self.hint_rows
        for i in range(K):
            vals[hr[f"x.p{i}"]] = 16 * a[:, i] + b[:, i]
        part = pub[HD._PART:HD._G]
        G = pub[HD._G]
        cands = []
        for g in range(GROUPS):
            vals[hr[f"x.s{g}"]] = 128 * a[:, K + g] + b[:, K + g]
            nprod = (pub[GROUP * g: GROUP * (g + 1)] * pub[K + GROUP * g: K + GROUP * (g + 1)] != 0).any(0)
            vals[hr[f"x.g{g}.any"]] = nprod.to(torch.int64)
            vals[hr[f"x.g{g}.part"]] = part[g]
            X = pub[HD._X + g]
            cands.append(torch.where(part[g] == 1, X - R.FRAC_BITS, torch.full_like(X, R.GRID_FLOOR)))
        cands.append(torch.full_like(G, R.GRID_FLOOR))
        vals[hr["x.G"]] = G
        taken = torch.zeros_like(G, dtype=torch.bool)
        for g, cg in enumerate(cands):
            hit = (cg == G) & ~taken
            vals[hr[f"x.G.sel{g}"]] = hit.to(torch.int64)
            taken |= hit
        bad |= ~taken.view(N, STEPS).all(1)
        return vals, bad, c

    # --- the derived rows ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _eval(e: Expr, vals: dict, like: torch.Tensor) -> torch.Tensor:
        out = None
        for j, v in e.lin.items():
            t = vals[j] if v == 1 else vals[j] * v
            out = t if out is None else out + t
        if out is None:
            return torch.full_like(like, e.c)
        return out + e.c if e.c else out

    def _program(self, vals: dict, like: torch.Tensor) -> None:
        ev = self._eval
        for op in self.ops:
            kind, idx = op[0], op[1]
            if kind == "prod":
                vals[idx] = ev(op[2], vals, like) * ev(op[3], vals, like)
            elif kind == "aux":
                vals[idx] = ev(op[2], vals, like)
            elif kind == "dig":
                x, c, w = op[2], op[3], op[4]
                vals[idx] = (ev(x, vals, like) >> c) & (2**w - 1)
            elif kind == "lk":
                rec, ci = op[2], op[3]
                lo, t = self.dense[(id(rec), ci)]
                key = ev(rec.key, vals, like) - lo
                vals[idx] = t[key.clamp(0, t.shape[0] - 1)]
            elif kind == "inv":
                vals[idx] = _field_pow(ev(op[2], vals, like), P - 2)
            elif kind == "invz":
                vals[idx] = _field_pow(ev(op[2], vals, like), P - 2)          # 0 -> 0
            else:
                raise ValueError(f"program op {kind} not supported")

    def run(self, A: torch.Tensor, B: torch.Tensor, Y: torch.Tensor, sync: bool = False, variant: str | None = None) -> Rows:
        t0 = time.perf_counter()
        N = A.shape[0]
        vals, bad, y = self._chain(A, B, Y, variant)
        if sync:
            torch.cuda.synchronize(self.dev)
        t1 = time.perf_counter()
        like = vals[self.hint_rows["acc.s"]]
        self._program(vals, like)
        U = N * STEPS
        units = torch.empty((U, 1 + len(self.col_rows)), dtype=torch.int64, device=self.dev)
        units[:, 0] = 1
        for c, r in enumerate(self.col_rows, 1):
            units[:, c] = vals[r]
        units.remainder_(P)
        pw = public_words(y)
        epi = torch.cat([torch.ones((N, 1), dtype=torch.int64, device=self.dev), pw], dim=1)
        if sync:
            torch.cuda.synchronize(self.dev)
        return Rows(units, epi, pw, bad, {"chain": t1 - t0, "program": time.perf_counter() - t1})


class GraphedNVF4Generator:
    """``NVF4Generator.run`` captured once as a CUDA graph on static operand buffers (uint8 words, int64 claims) and
    replayed per call; ``run`` returns the graph's ``Rows`` (overwritten by the next ``run``)."""

    def __init__(self, gen: NVF4Generator, N: int) -> None:
        dev = gen.dev
        self.gen = gen
        self.a8 = torch.zeros((N, STEPS, WORDS), dtype=torch.uint8, device=dev)
        self.b8 = torch.zeros((N, STEPS, WORDS), dtype=torch.uint8, device=dev)
        self.y = torch.zeros((N,), dtype=torch.int64, device=dev)
        side = torch.cuda.Stream(dev)
        side.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(side):
            self._body()
        torch.cuda.current_stream(dev).wait_stream(side)
        torch.cuda.synchronize(dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            rows = self._body()
        torch.cuda.synchronize(dev)
        self.rows = Rows(rows.units, rows.epilogue, rows.public, rows.bad, {})

    def _body(self) -> Rows:
        return self.gen.run(self.a8.to(torch.int64), self.b8.to(torch.int64), self.y)

    def run(self, a8, b8, y) -> Rows:
        """``a8`` / ``b8`` (N, 24, 68) host uint8 operand words, ``y`` (N,) host int64 claimed words."""
        self.a8.copy_(torch.as_tensor(a8))
        self.b8.copy_(torch.as_tensor(b8))
        self.y.copy_(torch.as_tensor(y))
        self.graph.replay()
        return self.rows


def check_rows(st: NVF4Statement, rows: Rows) -> dict:
    """Every linear assertion and quadratic of the unit and the epilogue on ``rows``, mod p: {constraint: failing units}."""
    bad: dict[str, int] = {}
    for ctx, M, ex in ((st.unit, rows.units, st.ue), (st.epi, rows.epilogue, st.ee)):
        col_of = {r: c for c, r in enumerate(ex.col_rows) if r >= 0}
        vals: dict[int, torch.Tensor] = {r: M[:, c] for r, c in col_of.items()}
        like = M[:, 0]
        for r in ctx.sys.rows:                             # product rows: recomputed from the columns (mod p)
            if r.kind == "prod":
                a, b = r.spec
                vals[r.idx] = (_mod_eval(a, vals, like) * _mod_eval(b, vals, like)) % P

        for lin in ctx.sys.linear:
            v = _mod_eval(Expr(None, lin.lin, lin.c), vals, like)
            n = int((v != 0).sum())
            if n:
                bad[lin.name] = n
        for q in ctx.sys.quadratic:
            v = (_mod_eval(q.a, vals, like) * _mod_eval(q.b, vals, like) - _mod_eval(q.c, vals, like)) % P
            n = int((v != 0).sum())
            if n:
                bad[q.name] = n
    return bad


def _mod_eval(e: Expr, vals: dict, like: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(like, e.c % P)
    for j, v in e.lin.items():
        out = (out + vals[j] * (v % P)) % P
    return out


def _instances_chunk(args):
    from archive.direct.ligero.fp4.chain import instances_fp4
    lo, n = args
    inst = instances_fp4(n, start=lo)
    A = np.stack([np.asarray(v[0], dtype=np.uint8) for v in inst])
    B = np.stack([np.asarray(v[1], dtype=np.uint8) for v in inst])
    Y = np.array([int(v[3]) for v in inst], dtype=np.int64)
    return A, B, Y


def load_instances(lo: int, n: int, procs: int = 1):
    """VUs ``[lo, lo + n)`` of the frozen NVFP4 set (``fp4/chain.instances_fp4``: seed, edge families, the model's
    accumulators) as ``A`` / ``B`` (n, 24, 68) uint8 and the claimed final words ``Y`` (n,) int64."""
    step = max(1, -(-n // (max(1, procs) * 4)))
    chunks = [(s, min(step, lo + n - s)) for s in range(lo, lo + n, step)]
    if procs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(procs) as pool:
            parts = pool.map(_instances_chunk, chunks)
    else:
        parts = [_instances_chunk(c) for c in chunks]
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def main(argv=None) -> int:
    import argparse
    import json

    from .circuit import statement

    ap = argparse.ArgumentParser(prog="python -m gpu.nvf4.witness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest", help="generator on the first --vus frozen VUs; every constraint checked on the rows")
    s.add_argument("--vus", type=int, default=28)
    s.add_argument("--device", default="cuda")
    s.add_argument("--procs", type=int, default=1)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    st = statement()
    gen = NVF4Generator(st, args.device)
    t1 = time.perf_counter()
    A, B, Y = load_instances(0, args.vus, args.procs)
    t2 = time.perf_counter()
    rows = gen.run(torch.from_numpy(A).to(gen.dev).long(), torch.from_numpy(B).to(gen.dev).long(), torch.from_numpy(Y).to(gen.dev), sync=True)
    t3 = time.perf_counter()
    fails = check_rows(st, rows)
    out = {"vus": args.vus, "bad_vus": int(rows.bad.sum()), "failing_constraints": len(fails), "fails": dict(list(fails.items())[:20]),
           "units_shape": list(rows.units.shape), "compile_s": t1 - t0, "instances_s": t2 - t1, "gen_s": t3 - t2, "seconds": rows.seconds}
    print(json.dumps(out, indent=1))
    return 0 if not fails and not out["bad_vus"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
