"""The checker-v2 VU witness on the device: the rows of ``gpu.v2.export`` (unit segment, limb epilogue, public words)
as array programs over every VU at once, instead of the export's per-VU Python loop.

    gen = Generator(ops_torch(device))          # preprocessing: circuits, dense tables (a function of Params only)
    rows = gen.run(A, B, Y16)                   # A, B: (N, K) operand words, Y16: (N,) claimed words -> Rows

Same values as the export, row for row (``python -m gpu.v2.witness selftest`` against ``gpu.v2.export.API.vu_rows``
on the laptop with numpy; ``python -m gpu.v2.witness compare DIR`` against an export directory on the pod).

* operand decode and products: one T_OP gather over all ``N x K`` words;
* the chain: the state passes from unit to unit, so the ``K / 16`` units run in order, each group step vectorised
  over the VUs and the group's candidates -- the arithmetic of ``checker.v2.step.gen_unit`` / ``_group``;
* the derived columns (lookup outputs, materialised keys, range limbs): ``gkr_export.v2.derive`` as one array
  program over all rows;
* the tables: dense arrays of ``checker.v2.tables`` (explicit tables from ``fn``, the four implicit ones from the
  ``itable`` closed forms of ``gkr_export.v2._implicit_row``, spot-checked against ``fn``).

A predicate the reference rejects on (an operand word with no T_OP row, a saturated non-final unit, any range or
identity of the gadgets) marks the VU in ``Rows.bad`` instead of raising; the export writes no row for such a VU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import numpy as np
from verity_numerical.checker import REAL, Params, Reject
from verity_numerical.checker.v2.params import V2
from verity_numerical.checker.v2.tables import (
    align4_table,
    lead_table,
    leadnorm_table,
    op_table,
    shift_table,
    sshift_hi_table,
    sshift_lo_table,
    tnorm_table,
)
from verity_numerical.gkr_export import v2 as GV2
from verity_numerical.gkr_export.circuit import CHUNK

from . import epilogue_limbs as EL
from . import fp8 as FP8

P = 2**31 - 2**27 + 1
STEP = 16


# ---------------------------------------------------------------------------------------------------------------------
# array backends
# ---------------------------------------------------------------------------------------------------------------------

class Ops:
    """The few array operations the generator needs, on numpy (host) or torch (device)."""

    def __init__(self, torch_device=None) -> None:
        self.torch = None
        self.device = None
        if torch_device is not None:
            import torch

            self.torch = torch
            self.device = torch.device(torch_device) if isinstance(torch_device, str) else torch_device

    def asarray(self, a):
        a = np.ascontiguousarray(np.asarray(a).astype(np.int64, copy=False))
        if self.torch is None:
            return a
        return self.torch.from_numpy(a).to(self.device)

    def full(self, shape, v: int):
        if self.torch is None:
            return np.full(shape, v, dtype=np.int64)
        return self.torch.full(shape if isinstance(shape, tuple) else (shape,), v, dtype=self.torch.int64, device=self.device)

    def falses(self, n: int):
        if self.torch is None:
            return np.zeros(n, dtype=bool)
        return self.torch.zeros(n, dtype=self.torch.bool, device=self.device)

    def i64(self, b):
        return b.astype(np.int64) if self.torch is None else b.long()

    def amax(self, a, axis: int):
        return a.max(axis=axis) if self.torch is None else self.torch.amax(a, dim=axis)

    def clamp(self, a, lo: int, hi: int):
        return np.clip(a, lo, hi) if self.torch is None else self.torch.clamp(a, lo, hi)

    def clamp_min(self, a, lo: int):
        return np.maximum(a, lo) if self.torch is None else self.torch.clamp(a, min=lo)

    def cat(self, xs, axis: int):
        return np.concatenate(xs, axis=axis) if self.torch is None else self.torch.cat(xs, dim=axis)

    def stack(self, xs, axis: int):
        return np.stack(xs, axis=axis) if self.torch is None else self.torch.stack(xs, dim=axis)

    def where(self, c, a, b):
        return np.where(c, a, b) if self.torch is None else self.torch.where(c, a, b)

    def mod(self, a, m: int):
        return np.mod(a, m) if self.torch is None else self.torch.remainder(a, m)

    def all(self, b, axis: int):
        return b.all(axis=axis) if self.torch is None else b.all(dim=axis)


# ---------------------------------------------------------------------------------------------------------------------
# dense tables
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Dense:
    """A table as ``outs[(key - lo)]`` (``(size, nouts)`` int64) and the domain mask (None = every key in range)."""

    name: str
    lo: int
    outs: object
    valid: object | None

    @property
    def size(self) -> int:
        return int(self.outs.shape[0])

    def gather(self, ops: Ops, key):
        """``(rows, ok)``: the rows at ``key`` (clamped; garbage where ``ok`` is false) and the in-domain mask."""
        idx = key - self.lo if self.lo else key
        ok = (idx >= 0) & (idx < self.size)
        idx = ops.clamp(idx, 0, self.size - 1)
        if self.valid is not None:
            ok = ok & self.valid[idx]
        return self.outs[idx], ok


def _explicit(t) -> tuple[np.ndarray, np.ndarray | None]:
    n = t.hi - t.lo
    outs = np.zeros((n, len(t.outs)), dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for k in range(t.lo, t.hi):
        if t.domain is not None and not t.domain(k):
            valid[k - t.lo] = False
            continue
        outs[k - t.lo] = t.fn(k)
    return outs, (None if valid.all() else valid)


def _implicit(t, p: Params) -> np.ndarray:
    """The ``itable`` closed form over every key (``gkr_export.v2._implicit_row``, vectorised)."""
    parts = GV2.implicit_table_header(t, p).split()
    keys = np.arange(t.hi - t.lo, dtype=np.int64)
    if parts[0] == "shift":
        scale, R = int(parts[2]), int(parts[3])
        x, rem = keys // R, keys % R
        d, s = rem // 2, rem % 2
        v = np.right_shift(np.left_shift(x, scale), d)
        out = np.where(s == 1, -v, v)
    else:
        vb, k_lo = int(parts[1]), int(parts[3])
        koff, v_lo = keys >> vb, keys & (2**vb - 1)
        k = koff + k_lo
        out = np.where(k >= 0, np.left_shift(v_lo, np.maximum(k, 0)), np.right_shift(v_lo, np.maximum(-k, 0)))
    rng = np.random.default_rng(0)
    for key in [0, 1, t.hi - 1, (t.hi - 1) // 3, *rng.integers(0, t.hi, 64).tolist()]:
        if (int(out[key]),) != tuple(t.fn(int(key))):
            raise AssertionError(f"{t.name}: closed form differs from fn at key {key}")
    return out.reshape(-1, 1)


def dense_tables(p: Params = REAL) -> dict[str, tuple[np.ndarray, np.ndarray | None, int]]:
    out = {}
    for t in (op_table(p), align4_table(p), lead_table(p), leadnorm_table(p)):
        outs, valid = _explicit(t)
        out[t.name] = (outs, valid, t.lo)
    for t in (shift_table(p), sshift_hi_table(p), sshift_lo_table(p), tnorm_table(p)):
        out[t.name] = (_implicit(t, p), None, t.lo)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# predicates on arrays (the reference's IntCtx, recording instead of raising)
# ---------------------------------------------------------------------------------------------------------------------

class ArrayCtx:
    """``rng`` / ``eq`` / ``boolean`` over arrays: a violated predicate marks the VU in ``bad``."""

    def __init__(self, ops: Ops, bad) -> None:
        self.ops = ops
        self.bad = bad

    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int):
            if not lo <= x < hi:
                raise Reject(f"{name}: constant {x} not in [{lo}, {hi})")
            return
        self.bad = self.bad | (x < lo) | (x >= hi)

    def eq(self, name: str, a, b) -> None:
        d = a - b
        if isinstance(d, int):
            if d:
                raise Reject(f"{name}: constant identity fails")
            return
        self.bad = self.bad | (d != 0)

    def boolean(self, name: str, b) -> None:
        if isinstance(b, int):
            if b not in (0, 1):
                raise Reject(f"{name}: constant not a bit")
            return
        self.bad = self.bad | ((b != 0) & (b != 1))


# ---------------------------------------------------------------------------------------------------------------------
# derive as an array program
# ---------------------------------------------------------------------------------------------------------------------

def derive_arrays(ctx, vals: list, tables: dict[str, Dense], ops: Ops, n: int) -> list:
    """``gkr_export.v2.derive`` over arrays of ``n`` rows: fills every derived column in column order (a lookup off
    its table gives 0, as there)."""
    cache: dict[int, object] = {}
    rows_cache: dict[tuple[str, int], tuple[object, object]] = {}

    def wire(w: int):
        kind = ctx.wires[w]
        if kind[0] == "in":
            v = vals[kind[1]]
            if v is None:
                raise Reject(f"column {ctx.cols[kind[1]]} evaluated before it is set")
            return v
        v = cache.get(w)
        if v is None:
            v = lin(kind[1]) * lin(kind[2])
            cache[w] = v
        return v

    def lin(x):
        acc = None
        for w, c in x.terms.items():
            t = wire(w)
            if c != 1:
                t = t * c
            acc = t if acc is None else acc + t
        if acc is None:
            return ops.full(n, x.const)
        return acc + x.const if x.const else acc

    for col, d in enumerate(ctx.derived):
        if d is None:
            continue
        kind = d[0]
        if kind == "const":
            vals[col] = ops.full(n, d[1])
        elif kind == "lin":
            vals[col] = lin(d[1])
        elif kind == "lo16":
            vals[col] = lin(d[1]) & (2**CHUNK - 1)
        elif kind == "hi16":
            vals[col] = lin(d[1]) >> CHUNK
        elif kind == "row":
            _, table, key, idx = d
            ck = (table.name, id(key))
            if ck not in rows_cache:
                rows_cache[ck] = tables[table.name].gather(ops, lin(key))
            rows, ok = rows_cache[ck]
            vals[col] = ops.where(ok, rows[..., idx], 0)
        else:
            raise ValueError(f"unknown derived kind {kind}")
    return vals


# ---------------------------------------------------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Rows:
    units: object        # (N * steps, unit columns) int64 mod P, VU-major
    epilogue: object     # (N, epilogue columns) int64 mod P
    public: object       # (N,) int64
    bad: object          # (N,) bool: the reference rejects this VU at witness generation
    seconds: dict


class Generator:
    """Preprocessing (circuits, dense tables, hint layout) once; ``run`` per batch."""

    def __init__(self, ops: Ops, p: Params = REAL, host_tables: dict | None = None) -> None:
        self.ops, self.p = ops, p
        self.fp8 = FP8.is_fp8(p)
        self.k = sum(p.groups)
        if not self.fp8 and self.k != STEP:
            raise Reject(f"the BF16 v2 unit has {STEP} products, got {self.k}")
        self.unit = GV2.build_unit_circuit_v2(p)
        self.epi = FP8.build_epilogue_circuit_fp8(p) if self.fp8 else EL.build_epilogue_circuit_limbs(p)
        host = host_tables if host_tables is not None else dense_tables(p)
        self.tables = {name: Dense(name, lo, ops.asarray(outs), None if valid is None else self._bool(valid))
                       for name, (outs, valid, lo) in host.items()}
        k = sum(p.groups)
        uctx, paths = self.unit
        want = [f"h.groups[{g}].{n}" for g in range(len(p.groups))
                for n in ("E", "acc_split.M_hi", "acc_split.M_lo", "sum.sgn", "sum.v_hi", "sum.v_lo")]
        if sorted(paths) != sorted(want):
            raise AssertionError(f"unit hint paths changed: {paths}")
        self.hint_order = [want.index(pth) for pth in paths]
        names = uctx.cols[:9 + 2 * k]
        if names[:5] != ["one", "x.s", "x.e", "x.M", "x.z"] or names[5 + 2 * k:9 + 2 * k] != ["y.s", "y.e", "y.M", "y.z"]:
            raise AssertionError("unit column layout changed")
        ectx, epaths = self.epi
        G = len(p.groups)
        if ectx.cols[:6 + 2 * G] != ["one", "s", "e", "M", "z", *[f"{w}[{g}]" for g in range(G) for w in ("ovf", "sgn")], "y16"][:6 + 2 * G]:
            raise AssertionError("epilogue column layout changed")
        self.epi_hints = [pth[2:] for pth in epaths]

    def _bool(self, valid: np.ndarray):
        if self.ops.torch is None:
            return valid
        return self.ops.torch.from_numpy(valid).to(self.ops.device)

    # -- the chain ----------------------------------------------------------------------------------------------------
    def chain(self, A, B):
        """Every unit's incoming state, outgoing state and 12 hints (``(N, steps, 4|4|12)``), the last unit's
        ``(ovf, sgn)`` per group, the final state, and ``bad``."""
        ops, p, T = self.ops, self.p, self.tables
        N, K = int(A.shape[0]), int(A.shape[1])
        if K % self.k:
            raise Reject(f"operand count must be a multiple of {self.k}")
        steps = K // self.k
        R = 2 * V2.dt_rows(p)
        mb, vb, L = V2.mb(p), V2.vb(p), p.l_hi + 1
        ra, oka = T["T_OP"].gather(ops, A)
        rb, okb = T["T_OP"].gather(ops, B)
        bad = ~(ops.all(oka, 1) & ops.all(okb, 1))
        u = ra[..., 1] * rb[..., 1]
        ep = ra[..., 2] + rb[..., 2]
        s4 = ra[..., 0] + 2 * rb[..., 0]
        del ra, rb
        s_c, e_c, M_c, z_c = ops.full(N, 0), ops.full(N, p.floor), ops.full(N, 0), ops.full(N, 1)
        xs, ys, hs = [], [], []
        flags = []
        for i in range(steps):
            xs.append(ops.stack([s_c, e_c, M_c, z_c], 1))
            start = i * self.k
            flags = []
            hint = []
            for size in p.groups:
                sl = slice(start, start + size)
                start += size
                cands = ops.cat([e_c[:, None], ep[:, sl]], 1)
                s4s = ops.cat([s_c[:, None], s4[:, sl]], 1)
                E = ops.clamp_min(ops.amax(cands, 1), p.floor)
                al, ok = T["ALIGN4"].gather(ops, 4 * (E[:, None] - cands) + s4s)
                dt, sp, isz = al[..., 0], al[..., 1], al[..., 2]
                bad = bad | ~ops.all(ok, 1)
                M_hi, M_lo = M_c >> mb, M_c & (2**mb - 1)
                bad = bad | (M_c < 0) | (M_c >= 2**p.acc_sig)
                hi, ok1 = T["SSHIFT_HI"].gather(ops, M_hi * R + 2 * dt[:, 0] + sp[:, 0])
                lo, ok2 = T["SSHIFT_LO"].gather(ops, M_lo * R + 2 * dt[:, 0] + sp[:, 0])
                tp, ok3 = T["SHIFT"].gather(ops, u[:, sl] * R + 2 * dt[:, 1:] + sp[:, 1:])
                S = hi[:, 0] + lo[:, 0] + tp[..., 0].sum(1)
                sgn = ops.i64(S < 0)
                v = S - 2 * sgn * S
                v_hi, v_lo = v >> vb, v & (2**vb - 1)
                lh, ok4 = T["LEAD"].gather(ops, v_hi)
                ll, ok5 = T["LEAD"].gather(ops, v_lo)
                nz_h = lh[:, 0]
                l = lh[:, 1] + nz_h * vb + ll[:, 1] - nz_h * ll[:, 1]
                ln, ok6 = T["LEADNORM"].gather(ops, l + L * (E - p.floor))
                koff, A_, e_hat, z, ovf, isfloor = (ln[:, j] for j in range(6))
                tn, ok7 = T["TNORM"].gather(ops, v_lo + 2**vb * koff)
                M = v_hi * A_ + tn[:, 0]
                att = isz.sum(1) + isfloor - 1
                bad = bad | (att < 0) | (att >= size + 3) | ~(ok1 & ok2 & ops.all(ok3, 1) & ok4 & ok5 & ok6 & ok7)
                hint += [E, M_hi, M_lo, sgn, v_hi, v_lo]
                s_c, e_c, M_c, z_c = sgn - sgn * z, e_hat, M, z
                flags.append((ovf, sgn))
            ys.append(ops.stack([s_c, e_c, M_c, z_c], 1))
            hs.append(ops.stack(hint, 1))
            if i < steps - 1:
                for ovf, _ in flags:
                    bad = bad | (ovf != 0)
        return ops.stack(xs, 1), ops.stack(ys, 1), ops.stack(hs, 1), flags, (s_c, e_c, M_c, z_c), bad

    # -- rows ---------------------------------------------------------------------------------------------------------
    def run(self, A, B, Y16, sync=None) -> Rows:
        ops, p = self.ops, self.p
        t0 = time.perf_counter()
        N, K = int(A.shape[0]), int(A.shape[1])
        steps, k = K // self.k, self.k
        xs, ys, hs, flags, state, bad = self.chain(A, B)
        if sync:
            sync()
        t1 = time.perf_counter()
        uctx, _ = self.unit
        n = N * steps
        vals: list = [None] * len(uctx.cols)
        vals[0] = ops.full(n, 1)
        for j in range(4):
            vals[1 + j] = xs[..., j].reshape(n)
            vals[5 + 2 * k + j] = ys[..., j].reshape(n)
        Au, Bu = A.reshape(N, steps, k), B.reshape(N, steps, k)
        for j in range(k):
            vals[5 + j] = Au[..., j].reshape(n)
            vals[5 + k + j] = Bu[..., j].reshape(n)
        base = 9 + 2 * k
        for i, h in enumerate(self.hint_order):
            vals[base + i] = hs[..., h].reshape(n)
        derive_arrays(uctx, vals, self.tables, ops, n)
        units = ops.mod(ops.stack(vals, 1), P)
        del vals
        if sync:
            sync()
        t2 = time.perf_counter()
        # the limb epilogue (``epilogue_limbs.epilogue_witness_limbs``) on the last state and flags; the FP8 epilogue
        # (``fp8.epilogue_witness_fp8``) has no hints: the packed word is an expression of the linked columns
        ectx, _ = self.epi
        h = {}
        if not self.fp8:
            ctx = ArrayCtx(ops, bad)
            s, e_hat, M, z = state
            h = {"M_hi8": M >> 16, "M_lo16": M & 0xFFFF}
            y_hi, y_lo = EL.pack_limbs(ctx, p, state, flags, h)
            lsb = y_hi & 1
            t = y_lo + lsb + (2**15 - 1)
            h.update({"hi2": y_hi >> 1, "lsb": lsb, "carry": t >> 16, "rem": t & 0xFFFF})
            EL.epilogue_limbs(ctx, p, y_hi, y_lo, h)
            bad = ctx.bad
        evals: list = [None] * len(ectx.cols)
        evals[0] = ops.full(N, 1)
        for j, v in enumerate(state):
            evals[1 + j] = v
        for g, (ovf, sgn) in enumerate(flags):
            evals[5 + 2 * g] = ovf
            evals[6 + 2 * g] = sgn
        off = 5 + 2 * len(flags)
        evals[off] = Y16
        for i, name in enumerate(self.epi_hints):
            evals[off + 1 + i] = h[name]
        derive_arrays(ectx, evals, self.tables, ops, N)
        epi = ops.mod(ops.stack(evals, 1), P)
        if sync:
            sync()
        t3 = time.perf_counter()
        return Rows(units, epi, Y16, bad, {"chain": t1 - t0, "unit_rows": t2 - t1, "epilogue": t3 - t2, "total": t3 - t0})


class GraphedGenerator:
    """``Generator.run`` on CUDA captured once as a graph on static operand buffers and replayed per call.  The chain is
    ~2e4 small elementwise / gather launches (96 steps x 2 groups x ~90 ops over ``N`` lanes) and eager torch is
    launch-bound on them; every op has fixed shapes and no host round trip, so the replay issues the same kernels back to
    back.  The operands enter as their 16-bit patterns (int16) and are widened inside the graph; the claimed words too,
    except for the FP8 relations, whose 22-bit packed words enter as int32.  ``run`` returns the graph's ``Rows``, which
    the next ``run`` overwrites; ``seconds`` is empty."""

    def __init__(self, gen: Generator, N: int, K: int) -> None:
        torch = gen.ops.torch
        dev = gen.ops.device
        self.gen, self.torch, self.dev = gen, torch, dev
        self.a16 = torch.zeros((N, K), dtype=torch.int16, device=dev)
        self.b16 = torch.zeros((N, K), dtype=torch.int16, device=dev)
        self.y16 = torch.zeros((N,), dtype=torch.int32 if gen.fp8 else torch.int16, device=dev)
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
        def u16(t):
            return t.to(self.torch.int64) & 0xFFFF

        y = self.y16.to(self.torch.int64) if self.gen.fp8 else u16(self.y16)
        return self.gen.run(u16(self.a16), u16(self.b16), y)

    def run(self, x16: np.ndarray, w16: np.ndarray, y16: np.ndarray) -> Rows:
        """``x16`` / ``w16`` ``(N, K)`` and ``y16`` ``(N,)``: host int16 arrays holding the uint16 operands' bits
        (``y16`` int32 for the FP8 relations)."""
        torch = self.torch
        self.a16.copy_(torch.from_numpy(x16))
        self.b16.copy_(torch.from_numpy(w16))
        self.y16.copy_(torch.from_numpy(y16))
        self.graph.replay()
        return self.rows


# ---------------------------------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------------------------------

def _to_np(a) -> np.ndarray:
    return a if isinstance(a, np.ndarray) else a.cpu().numpy()


def selftest(vus: int = 6, steps: int = 4, seed: int = 7) -> dict:
    """numpy generator vs ``gpu.v2.export.API.vu_rows`` (the export's rows) on random finite VUs, every word class:
    zeros, subnormals, large exponents (a saturating last unit), sign mixes; plus a non-finite operand and a
    saturated non-final unit, which both must be rejected."""
    import random

    from verity_numerical.checker.v2.step import vu_output

    from .export import API

    rng = random.Random(seed)
    p = REAL
    k = sum(p.groups)

    def word() -> int:
        c = rng.random()
        if c < 0.1:
            return rng.randrange(2) << 15                                        # +-0
        if c < 0.2:
            return (rng.randrange(2) << 15) | rng.randrange(1, 128)              # subnormal
        if c < 0.3:
            return (rng.randrange(2) << 15) | (rng.randrange(150, 175) << 7) | rng.randrange(128)   # large, finite products
        if c < 0.3 + 0.04 / steps:
            return (rng.randrange(2) << 15) | (rng.randrange(200, 255) << 7) | rng.randrange(128)   # huge: may saturate
        return (rng.randrange(2) << 15) | (rng.randrange(100, 150) << 7) | rng.randrange(128)

    unit, epi = API.build_unit(), API.build_epilogue()
    As, Bs, Ys, want = [], [], [], []
    for v in range(vus):
        a = [word() for _ in range(steps * k)]
        b = [word() for _ in range(steps * k)]
        if v == vus - 1:
            a[3] = 0x7F80                                                         # +inf: no T_OP row
        if v == vus - 2:
            a[:k] = b[:k] = [0x7F7F] * k                                          # unit 0 saturates
        try:
            y16 = vu_output(p, a, b)
        except Reject:
            y16 = 0
        try:
            rows, erow = API.vu_rows(unit, epi, a, b, y16, cross_check=False)
            want.append(([[x % P for x in r] for r in rows], [x % P for x in erow]))
        except Reject as e:
            want.append(str(e))
        As.append(a)
        Bs.append(b)
        Ys.append(y16)
    ops = Ops()
    gen = Generator(ops)
    got = gen.run(ops.asarray(As), ops.asarray(Bs), ops.asarray(Ys))
    units = got.units.reshape(vus, steps, -1)
    report = {"vus": vus, "steps": steps, "match": 0, "rejected_both": 0, "mismatch": []}
    for v in range(vus):
        if isinstance(want[v], str):
            if bool(got.bad[v]):
                report["rejected_both"] += 1
            else:
                report["mismatch"].append({"vu": v, "reference": want[v], "generator": "accepted"})
            continue
        if bool(got.bad[v]):
            report["mismatch"].append({"vu": v, "reference": "accepted", "generator": "rejected"})
            continue
        rows, erow = want[v]
        du = np.argwhere(units[v] != np.array(rows, dtype=np.int64))
        de = np.argwhere(got.epilogue[v] != np.array(erow, dtype=np.int64))
        if len(du) or len(de):
            report["mismatch"].append({"vu": v, "unit_cells": du[:5].tolist(), "unit_cols": sorted({int(c) for c in du[:, 1]})[:10] if len(du) else [],
                                       "epilogue_cols": de.ravel().tolist()[:10]})
        else:
            report["match"] += 1
    report["ok"] = not report["mismatch"] and report["rejected_both"] >= 2 and report["match"] > 0
    return report


def compare(export_dir: str, root: str, tier: str, lo: int, hi: int | None, device: str, reps: int) -> dict:
    """The device generator vs an export directory's ``units.bin`` / ``epilogue.bin`` / ``public.bin``."""
    from pathlib import Path

    import torch
    from verity_numerical.bench import instances as I

    from ..bb_export import read_rows

    d = Path(export_dir)
    man = json.loads((d / "manifest.json").read_text())
    tierdata = I.load(tier, root)
    hi = tierdata.n if hi is None else hi
    Kw = tierdata.K
    x = np.frombuffer(tierdata.x, dtype=np.uint16).reshape(tierdata.n, Kw)[lo:hi]
    w = np.frombuffer(tierdata.w, dtype=np.uint16).reshape(tierdata.n, Kw)[lo:hi]
    y = np.frombuffer(tierdata.y, dtype=np.uint16)[lo:hi]
    t = time.perf_counter()
    ops = Ops(device)
    gen = Generator(ops)
    prep = time.perf_counter() - t
    dev = ops.device
    times = []
    for _ in range(reps):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        A, B, Y = ops.asarray(x), ops.asarray(w), ops.asarray(y)
        rows = gen.run(A, B, Y, sync=lambda: torch.cuda.synchronize(dev))
        torch.cuda.synchronize(dev)
        times.append({"total_with_h2d": time.perf_counter() - t0, **rows.seconds})
    want_u = read_rows(d / "units.bin").astype(np.int64)
    want_e = read_rows(d / "epilogue.bin").astype(np.int64)
    want_p = read_rows(d / "public.bin").astype(np.int64)[:, 0]
    written = man.get("written")
    bad = _to_np(rows.bad)
    ok_ids = [i for i in range(hi - lo) if not bad[i]]
    if written is not None and [lo + i for i in ok_ids] != list(written):
        return {"ok": False, "why": "the accepted VU set differs from the export's", "bad": np.flatnonzero(bad).tolist()[:20]}
    steps = Kw // STEP
    gu = _to_np(rows.units).reshape(hi - lo, steps, -1)[ok_ids].reshape(-1, want_u.shape[1])
    ge = _to_np(rows.epilogue)[ok_ids]
    gp = _to_np(rows.public)[ok_ids] % P
    du = np.argwhere(gu != want_u)
    de = np.argwhere(ge != want_e)
    return {"ok": not len(du) and not len(de) and bool((gp == want_p).all()), "vus": hi - lo, "accepted": len(ok_ids),
            "unit_mismatches": int(len(du)), "first_unit_mismatch": du[:3].tolist(), "epilogue_mismatches": int(len(de)),
            "public_equal": bool((gp == want_p).all()), "preprocessing_s": prep, "reps": times,
            "unit_shape": list(gu.shape), "gpu": torch.cuda.get_device_name(dev)}


def recipe(relation: str, vus: int, device: str, procs: int) -> dict:
    """The device generator at a relation's Params (``backends/direct/ligero/relations.py``) vs the export's rows
    (``BabyBearV2Api(epilogue, p).vu_rows``, the model-recorded accumulator cross-checked at every unit and the cast /
    packed word against the chain-end word; ``fp8.relation_params``) on the relation's first ``vus`` recipe instances (``relchain.instances``)."""
    import torch
    from backends.direct.ligero import relchain
    from backends.direct.ligero.relations import relation as get_relation

    from .export import BabyBearV2Api

    rel = get_relation(relation)
    p, epilogue = FP8.relation_params(rel)
    t = time.perf_counter()
    inst = relchain.instances(rel, vus, procs=procs)
    t_inst = time.perf_counter() - t
    x = np.array([v[0] for v in inst], dtype=np.int64)
    w = np.array([v[1] for v in inst], dtype=np.int64)
    y = np.array([rel.y_public(int(v[3])) for v in inst], dtype=np.int64)
    api = BabyBearV2Api(epilogue, p)
    unit, epi = api.build_unit(), api.build_epilogue()
    want_u, want_e, rejected = [], [], {}
    for v, (a, b, accs, _) in enumerate(inst):
        try:
            rows, erow = api.vu_rows(unit, epi, a, b, int(y[v]), accs)
        except Reject as e:
            rejected[v] = str(e)
            rows, erow = [], []
        want_u.append(np.array([[c % P for c in r] for r in rows], dtype=np.int64))
        want_e.append(np.array([c % P for c in erow], dtype=np.int64))
    ops = Ops(device)
    got = Generator(ops, p).run(ops.asarray(x), ops.asarray(w), ops.asarray(y))
    steps = x.shape[1] // sum(p.groups)
    gu = _to_np(got.units).reshape(vus, steps, -1)
    ge = _to_np(got.epilogue)
    bad = _to_np(got.bad)
    mism = [v for v in range(vus) if v not in rejected and (bad[v] or not np.array_equal(gu[v], want_u[v]) or not np.array_equal(ge[v], want_e[v]))]
    return {"ok": not mism and not rejected and not bad.any(), "relation": rel.name, "model": rel.model.name, "vus": vus,
            "rejected_by_export": rejected, "generator_bad": np.flatnonzero(bad).tolist()[:20], "mismatched_vus": mism[:20],
            "unit_columns": int(gu.shape[-1]), "epilogue_columns": int(ge.shape[-1]), "instances_s": t_inst,
            "gpu": torch.cuda.get_device_name(ops.device) if ops.torch is not None and ops.device.type == "cuda" else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpu.v2.witness")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--vus", type=int, default=6)
    s.add_argument("--steps", type=int, default=4)
    s.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("compare")
    c.add_argument("dir")
    c.add_argument("--root", required=True)
    c.add_argument("--tier", default="vu-k1536")
    c.add_argument("--lo", type=int, default=0)
    c.add_argument("--hi", type=int, default=None)
    c.add_argument("--device", default="cuda")
    c.add_argument("--reps", type=int, default=3)
    r_ = sub.add_parser("recipe")
    r_.add_argument("--relation", required=True, help="a backends/direct/ligero/relations.py name, e.g. bf16-hopper")
    r_.add_argument("--vus", type=int, default=64)
    r_.add_argument("--device", default="cuda")
    r_.add_argument("--procs", type=int, default=8)
    args = ap.parse_args(argv)
    if args.cmd == "selftest":
        r = selftest(args.vus, args.steps, args.seed)
    elif args.cmd == "recipe":
        r = recipe(args.relation, args.vus, args.device, args.procs)
    else:
        r = compare(args.dir, args.root, args.tier, args.lo, args.hi, args.device, args.reps)
    print(json.dumps(r, indent=1, default=str))
    return 0 if r.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
