"""`ncp-v2`'s committed values computed natively: every unit's outputs from numpy and hashlib, each hashed as it is made.

A call's values are its Program's (`verity.core.protocols.pouw.circuit.ncp2.linear`), and evaluating that gate by gate costs about 26
hours per Qwen2.5-0.5B forward at 16 rows. `Native` computes the same values unit by unit instead, and `NativeWeight` its
weight side's (`ncp2.weight`). They follow the partitions' numbering (`verity.core.protocols.pouw.circuit.partition.units`) and each
template's output-leaf order, and work from the inputs and the byte layouts `ncp2`'s docstring states:

    form_y   Y = [B + F₁ | B − F₁∘π⁻¹ | B + F₁ − F₁∘π⁻¹] (s8) for 16 columns, then c₀ = −β·Σ Y mod 2³² (the weight side's)
    digest   D_s = SHA-512 over the strip's row digests, as big-endian u32 words
    node     SHA-512 over its two children, in `ncp2.rfc6962`'s emission order
    key      K = SHAKE256(D_A, salt, forward, position, weight id)[:32], as little-endian u32 words
    form_x   X + β = [A + E₁ | A + E₁∘π | −(A + E₁ + E₁∘π)] + β (u8) for 16 rows
    tile     route U from c₀: every checkpoint, step-major then row-major, the last one Z (float64 is exact here: every
             partial sum is an integer below 2⁵³)
    dequant  Bf16Add(F32ToBf16Rn(F32Mul(F32Mul(s, t), I32ToF32Rn(Z))), bias), over arrays (`dequant`)

`NativeWeight.commit` commits a weight's side once, and `Native.commit` a whole call, both one block of weight rows at a
time, the blocks on a thread pool, reading those rows from the layer, and keep only each unit's leaf
(`pouw_circuit.unit_leaf`) and the served y. `outputs(u)` regenerates one unit for the replay's openings, and `args(u, ...)`
gives the unit's template arguments from the inputs and its producers' opened outputs.
"""
from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cache, cached_property
from typing import Callable, Mapping

import numpy as np
from verity.core.protocols.pouw.circuit import ncp2
from verity.core.protocols.pouw.circuit import reference as R
from verity.core.protocols.pouw.circuit.ncp2 import D, Shape

from verity.ml.kernels.tc import f32_to_bf16_rn_batch
from verity.ml.circuits.ops.scalar import F32_NAN_PTX
from verity_vllm.protocol_options import pouw_circuit as PC

M32 = 0xFFFFFFFF
BETA = 128
ROLES = ("digest", "node", "key", "form_x", "tile", "dequant")
#: weight rows read, formed and multiplied at a time (a multiple of 16), one block per task of the commit's thread pool
BLOCK = 256
#: the fewest rows a block is cut to so that every thread of the pool has one (`block_for`)
MIN_BLOCK = 64
#: the pool's threads, one per CPU this process may run on: numpy's products and sums and hashlib over a tile's words release the GIL
THREADS = len(os.sched_getaffinity(0))


def block_for(n: int) -> int:
    """Rows per task for a weight of ``n`` rows: `BLOCK`, or fewer, down to `MIN_BLOCK`, when `BLOCK` would leave threads of
    the pool without a block. The commit does not depend on it."""
    per = -(-n // THREADS)
    return max(MIN_BLOCK, min(BLOCK, -(-per // D) * D))


@cache
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=THREADS, thread_name_prefix="pouw-native")


def _u64(v: int) -> bytes:
    return (v & ((1 << 64) - 1)).to_bytes(8, "little")


def _index(i: int) -> bytes:
    return _u64((i << 32) | i) * 5


def _noise(xof: bytes) -> np.ndarray:
    return (np.frombuffer(xof, np.uint8) & 63).astype(np.int64) - 31


def _be(d: bytes) -> np.ndarray:
    return np.frombuffer(d, ">u4").astype(np.uint32)


def _le(d: bytes) -> np.ndarray:
    return np.frombuffer(d, "<u4").astype(np.uint32)


def _lane(v: int) -> np.ndarray:
    return np.array([v & M32, (v >> 32) & M32], np.uint32)


def _ints(v) -> list[int]:
    return [int(x) for x in np.asarray(v).reshape(-1)]


def _i7(v) -> list[int]:
    return [int(x) & 0x7F for x in np.asarray(v).reshape(-1)]


def _tampers(tamper: Mapping[tuple[int, int], int] | None) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for (u, j), v in (tamper or {}).items():
        out.setdefault(u, []).append((j, v))
    return out


def _tampered(tamper: dict[int, list[tuple[int, int]]], u: int, v: np.ndarray) -> np.ndarray:
    hit = tamper.get(u)
    if hit:
        v = np.array(v, np.uint32).reshape(-1)
        for j, x in hit:
            v[j] = x
    return v


# ---- dequantization over arrays ------------------------------------------------------------------------------------------

def _f32(u) -> np.ndarray:
    return np.asarray(u, np.uint32).view(np.float32)


def _is_nan(u: np.ndarray) -> np.ndarray:
    return ((u & 0x7F800000) == 0x7F800000) & ((u & 0x7FFFFF) != 0)


def _mul(a, b) -> np.ndarray:
    """`F32Mul_v2`: RN(a·b), every NaN `F32_NAN_PTX`."""
    r = np.asarray(_f32(a) * _f32(b), np.float32).view(np.uint32)
    return np.where(_is_nan(r), np.uint32(F32_NAN_PTX), r).astype(np.uint32)


def dequant(z, s, t, bias) -> np.ndarray:
    """`NcpDequant` elementwise (numpy broadcasting): Bf16Add_v1(F32ToBf16Rn_v1(F32Mul_v2(F32Mul_v2(s, t), I32ToF32Rn(z))),
    bias). Every step is IEEE binary32 in numpy, as the primitives' evaluators compute it, with their NaN words."""
    zf = np.asarray(z, np.uint32).view(np.int32).astype(np.float32).view(np.uint32)       # cvt.rn.f32.s32
    v = f32_to_bf16_rn_batch(_mul(_mul(s, t), zf))[0].astype(np.uint32)
    add = np.asarray(_f32(v << 16) + _f32(np.asarray(bias, np.uint32) << 16), np.float32).view(np.uint32)
    return f32_to_bf16_rn_batch(add)[0]


# ---- one weight's side ---------------------------------------------------------------------------------------------------

class NativeWeight:
    """One weight's side: its Y strips and c₀, `NcpWeight`'s units (one per 16-column strip), from its rows as read.
    ``tamper`` ((unit, output leaf) -> encoding) is a prover that commits and opens those values instead: the tests'
    negative controls."""

    def __init__(self, rows: PC.Rows, scheme: str, salt: bytes, weight: int, colids: np.ndarray,
                 tamper: Mapping[tuple[int, int], int] | None = None) -> None:
        self.rows, self.scheme, self.kind, self.salt, self.weight = rows, scheme, PC.kind_of(scheme), salt, weight
        self.colids = np.asarray(colids, np.uint32)
        self.n, self.k = rows.n, rows.k
        Shape(D, self.k, self.n).check()
        self.population = self.n // D
        self.inv = np.argsort(np.asarray(ncp2.perm(self.kind, self.k)))
        self.tamper = _tampers(tamper)
        self._strips: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def of(cls, side: PC.WeightRecord) -> "NativeWeight":
        """The side a record commits, regenerated from its inputs."""
        return cls(side.rows, side.scheme, side.salt, side.weight, side.colids)

    def template(self, u: int = 0):
        return ncp2.templates(self.k, self.kind)["form_y"]

    def read(self, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
        b, t = self.rows.rows(lo, hi)
        return np.ascontiguousarray(b, np.int8), np.ascontiguousarray(t, np.uint32)

    def block(self, b: np.ndarray, lo: int) -> tuple[np.ndarray, np.ndarray]:
        """Y (s8 bytes) and c₀ for the weight rows ``b``, the columns from ``lo``."""
        k = self.k
        tail = ncp2.TAG_F1 + self.salt + _u64(self.weight)
        f = np.stack([_noise(hashlib.shake_256(_index(int(j)) + tail).digest(k)) for j in self.colids[lo:lo + len(b)]])
        bb, fi = b.astype(np.int64), f[:, self.inv]
        y = (np.concatenate([bb + f, bb - fi, bb + f - fi], axis=1) & 0xFF).astype(np.uint8)
        c0 = ((-BETA * y.view(np.int8).astype(np.int64).sum(axis=1)) & M32).astype(np.uint32)
        return y, c0

    def strip(self, cs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Column strip ``cs``'s Y (16 × 3k bytes), c₀ and scales t, as the rows give them."""
        hit = self._strips.get(cs)
        if hit is None:
            b, t = self.read(D * cs, D * cs + D)
            hit = self._strips[cs] = (*self.block(b, D * cs), t)
        return hit

    def outputs(self, u: int) -> np.ndarray:
        """Y unit ``u``'s output leaves (its strip's Y, then its c₀), as this prover commits and opens them."""
        if not 0 <= u < self.population:
            raise IndexError(f"Y unit {u} outside 0..{self.population - 1}")
        y, c0, _t = self.strip(u)
        return _tampered(self.tamper, u, np.concatenate([y.reshape(-1).astype(np.uint32), c0]))

    def args(self, u: int) -> list:
        """Y unit ``u``'s template arguments, in parameter order: its 16 weight rows, the salt, the weight id, its column ids."""
        b, _t = self.read(D * u, D * u + D)
        return [_i7(b), _ints(np.frombuffer(self.salt, "<u4")), _ints(_lane(self.weight)), _ints(self.colids[D * u:D * u + D])]

    def commit(self, block: int | None = None) -> PC.WeightRecord:
        """Every Y unit's leaf, hashed as it is computed, the inputs' digest and what a call reads (`WeightRecord.reads`);
        the rows are read ``block`` at a time."""
        n = self.n
        block = max(D, (block or block_for(n)) // D * D)
        leaves: list[bytes] = [b""] * self.population

        def columns(lo: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            hi = min(n, lo + block)
            b, t = self.read(lo, hi)
            y, c0 = self.block(b, lo)
            for j in range(0, hi - lo, D):
                u = (lo + j) // D
                leaves[u] = PC.unit_leaf(_tampered(self.tamper, u, np.concatenate([y[j:j + D].reshape(-1), c0[j:j + D]])))
            return b, y, c0, t

        rd, read = PC.RowsDigest(n, self.k), PC.Read(n, self.k)
        for b, y, c0, t in _pool().map(columns, range(0, n, block)):     # in block order: B, Y, c₀ and t digested in order
            rd.update(b)
            read.update(y, c0, t)
        blob = b"".join(leaves)
        return PC.WeightRecord(self.scheme, self.rows, self.salt, self.weight, self.colids,
                               PC.weight_inputs_digest(rd.digest(), self.salt, self.weight, self.colids), blob,
                               PC.units_root(blob), self.outputs, read.parts())

    def reads(self, block: int | None = None) -> tuple[bytes, bytes, bytes]:
        """What a call on this weight reads, digested (`pouw_circuit.Read`): its Y, its c₀ and the scales t, regenerated."""
        n, rd = self.n, PC.Read(self.n, self.k)
        block = max(D, (block or BLOCK) // D * D)
        for lo in range(0, n, block):
            b, t = self.read(lo, min(n, lo + block))
            rd.update(*self.block(b, lo), t)
        return rd.parts()


def commit_weight(scheme: str, call: PC.Call, tamper: Mapping[tuple[int, int], int] | None = None) -> PC.WeightRecord:
    """The call's weight side committed natively (`NativeWeight.commit`), once per run and weight."""
    return NativeWeight(call.b, scheme, call.salt, call.weight, call.colids, tamper).commit()


# ---- one call ------------------------------------------------------------------------------------------------------------

class Native:
    """One call's committed values: the whole call hashed as it is computed (`commit`), or one unit on demand
    (`outputs`). It reads Y and c₀ from its weight side ``side`` (a `NativeWeight` over the call's rows when None).
    ``tamper`` ((unit, output leaf) -> encoding) is a prover that commits and opens those values instead: the tests'
    negative controls."""

    def __init__(self, call: PC.Call, scheme: str, tamper: Mapping[tuple[int, int], int] | None = None,
                 side: NativeWeight | None = None) -> None:
        self.call, self.scheme, self.kind = call, scheme, PC.kind_of(scheme)
        sh = self.shape = call.shape
        sh.check()
        self.side = side or NativeWeight(call.b, scheme, call.salt, call.weight, call.colids)
        self.a = np.ascontiguousarray(call.a, np.int8)
        self.ms, self.ns, self.steps = sh.m // D, sh.n // D, sh.steps
        self.pi = np.asarray(ncp2.perm(self.kind, sh.k))
        ms, ns = self.ms, self.ns
        self.base = {"digest": 0, "node": ms, "key": 2 * ms - 1, "form_x": 2 * ms, "tile": 3 * ms, "dequant": 3 * ms + ms * ns}
        self.population = self.base["dequant"] + ms * ns * D * D
        self.tamper = _tampers(tamper)
        self._xs: dict[int, np.ndarray] = {}

    # ---- the partition's numbering

    def index(self, role: str, *where: int) -> int:
        b = self.base[role]
        if role == "key":
            return b
        if role == "tile":
            return b + where[0] * self.ns + where[1]
        if role == "dequant":
            return b + (where[0] * self.ns + where[1]) * D * D + where[2]
        return b + where[0]

    def locate(self, u: int) -> tuple[str, tuple[int, ...]]:
        if not 0 <= u < self.population:
            raise IndexError(f"unit {u} outside 0..{self.population - 1}")
        role = next(r for r in reversed(ROLES) if self.base[r] <= u)
        off = u - self.base[role]
        if role == "key":
            return role, ()
        if role == "tile":
            return role, divmod(off, self.ns)
        if role == "dequant":
            t, e = divmod(off, D * D)
            return role, (*divmod(t, self.ns), e)
        return role, (off,)

    def template(self, u: int):
        return ncp2.templates(self.shape.k, self.kind)[self.locate(u)[0]]

    # ---- values

    @cached_property
    def digests(self) -> list[bytes]:
        out = []
        for s in range(self.ms):
            rows = b"".join(hashlib.sha512(ncp2.TAG_ROW + self.a[D * s + r].tobytes()).digest() for r in range(D))
            out.append(hashlib.sha512(ncp2.TAG_STRIP + rows).digest())
        return out

    @cached_property
    def tree(self) -> tuple[list, tuple[str, int], bytes]:
        """The tree's nodes in emission order, each ((left, right), digest) with a child ("digest", s) or ("node", j); the
        root's reference and D_A."""
        nodes: list = []

        def leaf(i: int):
            return ("digest", i), self.digests[i]

        def node(left, right):
            nodes.append(((left[0], right[0]), R.node(left[1], right[1])))
            return ("node", len(nodes) - 1), nodes[-1][1]

        ref, root = ncp2.rfc6962(0, self.ms, leaf, node)
        return nodes, ref, root

    @cached_property
    def key(self) -> bytes:
        return R.key(self.call, self.tree[2])

    def x_strip(self, rs: int) -> np.ndarray:
        """X + β for strip ``rs``: 16 rows of 3k bytes."""
        hit = self._xs.get(rs)
        if hit is not None:
            return hit
        k = self.shape.k
        e = np.stack([_noise(hashlib.shake_256(_index(int(i)) + ncp2.TAG_E1 + self.key).digest(k))
                      for i in self.call.rowids[D * rs:D * rs + D]])
        a = self.a[D * rs:D * rs + D].astype(np.int64)
        ep = e[:, self.pi]
        x = (np.concatenate([a + e + BETA, a + ep + BETA, BETA - (a + e + ep)], axis=1) & 0xFF).astype(np.uint8)
        self._xs[rs] = x
        return x

    def read(self, cs: int) -> tuple[np.ndarray, np.ndarray]:
        """The Y strip (flat, 16 × 3k bytes) and c₀ this call reads at column strip ``cs``: its inputs there."""
        y, c0, _t = self.side.strip(cs)
        return y.reshape(-1), c0

    def _x64(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float64).reshape(D, self.steps, D).transpose(1, 0, 2)                   # [t, r, d]

    def tiles(self, x64: np.ndarray, y: np.ndarray, c0: np.ndarray) -> np.ndarray:
        """Every checkpoint of a strip's tiles against the columns of ``y``: [t, r, c] as u32."""
        yv = y.view(np.int8).astype(np.float64).reshape(-1, self.steps, D).transpose(1, 2, 0)    # [t, d, c]
        run = np.cumsum(np.matmul(x64, yv), axis=0) + c0.astype(np.float64)
        return (run.astype(np.int64) & M32).astype(np.uint32)

    def outputs(self, u: int) -> np.ndarray:
        """Unit ``u``'s output leaves, regenerated from the call's inputs."""
        role, w = self.locate(u)
        return _tampered(self.tamper, u, self._outputs(role, w))

    def _outputs(self, role: str, w: tuple[int, ...]) -> np.ndarray:
        if role == "digest":
            return _be(self.digests[w[0]])
        if role == "node":
            return _be(self.tree[0][w[0]][1])
        if role == "key":
            return _le(self.key)
        if role == "form_x":
            return self.x_strip(w[0]).reshape(-1).astype(np.uint32)
        if role == "tile":
            y, c0, _t = self.side.strip(w[1])
            return self.tiles(self._x64(self.x_strip(w[0])), y, c0).reshape(-1)
        rs, cs, e = w
        y, c0, t = self.side.strip(cs)
        z = self.tiles(self._x64(self.x_strip(rs)), y, c0)[-1].reshape(-1)[e]
        r, c = divmod(e, D)
        return dequant(z, self.call.s[D * rs + r], t[c], self.call.bias[D * cs + c]).reshape(1).astype(np.uint32)

    # ---- wiring

    def producers(self, u: int) -> list[int]:
        """The call's units whose outputs unit ``u`` reads (a tile's Y and c₀ are the call's inputs, `read`)."""
        role, w = self.locate(u)
        if role == "node":
            (left, right), _d = self.tree[0][w[0]]
            return [self.index(*left), self.index(*right)]
        if role == "key":
            return [self.index(*self.tree[1])]
        if role == "form_x":
            return [self.index("key")]
        if role == "tile":
            return [self.index("form_x", w[0])]
        if role == "dequant":
            return [self.index("tile", w[0], w[1])]
        return []

    def args(self, u: int, opened: Callable[[int], np.ndarray], read: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None) -> list:
        """Unit ``u``'s template arguments, in parameter order: the call's inputs it reads (a tile's Y and c₀ as ``read``
        gives them, `read` when None), and its producers' outputs as ``opened`` gives them."""
        c, (role, w) = self.call, self.locate(u)
        salt, wid = _ints(np.frombuffer(c.salt, "<u4")), _ints(_lane(c.weight))
        if role == "digest":
            return [_i7(self.a[D * w[0]:D * w[0] + D])]
        if role == "node":
            left, right = self.producers(u)
            return [_ints(opened(left)), _ints(opened(right))]
        if role == "key":
            return [salt, _ints(_lane(c.forward)), _ints(_lane(c.position)), wid, _ints(opened(self.producers(u)[0]))]
        if role == "form_x":
            return [_i7(self.a[D * w[0]:D * w[0] + D]), _ints(opened(self.index("key"))), _ints(c.rowids[D * w[0]:D * w[0] + D])]
        if role == "tile":
            y, c0 = (read or self.read)(w[1])
            return [_ints(opened(self.producers(u)[0])), _ints(y), _ints(c0)]
        rs, cs, e = w
        r, cc = divmod(e, D)
        z = opened(self.producers(u)[0])[(self.steps - 1) * D * D + e]
        t = self.side.strip(cs)[2]
        return [int(z), int(c.s[D * rs + r]), int(t[cc]), int(c.bias[D * cs + cc])]

    # ---- the whole call

    def commit(self, index: int, side: PC.WeightRecord, block: int | None = None) -> PC.CallRecord:
        """Every unit's leaf, hashed as it is computed, and the served y, reading its weight side's Y and c₀; the weight's rows
        are read ``block`` at a time (default `block_for`). ``side`` is the weight side's record, committed once per run."""
        c, sh = self.call, self.shape
        m, k, n = sh.m, sh.k, sh.n
        block = max(D, (block or block_for(n)) // D * D)
        leaves: list[bytes] = [b""] * self.population

        def put(u: int, v: np.ndarray) -> None:
            leaves[u] = PC.unit_leaf(_tampered(self.tamper, u, v))

        for s in range(self.ms):
            put(self.index("digest", s), _be(self.digests[s]))
        for j, (_children, d) in enumerate(self.tree[0]):
            put(self.index("node", j), _be(d))
        put(self.index("key"), _le(self.key))
        xs = [self.x_strip(rs) for rs in range(self.ms)]
        for rs, x in enumerate(xs):
            put(self.index("form_x", rs), x.reshape(-1))
        x64 = [self._x64(x) for x in xs]
        z, y_out = np.empty((m, n), np.uint32), np.empty((m, n), np.uint16)

        def columns(lo: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            hi = min(n, lo + block)
            b, t = self.side.read(lo, hi)
            y, c0 = self.side.block(b, lo)
            for rs in range(self.ms):
                words = self.tiles(x64[rs], y, c0)
                for j in range(0, hi - lo, D):
                    put(self.index("tile", rs, (lo + j) // D), np.ascontiguousarray(words[:, :, j:j + D]))
                z[D * rs:D * rs + D, lo:hi] = words[-1]
            y_out[:, lo:hi] = dequant(z[:, lo:hi], c.s[:, None], t[None, :], c.bias[None, lo:hi])
            return y, c0, t

        rd = PC.Read(n, k)
        for y, c0, t in _pool().map(columns, range(0, n, block)):          # in block order: Y, c₀ and t digested column by column
            rd.update(y, c0, t)
        base = self.base["dequant"]
        deq = y_out.reshape(self.ms, D, self.ns, D).transpose(0, 2, 1, 3).reshape(-1).astype(np.uint32)
        for u, hit in self.tamper.items():
            for _j, v in (hit if u >= base else ()):
                deq[u - base] = v
        raw = deq.astype("<u4").tobytes()
        leaves[base:] = [hashlib.sha256(PC.UNIT_TAG + raw[i:i + 4]).digest() for i in range(0, len(raw), 4)]
        served = deq.astype(np.uint16).reshape(self.ms, self.ns, D, D).transpose(0, 2, 1, 3).reshape(m, n)
        self._xs.clear()
        blob = b"".join(leaves)
        return PC.CallRecord(index, self.scheme, c, PC.inputs_digest(c, rd.parts()), blob, PC.units_root(blob), served, side,
                             self.outputs, self.read)


def served(call: PC.Call, block: int | None = None) -> np.ndarray:
    """The call's served y alone, without its units: Z = A·Bᵀ exactly (the noise cancels in the circuit's Z; float64 is exact,
    every sum an integer below 2⁵³) and `dequant`, as `Native.commit` serves it. The weight's rows are read ``block`` at a
    time."""
    (m, _k), n = call.a.shape, call.b.n
    a, y, block = np.asarray(call.a, np.float64), np.empty((m, n), np.uint16), block or BLOCK
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        b, t = call.b.rows(lo, hi)
        z = ((a @ np.asarray(b, np.float64).T).astype(np.int64) & M32).astype(np.uint32)
        y[:, lo:hi] = dequant(z, call.s[:, None], np.asarray(t, np.uint32)[None, :], call.bias[None, lo:hi])
    return y


def commit_call(index: int, scheme: str, call: PC.Call, tamper: Mapping[tuple[int, int], int] | None = None,
                side: PC.WeightRecord | None = None) -> PC.CallRecord:
    """One call committed natively (`Native.commit`), reading its weight side ``side`` (committed here when None, as a run's
    first call on the weight does); its record opens units by regenerating them."""
    nw = NativeWeight(call.b, scheme, call.salt, call.weight, call.colids)
    return Native(call, scheme, tamper, nw).commit(index, side or nw.commit())


__all__ = ["BLOCK", "MIN_BLOCK", "Native", "NativeWeight", "THREADS", "block_for", "commit_call", "commit_weight", "dequant",
           "served"]
