"""PoUW's H1 + fx1 (`eps1-sm120-h1fx1`, `verity.core.protocols.pouw.circuit.h1tile`) on the eager int8_exact row: each
served int8 linear call as its own Program on bits, which `verity.core.protocols.pouw.circuit.compile` takes to H1 tile units.

* **The Program** (`program(k, n, m)`): one call's linear. Its A is what the Commit taps as the call's `Int8RowQuant_v1`
  (``/0``, the s8 rows x [m, k]; ``/1``, their f32 scales), its B the layer's int8 weight rows and their f32 scales
  (registered), and output (i, j) is `ScaledMmS8Coordinate_v2` on row i and weight row j, the served linear's
  ``RN_bf16(RN32(RN32(f32(Y) * s_a) * s_w))`` on its exact s32 Y. The coordinates are one batch over the (row, column) grid,
  so each rest commits one BF16 (the width rule). Rows and weight rows are padded to whole tiles (`TILE`, a warp's
  32 × 32, the tile the H1 kernel computes): pad row p reads one constant zero bit of its own for every entry, so it is a
  distinct row of the site, and its outputs are not served. Decode's one row, a partial last block and a column count off
  the tile all become full grids, which the H1 tile takes; a pad row is zero, so the units' z there is zero.
* **Groups.** A call is one site and one seed group (decision 122: a batched call over a step's batch is one site): its
  (root(A), root(B)) seeds its units. Calls are never grouped together, since a later call reads an earlier one's output.
* **Positions** (`H1Run`): the install's calls take consecutive ranges of `pouw.positions`, a running base, so no (salt,
  position) repeats under the epoch's salt; `Call.inputs` is `Compiled.inputs` at the call's base.
* **z** is the call's exact s32 Y (`engine.int8_exact.int_mm`, the product the served `scaled_mm` dequantizes): each unit's z
  is Y's cells of its tile, zero on pad rows and columns (`Call.z`).
* **Roots** are stand-ins (`ROOTS`): root(A) an hm96 row tree over the call's s8 rows, root(B) a SHA-512 row tree over the
  weight's int8 rows, until circuits binds `pouw.roots` to the Commit's own tap and weight roots.
* **Salt.** `pouw.salt` is a SHA-512 digest and the epoch's salt (`audit.Epoch.start`, from the challenger's salt coin) is 32
  bytes, so the Program's salt is the epoch's under `SALT_TAG` (`program_salt`) until the challenger opens a 64-byte one.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from verity.core.protocols.pouw.audit import HM96, activation_binding, commit_rows, weight_binding
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, Strided, concat
from verity.ml.circuits.types import Array

SCHEME = H.SCHEME
TILE = (32, 32)
#: the bits of one s8 entry and of one bf16 output
I8_BITS, BF16_BITS = 8, 16
ROOTS = ("stand-in: root(A) an hm96 row tree over the call's s8 rows (owner: the call's index), root(B) a SHA-512 row tree over the "
         "weight's int8 rows (owner -2), not the Commit's Int8RowQuant_v1 tap root and weight root (circuits: CMP-36, CMP-38)")
RECORD = "verity-vllm/pouw-h1-calls/v1"
CALLS_FILE = "pouw-h1-calls.json"
SALT_TAG = b"verity/pouw/h1/program-salt/v1\0"


def program_salt(epoch_salt: bytes) -> bytes:
    """`pouw.salt` of the install's Programs: the epoch's 32-byte salt under `SALT_TAG`, a SHA-512 digest."""
    return hashlib.sha512(SALT_TAG + epoch_salt).digest()


def coordinate(k: int) -> Any:
    return bind(BG.ScaledMmS8Coordinate, K=k, **BG.GEMM_SM120_S8)


def padded(m: int, n: int) -> tuple[int, int]:
    """The call's rows and weight rows padded to whole tiles."""
    r, c = TILE
    return -(-m // r) * r, -(-n // c) * c


def _signature(S: Any) -> tuple:
    (_x, a), (_sx, f), _w, _sw = coordinate(S.K).signature()[0]
    ret = coordinate(S.K).signature()[1]
    return (("x", Array(S.M, a)), ("sx", Array(S.M, f)), ("w", Array(S.N, a)), ("sw", Array(S.N, f))), Array(S.M, Array(S.N, ret))


def _repeat(r: Refs, n: int) -> Refs:
    """`r` read `n` times in a row."""
    if isinstance(r, Affine):
        return Strided(r.space, r.idx, r.base, r.count, 0, 1, r.count * n)
    if isinstance(r, Strided) and (r.inner, r.outer_stride, r.inner_stride) == (1, 0, 0):
        return Strided(r.space, r.idx, r.base, 1, 0, 0, r.count * n)
    return concat([r] * n)


def _body(B: Any, S: Any, x: Coll, sx: Coll, w: Coll, sw: Coll) -> Coll:
    mp, np_ = padded(S.M, S.N)
    L, F = x.type.elem.leaves, sx.type.elem.leaves
    pads = mp - S.M + np_ - S.N
    zero = B.batch(BL.constant(0), axes=(), n=pads).refs if pads else None

    def pad(p: int, count: int) -> Refs:
        return Strided(zero.space, zero.idx, zero.base + p, 1, 0, 0, count)

    rows = [x.refs.slice(i * L, (i + 1) * L) for i in range(S.M)] + [pad(p, L) for p in range(mp - S.M)]
    scales = [sx.refs.slice(i * F, (i + 1) * F) for i in range(S.M)] + [pad(p, F) for p in range(mp - S.M)]
    wrows = concat([w.refs, *(pad(mp - S.M + p, L) for p in range(np_ - S.N))])
    wscales = concat([sw.refs, *(pad(mp - S.M + p, F) for p in range(np_ - S.N))])
    cells = mp * np_
    grid = [Coll(Array(cells, x.type.elem), concat([_repeat(r, np_) for r in rows])),
            Coll(Array(cells, sx.type.elem), concat([_repeat(s, np_) for s in scales])),
            Coll(Array(cells, w.type.elem), _repeat(wrows, mp)),
            Coll(Array(cells, sw.type.elem), _repeat(wscales, mp))]
    out = B.batch(coordinate(S.K), *grid, axes=(0, 0, 0, 0))
    served = concat([out.refs.slice(i * np_ * BF16_BITS, (i * np_ + S.N) * BF16_BITS) for i in range(S.M)])
    return Coll(_signature(S)[1], served)


VllmInt8LinearH1 = CompositeDefinition(
    "VllmInt8LinearH1", 1, ("K", "N", "M"), _signature, _body, conformance="protocol-defined", register=False,
    doc="One served int8_exact linear call on bits: `ScaledMmS8Coordinate_v2` over the grid of its M s8 rows and N weight "
        "rows, both padded with constant zero rows to whole 32 x 32 tiles, serving the real cells.")


@lru_cache(maxsize=256)
def program(k: int, n: int, m: int) -> Program:
    """One call's served Program: x, sx bound per run, w, sw at registration."""
    return Program(bind(VllmInt8LinearH1, K=k, N=n, M=m), registered=("w", "sw"))


def check_tiles(c: C.Compiled) -> None:
    """Every unit a full `TILE` grid: what the served kernel's warp computes and its receipt covers."""
    bad = Counter((u.tile.r, u.tile.c, u.tile.grid) for u in c.units if (u.tile.r, u.tile.c) != TILE or not u.tile.grid)
    if bad:
        raise C.CompileError("scheme", f"{c.served.fn.id}: units off the {TILE[0]} x {TILE[1]} grid: {dict(bad)}")


@lru_cache(maxsize=256)
def compiled(k: int, n: int, m: int) -> C.Compiled:
    """`program(k, n, m)` compiled to `SCHEME`, a seed group per call, every unit a full tile."""
    c = C.compile(program(k, n, m), SCHEME)
    check_tiles(c)
    return c


def units(k: int, n: int, m: int) -> int:
    """The call's units, without compiling: its padded grid's tiles."""
    mp, np_ = padded(m, n)
    return (mp // TILE[0]) * (np_ // TILE[1])


def weight_root(w: int, rows: np.ndarray) -> bytes:
    """root(B)'s stand-in (`ROOTS`): the weight's int8 rows [n, k] in a SHA-512 row tree."""
    rows = np.ascontiguousarray(rows, dtype=np.int8)
    return commit_rows(weight_binding(w), -2, [r.tobytes() for r in rows], hash="sha512").commitment.root


def activation_root(salt: bytes, index: int, rows: np.ndarray, salt_key: bytes) -> bytes:
    """root(A)'s stand-in (`ROOTS`): the call's s8 rows [m, k] in an hm96 row tree under the prover's salt key."""
    rows = np.ascontiguousarray(rows, dtype=np.int8)
    return commit_rows(activation_binding(salt, index), index, [r.tobytes() for r in rows], hash=HM96, salt_key=salt_key,
                       nonce=index).commitment.root


@dataclass(frozen=True)
class Call:
    """One served int8 linear call: its index in the install, its layer (the linear's ordinal) and module, its shape, its
    first position, and its seed group's (root(A), root(B))."""

    index: int
    layer: int
    module: str
    m: int
    k: int
    n: int
    base: int
    roots: tuple[bytes, bytes]

    @property
    def units(self) -> int:
        return units(self.k, self.n, self.m)

    def compiled(self) -> C.Compiled:
        return compiled(self.k, self.n, self.m)

    def inputs(self, salt: bytes) -> list[list[int]]:
        """`pouw.salt`, `pouw.positions` and `pouw.roots`' leaves for this call (`Compiled.inputs` at its base)."""
        return self.compiled().inputs(salt, [self.roots], self.base)

    def z(self, y: np.ndarray, unit: int) -> np.ndarray:
        """Unit `unit`'s z, in its cells' order, from the call's exact s32 Y [m, n] (pad cells zero)."""
        u = self.compiled().units[unit]
        full = np.zeros(padded(self.m, self.n), dtype=np.int32)
        full[:self.m, :self.n] = y
        return np.array([full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs], dtype=np.int32)

    def summary(self) -> dict[str, Any]:
        return {"index": self.index, "layer": self.layer, "module": self.module, "shape": [self.m, self.k, self.n],
                "padded": list(padded(self.m, self.n)), "base": self.base, "units": self.units,
                "root_a": self.roots[0].hex(), "root_b": self.roots[1].hex()}


class H1Run:
    """One install's H1 workload on the int8_exact row: every served linear call recorded as a `Call`, its positions a range
    after the previous call's. ``retain`` keeps the last calls' operands and exact Y for the prover (0: none). Without
    ``commit`` (a Match capture) the arm serves and nothing is recorded."""

    def __init__(self, entry: str, salt: bytes, weight_roots: Mapping[int, bytes], modules: Mapping[int, str],
                 retain: int = 0, transcript_dir: str | None = None, commit: bool = True) -> None:
        self.entry, self.salt, self.weight_roots, self.modules = entry, salt, dict(weight_roots), dict(modules)
        self.commit = commit
        self.salt_key = os.urandom(32)
        self.calls: list[Call] = []
        self.retained: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=retain or None)
        self.retain, self.transcript_dir = retain, transcript_dir
        self.next_base = 0
        self.commits: dict[str, Any] = {"scheme": SCHEME, "tile": list(TILE), "calls": 0, "units": 0, "positions": 0,
                                        "roots": ROOTS}
        self.details: dict[str, Any] = {"scheme": SCHEME, "calls": 0, "padded_rows": 0, "shapes": {},
                                        "uncovered": "attention (QK^T and P.V in Int8LutAttentionRow_v1) and lm_head",
                                        "z": "engine.int8_exact.int_mm: the exact s32 Y, computed beside the served scaled_mm"}

    def record(self, w: int, xq: np.ndarray, sa: np.ndarray, n: int, y: np.ndarray | None = None) -> Call:
        """Call `w`'s s8 rows [m, k], their f32 scales [m], its weight's n rows and, when kept, its exact Y [m, n]."""
        m, k = xq.shape
        index = len(self.calls)
        ra = activation_root(self.salt, index, xq, self.salt_key)
        call = Call(index, w, self.modules.get(w, str(w)), m, k, n, self.next_base, (ra, self.weight_roots[w]))
        self.calls.append(call)
        self.next_base += call.units
        if self.retain:
            self.retained.append((index, xq, sa, y if y is not None else np.zeros((0, 0), np.int32)))
        shape = f"{m}x{k}x{n}"
        self.details["shapes"][shape] = self.details["shapes"].get(shape, 0) + 1
        self.details.update(calls=len(self.calls), padded_rows=self.details["padded_rows"] + padded(m, n)[0] - m)
        self.commits.update(calls=len(self.calls), units=self.next_base, positions=self.next_base)
        return call

    def shapes(self) -> list[tuple[int, int, int]]:
        return sorted({(c.k, c.n, c.m) for c in self.calls})

    def check(self) -> dict[str, Any]:
        """Compile every served shape to `SCHEME` and hold each call's units to its compiled Program's: per shape (k, n, m) its
        Program digest, units and tile shapes."""
        out = {}
        for k, n, m in self.shapes():
            c = compiled(k, n, m)
            if len(c.units) != units(k, n, m) or c.groups != ((0,),):
                raise C.CompileError("position", f"{k} x {n} at m = {m}: {len(c.units)} units in {len(c.groups)} groups, not "
                                                 f"{units(k, n, m)} in one")
            out[f"{m}x{k}x{n}"] = {"digest": c.digest, "units": len(c.units), "groups": [list(g) for g in c.groups],
                                   "cover": list(c.cover), "tiles": sorted({f"{u.tile.r}x{u.tile.c}" for u in c.units})}
        return out

    def calls_root(self) -> bytes:
        """SHA-512 over the record's calls, in order: what the install commits to (its positions and roots)."""
        h = hashlib.sha512(b"verity/pouw/h1/calls/v1\0" + self.salt)
        for c in self.calls:
            h.update(json.dumps(c.summary(), sort_keys=True).encode() + b"\n")
        return h.digest()

    def finish(self) -> None:
        self.commits["calls_root"] = self.calls_root().hex()
        if self.transcript_dir:
            d = Path(self.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / CALLS_FILE).write_text(json.dumps({"schema": RECORD, "entry": self.entry, "commitment": self.commits,
                                                    "calls": [c.summary() for c in self.calls]}) + "\n")


def layers(model: Any, types: tuple[type, ...]) -> list[Any]:
    """The linear layers by type in module order; ValueError names every one not on the int8_exact plain arm (`engine.int8_vllm`):
    H1 reads the s8 rows that arm's quantizer serves, and its int8 weight."""
    out = [m for m in model.modules() if isinstance(m, types)]
    if not out:
        raise ValueError(f"pouw:{SCHEME}: the model has no linear layer")
    bad = []
    for w, layer in enumerate(out):
        arm = getattr(layer.quant_method, "int8_arm", None)
        dtype = str(getattr(getattr(layer, "weight", None), "dtype", None)).removeprefix("torch.")
        if arm is None or dtype != "int8" or getattr(layer, "weight_scale", None) is None:
            bad.append(f"linear {w}: {type(layer.quant_method).__name__} with a {dtype} weight is not the int8_exact plain arm")
    if bad:
        raise ValueError(f"pouw:{SCHEME} refused at install: " + "; ".join(bad))
    return out


def numpy_of(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def served(run: H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``apply``: the arm's own (it serves y, and the Commit's taps see its ops), then the call recorded from its s8
    rows (`engine.int8_exact.quantize_rows`, the arm's quantizer, on the same activations) and, when kept, its exact Y
    (`engine.int8_exact.int_mm`)."""
    from verity_vllm.engine.int8_exact import int_mm, quantize_rows

    def apply(layer: Any, x: Any, bias: Any = None) -> Any:
        y = orig(layer, x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        if run.commit and x2.shape[0]:
            xq, sa = quantize_rows(x2)
            z = numpy_of(int_mm(xq, layer.weight)) if run.retain else None
            run.record(w, numpy_of(xq), numpy_of(sa), int(layer.weight.shape[0]), z)
        return y
    return apply


__all__ = ["CALLS_FILE", "Call", "H1Run", "RECORD", "ROOTS", "SALT_TAG", "SCHEME", "TILE", "VllmInt8LinearH1", "activation_root",
           "check_tiles", "compiled", "coordinate", "layers", "numpy_of", "padded", "program", "program_salt", "served", "units", "weight_root"]
