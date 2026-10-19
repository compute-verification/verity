"""PoUW's H1 + fx1 (`eps1-sm120-h1fx1`, `verity.core.protocols.pouw.circuit.h1tile`) on the eager int8_exact row: each
served int8 linear call as its own Program on bits, which `verity.core.protocols.pouw.circuit.compile` takes to H1 tile units.

* **The Program** (`program(k, n, m)`): one call's linear. Its A is the call's A table (x [m, k] s8 rows, sx [m] f32 scales:
  the served rows as the Commit taps `Int8RowQuant_v1` /0 and /1, then empty rows), its B the layer's int8 weight rows and
  their f32 scales (registered), and output (i, j) is `ScaledMmS8Coordinate_v2` on row i and weight row j, the served
  linear's ``RN_bf16(RN32(RN32(f32(Y) * s_a) * s_w))`` on its exact s32 Y. The coordinates are one batch over the (row,
  column) grid, so each rest commits one BF16 (the width rule). Rows and weight rows off a whole tile (`TILE`, a warp's
  32 × 32, the tile the H1 kernel computes) are padded: pad row p reads constant zero bits of its own, contiguous, so it is a
  distinct row of the site, and its outputs are not served.
* **The static layout** (`H1Run`, Daniel's static-circuit rule): every call has the same T rows (``rows``, vLLM's
  ``max_num_batched_tokens``, a multiple of 32), so its Program is `program(k, n, T)` whatever the step served. The step's m
  served rows fill slots 0 .. m - 1 and every other slot is an empty row, zero bytes under scale 0 (`slots`). The window is S
  steps (``steps``) of one call per linear: call (s, w) is at positions s·U + off(w) + u (`layout`: U is a step's units,
  off(w) the units of the linears before w), and steps past the last served one are all empty. Nothing about a call's rows or
  positions comes from its tokens, batch or EOS. Credit is the served rows of the calls the credit gate passes only; the
  padded work is shown beside it.
* **Groups.** A call is one site and one seed group (decision 122: a batched call over a step's batch is one site): its
  (root(A), root(B)) seeds its units. Calls are never grouped together, since a later call reads an earlier one's output.
* **Units** are in compile's order, the order the manifest lists them in: the call, then its row block, then its column
  block, row-major.
* **The hook.** The arm hands every served call to `engine.int8_vllm.Int8PlainLinearMethod.observe` after both ops, and the
  option records it there as a `Call`: its engine step (counted at linear 0), its layer (the linear's ordinal), its served
  rows and its position. Then each listener (`H1Run.listen`) gets the call with the served tensors, on the device: A (the s8
  rows, as the Commit taps ``Int8RowQuant_v1/0``), s_a (``/1``), the int8 weight and its scales; `H1Run.close` hands it the
  empty steps' calls too, with no rows. A's slot i is leaf i of the Commit's row table of (step, layer) (circuits' AROW), so
  a tile's rows are 32 consecutive leaves. What a listener returns (its roots and receipt root, as hex) is recorded with
  the call (``bound``). `Call.inputs` is `Compiled.inputs` at the call's base, with the roots it is given: root(A) is the
  table's, and root(B) is the weight's, which compute accounting commits. This module commits neither.
* **z** is the call's exact s32 Y: the arm's own when its ``observe`` hands one over (a fused kernel's), else
  `engine.int8_exact.int_mm` beside the served `scaled_mm` (computed only to retain it): each unit's z is Y's cells of its
  tile, zero on empty slots and pad columns (`Call.z`). The run keeps the last ``retain`` calls' Y.
* **The credit gate** (`credit_gate`, compute accounting's credit rule): each served call's ``apply`` runs under a trace
  (`traced`), and the call is credited only if the y it hands back to vLLM was written by the GEMM invocation that wrote
  its z, the only GEMM of the call. Today's arm serves `scaled_mm`'s y with z a second ``_int_mm`` beside it, so every
  call is refused (``credit: served-output-not-credited-kernel: ...``) and the window credits nothing; a call recorded
  outside a traced ``apply`` is refused too (`UNTRACED`).
* **Salt.** `pouw.salt` is a SHA-512 digest and the epoch's salt (`audit.Epoch.start`, from the challenger's salt coin, opened
  after the weights' root, `weight_root`) is 32 bytes, so the Program's salt is the epoch's under `SALT_TAG`
  (`program_salt`) until the challenger opens a 64-byte one.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from verity.core.protocols.pouw.audit import commit_rows, weight_binding
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, Strided, concat
from verity.ml.circuits.types import Array

from verity_vllm.protocol_options import credit_gate as CG

SCHEME = H.SCHEME
TILE = (32, 32)
#: the bits of one s8 entry and of one bf16 output
I8_BITS, BF16_BITS = 8, 16
RECORD = "verity-vllm/pouw-h1-calls/v2"
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
    zero = B.batch(BL.constant(0), axes=(), n=pads * (L + F)).refs if pads else None

    def pad(p: int, scale: bool) -> Refs:
        # contiguous leaves of its own, not one leaf repeated: compile keys a row by its runs, one per repeat
        return zero.slice(p * (L + F) + (L if scale else 0), p * (L + F) + (L + F if scale else L))

    rows = [x.refs.slice(i * L, (i + 1) * L) for i in range(S.M)] + [pad(p, False) for p in range(mp - S.M)]
    scales = [sx.refs.slice(i * F, (i + 1) * F) for i in range(S.M)] + [pad(p, True) for p in range(mp - S.M)]
    wrows = concat([w.refs, *(pad(mp - S.M + p, False) for p in range(np_ - S.N))])
    wscales = concat([sw.refs, *(pad(mp - S.M + p, True) for p in range(np_ - S.N))])
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
    """The epoch's commitment to weight `w` (its int8 rows [n, k] in a SHA-512 row tree), which the salt coin is opened after
    (`audit.weights_root`); not root(B), which is compute accounting's."""
    rows = np.ascontiguousarray(rows, dtype=np.int8)
    return commit_rows(weight_binding(w), -2, [r.tobytes() for r in rows], hash="sha512").commitment.root


def layout(shapes: Sequence[tuple[int, int]], rows: int) -> tuple[list[int], int]:
    """off(w) of each linear (its (n, k), in order) and U, one step's units: linear w's call is its `units` at `rows`."""
    if rows % TILE[0] or rows <= 0:
        raise ValueError(f"T = {rows} rows is not a positive multiple of {TILE[0]}")
    offs, at = [], 0
    for n, k in shapes:
        offs.append(at)
        at += units(k, n, rows)
    return offs, at


def slots(a: Any, s_a: Any, rows: int) -> tuple[Any, Any]:
    """The call's A table of `rows` slots, on a's device: its m served rows and scales first, then empty rows, zero bytes
    under scale 0 (a served row's scale is never 0)."""
    m = int(a.shape[0])
    if m > rows:
        raise ValueError(f"{m} rows past the call's {rows} slots")
    out, sc = a.new_zeros((rows, int(a.shape[1]))), s_a.reshape(-1).new_zeros(rows)
    out[:m], sc[:m] = a, s_a.reshape(-1)
    return out, sc


@dataclass(frozen=True)
class Call:
    """One (step, linear) call of the window: its index, its engine step, its layer (the linear's ordinal) and module, its
    served rows m (0 for a step past the last served one), its `rows` slots (T), k, n and its first position. Its A table is
    the Commit's row table of (step, layer): leaf i is slot i, served for i < m and empty after."""

    index: int
    step: int
    layer: int
    module: str
    m: int
    k: int
    n: int
    base: int
    rows: int

    @property
    def units(self) -> int:
        return units(self.k, self.n, self.rows)

    def compiled(self) -> C.Compiled:
        return compiled(self.k, self.n, self.rows)

    def inputs(self, salt: bytes, roots: tuple[bytes, bytes]) -> list[list[int]]:
        """`pouw.salt`, `pouw.positions` and `pouw.roots`' leaves for this call, its seed group's (root(A), root(B)) given
        (`Compiled.inputs` at its base)."""
        return self.compiled().inputs(salt, [roots], self.base)

    def z(self, y: np.ndarray, unit: int) -> np.ndarray:
        """Unit `unit`'s z, in its cells' order, from the call's exact s32 Y [m, n] (empty slots and pad columns zero)."""
        u = self.compiled().units[unit]
        full = np.zeros(padded(self.rows, self.n), dtype=np.int32)
        full[:self.m, :self.n] = y
        return np.array([full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs], dtype=np.int32)

    def summary(self) -> dict[str, Any]:
        return {"index": self.index, "step": self.step, "layer": self.layer, "module": self.module, "served": self.m,
                "rows": self.rows, "k": self.k, "n": self.n, "base": self.base, "units": self.units, "leaves": [0, self.rows]}


#: ``listener(call, a, s_a, weight, weight_scale)``: the served tensors on the device, which it must not change (a [m, k]
#: and s_a [m], m = 0 for an empty step; `slots` makes the call's T-slot table); it returns None or what it bound for the
#: call, as JSON (its roots and receipt root, hex)
Listener = Callable[[Call, Any, Any, Any, Any], Mapping[str, Any] | None]


def numpy_of(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


class H1Run:
    """One install's H1 window on the int8_exact row, in the static form: its calls, rows and positions are a function of the
    public parameters alone, never of a step's tokens, batch or EOS. ``shapes`` are the linears' (n, k) in order, ``rows``
    (T) every call's slots and ``steps`` (S) the window's engine steps. Call (s, w) is at s·U + off(w) (`layout`); served
    calls are recorded as they come (`record`), and `close` adds the steps past the last served one, all empty. A call past
    step S or with more than T rows is refused: recorded in ``refused``, never positioned, and `static` is then False.
    ``retain`` keeps the last calls' A, s_a and exact Y on the host (0: none). Without ``commit`` (a Match capture) nothing
    is recorded."""

    def __init__(self, entry: str, salt: bytes, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]], rows: int,
                 steps: int, retain: int = 0, transcript_dir: str | None = None, commit: bool = True) -> None:
        self.entry, self.salt, self.modules, self.commit = entry, salt, dict(modules), commit
        self.shapes_, self.rows, self.steps = [tuple(map(int, s)) for s in shapes], int(rows), int(steps)
        self.offs, self.u_step = layout(self.shapes_, self.rows)
        self.calls: list[Call] = []
        self.listeners: list[Listener] = []
        self.bound: dict[int, Mapping[str, Any]] = {}
        self.refused: list[dict[str, int]] = []
        self.retained: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=retain or None)
        self.retain, self.transcript_dir = retain, transcript_dir
        self.step, self.last, self.order_ok, self.closed = -1, None, True, False
        self._weights: dict[int, tuple[Any, Any]] = {}
        #: each served call's credit verdict (`credit_gate.verdict`), by call index, and the call `record` just made with its z
        self.verdicts: dict[int, dict[str, Any]] = {}
        self._recorded: tuple[Call, Any] | None = None
        self.commits: dict[str, Any] = {"scheme": SCHEME, "tile": list(TILE), "rows": self.rows, "steps": self.steps,
                                        "u_step": self.u_step, "units": self.steps * self.u_step, "calls": 0}
        self.details: dict[str, Any] = {"scheme": SCHEME, "calls": 0, "refused": self.refused,
                                        "uncovered": "attention (QK^T and P.V in Int8LutAttentionRow_v1) and lm_head",
                                        "z": "the arm's own when its observe hands one over, else engine.int8_exact.int_mm "
                                             "beside the served scaled_mm"}

    def listen(self, fn: Listener) -> None:
        """Hand every later call to `fn` (`Listener`)."""
        self.listeners.append(fn)

    def _call(self, s: int, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any) -> Call:
        n, k = self.shapes_[w]
        call = Call(len(self.calls), s, w, self.modules.get(w, str(w)), int(a.shape[0]) if a is not None else 0, k, n,
                    s * self.u_step + self.offs[w], self.rows)
        self.calls.append(call)
        for fn in self.listeners:
            out = fn(call, a, s_a, weight, weight_scale)
            if out is not None:
                self.bound[call.index] = {**self.bound.get(call.index, {}), **out}
        self.details["calls"] = self.commits["calls"] = len(self.calls)
        return call

    def record(self, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, z: Any = None) -> Call | None:
        """Linear `w`'s served call: its s8 rows a [m, k], their f32 scales s_a [m], its int8 weight [n, k] and scales, and
        its exact s32 Y [m, n] if the arm's kernel wrote one (else `int_mm` computes it beside, only to retain it)."""
        if self.closed:
            raise RuntimeError(f"{self.entry}: a call after the window closed")
        if w == 0:
            self.step += 1
        elif self.last is None or w != self.last + 1:
            self.order_ok = False
        self.last = w
        self._weights[w] = (weight, weight_scale)
        m, k = (int(d) for d in a.shape)
        if not 0 <= self.step < self.steps or m > self.rows or (int(weight.shape[0]), k) != self.shapes_[w]:
            self.refused.append({"step": self.step, "linear": w, "m": m})
            return None
        call = self._call(self.step, w, a, s_a, weight, weight_scale)
        if z is None and self.retain:
            from verity_vllm.engine.int8_exact import int_mm
            z = int_mm(a, weight)
        if self.retain:
            self.retained.append((call.index, numpy_of(a), numpy_of(s_a), numpy_of(z)))
        self._recorded = (call, z)
        return call

    def judge(self, trace: CG.Trace, y: Any) -> dict[str, Any] | None:
        """The credit gate on the call `record` made inside ``trace`` (None if it made none): y is what the linear handed
        back to vLLM."""
        made, self._recorded = self._recorded, None
        if made is None:
            return None
        call, z = made
        v = self.verdicts[call.index] = CG.verdict(trace, y, z)
        return v

    def close(self) -> None:
        """The steps past the last served one, each linear's call empty (m = 0), handed to the listeners like the rest."""
        if self.closed:
            return
        self.closed = True
        if not self.commit:
            return
        self.details["steps_served"] = self.step + 1
        self.details["last_step_complete"] = self.last in (None, len(self.shapes_) - 1)
        for s in range(self.step + 1, self.steps):
            for w, (n, k) in enumerate(self.shapes_):
                weight, scale = self._weights.get(w, (None, None))
                a = weight.new_zeros((0, k)) if weight is not None else None
                self._call(s, w, a, scale.reshape(-1).new_zeros(0) if scale is not None else None, weight, scale)
        self.details["static"] = self.static()
        self.details["credit"] = self.credit()

    def static(self) -> bool:
        """Every (step, linear) call of the window, in order, at its public position, and nothing refused."""
        want = [(s, w, s * self.u_step + self.offs[w], units(k, n, self.rows))
                for s in range(self.steps) for w, (n, k) in enumerate(self.shapes_)]
        return self.order_ok and not self.refused and [(c.step, c.layer, c.base, c.units) for c in self.calls] == want

    def verdict(self, call: Call) -> dict[str, Any]:
        """The call's credit verdict; a served call recorded outside a traced ``apply`` (`traced`) is refused."""
        return self.verdicts.get(call.index) or {
            "linear": call.module, "credit": False,
            "reason": f"credit: {CG.UNTRACED[1:-1]}: {call.module} call {call.index} was recorded outside a traced apply"}

    def credit(self) -> dict[str, Any]:
        """Credited: the served rows' MACs of the calls the credit gate passes (`credit_gate`), only. The padded work (empty
        slots and pad columns) is shown beside it, not credited (Daniel, 18 Oct), and so are the refused calls' rows, with
        each refusal's reason."""
        served = [c for c in self.calls if c.m]
        verdicts = {c.index: self.verdict(c) for c in served}
        ok = [c for c in served if verdicts[c.index]["credit"]]
        no = [c for c in served if not verdicts[c.index]["credit"]]
        computed = sum(padded(c.rows, c.n)[0] * padded(c.rows, c.n)[1] * c.k for c in self.calls)
        all_served = sum(c.m * c.n * c.k for c in served)
        return {"rule": "credit the served rows of calls whose served y is the credited kernel's own output only; padded "
                        "work and refused calls are shown, not credited",
                "credited": {"rows": sum(c.m for c in ok), "macs": sum(c.m * c.n * c.k for c in ok)},
                "refused": {"calls": len(no), "rows": sum(c.m for c in no), "macs": sum(c.m * c.n * c.k for c in no),
                            "reasons": CG.reasons(verdicts[c.index] for c in no)},
                "padded": {"rows": sum(c.rows - c.m for c in self.calls), "macs": computed - all_served},
                "units": sum(c.units for c in self.calls),
                "served_frac_of_macs": round(all_served / computed, 6) if computed else 0.0}

    def shapes(self) -> list[tuple[int, int, int]]:
        return sorted({(c.k, c.n, c.rows) for c in self.calls})

    def check(self) -> dict[str, Any]:
        """Compile every call shape to `SCHEME` and hold each call's units to its compiled Program's: per shape (T, k, n) its
        Program digest, units, seed groups and tile shapes."""
        out = {}
        for k, n, t in self.shapes():
            c = compiled(k, n, t)
            if len(c.units) != units(k, n, t) or c.groups != ((0,),):
                raise C.CompileError("position", f"{k} x {n} at T = {t}: {len(c.units)} units in {len(c.groups)} groups, not "
                                                 f"{units(k, n, t)} in one")
            out[f"{t}x{k}x{n}"] = {"digest": c.digest, "units": len(c.units), "groups": [list(g) for g in c.groups],
                                   "cover": list(c.cover), "tiles": sorted({f"{u.tile.r}x{u.tile.c}" for u in c.units})}
        return out

    def manifest(self) -> list[dict[str, Any]]:
        """Each call's summary, with what its listeners bound."""
        return [{**c.summary(), **({"bound": dict(self.bound[c.index])} if c.index in self.bound else {})} for c in self.calls]

    def calls_root(self) -> bytes:
        """SHA-512 over the manifest, in order: what the install commits to (its positions and what was bound)."""
        h = hashlib.sha512(b"verity/pouw/h1/calls/v2\0" + self.salt)
        for row in self.manifest():
            h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
        return h.digest()

    def finish(self) -> None:
        self.close()
        if not self.commit:
            return
        self.commits.update(calls_root=self.calls_root().hex(), static=self.details["static"])
        if self.transcript_dir:
            d = Path(self.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / CALLS_FILE).write_text(json.dumps({"schema": RECORD, "entry": self.entry, "commitment": self.commits,
                                                    "public": {"rows": self.rows, "steps": self.steps,
                                                               "shapes": [list(s) for s in self.shapes_]},
                                                    "details": {k: v for k, v in self.details.items() if k != "refused"},
                                                    "refused": self.refused, "calls": self.manifest()}) + "\n")


def layers(model: Any, types: tuple[type, ...]) -> list[Any]:
    """The linear layers by type in module order; ValueError names every one not on the int8_exact plain arm
    (`engine.int8_vllm.Int8PlainLinearMethod`, whose ``observe`` the option wraps) with its int8 weight and scales."""
    out = [m for m in model.modules() if isinstance(m, types)]
    if not out:
        raise ValueError(f"pouw:{SCHEME}: the model has no linear layer")
    bad = []
    for w, layer in enumerate(out):
        method = layer.quant_method
        dtype = str(getattr(getattr(layer, "weight", None), "dtype", None)).removeprefix("torch.")
        if getattr(method, "int8_arm", None) is None or not callable(getattr(method, "observe", None)) or dtype != "int8" \
                or getattr(layer, "weight_scale", None) is None:
            bad.append(f"linear {w}: {type(method).__name__} with a {dtype} weight is not the int8_exact plain arm")
    if bad:
        raise ValueError(f"pouw:{SCHEME} refused at install: " + "; ".join(bad))
    return out


def observer(run: H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``observe``: the arm's own, then the call recorded with the layer's weight and the z the arm's kernel
    wrote, if it hands one over (`H1Run.record`)."""

    def observe(layer: Any, xq: Any, s_a: Any, y: Any, z: Any = None) -> None:
        orig(layer, xq, s_a, y) if z is None else orig(layer, xq, s_a, y, z)
        if run.commit:
            run.record(w, xq, s_a, layer.weight, layer.weight_scale, z)
    return observe


def traced(run: H1Run, w: int, orig: Any) -> Any:
    """Layer `w`'s ``apply``: the arm's own under the credit gate's trace (`credit_gate.Trace`), then the call `record` made
    inside it judged against the y it returns (`H1Run.judge`)."""

    def apply(layer: Any, x: Any, bias: Any = None) -> Any:
        if not run.commit:
            return orig(layer, x, bias)
        run._recorded = None
        with CG.Trace(run.modules.get(w, f"linear {w}")) as trace:
            y = orig(layer, x, bias)
        run.judge(trace, y)
        return y
    return apply


def bind_all(calls: Sequence[Call], bound: Mapping[int, Mapping[str, Any]]) -> list[tuple[bytes, bytes] | None]:
    """Each call's (root(A), root(B)) from what its listeners bound (``root_a``, ``root_b``, hex), None where either is missing."""
    out: list[tuple[bytes, bytes] | None] = []
    for c in calls:
        b = bound.get(c.index, {})
        out.append((bytes.fromhex(b["root_a"]), bytes.fromhex(b["root_b"])) if "root_a" in b and "root_b" in b else None)
    return out


__all__ = ["CALLS_FILE", "Call", "H1Run", "Listener", "RECORD", "SALT_TAG", "SCHEME", "TILE", "VllmInt8LinearH1", "bind_all",
           "check_tiles", "compiled", "coordinate", "layers", "layout", "numpy_of", "observer", "padded", "program", "program_salt",
           "slots", "traced", "units", "weight_root"]
