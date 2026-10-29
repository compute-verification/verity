"""PoUW's H1 + fx1 (`eps1-sm120-h1fx1`, `verity.core.protocols.pouw.circuit.h1tile`) on the eager int8_exact row: each
served int8 linear call as its own Program on bits, which `verity.core.protocols.pouw.circuit.compile` takes to H1 tile units.

* **The Program** (`program(k, n, m)`): one call's linear. Its A is the call's A table (x [m, k] s8 rows, sx [m] f32 scales:
  the served rows as the Commit taps `Int8RowQuant_v1` /0 and /1, then empty rows), its B the layer's int8 weight rows and
  their f32 scales (registered), and output (i, j) is `ScaledMmS8Coordinate_v2` on row i and weight row j, the served
  linear's ``RN_bf16(RN32(RN32(f32(Y) * s_a) * s_w))`` on its exact s32 Y. The coordinates are one batch over the (row,
  column) grid, so each rest commits one BF16 (the width rule). Rows and weight rows off a whole tile (`TILE`, a warp's
  32 × 32, the tile the H1 kernel computes) are padded: pad row p reads constant zero bits of its own, contiguous, so it is a
  distinct row of the site, and its outputs are not served.
* **The static layout** (`H1Run`, the static-circuit rule): every call has the same T rows (``rows``, vLLM's
  ``max_num_batched_tokens``, a multiple of 32), so its Program is `program(k, n, T)` whatever the step served. The step's m
  served rows fill slots 0 .. m - 1 and every other slot is an empty row, zero bytes under scale 0 (`slots`). The window is S
  steps (``steps``) of one call per linear: call (s, w) is at positions s·U + off(w) + u (`layout`: U is a step's proof blocks,
  off(w) the proof blocks of the linears before w), and steps past the last served one are all empty. Nothing about a call's rows or
  positions comes from its tokens, batch or EOS. Gate-passed is the served rows of the calls the credit gate passes only;
  the padded work is shown beside it.
* **u** (`H1Run.utilization`, the ``u`` of `benchmarks/pouw/period_cost.py`): the window's served rows over its
  declared rows, Σ_s m_s / (S·T). A `Static` declaration (the static schedule's ``sampling.ignore_eos``: every request admitted at
  step 0, prefilled there and run to its cap) fixes each step's rows ahead of serving, so its u is public: 5 batched
  prompts (44 tokens) at cap 16 are 119/1152, and `FILL`'s 64 one-token prompts fill all 16 served steps, 16/18. A served
  step short of its declared rows is padded in the layout (its calls keep their positions) and refused by `Static.check`.
* **Groups.** A call is one site and one seed group (a batched call over a step's batch is one site): its
  (root(A), root(B)) seeds its proof blocks. Calls are never grouped together, since a later call reads an earlier one's output.
* **Proof blocks** are in compile's order, the order the manifest lists them in: the call, then its row tile, then its column
  tile, row-major.
* **The hook.** The arm hands every served call to `engine.int8_vllm.Int8PlainLinearMethod.observe` after both ops, and the
  option records it there as a `Call`: its engine step (counted at a linear the step already holds: GLM-5.3's forward
  runs its indexer's wq_b, module order's first, third), its layer (the linear's ordinal), its served rows and its
  position. Then each listener (`H1Run.listen`) gets the call with the served tensors, on the device: A (the s8
  rows, as the Commit taps ``Int8RowQuant_v1/0``), s_a (``/1``), the int8 weight and its scales; `H1Run.close` hands it the
  empty steps' calls too, with no rows. A's slot i is the call's row i, so a tile's rows are 32 consecutive slots. What a
  listener returns (its roots and receipt root, as hex) is recorded with the call (``bound``). `Call.inputs` is
  `Compiled.inputs` at the call's base, with the roots it is given: root(A) is the table's, and root(B) is the weight's,
  which compute accounting commits. This module commits neither.
* **The Commit** (ROOTS1, `pouw_h1_commit`, ``served``'s tap): each round's input-ids table ``ids@<s>`` and A table of
  each K are the Commit's values, its record bound into `calls_root` before the draw root and held to them by core's
  credit (`credit._arow`); a window that fails it credits nothing (`pouw_h1_commit.UNTIED`).
* **z** is the call's z/y records (`pouw_h1_records`): the int8 arm serves ``verity_int8::scaled_mm_zy`` over T slots (its
  ``zy_rows``), writing the served y and one record a proof block, its exact z (`Call.z`'s cells) then y = dq(z). With none,
  none is committed (with ``retain``, `engine.int8_exact.int_mm` beside, to check). The last ``retain`` calls' Y are kept.
* **The credit gate** (`credit_gate`, compute accounting's credit rule): each served call's ``apply`` runs under a trace
  (`traced`); the call gate-passes only if the y it hands back to vLLM was written by the GEMM invocation that wrote its
  z, the only GEMM of the call, and z is its records, which hold together with its scales and served y
  (`pouw_h1_records.reason`, else ``credit: z-records: ...``). An arm serving ``scaled_mm``'s y with z an ``_int_mm``
  beside it is refused (``credit: served-output-not-credited-kernel: ...``), as is a call recorded outside ``apply`` (`UNTRACED`).
  The verdict is the serving process's own, so it is only the call's credit bit (`H1Run.credit_bits`, ``credit`` in its
  manifest row under calls_root, before the draw root): credit is the verifier's (core's `credit`).
* **Requests and rounds** (`pouw_h1_requests`, `Accounting.Served`): round s is step s. `admitting` hands each
  step's admitted requests and batch order to `H1Run.admit` before its forward, so each request's entry is a hiding leaf
  under the root of the round that admits it, and each served row's slot names its request, before that round's coin; a
  window serving a row of a request with no entry credits nothing. A window admits only in round 0 unless its
  ``admission`` is ``per-round`` (`pouw_h1_requests.ADMISSIONS`).
* **Round values.** Each call's A table (its slots' s8 rows and f32 scales) and z table (a leaf a proof block, its record) are
  hiding tables of the call (``tables`` in its manifest row), so round s's root commits them, z at its cut, before round s's
  coin (``details.requests.a_roots``: `pouw_h1_requests.tables_reason` at `close`); round s's receipt roots (`H1Run.receipt`, after its coin) go under round s + 1's root, and the window's draw coin
  (``draw``) opens at `H1Run.finish`, after the manifest, the last round's receipts and every round's root and coin.
* **Coins.** Credit needs the window's coins from the auditor host's challenger (``coin_source`` ``challenger``,
  `pouw_h1_challenger`); a window whose coins were held in the serving process credits nothing (`IN_PROCESS`).
* **Salt.** `pouw.salt` is a SHA-512 digest and the epoch's salt (`audit.salt_of` over the weights' root, `weight_root`,
  from the challenger's salt coin) is 32 bytes, so the Program's salt is the epoch's under `SALT_TAG` (`program_salt`)
  until the challenger opens a 64-byte one. A challenger's window opens its salt coin after its own open root
  (`salt.window_open_root` of the weights' root, round 0's request root and its session), once round 0 is admitted
  and before step 0's forward (`H1Run.open_window`); with the Commit, that request root takes in round 0's ids table.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from verity.core.protocols.pouw.audit import HM96_WEIGHTS, weight_binding
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw.credit import CREDIT
from verity.core.protocols.sampled_proofs.challenger import CoinError
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, Strided, concat
from verity.ml.circuits.types import Array

from verity_vllm.protocols.pouw import credit_gate as CG
from verity_vllm.protocols.pouw.h1 import commit as CT
from verity_vllm.protocols.pouw.h1.commit import calls_root
from verity_vllm.protocols.pouw.h1 import prover as PV
from verity_vllm.protocols.pouw.h1 import records as ZR
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1 import tables_device as TD
from verity_vllm.protocols.pouw.h1.requests import TABLE_NONCES, TABLE_TAGS, table_leaves
from verity_vllm.protocols.pouw.h1.static import FILL, Static, bind_all, graph_pad, utilization

SCHEME = H.SCHEME
TILE = (32, 32)
#: the bits of one s8 entry and of one bf16 output
I8_BITS, BF16_BITS = 8, 16
RECORD = "verity-vllm/pouw-h1-calls/v2"
CALLS_FILE = "pouw-h1-calls.json"
#: the declared window Program a server writes under its ``h1/`` (`served.write_program`): its descriptor, gzipped, and
#: what a period-open of it reads, so ``pouw_h1_circuits --program OUT/h1`` builds its circuit files
PROGRAM, PROGRAM_META = "window-program.json.gz", "window-program.meta.json"
SALT_TAG = b"verity/pouw/h1/program-salt/v1\0"
#: the one coin source a credited window's coins come from (`pouw_h1_challenger`)
CHALLENGER = "challenger"
#: the credit refusal of every call of a window whose coins the auditor's challenger didn't draw
IN_PROCESS = ("credit: coins-not-the-auditors: the window's coins were held in the serving process, not drawn by the "
              "auditor host's challenger (the pouw setting 'challenger'), so it credits nothing")


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
    """Every proof block a full `TILE` grid: what the served kernel's warp computes and its receipt covers."""
    bad = Counter((u.tile.r, u.tile.c, u.tile.grid) for u in c.units if (u.tile.r, u.tile.c) != TILE or not u.tile.grid)
    if bad:
        raise C.CompileError("scheme", f"{c.served.fn.id}: units off the {TILE[0]} x {TILE[1]} grid: {dict(bad)}")


@lru_cache(maxsize=256)
def compiled(k: int, n: int, m: int) -> C.Compiled:
    """`program(k, n, m)` compiled to `SCHEME`, a seed group per call, every proof block a full tile."""
    c = C.compile(program(k, n, m), SCHEME)
    check_tiles(c)
    return c


def units(k: int, n: int, m: int) -> int:
    """The call's proof blocks, without compiling: its padded grid's tiles."""
    mp, np_ = padded(m, n)
    return (mp // TILE[0]) * (np_ // TILE[1])


#: this process's 32-byte key for its weight trees' leaf salts (`weight_root`): from the OS's CSPRNG when the module is first
#: imported (a forked child keeps its parent's) and held only in memory, never written to a record, a manifest or the store;
#: a tensor-parallel rank draws its own, since each commits only the weights it holds
WEIGHT_KEY = os.urandom(32)


def weight_root(w: int, rows: np.ndarray, key: bytes | None = None) -> bytes:
    """The epoch's commitment to weight `w`, which the salt coin is opened after (`audit.weights_root`): its int8 rows [n, k]
    in a hiding tree, ``audit.commit_rows(weight_binding(w), -2, rows, HM96, key, HM96_WEIGHTS + w)`` (`requests.hiding_rows`,
    the same tree), so the root says nothing of the rows to anyone without ``key`` (`WEIGHT_KEY` when None). Not root(B),
    which is compute accounting's."""
    key = WEIGHT_KEY if key is None else key
    rows = np.ascontiguousarray(rows, dtype=np.int8)
    return RQ.hiding_rows(weight_binding(w), -2, [r.tobytes() for r in rows], key, HM96_WEIGHTS + w).commitment.root


def layout(shapes: Sequence[tuple[int, int]], rows: int) -> tuple[list[int], int]:
    """off(w) of each linear (its (n, k), in order) and U, one step's proof blocks: linear w's call is its `units` at `rows`."""
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
    rows m (0 for a step past the last served one), its `rows` slots (T), k, n, its first position and ``pad``, its rows
    past the step's batch (a CUDA graph's padding, `H1Run.owners`). Its A table holds slot i at leaf i, computed for i < m
    and empty after (with the Commit, its AROW of (step, layer)); its `served` rows are the m - pad the scheduler served."""

    index: int
    step: int
    layer: int
    module: str
    m: int
    k: int
    n: int
    base: int
    rows: int
    pad: int = 0

    @property
    def served(self) -> int:
        return self.m - self.pad

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
        """Proof block `unit`'s z, in its cells' order, from the call's exact s32 Y [m, n] (empty slots and pad columns zero)."""
        u = self.compiled().units[unit]
        full = np.zeros(padded(self.rows, self.n), dtype=np.int32)
        full[:self.m, :self.n] = y
        return np.array([full[u.rows[i], u.cols[j]] for i, j in u.tile.pairs], dtype=np.int32)

    def summary(self) -> dict[str, Any]:
        return {"index": self.index, "step": self.step, "layer": self.layer, "module": self.module, "served": self.served,
                "rows": self.rows, "k": self.k, "n": self.n, "base": self.base, "units": self.units, "leaves": [0, self.rows]}


#: ``listener(call, a, s_a, weight, weight_scale)``: the served tensors on the device, which it must not change (a [m, k]
#: and s_a [m], m = 0 for an empty step; `slots` makes the call's T-slot table); it returns None or what it bound for the
#: call, as JSON (its roots and receipt root, hex)
Listener = Callable[[Call, Any, Any, Any, Any], Mapping[str, Any] | None]
#: ``opener(requests_root)``: given round 0's request table's root, the window's Program salt, its round coins and draw
#: coins (`LiveRecord`s) and what its commitment records of the opening (`H1Run.open_window`)
Opener = Callable[[bytes], tuple[bytes, Any, Any, Mapping[str, Any]]]


def numpy_of(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


class H1Run:
    """One install's H1 window on the int8_exact row, in the static form: its calls, rows and positions are a function of the
    public parameters alone, never of a step's tokens, batch or EOS. ``shapes`` are the linears' (n, k) in order, ``rows``
    (T) every call's slots and ``steps`` (S) the window's engine steps. Call (s, w) is at s·U + off(w) (`layout`); served
    calls are recorded as they come (`record`), and `close` adds the steps past the last served one, all empty. A call past
    step S or with more than T rows is refused: recorded in ``refused``, never positioned, and `static` is then False.
    ``retain`` keeps the last calls' A, s_a and exact Y on the host (0: none). Without ``commit`` (a Match capture) nothing
    is recorded. ``graphs`` are the engine's CUDA-graph capture sizes (none when eager), whose padding `graph_pad` names.

    Round s is step s. The engine's admissions and batch order (`admit`, from `admitting`) go to the round of the step about
    to run, and round s closes once every linear of step s is recorded (or at `close`): its root (`pouw_h1_requests.round_row`)
    over the step's calls, its request table and its slot table, and with ``served`` the step's A table of each K
    ``a/<K>@<s>`` whole (`pouw_h1_served.Served.segments`), is recorded with
    ``rounds`` (a `LiveRecord` over a live challenger's round family), which opens round s's coin only after it, and
    `finish` opens ``draw``'s coin (a `LiveRecord` of the same challenger's draw family) after the window's `draw_root`.
    ``key`` is the prover's hm96 salt key for the request, slot and call tables, drawn for the run (`os.urandom`) unless
    given. ``prover`` (else ``served``'s) is the prover's private directory (`pouw_h1_prover`): it keeps the key and each
    served call's A rows, scales and z/y records, so a drawn proof block opens after the close (DRAWN1); without it their leaves
    open only while the run is held (`open_request`, `open_slot`). ``admission`` is `pouw_h1_requests.ROUND0` (admit in round 0 only) or ``PER_ROUND``; ``coin_source`` is
    `CHALLENGER` only for coins from the auditor host's challenger, and any other credits nothing. With ``served``
    (`pouw_h1_served.Served`) each call's z/y table, its step's A table of K, and the weights' B, are also registered values
    as the calls are recorded, their roots in its tables (``registered``) under its round's root; the commitment carries them.

    The Program's ``salt`` is given, or None with ``opener``: then `open_window` opens it once round 0's admissions are in
    (or at the first use of the salt), after its request table, and ``opener`` sets ``rounds`` and ``draw``."""

    def __init__(self, entry: str, salt: bytes | None, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]],
                 rows: int, steps: int, retain: int = 0, transcript_dir: str | None = None, commit: bool = True,
                 rounds: Any = None, key: bytes | None = None, draw: Any = None, admission: str = RQ.ROUND0,
                 coin_source: str = "in-process", opener: Opener | None = None, served: Any = None,
                 prover: str | None = None, session: str | None = None, graphs: Sequence[int] = ()) -> None:
        if admission not in RQ.ADMISSIONS:
            raise ValueError(f"{entry}: an admission is one of {RQ.ADMISSIONS}, not {admission!r}")
        if (salt is None) == (opener is None):
            raise ValueError(f"{entry}: a window's salt is given or opened by its opener, one of the two")
        self.entry, self._salt, self._opener, self.modules, self.commit = entry, salt, opener, dict(modules), commit
        self.shapes_, self.rows, self.steps = [tuple(map(int, s)) for s in shapes], int(rows), int(steps)
        self.offs, self.u_step = layout(self.shapes_, self.rows)
        #: the calls at their indices s·L + w; per step, how many are in; the step's linears as they came; step 0's
        self.calls, self._in, self._came, self._order = list[Call](), Counter[int](), list[int](), None
        self.listeners: list[Listener] = []
        self.bound: dict[int, Mapping[str, Any]] = {}
        self.refused: list[dict[str, int]] = []
        self.retained: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=retain or None)
        self.retain, self.transcript_dir, self.graphs = retain, transcript_dir, tuple(graphs)
        self.step, self.order_ok, self.closed = -1, True, False
        self._weights: dict[int, tuple[Any, Any]] = {}
        #: each served call's credit verdict (`credit_gate.verdict`), by call index, and the call `record` just made with its z
        self.verdicts: dict[int, dict[str, Any]] = {}
        self._recorded: tuple[Call, Any] | None = None
        #: why the window's salt opening was refused (also ``details["unopened"]``): a window opens it once
        self._unopened: str | None = None
        self.rounds, self._key, self.served = rounds, key if key is not None else os.urandom(32), served
        self.draw, self.admission, self.coin_source = draw, admission, coin_source
        #: the gate's reserved session (None for in-process coins), bound by round 0's input-ids table (`Served.ids_tree`)
        self.session = session
        #: the root of round 0's input-ids table, inside round 0's request root (`open_window`)
        self._ids0: bytes | None = None
        self.prover = prover if prover is not None else getattr(served, "prover", None)
        if self.prover is not None and commit:
            PV.keep_key(self.prover, self._key)
        #: each call's table roots (`tables`), the calls whose tables the GPU is still hashing, and each call's receipt root
        self._roots: dict[int, dict[str, str | None]] = {}
        self._hashing: deque[tuple[int, TD.Pending]] = deque()
        self.receipts: dict[int, bytes] = {}
        #: the admitted requests' entries in admission order, each round's, and the requests refused admission
        self.requests: list[RQ.RequestEntry] = []
        self.admitted: dict[int, list[RQ.RequestEntry]] = {}
        self.refused_requests: list[dict[str, Any]] = []
        #: each round's slot owners (the request index of each of the step's rows, in its batch order, None for a row of a
        #: request with no entry); requests scheduled again as new (resumed after a preemption), which keep their entry;
        #: the rows of requests with no entry; and the steps whose batch isn't their served rows
        self.owners: dict[int, list[int | None]] = {}
        self.readmitted: list[dict[str, int]] = []
        self.unadmitted: list[dict[str, int]] = []
        self.unbound: list[str] = []
        self._ids: dict[str, int] = {}
        #: each closed round's record: its root, the parts it is over and its coin's opening (None without ``rounds``)
        self.round_log: list[dict[str, Any]] = []
        self._tables: dict[int, Any] = {}
        self._slot_tables: dict[int, Any] = {}
        self.commits: dict[str, Any] = {"scheme": SCHEME, "tile": list(TILE), "rows": self.rows, "steps": self.steps,
                                        "u_step": self.u_step, "units": self.steps * self.u_step, "calls": 0,
                                        "rounds": self.round_log, "admission": self.admission}
        self.details: dict[str, Any] = {"scheme": SCHEME, "calls": 0, "refused": self.refused,
                                        "uncovered": "attention (QK^T and P.V in Int8LutAttentionRow_v1), lm_head and "
                                                     "a MoE layer's router (bf16), and the linears the checkpoint's "
                                                     "modules_to_not_convert keeps bf16",
                                        "z": "the call's z/y records from verity_int8::scaled_mm_zy, if its arm hands them over"}

    def listen(self, fn: Listener) -> None:
        """Hand every later call to `fn` (`Listener`)."""
        self.listeners.append(fn)

    @property
    def tables(self) -> dict[int, dict[str, str | None]]:
        """Each call's table roots (hex; z None when the run has no exact Y for it), once the GPU has hashed them all."""
        TD.settle(self, wait=True)
        if self.served is not None:
            self.served.settle()
        return self._roots

    @property
    def salt(self) -> bytes:
        return self._salt if self._salt is not None else self.open_window()

    def open_window(self, ids: bytes | None = None) -> bytes:
        """The Program's salt, opened once: round 0's request table is committed (its binding reads no salt) and round 0's
        request root (`_requests0`, over ``ids``, round 0's input-ids table's root, with the Commit, so the prompt tokens
        are fixed before the salt) handed to ``opener``, which opens the window's session and its salt coin after the
        window's open root over it (`pouw_h1_window.Window.open`); the commitment records what ``opener`` returns."""
        if ids is not None:
            self._ids0 = ids
        if self._salt is None:
            if self._unopened is not None:
                raise RuntimeError(f"{self.entry}: the window's salt was refused ({self._unopened}), and a "
                                   "window opens it once")
            table = self._tables[0] = (self._live("requests", 0)
                                       or RQ.request_table(None, 0, self.rows, self.admitted.get(0, []), self._key))
            requests = self._requests0(table.commitment.root)
            try:
                salt, self.rounds, self.draw, commits = self._opener(requests)
            except Exception as e:
                self._unopened = self.details["unopened"] = str(e)
                raise
            self.commits.update(commits)
            self._salt = salt
        return self._salt

    def _live(self, name: str, r: int) -> Any:
        """Round r's live request (round 0's) or slot table with the Commit (`pouw_h1_live`), else None."""
        return None if self.served is None else self.served.live_table(self, name, r)

    def _requests0(self, table: bytes) -> bytes:
        """Round 0's request root: `credit.requests_root` of `table` and round 0's ids table's root, else `table`."""
        requests = table if self._ids0 is None else CR.requests_root(table, self._ids0)
        self.commits["requests_root"], self.commits["request_table"] = requests.hex(), table.hex()
        return requests

    def _commit_tables(self, call: Call, a: Any, s_a: Any, records: np.ndarray | None, zy: Any = None) -> None:
        """The call's A and z tables (`pouw_h1_tables_device.commit_tables`)."""
        TD.commit_tables(self, call, a, s_a, records, zy)

    def call(self, index: int) -> Call:
        """Call `index`; KeyError when the window has none."""
        i = bisect.bisect_left(self.calls, index, key=lambda c: c.index)
        if i == len(self.calls) or self.calls[i].index != index:
            raise KeyError(f"{self.entry}: no call {index}")
        return self.calls[i]

    def _call(self, s: int, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, records: Any = None,
              zy: Any = None) -> Call:
        n, k = self.shapes_[w]
        m, batch = int(a.shape[0]) if a is not None else 0, len(self.owners.get(s, ()))
        call = Call(s * len(self.shapes_) + w, s, w, self.modules.get(w, str(w)), m, k, n, s * self.u_step + self.offs[w],
                    self.rows, graph_pad(m, batch, self.graphs) if s in self.owners else 0)
        bisect.insort(self.calls, call, key=lambda c: c.index)
        self._in[s] += 1
        if self.commit:
            self._commit_tables(call, a, s_a, records, zy)
            if self.served is not None:
                self._roots[call.index]["registered"] = self.served.commit(self, call, a, s_a, weight, weight_scale, records)
        for fn in self.listeners:
            out = fn(call, a, s_a, weight, weight_scale)
            if out is not None:
                self.bound[call.index] = {**self.bound.get(call.index, {}), **out}
        self.details["calls"] = self.commits["calls"] = len(self.calls)
        if self._in[s] == len(self.shapes_):
            self._close_rounds(s)
        return call

    def admit(self, new: Sequence[tuple[str, Sequence[int] | None, int]], order: Sequence[tuple[str, int]]
              ) -> list[RQ.RequestEntry]:
        """The step about to run: the requests the engine admits for it, each (req_id, prompt token ids, token cap), and its
        batch order, each scheduled request's (req_id, rows) in the order the step's A tables hold its rows
        (`pouw_h1_requests.batch_order`). Each new request's entry goes in the step's round, at its first slot; a req_id
        already admitted (a request resumed
        after a preemption) keeps its entry (`readmitted`). A request past the window's S steps, with no prompt token or
        cap, or not in the batch is refused (`refused_requests`): it has no entry, and its rows are a gap
        (`request_gaps`), as is every row of a scheduled request never admitted."""
        if self.closed:
            raise RuntimeError(f"{self.entry}: an admission after the window closed")
        if not self.commit:
            return []
        r = self.step + 1
        self._close_rounds(r - 1)
        if r < len(self.round_log):
            raise RuntimeError(f"{self.entry}: an admission into round {r}, which has closed")
        if r in self.owners:
            self.unbound.append(f"step {r}: scheduled twice before its forward")
            return []
        at, n = {}, 0
        for req_id, rows in order:
            at[req_id], n = n, n + int(rows)
        out = []
        for req_id, prompt, cap in new:
            if req_id in self._ids:
                self.readmitted.append({"round": r, "index": self._ids[req_id]})
                continue
            prompt = tuple(int(t) for t in prompt or ())
            why = (f"admitted for step {r}, past the window's {self.steps} steps" if r >= self.steps
                   else RQ.late_admission(r) if r and self.admission == RQ.ROUND0
                   else "admitted after round 0's request table was committed for the window's salt" if r == 0
                   and 0 in self._tables
                   else "no prompt token ids or no token cap" if not prompt or int(cap) < 1
                   else "admitted and not in its step's batch" if req_id not in at else None)
            if why is not None:
                self.refused_requests.append({"round": r, "reason": why})
                continue
            e = RQ.RequestEntry(len(self.requests), r, at[req_id], int(cap), prompt)
            self._ids[req_id] = e.index
            self.requests.append(e)
            self.admitted.setdefault(r, []).append(e)
            out.append(e)
        owners: list[int | None] = []
        for req_id, rows in order:
            index = self._ids.get(req_id)
            if index is None and int(rows):
                self.unadmitted.append({"round": r, "rows": int(rows)})
            owners += [index] * int(rows)
        self.owners[r] = owners
        if r == 0:
            self.open_window(None if self.served is None else self.served.open_ids(
                self, RQ.first_ids({q: self.requests[i].prompt for q, i in self._ids.items()}, order)))
        return out

    def _close_rounds(self, through: int) -> None:
        """Close every round up to `through`, in order: its request and slot tables, then its root, recorded before its coin
        opens. A step whose served rows (linear 0's) aren't its batch's is named in `unbound`."""
        while len(self.round_log) <= min(through, self.steps - 1):
            r = len(self.round_log)
            owners = self.owners.get(r, [])
            served = next((c.served for c in self.calls if c.step == r and c.layer == 0), 0)
            if len(owners) != served:
                self.unbound.append(f"step {r}: {served} rows served and {len(owners)} in the step's batch, so the slot "
                                    "table can't name each served row's request")
            salt = self.salt
            if r not in self._tables:
                self._tables[r] = ((r == 0 and self._live("requests", 0))
                                   or RQ.request_table(salt, r, self.rows, self.admitted.get(r, []), self._key))
            table = self._tables[r]
            slots = self._slot_tables[r] = (self._live("slots", r)
                                            or RQ.slot_table(self.salt, r, self.rows, owners, self._key))
            calls = RQ.step_digest(self.salt, r, self._rows([c for c in self.calls if c.step == r]))
            receipts = self.receipts_of(r - 1) if r else RQ.NO_RECEIPTS
            if self.served is not None:
                self.served.close_round(self, r)
            requests = self._requests0(table.commitment.root) if r == 0 else table.commitment.root
            row = RQ.round_row(self.salt, r, calls, requests, slots.commitment.root, receipts,
                               self.served.segments(self, r) if self.served is not None else {})
            if r == 0:
                row["request_table"] = table.commitment.root.hex()
            coin = self.rounds.record(bytes.fromhex(row["root"])) if self.rounds is not None else None
            self.round_log.append({**row, "coin": coin.to_json() if coin else None})

    def receipts_of(self, r: int) -> bytes:
        """Round r's receipts (`pouw_h1_requests.receipts_digest`): its calls' receipt roots as handed to `receipt`."""
        return RQ.receipts_digest(self.salt, r, [(c.index, self.receipts.get(c.index)) for c in self.calls if c.step == r])

    def receipt(self, index: int, root: bytes) -> None:
        """Call `index`'s receipt root, computed after its round's coin: it goes under the next round's root (the last
        round's in the draw's root), so it is refused before that coin, once the next round has closed, or twice."""
        call = self.call(index)
        if len(self.round_log) <= call.step:
            raise RuntimeError(f"{self.entry}: call {index}'s receipt before round {call.step}'s coin")
        if len(self.round_log) > call.step + 1 or "draw" in self.commits or index in self.receipts:
            raise RuntimeError(f"{self.entry}: call {index}'s receipt after the root that commits round {call.step}'s "
                               "receipts, or a second one")
        if len(root) != 64:
            raise ValueError(f"{self.entry}: a receipt root is 64 bytes")
        self.receipts[index] = bytes(root)

    def open_request(self, index: int) -> RQ.RoundOpening:
        """Request `index`'s entry opened under its round's request table, once that round has closed."""
        e = self.requests[index]
        o = self._tables[e.round].open(self.admitted[e.round].index(e))
        return RQ.RoundOpening(e.round, o.position, o.value, o.path)

    def open_slot(self, s: int, position: int) -> RQ.RoundOpening:
        """Slot `position` of step `s` opened under round s's slot table, once that round has closed."""
        o = self._slot_tables[s].open(position)
        return RQ.RoundOpening(s, o.position, o.value, o.path)

    def open_first(self, index: int) -> RQ.RoundOpening:
        """Request `index`'s first slot, opened under its round's slot table."""
        e = self.requests[index]
        return self.open_slot(e.round, e.slot)

    def request_gaps(self) -> list[str]:
        """Every served row the window can't name an entry for: rows of requests with no entry, and steps whose batch isn't
        their served rows. A window with one credits nothing (`credit`)."""
        return [f"step {u['round']}: {u['rows']} row{'' if u['rows'] == 1 else 's'} of a request with no entry"
                for u in self.unadmitted] + self.unbound

    def record(self, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, z: Any = None) -> Call | None:
        """Linear `w`'s served call: its s8 rows a [m, k], their f32 scales s_a [m], its int8 weight [n, k] and scales, and
        the z its arm handed over: its records (`pouw_h1_records.of`), committed as its z table; without them nothing is
        committed for z, and with ``retain`` `int_mm` computes its exact Y beside, only to retain it."""
        if self.closed:
            raise RuntimeError(f"{self.entry}: a call after the window closed")
        if self.step < 0 or w in self._came:
            self._order = self._came if self._order is None and self.step >= 0 else self._order
            self.step, self._came = self.step + 1, []
        self.order_ok &= self._order is None or self._order[len(self._came):len(self._came) + 1] == [w]
        self._came.append(w)
        self._weights[w] = (weight, weight_scale)
        m, k = (int(d) for d in a.shape)
        if not 0 <= self.step < self.steps or m > self.rows or (int(weight.shape[0]), k) != self.shapes_[w]:
            self.refused.append({"step": self.step, "linear": w, "m": m})
            return None
        n = self.shapes_[w][0]
        records = ZR.of(z, units(k, n, self.rows)) if n % TILE[1] == 0 else None
        zy = z if records is not None else None
        if z is None and self.retain:
            from verity_vllm.engine.int8_exact import int_mm
            z = int_mm(a, weight)
        call = self._call(self.step, w, a, s_a, weight, weight_scale, records, zy)
        if self.retain:
            y = ZR.split(ZR.host(records), call.rows, call.n)[0][:m] if records is not None else numpy_of(z)
            self.retained.append((call.index, numpy_of(a), numpy_of(s_a), y))
        self._recorded = (call, z, None if records is None else (records, s_a, weight_scale))
        return call

    def judge(self, trace: CG.Trace, y: Any) -> dict[str, Any] | None:
        """The credit gate on the call `record` made inside ``trace`` (None if it made none): y is what the linear handed
        back to vLLM; a call the trace credits whose records don't hold is refused (`pouw_h1_records.refusal`), read on
        the records' device first (`pouw_h1_records.holds_on`) and on the host's copy only when that doesn't hold."""
        made, self._recorded = self._recorded, None
        if made is None:
            return None
        call, z, held = made
        v = CG.verdict(trace, y, z)
        if v["credit"] and not (held is not None and ZR.holds_on(z, call.rows, call.n, y, held[1], held[2])) \
                and (why := ZR.refusal(held and (held[0], numpy_of(held[1]), numpy_of(held[2])), call,
                                       TD.row_words(y, call.m, "<i2"))):
            v = {**v, "credit": False, "reason": f"credit: {CG.RECORDS}: {trace.linear} {why}"}
        self.verdicts[call.index] = v
        return v

    def close(self) -> None:
        """The steps past the last served one, each linear's call empty (m = 0), handed to the listeners like the rest."""
        if self.closed:
            return
        self.closed = True
        if not self.commit or self._unopened is not None:
            return
        self.details["steps_served"] = self.step + 1
        self.details["last_step_complete"] = self.step < 0 or sum(c.step == self.step for c in self.calls) == len(self.shapes_)
        for s in range(self.step + 1, self.steps):
            for w, (n, k) in enumerate(self.shapes_):
                weight, scale = self._weights.get(w, (None, None))
                a = weight.new_zeros((0, k)) if weight is not None else None
                self._call(s, w, a, scale.reshape(-1).new_zeros(0) if scale is not None else None, weight, scale)
        self._close_rounds(self.steps - 1)
        if self.served is not None:
            self.served.close(self)
            if self.served.tap is not None:
                self.details["roots1"] = CT.roots1(self)
        unbound = RQ.tables_reason(self.salt, self.round_log, self.manifest())
        self.details["requests"] = {"admitted": len(self.requests), "refused": self.refused_requests,
                                    "per_round": [len(self.admitted.get(r, [])) for r in range(self.steps)],
                                    "readmitted": len(self.readmitted), "gaps": self.request_gaps(),
                                    "coins": "live" if self.rounds is not None else "none",
                                    "coin_source": self.coin_source, "admission": self.admission,
                                    "a_roots": "bound: each call's A table is under its round's root" if unbound is None
                                    else f"unbound: {unbound}"}
        self.details["static"] = self.static()
        self.details["step_rows"] = self.step_rows()
        self.details["u"] = str(self.utilization())
        self.details["gate"] = self.credit()

    def static(self) -> bool:
        """Every (step, linear) call of the window, in order, at its public position, and nothing refused."""
        want = [(s, w, s * self.u_step + self.offs[w], units(k, n, self.rows))
                for s in range(self.steps) for w, (n, k) in enumerate(self.shapes_)]
        return self.order_ok and not self.refused and [(c.step, c.layer, c.base, c.units) for c in self.calls] == want

    def step_rows(self) -> list[int]:
        """Each step's served rows, its first linear's (0 for a step with no call)."""
        rows = {c.step: c.served for c in self.calls if c.layer == 0}
        return [rows.get(s, 0) for s in range(self.steps)]

    def utilization(self) -> Fraction:
        return utilization(self.step_rows(), self.rows, self.steps)

    def verdict(self, call: Call) -> dict[str, Any]:
        """The call's credit verdict; a served call recorded outside a traced ``apply`` (`traced`) is refused."""
        return self.verdicts.get(call.index) or {
            "linear": call.module, "credit": False,
            "reason": f"credit: {CG.UNTRACED[1:-1]}: {call.module} call {call.index} was recorded outside a traced apply"}

    def gated(self) -> dict[int, dict[str, Any]]:
        """Each served call's verdict: the credit gate's, refused in a window that served a row it can't name an entry for
        (`request_gaps`), whose coins aren't the challenger's (`IN_PROCESS`) or, with the Commit, whose A tables or input
        ids aren't the Commit's (`pouw_h1_commit.UNTIED`, ``details.roots1``)."""
        out = {c.index: self.verdict(c) for c in self.calls if c.m}
        untied = bool((self.details.get("roots1") or {}).get("reason"))
        for why, on in ((RQ.UNENTERED, bool(self.request_gaps())), (IN_PROCESS, self.coin_source != CHALLENGER),
                        (CT.UNTIED, untied)):
            out = {i: {**v, "credit": False, "reason": why} if on and v["credit"] else v for i, v in out.items()}
        return out

    def credit_bits(self) -> dict[int, bool]:
        """Each call's credit bit, its manifest row's `credit.CREDIT` under calls_root (so fixed before the draw root): its
        `gated` verdict, true only for a served call with a z table."""
        gated = self.gated()
        return {c.index: c.index in gated and gated[c.index]["credit"] and (self.tables.get(c.index) or {}).get("z") is not None
                for c in self.calls}

    def credit(self) -> dict[str, Any]:
        """Gate-passed: the served rows' MACs of the calls the serving process's own credit gate passes (`gated`), only. No
        verifier reads this: a window's credit is its verifier's (`pouw_h1_challenger.credit_reason`). The padded
        work (empty slots and pad columns) is shown beside it, and so are the refused calls' rows, with each
        refusal's reason."""
        served = [c for c in self.calls if c.m]
        gaps, verdicts = self.request_gaps(), self.gated()
        ok = [c for c in served if verdicts[c.index]["credit"]]
        no = [c for c in served if not verdicts[c.index]["credit"]]
        computed = sum(padded(c.rows, c.n)[0] * padded(c.rows, c.n)[1] * c.k for c in self.calls)
        all_served = sum(c.served * c.n * c.k for c in served)
        return {"rule": "gate-pass the served rows of calls whose served y is the credited kernel's own output only, and none "
                        "in a window that served a row of a request with no entry or whose coins the auditor's challenger "
                        "didn't draw; padded work and refused calls are shown, not gate-passed", "requests": gaps,
                "coins": self.coin_source,
                "gate_passed": {"rows": sum(c.served for c in ok), "macs": sum(c.served * c.n * c.k for c in ok)},
                "refused": {"calls": len(no), "rows": sum(c.served for c in no), "macs": sum(c.served * c.n * c.k for c in no),
                            "reasons": CG.reasons(verdicts[c.index] for c in no)},
                "padded": {"rows": sum(c.rows - c.served for c in self.calls), "macs": computed - all_served},
                "units": sum(c.units for c in self.calls),
                "served_frac_of_macs": round(all_served / computed, 6) if computed else 0.0}

    def shapes(self) -> list[tuple[int, int, int]]:
        return sorted({(c.k, c.n, c.rows) for c in self.calls})

    def check(self) -> dict[str, Any]:
        """Compile every call shape to `SCHEME` and hold each call's proof blocks to its compiled Program's: per shape (T, k, n) its
        Program digest, proof blocks, seed groups and tile shapes."""
        out = {}
        for k, n, t in self.shapes():
            c = compiled(k, n, t)
            if len(c.units) != units(k, n, t) or c.groups != ((0,),):
                raise C.CompileError("position", f"{k} x {n} at T = {t}: {len(c.units)} units in {len(c.groups)} groups, not "
                                                 f"{units(k, n, t)} in one")
            out[f"{t}x{k}x{n}"] = {"digest": c.digest, "units": len(c.units), "groups": [list(g) for g in c.groups],
                                   "cover": list(c.cover), "tiles": sorted({f"{u.tile.r}x{u.tile.c}" for u in c.units})}
        return out

    def _rows(self, calls: Sequence[Call]) -> list[dict[str, Any]]:
        return [{**c.summary(), **({"tables": dict(self.tables[c.index])} if c.index in self.tables else {}),
                 **({"bound": dict(self.bound[c.index])} if c.index in self.bound else {})} for c in calls]

    def manifest(self) -> list[dict[str, Any]]:
        """Each call's summary, with what its listeners bound and its credit bit (`credit_bits`), which the step digests
        under the round roots leave out (`pouw_h1_requests.step_rows`): a step's last verdict comes after its round closes."""
        bits = self.credit_bits()
        return [{**row, CREDIT: bits[row["index"]]} for row in self._rows(self.calls)]

    def calls_root(self) -> bytes:
        """SHA-512 over the manifest, in order: what the install commits to (its positions and what was bound), and with
        ``served`` its registered trees' entries and the Commit's record (`calls_root`)."""
        return calls_root(self.salt, self.manifest(), None if self.served is None else self.served.digest(),
                          CT.record_of(self))

    def finish(self) -> None:
        """Close the window, commit its manifest, then open ``draw``'s coin after `pouw_h1_requests.draw_root` over the
        manifest, the last round's receipts and every round's root and coin, so the draw follows all of them."""
        self.close()
        if not self.commit or self._unopened is not None:
            return
        commit = CT.record_of(self)
        if commit is not None:
            self.commits["commit"] = commit
        calls_root = self.calls_root()
        self.commits.update(calls_root=calls_root.hex(), static=self.details["static"])
        if self.served is not None:
            self.commits["served"] = {"entries": self.served.entries(), "refused": list(self.served.refused)}
        if "draw" not in self.commits:
            receipts = self.receipts_of(self.steps - 1)
            root = RQ.draw_root(calls_root, receipts, self.round_log)
            self.commits["draw"] = {"root": root.hex(), "receipts": receipts.hex(), "coin": None}
            try:
                coin = self.draw.record(root) if self.draw is not None else None
            except CoinError as e:
                self.commits["draw"]["refused"] = str(e)
            else:
                self.commits["draw"]["coin"] = coin.to_json() if coin else None
        if self.transcript_dir:
            d = Path(self.transcript_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / CALLS_FILE).write_text(json.dumps({"schema": RECORD, "entry": self.entry, "commitment": self.commits,
                                                    "public": {"rows": self.rows, "steps": self.steps,
                                                               "shapes": [list(s) for s in self.shapes_]},
                                                    "details": {k: v for k, v in self.details.items() if k != "refused"},
                                                    "refused": self.refused, "calls": self.manifest()}) + "\n")


__all__ = ["CALLS_FILE", "CHALLENGER", "Call", "FILL", "H1Run", "IN_PROCESS", "Listener", "Opener", "PROGRAM", "PROGRAM_META", "RECORD", "SALT_TAG",
           "SCHEME", "Static", "TABLE_TAGS", "TILE", "VllmInt8LinearH1", "bind_all", "calls_root", "check_tiles", "compiled",
           "coordinate", "layout", "numpy_of", "padded", "program", "program_salt", "slots",
           "table_leaves", "units", "utilization", "weight_root", "TABLE_NONCES", "WEIGHT_KEY"]
