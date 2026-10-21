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
* **u** (`H1Run.utilization`, PERIOD-PATH's ``u`` in `benchmarks/pouw/period_cost.py`): the window's served rows over its
  declared rows, Σ_s m_s / (S·T). A `Static` declaration (VLLM-31's ``sampling.ignore_eos``: every request admitted at
  step 0, prefilled there and run to its cap) fixes each step's rows ahead of serving, so its u is public: SERVE-H1's 5
  prompts (44 tokens) at cap 16 are 119/1152, and `FILL`'s 64 one-token prompts fill all 16 served steps, 16/18. A served
  step short of its declared rows is padded in the layout (its calls keep their positions) and refused by `Static.check`.
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
* **Requests and rounds** (`pouw_h1_requests`, #2677's `Accounting.Served`): round s is step s. `admitting` hands each
  step's admitted requests and batch order to `H1Run.admit` before its forward, so each request's entry is a hiding leaf
  under the root of the round that admits it, and each served row's slot names its request, before that round's coin; a
  window serving a row of a request with no entry credits nothing. A window admits only in round 0 unless its
  ``admission`` is ``per-round`` (`pouw_h1_requests.ADMISSIONS`).
* **Round values.** Each call's A table (its slots' s8 rows and f32 scales), its served y and its exact z when the run has
  one are hiding tables of the call (``tables`` in its manifest row), so round s's root commits them before round s's
  coin (``details.requests.a_roots``: `pouw_h1_requests.tables_reason` at `close`); round s's receipt roots (`H1Run.receipt`, after its coin) go under round s + 1's root, and the window's draw coin
  (``draw``) opens at `H1Run.finish`, after the manifest, the last round's receipts and every round's root and coin.
* **Coins.** Credit needs the window's coins from the auditor host's challenger (``coin_source`` ``challenger``,
  `pouw_h1_challenger`); a window whose coins were held in the serving process credits nothing (`IN_PROCESS`).
* **Salt.** `pouw.salt` is a SHA-512 digest and the epoch's salt (`audit.salt_of` over the weights' root, `weight_root`,
  from the challenger's salt coin) is 32 bytes, so the Program's salt is the epoch's under `SALT_TAG` (`program_salt`)
  until the challenger opens a 64-byte one. A challenger's window opens its salt coin after its own open root
  (`salt.window_open_root` of the weights' root, round 0's request table's root and its session), once round 0 is
  admitted and before step 0's forward (`H1Run.open_window`).
"""
from __future__ import annotations

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
from verity.core.protocols.pouw.audit import HM96, commit_rows, weight_binding
from verity.core.protocols.pouw.circuit import compile as C
from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.protocol import tagged
from verity.core.protocols.sampled_proofs.challenger import CoinError
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, Strided, concat
from verity.ml.circuits.types import Array

from verity_vllm.protocol_options import credit_gate as CG
from verity_vllm.protocol_options import pouw_h1_requests as RQ

SCHEME = H.SCHEME
TILE = (32, 32)
#: the bits of one s8 entry and of one bf16 output
I8_BITS, BF16_BITS = 8, 16
RECORD = "verity-vllm/pouw-h1-calls/v2"
CALLS_FILE = "pouw-h1-calls.json"
SALT_TAG = b"verity/pouw/h1/program-salt/v1\0"
#: a call's hiding tables, each T leaves under the run's salt key (`H1Run`): A's slot rows and scales, y's rows, z's rows
TABLE_TAGS = {"a": "verity/pouw/h1/a-table/v1", "y": "verity/pouw/h1/y-table/v1", "z": "verity/pouw/h1/z-table/v1"}
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


def utilization(step_rows: Sequence[int], rows: int, steps: int) -> Fraction:
    """PERIOD-PATH's u: the window's served rows over its declared rows, Σ_s m_s / (S·T)."""
    if len(step_rows) > steps or any(not 0 <= m <= rows for m in step_rows):
        raise ValueError(f"{list(step_rows)} is not a window of {steps} steps of at most {rows} rows")
    return Fraction(sum(step_rows), rows * steps)


@dataclass(frozen=True)
class Static:
    """A static H1 window's declaration (VLLM-31's ``sampling.ignore_eos``): requests of ``prompt_lens`` tokens, all admitted
    at step 0 and prefilled there in one step, each running exactly ``max_tokens`` steps (no early stop, no preemption, no host
    prefix cache), at T ``rows`` a call and S ``steps``. Each step's rows are then a function of the declaration alone
    (`schedule`), and so is the window's u."""

    prompt_lens: tuple[int, ...]
    max_tokens: int
    rows: int
    steps: int

    def __post_init__(self) -> None:
        layout([], self.rows)
        if not self.prompt_lens or min(self.prompt_lens) < 1 or sum(self.prompt_lens) > self.rows:
            raise ValueError(f"prompts of {list(self.prompt_lens)} tokens don't prefill in one step of {self.rows} rows")
        if not 1 <= self.max_tokens <= self.steps:
            raise ValueError(f"a cap of {self.max_tokens} tokens doesn't fit a window of {self.steps} steps")

    @property
    def slots(self) -> int:
        return len(self.prompt_lens)

    def schedule(self) -> list[int]:
        """Each step's rows: the prompts' tokens at step 0, one row a request for its other ``max_tokens`` - 1 steps, then the
        window's empty steps."""
        return [sum(self.prompt_lens)] + [self.slots] * (self.max_tokens - 1) + [0] * (self.steps - self.max_tokens)

    @property
    def u(self) -> Fraction:
        return utilization(self.schedule(), self.rows, self.steps)

    def engine_args(self) -> dict[str, Any]:
        """vLLM's scheduler limits for the declaration: every request in flight at once, T tokens a step, no prefix cache."""
        return {"max_num_seqs": self.slots, "max_num_batched_tokens": self.rows, "enable_chunked_prefill": True,
                "enable_prefix_caching": False}

    def check(self, realized: Sequence[int]) -> list[str]:
        """Each step whose served rows aren't the declared ones, by name; empty when the window is the declaration."""
        want = self.schedule()
        if len(realized) > len(want):
            return [f"{len(realized)} served steps past the window's {len(want)}"]
        got = list(realized) + [0] * (len(want) - len(realized))
        return [f"step {s}: {m} rows served, {d} declared ({'fewer' if m < d else 'more'}: an early stop, a preemption or an "
                f"admission off the declaration decided it, not the public schedule)"
                for s, (m, d) in enumerate(zip(got, want)) if m != d]


#: PERIOD-PATH's ask 1: 64 one-token prompts at cap 16, every row of SERVE-H1's 16 served steps filled (u = 16/18)
FILL = Static((1,) * 64, 16, 64, 18)


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
#: ``opener(requests_root)``: given round 0's request table's root, the window's Program salt, its round coins and draw
#: coins (`LiveRecord`s) and what its commitment records of the opening (`H1Run.open_window`)
Opener = Callable[[bytes], tuple[bytes, Any, Any, Mapping[str, Any]]]


def numpy_of(t: Any) -> np.ndarray:
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _words(t: Any, m: int, dtype: str) -> np.ndarray:
    """`t`'s m rows as little-endian `dtype` words, a bf16 tensor by its bits."""
    if hasattr(t, "detach") and str(t.dtype) == "torch.bfloat16":
        import torch
        t = t.detach().contiguous().view(torch.int16)
    return np.ascontiguousarray(numpy_of(t).reshape(m, -1), dtype=dtype)


def table_leaves(rows: int, width: int, m: int, *parts: np.ndarray) -> list[bytes]:
    """A call's table leaves: slot i < m the bytes of row i of each of `parts` in turn, every later slot `width` zero bytes."""
    out = [b"".join(p[i].tobytes() for p in parts) for i in range(m)]
    return out + [bytes(width)] * (rows - m)


class H1Run:
    """One install's H1 window on the int8_exact row, in the static form: its calls, rows and positions are a function of the
    public parameters alone, never of a step's tokens, batch or EOS. ``shapes`` are the linears' (n, k) in order, ``rows``
    (T) every call's slots and ``steps`` (S) the window's engine steps. Call (s, w) is at s·U + off(w) (`layout`); served
    calls are recorded as they come (`record`), and `close` adds the steps past the last served one, all empty. A call past
    step S or with more than T rows is refused: recorded in ``refused``, never positioned, and `static` is then False.
    ``retain`` keeps the last calls' A, s_a and exact Y on the host (0: none). Without ``commit`` (a Match capture) nothing
    is recorded.

    Round s is step s. The engine's admissions and batch order (`admit`, from `admitting`) go to the round of the step about
    to run, and round s closes once step s's last linear is recorded (or at `close`): its root
    (`pouw_h1_requests.round_root`) over the step's calls, its request table and its slot table is recorded with
    ``rounds`` (a `LiveRecord` over a live challenger's round family), which opens round s's coin only after it, and
    `finish` opens ``draw``'s coin (a `LiveRecord` of the same challenger's draw family) after the window's `draw_root`.
    ``key`` is the prover's hm96 salt key for the request, slot and call tables, drawn for the run (`os.urandom`) unless
    given; nothing persists it or the tables, so their leaves open only while the run is held (`open_request`,
    `open_slot`). ``admission`` is `pouw_h1_requests.ROUND0` (admit in round 0 only) or ``PER_ROUND``; ``coin_source`` is
    `CHALLENGER` only for coins from the auditor host's challenger, and any other credits nothing. With ``served``
    (`pouw_h1_served.Served`) the window's A codes and B rows are also committed as the window Program's registered
    values, at `close`, and `calls_root` binds them; the commitment carries their entries (``served``).

    The Program's ``salt`` is given, or None with ``opener``: then `open_window` opens it once round 0's admissions are in
    (or at the first use of the salt), after its request table, and ``opener`` sets ``rounds`` and ``draw``."""

    def __init__(self, entry: str, salt: bytes | None, modules: Mapping[int, str], shapes: Sequence[tuple[int, int]],
                 rows: int, steps: int, retain: int = 0, transcript_dir: str | None = None, commit: bool = True,
                 rounds: Any = None, key: bytes | None = None, draw: Any = None, admission: str = RQ.ROUND0,
                 coin_source: str = "in-process", opener: Opener | None = None, served: Any = None) -> None:
        if admission not in RQ.ADMISSIONS:
            raise ValueError(f"{entry}: an admission is one of {RQ.ADMISSIONS}, not {admission!r}")
        if (salt is None) == (opener is None):
            raise ValueError(f"{entry}: a window's salt is given or opened by its opener, one of the two")
        self.entry, self._salt, self._opener, self.modules, self.commit = entry, salt, opener, dict(modules), commit
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
        #: why the window's salt opening was refused (also ``details["unopened"]``): a window opens it once
        self._unopened: str | None = None
        self.rounds, self._key, self.served = rounds, key if key is not None else os.urandom(32), served
        self.draw, self.admission, self.coin_source = draw, admission, coin_source
        #: each call's table roots (hex; z None when the run has no exact Y for it) and each call's receipt root
        self.tables: dict[int, dict[str, str | None]] = {}
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
                                        "uncovered": "attention (QK^T and P.V in Int8LutAttentionRow_v1) and lm_head",
                                        "z": "the arm's own when its observe hands one over, else engine.int8_exact.int_mm "
                                             "beside the served scaled_mm"}

    def listen(self, fn: Listener) -> None:
        """Hand every later call to `fn` (`Listener`)."""
        self.listeners.append(fn)

    @property
    def salt(self) -> bytes:
        return self._salt if self._salt is not None else self.open_window()

    def open_window(self) -> bytes:
        """The Program's salt, opened once: round 0's request table is committed (its binding reads no salt) and its root
        handed to ``opener``, which opens the window's session and its salt coin after the window's open root over it
        (`pouw_h1_window.Window.open`); the commitment records the root (``requests_root``) and what ``opener``
        returns of the opening."""
        if self._salt is None:
            if self._unopened is not None:
                raise RuntimeError(f"{self.entry}: the window's salt was refused ({self._unopened}), and a "
                                   "window opens it once")
            table = self._tables[0] = RQ.request_table(None, 0, self.rows, self.admitted.get(0, []), self._key)
            self.commits["requests_root"] = table.commitment.root.hex()
            try:
                salt, self.rounds, self.draw, commits = self._opener(table.commitment.root)
            except Exception as e:
                self._unopened = self.details["unopened"] = str(e)
                raise
            self.commits.update(commits)
            self._salt = salt
        return self._salt

    def _table(self, call: Call, name: str, leaves: list[bytes]) -> str:
        j = list(TABLE_TAGS).index(name)
        return commit_rows(tagged(TABLE_TAGS[name], self.salt, call.index), call.index, leaves, hash=HM96,
                           salt_key=self._key, nonce=RQ.TABLES_NONCE + 3 * call.index + j).commitment.root.hex()

    def _commit_tables(self, call: Call, a: Any, s_a: Any, y: Any, z: Any) -> None:
        """The call's A, y and z tables (`table_leaves`): y's only when it's handed over or the call is empty, z's only when
        the run has the call's exact Y."""
        m, t = call.m, self.rows
        if a is not None and m:
            av, sv = _words(a, m, "i1"), _words(s_a, m, "<f4")
        else:
            av = sv = np.zeros((0, 1), "i1")
        out: dict[str, str | None] = {"a": self._table(call, "a", table_leaves(t, call.k + 4, m, av, sv))}
        out["y"] = None if (y is None and m) else self._table(
            call, "y", table_leaves(t, 2 * call.n, m, _words(y, m, "<i2") if m else av))
        out["z"] = None if (z is None and m) else self._table(
            call, "z", table_leaves(t, 4 * call.n, m, _words(z, m, "<i4") if m else av))
        self.tables[call.index] = out

    def _call(self, s: int, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, y: Any = None, z: Any = None) -> Call:
        n, k = self.shapes_[w]
        call = Call(len(self.calls), s, w, self.modules.get(w, str(w)), int(a.shape[0]) if a is not None else 0, k, n,
                    s * self.u_step + self.offs[w], self.rows)
        self.calls.append(call)
        if self.commit:
            self._commit_tables(call, a, s_a, y, z)
            if self.served is not None:
                self.served.record(call, a, weight)
        for fn in self.listeners:
            out = fn(call, a, s_a, weight, weight_scale)
            if out is not None:
                self.bound[call.index] = {**self.bound.get(call.index, {}), **out}
        self.details["calls"] = self.commits["calls"] = len(self.calls)
        if w == len(self.shapes_) - 1:
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
            self.open_window()
        return out

    def _close_rounds(self, through: int) -> None:
        """Close every round up to `through`, in order: its request and slot tables, then its root, recorded before its coin
        opens. A step whose served rows (linear 0's) aren't its batch's is named in `unbound`."""
        while len(self.round_log) <= min(through, self.steps - 1):
            r = len(self.round_log)
            owners = self.owners.get(r, [])
            served = next((c.m for c in self.calls if c.step == r and c.layer == 0), 0)
            if len(owners) != served:
                self.unbound.append(f"step {r}: {served} rows served and {len(owners)} in the step's batch, so the slot "
                                    "table can't name each served row's request")
            salt = self.salt
            if r not in self._tables:
                self._tables[r] = RQ.request_table(salt, r, self.rows, self.admitted.get(r, []), self._key)
            table = self._tables[r]
            slots = self._slot_tables[r] = RQ.slot_table(self.salt, r, self.rows, owners, self._key)
            calls = RQ.step_digest(self.salt, r, self._rows([c for c in self.calls if c.step == r]))
            receipts = self.receipts_of(r - 1) if r else RQ.NO_RECEIPTS
            root = RQ.round_root(self.salt, r, calls, table.commitment.root, slots.commitment.root, receipts)
            coin = self.rounds.record(root) if self.rounds is not None else None
            self.round_log.append({"round": r, "root": root.hex(), "calls": calls.hex(),
                                   "requests": table.commitment.root.hex(), "slots": slots.commitment.root.hex(),
                                   "receipts": receipts.hex(), "coin": coin.to_json() if coin else None})

    def receipts_of(self, r: int) -> bytes:
        """Round r's receipts (`pouw_h1_requests.receipts_digest`): its calls' receipt roots as handed to `receipt`."""
        return RQ.receipts_digest(self.salt, r, [(c.index, self.receipts.get(c.index)) for c in self.calls if c.step == r])

    def receipt(self, index: int, root: bytes) -> None:
        """Call `index`'s receipt root, computed after its round's coin: it goes under the next round's root (the last
        round's in the draw's root), so it is refused before that coin, once the next round has closed, or twice."""
        call = self.calls[index]
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

    def record(self, w: int, a: Any, s_a: Any, weight: Any, weight_scale: Any, z: Any = None, y: Any = None) -> Call | None:
        """Linear `w`'s served call: its s8 rows a [m, k], their f32 scales s_a [m], its int8 weight [n, k] and scales, its
        exact s32 Y [m, n] if the arm's kernel wrote one (else `int_mm` computes it beside, only to retain it) and its
        served y [m, n] bf16, each committed in the call's tables."""
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
        if z is None and self.retain:
            from verity_vllm.engine.int8_exact import int_mm
            z = int_mm(a, weight)
        call = self._call(self.step, w, a, s_a, weight, weight_scale, y, z)
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
        if not self.commit or self._unopened is not None:
            return
        self.details["steps_served"] = self.step + 1
        self.details["last_step_complete"] = self.last in (None, len(self.shapes_) - 1)
        for s in range(self.step + 1, self.steps):
            for w, (n, k) in enumerate(self.shapes_):
                weight, scale = self._weights.get(w, (None, None))
                a = weight.new_zeros((0, k)) if weight is not None else None
                self._call(s, w, a, scale.reshape(-1).new_zeros(0) if scale is not None else None, weight, scale)
        self._close_rounds(self.steps - 1)
        if self.served is not None:
            self.served.close(self)
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
        self.details["credit"] = self.credit()

    def static(self) -> bool:
        """Every (step, linear) call of the window, in order, at its public position, and nothing refused."""
        want = [(s, w, s * self.u_step + self.offs[w], units(k, n, self.rows))
                for s in range(self.steps) for w, (n, k) in enumerate(self.shapes_)]
        return self.order_ok and not self.refused and [(c.step, c.layer, c.base, c.units) for c in self.calls] == want

    def step_rows(self) -> list[int]:
        """Each step's served rows, its first linear's m (0 for a step with no call)."""
        rows = {c.step: c.m for c in self.calls if c.layer == 0}
        return [rows.get(s, 0) for s in range(self.steps)]

    def utilization(self) -> Fraction:
        return utilization(self.step_rows(), self.rows, self.steps)

    def verdict(self, call: Call) -> dict[str, Any]:
        """The call's credit verdict; a served call recorded outside a traced ``apply`` (`traced`) is refused."""
        return self.verdicts.get(call.index) or {
            "linear": call.module, "credit": False,
            "reason": f"credit: {CG.UNTRACED[1:-1]}: {call.module} call {call.index} was recorded outside a traced apply"}

    def credit(self) -> dict[str, Any]:
        """Credited: the served rows' MACs of the calls the credit gate passes (`credit_gate`), only, and none in a window
        that served a row it can't name an entry for (`request_gaps`) or whose coins aren't the challenger's (`IN_PROCESS`).
        The padded work (empty slots and pad columns) is
        shown beside it, not credited (Daniel, 18 Oct), and so are the refused calls' rows, with each refusal's reason."""
        served = [c for c in self.calls if c.m]
        gaps = self.request_gaps()
        verdicts = {c.index: self.verdict(c) for c in served}
        if gaps:
            verdicts = {i: v if not v["credit"] else {**v, "credit": False, "reason": RQ.UNENTERED} for i, v in verdicts.items()}
        if self.coin_source != CHALLENGER:
            verdicts = {i: v if not v["credit"] else {**v, "credit": False, "reason": IN_PROCESS} for i, v in verdicts.items()}
        ok = [c for c in served if verdicts[c.index]["credit"]]
        no = [c for c in served if not verdicts[c.index]["credit"]]
        computed = sum(padded(c.rows, c.n)[0] * padded(c.rows, c.n)[1] * c.k for c in self.calls)
        all_served = sum(c.m * c.n * c.k for c in served)
        return {"rule": "credit the served rows of calls whose served y is the credited kernel's own output only, and none "
                        "in a window that served a row of a request with no entry or whose coins the auditor's challenger "
                        "didn't draw; padded work and refused calls are shown, not credited", "requests": gaps,
                "coins": self.coin_source,
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

    def _rows(self, calls: Sequence[Call]) -> list[dict[str, Any]]:
        return [{**c.summary(), **({"tables": dict(self.tables[c.index])} if c.index in self.tables else {}),
                 **({"bound": dict(self.bound[c.index])} if c.index in self.bound else {})} for c in calls]

    def manifest(self) -> list[dict[str, Any]]:
        """Each call's summary, with what its listeners bound."""
        return self._rows(self.calls)

    def calls_root(self) -> bytes:
        """SHA-512 over the manifest, in order: what the install commits to (its positions and what was bound), and with
        ``served`` its registered trees' entries."""
        return calls_root(self.salt, self.manifest(), None if self.served is None else self.served.digest())

    def finish(self) -> None:
        """Close the window, commit its manifest, then open ``draw``'s coin after `pouw_h1_requests.draw_root` over the
        manifest, the last round's receipts and every round's root and coin, so the draw follows all of them."""
        self.close()
        if not self.commit or self._unopened is not None:
            return
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


def calls_root(salt: bytes, manifest: Sequence[Mapping[str, Any]], served: bytes | None = None) -> bytes:
    """`H1Run.calls_root` over the manifest rows `manifest`, and the served trees' digest `served` when the window has them."""
    h = hashlib.sha512(b"verity/pouw/h1/calls/v2\0" + salt)
    for row in manifest:
        h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
    return h.digest() if served is None else hashlib.sha512(h.digest() + served).digest()


def bind_all(calls: Sequence[Call], bound: Mapping[int, Mapping[str, Any]]) -> list[tuple[bytes, bytes] | None]:
    """Each call's (root(A), root(B)) from what its listeners bound (``root_a``, ``root_b``, hex), None where either is missing."""
    out: list[tuple[bytes, bytes] | None] = []
    for c in calls:
        b = bound.get(c.index, {})
        out.append((bytes.fromhex(b["root_a"]), bytes.fromhex(b["root_b"])) if "root_a" in b and "root_b" in b else None)
    return out


__all__ = ["CALLS_FILE", "CHALLENGER", "Call", "FILL", "H1Run", "IN_PROCESS", "Listener", "Opener", "RECORD", "SALT_TAG",
           "SCHEME", "Static", "TABLE_TAGS", "TILE", "VllmInt8LinearH1", "bind_all", "calls_root", "check_tiles", "compiled",
           "coordinate", "layout", "numpy_of", "padded", "program", "program_salt", "slots",
           "table_leaves", "units", "utilization", "weight_root"]
