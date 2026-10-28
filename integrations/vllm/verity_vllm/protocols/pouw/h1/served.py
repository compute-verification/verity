"""H1's served tables, committed as registered values of compute accounting's window Program
(`h1window.program`, one tile template per K) as the window serves them, so the Lean verifier opens every row a proof block reads
(`flock-verify --zk --registered`, `h1window.table_reads_file`).

* **The Program** is declared before the window serves (`declared`): every step calls every linear ``(module, K, n)`` in
  order, call (s, w) reading A block ``s·B_K + i`` of its K (`layout`: ``i`` the place of ``blocks[w]``, each linear's
  input within a step, among the blocks its K's linears read, ``B_K`` their number), its A slot M `h1window.slot` of the
  window's T rows, over the served path's R × C tile (`pouw_h1.TILE`, 32 × 32) and the tile unit at each K
  (`h1window.tile_unit`, or ``unit``). Its parameters are named `PREFIX` and their role, one prefix for every window of
  a period, so every window of a period declares one Program, the one its open commits as ``circuit``: the weights'
  registration fixes the weights' root before any window's open root exists, so no window's root can name the Program.
  A window's own values are its through their domains, which bind its open root (`Served.key`: `salt.window_open_root`
  of the period's weights' root, round 0's request root and the session, or the key the run is given, 64 bytes); round
  0's input-ids table, whose root that request root contains, binds the session instead (`Served.ids_tree`).
* **The tables** are the ones a proof block's rows open at (`h1window.table`, `h1window.table_reads`): step s's A table of K
  ``a/<K>@<s>``, the M slot rows of each of step s's calls of K, call after call in call order (`step_calls`;
  the j-th starts at row M·j), each its K int8 codes then its f32 s_a (`h1zy.ROW`; the served rows, then zero codes of
  scale 0), and call c's z table ``zy@<c>``, one record a proof block of the call in its tile order (`h1zy.RECORD`: its cells' s32 z,
  row-major, then their bf16 y; a slot past the served rows is z = 0 and y = +0, dq(0, +0, s_w) for a finite positive
  s_w); and the weights' ``b/<K>``, K's linears' rows linear after linear, each its K weight codes then its f32 s_w (Σn
  rows every step's proof blocks read). With ``receipts`` (BRIDGE2), call c's receipt table ``receipt@<c>`` too: one row a proof block
  of the call in its tile order, the proof block's receipt, the statement's output row o0 (`receipt_words`: its nw = 2 (KS + m)
  u32 words, each little-endian, so 2 nw u16 words), its domain bound to the open root alike, and listed whole under
  round s's root (`segments`). A proof block's seeds take its call's roots, root(A) its step's ``a/<K>@<s>`` and root(B)
  ``b/<K>``, so a step's receipt tables of K are registered at its last call of K, once both are. Call c's public
  table ``pub@<c>`` holds its statement's public row ``pub`` (`h1window.PUBS`): row t proof block u0 + t's,
  `core_check.public_words` of the window's Program salt, u0 + t and (root(``a/<K>@<s>``), root(``b/<K>``)), salted
  `core_check.public_row_salt` of the open root and u0 + t, so anyone rebuilds it, as credit's `registered_file` and
  Lean's ``registeredFile`` do; its rows are kept. Those are the salt, position and roots the proof block's receipt is seeded
  from. At `close` the window commits its ok rows ``ok`` (`Served.ok`, the statement's output row o1): row u
  `core_check.OK_ROW` under `core_check.ok_salt` of the open root and u, in no entry, since credit registers them itself
  and refuses a window that does.
* **The weights are the period's** (`Weights`): each K's ``b/<K>`` is registered once a period, before its first
  window's salt (`register_weights`), under the period's declared Program, binding no window, its salts fixed for the
  registration. `credit.weights_root` of their roots is the weights' root every window's open root binds and the
  period-open signs, so a window can't serve other codes or s_w than the period registered: every window registers the
  same entries, and `Served.commit` refuses a call whose weight isn't its linear's registered rows.
* **The salts are the auditor's**: the auditor registers the served model's ``b/<K>`` from its published
  checkpoint under its own salt key (infra's ``model-register``, core's `weights.weights_roots`), and a period opens with
  ``period-open --model M``, which signs that registration's weights' root W. The prover reads the key with its gate key
  (``model-key --model M --circuit C``, `model_key`) and registers the same rows under it (`Served`'s ``registry``), so
  its ``b/<K>`` are the registration's openings, and a window whose served rows fold to a root other than W is refused.
* **Rows** are ``hm96-sha512/row/v2`` bit rows of 16-bit words (`V2`, `row_words`, `record_words`), committed as
  `registered.Registered.commit` commits them, from their NumPy words (`pouw_h1_registered.commit`), under the
  registered-value domain of the Program's digest, the value, its words and rows, and for a call's tables the window's
  open root (`registered.domain`'s ``window``, `Served.root`), row j's salt `audit.hm96_salt` of the run's key at
  the value's nonce (`nonce`). A window keeps its trees, not its rows, which the prover rebuilds from its own.
* **Chunk rows** (SPLITK). Past K = 4096 an A or B row of K is n = `weights.chunks` (K) leaves (6144 → 2, 14336 → 4):
  row r's leaf n·r + c is its codes [c·K/n, (c + 1)·K/n) then its scale (`chunk_words`, `h1zy.ROW` (K/n)), so
  ``a/<K>@<s>`` is n·M leaves a call and ``b/<K>`` n·Σn, leaf j salted at j, as `h1window.program`'s chunked
  parameters and credit's `table_rows` and `_weights` count them. The Commit's AROW (`Served.a_words`) is the same
  leaves. At K <= 4096 a row is one leaf, as before.
* **When.** `Served.commit` registers call c's z table as `H1Run` records the call, step s's A table of K at its last
  call of K (salted from its first call's nonce), and puts the period's ``b/<K>`` in the call that completes K's weights
  (the last of K's linears in step 0), then, once both roots exist, each of the step's calls of K's ``pub@<c>``
  (`Served._pubs`) and, with ``receipts``, its receipt table. Each root goes into that call's manifest row
  (``tables.registered``), so its
  round's root commits it before the round's coin, and round s's root also lists step s's A tables whole
  (`Served.segments`, `credit.round_root`). A weight is taken once, and one that changes within the window
  is refused. Without the period's `Weights` (an in-process window, which credits nothing) the window registers its own
  ``b/<K>`` there, under the run's key.
* **The Commit** (ROOTS1): with ``tap`` (`pouw_h1_commit.CommitTap`, the Commit in the serving engine) its
  AROW of each step's calls of K is that step's A table's value (`credit.commit_rows`), and the window also registers each round's input-ids table ``ids@<s>`` (`ids_words`: its M slots' token ids, two words each),
  outside the Program's values (`ids`, in `entries`), and lists it whole under round s's root (`segments`). Round 0's
  is its prompts' tokens in step 0's batch order, registered before the window opens (`open_ids`), its root inside
  round 0's request root and its domain bound to the session (`ids_tree`); each later round's
  is the Commit's ids of its step (`close_round`). The Commit registers the window's own tree as its
  value of the same rows when its rows are the same (`a_tree`, `ids_tree`, `pouw_h1_commit.CommitTap`). With the tap, the
  window's round tables are the live proof block's registered values too (`pouw_h1_live`, `live`, in `entries`): round 0's
  request headers ``requests@0`` and each round's slot table ``slots@<s>``, listed whole under round s's root.
* **Refused** (no table from then on, and `trees` raises the reason): a window with no key of 64 bytes, blocks that aren't
  one per linear, a call off the declared Program's order, a served call with no z/y record, a weight that isn't the
  linear's n × K int8 codes and n scales or that changes, an odd K, what `h1window.program` refuses, served weights that
  aren't the auditor's registration, and at `close` a window that isn't static or a table the Program reads that was
  never registered.
* **Statements.** A Lean ``--registered`` port holds one root, so a statement holds one call's proof blocks, and its file is
  `reads_file` of that call: `h1window.table_reads_file`, the verifier's own reads from its Program, each call table's
  domain bound to the window's open root and the weights' to none, as their trees were. `register` holds the window's
  trees to an earlier record of the log under the same domain (R7-held) and writes each call's file (`file_name`).
* `descriptor` is a declared Program's descriptor bytes, the period's ``circuit`` (their SHA-512) and the window
  evidence's ``descriptor``, from which core's credit reads the layout it draws from (`credit.layout_of`);
  `declared_layout` is that layout without building the Program.
"""
from __future__ import annotations

import functools
import gc
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as CR
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import weights as Wt
from verity.core.protocols.pouw.circuit import h1device as HD
from verity.core.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registered as G
from verity.ml.circuits.codec import canonical_json, descriptor_chunks, encode_program
from verity.ml.circuits.program import Program

from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import live as LV
from verity_vllm.protocols.pouw.h1 import registered as NR
from verity_vllm.protocols.pouw.h1 import requests as RQ
from verity_vllm.protocols.pouw.h1.served_files import FILE, Registration, file_name, reads_file, register, trees
from verity_vllm.protocols.pouw.h1.words import (EMPTY_ID, chunk_words, ids_words, receipt_words, record_words,
                                                         row_words, zy_words)

#: the rows' format: the tile's port rows
V2 = G.SCHEMA_V2
#: the window Program's parameter prefix, the same for every window of a period
PREFIX = "h1/"
#: the hm96 salt stream's first nonce of the window's registered tables, past every round's and call's tables
NONCE = RQ.REQUESTS_NONCE + (1 << 42)
#: ``b/<K>``'s nonces, past every call's two tables: the auditor's registration's (`weights.B_NONCE`)
B_NONCE = Wt.B_NONCE
#: round s's input-ids table's nonce is IDS_NONCE + s, past every ``b/<K>``'s
IDS_NONCE = B_NONCE + (1 << 39)
#: call c's receipt table's nonce is RECEIPT_NONCE + c, past every round's input-ids table's
RECEIPT_NONCE = IDS_NONCE + (1 << 38)
#: the live tables' nonces (`pouw_h1_live.nonce`): requests@0's LIVE_NONCE, round s's slots@<s>'s LIVE_NONCE + 1 + s
LIVE_NONCE = RECEIPT_NONCE + (1 << 37)
#: the session round 0's input-ids table binds under in-process coins, where no gate reserved one (such a window credits
#: nothing)
LOCAL_SESSION = "in-process"
DIGEST_TAG = b"verity/pouw/h1/served/v3\0"


def nonce(call: int, table: str) -> int:
    """The salt nonce of call ``call``'s table ``table`` (`h1window.TABLES`): a step's A table of K takes its first call of
    K's ``a`` nonce, which no other table takes."""
    return NONCE + 2 * int(call) + HW.TABLES.index(table)


def layout(shapes: Sequence[tuple[int, int]], calls: Sequence[tuple[int, int]],
           blocks: Sequence[int] | None = None) -> list[int]:
    """Each call's A block of its K, ``calls`` each call's (step, linear) and ``blocks`` each linear's input block within a
    step (`Served`'s): call (s, w) reads block s·B_K + i, i the place of ``blocks[w]`` among the blocks K's linears read."""
    per = list(blocks) if blocks is not None else list(range(len(shapes)))
    if len(per) != len(shapes):
        raise ValueError(f"blocks {per} are not one input block for each of the window's {len(shapes)} linears")
    of_k: dict[int, list[int]] = {}
    for w, (_n, k) in enumerate(shapes):
        of_k.setdefault(k, [])
        if per[w] not in of_k[k]:
            of_k[k].append(per[w])
    of_k = {k: sorted(b) for k, b in of_k.items()}
    return [s * len(of_k[shapes[w][1]]) + of_k[shapes[w][1]].index(per[w]) for s, w in calls]


def card_arch(device: Any = None) -> str:
    """The architecture of the CUDA card ``device`` (the current one when None) as H1's device records name it
    (`h1device.DEVICES`): an H100 or H200's, compute capability 9.0, or the RTX PRO 6000's, 12.0. ValueError on a card
    with no record."""
    import torch

    major, minor = torch.cuda.get_device_capability(device)
    return HD.device(f"sm_{major}{minor}").arch


def declared(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
             tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None, prefix: str = PREFIX,
             unit: Callable[[int], Any] | None = None, arch: str | None = None) -> Program:
    """The window Program a static window declares before it serves: ``steps`` steps, each calling every linear (``shapes``,
    each ``(n, K)``, named by ``modules``) in order, laid out by `layout`, its A slot for ``rows`` token rows, its
    parameters under ``prefix`` (`PREFIX`, which names no window): one Program for every window of the declaration, the
    period's ``--circuit`` (its `program_digest`) and each window's `Served.program`. Without ``unit``, each K's tile
    unit is `h1window.tile_unit`'s on ``arch`` (`card_arch`; sm_120 when None), which refuses a K outside that card's
    record's domain (ValueError); sm_90's Program is sm_120's byte for byte."""
    lay = declared_layout(shapes, modules, rows, steps, tile, blocks)
    if unit is None and arch is not None:
        unit = functools.partial(HW.tile_unit, tile=tuple(tile), arch=arch)
    return HW.program([tuple(x) for x in lay["linears"]], [tuple(x) for x in lay["calls"]], lay["m"], tile, prefix, unit)


def unit_of(name: str | None, tile: tuple[int, int] = PH.TILE) -> Callable[[int], Any] | None:
    """`declared`'s ``unit`` for the pouw setting ``unit`` ``name`` (`pouw_h1_settings.UNITS`): None for v1 (unset), the
    default; for ``j10r`` `h1window.tile_unit` over J10-R's `h1tile.PouwH1Tile5`, one noise stream a call, the tile a
    period's guarantee covers (core's `period_check.COVERED_TILE`)."""
    if name in (None, "v1"):
        return None
    if name == "j10r":
        return functools.partial(HW.tile_unit, tile=tuple(tile), definition=HT.PouwH1Tile5)
    raise ValueError(f"no tile unit {name!r}: the pouw setting 'unit' is v1 or j10r")


def descriptor(program: Program) -> bytes:
    """`program`'s descriptor as the period commits it: its canonical JSON without annotations, the bytes whose SHA-512
    is its `codec.program_digest` (the period-open's ``circuit``) and from which core's credit reads the window's
    layout (`credit.layout_of`)."""
    return canonical_json({k: v for k, v in encode_program(program).items() if k != "annotations"})


def write_program(program: Program, layout: Mapping[str, Any], out: Path | str, *, rows: int, steps: int) -> str:
    """`program`'s `codec.program_digest`, streamed once into ``out``'s `pouw_h1.PROGRAM` (its `descriptor`, gzipped) as it
    is hashed, and its `pouw_h1.PROGRAM_META`: the digest, the descriptor's bytes, its `layout` (`declared_layout`, which
    core's `credit.layout_of` reads from it), T ``rows`` and S ``steps``. A period-open commits the digest and the circuit
    files built from these (``served_verdicts.py period``)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    h, size, on = hashlib.sha512(), 0, gc.isenabled()
    gc.disable()
    try:
        with gzip.open(out / PH.PROGRAM, "wb", compresslevel=1) as f:
            for chunk in descriptor_chunks(program):
                h.update(chunk)
                f.write(chunk)
                size += len(chunk)
    finally:
        if on:
            gc.enable()
    digest = h.hexdigest()
    meta = {"program_digest": digest, "file_sha512": digest, "bytes": size, "rows": int(rows), "steps": int(steps),
            "layout": dict(layout)}
    (out / PH.PROGRAM_META).write_text(json.dumps(meta, indent=1) + "\n")
    return digest


def declared_layout(shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
                    tile: tuple[int, int] = PH.TILE, blocks: Sequence[int] | None = None) -> dict[str, Any]:
    """`declared`'s arguments to `h1window.program` as the window's layout, ``{calls: [[linear, A block], ...],
    linears: [[name, K, n], ...], m, tile}``: what core's `credit.layout_of` reads from `declared`'s `descriptor`, built
    without the Program (the proof system sizes the period's K from it, `pouw_h1_challenger.draw_k`)."""
    mods = [modules.get(w, str(w)) for w in range(len(shapes))]
    calls = [(s, w) for s in range(steps) for w in range(len(shapes))]
    return {"calls": [[mods[w], b] for (_s, w), b in zip(calls, layout(shapes, calls, blocks))],
            "linears": [[mods[w], k, n] for w, (n, k) in enumerate(shapes)], "m": HW.slot(rows, tile), "tile": list(tile)}


def weights_of(layers: Sequence[Any]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each served linear's B as the period registers it, by its index among `layers` (the int8 arm's linear layers, as
    `H1Run` numbers them): its int8 codes ``(n, K)`` and its f32 s_w ``(n,)``."""
    return {w: (PH.numpy_of(m.weight), PH.numpy_of(m.weight_scale).astype("<f4").reshape(-1)) for w, m in enumerate(layers)}


@dataclass(frozen=True)
class Weights:
    """A period's registered weights: each K's ``<prefix>b/<K>``, the B rows of K's linears in the window's
    linear order, each row its K int8 codes then its f32 s_w as `weights.chunks` (K) = n leaves (`chunk_words`; one
    leaf, `row_words`, at K <= 4096), registered once (`register_weights`) under the
    period's declared Program (``circuit``, its `program_digest`), binding no window, its salts fixed for the
    registration. ``at`` gives each linear's K and first row in its ``b/<K>``, whose first leaf is n times it. Every window of the period registers these
    same entries, and `root`, `credit.weights_root` of their roots, is the weights' root every window's open root binds
    and the period-open signs (``--model``'s registration, `ModelKey`)."""

    circuit: str
    registered: dict[int, G.Registered]
    at: dict[int, tuple[int, int]]

    def root(self) -> bytes:
        return CR.weights_root({k: r.root for k, r in self.registered.items()})

    def entries(self) -> list[dict[str, Any]]:
        return [self.registered[k].entry() for k in sorted(self.registered)]

    def matches(self, w: int, codes: np.ndarray, scales: np.ndarray) -> bool:
        """Whether linear `w`'s served codes and s_w are its registered rows: its n leaves a row from leaf n·lo."""
        k, lo = self.at[w]
        n, reg = Wt.chunks(k), self.registered[k]
        got = chunk_words(codes, scales, n)
        return np.array_equal(got, np.asarray(reg.rows[n * lo:n * lo + len(got)], np.uint16))


def register_weights(circuit: str, shapes: Sequence[tuple[int, int]], weights: Mapping[int, tuple[Any, Any]], key: bytes,
                     prefix: str = PREFIX) -> Weights:
    """The period's `Weights`: for each K, ascending, ``<prefix>b/<K>`` over K's linears' rows (``shapes`` each linear's
    ``(n, K)``, in the window's order; ``weights`` each linear's int8 codes ``(n, K)`` and f32 s_w ``(n,)``), each row
    `weights.chunks` (K) leaves (`chunk_words`), committed by
    `pouw_h1_registered.commit` under the declared Program's digest ``circuit`` with no window, leaf j's salt
    `audit.hm96_salt` of the period's key ``key`` at ``B_NONCE`` plus K's place, its rows kept (`Weights.matches`).
    Committed once, before the period's first window's salt. ValueError on an odd K, a K `weights.chunks` refuses, or a
    weight that isn't its linear's n × K int8 codes and n scales."""
    ks, reg, at = sorted({int(k) for _n, k in shapes}), {}, {}
    for i, k in enumerate(ks):
        if k % 2:
            raise ValueError(f"K = {k}: a row's codes are whole u16 words")
        x, rows, lo = Wt.chunks(k), [], 0
        for w, (n, kw) in enumerate(shapes):
            if kw != k:
                continue
            codes, scales = (np.asarray(x) for x in weights[w])
            if codes.dtype != np.int8 or codes.shape != (n, k) or scales.reshape(-1).shape != (n,):
                raise ValueError(f"linear {w}'s weight is not its {n} × {k} int8 codes and {n} scales")
            rows.append(chunk_words(codes, scales.astype("<f4").reshape(-1), x))
            at[w], lo = (k, lo), lo + n
        words = np.concatenate(rows)
        reg[k] = NR.commit(circuit, f"{prefix}b/{k}", words, key, B_NONCE + i, keep_rows=True)
    return Weights(circuit, reg, at)


@dataclass(frozen=True)
class ModelKey:
    """The auditor's registration of the served model under the period's circuit, as the gate's ``model-key --model M
    --circuit C`` answers a lane's gate key: its role ``model``, ``circuit``, the weights' root ``weights_root`` (hex) that
    ``period-open --model M`` signs, and ``key``, the 32-byte salt key its ``b/<K>`` rows were committed under."""

    model: str
    circuit: str
    weights_root: str
    key: bytes = field(repr=False)


def _hex512(v: Any) -> bool:
    return type(v) is str and len(v) == 128 and all(c in "0123456789abcdef" for c in v)


def model_key(answer: Mapping[str, Any]) -> ModelKey:
    """The gate's ``model-key`` answer as a `ModelKey`; ValueError on a refusal, or an answer whose key isn't 32 bytes
    whose SHA-512 is its ``key_sha512``, or whose circuit or weights' root isn't a SHA-512 digest."""
    if "refused" in answer:
        raise ValueError(f"model-key: refused: {answer['refused']}")
    try:
        key = bytes.fromhex(answer["key"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("model-key: the answer has no key in hex") from None
    if len(key) != 32 or hashlib.sha512(key).hexdigest() != answer.get("key_sha512"):
        raise ValueError("model-key: the key is not 32 bytes whose SHA-512 is the answer's key_sha512")
    if not (type(answer.get("model")) is str and _hex512(answer.get("circuit")) and _hex512(answer.get("weights_root"))):
        raise ValueError("model-key: the answer names no model, circuit and weights' root (SHA-512 digests in hex)")
    return ModelKey(answer["model"], answer["circuit"], answer["weights_root"], key)


@dataclass
class Served:
    """How an `H1Run` commits its calls' tables and its weights as the window Program's registered values. ``tile``: the
    Program's R × C (`pouw_h1.TILE`). ``unit(k)``: the tile unit at contraction k (default `h1window.tile_unit` on
    ``arch``, the serving card's architecture, `card_arch`; a stand-in of its parameter shape in tests). ``blocks``: each
    linear's input block within a step (default each its own; ``(0, 0, 0, 1)`` for q, k, v and o). ``window``: the window's key, else the window-open root its commitment names.
    ``weights``: the period's registered `Weights`, or ``registry`` the auditor's registration of the served model
    (`ModelKey`, the gate's ``model-key``), whose key registers them from the served model at install (`weights_root`,
    from `pouw._install_h1`) and whose root they must fold to; without either the window registers its own ``b/<K>``.
    ``program``: the period's declared Program (`declared` under ``prefix``), given or declared at the first call, and
    ``program_digest`` its digest, given with it or computed then. From the first call: ``root`` (the window's open root
    its tables' domains bind, hex), ``linears``, ``calls`` and ``m`` (the Program's); ``registered`` (each value's
    `registered.Registered`, in registration order) and ``refused`` (why the window has no trees)."""

    tile: tuple[int, int] = PH.TILE
    unit: Callable[[int], Any] | None = None
    blocks: Sequence[int] | None = None
    window: bytes | None = None
    prover: str | None = None
    prefix: str = PREFIX
    weights: Weights | None = None
    registry: ModelKey | None = None
    program: Program | None = None
    program_digest: str | None = None
    root: str | None = None
    registered: dict[str, G.Registered] = field(default_factory=dict)
    linears: list[tuple[str, int, int]] = field(default_factory=list)
    calls: list[tuple[str, int]] = field(default_factory=list)
    m: int = 0
    refused: list[str] = field(default_factory=list)
    #: the Commit's tap in the serving engine (`pouw_h1_commit.CommitTap`)
    tap: Any = None
    #: each call's receipts, one a proof block in its tile order (`receipt_words`' input), from ``(call, codes, s_a, weight
    #: codes, s_w)``: with it, each call's ``receipt@<c>`` is registered (BRIDGE2) at its step's last call of its K, once
    #: the roots its proof blocks' seeds take, its step's ``a/<K>@<s>`` and ``b/<K>``, are registered; one with a ``prepare`` is
    #: first handed those calls together (`_prepare`); one with a ``stage`` is handed each served call and its rows ``a``
    #: where they are on a CUDA device, before they're read to the host (it copies them and keeps no reference to ``a``,
    #: which the engine reuses; a ValueError refuses the window)
    receipts: Callable[[Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None
    #: the serving card's architecture (`card_arch`), whose H1 record bounds each K's tile unit; sm_120 when None
    arch: str | None = None
    #: with ``receipts``, each open step A table's calls, their served codes and s_a, until their step's last call of K
    _held: dict[str, list[tuple[Any, np.ndarray, np.ndarray]]] = field(default_factory=dict)
    #: the window's ok rows ``<prefix>ok`` (`h1window.OK_PORT`'s, o1), from `close`: in no entry, as credit registers them
    ok: G.Registered | None = None
    #: each round's input-ids table and its words, with the tap
    ids: dict[int, G.Registered] = field(default_factory=dict)
    _ids_words: dict[int, np.ndarray] = field(default_factory=dict)
    #: with the tap, the live proof block's tables by value (`pouw_h1_live`): requests@0, then each round's slots@<s>
    live: dict[str, G.Registered] = field(default_factory=dict)
    #: each open step A table's words so far, one block a call recorded, by call (they may come out of call order)
    _a: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    #: each linear's weight, the tensor and its codes, taken once, and its scales
    _weights: dict[int, tuple[Any, np.ndarray]] = field(default_factory=dict)
    _scales: dict[int, np.ndarray] = field(default_factory=dict)

    def key(self, commits: Mapping[str, Any]) -> bytes:
        if self.window is not None:
            return self.window
        session = commits.get("session") or {}
        need = (commits.get("weights_root"), commits.get("requests_root"), session.get("session"))
        if None in need:
            raise ValueError("the window has no key: neither one given nor a window-open root (its weights' root, round 0's "
                             "request table and the session)")
        return PS.window_open_root(bytes.fromhex(need[0]), bytes.fromhex(need[1]), need[2])

    def declare(self, shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int) -> str:
        """The period's declared Program (`declared` under ``prefix``) and its digest, each unless given (the period's,
        declared once for all its windows: at FILL the digest alone takes minutes and tens of GB). ValueError on an odd K
        or what `h1window.program` refuses."""
        if any(k % 2 for _n, k in shapes):
            raise ValueError(f"K {sorted({k for _n, k in shapes})}: a row's codes are whole u16 words")
        if self.program is None:
            self.program, self.program_digest = declared(shapes, modules, rows, steps, self.tile, self.blocks,
                                                         self.prefix, self.unit, self.arch), None
        if self.program_digest is None:
            self.program_digest = P.program_digest(self.program)
        S = self.program.fn.S()
        self.linears, self.calls, self.m = [tuple(x) for x in S.LINEARS], [tuple(x) for x in S.CALLS], int(S.M)
        return self.program_digest

    def weights_root(self, shapes: Sequence[tuple[int, int]], modules: Mapping[int, str], rows: int, steps: int,
                     layers: Sequence[Any], register: bool = True) -> bytes | None:
        """The period's weights' root (`Weights.root`, the one ``period-open --model`` signs): ``weights``', or, with only
        ``registry`` and ``register``, registered now from the served ``layers`` under the declared Program with the
        registration's key, before any salt; else None. With ``registry``, the window is refused
        unless they are its registration: under its circuit, folding to its weights' root."""
        if self.weights is None and self.registry is not None and register:
            self.weights = register_weights(self.declare(shapes, modules, rows, steps), shapes, weights_of(layers),
                                            self.registry.key, self.prefix)
        if self.weights is None:
            return None
        reg = self.registry
        if reg is not None and (self.weights.circuit, self.weights.root().hex()) != (reg.circuit, reg.weights_root):
            self.refused.append(f"the served b/<K> are not the auditor's registration of {reg.model} (weights' root "
                                f"{reg.weights_root[:16]}... under circuit {reg.circuit[:16]}...): they fold to "
                                f"{self.weights.root().hex()[:16]}... under circuit {self.weights.circuit[:16]}..., so "
                                "the window serves other weights than the period's open signed")
        return self.weights.root()

    def open(self, run: Any) -> None:
        """The window's open root, from its key, and the period's Program, at its first call (after its salt opened)."""
        try:
            self.root = G.window_root(self.key(run.commits).hex())
            digest = self.declare(run.shapes_, run.modules, run.rows, run.steps)
        except ValueError as e:
            self.refused.append(str(e))
            return
        if self.weights is not None and self.weights.circuit != digest:
            self.refused.append("the period's registered weights are under another Program than the window's: register "
                                "each K's b/<K> under the period's declared Program")

    def _tree(self, run: Any, value: str, words: np.ndarray, first: int, window: str | None,
              session: str | None = None, row_salts: bytes | None = None) -> G.Registered:
        return NR.commit(self.program_digest, value, words, run._key, first, row_salts is not None, row_salts,
                         window=window, session=session)

    def _register(self, run: Any, value: str, words: np.ndarray, first: int, window: str | None,
                  row_salts: bytes | None = None) -> str:
        reg = self.registered[value] = self._tree(run, value, words, first, window, row_salts=row_salts)
        return reg.root.hex()

    def a_words(self, call: Any, codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Call ``call``'s A table's words over the Program's M slots: its rows' int8 ``codes`` [m, K] and f32 ``scales``
        [m], then zero codes of scale 0, each row `weights.chunks` (K) = n leaves (`chunk_words`, n·M leaves)."""
        full, sc = np.zeros((self.m, call.k), np.int8), np.zeros(self.m, "<f4")
        full[:len(codes)], sc[:len(codes)] = codes, scales
        return chunk_words(full, sc, Wt.chunks(call.k))

    @property
    def steps(self) -> int:
        """The declared Program's steps: its calls, every linear once a step."""
        return len(self.calls) // max(1, len(self.linears))

    def step_calls(self, c: int) -> list[int]:
        """Call ``c``'s step's calls of its K, in call order: the calls its step's A table holds, M rows each, the j-th
        from row M·j (`credit.step_tables`, `h1window.step_of`)."""
        per, ks = len(self.linears), {name: k for name, k, _n in self.linears}
        s, k = c // per, ks[self.calls[c][0]]
        return [d for d in range(s * per, (s + 1) * per) if ks[self.calls[d][0]] == k]

    def a_table(self, c: int) -> str:
        """Call ``c``'s step's A table of its K, ``<prefix>a/<K>@<s>`` (`h1window.table`)."""
        return HW.table(self.program, "a", c, self.steps)

    def receipt_table(self, c: int) -> str:
        """Call ``c``'s receipt table, ``<prefix>receipt@<c>`` (`h1window.table`'s `h1window.RECEIPTS`)."""
        return HW.table(self.program, HW.RECEIPTS, int(c))

    def pub_table(self, c: int) -> str:
        """Call ``c``'s public table, ``<prefix>pub@<c>`` (`h1window.table`'s `h1window.PUBS`)."""
        return HW.table(self.program, HW.PUBS, int(c))

    def receipt_shape(self, c: int) -> tuple[int, int]:
        """Call ``c``'s receipt table's rows and u16 words a row: one row a proof block of the call, its tile unit's ``receipt``
        (the statement's output row o0, `h1window.RECEIPT_PORT`), as a ``row/v2`` row of its bits; 0 words when they
        are no whole u16 words, which `h1window.table_reads_file` refuses to read."""
        S = self.program.fn.S()
        k = {name: k for name, k, _n in S.LINEARS}[S.CALLS[c][0]]
        bits = sum(dict(dict(S.TILES)[k].signature()[1].fields)[HW.OUTPUTS[0]].leaf_widths())
        return HW.call_units(self.program)[c][1], bits // 16 if bits % 16 == 0 else 0

    def a_tree(self, run: Any, call: Any, words: np.ndarray) -> G.Registered:
        """The tree of call ``call``'s step's A table ``a/<K>@<s>`` over ``words`` (its step's calls of K's `a_words`,
        call after call) as `commit` would register it, under the same domain and salts, unregistered
        (`pouw_h1_commit.CommitTap.arow`)."""
        return self._tree(run, self.a_table(call.index), words, nonce(self.step_calls(call.index)[0], "a"), self.root)

    def ids_tree(self, run: Any, s: int, words: np.ndarray) -> G.Registered:
        """The tree of round s's input-ids table ``<PREFIX>ids@<s>`` over ``words`` (`ids_words`), unregistered. Its domain
        binds the window's open root as a call's tables' do, except round 0's: the open root contains its root
        (`credit.requests_root`), so it binds the window's session (`credit.domain_of`; `LOCAL_SESSION` under
        in-process coins, which credit nothing)."""
        value = self.program.fn.bindings["PREFIX"] + f"ids@{s}"
        if s == 0:
            return self._tree(run, value, words, IDS_NONCE, None, run.session or LOCAL_SESSION)
        return self._tree(run, value, words, IDS_NONCE + s, self.root)

    def _register_ids(self, run: Any, s: int, ids: Sequence[int | None]) -> G.Registered:
        words = self._ids_words[s] = self.id_words(ids)
        reg = self.ids[s] = self.ids_tree(run, s, words)
        return reg

    def ids_of(self, s: int) -> np.ndarray | None:
        return self._ids_words.get(s)

    def id_words(self, ids: Sequence[int | None]) -> np.ndarray:
        return ids_words(ids, self.m)

    def open_ids(self, run: Any, ids: Sequence[int | None]) -> bytes | None:
        """With the tap, round 0's input-ids table, registered before the window opens (`pouw_h1.H1Run.open_window`, which
        puts its root, returned, inside round 0's request root): ``ids``, step 0's prompt tokens in its batch order
        (`pouw_h1_requests.first_ids`), under the declared Program. None without the tap or once refused."""
        if self.tap is None or self.refused:
            return None
        try:
            self.declare(run.shapes_, run.modules, run.rows, run.steps)
            return self._register_ids(run, 0, ids).root
        except ValueError as e:
            self.refused.append(f"round 0's input ids: {e}")
            return None

    def close_round(self, run: Any, r: int) -> None:
        """With the tap, at round r's close before its root: round r's input-ids table (after round 0, the Commit's ids
        of its step), and the Commit's value of it (`pouw_h1_commit.CommitTap.input_ids`)."""
        if self.tap is None or self.refused or self.program is None:
            return
        ids = self.tap.ids_at(r)
        if r:
            try:
                self._register_ids(run, r, ids)
            except ValueError as e:
                self.refused.append(f"round {r}'s input ids: {e}")
                return
        self.tap.input_ids(run, self, r, ids)

    def live_table(self, run: Any, name: str, r: int) -> LV.Table | None:
        """With the tap, round r's live request (round 0's) or slot table, registered now (`pouw_h1_live.table`)."""
        return LV.table(self, run, name, r, LIVE_NONCE, run.session or LOCAL_SESSION)

    def _weight(self, call: Any, weight: Any, scale: Any) -> None:
        held = self._weights.get(call.layer)
        if weight is None or (held is not None and held[0] is weight):
            return
        w, s = PH.numpy_of(weight), None if scale is None else PH.numpy_of(scale).astype("<f4").reshape(-1)
        if w.dtype != np.int8 or w.shape != (call.n, call.k) or s is None or s.shape != (call.n,):
            self.refused.append(f"call {call.index}: its weight is not its {call.n} × {call.k} int8 codes and {call.n} scales")
        elif held is None:
            self._weights[call.layer], self._scales[call.layer] = (weight, w), s
        elif not (np.array_equal(held[1], w) and np.array_equal(self._scales[call.layer], s)):
            self.refused.append(f"call {call.index}: linear {call.layer}'s weight changed within the window")

    def commit(self, run: Any, call: Any, a: Any, s_a: Any, weight: Any, weight_scale: Any,
               records: np.ndarray | None) -> dict[str, str]:
        """Register call ``call``'s z table (its z/y ``records`` over its T slots, `pouw_h1_records`', split back to z and
        y and laid out in the Program's tile), add its A rows (its served rows ``a`` [m, K] and scales ``s_a`` [m]) to its
        step's A table of K at its rows in call order, registered whole once the step's calls of K are all in (in
        whichever order its forward ran them: its last call of K), and take ``b/<K>`` once K's linears' weights are all
        held (`_take_b`); with ``receipts``, at the step's last call of K, each of the step's calls of K's receipt
        table, in call order (`_receipts` of the call's own codes and s_a): ``{value: root}``, what the call's manifest
        row commits under its round's root. Nothing once the window is refused."""
        if self.root is None and not self.refused:
            self.open(run)
        if not self.refused:
            self._weight(call, weight, weight_scale)
        if not self.refused and (call.index >= len(self.calls) or call.index != call.step * len(self.linears) + call.layer):
            self.refused.append(f"call {call.index} (step {call.step}, linear {call.layer}) is off the declared Program's "
                                "order")
        if not self.refused and call.m and records is None:
            self.refused.append(f"call {call.index}: no z/y record (its arm handed over no scaled_mm_zy records)")
        if self.refused:
            return {}
        codes, scales = np.zeros((0, call.k), np.int8), np.zeros(0, "<f4")
        zy_rows = None
        if call.m:
            stage = getattr(self.receipts, "stage", None)
            if stage is not None and getattr(a, "is_cuda", False):
                try:
                    stage(call, a)
                except ValueError as e:
                    self.refused.append(f"call {call.index}: {e}")
                    return {}
            codes, scales = PH.numpy_of(a).reshape(call.m, call.k), PH.numpy_of(s_a).reshape(-1)
            zy_rows = zy_words(records, call.rows, call.n, self.m, self.tile)
        else:
            zy_rows = record_words(np.zeros((0, call.n), "<i4"), np.zeros((0, call.n), np.int16), self.m, self.tile)
        out, value, mine = {}, self.a_table(call.index), self.step_calls(call.index)
        rows = self._a.setdefault(value, {})
        rows[call.index] = self.a_words(call, codes, scales)
        whole = None
        if len(rows) == len(mine):
            whole = np.concatenate([b for _c, b in sorted(self._a.pop(value).items())])
            out[value] = self._register(run, value, whole, nonce(mine[0], "a"), self.root)
        zy = HW.table(self.program, "zy", call.index)
        out[zy] = self._register(run, zy, zy_rows, nonce(call.index, "zy"), self.root)
        if self.tap is not None:
            self.tap.arow(run, self, call, value, whole)
        b = self.program.fn.bindings["PREFIX"] + f"b/{call.k}"
        if b not in self.registered and not self._take_b(run, call, b, out):
            return {}
        if whole is not None and b in self.registered:
            out.update(self._pubs(run, mine, value, b))
        if self.receipts is not None:
            self._held.setdefault(value, []).append((call, codes, scales))
            if whole is not None:
                held = sorted(self._held.pop(value), key=lambda x: x[0].index)
                if not self._prepare(held):
                    return {}
                for c, cc, sc in held:
                    got = self._receipts(c, cc, sc)
                    if got is None:
                        return {}
                    rc = self.receipt_table(c.index)
                    out[rc] = self._register(run, rc, got, RECEIPT_NONCE + c.index, self.root)
        return out

    def _take_b(self, run: Any, call: Any, b: str, out: dict[str, str]) -> bool:
        """Put ``b/<K>`` in ``out`` once K's linears' weights are all held, the period's or the window's own; False when
        the window is refused for them."""
        mine, ks = [w for w, (_n, k) in enumerate(run.shapes_) if k == call.k], sorted({k for _n, k in run.shapes_})
        if not all(w in self._weights for w in mine):
            return True
        if self.weights is None:
            words = np.concatenate([chunk_words(self._weights[w][1], self._scales[w], Wt.chunks(call.k)) for w in mine])
            out[b] = self._register(run, b, words, B_NONCE + ks.index(call.k), None)
        elif bad := [w for w in mine if not self.weights.matches(w, self._weights[w][1], self._scales[w])]:
            self.refused.append(f"call {call.index}: linears {bad}' codes or s_w are not their rows of the period's "
                                f"registered {b}: a window serves the weights its period registered before its salt")
            return False
        else:
            self.registered[b] = self.weights.registered[call.k]
            out[b] = self.registered[b].root.hex()
        return True

    def _pubs(self, run: Any, calls: Sequence[int], a: str, b: str) -> dict[str, str]:
        """Each of ``calls``' (a step's calls of one K) public table ``pub@<c>`` (`pub_table`), registered once its step's
        A table ``a`` and the weights' ``b`` are: row t proof block u0 + t's public row, `core_check.public_words` of the
        window's Program salt, u0 + t and (root(a), root(b)), and in a J10-R window (`h1window.is_j10r`) u0 and the proof block's
        blocks (`h1window.call_blocks`), salted `core_check.public_row_salt` of the open root and u0 + t, as credit's
        `registered_file` builds it (`h1window.public_table`); its rows kept, as they are public."""
        units, out, j10r = HW.call_units(self.program), {}, HW.is_j10r(self.program)
        ctx = self.root.encode()
        head = CC.PUBLIC_ROW_SALT_DOMAIN + len(ctx).to_bytes(8, "big") + ctx
        x = np.frombuffer(run._salt + bytes(8) + self.registered[a].root + self.registered[b].root, np.uint8)
        for c in calls:
            u0, n = units[c]
            rows = np.tile(x, (n, 1))
            rows[:, len(run._salt):len(run._salt) + 8] = (
                np.arange(u0, u0 + n, dtype=">u8").view(np.uint8).reshape(n, 8))
            if j10r:
                tail = np.array([(u0, b0, b1) for b0, b1 in HW.call_blocks(self.program, c)], dtype=">u8")
                rows = np.concatenate([rows, tail.view(np.uint8).reshape(n, 24)], axis=1)
            words = np.ascontiguousarray(rows.reshape(n, -1, 4)[:, :, ::-1]).reshape(n, -1).view("<u2")
            salts = NR.public_salts(head, u0, n)
            v = self.pub_table(c)
            out[v] = self._register(run, v, words, 0, self.root, salts)
        return out

    def _weight_args(self, call: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        w = self._weights.get(call.layer)
        return None if w is None else w[1], self._scales.get(call.layer)

    def _prepare(self, held: list[tuple[Any, np.ndarray, np.ndarray]]) -> bool:
        """With a `receipts` that has a ``prepare``, hand it a step's calls of one K, each ``(call, codes, s_a, weight
        codes, s_w)`` as `_receipts` then asks it, in that order (a producer computes them together); False when the
        window is refused for it."""
        prepare = getattr(self.receipts, "prepare", None)
        if prepare is None:
            return True
        try:
            prepare([(c, cc, sc, *self._weight_args(c)) for c, cc, sc in held])
        except ValueError as e:
            self.refused.append(f"calls {[c.index for c, _cc, _sc in held]}: {e}")
            return False
        return True

    def _receipts(self, call: Any, codes: np.ndarray, scales: np.ndarray) -> np.ndarray | None:
        """Call ``call``'s receipts' words (`receipts`, `receipt_words`), refused unless one row a proof block of its tile
        unit's receipt width (`receipt_shape`)."""
        try:
            got = receipt_words(self.receipts(call, codes, scales, *self._weight_args(call)))
        except ValueError as e:
            self.refused.append(f"call {call.index}: {e}")
            return None
        if got.shape != (want := self.receipt_shape(call.index)):
            self.refused.append(f"call {call.index}: its receipts are {got.shape[0]} rows of {got.shape[1]} u16 words, not "
                                f"its {want[0]} units' receipts of {want[1]}")
            return None
        return got

    def segments(self, run: Any, r: int) -> dict[str, list[tuple[int, int, bytes]]]:
        """Round r's A segments (`credit.round_root`): step r's A table ``a/<K>@<r>`` of each K its calls have, whole, as
        the aligned subtrees (`credit.cover`, each ``(level, index, root)``) of its registered tree, so round r's root
        commits it before round r's coin and no other round lists it; with ``receipts``, each of step r's calls' receipt
        table alike; and round r's input-ids table and its live slot table ``slots@<r>`` (`pouw_h1_live`). Nothing once the
        window is refused."""
        if self.refused or self.program is None:
            return {}
        per, out = len(self.linears), {}
        mine = range(r * per, min((r + 1) * per, len(self.calls)))
        values = dict.fromkeys(self.a_table(c) for c in mine)
        if self.receipts is not None:
            values.update(dict.fromkeys(self.receipt_table(c) for c in mine))
        regs = [self.registered.get(v) for v in values]
        for reg in [*regs, self.ids.get(r), self.live.get(LV.value(self, LV.SLOTS, r))]:
            if reg is not None:
                out[reg.value] = [(lv, ix, reg.levels[lv][ix]) for lv, ix in CR.cover(0, reg.leaves)]
        return out

    def close(self, run: Any) -> None:
        """After the run's last round: refuse a window that isn't static, or that left a table its Program reads
        unregistered; else commit its ok rows (`ok`): row u proof block u's, `core_check.OK_ROW` under `core_check.ok_salt` of
        the open root and u, as credit's `registered_file` and Lean's ``registeredFile`` build them (`h1window.ok_table`)."""
        if self.refused:
            return
        if self.program is None:
            self.refused.append("no call was committed")
        elif not run.static():
            self.refused.append("the window isn't static: a call off its public position or refused, so the Program's calls "
                                "aren't the window's")
        elif missing := [v for v in self.tables() if v not in self.registered]:
            self.refused.append(f"{len(missing)} tables the Program reads were never registered: {missing[:4]}")
        else:
            n = sum(k for _u0, k in HW.call_units(self.program))
            self.ok = self._tree(run, self.program.fn.bindings["PREFIX"] + HW.OUTPUTS[1],
                                 np.tile(np.array(CC.OK_ROW, np.uint16), (n, 1)), 0, self.root,
                                 row_salts=b"".join(CC.ok_salt(self.root, u) for u in range(n)))

    def tables(self) -> list[str]:
        """Every value the Program's proof blocks open at: each K's ``b/<K>``, then each step's A tables, then each call's z
        table, then each call's public table; then, with ``receipts``, each call's receipt table, which its proof blocks'
        receipts are."""
        pre, ks, cs = self.program.fn.bindings["PREFIX"], sorted({k for _name, k, _n in self.linears}), range(len(self.calls))
        return ([pre + f"b/{k}" for k in ks] + list(dict.fromkeys(self.a_table(c) for c in cs))
                + [HW.table(self.program, "zy", c) for c in cs] + [self.pub_table(c) for c in cs]
                + ([self.receipt_table(c) for c in cs] if self.receipts is not None else []))

    def values(self) -> dict[str, int]:
        """Each registered value's words a row."""
        return {v: r.words for v, r in self.registered.items()}

    def roots(self) -> dict[str, str]:
        return {v: r.root.hex() for v, r in self.registered.items()}

    def ports(self) -> dict[str, str]:
        """The window's registered reads' circuit ports (`h1window.window_ports` of its Program; a lone row, R or C = 1,
        under its parameter's name, `h1window.row_key`)."""
        return HW.window_ports(self.program)

    def entries(self) -> list[dict[str, Any]]:
        """The Program's registered values' entries, then each round's input-ids table's, then the live tables'."""
        return ([r.entry() for r in self.registered.values()] + [self.ids[s].entry() for s in sorted(self.ids)]
                + [r.entry() for r in self.live.values()])

    def digest(self) -> bytes:
        """SHA-512 over the trees' entries and what was refused, canonical JSON: what `calls_root` binds."""
        return digest_of(self.entries(), self.refused)

    def query(self) -> dict[str, Any]:
        """The window Program's partition query: every tile of each K's template one proof block (`h1window.query`)."""
        return HW.query(self.program)


def digest_of(entries: Sequence[Mapping[str, Any]], refused: Sequence[str]) -> bytes:
    """`Served.digest` of the trees' `entries` and `refused`, as a record carries them (the commitment's ``served``)."""
    return hashlib.sha512(DIGEST_TAG + canonical_json({"entries": list(entries), "refused": list(refused)})).digest()


__all__ = ["B_NONCE", "EMPTY_ID", "FILE", "IDS_NONCE", "LOCAL_SESSION", "ModelKey", "NONCE", "PREFIX", "Registration", "Served",
           "unit_of",
           "V2", "Weights", "chunk_words", "declared", "declared_layout", "descriptor", "digest_of", "file_name", "ids_words", "layout",
           "model_key", "nonce", "RECEIPT_NONCE", "reads_file", "receipt_words", "record_words", "register", "register_weights", "row_words", "trees",
           "weights_of", "write_program"]
