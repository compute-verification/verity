"""Pearl+ as a resource scheme (`verity.protocols.accounting.scheme`), registered as ``pearl-plus``.

Its verifier is `verity/pouw/pearl-plus/v1` (`verifier/lean/pearlplus/PouwPearlPlus/Protocol.lean`). In C, a credited
linear is its grid's units of `PearlPlusTileCoreWords_v5` (`prover.circuit.pearl_plus_credited`), which write its y; its
block (`TILE`) is the tile, which costs its credited MACs (`tile_macs`): its real rows × TN × K, its statics TR, TN and K,
since the tile forms every word of the linear's depth K and holds its filler rows to zero. C's laying opens each tile into
many proof blocks of many templates (its noise's units and its core, the core itself opened at the served K = 2048):
`laid_tiles` gives each laid block its tile by a public rule from the laying of the C registered before any coin, each
tile the registry's own Definition at its statics (`foreign_tile`). The credit is the Lean's (`PouwPearlPlus`,
`creditAtAssumption`).

On a served run (`attach`) it computes every linear block with its device run (`verity.hooks.ComputeBlockWith`), the
workload's `PearlCDeviceRun` (`serving.tensors.pearl_c_device`), over the weight operands the run registers
(``memory``): one epoch a Commit phase (`PhaseOpened` to `PhaseClosed`), or one for the server's life where no phase
arrives (`Attached.step`); outside an epoch a block is the workload's own. At a Match capture (``ctx["stage"]``
``capture``) each call is one dispatcher op, ``verity_pouw::pearl_plus_linear`` (`serving.tensors.PearlCDevice.linear_op`),
in the first-stage linear's form, which the capture records; its coins are in-process and credit nothing. It ends the run's pass once each step's blocks
are issued (``steps``, `StepComputed`). It provides the ``cut`` (`Scheme.provides`): a phase's circuit is cut at the
linears it computes, whose calls commit their A rows and y. Its settings are `SETTINGS`. A served run launches it untied (`_hooks`): its verdict is its own over its own passes, held to no window
record, and credits nothing.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

from verity import hooks as H
from verity.protocols.accounting.compute.pouw.prover import serving, workload
from verity.protocols.accounting.compute.pouw.prover.schemes import runs_on
from verity.config import COMPOSITES
from verity.protocol import VERIFIER
from verity.protocols.accounting.scheme import (Block, Charge, Executable, Lacks, Launch, LaunchRefused, Order,
                                                             Scheme, Term, register)

#: the kernel scheme it runs, on every card its row allows
KERNEL = "pearl-c-h100-v1-h3"
OPTION = f"pouw:{KERNEL}"
#: the weight formats its kernel reads: BF16 only, since a weight's tree commits its FP32 words and its retained rows are
#: BF16 (run_device._weight_words); any other weight (FP16, FP32, FP8, INT8) is refused, not misread
READS = ("bf16",)


def tile_macs(statics: Mapping[str, int]) -> int:
    return statics["TR"] * statics["TN"] * statics["K"]


#: a credited tile's template in C before C is laid (`prover.circuit.pearl_plus_credited.PearlPlusTileCoreWords`): its
#: statics are the tile's, and `tile_macs` of them is what the tile is credited
TILE = "PearlPlusTileCoreWords_v5"
#: `laid_tiles`' refusals: C holds no tile, so there is nothing to credit and no stratum to hold to its draw; the laid C's
#: strata number blocks other than its root units, one template each, on which the stratified bound rests; a root unit
#: of the laid C lies partly inside a tile's gates, so the rule places it in no tile; a root unit carries the tile's id and
#: is not the registry's tile at its statics, so `tile_macs` of them says nothing of what it computes; a tile's device or
#: inert-work bounds are not the kernel scheme of record's (`scheme_statics`); a tile's N is not its linear's columns in
#: C (`tile_columns`); C laid has a digest
#: other than the one registered before any coin, so its tiles are not those of the C the blocks are proved against
TILE_ABSENT = "tile-absent"
BLOCK_NOT_UNIT = "tile-block-not-unit"
TILE_STRADDLES = "tile-straddles"
TILE_FOREIGN = "tile-body-foreign"
TILE_SCHEME = "tile-statics-not-the-scheme"
TILE_COLUMNS = "tile-columns-not-the-linears"
C_DIGEST = "tile-c-digest-differs"
#: a tile below the served floor (`test_instance`): a test instance's, which credits nothing
TEST_INSTANCE = "tile-test-instance"


def test_instance(statics: Mapping[str, Any]) -> str | None:
    """Why a tile of ``statics`` is a test instance's, or None when it is at the served floor: its depth K at least
    `pearl_c.K_MIN` and its tile `pearl_c.TILE` × `pearl_c.TILE` (TM × TN), the constants the Lean credit's `plan` reads
    (`Check.K_MIN`, `Check.TILE`), below which a pass credits nothing."""
    from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c as C

    k, tm, tn = statics["K"], statics["TM"], statics["TN"]
    if k >= C.K_MIN and (tm, tn) == (C.TILE, C.TILE):
        return None
    return (f"is K = {k} at {tm} × {tn}, not the served K of at least {C.K_MIN} at {C.TILE} × {C.TILE}: a test "
            "instance, which credits nothing")


@dataclass(frozen=True)
class LaidTile:
    """A credited tile of C as C's laying lays it: ``index`` its place among C's tiles in C's root order, ``gates`` its
    gates [lo, hi) in C, ``amount`` its credited MACs (`tile_macs` on its template's statics, public in C before any
    coin), ``blocks`` the laid C's proof blocks whose gates lie in its gates, ``kinds`` its m_s: its blocks in each
    stratum, by the stratum's descriptor id, and ``test`` why it is a test instance's (`test_instance`), or None."""

    index: int
    gates: tuple[int, int]
    amount: int
    blocks: tuple[int, ...]
    kinds: Mapping[str, int]
    test: str | None = None


#: Every static of `TILE`, by what holds it. ``credit``: `tile_macs` reads it. ``scheme``: `KERNEL` fixes it at the tile's
#: shape (its device, its tile inert-work bound ρ = 1/RHO, its row bound ρr = 1/RHOR), `scheme_statics`. ``columns``: N,
#: the linear's columns, held to the columns C gives the linear (`tile_columns`): the inert-work bound charges the tile
#: TN/N of its forming, so an N above the linear's would admit more inert work. ``free``: TM, the tile's rows with its
#: filler rows, which its body holds to zero word for word, `tile_macs` does not count, and the bound charges the forming
#: of, so a TM above the tile's rows admits no more. A static not listed here is refused (`scheme_statics`).
TILE_STATICS: Mapping[str, str] = MappingProxyType({"K": "credit", "TR": "credit", "TN": "credit", "DEV": "scheme",
                                                    "RHO": "scheme", "RHOR": "scheme", "N": "columns", "TM": "free"})
SCHEME_STATICS = tuple(s for s, why in TILE_STATICS.items() if why == "scheme")


def scheme_statics(fn: Any) -> str | None:
    """Why ``fn``'s statics are not `TILE_STATICS`, or its `SCHEME_STATICS` are not those of `KERNEL` at its shape (K,
    TM, TN, N, TR), as `pearl_plus_credited.credited_product` binds them, or None when they are: a tile is credited
    `tile_macs` of its shape only under the scheme of record's device and inert-work bounds."""
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited as PC

    b = getattr(fn, "bindings", {})
    statics = tuple(getattr(getattr(fn, "defn", None), "statics", ()))
    if set(statics) != set(TILE_STATICS):
        return f"has statics {sorted(statics)}, and the credit rule holds exactly {sorted(TILE_STATICS)}"
    try:
        want = PC.credited_product(KERNEL, b["K"], b["TM"], b["TN"], b["N"], b["TR"]).bindings
    except (KeyError, TypeError, ValueError) as e:
        return f"has no shape {KERNEL} binds ({type(e).__name__}: {e})"
    if bad := [s for s in SCHEME_STATICS if b.get(s) != want[s]]:
        return ", ".join(f"{s} = {b.get(s)!r} where {KERNEL} binds {want[s]!r}" for s in bad)
    return None


def tile_columns(program: Any, partition: Any, tiles: Sequence[tuple[int, Any]]) -> str | None:
    """Why a tile's N is not the columns C gives its linear, or None when every tile's is: each tile ``(u, fn)``, root
    unit u of ``partition`` (C's template partition), reads its W operand (port ``b``, one column tile of TN rows of W)
    as one whole row of a parameter of C, whose rows are the linear's column tiles, so the linear has TN × its rows
    columns, which must be the tile's N; and every row of that parameter is read by some tile, so no column C does not
    compute counts toward N."""
    from verity.circuits.refs import runs

    spans = []
    for i, (name, _t) in enumerate(program.fn.signature()[0]):
        rs = list(runs(program.input_coll(i).refs))
        if rs and all(x[3] == y[2] for x, y in zip(rs, rs[1:])):
            spans.append((range(rs[0][2], rs[-1][3]), name))
    read: dict[str, set[int]] = {}
    rows_of: dict[str, int] = {}
    for u, fn in tiles:
        at, per = 0, None
        for name, t in fn.signature()[0]:
            if name == "b":
                per = t.leaves
                break
            at += t.leaves
        gates = list(partition.inputs(u)[at:at + (per or 0)])
        hit = next(((r, v) for r, v in spans if gates and gates[0] in r), None)
        if per is None or hit is None:
            return f"C's root unit {u} reads its W operand from no parameter of C"
        (r, v), off = hit, gates[0] - hit[0].start
        if len(r) % per or off % per or gates != list(range(gates[0], gates[0] + per)):
            return f"C's root unit {u} reads its W operand as no one whole row of parameter {v!r}"
        rows, b = len(r) // per, fn.bindings
        if b["N"] != rows * b["TN"]:
            return (f"C's root unit {u} binds N = {b['N']}, and parameter {v!r}, its W operand, holds {rows} column "
                    f"tiles of TN = {b['TN']}: {rows * b['TN']} columns")
        read.setdefault(v, set()).add(off // per)
        rows_of[v] = rows
    for v, got in read.items():
        if unread := sorted(set(range(rows_of[v])) - got):
            return (f"no tile of C reads column tile {unread[0]} of parameter {v!r}, so its {rows_of[v]} column tiles "
                    "are not the linear's columns C computes")
    return None


def foreign_tile(fn: Any) -> str | None:
    """Why ``fn``, a root unit's Definition carrying `TILE`'s id, is not the registry's `TILE` bound at its statics, or
    None when it is: the registry's own object, or one of the registry's signature whose Definition and every Definition
    it reaches (its body's callees and the functions its statics bind, down to the primitives) have the registry's ids
    and definition digests (`codec.definition_digest`), compared pairwise from the root. The signature is compared first,
    so a unit of another signature never makes the registry build its tile at the unit's statics."""
    from verity.circuits.codec import definition_digest, definition_entry, descriptor_id
    from verity.circuits.defs import REGISTRY, FunctionDefinition, SpecializedDefinition, bind
    from verity.protocols.accounting.compute.pouw.prover.circuit import pearl_plus_credited  # noqa: F401

    def reached(f: SpecializedDefinition) -> list[Any]:
        def fns(v: Any) -> list[Any]:
            if isinstance(v, FunctionDefinition):
                return [v]
            if isinstance(v, (list, tuple)):
                return [g for x in v for g in fns(x)]
            if isinstance(v, dict):
                return [g for x in v.values() for g in fns(x)]
            return []
        return [nd.fn for nd in f.body.nodes] + fns([f.bindings[k] for k in f.defn.statics])

    if not isinstance(fn, SpecializedDefinition) or fn.base_id != TILE:
        return f"it is {getattr(fn, 'id', fn)!r}, no binding of {TILE}"
    try:
        want = bind(REGISTRY.defs[TILE], **fn.bindings)
    except (TypeError, ValueError, KeyError) as e:
        return f"the registry's {TILE} binds no such statics ({e})"
    if fn is want:
        return None
    (fp, fr), (wp, wr) = fn.signature(), want.signature()
    if [(n, t.to_json()) for n, t in fp] != [(n, t.to_json()) for n, t in wp] or fr.to_json() != wr.to_json():
        return f"its signature is not the registry's {descriptor_id(want)}'s"
    seen: set[str] = set()
    todo = [(fn, want)]
    while todo:
        f, w = todo.pop()
        if f is w:
            continue
        i, j = descriptor_id(f), descriptor_id(w)
        if i != j:
            return f"it reaches {i} where the registry's reaches {j}"
        if i in seen:
            continue
        seen.add(i)
        if definition_digest(definition_entry(f)) != definition_digest(definition_entry(w)):
            return f"its {i} is not the registry's (definition digests differ)"
        if isinstance(f, SpecializedDefinition):
            fs, ws = reached(f), reached(w)
            if len(fs) != len(ws):
                return f"its {i} reaches {len(fs)} Definitions where the registry's reaches {len(ws)}"
            todo += zip(fs, ws)
    return None


def laid_tiles(program: Any, fits: Callable[..., bool], strata: Sequence[tuple[str, Sequence[Sequence[int]]]],
               registered: str) -> list[LaidTile]:
    """C's credited tiles, ``program``'s root units of `TILE`, each with the proof blocks of C as
    `verity.circuits.laying.lay` lays it under ``fits`` that it holds, by a public rule from the laying: since laying keeps
    C's gates in C's order, a laid root unit is the tile's whose gates in C hold its gates, and no tile's where none do.
    C must hold a tile (`TILE_ABSENT`), each tile's device and inert-work bounds must be `KERNEL`'s at its shape
    (`TILE_SCHEME`, `scheme_statics`), each tile must be the registry's `TILE` at its statics, its body and every
    Definition it reaches (`TILE_FOREIGN`, `foreign_tile`), each tile's N the columns C gives its linear (`TILE_COLUMNS`,
    `tile_columns`), and the laid C must be the one ``registered`` names, its
    `codec.program_digest` registered before any coin (`C_DIGEST`): the tiles credited are those of the C the
    blocks are proved against. ``strata`` are the laid C's (`verity.circuits.partition_query.strata` under its
    template-instance query): their blocks must be its root units, each of the one template its stratum names
    (`BLOCK_NOT_UNIT`), and no unit may lie partly in a tile (`TILE_STRADDLES`)."""
    import bisect

    from verity.circuits import laying as LY
    from verity.circuits import partition_object as PO
    from verity.circuits.codec import program_digest

    before = PO.TemplatePartition(program, LY.query(program))
    nodes = program.root.body.nodes
    tiles = [(a + m, nodes[k].fn) for k, tid, a, n in before.ranges if tid == TILE or tid.startswith(TILE + "{")
             for m in range(n)]
    if not tiles:
        raise ValueError(f"{TILE_ABSENT}: C has no root unit of {TILE}, so no tile to credit")
    for u, fn in tiles:
        if why := scheme_statics(fn):
            raise ValueError(f"{TILE_SCHEME}: C's root unit {u}, {getattr(fn, 'id', fn)}, {why}")
        if why := foreign_tile(fn):
            raise ValueError(f"{TILE_FOREIGN}: C's root unit {u} carries {TILE}'s id and {why}, so its statics' MACs "
                             "are not what it computes")
    if why := tile_columns(program, before, tiles):
        raise ValueError(f"{TILE_COLUMNS}: {why}")
    laid = LY.lay(program, fits)
    if (got := program_digest(laid)) != registered:
        raise ValueError(f"{C_DIGEST}: C laid has digest {got[:16]}…, and the digest registered before any coin is "
                         f"{registered[:16]}…: its tiles are not the proved C's")
    after = PO.TemplatePartition(laid, LY.query(laid))
    of: list[str | None] = [None] * after.population
    for tid, rs in strata:
        for lo, hi in rs:
            if not 0 <= lo <= hi <= after.population or any(of[lo:hi]):
                raise ValueError(f"{BLOCK_NOT_UNIT}: stratum {tid}'s blocks [{lo}, {hi}) are not among the laid C's "
                                 f"{after.population} root units, each in one stratum")
            of[lo:hi] = [tid] * (hi - lo)
    if wrong := next(((a, tid) for _k, tid, a, n in after.ranges if of[a:a + n] != [tid] * n), None):
        raise ValueError(f"{BLOCK_NOT_UNIT}: the laid C's root unit {wrong[0]} is one {wrong[1]}, and the strata place "
                         f"block {wrong[0]} in {of[wrong[0]]}: a block is one root unit of its stratum's template")
    spans = [before.gates(u) for u, _fn in tiles]
    starts, edges = [lo for lo, _hi in spans], {g for span in spans for g in span}
    held: list[list[int]] = [[] for _ in tiles]
    root = laid.root.body
    for k, _tid, a, n in after.ranges:
        g, at = root.nodes[k].fn.gates, root.offsets[k]
        for m in range(n):
            lo, hi = at + m * g, at + (m + 1) * g
            i = bisect.bisect_right(starts, lo) - 1
            if i >= 0 and lo < spans[i][1] and hi <= spans[i][1] and (lo < hi or spans[i][0] < lo):
                held[i].append(a + m)
            elif (i >= 0 and lo < spans[i][1]) or (i + 1 < len(spans) and spans[i + 1][0] < hi) or \
                    (lo == hi and lo in edges):
                raise ValueError(f"{TILE_STRADDLES}: the laid C's root unit {a + m}, gates [{lo}, {hi}), lies partly "
                                 "in a tile's gates, or holds none at a tile's edge")
    out = []
    for i, ((_u, fn), span, blocks) in enumerate(zip(tiles, spans, held)):
        kinds: dict[str, int] = {}
        for b in blocks:
            kinds[of[b]] = kinds.get(of[b], 0) + 1
        out.append(LaidTile(i, span, tile_macs(fn.bindings), tuple(blocks), MappingProxyType(kinds),
                            test_instance(fn.bindings)))
    return out


#: Its settings (`Scheme.settings`), each a default of its type: ``kernel`` the kernel's library (required on a served
#: run), ``cubin`` its cubin, ``schedule`` the schedule each pass runs under (`serving.SCHEDULES`),
#: ``transcript_dir`` where passes are retained (a phase's under ``w<phase>/``, owner-only, with its ``served.json``),
#: the salt's source, ``coins`` (``in-process``: a challenger held in this
#: process, crediting nothing) or ``challenger`` (the auditor device's, HOST:PORT), whose sessions the launcher's
#: ``ctx["gate"]``, the gate in its own words, opens: ``session``, a reserve answer the gate already gave, opened by the
#: first epoch, and ``lane`` and ``peer``, under which each later epoch reserves its own (`Attached.reserve`);
#: and ``users``, a regular expression a phase's inputs' users must all match in full for its epoch to open (unset:
#: every phase's)
SETTINGS: dict[str, Any] = {"kernel": "", "cubin": "", "schedule": "serial", "transcript_dir": "", "coins": "",
                            "challenger": "", "session": {}, "lane": "", "peer": "", "users": ""}
COINS = ("", "in-process")


def check_settings(own: Mapping[str, Any]) -> str | None:
    from verity.protocols.accounting.compute.pouw.prover import serving as SV

    if own["schedule"] not in SV.SCHEDULES:
        return f"schedule is one of {', '.join(SV.SCHEDULES)}, not {own['schedule']!r}"
    if own["coins"] not in COINS:
        return f"coins is 'in-process' (a challenger held in this process) or unset, not {own['coins']!r}"
    if own["coins"] and own["challenger"]:
        return "coins and challenger are both set: the salt has one source"
    host, sep, port = own["challenger"].rpartition(":")
    if own["challenger"] and not (sep and host and port.isdecimal()):
        return f"challenger is the auditor device's challenger as HOST:PORT, not {own['challenger']!r}"
    if bool(own["lane"]) != bool(own["peer"]):
        return "lane and peer go together: the lane whose gate key reserves, and this machine's name in the auditor's peers"
    if (own["session"] or own["lane"]) and not own["challenger"]:
        return "session, lane and peer are a challenger's: they reserve or name its sessions"
    if own["challenger"] and not (own["session"] or own["lane"]):
        return "a challenger needs a session (the gate's reserve answer) or lane and peer to reserve each epoch's"
    if own["session"] and not isinstance(own["session"].get("session"), str):
        return f"session is the gate's reserve answer, {{'session': SESSION_ID, ...}}, not {own['session']!r}"
    try:
        re.compile(own["users"])
    except re.error as e:
        return f"users is a regular expression, and {own['users']!r} is not ({e})"
    return None


#: The transforms its device run applies to a checkpoint's weights before committing them: none.  The keyed 8-block
#: rotation (`registration.ROTATION_8`) is not built on the device, so it registers only a curated unrotated checkpoint,
#: and no curated list is wired.
SERVED_TRANSFORMS: tuple[str, ...] = ()
#: What its commitment says of weights that are not registered (`registration`)
UNREGISTERED = "nothing about which weights were served: the checkpoint is not registered"


def registration(scheme: Any, refused: type[Exception]) -> dict[str, Any]:
    """What it claims of the weights it serves, the commitment's ``registration``: registered weights only when
    `SERVED_TRANSFORMS` pass the registered-weights rule (``scheme.check_registration``, which raises ``refused``);
    otherwise the claim says it is about nothing (`UNREGISTERED`) and names the refusal, and serving goes on, since the
    salt is still the challenger's live coin."""
    out: dict[str, Any] = {"transforms": list(SERVED_TRANSFORMS), "curated": False}
    try:
        scheme.check_registration(SERVED_TRANSFORMS, False, False)
    except refused as e:
        return {**out, "registered": False, "claims": UNREGISTERED, "refused": str(e)}
    return {**out, "registered": True}


#: The file in the scheme's private directory (``ctx["private"]``) that holds the last salt epoch number it spent
EPOCH_FILE = "epoch"


def spend_epoch(private: str | os.PathLike[str]) -> int:
    """The next salt epoch's number (`salt.epoch_open_root`'s k), written to ``private``'s `EPOCH_FILE`, synced with its
    directory, before it is returned: a killed server's number is never handed out again, so no later server opens a
    session on a root already used (the gate's one session a root); a number may be skipped."""
    path = Path(private) / EPOCH_FILE
    if not path.exists():
        k = 0
    else:
        text = path.read_text()
        if not re.fullmatch(r"(0|[1-9][0-9]*)\n", text) or int(text) >= 2**64 - 1:
            raise ValueError(f"pearl-plus: {path} holds {text[:40]!r}, not the last salt epoch number spent below 2^64 - 1: "
                             "refused rather than risk a number already used")
        k = int(text) + 1
    tmp = path.with_name(f"{EPOCH_FILE}.next")
    with open(tmp, "w") as f:
        f.write(f"{k}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return k


def requests(ctx: Any) -> tuple[H.RequireProperty, ...]:
    return (H.RequireProperty("eager", True, "its kernel at each linear block is not graph-capturable"),
            H.RequireProperty("world", 1, "its passes and transcript are per rank, which is not wired"))


class Attached:
    """Pearl+ on one served run's exchange: the run's declaration (``declared``), its linear weights (``memory``), each
    step's ``steps`` events, and every linear block computed by the device run (`kernel`). What the adapter reads of it
    (a report, `verity.protocols.accounting.scheme`): ``commitment`` the run's commitment, which the composition's record hashes, ``record``
    and ``summary`` beside it, and `release`, which finishes the run once serving ends."""

    def __init__(self, ctx: Mapping[str, Any]) -> None:
        from verity.protocols.accounting.scheme import settings

        self.own: Mapping[str, Any] = settings(PEARL_PLUS, ctx.get("settings") or {})
        self.gate: Any = ctx.get("gate")
        self.private: Any = ctx.get("private")
        self.ledger: Any = ctx.get("ledger")
        self.capture = ctx.get("stage") == "capture"
        self.build: str = str(ctx.get("build") or "")
        if not self.own["coins"] and not self.own["challenger"]:
            raise ValueError("pearl-plus: neither 'coins' nor 'challenger' is among its settings (ctx['settings']): the salt is "
                             "the challenger's salt coin, opened after the weights' root, and there is no beacon")
        if self.capture and self.own["challenger"]:
            raise ValueError("pearl-plus: a Match capture (ctx['stage'] 'capture') opens no session on the auditor device's "
                             "challenger, whose sessions are a served run's: give 'coins': 'in-process'")
        if self.own["challenger"] and not callable(self.gate):
            raise ValueError("pearl-plus: with a 'challenger', its launcher passes ctx['gate'], the auditor device's "
                             "challenger gate in its own words, gate(*words) -> answer (commit_coins.ChallengerGate.call)")
        if self.own["challenger"] and self.private is None:
            raise ValueError("pearl-plus: with a 'challenger', its launcher passes ctx['private'], the scheme's private "
                             "directory: the salt epoch's number must survive a restart, or a later server would open a "
                             "session on a root already used")
        if self.own["lane"] and not callable(self.ledger):
            raise ValueError("pearl-plus: with a 'lane', its launcher passes ctx['ledger'], the gate's ledger as lines "
                             "(commit_coins.ChallengerGate.ledger): each epoch's reserve joins the lane's open period")
        if not self.own["kernel"]:
            raise ValueError("pearl-plus: no 'kernel' among its settings (ctx['settings']): its run computes on the device, through "
                             "the kernel's library")
        self.declared: H.Declared | None = None
        self.weights: dict[int, H.Handle] | None = None
        self.run: Any = None
        """The open epoch's device run, which benchmarks' harness gate reads (`verity_bench.vllm.runner._pearl_c_gate`)."""
        self.session: Any = None
        self.reserved: dict[str, Any] | None = None
        self.epochs = 0
        self.phase: int | None = None
        self.phased = False
        self.first_step = 0
        self.life: dict[str, Any] | None = None
        self.phases: dict[int, dict[str, Any]] = {}
        self.busy: dict[int, int] = {}
        self.computed: set[int] = set()
        self.users = re.compile(self.own["users"]) if self.own["users"] else None
        self.user_of: dict[str, str | None] = {}
        self.held = False
        self.unopened: set[int] = set()
        self.life_unopened = False

    def declare(self, event: H.Declared) -> None:
        self.declared = event

    def registered(self, event: Any) -> None:
        """The ``memory`` hook's `RegisteredValues`; its ranges' events (`RangesDeclared` and the rest) are not Pearl+'s."""
        if isinstance(event, H.RegisteredValues):
            self.weights = self.keyed(event)

    def keyed(self, event: H.RegisteredValues) -> dict[int, H.Handle]:
        """The linear weights by block key (``operands``, the key `BlockDue` carries), every one committed before the
        first block."""
        values = dict(event.values)
        if not event.operands:
            raise ValueError("pearl-plus: the registered values name no weight operand (RegisteredValues.operands is "
                             "empty), so there is no linear weight to commit")
        if missing := sorted(set(event.operands) - set(values)):
            raise ValueError(f"pearl-plus: weight operands {missing} are not among the registered values")
        return {key: values[name] for name, key in event.operands.items()}

    def step(self, event: Any) -> None:
        """Its epochs, keyed on whether phase events arrive: one epoch a phase, opened at `PhaseOpened` and finished at
        `PhaseClosed`, its committed root taken from `PhaseCommitted` whenever it comes, after the next phase has opened
        too; with no phase, one epoch for the server's life, opened at the first step with tokens and finished at
        `release`. A pass ends at each `StepComputed`, once the step's blocks are all issued. A step that computed and
        ended without one is refused: under a multiprocess executor the adapter emits no `StepComputed`, and no pass
        would end. Under ``users``, an epoch opens instead at its phase's (or the server's life's) first step whose inputs'
        users all match it, if no step of the phase computed before it; a phase that ran another's inputs first serves
        no epoch and takes no coins, and a step deciding it with an input that named no user is refused."""
        if isinstance(event, H.InputArrived):
            if self.users is not None:
                self.user_of[event.input] = event.user
            return
        if isinstance(event, H.OutputsComplete):
            self.user_of.pop(event.input, None)
        if isinstance(event, H.PhaseOpened):
            if self.run is not None and not self.phased:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens at step {event.step}, after steps outside any "
                                   "phase opened its epoch for the server's life: its committer opens phase 0 before "
                                   "the first step")
            if self.phase is not None:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens while phase {self.phase} is open")
            if self.weights is None:
                raise RuntimeError(f"pearl-plus: phase {event.phase} opens before its linear weights registered")
            self.phased, self.phase, self.first_step = True, event.phase, event.step
            if self.users is None:
                self.run = self.open(self.weights, self.own)
            else:
                self.held = True
            return
        if isinstance(event, H.PhaseClosed):
            if self.phase != event.phase:
                raise RuntimeError(f"pearl-plus: phase {event.phase} closes, and the open phase is {self.phase}")
            if self.run is None:
                self.unopened.add(event.phase)
                self.held, self.phase = False, None
                return
            self.phases[event.phase] = {**self.close(event.step - self.first_step + 1), "first_step": self.first_step,
                                        "last_step": event.step,
                                        "program": event.program.hex(), "root": None}
            self.phase = None
            return
        if isinstance(event, H.PhaseCommitted):
            if event.phase in self.unopened:
                self.unopened.discard(event.phase)
                return
            if event.phase not in self.phases:
                raise RuntimeError(f"pearl-plus: phase {event.phase}'s root comes before the phase closed")
            if self.phases[event.phase]["root"] is not None:
                raise RuntimeError(f"pearl-plus: phase {event.phase}'s root comes twice")
            self.phases[event.phase]["root"] = event.root.hex()
            return
        if isinstance(event, H.StepBegan) and self.run is None and not self.phased and event.slots \
                and self.weights is not None and not self.life_unopened:
            if self.users is None or self.tied(event.inputs):
                self.run = self.open(self.weights, self.own)
            else:
                self.life_unopened = True
        if isinstance(event, H.StepBegan) and self.held:
            if self.tied(event.inputs):
                self.held, self.run = False, self.open(self.weights, self.own)
            elif event.slots:
                self.held = False
        if self.run is None:
            return
        if isinstance(event, H.StepBegan):
            if (limit := self.run.details.get("max_step_tokens")) is not None and event.slots > limit:
                raise RuntimeError(f"pearl-plus: a step of {event.slots} tokens is over one pass ({limit} tokens a "
                                   f"step); serve with max_num_batched_tokens at most {limit}")
            self.busy[event.step] = event.slots
        elif isinstance(event, H.StepComputed):
            self.run.end_pass(step=True)
            self.computed.add(event.step)
        elif isinstance(event, H.StepEnded):
            if self.busy.pop(event.step, 0) and event.step not in self.computed:
                raise RuntimeError(f"pearl-plus: step {event.step} computed and ended with no StepComputed, so its pass "
                                   "never ends (a multiprocess executor emits none); serve with a uniprocess executor")
            self.computed.discard(event.step)

    def tied(self, inputs: tuple[str, ...]) -> bool:
        """Whether ``inputs`` are some, and every one's user matches ``users`` in full; refused where one arrived naming
        no user (`InputArrived.user`), since no answer about it is the setting's. A user is matched here only, and never
        written anywhere."""
        if missing := [i for i in inputs if self.user_of.get(i) is None]:
            raise RuntimeError(f"pearl-plus: inputs {missing[:4]} name no user, and its 'users' setting "
                               f"{self.users.pattern!r} opens an epoch only on a step whose inputs' users all match it: "
                               "the adapter fills InputArrived.user with each input's user")
        return bool(inputs) and all(self.users.fullmatch(self.user_of[i]) for i in inputs)

    def kernel(self, due: H.BlockDue, original: Callable[[Mapping[str, H.Handle] | None], H.Handle]) -> H.Handle:
        if due.format not in READS:
            raise ValueError(f"pearl-plus: block {due.block}'s weight is {due.format}, and its kernel reads {READS}")
        if self.weights is None:
            raise RuntimeError(f"pearl-plus: block {due.block} is due before the run registered its linear weights")
        if self.run is None:
            return original(None)
        if self.capture:
            from verity.protocols.accounting.compute.pouw.prover.serving.tensors import pearl_c_device

            return pearl_c_device().linear_op()(due.inputs["x"], due.inputs["weight"], due.inputs.get("bias"), self.build,
                                                due.key)
        return self.run.gemm(due.key, due.inputs["x"], due.inputs["weight"], due.inputs.get("bias"))

    def commitment(self) -> Mapping[str, Any]:
        """The epoch's commitment for the server's life; with phases, each phase's by number, with its first and last
        step, its Program's digest and its committed root (None until `PhaseCommitted`)."""
        if self.phased:
            return {"phases": {str(p): {k: v for k, v in e.items() if k != "summary"} for p, e in self.phases.items()}}
        return dict(self.life["commitment"]) if self.life else dict(self.run.commits) if self.run is not None else {}

    def record(self) -> Mapping[str, Any]:
        return {"entry": OPTION, "schedule": self.own["schedule"]}

    def summary(self) -> Mapping[str, Any]:
        if self.phased:
            return {"phases": {str(p): e["summary"] for p, e in self.phases.items()}}
        return dict(self.life["summary"]) if self.life else dict(self.run.details) if self.run is not None else {}

    def release(self) -> None:
        """The open epoch finished when serving stops: the server's life's, or a phase that never closed, recorded
        with ``closed`` False."""
        if self.run is None:
            return
        if self.phase is None:
            self.life = self.close()
            return
        self.phases[self.phase] = {**self.close(), "first_step": self.first_step, "last_step": None, "program": None,
                                   "root": None, "closed": False}
        self.phase = None

    def close(self, steps: int | None = None) -> dict[str, Any]:
        """The open epoch's device run finished (its session closed with it) and freed, before another opens; what it
        committed and its details. A retained epoch's record is written beside its passes (`served`)."""
        import gc

        from verity.protocols.accounting.compute.pouw.prover import serving as SV

        from verity.protocols.accounting.compute.pouw.prover.serving.tensors import pearl_c_device

        run, session = self.run, self.session
        self.run = self.session = None
        if self.capture:
            pearl_c_device().runs.pop(self.build, None)
        if session is None:
            run.finish()
        else:
            SV.finish_window(run, session)
        if getattr(run, "retain_dir", None) is not None:
            self.served(run, steps)
        out = {"commitment": dict(run.commits), "summary": dict(run.details), **({"reserved": self.reserved}
                                                                                   if self.reserved else {})}
        self.reserved = None
        del run
        gc.collect()
        return out

    def reserve(self, k: int) -> dict[str, Any]:
        """Salt epoch ``k``'s session, reserved before it opens, since a reserve opens once: the first epoch's the
        ``session`` setting, if given; every other one reserved at the gate under ``lane`` as ``peer``, joining the
        lane's open period (`serving.reserve_salt`, epoch ``k``). The reserve answer and its period go in the epoch's
        record."""
        from verity.protocols.accounting.compute.pouw.prover import serving as SV

        if self.epochs == 1 and self.own["session"]:
            reserved, period = dict(self.own["session"]), None
        elif self.own["lane"]:
            reserved, period = SV.reserve_salt(self.gate, self.ledger, self.own["lane"], self.own["peer"], k)
        else:
            raise ValueError(f"pearl-plus: salt epoch {k} needs a session of its own, and the 'session' setting was the "
                             "first epoch's (a reserve opens once): give 'lane' and 'peer' to reserve one an epoch")
        self.reserved = {"session": reserved, "period": period}
        return reserved

    def served(self, run: Any, steps: int | None) -> None:
        """A retained epoch's ``served.json`` (`serving.SERVED_SCHEMA`, as ``served_passes.py retain --window-record`` and
        ``verify --tie`` read it) and ``coins.json`` (its coin commitment) in its retained directory: its passes with
        calls (`serving.served_passes`), its coins (`serving.coins_of`), and ``error`` when they aren't one round a step
        of its phase's ``steps``, each with one call a linear and a manifest."""
        import json

        from verity.protocols.accounting.compute.pouw.prover import serving as SV

        d, linears = Path(run.retain_dir), len(self.weights or {})
        passes = [p for p in SV.served_passes(run.details["passes"]) if p["units"]]
        bad = ([] if steps is None or len(passes) == steps else
               [f"its passes with calls are {len(passes)} and its phase's steps {steps}, so round s isn't step s"])
        bad += [f"pass {p['pass']} has {p['units']} calls, not one a linear ({linears})" for p in passes
                if p["units"] != linears][:4]
        bad += [f"pass {p['pass']} retained no manifest" for p in passes if not p.get("manifest")][:4]
        held = SimpleNamespace(commitment=lambda: run.commits, info=lambda: run.details)
        doc = {"schema": SV.SERVED_SCHEMA, "entry": OPTION, "window": self.phase, "retained": True, "commit_steps": steps,
               "passes": passes, "passes_with_calls": len(passes),
               "coins": SV.coins_of(held, (self.reserved or {}).get("session")), "ok": not bad,
               "error": "; ".join(bad) or None}
        (d / "coins.json").write_text(json.dumps(run.commits["coins"]) + "\n")
        (d / "served.json").write_text(json.dumps(doc, indent=1) + "\n")

    def open(self, weights: Mapping[int, H.Handle], own: Mapping[str, Any]) -> Any:
        """The device run over ``weights``, under Pearl+'s own settings ``own`` (``kernel``, ``cubin``, ``schedule``,
        ``transcript_dir``, and ``coins`` or ``challenger``, whose session `reserve` gives and the launcher's gate
        opens): their B rows salted with fresh bytes each epoch (the served checkpoint is not registered), their
        roots committed before any salt, the salt coin opened on their root (with ``challenger``, one
        session an epoch, closed at its `close`, on an epoch number first spent in ``ctx["private"]``, `spend_epoch`; else
        the in-process coins of this process's epoch count), and the run's pass limit, retained passes and ``registration``,
        what it claims of the weights it serves."""
        from verity.protocols.accounting.compute.pouw.prover import serving as SV
        from verity.protocols.accounting.compute.pouw.prover.serving.tensors import pearl_c_device

        device, name, weights = pearl_c_device(), OPTION.partition(":")[2], dict(weights)
        self.epochs += 1
        if bad := SV.pearl_c_refusals(name, {w: tuple(t.shape) for w, t in weights.items()}):
            raise ValueError(f"pearl-plus: its weights are refused before any commitment: {'; '.join(bad)}")
        kernel = SV.PearlCKernel.load(own["kernel"], own["cubin"] or None)
        salts = {w: b"".join(SV.fresh_salts(t.shape[0])) for w, t in weights.items()}
        roots = device.weight_roots(kernel, weights, name, weight_salts=salts)
        k, carried = None, None
        if own["challenger"]:
            k = spend_epoch(self.private)
            given = SimpleNamespace(challenger=own["challenger"], session=self.reserve(k), gate=SV.gate_of(self.gate))
            self.session, coins, opening, _, carried, k = SV.salt_session(OPTION, given, SV.weights_root(roots), epoch=k)
        else:
            held = SV.in_process_coins(self.epochs - 1)
            coins, opening = held.commitment, held.open("salt", SV.weights_root(roots))
        with SV.closed_on_error(self.session):
            scheme = SV.pearl_c_scheme(name)
            run = device.PearlCDeviceRun(scheme, OPTION, kernel, weights, roots, coins, opening, pass_index=k,
                                         schedule=own["schedule"], weight_salts=salts)
        if self.session is not None:
            run.commits.update(coin_source=SV.PEARL_C_CHALLENGER, session=carried)
        run.commits["registration"] = registration(scheme, SV.DomainError)
        run.details["max_step_tokens"] = device.step_tokens(run, SV.PEARL_C_CALL_CAPACITY)
        if self.capture:
            device.runs[self.build] = run
        if own.get("transcript_dir"):
            run.retain_dir = Path(own["transcript_dir"]) / (f"w{self.phase}" if self.phased else "")
            for d in {Path(own["transcript_dir"]), run.retain_dir}:
                d.mkdir(mode=0o700, parents=True, exist_ok=True)
                d.chmod(0o700)
            if not run.cur.matmuls:
                run.begin()
        return run


def served_record(d: str | os.PathLike[str]) -> dict[str, Any]:
    """A served run's report as vLLM's scheme plugin writes it (``out/schemes/pearl-plus/``: ``configuration.json``,
    ``summary.json``, ``commitment.json``), read as the served verdicts' Pearl+ piece reads ``served.json``: its
    ``arm``, ``passes`` (`pearl_c.window.served_passes`), ``coins`` (`pearl_c.window.coins_of`, the gate's reserve answer
    its ``session`` setting), pass limit and pass ends. What only the load's driver measures (``generate_s``, the
    requests and their tokens, ``ok``) is not in it, and a pass's ``manifest`` is there only where it was retained."""
    import json

    from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import window as PW

    configuration, summary, commitment = (json.loads((Path(d) / f"{n}.json").read_text())
                                          for n in ("configuration", "summary", "commitment"))
    own = configuration.get("settings") or {}
    held = SimpleNamespace(commitment=lambda: commitment, info=lambda: summary)
    return {"arm": configuration["record"]["entry"], "retained": bool(own.get("transcript_dir")),
            "max_step_tokens": summary.get("max_step_tokens"), "pass_ends": summary.get("pass_ends"),
            "passes": PW.served_passes(summary.get("passes") or []), "coins": PW.coins_of(held, own.get("session") or None)}


def attach(exchange: Any, ctx: Any) -> Attached:
    on = Attached(ctx)
    exchange.subscribe(PEARL_PLUS.name, "declared", on.declare)
    exchange.subscribe(PEARL_PLUS.name, "memory", on.registered)
    exchange.subscribe(PEARL_PLUS.name, "steps", on.step)
    exchange.request(PEARL_PLUS.name, H.ComputeBlockWith(on.kernel))
    return on


PEARL_PLUS = register(Scheme(
    name="pearl-plus",
    catalog={part: f"verity.protocols.accounting.compute.pouw.prover.circuit.pearl_plus_{part}"
             for part in ("credited", "tile")},
    messages=tuple(m for m in workload.MESSAGES if m.type in ("PoUWScheme", "PearlPass")),
    protocol="verity/pouw/pearl-plus/v1",
    resource="compute",
    blocks=(Block(TILE, tile_macs),),
    assumptions=("Pouw.PearlC.Assumptions.TTOutTilePearlCH100Rev1",),
    log=(Order("weightsRoot", "pouw/salt"), Order("commitment", "pouw/draw")),
    charges=(Charge("inert-work", "rho"), Charge("sample", "epsilon"), Charge("over-bound-tiles", "over_cap_draws"),
             Charge("excluded-rows", "give_up")),
    delta_scheme=Term("Pouw.PearlC.δPearlC", ("q", "N"), lambda q, N: Fraction(q + N, 2**128)),
    over="399 × the MACs credited: each passed tile's credited cells times the record's kept width |K|",
    under="400 × the MACs spent: the prover's cost in the cost model, priced at a credited MAC's honest cost (TT_OUT's T "
          "at g₀ = 1/400)",
    settings=SETTINGS,
    check_settings=check_settings,
    option=OPTION,
    hooks=("blocks", "declared", "memory", "steps"),
    provides=("cut",),
    requires={"one-labelling": "its own pass manifests, held to no window record",
              "log-order": "its own coin sessions: the salt's opened on the weights' root, the sample's on the run's "
                           "commitment",
              "one-design": "its own tile sampling by ticket weight"},
    requests=requests,
    attach=attach,
    parties=(*workload.PEARL_PARTIES, VERIFIER),
    channels=workload.PEARL_CHANNELS,
    colocated=(),
    needs=frozenset({"windows"}),
    depends=(),
    pieces={"workers": Launch(optional=("gate", "peer", "key-sha512", "lane", "main-file"), development="gate",
                              starts=("gate", "peer", "lane", "key-sha512", "main-file"),
                              starts_optional=("lane", "main-file"),
                              entry="verity.protocols.accounting.compute.pouw.scheme:launch_pearlplus",
                              executables=(Executable(
                                  "verity/protocols/accounting/compute/pouw/verifier/lean/pearlplus",
                                  "pouw-pearlplus"),))},
    answers=Lacks("has no piece that answers outside an exchange"),
    # until the salt and sample sessions' signature pin is in Lean
    unchecked={"coins-signatures-checked-by-the-auditor":
               "coins' signatures checked on the served node only; not checked by the auditor",
               # no record the verifier reads states vLLM's graph mode: `requests`' eager is held on the serving node
               "cuda-graph-rows-outside-the-verdict":
               "Pearl+'s verdict covers a workload served eager (vLLM's graph mode NONE): under a CUDA graph (FULL or "
               "PIECEWISE) a step's linear runs at the capture size, with rows no request owns, and those rows are "
               "outside what its verdict covers"},
    record="work",
    symbol="V",
    published=("MACs", Fraction(1)),
    channel_schemas=(workload.PEARL_SALT,),
    check=Lacks("Pearl+'s verifier of record is its owner's to declare"),
    premises=("Verity.Vocab.PouwSound", "Pouw.PearlPlus.Setting.ChargeHolds"),
    reader=Lacks("the record's work row is H1's (W_cert, in work); Pearl+'s reader is its owner's to declare"),
    tables=("pearl-a", "pearl-y", "pearl-b"),
    output_format="bf16",
    product_family="fp8-e4m3-rowwise",
    y_committed=True,
    salt_open_root=serving.epoch_open_root,
    kernels=runs_on(KERNEL),
    keeps_weight_copy=True,
    copy_beside_storage_ruled="Storage's bound is over a responder that is handed the weights, so it does not depend on "
                              "what else holds them. Beside Pearl+ the plain weights stay resident for the whole serve, "
                              "with Pearl+'s formed FP8 buffers as a third form, and storage's release of a weight frees "
                              "nothing.",
))

SCRIPT = "benchmarks/one_stage/served_pearlplus.sh"
#: served_verdicts.sh's variable for each binding of Pearl+'s live challenger, the party that sends its salt
_LIVE = {"gate": "PEARLC_CHALLENGER_GATE", "peer": "PEARLC_CHALLENGER_PEER", "key-sha512": "H1_KEY_SHA512"}
_CHALLENGER = workload.PEARL_CHANNELS[0].sender


def _hooks(cfg, params, bound, pinned):
    """Pearl+ on the workload's engine (POUW=pearlplus), untied: each engine step's passes retained under the gate's
    salts, then its verdict over its own passes, held to no window record, and the Lean verifier
    (served_pearlplus.sh). With no gate bound the gate variables are set empty, since unset served_verdicts.sh takes the
    lane's gate: Pearl+'s epoch coins are then an in-process challenger's on the workload's machine, its verdict claims
    none, and the run is a development one."""
    product = "remote" if sorted(cfg["protocols"]) == sorted(COMPOSITES["verity/remote/v1"]) else "onsite"
    hooks = {"POUW": "pearlplus", "PRODUCT": product}
    if given := [k for k in _LIVE if k in bound]:
        if lacking := [k for k in _LIVE if k not in bound]:
            raise LaunchRefused("unbound", f"Pearl+'s live challenger takes {', '.join(_LIVE)} together, and the "
                                           f"deploy bindings bind {', '.join(given)} without "
                                           f"{', '.join(f'{_CHALLENGER}.pearl-plus.{k}' for k in lacking)}: bind "
                                           f"them, or none for coins flipped on the workload's machine")
        hooks |= {v: bound[k] for k, v in _LIVE.items()}
    else:
        hooks |= {"PEARLC_CHALLENGER_GATE": "", "PEARLC_CHALLENGER_PEER": "", "DEVELOPMENT": "1"}
    return hooks | {v: bound[k] for k, v in (("lane", "PEARLC_CHALLENGER_LANE"), ("main-file", "MAIN_FILE"))
                    if k in bound}


def launch_pearlplus(declaration: Scheme) -> tuple[Any, tuple[str, ...]]:
    """Pearl+'s workers piece (its `Launch.entry`): its hooks and the script they run. It launches and collects, and
    decides no verdict."""
    return _hooks, (SCRIPT,)
