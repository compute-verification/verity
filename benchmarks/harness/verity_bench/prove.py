"""`verity bench prove`'s rules: what a proof system's prover costs on a workload, as one labelled table a block type.

Standard library only: `prove_records` reads the inputs from the store's records and renders the table.

**The workload** is its block census (`Census`): every block type (a Definition at its statics) with the proof blocks of
it a served token takes and log2 of its witness bits (k_log), and the MACs a served token takes. A cut (a `partition`
card) replaces the block types of the served rows it recuts.

**A proof system** (`ProofSystem`) turns a census into a `Table`: a row a block type with the blocks of it proved a
served token, the prover GPU-seconds a block and where that figure is from (`SOURCES`, best first):

- `steady`: a row of the proof server's rate (`Steady`) at the stated prover count and the stated host cores a card:
  statements of the type proved back to back by that many provers holding that many CPU cores, every session accepted
  and the statement cache's hit rate flat. A type whose rows at that prover count were all timed at more cores takes
  the one at the fewest, and its row says so; a row timed at fewer cores than stated prices nothing. A row whose
  staged Definition is no block type of the census is on no row of the table and in no figure, the range's other end
  included: the table lists it as unread (`Table.unread`). A row may be staged for its census type in another
  Definition than the type's own (`Steady.form`): the type's row names the Definition its statements are of, as the
  type's Boolean version where the tree registers it as that (`Flock.boolean`) and as another Definition where it
  doesn't. A type's rows at the same prover count and cores that timed statements of other sizes are carried beside
  the one that prices it (`Cost.sizes`, a row a size), so a reader can split a statement's fixed seconds from its
  seconds a block; they move no figure of the table;
- `warm`: no steady-state row, but its own statements proved on one card from warm sessions (`Warm`): after a stated
  count of warm sessions, the median of a stated count of timed ones (`WARM` and `RUNS`, which the row names), a
  statement's GPU-seconds over its blocks. It is measured, not at steady state: the steady-state share and the measured
  status count it as neither, and no other type is priced from it;
- `family`: no row of its own, but a steady-state row of its Definition and K at other statics, chosen as above, at
  2^Δk its seconds a block (Δk between the two k_logs), the dearest such where there are several, named with the
  Definition its statements are of as a type's own row is. It is priced, not measured: the steady-state share and the
  measured status count it as neither;
- `single`: one warm process's sessions on the type alone, a block's marginal seconds plus its share of a full
  statement's fixed seconds;
- `k_log`: no measurement of the type, so the single-session types' median seconds a witness bit times its 2^k_log
  bits, and a statement's fixed seconds on their least-squares line in k_log;
- `missing`: neither a measurement nor a k_log, so no figure, and the totals leave the type out and say so.

`Flock` (`prove_flock`, with the measurements it reads) proves every block: its row counts are the census's. It
prices a type at a stated prover count (`provers`) and a stated count of host CPU cores a card (`cores`), from the
steady-state rows at that prover count only and never from one timed at fewer cores, so no type's figure depends on
which counts were run or in what order. `Spot`
wraps a proof system with a sampling plan: the same costs, and the blocks the plan samples a served token. A plan that
states the prices its own planner sized the sample at gets a second total beside the table's, the same blocks at those
prices (`planned`), with where the two differ: a plan's sample fits its planner's prices, and the table's sources may
hold others.

**The totals** (`report`) are the same for every system: the prover GPU-seconds a served token, the served tokens a
prover GPU covers a second (its inverse) and the prover GPUs a serving GPU needs (over a served token's GPU-seconds,
`Serving`, a stated and modeled input), with the share of the proving time each source covers and the share that rests
on rows timed at more cores than the stated host's. Their range's other end
prices every type not measured at steady state at the steady-state types' median seconds a witness bit. The totals are
`measured` only when every block type is measured at steady state, `partial` when a type has no figure (they leave its
proving out, so they name each such type with its share of the blocks, and of the MACs where the census holds them: a
type with no time has no share of the proving time), and `modeled` otherwise.

**The Lean verifier's seconds and the proof's bytes** are a block's too, from the statements the Lean verifier was
run on (`LeanVerify`, a `lean-verify` record: one statement of a block type that the Lean verifier's `verify` accepted, each
run's seconds and its proof files' bytes). Each figure is priced by a size: the seconds by a block's AND gates (the
census's `ands`), the bytes by its 2^k_log witness bits (`LEAN_SECONDS_BASIS` and `PROOF_BYTES_BASIS` say why, and
what no measurement backs). A block type takes each from the best source that holds it (`LEAN_SOURCES`): its own
record (`measured`: its statement's seconds, the median of its runs, and bytes over the statement's blocks); else its
Definition's measured types (`family`: the same Definition, whichever form a statement was staged in, since a record
names the census type it was staged for), the dearest of their rates a unit of size times its own size; else the
rate fitted over every measured statement (`fitted`: their seconds, or bytes, over their sizes) times its own size;
else none, where it has no size. Each total a served token is `measured` only when every block type has a record of
its own and `modeled` otherwise, names each measured type with its share of the proving time, and gives the share of
the figure each source holds.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from verity_bench.prove_flock import LeanVerify, Steady

STEADY, WARM, FAMILY, SINGLE, MODELED, MISSING = "steady", "warm", "family", "single", "k_log", "missing"
#: where a block type's proving time is from, best first, as the table words it
SOURCES = {STEADY: "measured at steady state", WARM: "measured from warm sessions", FAMILY: "steady state, its family's",
           SINGLE: "measured in a single session", MODELED: "modeled from k_log", MISSING: "missing"}
#: a full statement's witness bits: a block type's statement holds 2^35 / 2^k_log of its blocks
STATEMENT_BITS = 2**35
#: a MAC is a fused multiply-add, which the data sheets' rates count as two FLOP
FLOP_PER_MAC = 2
SCHEMA = "verity-bench/prove/v1"
#: the CPU cores a card's provers hold on the host a table prices for, where none is stated: the whole cores of the
#: share node A's provider gives a card
HOST_CORES = 10
#: the CPU cores a steady-state row is read as timed at where its record states none
UNSTATED_CORES = 24
#: where a block type's Lean verifier seconds and proof bytes are from, best first: its own `lean-verify` record, its
#: Definition's measured types at its size, or the rate fitted over every measured statement at its size
LEAN_MEASURED, LEAN_FAMILY, LEAN_FITTED = "measured", "family", "fitted"
#: each source of a Lean figure as its line words it
LEAN_SOURCES = {LEAN_MEASURED: "own", LEAN_FAMILY: "family", LEAN_FITTED: "fitted"}
#: each source of a block type's Lean verifier seconds as its row marks it: its own statement is a measurement; its
#: Definition's measured rate at its AND gates is modeled on a measurement of that Definition; the rate fitted over
#: every measured statement is assumed to hold for a Definition no measurement backs
LEAN_MARKS = {LEAN_MEASURED: "measured", LEAN_FAMILY: "modeled", LEAN_FITTED: "assumed"}
#: the circuit file a measured statement was read from, by the statement id its record names
LEAN_FORMS = {"verity/flock-circuit": "the unrolled circuit file", "verity/flock-circuit/tmpl": "its circuit's templates"}
#: why a type without a record of its own takes its Lean verifier seconds by its AND gates, and what that leaves unbacked
LEAN_SECONDS_BASIS = (
    "the Lean verifier reads and sets up a statement from its circuit file, then checks the session: the lincheck's "
    "fold over the circuit's matrix entries, and the rounds and Ligerito queries that the statement's size m fixes "
    "(`Flock.Schedule.fast100`). The first two grow with a block's gates, so a type is priced by its AND gates, at its "
    "own Definition's rate where a statement of it was measured: the rate a measured Definition shows prices its other "
    "statics, and the fitted rate every other Definition, which no measurement of that Definition backs. The last is "
    "the same for every full statement, and an AND rate prices it in proportion to a statement's gates")
#: why a type without a record of its own takes its proof bytes by its witness bits
PROOF_BYTES_BASIS = (
    "a proof file's length is fixed by its statement's shape and by nothing else of the block type: its size m, the "
    "block's k_log and the opening's claims (`Flock.Proof.decodeRead`, `Flock.Zk.reader`). Every full statement has "
    "m = 35, so its proof is a measured one's but for the lincheck's rounds, two field elements a unit of k_log, and a "
    "ring switch for each opening claim past the first two: a type is priced by its witness bits, not its gates")


# ---- the workload ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A block type of a census: a Definition at its statics as the census spells it, the served rows it is cut from
    (`kind`, which a cut replaces by), the proof blocks of it a served token takes, log2 of its witness bits, and a
    block's served MACs and AND gates where the census holds them."""

    name: str
    kind: str
    per_token: float
    k_log: int | None = None
    macs: float | None = None
    ands: int | None = None


@dataclass(frozen=True)
class Census:
    """A workload's block census: its block types, the MACs a served token takes, those of them inside a proof block
    (the rest are served rows no cut fits, which no table prices), and the cut applied to it."""

    blocks: tuple[Block, ...]
    served_macs_per_token: float
    block_macs_per_token: float
    cut: str | None = None

    @property
    def macs_in_blocks(self) -> float:
        """The share of a served token's MACs that are inside a proof block."""

        return self.block_macs_per_token / self.served_macs_per_token

    @classmethod
    def of(cls, doc: Mapping[str, Any]) -> Census:
        """The census of a `block-census/v1` record's `census.json`, one request's counts over its tokens."""

        tokens = sum(doc["tokens"].values())
        blocks = tuple(Block(name, ", ".join(sorted(d["of"])), count / tokens, d.get("k_log"), d.get("macs"), d.get("ands"))
                       for name, d in doc["definitions"].items() if (count := d["prefill"] + d["decode"]))
        return cls(blocks, doc["per_request"]["circuit_macs"] / tokens, doc["per_request"]["macs"] / tokens)

    def recut(self, card: Mapping[str, Any]) -> Census:
        """The census under a `partition` card's cut: its strata in place of the block types of the kinds it replaces.
        A cut recuts served rows that were already in blocks, so the MACs inside a block stay."""

        replaced = set(card["replaces"])
        if absent := sorted(replaced - {b.kind for b in self.blocks}):
            raise ValueError(f"the cut `{card['name']}` replaces {', '.join(absent)}, which the census doesn't hold: it is "
                             "another census's cut")
        kept = tuple(b for b in self.blocks if b.kind not in replaced)
        cut = tuple(Block(x["block"], x["kind"], x["per_token"], x.get("k_log"), x.get("macs"), x.get("ands"))
                    for x in card["strata"] if x["per_token"])
        return replace(self, blocks=kept + cut, cut=card["name"])


@dataclass(frozen=True)
class Serving:
    """The serving rate, a stated and modeled input: a served token's MACs at the fleet's assumed serving MFU of the
    card's data-sheet peak."""

    mfu: float
    flop_per_second: float
    basis: str

    def gpu_seconds_per_token(self, census: Census) -> float:
        return FLOP_PER_MAC * census.served_macs_per_token / (self.mfu * self.flop_per_second)


_FN = re.compile(r'\{"fn":"([^"]*)"\}')
_F64 = re.compile(r'\{"f64":"([^"]*)"\}')
_BOOL = re.compile(r"\b(true|false)\b")


def definition_id(spec: str) -> str:
    """A census's name of a block type, in the spelling of the Definition's id a staged statement records: a function's
    and a float's static written out, Python's booleans and tuples."""

    spelled = _F64.sub(lambda m: repr(float.fromhex(m[1])), _FN.sub(r"\1", spec))
    return _BOOL.sub(lambda m: m[1].capitalize(), spelled).replace("[", "(").replace("]", ")")


# ---- the table ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """What proving one block of a type costs, and where each figure is from: the prover GPU-seconds (None when
    `missing`), the run or art that holds it, the staged type that measured it as its record names it (a sampling plan
    names its cells' the same way), the same block at the steady-state types' median seconds a witness bit (the range's
    other end; None for a type measured at steady state), the proof server's verifier's seconds where a record holds
    them, the prover count, the host CPU cores a card and the blocks a statement held that a steady-state figure was
    measured at, the Lean verifier's seconds and the proof's bytes with where each is from (`lean_source` and
    `proof_bytes_source`, a key of `LEAN_SOURCES`; None with no figure), the measured statement the seconds rest on
    (`lean_statement`: its own, or its Definition's that prices it; None at the fitted rate), and the Definition the
    statements a steady-state figure was measured on are of, by its id (`definition`: the type's own row's, or that of
    the family's row it is scaled from; None from any other source). `sizes` is each statement size the type's own
    steady-state rows timed at its prover count and cores, as (blocks, a statement's GPU-seconds, the run or art that
    holds it), the size that prices it first; empty where they timed one size or none."""

    source: str
    seconds: float | None = None
    record: str | None = None
    note: str = ""
    staged: str | None = None
    at_steady_rate: float | None = None
    verifier_seconds: float | None = None
    proof_bytes: float | None = None
    provers: int | None = None
    cores: int | None = None
    blocks: int | None = None
    lean_seconds: float | None = None
    lean_source: str | None = None
    definition: str | None = None
    proof_bytes_source: str | None = None
    sizes: tuple[tuple[int, float, str], ...] = ()
    lean_statement: LeanVerify | None = None


@dataclass(frozen=True)
class Row:
    """A block type's row: the blocks of it proved a served token, a block's cost, a block's served MACs where the
    census holds them, the GPU-seconds a block its sampling plan's own planner priced it at, where the plan states
    them, and every staged type a steady-state row of it was staged as, at any prover count, measured or not."""

    block: str
    k_log: int | None
    per_token: float
    cost: Cost
    macs: float | None = None
    planned: float | None = None
    staged_as: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    """A proof system's costs on a census, a row a block type; `range_basis` says what the range's other end is,
    `verifier_basis` which verifier's seconds the rows hold, `unread` the steady-state rows it was given whose
    staged Definition is no block type of the census (they are on no row and enter no figure), `cores` the host CPU
    cores a card its types are priced at, where its proof system states them, `lean` the statements the rows' Lean
    verifier seconds and proof bytes are measured on, and `lean_fitted` the rates fitted over them, seconds an AND gate
    and bytes a witness bit (None where no measured statement has the size)."""

    system: str
    rows: tuple[Row, ...]
    range_basis: str | None = None
    verifier_basis: str | None = None
    unread: tuple[Steady, ...] = ()
    cores: int | None = None
    lean: tuple[LeanVerify, ...] = ()
    lean_fitted: tuple[float | None, float | None] = (None, None)

    def above(self, row: Row) -> bool:
        """Whether `row`'s figure rests on a steady-state row timed at more cores than the table's host holds."""

        return self.cores is not None and row.cost.cores is not None and row.cost.cores > self.cores

    def gpu_seconds_per_token(self) -> float:
        """The prover GPU-seconds a served token, over the rows that have a figure."""

        return sum(r.per_token * r.cost.seconds for r in self.rows if r.cost.seconds is not None)

    def gpu_seconds_range(self) -> tuple[float, float]:
        """The total and the total with every type not measured at steady state at the steady-state rate, in order."""

        other = sum(r.per_token * (r.cost.seconds if r.cost.at_steady_rate is None else r.cost.at_steady_rate)
                    for r in self.rows if r.cost.seconds is not None)
        total = self.gpu_seconds_per_token()
        return min(total, other), max(total, other)


class ProofSystem(Protocol):
    """A proof system's prover on a workload: the blocks it proves a served token and what each costs."""

    @property
    def name(self) -> str: ...

    def table(self, census: Census) -> Table: ...


# ---- SPOT: a proof system at a sampling plan -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingPlan:
    """A sampling plan: over `period_s` seconds of `scope`'s serving (`serving_gpu_seconds` in all) at sampling rate
    `q`, the proof blocks it samples of each block type (`sampled`, by the plan's name of the type), the staged type
    of each (`staged`), where it names one, and the GPU-seconds a block its own planner priced each type at
    (`priced`), where the plan states them: every cell's, or none."""

    name: str
    scope: str
    period_s: float
    q: float
    serving_gpu_seconds: float
    sampled: Mapping[str, float]
    staged: Mapping[str, str]
    record: str
    priced: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def of(cls, doc: Mapping[str, Any], record: str) -> SamplingPlan:
        """The plan of a record's `mix.json`: each cell takes `taken` blocks of every sampled replay block, and may
        state its planner's GPU-seconds for them (`prove_s`, a statement's fixed seconds included)."""

        cells = doc["cells"]
        priced = {c["block"]: c["prove_s"] / c["taken"] if c["taken"] else 0.0
                  for c in cells if c.get("prove_s") is not None}
        bare = [c["block"] for c in cells if c["block"] not in priced]
        if priced and bare:
            raise ValueError(f"the sampling plan {record} states its planner's seconds (`prove_s`) for {len(priced)} "
                             f"cell(s) and not for {len(bare)}{_some(bare)}: a total at its prices would leave those "
                             "out. A plan prices every cell or none")
        return cls(doc["case"], doc["scope"], doc["period_s"], doc["q"], doc["gpu_seconds"],
                   {c["block"]: doc["replay_sample"] * c["taken"] for c in cells},
                   {c["block"]: c["type"] for c in cells if c.get("type")}, record, priced)


@dataclass(frozen=True)
class Spot:
    """SPOT over a proof system: `prover`'s cost of each block type, at the blocks `plan` samples a served token. A
    plan's cell is a row's by the block type's name, else by a staged type any steady-state row of the type was staged
    as, the one that priced it first (a plan names a block type no served run has taken by its staged type alone, and
    a newer row may stage it under another name); a plan and a census that don't cover each
    other are refused, since the total would price another workload. Each row keeps its cell's price in the plan
    (`Row.planned`), where the plan states one."""

    prover: ProofSystem
    plan: SamplingPlan
    serving: Serving
    name: str = "spot"

    def table(self, census: Census) -> Table:
        inner = self.prover.table(census)
        tokens = self.plan.serving_gpu_seconds / self.serving.gpu_seconds_per_token(census)
        by_staged = {staged: cell for cell, staged in self.plan.staged.items()}
        rows, used, bare = [], set(), []
        for r in inner.rows:
            cell = r.block if r.block in self.plan.sampled else next(
                (by_staged[n] for n in (r.cost.staged, *r.staged_as) if n in by_staged), None)
            if cell is None or cell in used:
                bare.append(r.block)
                continue
            used.add(cell)
            rows.append(replace(r, per_token=self.plan.sampled[cell] / tokens, planned=self.plan.priced.get(cell)))
        unread = sorted(set(self.plan.sampled) - used)
        if bare or unread:
            raise ValueError(
                f"the sampling plan {self.plan.record} and the census don't cover each other: the plan has no cell for "
                f"{len(bare)} block type(s){_some(bare)}, and the census no block type for {len(unread)} cell(s)"
                f"{_some(unread)}. It is another census's or cut's plan, or no labelled steady-state row names "
                "those cells' staged types")
        return Table(f"{self.name}[{inner.system}]", tuple(rows), inner.range_basis, inner.verifier_basis, inner.unread,
                     inner.cores, inner.lean, inner.lean_fitted)


def _some(names: Sequence[str]) -> str:
    return "" if not names else " (" + ", ".join(n[:60] for n in names[:3]) + (", ..." if len(names) > 3 else "") + ")"


# ---- the report --------------------------------------------------------------------------------------------------------------


def report(table: Table, serving_seconds: float) -> dict[str, Any]:
    """A table's rows, the share of the proving time each source covers, and the totals, given a served token's
    GPU-seconds. The totals are `measured` only when every block type is measured at steady state, and `partial` when a
    type has no figure: `missing` then names each such type, the most blocks first, with its share of the blocks, and
    of the MACs where every row holds them (a type with no time has no share of the proving time). `planned` is the
    same blocks at their sampling plan's own prices, where the plan states them (`_planned`). `above_cores` is the
    share of the proving time that rests on steady-state rows timed at more cores than the table's host holds
    (`_above_cores`). `unread` names each steady-state row that is on no block type's row: what holds it, the run that
    measured it, the type and Definition it staged, its prover count and cores, and a block's GPU-seconds where it is
    a steady-state measurement, else why it isn't. `lean_verifier_seconds_per_token` and `proof_bytes_per_token` are
    the Lean verifier's seconds and the proof's bytes a served token, with each source's share (`_scaled`), None
    where no statement was measured, and `lean_verify` names each measured statement, the most proving time first,
    with its block type's share of the proving time and the statement id it was read as (`form`). A row's
    `lean_mark` is its Lean verifier seconds' source as `LEAN_MARKS` words it, and `lean_record`, `lean_definition`,
    `lean_form` and `lean_statement_blocks` the measured statement they rest on (`_lean_statement`). A row's `definition` is the Definition the statements its figure was measured on are of, by its id,
    where the figure is from a steady-state row, the type's own or its family's; its note says which form of the
    type that Definition is. A row's `sizes` is each statement size its own steady-state rows timed (`Cost.sizes`: its
    blocks, a statement's GPU-seconds and what holds it, the pricing size first), None where they timed one."""

    total = table.gpu_seconds_per_token()
    blocks = sum(r.per_token for r in table.rows)
    low, high = table.gpu_seconds_range()

    def share(part: float, whole: float) -> float | None:
        return part / whole if whole else None

    rows = [{"block": r.block, "k_log": r.k_log, "per_token": r.per_token, "seconds": r.cost.seconds,
             "planned_seconds": r.planned, "source": r.cost.source, "record": r.cost.record, "provers": r.cost.provers,
             "cores": r.cost.cores, "statement_blocks": r.cost.blocks, "definition": r.cost.definition, "note": r.cost.note,
             "verifier_seconds": r.cost.verifier_seconds, "proof_bytes": r.cost.proof_bytes,
             "lean_verifier_seconds": r.cost.lean_seconds, "lean_source": r.cost.lean_source,
             "lean_mark": LEAN_MARKS.get(r.cost.lean_source or ""), **_lean_statement(r.cost.lean_statement),
             "proof_bytes_source": r.cost.proof_bytes_source,
             "sizes": [{"blocks": n, "statement_seconds": g, "record": rec} for n, g, rec in r.cost.sizes] or None,
             "time_share": None if r.cost.seconds is None else share(r.per_token * r.cost.seconds, total)}
            for r in table.rows]
    sources = []
    for source in SOURCES:
        mine = [r for r in table.rows if r.cost.source == source]
        seconds = sum(r.per_token * r.cost.seconds for r in mine if r.cost.seconds is not None)
        sources.append({"source": source, "block_types": len(mine), "per_token": sum(r.per_token for r in mine),
                        "block_share": share(sum(r.per_token for r in mine), blocks),
                        "gpu_seconds_per_token": seconds, "time_share": share(seconds, total)})
    verified = [r for r in table.rows if r.cost.verifier_seconds is not None]
    timed = {v.definition: v for v in table.lean}
    lean = [{"block": r.block, "time_share": None if r.cost.seconds is None else share(r.per_token * r.cost.seconds, total),
             "k_log": v.k_log, "blocks": v.blocks, "full": v.full, "seconds": list(v.seconds),
             "statement_seconds": v.statement_seconds, "proof_bytes": v.proof_bytes, "run": v.run, "host": v.host,
             "label": v.label, "form": v.form}
            for r in table.rows if (v := timed.get(definition_id(r.block)))]
    steady = sources[0]["time_share"] or 0.0
    known = [r.per_token * r.macs for r in table.rows if r.macs is not None]
    macs = sum(known) if len(known) == len(table.rows) else None
    missing = [{"block": r.block, "per_token": r.per_token, "block_share": share(r.per_token, blocks),
                "mac_share": None if macs is None or r.macs is None else share(r.per_token * r.macs, macs)}
               for r in sorted(table.rows, key=lambda r: -r.per_token) if r.cost.seconds is None]
    if missing:
        status = "partial"
    else:
        status = "measured" if table.rows and all(r.cost.source == STEADY for r in table.rows) else "modeled"
    unread = [{"record": s.record, "run": s.run, "type": s.type, "definition": s.definition, "provers": s.provers,
               "cores": s.cores, "seconds": s.seconds / s.blocks if s.measured else None, "why": s.refused}
              for s in table.unread]
    return {
        "system": table.system, "rows": sorted(rows, key=lambda r: -(r["time_share"] or 0.0)), "sources": sources,
        "unread": unread,
        "totals": {
            "status": status, "steady_time_share": steady, "above_cores": _above_cores(table), "missing": missing,
            "blocks_per_token": blocks, "gpu_seconds_per_token": total, "gpu_seconds_per_token_range": [low, high],
            "tokens_per_gpu_second": share(1.0, total),
            "tokens_per_gpu_second_range": [share(1.0, high), share(1.0, low)],
            "gpus_per_serving_gpu": total / serving_seconds,
            "gpus_per_serving_gpu_range": [low / serving_seconds, high / serving_seconds],
            "range_basis": table.range_basis, "verifier_basis": table.verifier_basis,
            "verifier_seconds_per_token": _covered(verified, lambda r: r.cost.verifier_seconds, blocks),
            "lean_verifier_seconds_per_token": _scaled(table.rows, lambda r: (r.cost.lean_seconds, r.cost.lean_source),
                                                       table.lean_fitted[0], LEAN_SECONDS_BASIS),
            "proof_bytes_per_token": _scaled(table.rows, lambda r: (r.cost.proof_bytes, r.cost.proof_bytes_source),
                                             table.lean_fitted[1], PROOF_BYTES_BASIS),
            "lean_verify": sorted(lean, key=lambda x: -(x["time_share"] or 0.0)),
            "planned": _planned(table, serving_seconds),
        },
    }


def _lean_statement(v: LeanVerify | None) -> dict[str, Any]:
    """The measured statement a row's Lean verifier seconds rest on: its run, Definition, statement id, and its
    blocks of a full statement; each None at the fitted rate."""

    return {"lean_record": v and v.run, "lean_definition": v and v.definition, "lean_form": v and v.form,
            "lean_statement_blocks": v and [v.blocks, v.full]}


def steady_rows(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A report's block types measured at steady state, a row each as the break-even planner reads a steady-state row
    (`benchmarks/spot_figures/breakeven/reprice.py`'s `steady`): the block type by its census name, a statement's
    GPU-seconds, the blocks it held, its provers, their cores and the run or art that holds the measurement. A type
    priced any other way (its family's row, a single session, the k_log model) is no measurement of it and gets no
    row."""

    return {"rows": [{"block": r["block"], "statement_seconds": r["seconds"] * r["statement_blocks"],
                      "blocks": r["statement_blocks"], "provers": r["provers"], "cores": r["cores"], "record": r["record"]}
                     for r in doc["rows"] if r["source"] == STEADY]}


def _above_cores(table: Table) -> dict[str, Any] | None:
    """How much of the table's proving time rests on steady-state rows timed at more cores than its host holds, None
    where its proof system states no host cores: the share of the proving time, the block types, the core counts
    those rows were timed at, and the same share and count for each source that holds one (a type's own row, or its
    family's)."""

    if table.cores is None:
        return None
    total = table.gpu_seconds_per_token()
    above = [r for r in table.rows if table.above(r)]

    def of(rows: Sequence[Row]) -> dict[str, Any]:
        return {"block_types": len(rows),
                "time_share": sum(r.per_token * r.cost.seconds for r in rows if r.cost.seconds is not None) / total
                if total else None}

    return {"cores": table.cores, **of(above), "timed_at": sorted({r.cost.cores for r in above}),
            "sources": [{"source": source, **of(mine)} for source in SOURCES
                        if (mine := [r for r in above if r.cost.source == source])]}


def _planned(table: Table, serving_seconds: float) -> dict[str, Any] | None:
    """The table's blocks at their sampling plan's own prices, None unless the plan prices every row: the totals, and
    where they leave the table's. Over the rows the table prices, each source's GPU-seconds a served token at both
    prices, and the row that holds most of the difference, with its share of it."""

    planned = [(r, r.planned) for r in table.rows if r.planned is not None]
    if not table.rows or len(planned) != len(table.rows):
        return None
    total = sum(r.per_token * p for r, p in planned)
    both = [(r, s, p) for r, p in planned if (s := r.cost.seconds) is not None]
    gap = sum(r.per_token * (s - p) for r, s, p in both)
    top = max(both, key=lambda x: abs(x[0].per_token * (x[1] - x[2])), default=None)
    return {
        "gpu_seconds_per_token": total, "gpus_per_serving_gpu": total / serving_seconds,
        "sources": [{"source": source, "block_types": len(mine),
                     "gpu_seconds_per_token": sum(r.per_token * s for r, s, _ in mine),
                     "planned_gpu_seconds_per_token": sum(r.per_token * p for r, _, p in mine)}
                    for source in SOURCES if (mine := [x for x in both if x[0].cost.source == source])],
        "largest": None if top is None or not gap else {
            "block": top[0].block, "seconds": top[1], "planned_seconds": top[2],
            "difference_share": top[0].per_token * (top[1] - top[2]) / gap},
    }


def _covered(rows: Sequence[Row], value: Any, blocks: float) -> dict[str, Any] | None:
    """A per-block quantity summed a served token over the rows a record holds it for, with the blocks they cover; None
    where no record holds it."""

    if not rows:
        return None
    return {"value": sum(r.per_token * value(r) for r in rows), "block_types": len(rows),
            "block_share": sum(r.per_token for r in rows) / blocks if blocks else None}



def _scaled(rows: Sequence[Row], figure: Callable[[Row], tuple[float | None, str | None]], fitted: float | None,
            basis: str) -> dict[str, Any] | None:
    """A per-block Lean figure (`figure`: a row's figure and its source) summed a served token, None where no statement
    was measured: `measured` only when every block type has a record of its own, else `modeled`, with the block types
    each source prices and its share of the figure (`sources`, by `LEAN_SOURCES`), those left out (no size), the
    fitted rate a unit of size, and the basis."""

    priced = [(r, x, source) for r in rows for x, source in [figure(r)] if x is not None]
    if not priced:
        return None
    total = sum(r.per_token * x for r, x, _ in priced)
    sources = {source: {"block_types": len(mine), "share": sum(r.per_token * x for r, x, _ in mine) / total if total else None}
               for source in LEAN_SOURCES for mine in [[p for p in priced if p[2] == source]]}
    return {"value": total, "status": "measured" if sources[LEAN_MEASURED]["block_types"] == len(rows) else "modeled",
            "sources": sources, "unpriced_block_types": len(rows) - len(priced), "fitted_rate": fitted, "basis": basis}
