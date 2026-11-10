"""`verity bench prove`'s rules: what a proof system's prover costs on a workload, as one labelled table a block type.

Standard library only: `prove_records` reads the inputs from the store's records and renders the table.

**The workload** is its block census (`Census`): every block type (a Definition at its statics) with the proof blocks of
it a served token takes and log2 of its witness bits (k_log), and the MACs a served token takes. A cut (a `partition`
card) replaces the block types of the served rows it recuts.

**A proof system** (`ProofSystem`) turns a census into a `Table`: a row a block type with the blocks of it proved a
served token, the prover GPU-seconds a block and where that figure is from (`SOURCES`, best first):

- `steady`: a proof server's report, statements of the type proved back to back until its statement cache's hit rate is
  flat, every one accepted;
- `single`: one warm process's sessions on the type alone, a block's marginal seconds plus its share of a full
  statement's fixed seconds;
- `k_log`: no measurement of the type, so the single-session types' median seconds a witness bit times its 2^k_log
  bits, and a statement's fixed seconds on their least-squares line in k_log;
- `missing`: neither a measurement nor a k_log, so no figure, and the totals leave the type out and say so.

`Flock` proves every block: its row counts are the census's. `Spot` wraps a proof system with a sampling plan: the
same costs, and the blocks the plan samples a served token. A plan that states the prices its own planner sized the
sample at gets a second total beside the table's, the same blocks at those prices (`planned`), with where the two
differ: a plan's sample fits its planner's prices, and the table's sources may hold others.

**The totals** (`report`) are the same for every system: the prover GPU-seconds a served token, the served tokens a
prover GPU covers a second (its inverse) and the prover GPUs a serving GPU needs (over a served token's GPU-seconds,
`Serving`, a stated and modeled input), with the share of the proving time each source covers. Their range's other end
prices every type not measured at steady state at the steady-state types' median seconds a witness bit. The totals are
`measured` only when every block type is measured at steady state, `partial` when a type has no figure (they leave its
proving out, so they name each such type with its share of the blocks, and of the MACs where the census holds them: a
type with no time has no share of the proving time), and `modeled` otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import linear_regression, median
from typing import Any, Protocol

STEADY, SINGLE, MODELED, MISSING = "steady", "single", "k_log", "missing"
#: where a block type's proving time is from, best first, as the table words it
SOURCES = {STEADY: "measured at steady state", SINGLE: "measured in a single session",
           MODELED: "modeled from k_log", MISSING: "missing"}
#: a full statement's witness bits: a block type's statement holds 2^35 / 2^k_log of its blocks
STATEMENT_BITS = 2**35
#: a MAC is a fused multiply-add, which the data sheets' rates count as two FLOP
FLOP_PER_MAC = 2
SCHEMA = "verity-bench/prove/v1"
#: the verifier a steady-state report times
VERIFIER = ("the proof server's own verifier (`flock-circuit serve --zk`), its wall seconds a statement over the "
            "statement's blocks; the Lean verifier's seconds aren't in these records")


# ---- the workload ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """A block type of a census: a Definition at its statics as the census spells it, the served rows it is cut from
    (`kind`, which a cut replaces by), the proof blocks of it a served token takes, log2 of its witness bits, and a
    block's served MACs where the census holds them."""

    name: str
    kind: str
    per_token: float
    k_log: int | None = None
    macs: float | None = None


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
        blocks = tuple(Block(name, ", ".join(sorted(d["of"])), count / tokens, d.get("k_log"), d.get("macs"))
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
        cut = tuple(Block(x["block"], x["kind"], x["per_token"], x.get("k_log"), x.get("macs"))
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


# ---- the table ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """What proving one block of a type costs, and where each figure is from: the prover GPU-seconds (None when
    `missing`), the run or art that holds it, the staged type that measured it as its record names it (a sampling plan
    names its cells' the same way), the same block at the steady-state types' median seconds a witness bit (the range's
    other end; None for a type measured at steady state), and the verifier's seconds and the proof's bytes where a record
    holds them."""

    source: str
    seconds: float | None = None
    record: str | None = None
    note: str = ""
    staged: str | None = None
    at_steady_rate: float | None = None
    verifier_seconds: float | None = None
    proof_bytes: float | None = None


@dataclass(frozen=True)
class Row:
    """A block type's row: the blocks of it proved a served token, a block's cost, a block's served MACs where the
    census holds them, and the GPU-seconds a block its sampling plan's own planner priced it at, where the plan states
    them."""

    block: str
    k_log: int | None
    per_token: float
    cost: Cost
    macs: float | None = None
    planned: float | None = None


@dataclass(frozen=True)
class Table:
    """A proof system's costs on a census, a row a block type; `range_basis` says what the range's other end is and
    `verifier_basis` which verifier's seconds the rows hold."""

    system: str
    rows: tuple[Row, ...]
    range_basis: str | None = None
    verifier_basis: str | None = None

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

    name: str

    def table(self, census: Census) -> Table: ...


# ---- Flock: every block proved ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """A block type's single-session measurement (a `proof_overhead` card's block): a block's marginal seconds, a
    statement's fixed seconds, and the blocks a full statement holds at the k_log it was staged at."""

    block: str
    k_log: int
    seconds_per_block: float
    fixed_seconds: float
    statement_blocks: int


@dataclass(frozen=True)
class Steady:
    """One staged type of a proof server's report: the Definition it staged (by its id), its statements of `blocks`
    blocks where a full one holds `full`, the mean GPU-seconds and verifier's seconds a statement, and why the report
    isn't a steady-state measurement (`refused`, None when it is one)."""

    type: str
    definition: str
    k_log: int
    blocks: int
    full: int
    run: str
    report: str
    gpu_seconds: float | None = None
    verifier_seconds: float | None = None
    refused: str | None = None

    @property
    def measured(self) -> bool:
        return self.refused is None and self.gpu_seconds is not None

    @property
    def fill(self) -> str:
        return "full statements" if self.blocks == self.full else f"statements {self.blocks}/{self.full} full"


def _fullest(s: Steady) -> tuple[int, str]:
    return s.blocks, s.run


def _newest(s: Steady) -> tuple[str, int]:
    return s.run, s.blocks


_FN = re.compile(r'\{"fn":"([^"]*)"\}')
_F64 = re.compile(r'\{"f64":"([^"]*)"\}')
_BOOL = re.compile(r"\b(true|false)\b")
_K = re.compile(r"[{,]K=(\d+)")


def definition_id(spec: str) -> str:
    """A census's name of a block type, in the spelling of the Definition's id a staged statement records: a function's
    and a float's static written out, Python's booleans and tuples."""

    spelled = _F64.sub(lambda m: repr(float.fromhex(m[1])), _FN.sub(r"\1", spec))
    return _BOOL.sub(lambda m: m[1].capitalize(), spelled).replace("[", "(").replace("]", ")")


def _family(block: str) -> tuple[str, int | None]:
    """A block type's Definition and K: the types a single-session measurement prices beside its own."""

    k = _K.search(block)
    return block.split("{")[0], int(k[1]) if k else None


@dataclass(frozen=True)
class Flock:
    """Flock's prover with every block proved. A type takes its steady-state measurement at the fullest statement any
    run measured, the newest such run's; else `card`'s single-session one: its own block's, or its Definition and K's at
    2^Δk the seconds and 2^-Δk the statement, Δk between the two k_logs as the census holds them; else the k_log model
    fitted on the card's types. A type no report measured names its newest attempt."""

    steady: Sequence[Steady] = ()
    sessions: Sequence[Session] = ()
    card: str | None = None
    name: str = "flock"

    def table(self, census: Census) -> Table:
        measured: dict[str, Steady] = {}
        tried: dict[str, Steady] = {}
        for s in self.steady:
            best, order = (measured, _fullest) if s.measured else (tried, _newest)
            if (old := best.get(s.definition)) is None or order(s) > order(old):
                best[s.definition] = s
        rates = [s.gpu_seconds / s.blocks / 2**s.k_log for s in measured.values() if s.blocks == s.full]
        steady_rate = median(rates) if rates else None
        single = self._single(census)
        rows = []
        for b in census.blocks:
            spelled = definition_id(b.name)
            if s := measured.get(spelled):
                cost = Cost(STEADY, s.gpu_seconds / s.blocks, s.run, f"`{s.type}`, {s.fill} ({s.report})", s.type,
                            verifier_seconds=None if s.verifier_seconds is None else s.verifier_seconds / s.blocks)
            else:
                cost = single(b)
                if s := tried.get(spelled):
                    why = f"steady state tried as `{s.type}` in {s.run} ({s.report}): {s.refused}"
                    cost = replace(cost, staged=s.type, note="; ".join(filter(None, [cost.note, why])))
                if steady_rate is not None and b.k_log is not None and cost.seconds is not None:
                    cost = replace(cost, at_steady_rate=steady_rate * 2**b.k_log)
            rows.append(Row(b.name, b.k_log, b.per_token, cost, b.macs))
        basis = (None if steady_rate is None else
                 f"every type not measured at steady state priced at the full-statement steady-state types' median, "
                 f"{steady_rate:.3g} GPU-seconds a witness bit")
        return Table(self.name, tuple(rows), basis, VERIFIER)

    def _single(self, census: Census) -> Callable[[Block], Cost]:
        """Each block type's cost off the steady state: `single`, `k_log` or `missing`."""

        by_name = {s.block: s for s in self.sessions}
        families = {_family(s.block): s for s in self.sessions}
        census_k = {b.name: b.k_log for b in census.blocks if b.k_log is not None}
        held = {_family(s.block): census_k[s.block] for s in self.sessions if s.block in census_k}
        ks = [s.k_log for s in families.values()]
        model = None
        if len(set(ks)) > 1:
            rate = median(s.seconds_per_block / 2**s.k_log for s in families.values())
            slope, intercept = linear_regression(ks, [s.fixed_seconds for s in families.values()])
            model = rate, slope, intercept

        def cost(b: Block) -> Cost:
            if s := by_name.get(b.name):
                return Cost(SINGLE, s.seconds_per_block + s.fixed_seconds / s.statement_blocks, self.card)
            if s := families.get(_family(b.name)):
                dk = 0 if b.k_log is None else b.k_log - held.get(_family(b.name), s.k_log)
                full = s.statement_blocks << -dk if dk < 0 else max(1, s.statement_blocks >> dk)
                return Cost(SINGLE, s.seconds_per_block * 2.0**dk + s.fixed_seconds / full, self.card,
                            f"its Definition and K's, `{s.block}`, at 2^{dk} its seconds")
            if model and b.k_log is not None:
                rate, slope, intercept = model
                fixed = max(0.0, intercept + slope * b.k_log)
                return Cost(MODELED, rate * 2**b.k_log + fixed / max(1, STATEMENT_BITS >> b.k_log), self.card)
            return Cost(MISSING)

        return cost


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
        priced = {c["block"]: c["prove_s"] / c["taken"] for c in cells if c.get("prove_s") is not None and c["taken"]}
        bare = [c["block"] for c in cells if c["taken"] and c["block"] not in priced]
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
    plan's cell is a row's by the block type's name, else by the staged type that measured or tried the row (a plan
    names a block type no served run has taken by its staged type alone); a plan and a census that don't cover each
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
            cell = r.block if r.block in self.plan.sampled else by_staged.get(r.cost.staged or "")
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
                f"{_some(unread)}. It is another census's or cut's plan, or no steady-state report of record names "
                "those cells' staged types")
        return Table(f"{self.name}[{inner.system}]", tuple(rows), inner.range_basis, inner.verifier_basis)


def _some(names: Sequence[str]) -> str:
    return "" if not names else " (" + ", ".join(n[:60] for n in names[:3]) + (", ..." if len(names) > 3 else "") + ")"


# ---- the report --------------------------------------------------------------------------------------------------------------


def report(table: Table, serving_seconds: float) -> dict[str, Any]:
    """A table's rows, the share of the proving time each source covers, and the totals, given a served token's
    GPU-seconds. The totals are `measured` only when every block type is measured at steady state, and `partial` when a
    type has no figure: `missing` then names each such type, the most blocks first, with its share of the blocks, and
    of the MACs where every row holds them (a type with no time has no share of the proving time). `planned` is the
    same blocks at their sampling plan's own prices, where the plan states them (`_planned`)."""

    total = table.gpu_seconds_per_token()
    blocks = sum(r.per_token for r in table.rows)
    low, high = table.gpu_seconds_range()

    def share(part: float, whole: float) -> float | None:
        return part / whole if whole else None

    rows = [{"block": r.block, "k_log": r.k_log, "per_token": r.per_token, "seconds": r.cost.seconds,
             "planned_seconds": r.planned, "source": r.cost.source, "record": r.cost.record, "note": r.cost.note,
             "verifier_seconds": r.cost.verifier_seconds, "proof_bytes": r.cost.proof_bytes,
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
    sized = [r for r in table.rows if r.cost.proof_bytes is not None]
    steady = sources[0]["time_share"] or 0.0
    macs = None if any(r.macs is None for r in table.rows) else sum(r.per_token * r.macs for r in table.rows)
    missing = [{"block": r.block, "per_token": r.per_token, "block_share": share(r.per_token, blocks),
                "mac_share": None if macs is None else share(r.per_token * r.macs, macs)}
               for r in sorted(table.rows, key=lambda r: -r.per_token) if r.cost.seconds is None]
    if missing:
        status = "partial"
    else:
        status = "measured" if table.rows and all(r.cost.source == STEADY for r in table.rows) else "modeled"
    return {
        "system": table.system, "rows": sorted(rows, key=lambda r: -(r["time_share"] or 0.0)), "sources": sources,
        "totals": {
            "status": status, "steady_time_share": steady, "missing": missing,
            "blocks_per_token": blocks, "gpu_seconds_per_token": total, "gpu_seconds_per_token_range": [low, high],
            "tokens_per_gpu_second": share(1.0, total),
            "tokens_per_gpu_second_range": [share(1.0, high), share(1.0, low)],
            "gpus_per_serving_gpu": total / serving_seconds,
            "gpus_per_serving_gpu_range": [low / serving_seconds, high / serving_seconds],
            "range_basis": table.range_basis, "verifier_basis": table.verifier_basis,
            "verifier_seconds_per_token": _covered(verified, lambda r: r.cost.verifier_seconds, blocks),
            "proof_bytes_per_token": _covered(sized, lambda r: r.cost.proof_bytes, blocks),
            "planned": _planned(table, serving_seconds),
        },
    }


def _planned(table: Table, serving_seconds: float) -> dict[str, Any] | None:
    """The table's blocks at their sampling plan's own prices, None unless the plan prices every row: the totals, and
    where they leave the table's. Over the rows the table prices, each source's GPU-seconds a served token at both
    prices, and the row that holds most of the difference, with its share of it."""

    if not table.rows or any(r.planned is None for r in table.rows):
        return None
    total = sum(r.per_token * r.planned for r in table.rows)
    both = [r for r in table.rows if r.cost.seconds is not None]
    gap = sum(r.per_token * (r.cost.seconds - r.planned) for r in both)
    top = max(both, key=lambda r: abs(r.per_token * (r.cost.seconds - r.planned)), default=None)
    return {
        "gpu_seconds_per_token": total, "gpus_per_serving_gpu": total / serving_seconds,
        "sources": [{"source": source, "block_types": len(mine),
                     "gpu_seconds_per_token": sum(r.per_token * r.cost.seconds for r in mine),
                     "planned_gpu_seconds_per_token": sum(r.per_token * r.planned for r in mine)}
                    for source in SOURCES if (mine := [r for r in both if r.cost.source == source])],
        "largest": None if top is None or not gap else {
            "block": top.block, "seconds": top.cost.seconds, "planned_seconds": top.planned,
            "difference_share": top.per_token * (top.cost.seconds - top.planned) / gap},
    }


def _covered(rows: Sequence[Row], value: Any, blocks: float) -> dict[str, Any] | None:
    """A per-block quantity summed a served token over the rows a record holds it for, with the blocks they cover; None
    where no record holds it."""

    if not rows:
        return None
    return {"value": sum(r.per_token * value(r) for r in rows), "block_types": len(rows),
            "block_share": sum(r.per_token for r in rows) / blocks if blocks else None}

