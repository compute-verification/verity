"""Flock's prover priced on a workload (`verity bench prove --system flock`): the measurements it reads, a block type's
single session (`Session`), a row of the proof server's steady-state rate (`Steady`), a statement proved from warm
sessions (`Warm`) and a statement the Lean verifier accepted (`LeanVerify`), and `Flock`, the proof system that turns a census into a table from them. The
sources, the totals and the Lean figures' rules are `prove`'s.

Standard library only: `prove_records` reads these measurements from the store's records.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from statistics import linear_regression, median

from verity_bench.prove import (
    FAMILY,
    HOST_CORES,
    LEAN_FAMILY,
    LEAN_FITTED,
    LEAN_MEASURED,
    MISSING,
    MODELED,
    SINGLE,
    STATEMENT_BITS,
    STEADY,
    UNSTATED_CORES,
    WARM,
    Block,
    Census,
    Cost,
    Row,
    Table,
    definition_id,
)

#: the verifier a steady-state row times
VERIFIER = ("the proof server's own verifier (`flock-circuit serve --zk`), its seconds a statement (a row's `verify_s`) "
            "over the statement's blocks; the Lean verifier's seconds aren't in these figures")


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
class Warm:
    """A statement of a block type proved on one card from warm sessions (`stage_prove.sh`, `window.py prove`): the
    Definition of the block type it was staged for (`definition`, by its id), `blocks` blocks where a full statement
    holds `full`, after `warm` warm sessions the median GPU-seconds of its `runs` timed ones (`seconds`), every session
    accepted. `statement` is its name in the run, `run` the run that proved it."""

    definition: str
    k_log: int
    blocks: int
    full: int
    seconds: float
    warm: int
    runs: int
    run: str
    statement: str

    @property
    def note(self) -> str:
        return (f"`{self.statement}`, {self.blocks} of a full statement's {self.full} blocks, on one card: after "
                f"{self.warm} warm session{'' if self.warm == 1 else 's'}, the median of {self.runs} timed ones "
                f"(WARM={self.warm}, RUNS={self.runs})")


@dataclass(frozen=True)
class Steady:
    """One row of a proof server's rate, a staged type at a prover count: the Definition of the block type it was
    staged for (`definition`, by its id), its statements of `blocks` blocks where a full one holds `full`, the
    GPU-seconds and the verifier's seconds a statement, the provers that proved them, the host CPU cores they held for
    their card (`UNSTATED_CORES` where its record states none), and why the row isn't a steady-state measurement
    (`refused`, None when it is one). `run` is the run that measured it and `report` the file that holds it: in the
    run's own files, or in `art` where an art holds the run's rows in their place. `form` is the Definition its
    statements are of, by its id, where that is another than `definition` (the type's Boolean version, say); None
    where they are of `definition` itself."""

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
    provers: int = 1
    art: str | None = None
    cores: int = UNSTATED_CORES
    form: str | None = None

    @property
    def measured(self) -> bool:
        return self.refused is None and self.gpu_seconds is not None

    @property
    def seconds(self) -> float:
        """A measured row's GPU-seconds a statement; ValueError on a row that measured none."""

        if self.gpu_seconds is None:
            raise ValueError(f"`{self.type}` in {self.run} measured no GPU-seconds: {self.refused}")
        return self.gpu_seconds

    @property
    def proved(self) -> str:
        """The Definition its statements are of, by its id: `form`, else `definition`."""

        return self.form or self.definition

    @property
    def record(self) -> str:
        """What holds the row: its art, else its run."""

        return self.art or self.run

    @property
    def held(self) -> str:
        """The file that holds the row, with the art it is in where the run's own files don't hold it."""

        return self.report if self.art is None else f"{self.report} of {self.art[:16]}"

    @property
    def fill(self) -> str:
        return "full statements" if self.blocks == self.full else f"statements {self.blocks}/{self.full} full"

    @property
    def at(self) -> str:
        return f"{self.provers} prover{'' if self.provers == 1 else 's'}"


@dataclass(frozen=True)
class LeanVerify:
    """A statement the Lean verifier accepted, as a `lean-verify` record holds it: the staged Definition (by its id),
    its `blocks` blocks of 2^k_log witness bits each where a full statement holds `full`, each run's wall seconds (a
    run is one process of the Lean verifier's `verify`: the statement read and set up, and the one session verified), the bytes
    of the session's proof files, the run that measured it, its host as the report words it, what the accept holds
    (`label`, the record's own words: on the provisional unit cover the verifier compares no Program with the circuit),
    and the statement id it was verified as (`form`: `LEAN_FORMS` words the two it knows, the unrolled circuit file
    and its templates; None where the record names none)."""

    definition: str
    k_log: int
    blocks: int
    full: int
    seconds: tuple[float, ...]
    proof_bytes: int
    run: str
    host: str
    label: str
    form: str | None = None

    @property
    def bits(self) -> int:
        """The statement's witness bits."""

        return self.blocks << self.k_log

    @property
    def statement_seconds(self) -> float:
        """The verifier's seconds on the statement: the median of its runs."""

        return median(self.seconds)


@dataclass(frozen=True)
class _LeanFigure:
    """One Lean figure's prices: a block's figure from its own statement (`value` over the statement's blocks), else
    its Definition's measured statements' figure a unit of size (`statement`: a statement's size), the dearest of them
    so a family's cheapest member never prices the rest, times the block's `size`; else the rate fitted over every
    measured statement that has a size, the sum of their figures over the sum of their sizes, times the block's `size`;
    else none, where the block has no size or no measured statement does."""

    timed: Mapping[str, LeanVerify]
    value: Callable[[LeanVerify], float]
    statement: Callable[[LeanVerify], float | None]
    size: Callable[[Block], float | None]

    @property
    def sized(self) -> list[tuple[LeanVerify, float]]:
        return [(v, n) for v in self.timed.values() if (n := self.statement(v))]

    @property
    def fitted(self) -> float | None:
        sized = self.sized
        return sum(self.value(v) for v, _ in sized) / sum(n for _, n in sized) if sized else None

    def prices(self) -> Callable[[Block], tuple[float, str, LeanVerify | None] | None]:
        """A block's figure, its source and the measured statement it rests on (None at the fitted rate)."""

        family: dict[str, tuple[float, str, LeanVerify]] = {}
        for v, n in self.sized:
            name = _family(v.definition)[0]
            rate = (self.value(v) / n, v.definition, v)
            if (old := family.get(name)) is None or rate[:2] > old[:2]:
                family[name] = rate
        fitted = self.fitted

        def price(b: Block) -> tuple[float, str, LeanVerify | None] | None:
            if (own := self.timed.get(definition_id(b.name))) is not None:
                return self.value(own) / own.blocks, LEAN_MEASURED, own
            if (n := self.size(b)) is None:
                return None
            if kin := family.get(_family(b.name)[0]):
                return kin[0] * n, LEAN_FAMILY, kin[2]
            return None if fitted is None else (fitted * n, LEAN_FITTED, None)

        return price


def _lean_figures(census: Census, timed: Mapping[str, LeanVerify]) -> tuple[_LeanFigure, _LeanFigure]:
    """The Lean verifier's seconds, by AND gates (a measured statement's: its blocks' census `ands`), and the proof's
    bytes, by witness bits."""

    ands = {definition_id(b.name): b.ands for b in census.blocks}

    def statement_ands(v: LeanVerify) -> float | None:
        n = ands.get(v.definition)
        return None if n is None else v.blocks * n

    return (_LeanFigure(timed, lambda v: v.statement_seconds, statement_ands, lambda b: b.ands),
            _LeanFigure(timed, lambda v: v.proof_bytes, lambda v: v.bits,
                        lambda b: None if b.k_log is None else 2**b.k_log))


def _preferred(s: Steady) -> tuple[int, int, str]:
    """The order a type's steady-state measurements price it in, among rows at the host's cores or more: the fewest
    cores (a row at the host's own before any above it), then the fullest statement, then the newest run."""

    return -s.cores, s.blocks, s.run


def _newest(s: Steady) -> tuple[str, int]:
    return s.run, s.blocks


def _sizes(s: Steady, rows: Sequence[Steady]) -> list[Steady]:
    """The rows of `rows` beside `s`, the row that prices its type, that timed statements of another size at its cores
    in its Definition, one a size, the most blocks first: `s`'s run's where it timed that size, else the newest."""

    by: dict[int, Steady] = {}
    for x in rows:
        if x.blocks != s.blocks and x.cores == s.cores and x.proved == s.proved:
            old = by.get(x.blocks)
            if old is None or ((x.run == s.run), x.run) > ((old.run == s.run), old.run):
                by[x.blocks] = x
    return [by[n] for n in sorted(by, reverse=True)]


_K = re.compile(r"[{,]K=(\d+)")


def _family(block: str) -> tuple[str, int | None]:
    """A block type's Definition and K: the types a single-session or steady-state measurement prices beside its own."""

    k = _K.search(block)
    return block.split("{")[0], int(k[1]) if k else None


def _above(s: Steady, cores: int) -> str:
    """What a figure's note says of a row timed at more cores than the host's `cores`; nothing of one at the host's."""

    return f"; timed at {s.cores} cores, above the host's {cores}" if s.cores > cores else ""


def _form(s: Steady, boolean: bool) -> str:
    """How a figure's note names the Definition a row's statements are of: by its id alone where it is the Definition
    of the block type the row was staged for; else as that type's Boolean version where the tree registers it as one
    (`boolean`), and as another Definition where it doesn't."""

    if s.form is None:
        return f"`{s.definition}`"
    return f"{'its Boolean version' if boolean else 'another Definition'}, `{s.form}`"


def _kin(block: Block, kin: Sequence[Steady], cores: int, form: Callable[[Steady], str]) -> Cost:
    """`block` priced from the steady-state rows of its Definition and K at other statics, each the row that prices
    its own type on a host of `cores` cores a card: each at 2^Δk its seconds a block, Δk from the row's k_log to the
    block's, and the dearest of them, so a family's cheapest member never prices the rest. Its note names that row's
    block type and, where the row was staged in another Definition, that one after it (`form`: a row's `_form`). The
    verifier's seconds stay the type's own: none."""

    def dk(s: Steady) -> int:
        return 0 if block.k_log is None else block.k_log - s.k_log

    s = max(kin, key=lambda s: (s.seconds / s.blocks * 2.0**dk(s), s.definition))
    others = f"; the dearest of {len(kin)} measured" if len(kin) > 1 else ""
    named = form(s) if s.form is None else f"`{s.definition}` in {form(s)}"
    return Cost(FAMILY, s.seconds / s.blocks * 2.0**dk(s), s.record,
                f"its Definition and K's, `{s.type}` ({named}), {s.at}, {s.fill}, at 2^{dk(s)} its seconds{others}"
                f"{_above(s, cores)}", s.type, provers=s.provers, cores=s.cores, definition=s.proved)


@dataclass(frozen=True)
class Flock:
    """Flock's prover with every block proved, each type priced at `provers` provers on a host that gives a card
    `cores` CPU cores. A type takes its steady-state measurement from the rows at that prover count alone, and among
    them from those timed at `cores` cores: the fullest statement any of them measured, the newest such run's (a row
    an art holds is its run's). Its rows at those cores that timed another statement size are its `sizes`, each size
    the row of the pricing row's run where that run timed it, else the newest, and its note names them. A type those
    rows measured only at more cores takes the rows at the fewest cores
    above, by the same order, and its note says which; a row timed at fewer cores prices nothing, since the host's
    share is the least a prover holds. Else it takes its own statements proved from warm sessions (`warm`), the fullest
    statement's, the newest run's among equals. Else it takes `card`'s single-session measurement: its own block's, or its
    Definition and K's at 2^Δk the seconds and 2^-Δk the statement, Δk between the two k_logs as the census holds
    them; else the k_log model fitted on the card's types. A type no row measured for it names its newest attempt at
    that prover count, the other prover counts that measured it, and the core counts under the host's that did. A row
    whose staged Definition is no block type of the census, at any prover count, is the table's `unread` and enters no
    figure, the range's other end included. `lean`: the statements the Lean verifier accepted, a block type's the
    newest run's; one of no block type of the census measures nothing. `boolean`: whether the tree registers a
    Definition (by its id) as the Boolean version of a census type (by its census name). A row staged in another
    Definition than its type's own is named as the type's Boolean version only where `boolean` says it is one, and as
    another Definition where it says not or none is given."""

    steady: Sequence[Steady] = ()
    sessions: Sequence[Session] = ()
    card: str | None = None
    name: str = "flock"
    provers: int = 1
    cores: int = HOST_CORES
    #: (staged Definition, staged type) of superseded rows: they price nothing, but a plan's cell may still name a type
    #: by the staged type only they were staged as
    aliases: Sequence[tuple[str, str]] = ()
    lean: Sequence[LeanVerify] = ()
    boolean: Callable[[str, str], bool] | None = None
    warm: Sequence[Warm] = ()

    def table(self, census: Census) -> Table:
        measured: dict[str, Steady] = {}
        tried: dict[str, Steady] = {}
        elsewhere: dict[str, set[int]] = {}
        under: dict[str, set[int]] = {}
        held = {definition_id(b.name): b.name for b in census.blocks}

        def form(s: Steady) -> str:
            return _form(s, s.form is not None and self.boolean is not None and self.boolean(held[s.definition], s.form))

        unread = []
        names: dict[str, set[str]] = {}
        sized: dict[str, list[Steady]] = {}
        for definition, type_ in self.aliases:
            if definition in held:
                names.setdefault(definition, set()).add(type_)
        for s in self.steady:
            if s.definition not in held:
                unread.append(s)
                continue
            names.setdefault(s.definition, set()).add(s.type)
            if s.provers != self.provers:
                if s.measured:
                    elsewhere.setdefault(s.definition, set()).add(s.provers)
                continue
            if s.measured and s.cores < self.cores:
                under.setdefault(s.definition, set()).add(s.cores)
                continue
            if s.measured:
                sized.setdefault(s.definition, []).append(s)
            best, order = (measured, _preferred) if s.measured else (tried, _newest)
            if (old := best.get(s.definition)) is None or order(s) > order(old):
                best[s.definition] = s
        rates = [s.seconds / s.blocks / 2**s.k_log for s in measured.values() if s.blocks == s.full]
        steady_rate = median(rates) if rates else None
        families: dict[tuple[str, int | None], list[Steady]] = {}
        for s in measured.values():
            families.setdefault(_family(s.definition), []).append(s)
        single = self._single(census)
        warmed: dict[str, Warm] = {}
        for w in self.warm:
            if (kept := warmed.get(w.definition)) is None or (w.blocks, w.run) > (kept.blocks, kept.run):
                warmed[w.definition] = w
        timed: dict[str, LeanVerify] = {}
        for v in self.lean:
            if v.definition in held and ((prior := timed.get(v.definition)) is None or v.run > prior.run):
                timed[v.definition] = v
        lean_seconds, lean_bytes = _lean_figures(census, timed)
        seconds_of, bytes_of = lean_seconds.prices(), lean_bytes.prices()
        rows = []
        for b in census.blocks:
            spelled = definition_id(b.name)
            if m := measured.get(spelled):
                where = m.held if m.art is None else f"{m.held}, measured in {m.run}"
                others = _sizes(m, sized[spelled])
                also = "".join(f"; another size, {o.fill} ({o.record})" for o in others)
                cost = Cost(STEADY, m.seconds / m.blocks, m.record,
                            f"`{m.type}` ({form(m)}), {m.at}, {m.fill} ({where}){also}{_above(m, self.cores)}", m.type,
                            verifier_seconds=None if m.verifier_seconds is None else m.verifier_seconds / m.blocks,
                            provers=m.provers, cores=m.cores, blocks=m.blocks, definition=m.proved,
                            sizes=tuple((x.blocks, x.seconds, x.record) for x in [m, *others]) if others else ())
            else:
                kin = families.get(_family(spelled), [])
                if proved := warmed.get(spelled):
                    cost = Cost(WARM, proved.seconds / proved.blocks, proved.run, proved.note, blocks=proved.blocks)
                else:
                    cost = _kin(b, kin, self.cores, form) if kin else single(b)
                why = []
                if t := tried.get(spelled):
                    why.append(f"steady state tried as `{t.type}` at {t.at} in {t.run} ({t.held}): {t.refused}")
                    cost = replace(cost, staged=t.type)
                if counts := elsewhere.get(spelled):
                    why.append(f"measured at steady state at {', '.join(map(str, sorted(counts)))} provers, which "
                               f"price no type at {self.provers}")
                if fewer := under.get(spelled):
                    why.append(f"measured at steady state at {', '.join(map(str, sorted(fewer)))} cores, which price no "
                               f"type at {self.cores}")
                if why:
                    cost = replace(cost, note="; ".join(filter(None, [cost.note, *why])))
                if steady_rate is not None and b.k_log is not None and cost.seconds is not None:
                    cost = replace(cost, at_steady_rate=steady_rate * 2**b.k_log)
            seconds, size = seconds_of(b) or (None, None, None), bytes_of(b) or (None, None, None)
            cost = replace(cost, lean_seconds=seconds[0], lean_source=seconds[1], proof_bytes=size[0],
                           proof_bytes_source=size[1], lean_statement=seconds[2])
            rows.append(Row(b.name, b.k_log, b.per_token, cost, b.macs, staged_as=tuple(sorted(names.get(spelled, ())))))
        basis = (None if steady_rate is None else
                 f"every type not measured at steady state at {self.provers} prover{'' if self.provers == 1 else 's'} "
                 f"priced at the full-statement steady-state types' median, "
                 f"{steady_rate:.3g} GPU-seconds a witness bit")
        return Table(self.name, tuple(rows), basis, VERIFIER, tuple(unread), self.cores, tuple(timed.values()),
                     (lean_seconds.fitted, lean_bytes.fitted))

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
