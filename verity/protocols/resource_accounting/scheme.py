"""A resource scheme as the resource accounting layer takes it, and the one registry a configuration names it from.

The proof layer gives every scheme the same four things; a scheme reads them and owns none of them:

- the registered circuit, with its blocks and wires (`verity.commitments.periods.interface.Circuit`);
- the commitments to its trace, logged as they arrive (`verity.commitments.periods.interface.Window`'s records);
- the verifier's coins, each logged after the commitments it must follow (the challenger's ledger entries,
  `Core.Record`, whose Python is `verity.channel.Record`);
- the audit: one sampling design over all blocks, and what its sample established
  (`verity.protocols.proofs.spot.verifier.profile.IntegrityProfile`, `Core.Integrity` in Lean).

A scheme supplies a `Scheme`: its blocks (`Block`), the cost of each in its resource, its assumptions, its conditions on
the log (`Order`), its other charges (`Charge`) and its settings. It gets back a `Bound`. Where the proof layer does not
yet provide one of its four things uniformly, a scheme states what it requires (`REQUIREMENTS`) and the stand-in it runs
on today (`Scheme.requires`).

The definition is the Lean structure `Verity.Scheme` and its `Verity.Scheme.GetsBack` (`security/Definitions/Core/
Scheme.lean`); `Scheme`, `Block`, `Order`, `Charge` and `Bound` mirror the fields a bound reads under its names (``δ``
spelt ``delta``), and a change to either names the other's line in its message. `hooks`, `settings`, `option`,
`ports` and `requires` are read by the adapter and the runner, by no guarantee, and have no Lean field.

**Declaring a scheme.** One module, ``<protocol>/scheme.py``, calls `register(Scheme(...))`, and each name it registers
is a line of `verity.protocols.compose.schemes.DECLARATIONS`, from which every reader gets it (`get`, `load`). Its fields, with the Lean they mirror
(`Verity.Scheme`):

- ``blocks``: `Block(definition, cost)`, `Block` and `cost`; ``cost`` reads public values logged before the sample's coin.
- ``charges``: `Charge(name, bound)`, summed in `charges`; a known channel with no bound yet is ``bounded=False``.
- ``log``: `Order`s, `logOK`; a deadline (``within_ns``) names one of ``assumptions``, the names `breaks` ranges over.
- ``over``, ``under``: `over` and `under`, each side said in words with its natural unit; a scheme whose term is said
  says both. A scheme bounding what it credits by what was spent or held puts the credit ``over``; a factor in its
  guarantee is cleared into the two sides, never carried beside them.
- ``delta_scheme``: `δscheme`, said by name: a number, zero included, or a `Term`, the Lean constant of a family of
  schemes (one for each value of its parameters) and its code, which `bound` evaluates at parameters its caller gives;
  unsaid, `bound` refuses.
- code only: ``name`` (the configuration's), ``protocol`` (`Core.Protocol.id`), ``resource`` (`Core.Account.Part`),
  ``settings`` (typed defaults, the application's ``[params]`` over them through `settings`, which is the one place
  they are checked, with ``check_settings``; `verity.hooks.Declared.settings` carries them into the served run),
  ``hooks``, ``ports`` (each with its `Layout` at a public shape when an adapter's circuit reads it, `Port.layout`),
  ``option``, ``requires``, and what it does on a served run: ``requests(ctx)``, the requests it makes of the run
  (`verity.hooks`), ``attach(exchange, ctx)``, which subscribes to the run's exchange and installs its kernels (``ctx``'s keys:
  `verity.hooks.FRONT`), returning what the adapter reads of it (a `verity.hooks.Front`, a `verity.hooks.Gate`, or a report: any of
  ``summary()``, the run's totals, ``record()``, what it ran with, ``release()``, called once when the workload stops,
  and ``commitment()``, what it committed to, read after ``release``), and
  ``processes``, what it starts beside the workload by name (`verity.hooks.StartBeside`, entered in `PROCESSES`), and
  ``messages``, the message types its workload writes and reads on disk (`verity.channel.Schema`s), which an adapter's
  codec reads by the scheme's name, and ``framings``, the completion framings it serves by name, each built over the
  adapter's own framing (``framing(base=..., **settings)``), and ``provides``, the roles an adapter asks a scheme for
  by role rather than by name (``admission``: with ``admit`` in its ``ctx``, `attach` installs and returns its gate;
  ``cut``: an execution phase's circuit is cut at the blocks it computes, reading the values each call commits there in
  place of their computation).
  ``ctx`` is a mapping: ``settings``, the scheme's own (`settings`, the application's over its defaults); ``build``,
  the key of the build it attaches to (one per exchange); ``gate``, the auditor device's challenger gate in its own
  words, ``gate(*words) -> answer`` (``open --session S --root R`` opens a reserved session on a root; a scheme forms
  its own requests of it), or None where the launcher holds none; ``private``, the scheme's own directory
  (``<launcher's private>/<scheme's name>``), outside every run output, where it keeps what never leaves the node across
  a server's lives, or None where the launcher holds none; and what the launcher passes: ``process``, ``feed``
  (`verity.hooks.FRONT`), for the network attester ``admit``, ``source``, ``submit``, ``abort``, ``take``, ``pull``,
  ``log``, ``clock`` and ``schedule``, and for Pearl+ reserving a session an epoch ``ledger``, the gate's ledger as lines. The adapter's exchange attaches it (``Exchange.attach``), files those of
  ``requests(ctx)`` that ``attach`` did not, and declares it with its settings under its name (`verity.hooks.Declared`).

What it gets back: `bound` gives `Bound`, `GetsBack`'s numbers; `credited` gives `Placed.credited`, the first-level blocks
the integrity profile's ``credited`` takes.

A served run offers every scheme the same five hooks (`verity.hooks.HOOKS`), and a scheme names those it attaches
through (`Scheme.hooks`).

A scheme's `name` is the name a configuration lists (`protocols = [...]` in `applications/NAME.toml`, which
`verity.config.Component.name` declares), so changing schemes is changing that name. Its `protocol` is the id of the
Lean check that decides it (`Core.Protocol.id`), and `resource` is the part of the registered hardware its bound is
about, as `Core.Account.Part` names it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Literal, Protocol

from verity.hooks import HOOKS


def _no_requests(ctx: Any) -> tuple[Any, ...]:
    return ()


def _attaches_nothing(exchange: Any, ctx: Any) -> None:
    return None


def _no_settings_check(values: Mapping[str, Any]) -> str | None:
    return None


@dataclass(frozen=True)
class Lacks:
    """What a declaration doesn't fill, and why: read through `declared`, a field holding it refuses by its name. A field
    no declaration sets holds `UNDECLARED`, so nothing unset reads as declared."""

    why: str = ""


UNDECLARED = Lacks()


class Kernels(Protocol):
    """A scheme's ``kernels``: the kernel scheme it runs on the card whose short name is ``card`` and whose card file's
    ``arch`` is ``arch`` (its caller reads the card file), with registration pinning the weights or not."""

    def __call__(self, card: str, *, pinned: bool, arch: str) -> str: ...


#: `Core.Account.Part`'s names, each with the unit its rows are in (`Account.Part.unit`); a row may publish its amount in
#: another measure (`Account.Measure`: PoUB P2's ``q_bytes`` in bytes, 8 bits each), which the row converts to this one
RESOURCES: Mapping[str, str] = MappingProxyType({"compute": "MACs", "memory": "bits", "read": "bits",
                                                 "egress": "bits"})
#: what a scheme needs of the proof layer that the layer doesn't provide every scheme the same way yet
REQUIREMENTS = ("one-labelling", "log-order", "log-times", "coins-at-rate", "one-design")
_NAME = re.compile(r"[a-z][a-z0-9-]*")


@dataclass(frozen=True)
class Block:
    """A kind of block the scheme adds to the registered circuit: its Definition, an exact function of its inputs, by the
    id `verity.commitments.periods.interface.resolve` reads, and ``cost``, the resource credited for one correct block.

    ``cost`` takes the block's public values (its Definition's public shape and the record's values for it, such as a
    kept width), every one of which is logged before the sample's coin: a value the prover could set after seeing the
    sample would let it raise the cost of the sampled blocks. `Verity.Scheme`'s block is one sampled unit; a `Block`
    here is a kind whose every unit of one kind and public values costs the same, which is how the two agree. No
    declared cost may be below what the verdict credits for the block: a larger cost only weakens the bound."""

    definition: str
    cost: Callable[[Mapping[str, int]], int]


@dataclass(frozen=True)
class Order:
    """A condition on the verifier's log (`Verity.Scheme.logOK`): ``committed`` is logged ``direction`` (``before`` or
    ``after``) ``coin``, a coin drawn or another logged event, and, where ``within_ns`` is set, within that many
    nanoseconds of it: a deadline, which no guarantee of the proof layer reads, so it names the scheme's ``assumption``
    that does (a bound on the prover's speed, say). A deadline set per session is the name of the public parameter,
    logged before the coin, that carries it in nanoseconds (PoUB's ``deadline_ns``); no value of it is declared."""

    committed: str
    coin: str
    direction: Literal["before", "after"] = "before"
    within_ns: int | str | None = None
    assumption: str | None = None

    def __post_init__(self) -> None:
        if self.direction not in ("before", "after"):
            raise ValueError(f"{self.committed}: an order is before or after {self.coin}, not {self.direction!r}")
        if (self.within_ns is None) != (self.assumption is None):
            raise ValueError(f"{self.committed} {self.direction} {self.coin}: a time is read only by an assumption, "
                             "and names it")
        if isinstance(self.within_ns, str) and not _NAME.fullmatch(self.within_ns.replace("_", "-")):
            raise ValueError(f"{self.committed} {self.direction} {self.coin}: a deadline set per session names its "
                             f"logged parameter, not {self.within_ns!r}")
        if isinstance(self.within_ns, int) and self.within_ns <= 0:
            raise ValueError(f"{self.committed} {self.direction} {self.coin}: a deadline is a positive time")


@dataclass(frozen=True)
class Charge:
    """A known way to a correct block without spending the resource: its name, and where its bound is stated (the key
    of the scheme's published answer that carries it, or the Lean declaration that bounds it). A known channel is
    declared even before anything bounds it, with ``bounded`` False and ``bound`` naming the statement to come: then
    `bound` refuses the scheme, since a bound without it would read as complete (`Verity.Scheme.charges` is a number)."""

    name: str
    bound: str
    bounded: bool = True


@dataclass(frozen=True)
class Layout:
    """A port's row at one public shape: its ``bits`` and each field's ``(offset, width)`` in bits, least significant bit
    first, and ``counts``, the public integers the shape derives that place a field's own items (the scan state's ``K0``
    prompt ids in a first cell and ``K`` in each continuation). A layout is protocol structure, one of the four public
    kinds: it says where a value's bits sit, never what they are. Refused by name: a field that runs past ``bits`` or
    overlaps another (a field of width 0 overlaps nothing)."""

    bits: int
    fields: Mapping[str, tuple[int, int]]
    counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.bits) is not int or self.bits < 1:
            raise ValueError(f"a layout's bits are a positive integer, not {self.bits!r}")
        for name, (at, width) in self.fields.items():
            if type(at) is not int or type(width) is not int or at < 0 or width < 0:
                raise ValueError(f"field {name}'s offset and width are integers >= 0, not ({at!r}, {width!r})")
            if at + width > self.bits:
                raise ValueError(f"field {name} at ({at}, {width}) runs past the layout's {self.bits} bits")
        laid = sorted((at, at + width, name) for name, (at, width) in self.fields.items() if width)
        for (_, end, before), (at, _, name) in zip(laid, laid[1:]):
            if at < end:
                raise ValueError(f"field {name} at bit {at} overlaps field {before}, which ends at bit {end}")
        for name, v in self.counts.items():
            if type(v) is not int:
                raise ValueError(f"count {name} is an integer, not {v!r}")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))


@dataclass(frozen=True)
class Port:
    """An external port the scheme reads (``input``) or writes (``output``), laid out as the field ``field`` of the
    public signature of its block's Definition ``definition``: the scan's state row, the released bytes' positions.
    ``layout``, when declared, gives the port's `Layout` at a public shape (named integers, as `Block.cost` reads), which
    an adapter's lowering reads through `layout` when it builds a circuit; no guarantee reads it. ``value``, where the
    port's rows are a registered value, is that value's name."""

    name: str
    direction: Literal["input", "output"]
    definition: str
    field: str
    layout: Callable[[Mapping[str, int]], Layout] | None = None
    value: str = ""


@dataclass(frozen=True)
class Term:
    """A scheme's own term that depends on public parameters: in the Lean a family of schemes, one for each value, whose
    `δscheme` is the constant ``lean`` at them. ``parameters`` are its parameters' names (``q``, ``N``), and ``value``
    evaluates it at them by name; no parameter's value is written into a declaration (`bound` takes them)."""

    lean: str
    parameters: tuple[str, ...]
    value: Callable[..., Fraction]

    def __post_init__(self) -> None:
        if not self.lean or not self.parameters or len(set(self.parameters)) != len(self.parameters):
            raise ValueError(f"a term names its Lean constant and its parameters once each, not {self.lean!r} "
                             f"{self.parameters}")


@dataclass(frozen=True)
class Reading:
    """What a scheme's ``reader`` reads of the row its run left: its certified quantity's ``value``, in the unit it
    publishes (``published``); the ``error`` the run established it except with, and ``error_from``, the key of the
    row it is read from; and the quantity's other public ``figures`` by name, as the record writes them."""

    value: int
    error: Fraction
    error_from: str
    figures: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scheme:
    """What a resource scheme supplies. ``settings`` are its settings with their defaults, which a configuration's
    ``[params."<protocol>"]`` overrides (`settings`), and ``check_settings`` what it refuses of their values beyond their
    types (a count below 1), as a sentence, None where it accepts them; ``option`` is the adapter's entry that attaches it to a served run
    (``NAME:scheme``, an adapter's ``protocols`` spelling), empty when it attaches through none; ``hooks`` the served
    run's hooks it attaches through (`HOOKS`); ``ports`` the external ports it reads or writes; ``requires`` maps each
    requirement it has of the proof layer to today's stand-in. ``delta_scheme`` is `Verity.Scheme.δscheme`, the bound on
    the probability that one of its ``assumptions`` fails (a key guessed, a collision), said by name in every
    declaration: a number, zero where none can fail, or a `Term` where it depends on public parameters; None, not said,
    lets a scheme be registered and read but not bounded (`bound`). ``over`` and ``under`` are `Verity.Scheme.over` and
    `under` in words, each with its natural unit: the bound is that ``over`` exceeds ``under`` by at most the incorrect
    blocks' cost and the charges; a scheme whose term is said says both.

    What a configuration, a launcher and the combined record read of it, each a `Lacks` (refused by `declared`, with
    the field's name) until the declaration fills it: ``parties`` and ``channels`` (`verity.protocol.Role`, `Channel`;
    `verity.protocol.VERIFIER` is the party its answer goes to) and ``channel_schemas``, the `verity.channel.Schema` of
    each message its channels carry that has one, by its type, which this module types by name only, so it loads
    neither module; ``colocated``, the pairs of its parties on one machine; ``needs``, what
    the workload provides it (``windows``, ``noise``, ``store``, ``receipts``, ``ingress-relay``: provisions, which
    ``requests`` doesn't state, since it answers only for a serve); ``depends``, the schemes a configuration must run
    beside it (a scheme's names, unlike ``requires``, the proof layer's requirements); ``record``, its row's name in
    the record; ``symbol``, its certified quantity's symbol; ``published``, the unit the record publishes it in and
    the exact factor from ``unit`` (``("bytes", 1/8)`` for an amount in bits published in bytes, ``(unit, 1)`` where
    it is the same); ``check``, its verifier of record, the entry its Lean port is held to, with that entry's own
    arguments (a run's record and what the verifier is registered with, its trusted keys among them), which answers
    accept or names its refusal; ``reader``, which takes its row as the runner read it from what its piece wrote
    (``answer``, the answer, and ``quantity``, the figures read of it) and returns its `Reading`; and ``premises``,
    each Lean `Prop`, by name, that its charge or bound is proved under, empty where its instance takes none (an
    empty tuple claims that), which `bound` carries and the record writes beside the scheme's figure.
    ``tables``, the names of the value tables its run registers in the window record, a value of table ``t`` named
    ``t/...``; ``output_format``, the number format the output y leaves its blocks in for the served model
    (``bf16``), not the format of the arithmetic that made it; ``product_family``, the number family of its useful
    products, the operands' format and scaling (``fp8-e4m3-rowwise``: E4M3 operands, one scale per row of each, an
    FP32 accumulator), named as the harness names a stock kernel's family; ``y_committed``, whether each of its calls
    commits that y; ``salt_open_root``, the root its salt coin is opened after, a function of the weights' root
    and the salt epoch's number; and ``kernels``, the kernel scheme it runs on a card (`Kernels`), of the card's short
    name, ``pinned=`` (whether registration pins the weights) and ``arch=`` (its card file's), `LookupError` naming the
    cards and pinnings it runs on otherwise; and ``keeps_weight_copy``, whether it keeps a weight-derived copy resident for the serve's life, checked
    against the committed root once at build.

    A component that charges nothing (the sampled proofs, the inference-only check) is a `Scheme` whose ``resource``
    and ``blocks`` are each a `Lacks` saying so: it is in the one table with its parties and record; asked of it alone,
    its resource, blocks, unit and `bound` each refuse by name, and `credited`, which goes over every attached scheme,
    passes over it and gives it no block. A declaration that names a resource places at least one block, or says with
    ``blocks=Lacks(why)`` why no block of the circuit carries its cost (an audit proved outside the workload's
    circuit), and is held to every check below."""

    name: str
    protocol: str
    resource: str | Lacks
    blocks: tuple[Block, ...] | Lacks
    assumptions: tuple[str, ...]
    log: tuple[Order, ...]
    charges: tuple[Charge, ...]
    settings: Mapping[str, Any] = field(default_factory=dict)
    check_settings: Callable[[Mapping[str, Any]], str | None] = _no_settings_check
    option: str = ""
    hooks: tuple[str, ...] = ()
    requires: Mapping[str, str] = field(default_factory=dict)
    ports: tuple[Port, ...] = ()
    delta_scheme: Fraction | Term | Lacks = UNDECLARED
    over: str = ""
    under: str = ""
    requests: Callable[[Any], Sequence[Any]] = _no_requests
    attach: Callable[[Any, Any], Any] = _attaches_nothing
    processes: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    messages: tuple[Any, ...] = ()
    framings: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
    provides: tuple[str, ...] = ()
    parties: tuple[Any, ...] | Lacks = UNDECLARED
    channels: tuple[Any, ...] | Lacks = UNDECLARED
    colocated: tuple[tuple[str, str], ...] | Lacks = UNDECLARED
    needs: frozenset[str] | Lacks = UNDECLARED
    depends: tuple[str, ...] | Lacks = UNDECLARED
    record: str | Lacks = UNDECLARED
    symbol: str | Lacks = UNDECLARED
    channel_schemas: tuple[Any, ...] | Lacks = UNDECLARED
    published: tuple[str, Fraction] | Lacks = UNDECLARED
    check: Callable[..., Mapping[str, Any]] | Lacks = UNDECLARED
    reader: Callable[[Mapping[str, Any]], Reading] | Lacks = UNDECLARED
    premises: tuple[str, ...] | Lacks = UNDECLARED
    tables: tuple[str, ...] | Lacks = UNDECLARED
    output_format: str | Lacks = UNDECLARED
    product_family: str | Lacks = UNDECLARED
    y_committed: bool | Lacks = UNDECLARED
    salt_open_root: Callable[[bytes, int], bytes] | Lacks = UNDECLARED
    kernels: Kernels | Lacks = UNDECLARED
    keeps_weight_copy: bool | Lacks = UNDECLARED

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"a scheme's name is a configuration's name ({_NAME.pattern}), not {self.name!r}")
        if unknown := sorted(set(self.hooks) - set(HOOKS)):
            raise ValueError(f"{self.name}: {unknown} are not hooks of a served run {HOOKS}")
        if not isinstance(self.resource, Lacks) and self.resource not in RESOURCES:
            raise ValueError(f"{self.name}: its resource is one of {sorted(RESOURCES)}, not {self.resource!r}")
        if unknown := sorted(set(self.requires) - set(REQUIREMENTS)):
            raise ValueError(f"{self.name}: {unknown} are not requirements of the proof layer {REQUIREMENTS}")
        names = [c.name for c in self.charges]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name} names a charge twice: {names}")
        if unknown := sorted({o.assumption for o in self.log if o.assumption} - set(self.assumptions)):
            raise ValueError(f"{self.name}: its deadlines name {unknown}, which are not among its assumptions")
        if isinstance(self.resource, Lacks) and not isinstance(self.blocks, Lacks):
            raise ValueError(f"{self.name}: a scheme that charges nothing says why it has no blocks with Lacks(why)")
        if not isinstance(self.blocks, Lacks) and not self.blocks:
            raise ValueError(f"{self.name}: it names a resource and places no block; where no block of the circuit "
                             "carries its cost, its blocks say why with Lacks(why)")
        placed = () if isinstance(self.blocks, Lacks) else self.blocks
        if unknown := sorted({p.definition for p in self.ports} - {b.definition for b in placed}):
            raise ValueError(f"{self.name}: its ports lie in {unknown}, which are not its blocks' Definitions")
        said = not isinstance(self.delta_scheme, Lacks)
        if said and not isinstance(self.delta_scheme, Term) and not 0 <= self.delta_scheme < 1:
            raise ValueError(f"{self.name}: its own term δscheme lies in [0, 1), not {self.delta_scheme}")
        if said and not (self.over and self.under):
            raise ValueError(f"{self.name}: a scheme whose term is said says its two sides (Verity.Scheme.over and under) "
                             "in words, each with its unit")
        if not isinstance(self.channel_schemas, Lacks) and not isinstance(self.channels, Lacks):
            carried = {m for c in self.channels for m in c.messages}
            if stray := sorted({x.type for x in self.channel_schemas} - carried):
                raise ValueError(f"{self.name}: its channel_schemas name {stray}, which none of its channels carries")
        if not isinstance(self.published, Lacks):
            unit, per = self.published
            if not unit or type(per) is not Fraction or per <= 0:
                raise ValueError(f"{self.name}: published is (the record's unit, a positive Fraction from its own "
                                 f"unit), not {self.published!r}")
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))
        object.__setattr__(self, "requires", MappingProxyType(dict(self.requires)))

    @property
    def unit(self) -> str:
        """The unit of its sides, costs and charges (`Core.Account.Part.unit`); refused by name for a scheme with no
        resource."""
        return RESOURCES[declared(self, "resource")]


def settings(scheme: Scheme, given: Mapping[str, Any]) -> dict[str, Any]:
    """``scheme``'s settings with ``given`` over its defaults, the one place a setting is checked: refuses a key it
    doesn't have, a value of another type than its default's, and what its ``check_settings`` refuses. Every setting is
    returned, its default included: what a run used is what an application's hash covers (`verity.config.check` fills
    the defaults), so stating a default moves no hash and changing a default in code does."""
    if unknown := sorted(set(given) - set(scheme.settings)):
        raise ValueError(f"{scheme.name} has no setting {unknown}: its settings are {sorted(scheme.settings)}")
    for k, v in given.items():
        if type(v) is not type(scheme.settings[k]):
            raise ValueError(f"{scheme.name}.{k} is a {type(scheme.settings[k]).__name__}, not {v!r}")
    out = {**scheme.settings, **given}
    if refused := scheme.check_settings(out):
        raise ValueError(f"{scheme.name}: {refused}")
    return out


@dataclass(frozen=True)
class Bound:
    """What the audit's guarantee gives a scheme (`Verity.Scheme.GetsBack`): at cutoff ``delta``, its ``over`` exceeds
    its ``under`` by more than ``missable``, the largest cost of a set of its blocks that the sample misses with
    probability above ``delta``, plus each of its charges' bounds, with probability at most ``delta`` plus ``epsilon``,
    the audit's own error (its knowledge and binding errors, the audit's to give), plus ``delta_scheme``, the scheme's own
    term: where that is a `Term`, ``term`` names its Lean constant and
    ``parameters`` the values it was evaluated at; all of it under ``premises``, each Lean `Prop`, by name, the
    scheme's guarantee is stated under, which whatever writes the bound writes beside it."""

    scheme: str
    delta: Fraction
    missable: int
    charges: Mapping[str, int]
    delta_scheme: Fraction
    epsilon: Fraction = Fraction(0)
    term: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    premises: tuple[str, ...] = ()

    @property
    def excess(self) -> int:
        """The most ``over`` exceeds ``under`` by, but with `probability`."""
        return self.missable + sum(self.charges.values())

    @property
    def probability(self) -> Fraction:
        """The probability that ``over`` exceeds ``under`` by more than `excess`: ``delta + epsilon + delta_scheme``
        (`GetsBack`'s ``δ + ε + δscheme``)."""
        return self.delta + self.epsilon + self.delta_scheme


class Unbounded(ValueError):
    """`bound`'s refusal of a declaration whatever its caller gives: the sentence, and ``terms``, the name of each
    term it lacks (each charge nothing bounds, by its name; ``Verity.Scheme.δscheme``; ``premises``)."""

    def __init__(self, why: str, terms: tuple[str, ...]) -> None:
        super().__init__(why)
        self.terms = terms


def unbounded(scheme: Scheme) -> Unbounded | None:
    """Why ``scheme``'s declaration lets `bound` give no bound at any inputs, None where it can: a known charge
    nothing bounds yet, its own term not said, or its premises not said, the first of these."""
    if open_ := [c for c in scheme.charges if not c.bounded]:
        return Unbounded(f"{scheme.name} has known charges nothing bounds yet: "
                         f"{', '.join(f'{c.name} ({c.bound})' for c in open_)}; a bound without them would read as "
                         "complete", tuple(c.name for c in open_))
    if isinstance(scheme.delta_scheme, Lacks):
        return Unbounded(f"{scheme.name}: its delta_scheme (Verity.Scheme.δscheme) is not said: a declaration says its "
                         "own term by name, zero where none of its assumptions can fail"
                         + (f"; it says why not: {scheme.delta_scheme.why}" if scheme.delta_scheme.why else ""),
                         ("Verity.Scheme.δscheme",))
    if isinstance(scheme.premises, Lacks):
        return Unbounded(f"{scheme.name}: its premises are not said, so a bound would read as unconditional; a "
                         "declaration names each Lean Prop its guarantee is stated under, () where none"
                         + (f"; it says why not: {scheme.premises.why}" if scheme.premises.why else ""), ("premises",))
    return None


def bound(scheme: Scheme, delta: Fraction, missable: int, charges: Mapping[str, int],
          parameters: Mapping[str, Any] | None = None, *, epsilon: Fraction | None = None) -> Bound:
    """``scheme``'s `Bound`, carrying its own term, evaluated at ``parameters`` where it is a `Term` (``N`` read from
    the registered circuit, ``q`` and the rest the caller's); refuses charges other than exactly the ones it states, a
    scheme with a known charge nothing bounds yet or whose own term is not said, parameters other than exactly the
    term's, and an audit's error ``epsilon`` not given (`GetsBack`'s ε, which the audit supplies, never the scheme)."""
    declared(scheme, "resource")
    if (refused := unbounded(scheme)) is not None:
        raise refused
    if sorted(charges) != sorted(c.name for c in scheme.charges):
        raise ValueError(f"{scheme.name}'s charges are {sorted(c.name for c in scheme.charges)}, not {sorted(charges)}")
    if not 0 < delta < 1 or missable < 0 or any(v < 0 for v in charges.values()):
        raise ValueError(f"{scheme.name}: δ lies in (0, 1) and every amount is at least 0")
    if epsilon is None or not 0 <= epsilon < 1:
        raise ValueError(f"{scheme.name}: the audit's error ε (Verity.Scheme.GetsBack's δ + ε + δscheme) is given by the "
                         f"audit, in [0, 1), not {epsilon}")
    given = dict(parameters or {})
    term = scheme.delta_scheme
    if not isinstance(term, Term):
        if given:
            raise ValueError(f"{scheme.name}: its own term is the number {term}, which takes no parameters {sorted(given)}")
        return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)), Fraction(term),
                     Fraction(epsilon), premises=scheme.premises)
    if missing := [p for p in term.parameters if p not in given]:
        raise ValueError(f"{scheme.name}: its own term {term.lean} is evaluated at {list(term.parameters)}, and "
                         f"{missing} were not given")
    if extra := sorted(set(given) - set(term.parameters)):
        raise ValueError(f"{scheme.name}: its own term {term.lean} has no parameters {extra}")
    value = Fraction(term.value(**given))
    if not 0 <= value < 1:
        raise ValueError(f"{scheme.name}: its own term {term.lean} at {given} is {value}, outside [0, 1): no bound")
    return Bound(scheme.name, Fraction(delta), missable, MappingProxyType(dict(charges)), value, Fraction(epsilon),
                 term.lean, MappingProxyType(given), scheme.premises)


#: the registered schemes, by name: a scheme's module registers it when imported, and a reader gets it through
#: `verity.protocols.compose.schemes.get`
SCHEMES: dict[str, Scheme] = {}


def declared(scheme: Scheme, name: str) -> Any:
    """``scheme``'s field ``name``, refused (LookupError, naming the scheme, the field and what the declaration says it
    lacks) where it holds a `Lacks`."""
    value = getattr(scheme, name)
    if isinstance(value, Lacks):
        raise LookupError(f"{scheme.name} declares no {name}" + (f": {value.why}" if value.why else ""))
    return value


def layout(scheme: Scheme, port: str, shape: Mapping[str, int]) -> Layout:
    """``port``'s `Layout` at ``shape``, as ``scheme`` declares it; KeyError naming the scheme's ports, or ValueError when
    the port declares no layout."""
    found = {p.name: p for p in scheme.ports}
    if port not in found:
        raise KeyError(f"{scheme.name} declares no port {port!r}; its ports: {', '.join(found) or 'none'}")
    if found[port].layout is None:
        raise ValueError(f"{scheme.name}'s port {port} declares no layout (`Port.layout`)")
    got = found[port].layout(dict(shape))
    if not isinstance(got, Layout):
        raise ValueError(f"{scheme.name}'s port {port}'s layout gives a {type(got).__name__}, not a Layout")
    return got


def register(scheme: Scheme) -> Scheme:
    if scheme.name in SCHEMES:
        raise KeyError(f"a scheme named {scheme.name!r} is registered already")
    SCHEMES[scheme.name] = scheme
    for name, start in scheme.processes.items():
        register_process(name, start)
    return scheme


#: the processes schemes start beside a workload, by name (`verity.hooks.StartBeside`): each called with the readers
#: its request names
PROCESSES: dict[str, Callable[..., Any]] = {}


def register_process(name: str, start: Callable[..., Any]) -> Callable[..., Any]:
    if name in PROCESSES:
        raise KeyError(f"a process named {name!r} is registered already")
    PROCESSES[name] = start
    return start


@dataclass(frozen=True)
class Circuits:
    """What a scheme's executed computation is traced as: ``schemes``, the schemes with a circuit, and ``traced_as(scheme,
    m, k, n)``, one executed call's circuit (None without one)."""

    schemes: Mapping[str, Any]
    traced_as: Callable[[str, int, int, int], Any]


#: the circuits of each scheme's executed computation, by the name a configuration lists it under
CIRCUITS: dict[str, Circuits] = {}


def register_circuits(name: str, circuits: Circuits) -> Circuits:
    if name in CIRCUITS:
        raise KeyError(f"circuits named {name!r} are registered already")
    CIRCUITS[name] = circuits
    return circuits


def credited(schemes: Sequence[Scheme], strata: Iterable[tuple[str, Sequence[Sequence[int]]]]) -> frozenset[int]:
    """`Verity.Scheme.Placed.credited`: the numbers of the first-level proof blocks that ``schemes`` credit, in the
    numbering the integrity profile's ``credited`` takes. ``strata`` are the registered circuit's first-level proof blocks
    by template, ``(descriptor id, ascending [lo, hi) ranges)``, as each party derives them from the public Program and
    query (`verity.protocols.proofs.spot.verifier.one_stage.partition.strata`). A declared `Block` places on the
    templates whose descriptor id is its Definition id, bare or followed by its statics (``Name_vN{...}``). Refuses a
    declared Definition id that resolves to no block, and a template two blocks would place on, of one scheme or two
    (`Placed.inj`, and one credited set for all). It passes over a scheme that charges nothing, or whose blocks say
    why none is in the circuit: neither places a block."""
    schemes = [s for s in schemes if not isinstance(s.resource, Lacks) and not isinstance(s.blocks, Lacks)]
    strata = [(tid, rs) for tid, rs in strata]
    placed: dict[str, str] = {}
    out: set[int] = set()
    for s in schemes:
        for b in s.blocks:
            hits = [(tid, rs) for tid, rs in strata if tid == b.definition or tid.startswith(b.definition + "{")]
            if not hits:
                raise ValueError(f"{s.name}: its block {b.definition} resolves to no block of the circuit, whose "
                                 f"templates are {[tid for tid, _ in strata]}")
            for tid, rs in hits:
                if tid in placed:
                    raise ValueError(f"{tid} would be credited twice: by {placed[tid]} and by {s.name}'s {b.definition}")
                placed[tid] = f"{s.name}'s {b.definition}"
                out.update(u for lo, hi in rs for u in range(lo, hi))
    return frozenset(out)


__all__ = ["CIRCUITS", "PROCESSES", "REQUIREMENTS", "RESOURCES", "SCHEMES", "UNDECLARED", "Block", "Bound",
           "Charge", "Circuits", "Kernels", "Lacks", "Layout", "Order", "Port", "Reading", "Scheme", "Term", "Unbounded",
           "bound",
           "credited", "declared", "layout", "register", "register_circuits", "register_process", "settings",
           "unbounded"]
