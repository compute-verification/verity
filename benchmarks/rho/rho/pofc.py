"""The proof of forward computation (PoFC): Daniel's Notion page "Proof of forward computation", in ρ's layers.

The prover commits to a chain of operators `F_1 → … → F_k` (one a transformer layer, say) with parameters `θ_j`
(`|θ_j| ≤ p`, `Σ |θ_j| ≤ P`) and works in isolated **episodes** of at most `n_max` tokens: `F_j` reads only `θ_j`, its
own state in the episode and `F_{j−1}`'s output; only `F_1` reads novel input (`X`) and only `F_k` emits (`Y`); each
operator writes at most `s` bits a token to the next (about `d_model` values at serving precision). The conjecture:
updating a layer needs its weights together with a backward signal only later operators produce, so every trained
layer's weights must reach the operator holding the loss, either committed a second time (charged to `P`) or written
forward as activations (charged to the `n_max s` bits an episode's boundary carries). The **toll** is the share of those
bits the weights take,

    t = (weight bits behind the boundary) / (n_max s)

so exact backprop inside an episode is blocked outright at `t ≥ 1`, and gives a share `t` of the episode's positions to
weights instead of tokens at `t < 1`. Everything is counted in bits, `P` and `p` included (Daniel, 26 Oct: circuit
language), so a weight crosses at the bits the forward circuit reads it, the committed precision.

**Where an update goes** depends on the regime's threat model (Daniel, 26 Oct). Registration happens once (25 Oct) and
every declared episode reads the committed `θ`, so an update made inside an episode reaches a later declared episode
only as its output `Y` coming back in as `X` (`Rules.persist = "outputs"`); `"recommit"` is the page's reading, a prover
that commits new parameters and pays `P`. What an episode computes also stays with whoever ran it, for covert
computation: onsite, the server, until its wipe, which the serving cut bounds with everything else the serving block
holds (other servers learn only its certified outputs, timing and side channels); remote, the prover, whose covert
computation only the uncertified gates bound. Both are cuts the wrapped regime already has.

PoFC enters ρ beside the layers it touches, never in them:

* `Pofc` wraps any regime: every cut with inference blocks gains a copy whose declared work runs as episodes
  (`episode`), and so does the layout. With `link`, the physical instantiation (forward-only links, each GPU reading only
  its own static memory), every other block of those copies gets the link's bits over its time as its forward cut.
  `apply` wraps a scenario's regime and adds PoFC's bounds and attacks to its own.
* `Backprop` is the training class `pofc-backprop` (draft): what a block under a forward cut finishes of exact backprop
  that must reach depth `L`.
* `LastLayers`, `Palindrome`, `WeightsAsActivations` and `JunkEpisodes` are the page's constructions (attacks b to e);
  `FAMILIES` says where its other attacks, algorithms with their own efficiency, enter the families layer.

The arguments read a `Trained` (`trained`: a `Profile`, a model and the `Rules`) rather than new `Profile` fields, and
`versus` gives ρ against `n_max` at a scenario.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rho.classes import Allowance, FlopBudget, Outside
from rho.params import Registry
from rho.profile import Profile
from rho.theorem import Blocks, Budget, Cut, Partition

if TYPE_CHECKING:
    from rho.envelopes import Context, Delta, Regime
    from rho.families import Family
    from rho.instances import Model, Run
    from rho.scenario import Scenario

#: the page's conjecture, its rules (enforced on declared episodes), its physical instantiation and its known attacks;
#: `SIGNAL` is the premise that bounds sharded backprop (`Backprop.signal`)
CONJECTURE = "draft:pofc-conjecture"
RULES = "draft:pofc-rules"
PHYSICAL = "draft:pofc-physical"
SIGNAL = "draft:pofc-signal"
ATTACKS = "draft:pofc-attacks"

DAY = 86400.0

#: `outputs`: through `Y` and back in through `X` (registration once); `recommit`: new parameters, charged to `P`
PERSIST = ("outputs", "recommit")


class Layered(Protocol):
    """What the toll reads about a model: an `instances.Model` or a `Stack`."""

    name: str
    n_total: float
    layers: int
    d_model: int


@dataclass(frozen=True)
class Stack:
    """A model as the toll sees it: its parameters, its layers and its width."""

    name: str
    n_total: float
    layers: int
    d_model: int


@dataclass(frozen=True)
class Rules:
    """The PoFC policy's numbers (the page's rules 1 to 6)."""

    name: str
    #: `n_max`: tokens an episode may process
    n_max: float
    #: the write allowance `s`, in values of `d_model` a position
    write: float
    #: bits a written value carries (the serving precision)
    act_bits: float
    #: bits the backward pass reads a weight at: the forward circuit's (`commit_bits`), unless an attacker's training
    #: needs more
    weight_bits: float
    #: bits the honest model's weights are committed at
    commit_bits: float
    #: `P`, in multiples of the honest model's committed bits, and `p`, in multiples of one honest layer's
    commit: float
    per_operator: float
    #: `X` and `Y`, in bits a token of the episode
    input_bits: float
    output_bits: float
    #: episodes a location may run a day
    episodes_per_day: float
    #: how an update made inside an episode reaches the next (`PERSIST`)
    persist: str = "outputs"

    def __post_init__(self) -> None:
        if self.persist not in PERSIST:
            raise ValueError(f"pofc rules {self.name}: persist {self.persist!r} is not one of {PERSIST}")

    def s(self, m: Layered) -> float:
        """`s = write · d_model · act_bits`: the bits one position carries across a boundary."""
        return self.write * m.d_model * self.act_bits

    @property
    def x(self) -> float:
        """`X`: the novel bits an episode may read."""
        return self.n_max * self.input_bits

    @property
    def y(self) -> float:
        """`Y`: the bits an episode may emit."""
        return self.n_max * self.output_bits


def toll(layers_behind_bits: float, n_max: float, s_bits_per_position: float) -> float:
    """`t = bits / (n_max s)`: the share of an episode's boundary positions the weights behind it take."""
    return layers_behind_bits / (n_max * s_bits_per_position)


def layer_params(m: Layered) -> float:
    """One layer's parameters: the model's over its layers, every expert counted (a batch activates nearly all within
    a few hundred tokens)."""
    return m.n_total / m.layers


def positions(m: Layered, weight_bits: float, act_bits: float, write: float = 1.0) -> float:
    """Boundary positions one layer's weights take: `layer_params · weight_bits / (write · d_model · act_bits)`."""
    return layer_params(m) * weight_bits / (write * m.d_model * act_bits)


def depth_toll(m: Layered, depth: int, n_max: float, weight_bits: float, act_bits: float,
               write: float = 1.0) -> float:
    """The toll when the gradient must reach the top `depth` layers (the whole model past its own depth), all of
    whose weights cross the boundary in front of the operator holding the loss."""
    return min(depth, m.layers) * positions(m, weight_bits, act_bits, write) / n_max


def tolls(m: Layered, depths: Iterable[int], n_maxes: Iterable[float], rules: Rules) -> dict[tuple[int, float], float]:
    """`depth_toll` over a grid, at the rules' precisions: the page's tables."""
    n_maxes = tuple(n_maxes)
    return {(d, n): depth_toll(m, d, n, rules.weight_bits, rules.act_bits, rules.write) for d in depths for n in n_maxes}


@dataclass(frozen=True)
class Trained:
    """What PoFC's arguments read about one algorithm on one model under the rules."""

    name: str
    kappa: float
    #: `L`: the layers the gradient must reach, from the top
    depth: int
    layer_params: float
    #: bits a weight crosses at (`Rules.weight_bits`)
    weight_bits: float
    #: top layers the operator holding the loss commits with it at `weight_bits`, within `p`: they never cross
    held: int
    #: `P` less the honest model's commitment, in bits: weights committed a second time rather than crossing
    spare: float
    #: `s`: bits a position carries across a boundary
    position_bits: float
    #: the reference algorithm's FLOPs a trained token: the forward through the model, the backward through `depth`
    #: layers (`6 N_active` for the whole model)
    token_flops: float

    def behind(self, bits: float | None = None) -> float:
        """Weight bits that must cross the boundary in front of the operator holding the loss: the trained layers it
        doesn't hold, at `bits` a weight (`weight_bits` unless given), less what the spare commitment holds."""
        b = self.weight_bits if bits is None else bits
        return max(0.0, (self.depth - self.held) * self.layer_params * b - self.spare)

    def toll(self, budget: FlopBudget, bits: float | None = None) -> float:
        return self.behind(bits) / budget.cut

    def tokens(self, bits: float) -> float:
        """Counted products of `bits / s` trained tokens: `κ/2 · token_flops · bits / s`."""
        return self.kappa / 2 * self.token_flops * bits / self.position_bits

    def within(self, budget: FlopBudget, bits: float | None = None) -> float:
        """Counted products of the tokens whose positions the weights leave free: `tokens((cut − behind)⁺)`."""
        return self.tokens(max(0.0, budget.cut - self.behind(bits)))


def trained(p: Profile, model: Model, rules: Rules, depth: int | None = None) -> Trained:
    """`p` on `model` under `rules`, its gradient reaching the top `depth` layers: the given depth, else the profile's
    `depth` if it has one, else the whole model."""
    depth = min(model.layers, depth if depth is not None else getattr(p, "depth", None) or model.layers)
    return Trained(
        name=p.name, kappa=p.kappa, depth=depth, layer_params=layer_params(model), weight_bits=rules.weight_bits,
        held=math.floor(rules.per_operator * rules.commit_bits / rules.weight_bits),
        spare=max(0.0, (rules.commit - 1) * model.n_total * rules.commit_bits), position_bits=rules.s(model),
        token_flops=2 * model.n_active * (1 + 2 * depth / model.layers))


@dataclass(frozen=True)
class Backprop:
    """Exact backprop whose gradient must reach depth `L`, completed inside one episode (the page's conjecture). What
    it caps is the counted products of the tokens a block trains to depth `L` (a token run only forward, or whose
    gradient stops short of `L`, finishes none of them). A block under a forward cut finishes at most those of the
    tokens whose positions the weights behind the cut leave free, `κ/2 · token_flops · (cut − behind) / s`
    (`Trained.within`): every trained token's activation takes a position, and trains at the reference algorithm's
    cost. `theorem.terms` caps it at `κ C / 2`, so it binds only when `(1 − t) n_max · token_flops < C`; with
    `token_flops = C / n_max` (rule 6 charged a position) it is the page's useful share, `(1 − t) κ C / 2`. Like every
    class it is in the FLOP model (`classes.FlopBudget`).

    At `t ≥ 1` the within-episode route is blocked and only sharding across episodes remains, which this class doesn't
    bound and the NCI classes price at the episode's `X`: it gives no cap (`math.inf`) there, nor on a block with no
    forward cut. With `signal`, it bounds the sharded route too, under the premise that a token's backward signal is at
    least a position and enters a later episode only through `X`: `κ/2 · token_flops · ((cut − behind)⁺ + X) / s`,
    finite at every toll and a valid cap for an algorithm that mixes both routes."""

    model: Model
    rules: Rules
    depth: int | None = None
    signal: bool = False
    name: str = "pofc-backprop"
    side: Literal["bound", "attack"] = "bound"
    cites: tuple[str, ...] = (CONJECTURE, RULES)

    def allowance(self, p: Profile) -> Allowance:
        t = trained(p, self.model, self.rules, self.depth)

        def a(budget: FlopBudget) -> float:
            if math.isinf(budget.cut):
                return math.inf
            if self.signal:
                return t.within(budget) + t.tokens(budget.inputs)
            return t.within(budget) if t.behind() < budget.cut else math.inf
        return a


def _cut(budget: FlopBudget) -> bool:
    return math.isfinite(budget.cut)


def persists(p: Profile, rules: Rules) -> None:
    """Raise `Outside` unless what a construction trains inside an episode reaches the next one: under `outputs`, only
    a trained state that fits through `Y` and back in through `X`."""
    if rules.persist == "outputs" and p.state_bits > min(rules.x, rules.y):
        raise Outside(f"{p.name}'s {p.state_bits:.3g}-bit state leaves an episode only through Y "
                      f"({rules.y:.3g} bits): registration happens once")


@dataclass(frozen=True)
class LastLayers:
    """Train only the top layers the operator holding the loss commits with it (attack b): its weights, activations and
    the loss sit in one operator, so nothing crosses and a block under a forward cut trains unthrottled. The prover
    chooses the cuts, so what it may hold is what `p` fits at `weight_bits` (`Trained.held`); outside unless
    the gradient reaches no deeper. Blocks without a forward cut comply."""

    model: Model
    rules: Rules
    depth: int | None = None
    name: str = "pofc-last-layers"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = (ATTACKS,)

    def allowance(self, p: Profile) -> Allowance:
        persists(p, self.rules)
        t = trained(p, self.model, self.rules, self.depth)
        if t.depth > t.held:
            raise Outside(f"{p.name} trains {t.depth} layers; the operator holding the loss fits {t.held} within p")
        return lambda budget: t.kappa * budget.compute / 2 if _cut(budget) else 0.0


@dataclass(frozen=True)
class Palindrome:
    """Commit layer `i`'s weights to operators `i` and `k − i + 1`, so the chain's first half runs the forward and its
    second the backward (attack c): it pays by commitment, so it is open when the spare commitment holds every trained
    layer the last operator doesn't and `p` holds one at `weight_bits` (a backward operator's copy), and then
    trains unthrottled under a forward cut. Blocks without one comply."""

    model: Model
    rules: Rules
    depth: int | None = None
    name: str = "pofc-palindrome"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = (ATTACKS,)

    def allowance(self, p: Profile) -> Allowance:
        persists(p, self.rules)
        t = trained(p, self.model, self.rules, self.depth)
        need = max(0, t.depth - t.held) * t.layer_params * t.weight_bits
        if need > t.spare:
            raise Outside(f"{p.name} needs {need:.3g} bits committed again; P leaves {t.spare:.3g}")
        if self.rules.per_operator * self.rules.commit_bits < self.rules.weight_bits:
            raise Outside(f"p holds {self.rules.per_operator:g} honest layers, short of one at weight_bits")
        return lambda budget: t.kappa * budget.compute / 2 if _cut(budget) else 0.0


@dataclass(frozen=True)
class WeightsAsActivations:
    """Stream the trained layers' weights forward as activations beside the tokens (attack d): `c d_model²` parameters
    cross in about `c d_model` positions, every upstream layer sharing the one boundary, so a block under a forward cut
    trains the tokens the weights leave room for (`Trained.within`), nothing at `t ≥ 1`. At `bits` a weight,
    `weight_bits` unless given: an attacker content with serving-precision weights claws back the surcharge.
    Blocks without a forward cut comply."""

    model: Model
    rules: Rules
    depth: int | None = None
    bits: float | None = None
    name: str = "pofc-weights-as-activations"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = (ATTACKS,)

    def allowance(self, p: Profile) -> Allowance:
        persists(p, self.rules)
        t = trained(p, self.model, self.rules, self.depth)
        return lambda budget: t.within(budget, self.bits) if _cut(budget) else 0.0


@dataclass(frozen=True)
class JunkEpisodes:
    """Pad every episode to `n_max` with junk so each carries the whole cut, and stream weights in it (attack e). The
    wrapper's episodes are already `n_max` tokens; what the attack adds is the cap: a location runs at most
    `episodes_per_day` episodes, so padded episodes cover at most `share` of its declared work (`junk_share`)."""

    model: Model
    rules: Rules
    share: float
    depth: int | None = None
    name: str = "pofc-junk-episodes"
    side: Literal["bound", "attack"] = "attack"
    cites: tuple[str, ...] = (ATTACKS,)

    def allowance(self, p: Profile) -> Allowance:
        persists(p, self.rules)
        t = trained(p, self.model, self.rules, self.depth)
        return lambda budget: self.share * t.within(budget) if _cut(budget) else 0.0


#: The page's attacks that are algorithms with their own efficiency, not constructions in a block, and where each enters
#: the model: a family whose `efficiency.<run kind>` is that algorithm's compute to the target in backprop's multiples.
FAMILIES: Mapping[str, str] = {
    "small-models": "attack a: a model whose trained weights fit under the spare commitment plus one episode's boundary"
                    " traffic is the toll below 1 (`WeightsAsActivations`) or the palindrome, at a small model",
    "zeroth-order": "attack f: perturbations and losses read no backward signal, so it meets every rule at unit overhead"
                    " and is outside `pofc-backprop`; a family whose efficiency is zeroth-order's inefficiency, its"
                    " per-episode losses leaving through Y",
    "feedback-alignment": "attack g: its backward matrices are seed-generated, so no weights cross and it is outside"
                          " `pofc-backprop`; a family whose efficiency is its measured inefficiency at scale, the one"
                          " thing that excludes it",
    "compression": "attack h: seed-derivable weight state (a fresh init) regenerates free, so the toll counts trained"
                   " weights only; a run from scratch pays it on what it has trained so far",
}


@dataclass(frozen=True)
class Pofc:
    """`regime` under PoFC. Every cut with inference blocks gains a copy (`<cut>+pofc`) whose declared work runs as
    `episode`s (`n_max` tokens forward, `X` novel bits, `n_max s` bits across the narrowest forward cut), valid under
    the page's rules as well as the cut's own citations; NCI's cuts keep their blocks, whose cut stays `math.inf`. The
    layout's declared work runs as episodes. With `link` (bit/s; the physical instantiation: forward-only links, each
    GPU reading only its own static memory), every other block of the copies gets the link's bits over its time as its
    cut, so the serving cut gains a copy too. The layout's other blocks keep none: the constructions act on episodes
    only, and a strategy there still carries its state across the wipe through the block's inputs. `episode` is in the
    FLOP model; it enters a serving block in gates (`hardware.Units`)."""

    name: str
    regime: Regime
    episode: FlopBudget
    link: float | None = None

    def _block(self, b: Blocks, episode: Budget, rate: float, link: float | None) -> Blocks:
        if b.kind == "inference":
            return Blocks("episode", episode, b.count * b.budget.gates / episode.gates)
        if link is None:
            return b
        return replace(b, budget=replace(b.budget, cut=min(b.budget.cut, link * b.budget.gates / rate)))

    def _under(self, c: Cut, name: str, episode: Budget, rate: float, link: float | None) -> Cut:
        physical = link is not None and any(b.kind != "inference" for b in c.blocks)
        cites = c.cites + (RULES,) + ((PHYSICAL,) if physical else ())
        return Cut(name, tuple(self._block(b, episode, rate, link) for b in c.blocks), c.uncertified,
                   tuple(dict.fromkeys(cites)))

    def _touches(self, c: Cut) -> bool:
        return any(b.kind == "inference" or self.link is not None for b in c.blocks)

    def delta(self, ctx: Context) -> Delta:
        return self.regime.delta(ctx)

    def partition(self, ctx: Context) -> Partition:
        inner = self.regime.partition(ctx)
        e = self.episode
        episode = Budget(ctx.units.gates(e.compute), e.inputs, e.cut)
        # the blocks a link cuts run at capacity (the serving block is Φ_s Δ_wipe), so their time is gates over Φ
        rate = ctx.location.phi
        serving = tuple(
            replace(b, cuts=b.cuts + tuple(self._under(c, f"{c.name}+pofc", episode, rate, self.link)
                                           for c in b.cuts if self._touches(c)),
                    layout=self._under(b.layout, b.layout.name, episode, rate, None) if b.layout else None)
            for b in inner.serving)
        return replace(inner, serving=serving)


def rules(registry: Registry, name: str = "reference") -> Rules:
    return registry.fields(Rules, f"pofc.rules.{name}")


def stack(registry: Registry, name: str) -> Stack:
    return registry.fields(Stack, f"pofc.model.{name}")


def link(registry: Registry, location: str) -> float:
    return registry[f"pofc.link.{location}.bandwidth"]


def episode(rules: Rules, model: Model) -> FlopBudget:
    """One episode's budget: the forward of `n_max` tokens (`2 N_active n_max`: rule 6 calibrated to inference), `X`
    novel bits, and `n_max s` bits across its narrowest forward cut."""
    return FlopBudget(2 * model.n_active * rules.n_max, rules.x, rules.n_max * rules.s(model))


def wrap(regime: Regime, rules: Rules, model: Model, link: float | None = None) -> Pofc:
    return Pofc(f"{regime.name}+pofc", regime, episode(rules, model), link)


def bounds(model: Model, rules: Rules, depth: int | None = None) -> tuple[Backprop, ...]:
    return (Backprop(model, rules, depth),
            Backprop(model, rules, depth, signal=True, name="pofc-backprop-signal", cites=(CONJECTURE, RULES, SIGNAL)))


def matched_link(rules: Rules, model: Model, ctx: Context) -> float:
    """The link bandwidth (bit/s) that carries exactly the write allowance at the location's serving rate:
    `s · Φ · mfu_serve / (2 N_active)`, `Φ` in FLOP/s. A link `B` over a window `Δτ` carries `B Δτ / (n_max s)` episodes' cuts, so the
    physical instantiation bounds an episode as the declared rules do only with links this slow and dynamic memory
    cleared between episodes."""
    return rules.s(model) * ctx.units.flops(ctx.location.phi) * ctx.fleet.mfu_serve / (2 * model.n_active)


def junk_share(rules: Rules, model: Model, ctx: Context) -> float:
    """The share of a location's declared work that episodes padded to `n_max` cover when it runs at most
    `episodes_per_day` of them."""
    return min(1.0, rules.episodes_per_day * episode(rules, model).compute
               / (ctx.units.flops(ctx.location.phi) * DAY * ctx.fleet.mfu_serve))


def attacks(model: Model, rules: Rules, ctx: Context, depth: int | None = None) -> tuple[Any, ...]:
    return (LastLayers(model, rules, depth), Palindrome(model, rules, depth), WeightsAsActivations(model, rules, depth),
            WeightsAsActivations(model, rules, depth, rules.act_bits, "pofc-weights-as-activations-served"),
            JunkEpisodes(model, rules, junk_share(rules, model, ctx), depth))


def apply(s: Scenario, rules: Rules, link: float | None = None, depth: int | None = None) -> Scenario:
    """`s` under PoFC: its regime wrapped (`wrap`), and PoFC's bounds and attacks beside its own."""
    return replace(s, regime=wrap(s.regime, rules, s.model, link),
                   bounds=s.bounds + bounds(s.model, rules, depth),
                   attacks=s.attacks + attacks(s.model, rules, s.context, depth))


@dataclass(frozen=True)
class Top:
    """`family` training only the top `depth` layers: its state is theirs, and its target its own work (`m = 1`)."""

    family: Family
    depth: int

    @property
    def name(self) -> str:
        return f"{self.family.name}-top{self.depth}"

    def profile(self, model: Model, run: Run) -> Profile:
        p = self.family.profile(model, run)
        return replace(p, name=self.name, state_bits=p.state_bits * self.depth / model.layers, efficiency=1.0)


@dataclass(frozen=True)
class Point:
    """ρ for one family at one `n_max` on the scenario under PoFC: every bound at once (`lo`), `pofc-backprop` alone
    and with its signal premise, the best attack (`hi`), and `lo` without PoFC (`unwrapped`); `toll` is one episode's,
    at the depth the gradient reaches (`depth`, the whole model for full training)."""

    family: str
    depth: int
    n_max: float
    toll: float
    lo: float
    pofc: float
    signal: float
    hi: float
    unwrapped: float

    def json(self) -> dict[str, Any]:
        return asdict(self)


N_MAX = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8)


def versus(registry: Registry, scenario: str = "reference-onsite-h200", n_maxes: Iterable[float] = N_MAX,
           depths: Iterable[int | None] = (None, 1, 4, 16), rules_name: str = "reference",
           physical: bool = False) -> list[Point]:
    """ρ against `n_max` on `scenario` under PoFC, for its families' full training (`None`) and for training their top
    `depth` layers (`Top`), with PoFC's bounds and attacks beside the scenario's."""
    from rho.scenario import build, evaluate  # scenario may import this module

    base = build(registry, scenario)
    r0 = rules(registry, rules_name)
    bandwidth = link(registry, base.context.location.name) if physical else None
    n_maxes = tuple(n_maxes)
    out = []
    for depth in depths:
        families = base.families if depth is None else tuple(Top(f, depth) for f in base.families)
        plain = evaluate(replace(base, families=families))
        for n in n_maxes:
            r = replace(r0, n_max=n)
            s = apply(replace(base, families=families), r, bandwidth, depth)
            for f, f0 in zip(evaluate(s).families, plain.families, strict=True):
                rho = {e.argument: e.rho for e in f.bounds}
                t = trained(f.profile, base.model, r, depth)
                out.append(Point(f.family, t.depth, n, t.toll(episode(r, base.model)), f.lo, rho["pofc-backprop"],
                                 rho["pofc-backprop-signal"], f.hi, f0.lo))
    return out
