import itertools
import math
from fractions import Fraction

from rho import envelopes, operating
from rho.envelopes import Detection
from rho.params import Registry
from rho.scenario import build

from verity.protocols.spot.verifier.profile import MARGIN, WINDOW_DELTA, Bernoulli, IntegrityProfile

REG = Registry.load()


def test_the_injected_bits_are_held_to_cores_bound_for_one_bernoulli_level():
    """The calculator's injected bits and core's bound on the incorrect checked members stay two definitions.
    `Detection.injected` (and `operating.Incorrect.capacity`, the same expression) counts over any span, with no
    population: the proof blocks one identity gets accepted at sample rate `q` before its first detection,
    `ln(1/δ) / q`, times the identities `J`, plus a floor class's steps. Core's `harm_bound` bounds one profile, a
    population under its sampling design. This holds them together where they meet, one `Bernoulli(q)` level over a
    row's served proof blocks, each weighing its output bits. The count less its floor term lies between
    `core / (1 + 3 * MARGIN)` and `core * (1 + q)`: at most `q` above core's rounded bound and, the sharper lower
    end, never below core's optimum before its directed rounding, `ln(1/δ) / ρ(1)` blocks at the level's risk
    `ρ(1) = -ln(1 - q)` (to one part in 10¹², the floats' room).

    The corners are the registry's ranges of `proof_share` and `overhead` (which set `q`) and of `p_evade`, each at
    its low end, its value and its high end, on every fleet and monitoring period `periods.py` reports; none of those
    populations is small enough for its whole harm to limit core's bound. Each `δ` is the calculator's own, and this
    changes none: `injected` is at `p_evade`, which the registry sets to core's `WINDOW_DELTA`, 2⁻⁴⁰, and
    `operating.incorrect`'s row takes the same expression at `incorrect.delta`, the least reporting `δ`, 10⁻⁶, and
    is held here at that `δ`.

    A `Bernoulli` level's bound is `worst_case`'s, its risks computed from exact integers: it never runs the
    stratified design's search (`Stratified.harm_bound`), so this rests on nothing about that search."""
    import periods

    name = "reference-remote-h200-sampled"
    regime = REG[f"scenario.{name}.regime"]
    ranged = [REG.param(f"regime.{regime}.{leaf}") for leaf in ("proof_share", "overhead", "p_evade")]
    rows = [build(REG.override(periods.choices(name, fleet, period)), name).context
            for fleet, _ in periods.FLEETS for period in periods.PERIODS]
    assert rows and all(p.low < p.value < p.high for p in ranged)

    def hold(bits: float, d: Detection, q: float, delta: float) -> None:
        """`bits`, counted at sample rate `q` and `delta`, against core's bound over each row's served proof blocks."""
        for ctx in rows:
            served = math.floor(ctx.units.flops(ctx.capacity) * ctx.fleet.mfu_serve / d.step_flop)
            level = IntegrityProfile(("unit",), (served,), {"unit": Bernoulli(Fraction(q))}, delta)
            core = d.identities * level.harm_bound(d.step_bits)
            unrounded = d.identities * d.step_bits * math.log(1 / delta) / level.rho(1)
            sampled = bits - d.identities * math.log(1 / delta) * d.floor_steps * d.step_bits
            at = (q, delta, ctx.fleet.name, ctx.period.name)
            assert core / (1 + 3 * MARGIN) <= sampled <= core * (1 + q), at
            assert unrounded * (1 - 1e-12) <= sampled, at

    for corner in itertools.product(*((p.low, p.value, p.high) for p in ranged)):
        d = envelopes.regime(REG.override({p.key: x for p, x in zip(ranged, corner, strict=True)}), regime).detection
        assert d.sampled and (d.proof_share, d.overhead, d.p_evade) == corner
        assert operating.Incorrect(d.q(), d.p_evade, d.step_bits, d.floor_steps, d.identities).capacity == d.injected()
        hold(d.injected(), d, d.q(), d.p_evade)
    reference = build(REG, name)
    d, row = reference.regime.detection, operating.incorrect(REG, reference)
    assert d.p_evade == float(WINDOW_DELTA) == 2.0 ** -40 and row.delta == REG["incorrect.delta"] == 1e-6
    assert row.q == d.q()
    hold(row.capacity, d, row.q, row.delta)
