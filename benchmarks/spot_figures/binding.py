"""The two-stage first-hit finder's bound, evaluated exactly at a sampling design: each term under its premise.

The finder bounds the probability that an accepted run opens a value off the committed trace (the boundary's trace
or a sampled replay block's committed interior) by what a collision of SHA-512 costs it. A design is stated per
sampled replay block: the sample is one replay block's, its sessions and the gates they need. A run whose padded
first stage samples K0 replay blocks has K0 times the sessions and the gates (`at_run`), and a monitoring period is
the sum over its runs. Every figure is per sampled replay block or per run, and says which.

By position the bound is four terms, at e_s = 1 + 210·k: the first run with every session extracted and, for each
position the sample needs, a run with the sessions holding it extracted. Each term holds under its own premise on
SHA-512 (`ExpectedCR` and `StrictBirthday`, `security/Definitions/Core/Interaction.lean`), so a figure is a bound
only under the premises its line names:

    sessions    κ_s                                           the sampled sessions' knowledge error, as given
    rewinding   √(r·γ·Pr[B])                                  Pr[B] ≤ 2T/2^256 under `ExpectedCR` at B, or
                                                              Pr[B] ≤ (2T)²/2^513 under `StrictBirthday` at B
    conflict    2·T·(1 + N_s·e_s)/2^256                       `ExpectedCR` at the conflict finder, at that cost
    first hit   4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c)/2^256
                                                              `ExpectedCR` at the two first-hit finders, at every limit

By classes is by position with the gates counted by the classes one reveal opens whole (a row's gates): K_s and
K_m count the classes, and the first-hit finders extract a class at a time (`by_classes`). By gate is the union over
the gates of a sample of one finder a gate, with a limit on each finder's further runs:

    2·K·(T·(1 + d·e_s) + d·openCalls)/2^256 + p_ref·|opensSet| + |opensSet|/(M + 1)

* T: the calls of SHA-512 in a run, the prover's and the verifier's together, each call on a whole message. T bounds
  every run, not the mean. B is two runs, so its calls are at most 2T.
* k: a session's accepted reruns a wait, ⌈4·N₀/e⌉ at a session whose level-0 codewords have N₀ symbols in all
  (`queries`). The extractor waits 210 times a session (`WAITS`), so e_s bounds the expected reruns of one session's
  extraction.
* N_s (`sessions`): the sessions of a sampled replay block's sample, in the mean. A design that states it is
  evaluated by position, or by classes (`by_class`).
* K_s (`K`) and K_m (`Km`, K_s unless stated): the gates the sample needs, in the mean, counted once and counted
  session by session; by classes, the classes.
* c: the calls of the run's own opening checks, n_PU·openCalls (`units`, `open_calls`).
* r (`rounds`) and γ (`gamma`): the verifier's rounds and the sessions' γ, by default 4 rounds and 4,402 a session
  (`ROUNDS`) and 2^51 (`GAMMA`).
* K, by gate: the gates the sample needs, the most over the coins the verifier does not refuse. `card_gates` is its
  value at one proof block a stratum of a model's `workload` card, the design with no limit on a stratum's sample.
* d: the most sessions of one run that hold a sampled proof block needing one gate; by gate openCalls is one
  session's.
* p_ref: the probability that the verifier refuses the coins; |opensSet|: the gates any run can open; M: the limit on
  the further runs the finder waits for at one gate.

The premises count calls: a finder's cost is its calls of SHA-512, a call on 64 KiB counted as one on 64 bytes. So
each figure is given twice: in calls, as the Lean says, and in compressions, converted at the longest call. The
second is the first at L·T and L·c, where the longest call is L compressions (`compressions`, `LONGEST`): what a
premise counted in compressions gives when every call may be that long (`converted`). A design by position that
also states its classes is printed in four columns: by gates and by classes, each in calls and in compressions.

K_s and K_m as given are means over the verifier's coins. A count of the sample bounds them by less: the blocks it
checks times the most gates one block needs (`as_counted`). A design that states that largest block (`largest`, and
`largest_classes` by classes) is printed in two more columns for each, beside the mean.

Beside the bound, per run: the term of the limit on the extractor's waits, (2·k/K + e^(−k/8)) at the outer waits plus
c times that at the inner, K = 2^n·k and c = 2^(1/1024), at each n of `CAPS` (`waits`). It is no term of the
first-hit bound and is in no sum.

`most_gates` is the largest K a target leaves the collision term (by position the first-hit term at K_s = K_m = K).

Every value is an integer or a `Fraction` until it is printed. The rewinding term is a rational at or above the
square root (`root_up`), and what is printed is ⌊−log₂⌋ in hundredths of a bit, found by integer arithmetic
(`centibits`): a term printed as 2^-132.03 is at most 2^-132.03.

    uv run python benchmarks/spot_figures/binding.py --K 2^41.81 --classes 2^26.24 --sessions 15694 --symbols 2^23 \
        --first-stage 6                                                                 # by gates and by classes
    uv run python benchmarks/spot_figures/binding.py --K 2^41.81 --sessions 10046500 --largest 632128 \
        --symbols 2^23 --first-stage 6                                    # the gates a mean, and as a count gives them
    uv run python benchmarks/spot_figures/binding.py --card GLM-5.3 --symbols 2^10          # by gate, K from the card
    uv run python benchmarks/spot_figures/binding.py --K 2^40.23 --k 12343986 --d 167       # by gate, K, k and d given
                            # [--T 2^80] [--longest-call 65536] [--Km K] [--units N --open-calls C] [--rounds R]
                            # [--gamma G] [--kappa P] [--waits-k 2^10] [--p-ref P] [--opens N --M N]
                            # [--largest M] [--largest-classes M] [--checked C]
                            # [--first-stage K0] [--phase-seconds 600] [--machines N] [--target 100] [--json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from math import ceil, factorial, floor, isqrt

from verity.protocols.reporting import PERIODS

#: the extractor's waits a session
WAITS = 210
#: a collision of SHA-512 is found with probability at most a finder's expected calls over this
HASH = 2 ** 256
#: a SHA-512 block's bytes, and the least padding a message takes (0x80 and its 16-byte length)
BLOCK, PAD = 128, 17
#: the calls of SHA-512 in a run the bound is quoted at
BUDGET = 2 ** 80
#: the bytes of the longest call the figures in compressions are converted at, a row of 64 KiB
LONGEST = 65536
#: an execution phase's seconds, one run each
PHASE = 600
#: the verifier's rounds the rewinding term is evaluated at unless they are stated: these, and these a session
ROUNDS = (4, 4402)
#: the sessions' γ the rewinding term is evaluated at unless it is stated
GAMMA = 2 ** 51
#: the limits on the extractor's waits the waits' term is given at, each as the power of two that k is multiplied by
CAPS = (128, 192)
#: the least k of a wait, where the waits' term is largest
WAITS_K = 2 ** 10
#: at or above c = 2^(1/1024), the factor on the inner waits' term
INNER = Fraction(1025, 1024)
#: what one figure covers
BLOCK_UNIT = "per sampled replay block"
#: e between two rationals: its series to 1/39!, and that plus 2/40!, more than the series' tail
E_LOW = sum(Fraction(1, factorial(n)) for n in range(40))
E_HIGH = E_LOW + Fraction(2, factorial(40))


def run_unit(first_stage: int) -> str:
    """What a figure at a run of K0 sampled replay blocks covers, in words."""
    return f"per run at a padded first stage of K0 = {first_stage} replay blocks"


def queries(symbols: int) -> int:
    """⌈4·symbols/e⌉, a session's accepted reruns a wait at a level-0 codeword of `symbols` symbols."""
    if symbols < 1:
        raise ValueError("a codeword has at least one symbol")
    low, high = ceil(Fraction(4 * symbols) / E_HIGH), ceil(Fraction(4 * symbols) / E_LOW)
    if low != high:
        raise ValueError(f"4·{symbols}/e is too near an integer for e's enclosure: {low} or {high}")
    return low


def compressions(row_bytes: int) -> int:
    """The SHA-512 compressions of one call on `row_bytes` bytes."""
    if row_bytes < 0:
        raise ValueError("a row has no negative length")
    return (row_bytes + PAD + BLOCK - 1) // BLOCK


def card_gates(card: dict) -> int:
    """K at one proof block a stratum of a `workload` card: the sum of each stratum's input and output bits."""
    total = 0
    for stratum in card["value"]["strata"]:
        for field in ("in_bits", "out_bits"):
            bits = stratum.get(field)
            if bits is None or bits != int(bits) or bits < 0:
                raise ValueError(f"{stratum['block']}: the workload card gives no whole {field}, so K is not known")
            total += int(bits)
    return total


def root_up(p: Fraction) -> Fraction:
    """A rational at or above √p, and within 2^-256 of it relatively where p's numerator is not 0."""
    p = Fraction(p)
    if p < 0:
        raise ValueError("a probability is not negative")
    square = p.numerator * p.denominator * HASH * HASH
    root = isqrt(square)
    return Fraction(root + (root * root < square), p.denominator * HASH)


#: by position's fields, which by gate does not read, each with the value that leaves it unstated
_POSITION = {"Km": None, "units": 1, "rounds": None, "gamma": GAMMA, "kappa": None, "by_class": False}
#: by gate's fields, which by position does not read
_GATE = {"d": 1, "opens": 0, "p_ref": Fraction(0)}


@dataclass(frozen=True, slots=True)
class Design:
    """What the bound reads of a sampling design and a run's calls (the module docstring names each field)."""

    K: int
    k: int
    d: int = 1
    T: int = BUDGET
    sessions: int | None = None
    Km: int | None = None
    units: int = 1
    open_calls: int = 0
    rounds: int | None = None
    gamma: int = GAMMA
    kappa: Fraction | None = None
    by_class: bool = False
    p_ref: Fraction = Fraction(0)
    opens: int = 0
    M: int | None = None

    def __post_init__(self) -> None:
        def whole(name: str, least: int) -> None:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < least:
                raise ValueError(f"{name} is {'a positive' if least else 'a non-negative'} integer, not {value!r}")

        for name in ("K", "k", "d", "T", "units", "gamma"):
            whole(name, 1)
        for name in ("open_calls", "opens"):
            whole(name, 0)
        for name in ("sessions", "Km", "rounds"):
            if getattr(self, name) is not None:
                whole(name, 1)
        for name in ("p_ref", "kappa"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, Fraction(getattr(self, name)))
                if not 0 <= getattr(self, name) <= 1:
                    raise ValueError(f"{name} is a probability, not {getattr(self, name)}")
        if self.opens and (not isinstance(self.M, int) or isinstance(self.M, bool) or self.M < 0):
            raise ValueError("a design with opens > 0 states M, the limit on the finder's further runs a gate")
        form, unread = ("by gate", _POSITION) if self.sessions is None else ("by position", _GATE)
        stated = [name for name, unstated in unread.items() if getattr(self, name) != unstated]
        if stated:
            raise ValueError(f"{form} the bound reads no {', '.join(stated)}")

    @property
    def form(self) -> str:
        """Which form the design is evaluated in: `gate` unless it states its sessions, then `position`, or
        `classes` where its gates are counted by classes."""
        return "gate" if self.sessions is None else "classes" if self.by_class else "position"

    @property
    def reruns(self) -> int:
        """e_s = 1 + 210·k."""
        return 1 + WAITS * self.k

    @property
    def own(self) -> int:
        """c, the calls of the run's own opening checks: n_PU·openCalls."""
        return self.units * self.open_calls

    @property
    def rewinds(self) -> int:
        """r, the verifier's rounds: as stated, for a run as for a sampled replay block, else `ROUNDS`' own and a
        session's."""
        return self.rounds if self.rounds is not None else ROUNDS[0] + ROUNDS[1] * (self.sessions or 0)

    @property
    def calls(self) -> int:
        """The collision term's numerator over 2^256: by position 4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c),
        the first-hit finders' expected calls four times; by gate 2·K·(T·(1 + d·e_s) + d·openCalls)."""
        e = self.reruns
        if self.sessions is None:
            return 2 * self.K * (self.T * (1 + self.d * e) + self.d * self.open_calls)
        Km = self.K if self.Km is None else self.Km
        return 4 * (self.T * (1 + e * self.sessions + self.K + e * Km) + (1 + self.K) * self.own)


def converted(design: Design, longest: int = LONGEST) -> Design:
    """The design counted in compressions, each call taken as long as the longest, one of `longest` bytes: T and the
    opening checks' calls times that call's compressions."""
    each = compressions(longest)
    return replace(design, T=each * design.T, open_calls=each * design.open_calls)


def at_run(design: Design, first_stage: int) -> Design:
    """The design of a run whose padded first stage samples `first_stage` replay blocks: that many times the
    sessions and the gates of one."""
    if first_stage < 1:
        raise ValueError("a padded first stage samples at least one replay block")
    return replace(design, K=first_stage * design.K, Km=None if design.Km is None else first_stage * design.Km,
                   sessions=None if design.sessions is None else first_stage * design.sessions)


def by_classes(design: Design, classes: int, Km: int | None = None) -> Design:
    """The design by position counted by classes: `classes` of them in the mean, `Km` session by session."""
    return replace(design, K=classes, Km=Km, by_class=True)


def as_counted(design: Design, largest: int, checked: int | None = None) -> Design:
    """The design by position with K_s and K_m as a count of the sample gives them: its `checked` blocks (its
    sessions unless stated, one session a checked block) times `largest`, the most gates, or classes, one block
    needs."""
    if design.sessions is None:
        raise ValueError("a count of the sample's checked blocks is of a design by position: state its sessions")
    if largest < 1 or (checked is not None and checked < 1):
        raise ValueError("the checked blocks and the largest block's needs are positive integers")
    return replace(design, K=(design.sessions if checked is None else checked) * largest, Km=None)


def waits(cap: int, k_out: int = WAITS_K, k_in: int = WAITS_K) -> Fraction:
    """A rational at or above the term of the limit on the extractor's waits, K = 2^`cap`·k:
    (2·k_out/K_out + e^(−k_out/8)) + c·(2·k_in/K_in + e^(−k_in/8)), with e^(−k/8) taken at ⌊k/8⌋ and e at `E_LOW`,
    c at `INNER`, and the sum rounded up to a multiple of 2^-512."""
    if cap < 0 or k_out < 1 or k_in < 1:
        raise ValueError("a limit on the waits is a power of two times a positive k")
    out, inner = (Fraction(2, 2 ** cap) + 1 / E_LOW ** (k // 8) for k in (k_out, k_in))
    return Fraction(ceil((out + INNER * inner) * HASH * HASH), HASH * HASH)


#: by position, the premise at B each sum takes the rewinding term under
READINGS = ("ExpectedCR", "StrictBirthday")


def terms(design: Design, at_B: str = "StrictBirthday") -> dict[str, Fraction]:
    """The bound's terms per sampled replay block, by name: by gate the collision, refused and cap terms; by position
    the four, the rewinding term under the premise `at_B` at B and the sessions' term 0 where κ_s is not stated."""
    if design.sessions is None:
        return {"collision": Fraction(design.calls, HASH), "refused": design.p_ref * design.opens,
                "cap": Fraction(design.opens, design.M + 1) if design.opens else Fraction(0)}
    if at_B not in READINGS:
        raise ValueError(f"the premise at B is one of {', '.join(READINGS)}, not {at_B!r}")
    two = 2 * design.T
    collision = Fraction(two, HASH) if at_B == "ExpectedCR" else Fraction(two * two, 2 * HASH * HASH)
    return {"sessions": Fraction(0) if design.kappa is None else design.kappa,
            "rewinding": root_up(design.rewinds * design.gamma * collision),
            "conflict": Fraction(two * (1 + design.sessions * design.reruns), HASH),
            "first hit": Fraction(design.calls, HASH)}


def premises(design: Design, at_B: str = "StrictBirthday") -> dict[str, str]:
    """What each of `terms`' figures is, and the premise it holds under."""
    if design.sessions is None:
        return {"collision": "2·K·(T·(1 + d·e_s) + d·openCalls)/2^256: `ExpectedCR` at each gate's finder",
                "refused": "p_ref·|opensSet|: the probability of refused coins, as given",
                "cap": "|opensSet|/(M + 1): the limit on a finder's further runs"}
    collision = "2T/2^256: `ExpectedCR` at B" if at_B == "ExpectedCR" else "(2T)²/2^513: `StrictBirthday` at B"
    kappa = "as given" if design.kappa is not None else "not given, counted as 0"
    return {"sessions": f"κ_s, the sampled sessions' knowledge error: {kappa}",
            "rewinding": f"√(r·γ·Pr[B]) at r = {design.rewinds} rounds, γ = 2^{log2(design.gamma)} and Pr[B] ≤ "
                         f"{collision}",
            "conflict": "2·T·(1 + N_s·e_s)/2^256: `ExpectedCR` at the conflict finder, its expected calls taken as "
                        "that",
            "first hit": "4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c)/2^256: `ExpectedCR` at the two "
                         f"first-hit finders{' by classes, one reveal opening a class whole' * design.by_class}, at "
                         "every limit"}


def bound(design: Design, at_B: str = "StrictBirthday") -> Fraction:
    """The bound per sampled replay block: the sum of `terms`."""
    return sum(terms(design, at_B).values(), Fraction(0))


def runs(seconds: int, phase: int = PHASE, machines: int = 1) -> int:
    """The runs of `seconds` on `machines` machines, one an execution phase of `phase` seconds, a started phase
    counted."""
    if phase < 1 or machines < 1 or seconds < 0:
        raise ValueError("a period has a positive phase and at least one machine")
    return machines * -(-seconds // phase)


def periods(design: Design, phase: int = PHASE, machines: int = 1, first_stage: int = 1,
            at_B: str = "StrictBirthday") -> dict[str, tuple[int, Fraction]]:
    """Each monitoring period's runs and its bound: the sum over its runs, each of `first_stage` sampled replay
    blocks (`at_run`)."""
    one = bound(at_run(design, first_stage), at_B)
    return {name: (n, n * one) for name, seconds in PERIODS for n in [runs(seconds, phase, machines)]}


def most_gates(design: Design, target: int, runs: int = 1, first_stage: int = 1) -> int:
    """The most gates K of a sampled replay block at which the collision term, summed over `runs` runs of
    `first_stage` replay blocks each, is at most 2^-`target`, the design's other fields kept: by gate less its
    refused and cap terms, 0 where those are above the target alone; by position the first-hit term at
    K_s = K_m = K."""
    one, two = (at_run(replace(design, K=K, Km=None), first_stage).calls for K in (1, 2))
    room = Fraction(HASH, 2 ** target * runs)
    if design.sessions is None:
        fixed = terms(design)
        room -= (fixed["refused"] + fixed["cap"]) * HASH
    return max(0, floor((room - (2 * one - two)) / (two - one)))


def centibits(p: Fraction) -> int | None:
    """⌊−100·log₂ p⌋ for a positive `p`, exactly; None at 0."""
    p = Fraction(p)
    if p < 0:
        raise ValueError("a probability is not negative")
    if p == 0:
        return None
    a, b = p.denominator ** 100, p.numerator ** 100
    shift = a.bit_length() - b.bit_length()
    over = a >= b << shift if shift >= 0 else a << -shift >= b
    return shift if over else shift - 1


def bits(p: Fraction) -> str:
    """`p` as a power of two with its exponent cut to hundredths, never understating `p`: `2^-132.03`, `0`."""
    c = centibits(p)
    if c is None:
        return "0"
    whole, hundredths = divmod(abs(c), 100)
    return f"2^{'-' if c > 0 else ''}{whole}.{hundredths:02d}"


def log2(n: int) -> str:
    """log₂ of a positive count, cut to hundredths."""
    c = -centibits(Fraction(n)) - 1 if n & (n - 1) else 100 * (n.bit_length() - 1)
    return f"{c // 100}.{c % 100:02d}"


def natural(text: str) -> int:
    """A count as written: a product of integers and powers (`3*2^80`). A power's exponent may be a decimal
    (`2^40.23`), and that factor is then the least integer at or above it."""
    total = 1
    for factor in text.replace("**", "^").replace(",", "").replace("_", "").split("*"):
        base, _, exponent = factor.partition("^")
        b, e = int(base), Fraction(exponent or 1)
        if b < 1 or e < 0:
            raise ValueError(f"{factor!r} is not a count")
        total *= _root_up(b ** e.numerator, e.denominator)
    return total


def _root_up(n: int, q: int) -> int:
    """The least integer r with r^q >= n."""
    low, high = 0, 1
    while high ** q < n:
        high *= 2
    while high - low > 1:
        mid = (low + high) // 2
        low, high = (low, mid) if mid ** q >= n else (mid, high)
    return high


def probability(text: str) -> Fraction:
    """A probability as written: 0, a count, or one count over another (`1/2^300`)."""
    top, _, bottom = text.partition("/")
    return Fraction(0) if top.strip() == "0" else Fraction(natural(top), natural(bottom or "1"))


def report(design: Design, phase: int = PHASE, machines: int = 1, target: int = 100, first_stage: int = 1,
           longest: int = LONGEST, classes: int | None = None, waits_k: int = WAITS_K, largest: int | None = None,
           largest_classes: int | None = None, checked: int | None = None) -> dict:
    """The bound per sampled replay block, term by term with each term's premise, per run at a padded first stage of
    `first_stage` replay blocks and over each monitoring period. Each figure is given in each of `columns`: by gates
    in calls and in compressions (`converted` at a longest call of `longest` bytes) and, where a design by position
    states its `classes`, by classes in both, with whether it is at most 2^-`target`. Where it states the `largest`
    block's gates, or its `largest_classes`, the same with that count `as_counted` over the `checked` blocks, beside
    the count as given. By position each sum is given under each premise at B (`READINGS`), which its `at_B` names.
    `waits` is the waits' term at each limit, per run."""
    goal = Fraction(1, 2 ** target)
    readings = READINGS if design.sessions is not None else (None,)
    counts = {"gates": design}
    if largest is not None:
        counts["gates as counted"] = as_counted(design, largest, checked)
    if classes is not None:
        counts["classes"] = by_classes(design, classes)
    if largest_classes is not None:
        counts["classes as counted"] = by_classes(design, as_counted(design, largest_classes, checked).K)
    columns = {f"{count}, {unit}": d for count, counted in counts.items()
               for unit, d in (("calls", counted), ("compressions", converted(counted, longest)))}
    run = run_unit(first_stage)

    def cell(p: Fraction) -> dict:
        return {"bound": bits(p), "centibits": centibits(p), "exact": str(p), "meets": p <= goal}

    def cells(figure, **more) -> dict:
        """One line: `figure` of the design in each column."""
        return {**more, "columns": {column: cell(figure(d)) for column, d in columns.items()}}

    def sums(times: int, stage: int, unit: str, **more) -> list[dict]:
        return [cells(lambda d, at=at_B or READINGS[-1]: times * bound(at_run(d, stage), at), at_B=at_B, unit=unit,
                      **more) for at_B in readings]

    rows = []
    for name in terms(design):
        for at in (at_B or READINGS[-1] for at_B in readings):
            row = cells(lambda d, at=at, name=name: terms(d, at)[name], term=name, unit=BLOCK_UNIT,
                        premise=premises(design, at)[name])
            if row not in rows:
                rows.append(row)
    spans = {name: n for name, (n, _) in periods(design, phase, machines, first_stage).items()}

    def most(d: Design) -> dict[str, int]:
        return {"block": most_gates(d, target), "run": most_gates(d, target, 1, first_stage),
                **{name: most_gates(d, target, n, first_stage) for name, n in spans.items()}}

    counted = {count: {"checked": design.sessions if checked is None else checked, "largest": most_of,
                       "K": counts[count].K}
               for count, most_of in (("gates as counted", largest), ("classes as counted", largest_classes))
               if most_of is not None}
    return {"design": {"form": design.form, "K": design.K, "Km": design.Km, "classes": classes, "counted": counted,
                       "k": design.k,
                       "d": design.d, "T": design.T, "sessions": design.sessions, "units": design.units,
                       "open_calls": design.open_calls, "own_calls": design.own, "rounds": design.rewinds,
                       "gamma": design.gamma, "kappa": None if design.kappa is None else str(design.kappa),
                       "p_ref": str(design.p_ref), "opens": design.opens, "M": design.M, "reruns": design.reruns},
            "target": target, "first_stage": first_stage, "phase_seconds": phase, "machines": machines,
            "longest_call": {"bytes": longest, "compressions": compressions(longest)}, "columns": list(columns),
            "terms": rows, "block": sums(1, 1, BLOCK_UNIT), "run": sums(1, first_stage, run),
            "periods": {name: sums(n, first_stage, run, runs=n) for name, n in spans.items()},
            "waits": [{"cap": cap, "k": waits_k, "unit": run, **cell(waits(cap, waits_k, waits_k))} for cap in CAPS],
            "most_gates": {column: most(d) for column, d in columns.items() if "as counted" not in column}}


#: what a finder's cost is counted in, each a column of `over_phases`: calls, as the Lean counts, and compressions
COUNTED = ("calls", "compressions")


@dataclass(frozen=True, slots=True)
class Instance:
    """One of Lean's instances of the bound by classes at numbers (`theorem`, in `Claims.Flock.Numbers`): at a run
    whose budget of calls `T`, e_s (`e`), N_s, K_s, K_m and c = n_PU·openCalls are within these numerals, the conflict
    finder's cost and the first-hit finders', summed over 2^256, are at most 2^-`bits`."""

    theorem: str
    T: int
    e: int
    Ns: int
    Ks: int
    Km: int
    c: int
    bits: int


#: e_s at a session of one table of at most 2^23 level-0 symbols, 1 + 210·⌈4·2^23/e⌉: the instances' own
ONE_TABLE = 1 + WAITS * 12343986
#: Lean's instances on the right-hand side of `CompiledTwo.eventTwoX_le_cls_at_budget`, as their statements give
#: them (`security/Proofs/Flock/SamplingGeneric/TwoFirstHitClassNumbers.lean`): one at runs of 2^80 calls, the others
#: at 513 times that, every call charged as the compressions of one on 64 KiB
LEAN = (Instance("εclsAt_le_table1", BUDGET, ONE_TABLE, 2 ** 30, 2 ** 42, 2 ** 42, BUDGET, 100),
        Instance("εclsAt_le_table1_513", 513 * BUDGET, ONE_TABLE, 2 ** 30, 2 ** 33, 2 ** 33, 513 * BUDGET, 100),
        Instance("εclsAt_le_table1_513_narrow", 513 * BUDGET, ONE_TABLE, 2 ** 27, 14 * 10 ** 9, 14 * 10 ** 9,
                 513 * BUDGET, 100),
        Instance("εclsAt_le_table1_513_k35", 513 * BUDGET, ONE_TABLE, 2 ** 30, 2 ** 35, 2 ** 35, 513 * BUDGET, 98))


def sizes(design: Design) -> dict[str, int]:
    """What an `Instance`'s numerals bound, of a run's design by position or by classes."""
    if design.sessions is None:
        raise ValueError("Lean's instances are of a design by classes: state its sessions")
    return {"T": design.T, "e": design.reruns, "Ns": design.sessions, "Ks": design.K,
            "Km": design.K if design.Km is None else design.Km, "c": design.own}


def proved(design: Design) -> dict:
    """What Lean proves at numbers of one run's design by classes: among `LEAN`'s instances whose numerals the design
    is within, the one of the most bits (`theorem`, `bits`). Where it is within none's, `theorem` is None, beside
    the instance it is past in the fewest numerals, among those at its budget or above (`nearest`, `bits`), and
    each numeral it is past with the design's value and the instance's (`past`)."""
    if design.form != "classes":
        raise ValueError("Lean's instances are of the bound by classes: count the design's gates by classes")
    mine = sizes(design)
    past = {i.theorem: [name for name, value in mine.items() if value > getattr(i, name)] for i in LEAN}
    within = [i for i in LEAN if not past[i.theorem]]
    if within:
        best = max(within, key=lambda i: i.bits)
        return {"theorem": best.theorem, "bits": best.bits}
    near = min([i for i in LEAN if "T" not in past[i.theorem]] or LEAN, key=lambda i: (len(past[i.theorem]), -i.Ks))
    return {"theorem": None, "nearest": near.theorem, "bits": near.bits,
            "past": {name: {"design": mine[name], "instance": getattr(near, name)} for name in past[near.theorem]}}


def over_phases(sessions: int, classes: int, k: int, phases: int, T: int = BUDGET, longest: int = LONGEST) -> dict:
    """The bound by classes of one verification phase and of `phases` of them, in each of `COUNTED`.

    A verification phase is one run: its sample has `sessions` sessions and needs `classes` classes of gates, counted
    once and session by session (K_s and K_m), and `k` is the accepted reruns a wait of its largest session. The
    figure is `bound` at that design: the rewinding term under `StrictBirthday` at B, with κ_s and the run's own
    opening checks not stated and counted as 0. In calls that is the right-hand side of
    `CompiledTwo.eventTwoX_le_cls_at_budget` at those numbers, and in compressions the same at the budget times the
    longest call's compressions (`converted`). `phases` of them is the sum over them, which no Lean theorem states.
    Beside each counting: what Lean proves at numbers of one phase's part over 2^256 (`proved`)."""
    if phases < 1:
        raise ValueError("a period has at least one verification phase")
    design = Design(K=classes, k=k, T=T, sessions=sessions, by_class=True)

    def cell(p: Fraction) -> dict:
        return {"bound": bits(p), "centibits": centibits(p)}

    return {"design": {"sessions": sessions, "classes": classes, "k": k, "reruns": design.reruns,
                       "calls_a_run": f"2^{log2(T)}", "rounds": design.rewinds, "phases": phases,
                       "longest_call": {"bytes": longest, "compressions": compressions(longest)}},
            "counted": {counted: {"phase": cell(one), "phases": cell(phases * one), "lean": proved(d)}
                        for counted, d in zip(COUNTED, (design, converted(design, longest)), strict=True)
                        for one in [bound(d)]}}


def render(out: dict) -> str:
    """The report as lines, each saying what it covers: the design, each term per sampled replay block with its
    premise, the sums per sampled replay block and per run, each monitoring period, then the waits' term. Each
    figure is printed in each column."""
    d = out["design"]
    by_gate = d["form"] == "gate"
    longest, run = out["longest_call"], run_unit(out["first_stage"])

    def marked(c: dict) -> str:
        return f"{c['bound'] + ('' if c['meets'] else '!'):<12}"

    def line(name: str, c: dict, text: str) -> str:
        return f"  {name:<11}{''.join(marked(c['columns'][column]) for column in out['columns'])}{text}"

    def summed(c: dict) -> str:
        return c["unit"] + (f", the rewinding term under `{c['at_B']}` at B" if c["at_B"] else "")

    def count(n: int, of: str = "gates") -> str:
        return f"{n} {of} (2^{log2(n)})"

    budget = (f"T = 2^{log2(d['T'])} calls of SHA-512 in every run, the prover's and the verifier's, k = {d['k']}, "
              f"e_s = {d['reruns']}")
    if by_gate:
        lines = [f"by gate, {BLOCK_UNIT}: K = {count(d['K'])}, d = {d['d']}, openCalls = {d['open_calls']}, {budget}"]
    else:
        of = "classes" if d["form"] == "classes" else "gates"
        classes = f", K_cls = {count(d['classes'], 'classes')}" if d["classes"] is not None else ""
        own = f"c = n_PU·openCalls = {d['units']}·{d['open_calls']} = {d['own_calls']}"
        lines = [f"by {d['form']}, {BLOCK_UNIT}: N_s = {d['sessions']} sessions, K_s = {count(d['K'], of)}, "
                 f"K_m = {count(d['K'] if d['Km'] is None else d['Km'], of)}{classes}, {own}, {budget}"]
    for name, c in d["counted"].items():
        of = name.split()[0]
        lines.append(f"{name}, {BLOCK_UNIT}: K_s = K_m = {c['checked']} checked blocks times the {c['largest']} {of} "
                     f"the largest block needs = {count(c['K'], of)}; as given, K_s and K_m are means over the coins")
    lines.append(f"each figure {BLOCK_UNIT} or {run}, in {len(out['columns'])} columns: "
                 f"{'; '.join(f'by {column}' for column in out['columns'])}. Calls are as the Lean says; compressions "
                 f"are converted at the longest call ({longest['bytes']} bytes, {longest['compressions']} "
                 f"compressions). A run has {out['first_stage']} times the sessions and the gates. ! is above "
                 f"2^-{out['target']}")
    lines += [line(c["term"], c, f"{c['unit']}; {c['premise']}") for c in out["terms"]]
    lines += [line("sum", c, summed(c)) for span in ("block", "run") for c in out[span]]
    lines += [line(name, c, f"{c['runs']} runs, one an execution phase of {out['phase_seconds']} s on "
                            f"{out['machines']} machine(s), {summed(c)}")
              for name, cells in out["periods"].items() for c in cells]
    lines += [f"  {'waits':<11}{marked(c) * len(out['columns'])}{c['unit']}; the limit on the extractor's waits at "
              f"2^{c['cap']}·k, k = {c['k']} at both waits: (1 + c)·(2/2^{c['cap']} + e^(−k/8)); in no sum"
              for c in out["waits"]]
    units = {"block": BLOCK_UNIT, "run": run}
    term = "the collision term beside the other two" if by_gate else "the first-hit term at K_s = K_m = K"
    for column, most in out["most_gates"].items():
        upto = "; ".join(f"{f'2^{log2(n)}' if n else 'none'} {units.get(name, f'over a {name} of such runs')}"
                         for name, n in most.items())
        lines.append(f"by {column}, {term} is at most 2^-{out['target']} up to K = {upto}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    k = ap.add_mutually_exclusive_group(required=True)
    k.add_argument("--K", type=natural,
                   help="the gates a sampled replay block's sample needs: the most by gate, the mean by position "
                        "(2^40.23 is rounded up)")
    k.add_argument("--card", metavar="MODEL", help="K at one proof block a stratum of the model's workload card")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--k", type=natural, help="a session's accepted reruns a wait")
    q.add_argument("--symbols", type=natural, help="a table's level-0 codeword symbols: k = ⌈4·tables·symbols/e⌉")
    ap.add_argument("--tables", type=natural, default=1, help="the tables of one session, with --symbols (default 1)")
    ap.add_argument("--sessions", type=natural, help="the sample's sessions, in the mean: evaluate by position")
    ap.add_argument("--classes", type=natural, metavar="K_CLS",
                    help="by position, the classes of gates one reveal opens whole: print by classes beside by gates")
    ap.add_argument("--largest", type=natural, metavar="M",
                    help="by position, the most gates one block needs: print the gates as a count of the checked "
                         "blocks gives them, beside the mean")
    ap.add_argument("--largest-classes", type=natural, metavar="M",
                    help="by position, the most classes one block needs: the same by classes")
    ap.add_argument("--checked", type=natural, metavar="C",
                    help="the blocks a sampled replay block's sample checks, with --largest (default --sessions)")
    ap.add_argument("--Km", type=natural, help="by position, the gates counted session by session (default --K)")
    ap.add_argument("--units", type=natural, default=1, help="by position, n_PU: own calls are units·open-calls")
    ap.add_argument("--open-calls", type=natural, default=0, help="the calls of the opening checks")
    ap.add_argument("--rounds", type=natural, help="by position, the verifier's rounds (default 4 + 4402·sessions)")
    ap.add_argument("--gamma", type=natural, default=GAMMA, help="by position, the sessions' γ (default 2^51)")
    ap.add_argument("--kappa", type=probability, help="by position, the sessions' knowledge error (1/2^203)")
    ap.add_argument("--waits-k", type=natural, default=WAITS_K,
                    help="the k of a wait the waits' term is given at, at both waits (default 2^10)")
    ap.add_argument("--d", type=natural, default=1, help="by gate, the most sessions holding one gate (default 1)")
    ap.add_argument("--T", type=natural, default=BUDGET,
                    help="the calls of SHA-512 in every run, the prover's and the verifier's (default 2^80)")
    ap.add_argument("--longest-call", type=natural, default=LONGEST, metavar="BYTES",
                    help="the longest call's bytes, at which the figures in compressions are converted (default 65536)")
    ap.add_argument("--p-ref", type=probability, default=Fraction(0), help="the probability of refused coins")
    ap.add_argument("--opens", type=natural, default=0, help="the gates any run can open")
    ap.add_argument("--M", type=natural, help="the limit on the finder's further runs a gate (with --opens)")
    ap.add_argument("--first-stage", type=natural, default=1, metavar="K0",
                    help="the replay blocks a run's padded first stage samples (default 1)")
    ap.add_argument("--phase-seconds", type=natural, default=PHASE, help="an execution phase's seconds (default 600)")
    ap.add_argument("--machines", type=natural, default=1, help="the machines audited over a period (default 1)")
    ap.add_argument("--target", type=int, default=100, help="the bits each bound is held to (default 100)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    gates = a.K
    if a.card:
        import cards
        gates = card_gates(cards.load("workload", a.card))
    design = Design(K=gates, k=a.k if a.k else queries(a.tables * a.symbols), d=a.d, T=a.T, sessions=a.sessions,
                    Km=a.Km, units=a.units, open_calls=a.open_calls, rounds=a.rounds, gamma=a.gamma, kappa=a.kappa,
                    p_ref=a.p_ref, opens=a.opens, M=a.M)
    stated = [flag for flag, value in (("--classes", a.classes), ("--largest", a.largest),
                                       ("--largest-classes", a.largest_classes), ("--checked", a.checked))
              if value is not None]
    if stated and a.sessions is None:
        ap.error(f"{', '.join(stated)} counts a design by position: state --sessions")
    if a.checked is not None and a.largest is None and a.largest_classes is None:
        ap.error("--checked is the count --largest or --largest-classes multiplies: state one")
    out = report(design, a.phase_seconds, a.machines, a.target, a.first_stage, a.longest_call, a.classes, a.waits_k,
                 a.largest, a.largest_classes, a.checked)
    print(json.dumps(out, indent=1) if a.json else render(out))


if __name__ == "__main__":
    main()
