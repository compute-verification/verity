"""The two-stage first-hit finder's bound, evaluated exactly at a sampling design: each term under its premise.

The finder bounds the probability that an accepted run opens a value off the committed trace (the boundary's trace
or a sampled replay block's committed interior) by what a collision of SHA-512 costs it. Every figure is per sampled
replay block: the sample is one replay block's, its sessions and the gates they need. A run whose padded first stage
samples K0 replay blocks takes K0 times each figure, and a monitoring period the sum over its runs.

By position the bound is four terms, at e_s = 1 + 210·k: the first run with every session extracted and, for each
position the sample needs, a run with the sessions holding it extracted. Each term holds under its own premise on
SHA-512 (`ExpectedCR` and `StrictBirthday`, `security/Definitions/Core/Interaction.lean`), so a figure is a bound
only under the premises its line names:

    sessions    κ_s                                           the sampled sessions' knowledge error, as given
    rewinding   √(r·γ·Pr[B])                                  Pr[B] ≤ 2T/2^256 under `ExpectedCR` at B, or
                                                              Pr[B] ≤ (2T)²/2^513 under `StrictBirthday` at B
    conflict    2·T·(1 + N_s·e_s)/2^256                       `ExpectedCR` at the conflict finder, at that cost
    first hit   4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c)/2^256
                                                              `ExpectedCR` at the two first-hit finders, at every cap

By rows is by position with the gates counted by the rows that open together: K_s and K_m count those classes, and
the first-hit finders extract a class at a time. By gate is the union over the gates of a sample of one finder a
gate, with a cap on each finder's further runs:

    2·K·(T·(1 + d·e_s) + d·openCalls)/2^256 + p_ref·|opensSet| + |opensSet|/(M + 1)

* T: the calls of SHA-512 in a run, the prover's and the verifier's together, each call on a whole message. T caps
  every run, not the mean. B is two runs, so its calls are at most 2T.
* k: a session's accepted reruns a wait, ⌈4·N₀/e⌉ at a session whose level-0 codewords have N₀ symbols in all
  (`queries`). The extractor waits 210 times a session (`WAITS`), so e_s bounds the expected reruns of one session's
  extraction.
* N_s (`sessions`): the sessions of a sampled replay block's sample, in the mean. A design that states it is
  evaluated by position, or by rows (`rows`).
* K_s (`K`) and K_m (`Km`, K_s unless stated): the gates the sample needs, in the mean, counted once and counted
  session by session; by rows, the classes.
* c: the calls of the run's own opening checks, n_PU·openCalls (`units`, `open_calls`).
* r (`rounds`) and γ (`gamma`): the verifier's rounds and the sessions' γ, by default 4 rounds and 4,402 a session
  (`ROUNDS`) and 2^51 (`GAMMA`).
* K, by gate: the gates the sample needs, the most over the coins the verifier does not refuse. `card_gates` is its
  value at one proof block a stratum of a model's `workload` card, the design with no cap on a stratum's sample.
* d: the most sessions of one run that hold a sampled proof block needing one gate; by gate openCalls is one
  session's.
* p_ref: the probability that the verifier refuses the coins; |opensSet|: the gates any run can open; M: the cap on
  the further runs the finder waits for at one gate.

The premises count calls: a finder's cost is its calls of SHA-512, a call on 64 KiB counted as one on 64 bytes. So
each figure is given twice: in calls, as the Lean says, and in compressions, converted at the longest call. The
second is the first at L·T and L·c, where the longest call is L compressions (`compressions`, `LONGEST`): what a
premise counted in compressions gives when every call may be that long (`converted`).

`most_gates` is the largest K a target leaves the collision term (by position the first-hit term at K_s = K_m = K).

Every value is an integer or a `Fraction` until it is printed. The rewinding term is a rational at or above the
square root (`root_up`), and what is printed is ⌊−log₂⌋ in hundredths of a bit, found by integer arithmetic
(`centibits`): a term printed as 2^-132.03 is at most 2^-132.03.

    uv run python benchmarks/spot_figures/binding.py --K 2^37.69 --sessions 15694 --symbols 2^23 --first-stage 6
    uv run python benchmarks/spot_figures/binding.py --K 2^25.57 --sessions 15694 --symbols 2^23 --rows
    uv run python benchmarks/spot_figures/binding.py --card GLM-5.3 --symbols 2^10          # by gate, K from the card
    uv run python benchmarks/spot_figures/binding.py --K 2^40.23 --k 12343986 --d 167       # by gate, K, k and d given
                            # [--T 2^80] [--longest-call 65536] [--Km K] [--units N --open-calls C] [--rounds R]
                            # [--gamma G] [--kappa P] [--p-ref P] [--opens N --M N] [--first-stage K0]
                            # [--phase-seconds 600] [--machines N] [--target 100] [--json]
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
#: a collision of SHA-512 is found with probability at most a finder's expected compressions over this
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
#: what one figure covers, and what K0 of them cover
BLOCK_UNIT = "per sampled replay block"
#: e between two rationals: its series to 1/39!, and that plus 2/40!, more than the series' tail
E_LOW = sum(Fraction(1, factorial(n)) for n in range(40))
E_HIGH = E_LOW + Fraction(2, factorial(40))


def run_unit(first_stage: int) -> str:
    """What K0 sampled replay blocks cover, in words."""
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
_POSITION = {"Km": None, "units": 1, "rounds": None, "gamma": GAMMA, "kappa": None, "rows": False}
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
    rows: bool = False
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
            raise ValueError("a design with opens > 0 states M, the cap on the finder's further runs a gate")
        form, unread = ("by gate", _POSITION) if self.sessions is None else ("by position", _GATE)
        stated = [name for name, unstated in unread.items() if getattr(self, name) != unstated]
        if stated:
            raise ValueError(f"{form} the bound reads no {', '.join(stated)}")

    @property
    def form(self) -> str:
        """Which form the design is evaluated in: `gate` unless it states its sessions, then `position`, or `rows`
        where its gates are counted by rows."""
        return "gate" if self.sessions is None else "rows" if self.rows else "position"

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
        """r, the verifier's rounds: as stated, else `ROUNDS`' own and a session's."""
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
                "cap": "|opensSet|/(M + 1): the cap on a finder's further runs"}
    collision = "2T/2^256: `ExpectedCR` at B" if at_B == "ExpectedCR" else "(2T)²/2^513: `StrictBirthday` at B"
    kappa = "as given" if design.kappa is not None else "not given, counted as 0"
    return {"sessions": f"κ_s, the sampled sessions' knowledge error: {kappa}",
            "rewinding": f"√(r·γ·Pr[B]) at r = {design.rewinds} rounds, γ = 2^{log2(design.gamma)} and Pr[B] ≤ "
                         f"{collision}",
            "conflict": "2·T·(1 + N_s·e_s)/2^256: `ExpectedCR` at the conflict finder, its expected calls taken as "
                        "that",
            "first hit": "4·(T·(1 + e_s·N_s + K_s + e_s·K_m) + (1 + K_s)·c)/2^256: `ExpectedCR` at the two "
                         f"first-hit finders{' by rows, the gates of a row opening together' * design.rows}, at "
                         "every cap"}


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
    blocks."""
    if first_stage < 1:
        raise ValueError("a padded first stage samples at least one replay block")
    one = first_stage * bound(design, at_B)
    return {name: (n, n * one) for name, seconds in PERIODS for n in [runs(seconds, phase, machines)]}


def most_gates(design: Design, target: int, blocks: int = 1) -> int:
    """The most gates K at which the collision term over `blocks` sampled replay blocks is at most 2^-`target`, the
    design's other fields kept: by gate less its refused and cap terms, 0 where those are above the target alone; by
    position the first-hit term at K_s = K_m = K."""
    one, two = (replace(design, K=K, Km=None).calls for K in (1, 2))
    room = Fraction(HASH, 2 ** target * blocks)
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
           longest: int = LONGEST) -> dict:
    """The bound per sampled replay block, term by term with each term's premise, per run at a padded first stage of
    `first_stage` replay blocks and over each monitoring period. Each figure is given in `calls` and in
    `compressions` (`converted` at a longest call of `longest` bytes), with whether it is at most 2^-`target`. By
    position each sum is given under each premise at B (`READINGS`), which its `at_B` names."""
    goal = Fraction(1, 2 ** target)
    readings = READINGS if design.sessions is not None else (None,)
    counted = {"calls": design, "compressions": converted(design, longest)}
    run = run_unit(first_stage)

    def cells(figure, **more) -> dict:
        """One line: `figure` of the design in calls and of the design in compressions."""
        return {**more, **{count: {"bound": bits(p), "centibits": centibits(p), "exact": str(p), "meets": p <= goal}
                           for count, d in counted.items() for p in [figure(d)]}}

    def sums(times: int, unit: str, **more) -> list[dict]:
        return [cells(lambda d, at=at_B or READINGS[-1]: times * bound(d, at), at_B=at_B, unit=unit, **more)
                for at_B in readings]

    rows = []
    for name in terms(design):
        for at in (at_B or READINGS[-1] for at_B in readings):
            row = cells(lambda d, at=at, name=name: terms(d, at)[name], term=name, premise=premises(design, at)[name],
                        unit=BLOCK_UNIT)
            if row not in rows:
                rows.append(row)
    spans = {name: n for name, (n, _) in periods(design, phase, machines, first_stage).items()}

    def most(d: Design) -> dict[str, int]:
        return {"block": most_gates(d, target), "run": most_gates(d, target, first_stage),
                **{name: most_gates(d, target, n * first_stage) for name, n in spans.items()}}

    return {"design": {"form": design.form, "K": design.K, "Km": design.Km, "k": design.k, "d": design.d,
                       "T": design.T, "sessions": design.sessions, "units": design.units,
                       "open_calls": design.open_calls, "own_calls": design.own, "rounds": design.rewinds,
                       "gamma": design.gamma, "kappa": None if design.kappa is None else str(design.kappa),
                       "p_ref": str(design.p_ref), "opens": design.opens, "M": design.M, "reruns": design.reruns},
            "target": target, "first_stage": first_stage, "phase_seconds": phase, "machines": machines,
            "longest_call": {"bytes": longest, "compressions": compressions(longest)},
            "terms": rows, "block": sums(1, BLOCK_UNIT), "run": sums(first_stage, run),
            "periods": {name: sums(n * first_stage, run, runs=n) for name, n in spans.items()},
            "most_gates": {count: most(d) for count, d in counted.items()}}


def render(out: dict) -> str:
    """The report as lines, each saying what it covers: the design, each term per sampled replay block with its
    premise, the sums per sampled replay block and per run, then each monitoring period. Each figure is printed in
    calls, then in compressions."""
    d = out["design"]
    by_gate = d["form"] == "gate"
    longest, run = out["longest_call"], run_unit(out["first_stage"])

    def line(name: str, c: dict, text: str) -> str:
        pair = "".join(f"{c[count]['bound'] + ('' if c[count]['meets'] else '!'):<12}"
                       for count in ("calls", "compressions"))
        return f"  {name:<11}{pair}{text}"

    def summed(c: dict) -> str:
        return c["unit"] + (f", the rewinding term under `{c['at_B']}` at B" if c["at_B"] else "")

    def gates(n: int) -> str:
        return f"{n} {'row classes' if d['form'] == 'rows' else 'gates'} (2^{log2(n)})"

    budget = (f"T = 2^{log2(d['T'])} calls of SHA-512 in every run, the prover's and the verifier's, k = {d['k']}, "
              f"e_s = {d['reruns']}")
    if by_gate:
        lines = [f"by gate, {BLOCK_UNIT}: K = {gates(d['K'])}, d = {d['d']}, openCalls = {d['open_calls']}, {budget}"]
    else:
        own = f"c = n_PU·openCalls = {d['units']}·{d['open_calls']} = {d['own_calls']}"
        lines = [f"by {d['form']}, {BLOCK_UNIT}: N_s = {d['sessions']} sessions, K_s = {gates(d['K'])}, "
                 f"K_m = {gates(d['K'] if d['Km'] is None else d['Km'])}, {own}, {budget}"]
    lines.append(f"each figure {BLOCK_UNIT} or {run}, twice: in calls, as the Lean says, then in compressions, "
                 f"converted at the longest call ({longest['bytes']} bytes, {longest['compressions']} compressions); "
                 f"! is above 2^-{out['target']}")
    lines += [line(c["term"], c, f"{c['unit']}; {c['premise']}") for c in out["terms"]]
    lines += [line("sum", c, summed(c)) for span in ("block", "run") for c in out[span]]
    lines += [line(name, c, f"{c['runs']} runs, one an execution phase of {out['phase_seconds']} s on "
                            f"{out['machines']} machine(s), {summed(c)}")
              for name, cells in out["periods"].items() for c in cells]
    units = {"block": BLOCK_UNIT, "run": run}
    term = "the collision term beside the other two" if by_gate else "the first-hit term at K_s = K_m = K"
    for count, most in out["most_gates"].items():
        upto = "; ".join(f"{f'2^{log2(n)}' if n else 'none'} {units.get(name, f'over a {name} of such runs')}"
                         for name, n in most.items())
        lines.append(f"in {count}, {term} is at most 2^-{out['target']} up to K = {upto}")
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
    ap.add_argument("--rows", action="store_true",
                    help="by position with the gates counted by the rows that open together: --K and --Km count them")
    ap.add_argument("--Km", type=natural, help="by position, the gates counted session by session (default --K)")
    ap.add_argument("--units", type=natural, default=1, help="by position, n_PU: own calls are units·open-calls")
    ap.add_argument("--open-calls", type=natural, default=0, help="the calls of the opening checks")
    ap.add_argument("--rounds", type=natural, help="by position, the verifier's rounds (default 4 + 4402·sessions)")
    ap.add_argument("--gamma", type=natural, default=GAMMA, help="by position, the sessions' γ (default 2^51)")
    ap.add_argument("--kappa", type=probability, help="by position, the sessions' knowledge error (1/2^203)")
    ap.add_argument("--d", type=natural, default=1, help="by gate, the most sessions holding one gate (default 1)")
    ap.add_argument("--T", type=natural, default=BUDGET,
                    help="the calls of SHA-512 in every run, the prover's and the verifier's (default 2^80)")
    ap.add_argument("--longest-call", type=natural, default=LONGEST, metavar="BYTES",
                    help="the longest call's bytes, at which the figures in compressions are converted (default 65536)")
    ap.add_argument("--p-ref", type=probability, default=Fraction(0), help="the probability of refused coins")
    ap.add_argument("--opens", type=natural, default=0, help="the gates any run can open")
    ap.add_argument("--M", type=natural, help="the cap on the finder's further runs a gate (with --opens)")
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
                    rows=a.rows, p_ref=a.p_ref, opens=a.opens, M=a.M)
    out = report(design, a.phase_seconds, a.machines, a.target, a.first_stage, a.longest_call)
    print(json.dumps(out, indent=1) if a.json else render(out))


if __name__ == "__main__":
    main()
