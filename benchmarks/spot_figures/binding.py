"""SPOT's binding error by the two-stage first-hit finder, evaluated exactly at a sampling design.

The finder bounds the probability that an accepted run opens a value off the committed trace (the boundary's trace
or a sampled replay block's committed interior) by

    2·K·(T·(1 + d·e_s) + d·openCalls)/2^256 + p_ref·|opensSet| + |opensSet|/(M + 1),        e_s = 1 + 210·k

* K: the gates the proof blocks of one audit's sample need, the most over the coins the verifier does not refuse.
  `card_gates` is its value at one proof block a stratum of a model's `workload` card, the design with no cap on a
  stratum's sample; any other design's K is a parameter.
* T: the SHA-512 compressions of one run, the prover's and the verifier's (`compressions` converts calls on rows).
* d: the most sessions of one run that hold a sampled proof block needing one gate.
* k: a session's accepted reruns a wait, ⌈4·N₀/e⌉ at a session whose level-0 codeword has N₀ symbols (`queries`).
  The extractor waits 210 times a session (`WAITS`), so e_s bounds the expected reruns of one session's
  extraction.
* openCalls: the compressions of one session's opening checks.
* p_ref: the probability that the verifier refuses the coins; |opensSet|: the gates any run can open.
* M: the cap on the further runs the finder waits for at one gate.

The first term is what a collision of SHA-512 costs the finder: its expected compressions over 2^256, twice, at each
of the K gates. The other two are zero at the defaults (`open_calls`, `p_ref` and `opens` all 0).

An audit's bound holds for each audit of a monitoring period, so a period's is the sum over its audits: one audit an
execution phase on each machine (`periods`, over `verity.protocols.reporting.PERIODS`). `most_gates` is the largest K
a target leaves, at one audit or over a period's.

Every value is an integer or a `Fraction` until it is printed, and what is printed is ⌊−log₂⌋ in hundredths of a bit,
found by integer arithmetic (`centibits`): a bound printed as 2^-132.03 is at most 2^-132.03.

    uv run python benchmarks/spot_figures/binding.py --card GLM-5.3 --symbols 2^10          # K from the card
    uv run python benchmarks/spot_figures/binding.py --K 2^40.23 --k 12343986 --d 167       # K, k and d given
                            # [--T 2^80] [--row-bytes 65536] [--tables N] [--open-calls C] [--p-ref P]
                            # [--opens N --M N] [--phase-seconds 600] [--machines N] [--target 100] [--json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, factorial, floor

from verity.protocols.reporting import PERIODS

#: the extractor's waits a session
WAITS = 210
#: a collision of SHA-512 is found with probability at most a finder's expected compressions over this
HASH = 2 ** 256
#: a SHA-512 block's bytes, and the least padding a message takes (0x80 and its 16-byte length)
BLOCK, PAD = 128, 17
#: the prover's SHA-512 compressions the bound is quoted at
BUDGET = 2 ** 80
#: an execution phase's seconds, one audit each
PHASE = 600
#: e between two rationals: its series to 1/39!, and that plus 2/40!, more than the series' tail
E_LOW = sum(Fraction(1, factorial(n)) for n in range(40))
E_HIGH = E_LOW + Fraction(2, factorial(40))


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


@dataclass(frozen=True, slots=True)
class Design:
    """What the bound reads of a sampling design and a prover's budget (the module docstring names each field)."""

    K: int
    k: int
    d: int = 1
    T: int = BUDGET
    open_calls: int = 0
    p_ref: Fraction = Fraction(0)
    opens: int = 0
    M: int | None = None

    def __post_init__(self) -> None:
        for name in ("K", "k", "d", "T"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} is a positive integer, not {value!r}")
        for name in ("open_calls", "opens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} is a non-negative integer, not {value!r}")
        object.__setattr__(self, "p_ref", Fraction(self.p_ref))
        if not 0 <= self.p_ref <= 1:
            raise ValueError(f"p_ref is a probability, not {self.p_ref}")
        if self.opens and (not isinstance(self.M, int) or isinstance(self.M, bool) or self.M < 0):
            raise ValueError("a design with opens > 0 states M, the cap on the finder's further runs a gate")

    @property
    def reruns(self) -> int:
        """e_s = 1 + 210·k."""
        return 1 + WAITS * self.k

    @property
    def cost(self) -> int:
        """T·(1 + d·e_s) + d·openCalls: a run, and the d sessions' extractions and opening checks."""
        return self.T * (1 + self.d * self.reruns) + self.d * self.open_calls


def terms(design: Design) -> dict[str, Fraction]:
    """The bound's three terms at one audit, by name."""
    return {"collision": Fraction(2 * design.K * design.cost, HASH),
            "refused": design.p_ref * design.opens,
            "cap": Fraction(design.opens, design.M + 1) if design.opens else Fraction(0)}


def bound(design: Design) -> Fraction:
    """The bound at one audit: the sum of `terms`."""
    return sum(terms(design).values(), Fraction(0))


def audits(seconds: int, phase: int = PHASE, machines: int = 1) -> int:
    """The audits of `seconds` on `machines` machines, one an execution phase of `phase` seconds, a started phase
    counted."""
    if phase < 1 or machines < 1 or seconds < 0:
        raise ValueError("a period has a positive phase and at least one machine")
    return machines * -(-seconds // phase)


def periods(design: Design, phase: int = PHASE, machines: int = 1) -> dict[str, tuple[int, Fraction]]:
    """Each monitoring period's audits and its bound, the sum of its audits' bounds."""
    one = bound(design)
    return {name: (n, n * one) for name, seconds in PERIODS for n in [audits(seconds, phase, machines)]}


def most_gates(design: Design, target: int, audits: int = 1) -> int:
    """The most gates K at which `audits` audits of the design sum to at most 2^-`target`, its other fields kept: 0
    where the terms that do not read K are above it alone."""
    fixed = terms(design)
    room = Fraction(1, 2 ** target) - audits * (fixed["refused"] + fixed["cap"])
    return max(0, floor(room * HASH / (2 * design.cost * audits)))


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


def report(design: Design, phase: int = PHASE, machines: int = 1, target: int = 100) -> dict:
    """The bound at one audit, term by term, and over each monitoring period, with whether each is at most
    2^-`target`."""
    goal = Fraction(1, 2 ** target)

    def cell(p: Fraction, **more) -> dict:
        return {**more, "bound": bits(p), "centibits": centibits(p), "exact": str(p), "meets": p <= goal}

    return {"design": {"K": design.K, "k": design.k, "d": design.d, "T": design.T, "open_calls": design.open_calls,
                       "p_ref": str(design.p_ref), "opens": design.opens, "M": design.M, "reruns": design.reruns},
            "target": target, "audit": cell(bound(design)),
            "terms": {name: cell(p) for name, p in terms(design).items()},
            "phase_seconds": phase, "machines": machines,
            "periods": {name: cell(p, audits=n) for name, (n, p) in periods(design, phase, machines).items()},
            "most_gates": {"audit": most_gates(design, target),
                           **{name: most_gates(design, target, n)
                              for name, (n, _) in periods(design, phase, machines).items()}}}


def render(out: dict) -> str:
    """The report as lines: the design, one audit and its terms, then each monitoring period."""
    d = out["design"]

    def cell(c: dict) -> str:
        return c["bound"] + ("" if c["meets"] else f"   above 2^-{out['target']}")

    lines = [f"K = {d['K']} gates (2^{log2(d['K'])}), T = 2^{log2(d['T'])} compressions, d = {d['d']}, "
             f"k = {d['k']}, e_s = {d['reruns']}",
             f"one audit   {cell(out['audit'])}"]
    lines += [f"  {name:<10}{c['bound']}" for name, c in out["terms"].items()]
    lines.append(f"a monitoring period, an audit every {out['phase_seconds']} s on {out['machines']} machine(s):")
    lines += [f"  {name:<6}{c['audits']:>16} audits   {cell(c)}" for name, c in out["periods"].items()]
    most = ", ".join(f"{f'2^{log2(n)}' if n else 'none'} {'at one' if name == 'audit' else 'over a'} {name}"
                     for name, n in out["most_gates"].items())
    lines.append(f"at most 2^-{out['target']} up to K = {most}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    k = ap.add_mutually_exclusive_group(required=True)
    k.add_argument("--K", type=natural, help="the gates one audit's sample needs (2^40.23 is rounded up)")
    k.add_argument("--card", metavar="MODEL", help="K at one proof block a stratum of the model's workload card")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--k", type=natural, help="a session's accepted reruns a wait")
    q.add_argument("--symbols", type=natural, help="a table's level-0 codeword symbols: k = ⌈4·tables·symbols/e⌉")
    ap.add_argument("--tables", type=natural, default=1, help="the tables of one session, with --symbols (default 1)")
    ap.add_argument("--d", type=natural, default=1, help="the most sessions holding one gate (default 1)")
    ap.add_argument("--T", type=natural, default=BUDGET, help="the prover's SHA-512 calls (default 2^80)")
    ap.add_argument("--row-bytes", type=natural,
                    help="count T in calls on rows of this many bytes, not in compressions")
    ap.add_argument("--open-calls", type=natural, default=0, help="the compressions of a session's opening checks")
    ap.add_argument("--p-ref", type=probability, default=Fraction(0), help="the probability of refused coins")
    ap.add_argument("--opens", type=natural, default=0, help="the gates any run can open")
    ap.add_argument("--M", type=natural, help="the cap on the finder's further runs a gate (with --opens)")
    ap.add_argument("--phase-seconds", type=natural, default=PHASE, help="an execution phase's seconds (default 600)")
    ap.add_argument("--machines", type=natural, default=1, help="the machines audited over a period (default 1)")
    ap.add_argument("--target", type=int, default=100, help="the bits each bound is held to (default 100)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    gates = a.K
    if a.card:
        import cards
        gates = card_gates(cards.load("workload", a.card))
    design = Design(K=gates, k=a.k if a.k else queries(a.tables * a.symbols), d=a.d,
                    T=a.T * (compressions(a.row_bytes) if a.row_bytes is not None else 1),
                    open_calls=a.open_calls, p_ref=a.p_ref, opens=a.opens, M=a.M)
    out = report(design, a.phase_seconds, a.machines, a.target)
    print(json.dumps(out, indent=1) if a.json else render(out))


if __name__ == "__main__":
    main()
