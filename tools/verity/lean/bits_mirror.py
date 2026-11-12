"""The mirrors `verity lean bits` prints a `mirrored` number from: Python restatements of Lean error terms, one function an
error term, each named with the Lean definition it restates, until a Lean row or an evaluator replaces it (`bits.py`).

A mirror is a second copy of a formula, so it is held to the Lean (`tests/test_bits.py`, no Lean build):

- **pins**: each `Mirror` lists the guarantees whose bound it evaluates, with the hash their record in the lock holds
  (`unfolded_hash`, the statement with its definition unfolded once, else `type_hash`), and the definitions it restates
  with the hash the lock's `reads` holds for each one a lock records. A statement's hash names the definitions it reads
  only by name, so a restated error term moves its definition's `reads` hash and not the guarantee's: both are pinned.
  A definition of the proofs tree has no `reads` record, and a guarantee whose statement is still in the proofs tree
  has only a `type_hash`, the name of its statement: those pins say less, and the test's instance check carries the
  rest. A moved pin fails the test, and `bits.py` prints the guarantee with no number until the mirror is read again.
- **instances**: where a Lean theorem proved by `norm_num` evaluates the mirrored expression at numerals, the test reads
  its arguments and right side from the source and holds the mirror's exact value at them under that right side
  (`Point.instance`).

Values are exact rationals (`Fraction`) where the formula has no root or logarithm, else floats. A point's numbers are the
parameters of the formula: their `about` says where each comes from, and one with no source says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import sqrt
from typing import Callable


@dataclass(frozen=True)
class Point:
    """Where a mirror is evaluated: ``args`` (with a budget's, the mirror's arguments), what they are (``about``) and, when a
    Lean theorem bounds the mirrored expression at exactly these arguments, its name (``instance``)."""
    name: str
    about: str
    args: dict
    instance: str | None = None


@dataclass(frozen=True)
class Mirror:
    """One error term: the Lean definition it restates (``lean``), the term as that definition writes it (``term``), the
    guarantees whose bound it is part of ({name: (lock key, hash)}), the definitions it restates ({name: `reads` hash}),
    its budgets ({name: args}) and points, ``value``, the term at a point's and a budget's args together,
    ``params``, the Lean definition's arguments in order where an instance applies it to numerals, and ``per_audit``,
    whether the term bounds one audit, so that a period's audits add it up."""
    lean: str
    term: str
    guarantees: dict[str, tuple[str, str]]
    definitions: dict[str, str]
    budgets: dict[str, dict]
    points: tuple[Point, ...]
    value: Callable[..., Fraction | float] = field(repr=False)
    params: tuple[str, ...] = ()
    per_audit: bool = True


# ---- Claims.Flock.Numbers.hashBound: the by-class first-hit bound's part over 2^256 (the proofs tree) ----

def first_hit_calls(T: Fraction, e: Fraction, Ns: Fraction, Ks: Fraction, Km: Fraction, c: Fraction) -> Fraction:
    """`Claims.Flock.Numbers.firstHitCalls`: `T·(1 + e·N_s + K_s + e·K_m) + (1 + K_s)·c`."""
    return T * (1 + e * Ns + Ks + e * Km) + (1 + Ks) * c


def hash_bound(T: Fraction, e: Fraction, Ns: Fraction, Ks: Fraction, Km: Fraction, c: Fraction) -> Fraction:
    """`Claims.Flock.Numbers.hashBound`: `(2·(T·(1 + e·N_s) + c) + 4·firstHitCalls)/2^256`."""
    return (2 * (T * (1 + e * Ns) + c) + 4 * first_hit_calls(T, e, Ns, Ks, Km, c)) / Fraction(2) ** 256


# ---- Claims.SamplingGeneric.Compiled.εregAt: the committed form's binding terms at a budget ----

def bconf_cost(T: float, n_sess: float, es: float, n_pu: float, open_calls: float) -> float:
    """`Compiled.bconfCost`: `c_T = 2T·(1 + n_sess·e_s) + 2·n_PU·openCalls`."""
    return 2 * T * (1 + n_sess * es) + 2 * (n_pu * open_calls)


def eps_reg_at_binding(T: float, nG: float, n_sess: float, es: float, n_pu: float, open_calls: float) -> float:
    """`Compiled.εregAt`'s binding terms: `N·√(c_T/2^256) + c_T/2^256`, `N` the program's gates (`nG`)."""
    c = bconf_cost(T, n_sess, es, n_pu, open_calls) / 2.0 ** 256
    return nG * sqrt(c) + c


# ---- Claims.SamplingGeneric.Compiled.KSAt: the extractor term ecrBound(B_ext at T, 2·T·work) ----

def work(k_out: Fraction, k_in: Fraction, n_sess: Fraction, es: Fraction, n_pu: Fraction,
         open_calls: Fraction) -> Fraction:
    """`Compiled.work`: `1 + k_out + k_out·k_in·(1 + n_sess·e_s) + k_out·k_in·n_PU·openCalls`."""
    return 1 + k_out + k_out * k_in * (1 + n_sess * es) + k_out * k_in * n_pu * open_calls


def ks_at_extractor(T: Fraction, n_pu: Fraction, n_sess: Fraction, es: Fraction, open_calls: Fraction) -> Fraction:
    """`Claims.ecrBound` at `Sampling.KSAt`'s `B_ext` (`2·T·work`), where `ExpectedCR` holds at it: `2T·work/2^256`, at
    `k_out = k_in = 2^slackExp·(2·n_PU + 1)` (`Compiled.kOut`, `kIn`, with one proof block a wait's unit)."""
    k = Fraction(2) ** 10 * (2 * n_pu + 1)
    return 2 * T * work(k, k, n_sess, es, n_pu, open_calls) / Fraction(2) ** 256


# ---- Claims.KS: κ + √(γ·Pr[B]) at Flock's deployed zero-knowledge record ----

def ks(T: float, kappa: float, gamma: float, finder: str) -> float:
    """`Claims.KS`'s right side `κ + √(γ·Pr[B])`, with `Pr[B]` at a finder of two runs of `T` calls read under
    `StrictBirthday` (`(2T)²/2^513`) or `ExpectedCR` (`2T/2^256`)."""
    pr_b = (2 * T) ** 2 / 2.0 ** 513 if finder == "StrictBirthday" else 2 * T / 2.0 ** 256
    return kappa + sqrt(gamma * pr_b)


# ---- Pouw.PearlC.δPearlC: Pearl-C's noise term, per challenge ----

def delta_pearl_c(q: Fraction, N: Fraction) -> Fraction:
    """`Pouw.PearlC.δPearlC`: `(q + N)/2^128`, ``N`` a statement's rows times its row width."""
    return (q + N) / Fraction(2) ** 128


# ---- Pouw.Guarantees.PeriodBindingPerProver: a verification phase's binding per prover ----

def period_binding(T: float, L: float, kappa: float) -> float:
    """`PeriodBindingPerProver`'s bound `|L|·√(q/2^256 + κ)`, ``L`` the period's opened positions and ``q = T``."""
    return L * sqrt(T / 2.0 ** 256 + kappa)


#: SPOT's three sizes: the program's gates `N` and its proof blocks `n_PU`, with `k = 1507` rounds at one table, `e_s =
#: 1 + 210·k` and `openCalls = 2^25`. No card declares them: `N` counts MACs where `εregAt` counts gates, `n_PU` at the
#: floor is a workload card's `macs` field, and `openCalls`'s tree depth is a guess.
SPOT_SIZES = {"floor": (43.73, 57344.0), "B": (47.9, 2.0 ** 24.2), "A": (58.8, 2.0 ** 35.1)}
SPOT_K = 1507
SPOT_ES = 1 + 210 * SPOT_K
SPOT_OPEN_CALLS = 2.0 ** 25
CALLS = {"T = 2^64 calls a run": {"T": 2 ** 64}, "T = 2^80 calls a run": {"T": 2 ** 80}}
SPOT_UNSOURCED = "SPOT's {name} (N = 2^{n}, n_PU = {npu:g}, k = 1507, e_s = 1 + 210·k, openCalls = 2^25; no card declares them)"


def _spot_points(sessions: tuple[str, ...]) -> tuple[Point, ...]:
    out = []
    for name, (n_lg, npu) in SPOT_SIZES.items():
        for s in sessions:
            n_sess = 1.0 if s == "1 session" else npu
            out.append(Point(f"spot/{name}, {s}", SPOT_UNSOURCED.format(name=name, n=n_lg, npu=npu) + f", {s} an audit",
                             {"nG": 2.0 ** n_lg, "n_pu": npu, "n_sess": n_sess, "es": float(SPOT_ES),
                              "open_calls": SPOT_OPEN_CALLS}))
    return tuple(out)


def _exact(args: dict) -> dict:
    return {k: Fraction(v) for k, v in args.items()}


MIRRORS: tuple[Mirror, ...] = (
    Mirror(
        lean="Claims.Flock.Numbers.hashBound",
        term="hashBound(T, e, N_s, K_s, K_m, c) = (2·(T·(1 + e·N_s) + c) + 4·firstHitCalls)/2^256, read twice in "
             "SpotFlock.εoneWaitAt",
        guarantees={"Flock.SecurityProofs.SpotOneWaitTwo":
                    ("type_hash", "8acc03ad45f091d7827560e89cfbb9c551d450fc3fea21d4ad34396eb914d372")},
        definitions={},
        budgets={"T = 513·2^80 calls a run, a call charged as 513 compressions": {"T": 513 * 2 ** 80}},
        points=(Point("one table, one wait a session",
                      "sessions of one table (e ≤ 1 + 12343986), N_s ≤ 2^26, K_s, K_m ≤ 2^38, n_PU·openCalls ≤ 513·2^80",
                      {"e": 12343987, "Ns": 2 ** 26, "Ks": 2 ** 38, "Km": 2 ** 38, "c": 513 * 2 ** 80},
                      instance="Claims.Flock.Numbers.hashBound_oneWait_513"),
                Point("one table, one wait a session, K up to 2^40",
                      "as above at N_s ≤ 2^30 and K_s, K_m ≤ 2^40",
                      {"e": 12343987, "Ns": 2 ** 30, "Ks": 2 ** 40, "Km": 2 ** 40, "c": 513 * 2 ** 80},
                      instance="Claims.Flock.Numbers.hashBound_oneWait_513_k40")),
        value=lambda **a: hash_bound(**_exact(a)),
        params=("T", "e", "Ns", "Ks", "Km", "c"),
    ),
    Mirror(
        lean="Claims.SamplingGeneric.Compiled.εregAt",
        term="N·√(c_T/2^256) + c_T/2^256, c_T = 2T·(1 + n_sess·e_s) + 2·n_PU·openCalls (εregAt's binding terms)",
        guarantees={
            "Claims.SamplingGeneric.compilerKSReg":
                ("unfolded_hash", "fe205dc2f44b0a7d088bf33478098cd7bd13b7f888661ee7e4b31369f963c2f7"),
            "Claims.SamplingGeneric.compilerKSRegAt":
                ("unfolded_hash", "e8a286c2f3b096ef084393f165a712e95bb1c252594b184314e8b4c6552bf7e0"),
        },
        definitions={
            "Claims.SamplingGeneric.Compiled.εreg": "319db40121955316d771a9abeb026ceab37238c8205b1864e4324da841c18ed3",
            "Claims.SamplingGeneric.Compiled.εregAt": "9ae3d95b2fad4ad83a362a34d9b3d72e6a06cb8cb184aaf180628a83a268b02e",
            "Claims.SamplingGeneric.Compiled.bconfCost":
                "7b2c6f26cf47813e21fd5428dff2f11399192cb548dc6b01eee88dea18190a9b",
        },
        budgets=CALLS,
        points=_spot_points(("1 session", "n_PU sessions")),
        value=lambda **a: eps_reg_at_binding(**{k: float(v) for k, v in a.items()}),
    ),
    Mirror(
        lean="Claims.SamplingGeneric.Compiled.KSAt",
        term="2T·work/2^256, work = 1 + k_out + k_out·k_in·(1 + n_sess·e_s) + k_out·k_in·n_PU·openCalls (KSAt's "
             "extractor term)",
        guarantees={
            "Claims.SamplingGeneric.compilerKSAt":
                ("unfolded_hash", "92e403cec8f991118e93542458f9b0b9fe731ba1be4047fe6db27b0a54892a52"),
            "Claims.SamplingGeneric.compilerKSAtTwo":
                ("unfolded_hash", "9aecf6855d726770a3c07a074d98db78539f7117af4344f3523b6c8dbbd3b846"),
        },
        definitions={
            "Claims.SamplingGeneric.Compiled.KSAt": "d03a6d5adb65c1d7c168a76d67cc189fe3941bdf3d95f128cbc9aaf5a271688a",
            "Claims.SamplingGeneric.Compiled.work": "53bc0d301efa9646ca0140dcac25c7fefe63de2d2cc189d3f35e21a2882b4759",
            "Claims.Sampling.KSAt": "04fe1e51dbb3cf1622dc096a134663dbc77981153b9da5c8486179f783056d30",
            "Claims.ecrBound": "5a63f16b03af8221c63e8386dc91653577afde66001257db754f76c01e644288",
        },
        budgets=CALLS,
        points=_spot_points(("n_PU sessions",)),
        value=lambda **a: ks_at_extractor(**_exact({k: a[k] for k in ("T", "n_pu", "n_sess", "es", "open_calls")})),
    ),
    Mirror(
        lean="Claims.KS",
        term="κ + √(γ·Pr[B]), Pr[B] at a finder of two runs of T calls (Claims.KS at FlockDeploy.deployZk)",
        guarantees={"Flock.SecurityProofs.MaskedKS":
                    ("unfolded_hash", "7da7b31fc745cf3916021e210ad7423f0aa879a96bf961a4235b4b451f50fc87")},
        definitions={
            "Claims.KS": "323a73dbf6bb956ecf567a60d3e4471fd8273e2c13abbb0b2730b1e51494202b",
            "Claims.Flock.KS": "df44b3647fe39e67f9008c05b7cc220bf9f4797c488eaa1a986f3bf373b9599f",
        },
        budgets=CALLS,
        points=(
            Point("one table, StrictBirthday", "κ ≤ 2^-202.98 and γ ≤ 2^51 at one table (the deployed record's)",
                  {"kappa": 2.0 ** -202.98, "gamma": 2.0 ** 51, "finder": "StrictBirthday"}),
            Point("64 tables, StrictBirthday", "κ ≤ 2^-196.98 at 64 tables; γ = 2^62.1, no Lean bound",
                  {"kappa": 2.0 ** -196.98, "gamma": 2.0 ** 62.1, "finder": "StrictBirthday"}),
            Point("one table, ExpectedCR", "κ ≤ 2^-202.98 and γ ≤ 2^51 at one table, Pr[B] read under ExpectedCR",
                  {"kappa": 2.0 ** -202.98, "gamma": 2.0 ** 51, "finder": "ExpectedCR"}),
        ),
        value=lambda T, kappa, gamma, finder: ks(float(T), kappa, gamma, finder),
    ),
    Mirror(
        lean="Pouw.PearlC.δPearlC",
        term="δPearlC q N = (q + N)/2^128, q = T and N the rows times the row width (each statement's δ, a term of "
             "its gap)",
        guarantees={
            "Pouw.PearlC.pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192":
                ("unfolded_hash", "507fb03572a319ad07c133042c9fd96c2e7b9e6b40c419c30ed2b10da33f5235"),
            "Pouw.PearlC.pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCap1000_8192":
                ("unfolded_hash", "2f5e38246498aa870adda0a71a77d943d4eead4922e041513dc71c7e890ff65c"),
            "Pouw.PearlC.pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192":
                ("unfolded_hash", "18caf59f47dc6f09d5227c2bb6ef405bf3268c2af7cdf43b360b2af697561994"),
            "Pouw.PearlC.pearlCSampledH100v1RowSeedCodeRowCap100_6144x256":
                ("unfolded_hash", "2fb81f5ebfd295195002ead7457dfe7e85bd2f12d987d64229054c30154beaba"),
            "Pouw.PearlC.pearlCSampledH100v1RowSeedCodeRowCap100_6144x256Tol":
                ("unfolded_hash", "4eb3183f3916e1971f474d4628543951edd35fd8484c238e4e75c83667b37381"),
            "Pouw.PearlC.pearlCSampledH100v1RowSeedRowCap100_6144x256":
                ("unfolded_hash", "45768ac8198706181626edcfbe759c283d0684c24f5823c7a41384e9f14bd52f"),
            "Pouw.PearlC.pearlCSampledH100v1RowSeedRowCap100_6144x256Tol":
                ("unfolded_hash", "4f1a50ddaaf1f6cacca8e669d640336fdf1ee8c116859db708fa18086b14f9ef"),
            "Pouw.PearlC.pearlCSampledSm120v1RowSeed_8192":
                ("unfolded_hash", "1363de33bf93059ce2e8c560c0f7d9fce3681306c374da6c056fb2d1caf9500d"),
            "Pouw.PearlC.pearlCSampledSm120v2RowSeedCap1000_8192":
                ("unfolded_hash", "6597197cd72d99a894102682fe0d2bb74e13a8e65d45bb37bb750bec2fc9c17f"),
        },
        definitions={"Pouw.PearlC.δPearlC": "d59867555ea584f9dc1ae4ba8f6247384c1b116b37fbe1808efba06bbef504a8"},
        budgets=CALLS,
        points=(Point("rows·width = 2^30", "N = 2^30 (an illustrative size; no card declares it)", {"N": 2 ** 30}),
                Point("rows = 2^64, width 8192", "N = 2^64·8192, the most the sm120 statements' domain admits "
                      "(`rowsOf L ≤ 2^64`)", {"N": 2 ** 64 * 8192})),
        value=lambda T, N: delta_pearl_c(Fraction(T), Fraction(N)),
        per_audit=False,
    ),
    Mirror(
        lean="Pouw.Guarantees.PeriodBindingPerProver",
        term="|L|·√(q/2^256 + κ) at κ = 0, its least, q = T and L the period's opened positions (a verification "
             "phase's, per prover)",
        guarantees={"Pouw.SecurityProofs.PeriodBindingPerProver":
                    ("unfolded_hash", "64f30f40c6ae82fca10844cdd2b2d0782a38949e2b06a7d744a9c1710d5ada70")},
        definitions={"Pouw.Guarantees.PeriodBindingPerProver":
                     "a7149770af8e66c7ae22a6189d06919d9b319a9b6100925dcd5500d95d353a43"},
        budgets=CALLS,
        points=tuple(Point(f"|L| = 2^{n}, κ = 0", f"2^{n} opened positions (an illustrative size; no card declares "
                           f"it), κ = 0", {"L": 2 ** n, "kappa": 0.0}) for n in (20, 30)),
        value=lambda T, L, kappa: period_binding(float(T), float(L), kappa),
        per_audit=False,
    ),
)
