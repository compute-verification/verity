"""SPOT's cryptographic error in the remote application, its part of the composite `verity/remote/v1`'s soundness
error: the chance that, for a prover making at most T SHA-512 calls, SPOT's cryptography is fooled, per window and
pooled over the monitoring periods, over one datacenter and over the world's compute:

    uv run python benchmarks/remote_error/run.py [--json] [--blocks N]

A security figure, not a measurement: `DESCRIPTION`. Lean doesn't instantiate the remote guarantee at numeric
parameters (its one instance, the honest witness, has every error ⊤), so no lock guarantee states any term's value. The
formula is the guarantee's, and its value is evaluated here in exact rationals, each irrational rounded on the safe side.

The guarantee is `GUARANTEE`, read from its lock with its open premises and named assumptions. Soundness at the composite
is the union bound over its subprotocols' errors (`Verity.Vocab.error`): each one's δ at the prover's calls t, plus
ε_forge at its forger's budget, plus its loss. The total holds the terms of SPOT's δ (`Verity.Sampled.Live.delta`) that
cryptography, not sampling, bounds, and SPOT's forgery term. PoUW's and the inference-only policy's errors are open
(`OPEN`), and SPOT's sampling term is not this figure.

Each parameter says its kind (`KINDS`) and where its value comes from, and each term's basis follows from the kinds of the
parameters it reads (`basis`): ``formula proved, value evaluated here`` where every one is determined by the remote
application or chosen freely, ``assumed`` where one is an assumption or a value the tree doesn't determine. The free
parameters are chosen to minimize the total where the bound has a minimum (k, and ρ with it), and `sensitivity` shows the
total at two other choices of each.

Pooling: a period holds one play a machine a window, plays = machines · period / window. A term a + b·t + c·t² a play
sums over the plays to at most plays·a + b·T + c·T², with T the fleet's SHA-512 calls over the period (Σ t² ≤ (Σ t)²):
its GPUs at the figure's one adversary parameter, `hash_rate` calls a second each. The world's compute (the adversary
resource model, `benchmarks/census/adversary.json`'s reference) counts as machines of the datacenter's size. Each total's
share is per machine-window audited: the census states no AND gates a machine-window. Under a research run it writes
`METRICS` into the run's result.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench import kit, settings

from verity.protocol import MONITORING_PERIODS

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "benchmarks" / "census"
LOCKS = (ROOT / "security" / "lean-audit.json", ROOT / "security" / "Properties" / "lean-audit.json")
#: the remote application's configuration, whose sampling design the sample follows
APPLICATION = ROOT / "applications" / "remote.toml"
#: the measured SHA-512 rate the adversary's rate is stated against
HASH_CARD = ROOT / "benchmarks" / "spot_figures" / "cards" / "commitment.all.json"
#: the lock guarantee whose δ these terms are, and the composite's union bound it instantiates
GUARANTEE = "Verity.Sampled.remote_soundness_deployedReady"
COMPOSITE = "Verity.remote_soundness_of_subprotocols"
DESCRIPTION = ("bound's formula proved in Lean under four open premises; value evaluated in Python; "
               "SPOT's part only")
#: the composite's errors the total leaves out, each with the Lean premise or target it rests on
OPEN = ("pouw (Pearl+): open, not in the total (Verity.Vocab.PouwSound, an open premise of the guarantee; its error is "
        "the base vocabulary's δ, ε_forge at its forger's budget and loss, none of which Lean gives a value)",
        "inference-only policy: open, not in the total (Verity.Vocab.InferenceOnlySound, proved by "
        "Verity.Vocab.inferenceOnlySound with an empty event; its error is the base vocabulary's δ and ε_forge at its "
        "forger's budget, none of which Lean gives a value)")
#: what `ecr/sha-512` is taken at in the binding term, and why M and Rw may be as large as the bound likes
ECR_NOTE = ("The premise `ecr/sha-512` is taken at a finder that plays up to M extraction trials and Rw waits, which nobody "
            "could run to its end; the premise bounds it by its expected SHA-512 evaluations over 2^256 "
            "(`Assumptions.SHA512CRExpected`), and that expected cost does not read M or Rw.")
#: the metrics `metrics.toml` declares: the total over each fleet and period, and its share a machine-window
METRICS = {"soundness/remote-spot-crypto-error-log2": "total", "soundness/remote-spot-crypto-error-share-log2": "share"}
#: the metrics' fleet and period each id names, written last, and the one the free parameters are chosen at
HEADLINE = ("world", "month")
FORMAT = "remote-spot-crypto-error/v2"
KEPT = "out/remote-error.json"

SQRT_BITS = 64
#: e from below, so that 1/(eM), k/(e·Rw) and the rate 2^logLen₀/(e·k) that ρ must exceed are bounded above
E_LOW = Fraction(2718281, 10 ** 6)
#: a SHA-512 compression's block, in bytes
SHA512_BLOCK = 128

#: what a parameter is: what decides its value
KINDS = {
    "a": "determined by the remote application's parameters, computed from the file it names",
    "a, upper bound": "determined by the remote application's parameters; an upper bound computed from the files it names",
    "a, not derived": "determined by the remote application's parameters, which the tree doesn't hold; a stated value",
    "b": "a free parameter of the bound, which the theorem quantifies over",
    "c": "an assumption about the adversary",
    "assumption": "a named assumption at a stated level",
}
#: the kinds a term may read and still have its value follow from the proved formula
DETERMINED = ("a", "a, upper bound", "b")
BASES = ("proved at these parameters", "formula proved, value evaluated here", "assumed", "open", "excluded")

FLOCK = "verity/protocols/proofs/flock/verifier/Flock"
S = "security/Proofs/Properties/Verity/Sampled.lean"
INSTANCE = "security/Proofs/Properties/Verity/Sampled/Instance.lean"
STRICTZ = "security/Proofs/Flock/Soundness/Discharge/ZkLink/StrictZ.lean"
GAME = "security/Properties/Verity/Game.lean"
RATE = "security/Proofs/Flock/Soundness/Discharge/ZkBind/Rate.lean"
TABLE = "security/Proofs/Flock/Claims/NumbersTable.lean"
INNER = "security/Proofs/Flock/Soundness/Discharge/ZkExec/Statements.lean"
#: Flock's composer's sizes, read from the file rather than imported: importing it loads the circuits
COMPOSER = "verity/commitments/periods/interface.py"


@dataclass(frozen=True)
class Param:
    value: Fraction
    kind: str
    source: str


def _text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _nat(expr: str) -> int:
    """A Lean natural literal as the files write one: ``192`` or ``2 ^ 23``."""
    a, _, b = expr.partition("^")
    return int(a) ** int(b) if b else int(a)


def _find(rel: str, pattern: str) -> re.Match[str]:
    m = re.search(pattern, _text(rel), re.M)
    if m is None:
        raise SystemExit(f"{rel}: nothing matches {pattern!r}: its Lean changed, so update what {Path(__file__).name} reads")
    return m


def _block(rel: str, start: str) -> str:
    """The declaration starting with `start` in `rel`, to its first blank line."""
    text = _text(rel)
    i = text.find(start)
    if i < 0:
        raise SystemExit(f"{rel}: no {start!r}: its Lean changed, so update what {Path(__file__).name} reads")
    return text[i:text.find("\n\n", i)]


def schedule() -> dict[str, int]:
    """What the deployed `--zk` schedule allows, read from the Lean verifier: the largest m `Flock.Zk.queries0` accepts and
    its largest q₀, fast100's most levels and largest k₀ at those m, level 0's largest log length, the inner proof's
    padding and rate, and the coin tree's rounds."""
    q0 = _block(f"{FLOCK}/Zk.lean", "def queries0")
    m_max = max(int(x) for x in re.findall(r"m ≤ (\d+) then pure", q0))
    queries = max(int(x) for x in re.findall(r"pure \(?(?:if wide then )?(\d+)", q0) + re.findall(r"else (\d+)\)", q0))
    sched = f"{FLOCK}/Schedule.lean"
    rule = _find(sched, r"let levels := (.+)$").group(1)
    steps = [(int(a), int(b)) for a, b in re.findall(r"m ≤ (\d+) then (\d+)", rule)]
    last = int(re.search(r"else (\d+)$", rule).group(1))
    levels = max(next((v for top, v in steps if m <= top), last) for m in range(22, m_max + 1))
    k0 = max(int(x) for x in re.findall(r"(?:then|else) (\d+)", _find(sched, r"let k0 := (.+)$").group(1)))
    return {"m_max": m_max, "queries": queries, "levels": levels, "k0": k0,
            "log_len": int(_find(RATE, r"theorem zkLogLen\w+ : \w+ (\d+)").group(1)),
            "k_pad": int(_find(f"{FLOCK}/Zk.lean", r"^def K_PAD : Nat := (\d+)").group(1)),
            "inner_log_inv_rate": int(_find(f"{FLOCK}/Zk.lean", r"^def INNER_LOG_INV_RATE : Nat := (\d+)").group(1)),
            "rounds_log": int(re.search(r", (\d+), \d+⟩", _block(f"{FLOCK}/Zk.lean", "def coinSpec")).group(1))}


def composer() -> tuple[int, int]:
    """The hm96 slot's log size and the most rows a proof block may have, as `COMPOSER` defines them."""
    m = _find(COMPOSER, r"^K_MAX, K_MIN, SLOT_LOG, \w+, \w+, \w+ = (\d+), \d+, (\d+),")
    k_max, slot_log = int(m.group(1)), int(m.group(2))
    _find(COMPOSER, r"^MAX_ROWS = 1 << \(K_MAX - SLOT_LOG - 1\)$")
    return slot_log, 1 << (k_max - slot_log - 1)


def inner_n(m: int, k_log: int, k0: int, k_pad: int, inner_log_inv_rate: int) -> int:
    """`innerN`: the inner proof's N at `Flock.Zk.PadLayout.new m kLog k0`, nextPow2(len + K_PAD) · 2^INNER_LOG_INV_RATE,
    with `PadLayout.len` = 204 + 2·nMlv + 2·nLc + 4·k0 at nMlv = m − 6, nLc = kLog − 6 and nOod0 = 1."""
    length = 204 + 2 * (m - 6) + 2 * (k_log - 6) + 4 * k0
    return 1 << (length + k_pad - 1).bit_length() << inner_log_inv_rate


def design() -> tuple[int, int, int]:
    """The remote application's sampling design `rate:NUM/DEN[:T]`, as (NUM, DEN, T)."""
    text = tomllib.loads(APPLICATION.read_text(encoding="utf-8"))["service"]["law"]
    m = re.fullmatch(r"rate:(\d+)/(\d+)(?::(\d+))?", text)
    if m is None:
        raise SystemExit(f"{APPLICATION.relative_to(ROOT)}: sampling design {text!r} is no rate design, which this figure reads")
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)


def card_rate_log2() -> float:
    """The commitment card's SHA-512 rate on one GPU, in compressions a second: its bytes a second over a block's 128."""
    value = json.loads(HASH_CARD.read_text(encoding="utf-8"))["value"]
    return math.log2(value["hash_rate"] / SHA512_BLOCK)


def params(blocks: int | None = None) -> dict[str, Param]:
    """Every parameter the terms read, with its kind and source; k and ρ at the headline's minimum (`optimize_k`)."""
    s = schedule()
    num, den, tables = design()
    slot_log, max_rows = composer()
    n_max = max(inner_n(m, m, k0, s["k_pad"], s["inner_log_inv_rate"]) for m in range(25, s["m_max"] + 1)
                for k0 in range(s["k0"] + 1))
    slots = 2 ** (s["m_max"] - slot_log)
    positions, pos_src = slots, (
        f"Σ_p ratio_p, the commit strings a window's sample reads in expectation, at most the hm96 rows its one --zk table "
        f"opens: the session proves one table (Verity.Sampled.LiveSpec.count), of at most 2^{s['m_max']} bits "
        f"(Flock.Zk.queries0's largest m, {FLOCK}/Zk.lean), and each row a sampled proof block reads or writes takes an "
        f"hm96 slot of 2^SLOT_LOG = 2^{slot_log} bits in its Flock block ({COMPOSER})")
    if blocks is not None:
        k = tables * -(-(num * blocks) // (den * tables))
        positions = min(slots, k * max_rows)
        pos_src += (f"; with {blocks} proof blocks registered, the sample is ceil({num}·N/{den}) = {k} of them "
                    f"({APPLICATION.relative_to(ROOT)}'s sampling design, Spot.Rate.count), each reading at most MAX_ROWS = {max_rows} rows "
                    f"({COMPOSER})")
    else:
        pos_src += (f"; the exact count is K · (rows a sampled block reads), K = ceil({num}·N/{den}) by "
                    f"{APPLICATION.relative_to(ROOT)}'s sampling design, and N, the class circuit's proof blocks, is a lowering's "
                    "output the tree doesn't hold (--blocks N takes it)")
    stated = ("a stated value: the finders the guarantee's constructions build must run within it (Live's qF, qS, tL and "
              f"qM, {S}), so the verifier's code fixes it, but the tree holds no count of it, and `sensitivity` shows the "
              "total at larger ones")
    p = {
        "positions": Param(Fraction(positions), "a, upper bound", pos_src),
        "N": Param(Fraction(n_max), "a", f"innerN (the inner proof's N, {INNER}) at Flock.Zk.PadLayout, its largest at "
                   f"m ≤ {s['m_max']}, kLog ≤ m and k₀ ≤ {s['k0']}"),
        "levels": Param(Fraction(s["levels"]), "a", f"Flock.fast100's most levels at m ≤ {s['m_max']} ({FLOCK}/Schedule.lean)"),
        "log_len": Param(Fraction(s["log_len"]), "a", f"level 0's log length under Flock.Zk.schedule, which no later level "
                         f"exceeds, proved at each m ({RATE})"),
        "queries": Param(Fraction(s["queries"]), "a", f"Flock.Zk.queries0's largest q₀ ({FLOCK}/Zk.lean); later levels "
                         "query fewer"),
        "rounds": Param(Fraction(2 * 2 ** s["rounds_log"] + 1), "a", f"Verity.Sampled.Live.rounds, 2·2^roundsLog + 1 at "
                        f"Flock.Zk.coinSpec's roundsLog ({FLOCK}/Zk.lean)"),
        "max_leaves": Param(Fraction(_nat(_find(GAME, r"^def maxTreeLines : ℕ := (.+)$").group(1))
                                     * _nat(_find(GAME, r"^def maxTreeCoins : ℕ := (.+)$").group(1))), "a",
                            f"Verity.maxLeaves = maxTreeLines · maxTreeCoins ({GAME})"),
        "calls_mul": Param(Fraction(int(_find(INSTANCE, r"def callsMulOf[\s\S]*?\| \.sampled => (\d+)").group(1))), "a",
                           f"Verity.Sampled.Instance.callsMulOf .sampled, SPOT's finder budget a call ({INSTANCE})"),
        "v": Param(Fraction(2 ** 40), "a, not derived", "SHA-512 calls of one opening's verification (SessionBound.v): "
                   + stated),
        "w": Param(Fraction(2 ** 40), "a, not derived", "SHA-512 calls of one acceptance check (SessionBound.w): " + stated),
        "h": Param(Fraction(2 ** 40), "a, not derived", "SHA-512 calls hashing one message (SessionBound.h): " + stated),
        "roots_calls": Param(Fraction(2 ** 40), "a, not derived", "SHA-512 calls recomputing the registration's roots "
                             "from the public file (Flock.Bridge.rootsCostL): " + stated),
        "M": Param(Fraction(2 ** 256), "b", "the link finder's extraction trials (SessionBound.M, any M ≥ 1); the bound "
                   "falls as M grows and has no minimum. " + ECR_NOTE),
        "Rw": Param(Fraction(2 ** 256), "b", "the link finder's rewinding trials (SessionBound.Rw, any Rw ≥ 1); the bound "
                    "falls as Rw grows and has no minimum. " + ECR_NOTE),
        "k": Param(Fraction(1), "b", "the extraction budget (SessionBound.k, any k ≥ 1 with level 0's rate within ρ); "
                   "chosen where the headline's total is least"),
        "rho": Param(Fraction(1), "b", "the rate margin (SessionBound.ρ < 1), the least the rate condition "
                     "2^logLen₀/(e·k) ≤ ρ admits at k"),
        "table_bound": Param(Fraction(1, 2 ** int(_find(TABLE, r"theorem εtable_zk[\s\S]*?\(-\((\d+) : ℝ\)\)").group(1))),
                             "assumption", f"Claims.Flock.Numbers.εtable_zk ({TABLE}): a lemma bounding a --zk table's "
                             "bound where the table meets TabZK.M1; that the deployed table meets it is not in the guarantee"),
        "forge_rate": Param(Fraction(1, 2 ** 256), "assumption", "euf-cma/lms-sha512-n64 (Hash.Assumptions.LmsUnforgeable, "
                            "which takes ε_forge as a parameter) at the level ε_forge(q, Q) = q/2^256"),
        "hash_rate": Param(Fraction(2 ** 33), "c", f"SHA-512 compressions a second a GPU, the figure's one adversary "
                           f"parameter: at least the measured rate the commitment card states "
                           f"({HASH_CARD.relative_to(ROOT)}, its bytes a second over a {SHA512_BLOCK}-byte block)"),
    }
    return optimize_k(p)


def with_k(p: dict[str, Param], k: int) -> dict[str, Param]:
    """`p` at extraction budget k and the least ρ the rate condition admits there, 2^logLen₀/(e·k) bounded above."""
    rho = Fraction(2 ** int(p["log_len"].value)) / (E_LOW * k)
    return {**p, "k": replace(p["k"], value=Fraction(k)), "rho": replace(p["rho"], value=rho)}


def optimize_k(p: dict[str, Param], fl: dict[str, dict[str, int]] | None = None) -> dict[str, Param]:
    """`p` at the k, over a grid of 1/64 of a bit, whose headline total is least."""
    fl = fl or fleets()
    low = 2 ** int(p["log_len"].value) / E_LOW
    ks = sorted({int(2 ** (j / 64)) for j in range(16 * 64, 32 * 64)} - {0})
    best = min((k for k in ks if k > low), key=lambda k: headline_total(with_k(p, k), fl))
    return with_k(p, best)


def sqrt_up(x: Fraction) -> Fraction:
    """A rational at least √x, within 2^-SQRT_BITS of it relatively."""
    n, d = x.numerator, x.denominator
    return Fraction(math.isqrt(n * d * 4 ** SQRT_BITS) + 1, d * 2 ** SQRT_BITS)


def log2(x: Fraction) -> float:
    return math.log2(x.numerator) - math.log2(x.denominator)


@dataclass(frozen=True)
class Term:
    """One summand of the error a play, a + b·t + c·t² at t SHA-512 calls, or None where the total leaves it out."""
    name: str
    what: str
    lean: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    coeffs: tuple[Fraction, Fraction, Fraction] | None = None
    left_out: str = field(default="")
    note: str = ""

    def at(self, plays: int, calls: int) -> Fraction | None:
        if self.coeffs is None:
            return None
        a, b, c = self.coeffs
        return plays * a + b * calls + c * calls * calls


def basis(t: Term, p: dict[str, Param]) -> str:
    """A term's basis: none is proved at these parameters, since no lock guarantee states a term's value; one whose every
    parameter is determined or free has its formula proved and its value evaluated here; any other is assumed."""
    if t.coeffs is None:
        return t.left_out
    return BASES[1] if all(p[x].kind in DETERMINED for x in t.params) else BASES[2]


def terms(p: dict[str, Param]) -> list[Term]:
    """SPOT's cryptographic terms of `Verity.Sampled.Live.delta`, each part a stated value reaches apart, and its forgery
    term, then the terms the total leaves out, at the parameters `p`."""
    v = {k: x.value for k, x in p.items()}
    sq = sqrt_up(Fraction(1, 2 ** 513))
    q513, q256, zero = Fraction(1, 2 ** 513), Fraction(1, 2 ** 256), Fraction(0)
    rate = 1 / (1 - v["rho"])
    s = 2 * v["levels"] * sqrt_up(2 ** v["log_len"] * v["queries"]) + 2 * v["N"] + 2 + sqrt_up(v["k"])
    bind = v["positions"] / (1 - v["rho"])
    ks = (("FlockSoundness.Discharge.ZkLink.ksAvgStrictZ", STRICTZ), ("FlockSoundness.Discharge.ZkLink.ksStrictZ", STRICTZ),
          ("FlockSoundness.Discharge.ZkLink.rateZ", "security/Proofs/Flock/Soundness/Discharge/ZkLink/Compiled.lean"))
    link = (("Verity.Sampled.Live.sessionDelta", S), ("Verity.Sampled.Live.tL", S),
            ("FlockSoundness.Discharge.ZkLink.LinkCRZC", "security/Proofs/Flock/Soundness/Discharge/ZkLink/CompiledLink.lean"))
    two = (("Verity.Sampled.Live.twoRun", S), ("Verity.Sampled.Live.rounds", S), ("Verity.Sampled.Live.qM", S))
    return [
        Term("table_iop", "the --zk table's bound, over 1 − ρ", (*ks, ("Claims.Flock.Numbers.εtable_zk", TABLE),
             ("TabZK.M1", "security/Proofs/Flock/Soundness/Discharge/ZkSession/Numbers.lean")),
             params=("table_bound", "rho"), coeffs=(rate * v["table_bound"], zero, zero)),
        Term("table_collisions", "the table's rewinding and self-clash finders at the prover's calls: its levels', pads', "
             "coins' and k's √(·q_F²/2^513) terms at q_F's 2t and q_S²/2^513's t², over 1 − ρ", ks, ("cr/sha-512",),
             ("levels", "log_len", "queries", "N", "k", "rho"), (zero, rate * 2 * s * sq, rate * q513)),
        Term("table_collisions_v", "the same finders' share from the verifier's own calls: q_F's 2v and q_S²'s 2vt + v²", ks,
             ("cr/sha-512",), ("levels", "log_len", "queries", "N", "k", "rho", "v"),
             (rate * (2 * s * sq * v["v"] + v["v"] ** 2 * q513), rate * 2 * v["v"] * q513, zero)),
        Term("binding", "the link finders and extraction over the commit strings a sample reads: "
             "Σ_p ratio_p · (2t(1 + k)/2^256 + 1/(eM) + k/(e·Rw)) / (1 − ρ)", link, ("ecr/sha-512",),
             ("positions", "rho", "k", "M", "Rw"),
             (bind * (1 / (E_LOW * v["M"]) + v["k"] / (E_LOW * v["Rw"])), bind * 2 * (1 + v["k"]) * q256, zero),
             note=ECR_NOTE),
        Term("binding_w", "the link finders' share from the acceptance check's calls: Σ_p ratio_p · 2w(1 + k)/2^256 / (1 − ρ)",
             link, ("ecr/sha-512",), ("positions", "rho", "k", "w"), (bind * 2 * v["w"] * (1 + v["k"]) * q256, zero, zero)),
        Term("roots", "the finder recomputing the registration's roots", (("Verity.Sampled.Live.rootsErr", S),
             ("Claims.ecrBound", "security/Definitions/Core/Interaction.lean")), ("ecr/sha-512",), ("roots_calls",),
             (v["roots_calls"] * q256, zero, zero)),
        Term("two_continuation", "a two-continuation finder a round opened after a digest, one for the record and one for "
             "the digests the composition binds: (rounds + 2)·√(q_M²/2^513) at q_M's 2t", two, ("cr/sha-512",),
             ("rounds",), (zero, (v["rounds"] + 2) * sq * 2, zero)),
        Term("two_continuation_h", "the same finders' share from hashing the two revealed messages, q_M's 2h", two,
             ("cr/sha-512",), ("rounds", "h"), ((v["rounds"] + 2) * sq * 2 * v["h"], zero, zero)),
        Term("hiding", "the coin trees' hiding term, 2·maxLeaves/2^257", (("Verity.Sampled.Live.hidingTerm", S),
             ("Verity.maxLeaves", GAME)), (), ("max_leaves",), (2 * v["max_leaves"] / 2 ** 257, zero, zero)),
        Term("file_finders", "the two file finders, 2·2²/2^513", (("Verity.Sampled.Live.delta", S),), ("cr/sha-512",), (),
             (2 * 4 * q513, zero, zero)),
        Term("read_finder", "SPOT's read finder, ε_cr at its budget callsMul·t: callsMul·t/2^256",
             (("Verity.Vocab.crBound", GAME), ("Verity.Sampled.Instance.callsMulOf", INSTANCE),
              ("Verity.Sampled.Instance.sampledBound", INSTANCE)), ("ecr/sha-512",), ("calls_mul",),
             (zero, v["calls_mul"] * q256, zero)),
        Term("forgery", "SPOT's ε_forge at its forger's budget callsMul·t, under euf-cma/lms-sha512-n64 at its stated level",
             (("Verity.Vocab.error", GAME), ("Hash.Assumptions.LmsUnforgeable", "security/Definitions/Hash/Assumptions.lean")),
             ("euf-cma/lms-sha512-n64",), ("forge_rate", "calls_mul"), (zero, v["calls_mul"] * v["forge_rate"], zero)),
        Term("pouw", OPEN[0], (("Verity.Vocab.PouwSound", GAME), ("Verity.Vocab.error", GAME)), left_out="open"),
        Term("policy", OPEN[1], (("Verity.Vocab.InferenceOnlySound", GAME),
             ("Verity.Vocab.inferenceOnlySound", "security/Proofs/Properties/Verity/InferenceOnly.lean")), left_out="open"),
        Term("sample_miss", "SPOT's sampling term, ε at δ: the sampling error, not cryptography's",
             (("Verity.Sampled.Live.sampleMiss", S),), left_out="excluded"),
    ]


def guarantee(locks: tuple[Path, ...] = LOCKS) -> dict[str, Any]:
    """`GUARANTEE`'s and `COMPOSITE`'s open premises and named assumptions (their reads the locks give an id), from the
    locks."""
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in locks]
    ids = {k: v for d in docs for k, v in d.get("ids", {}).items()}
    out = {}
    for name in (GUARANTEE, COMPOSITE):
        (rec,) = [d["guarantees"][name] for d in docs if name in d.get("guarantees", {})]
        prem = rec["premises"]
        out[name] = {"open": prem["open"], "assumptions": sorted({ids[r] for r in prem["reads"] if r in ids})}
    return out


def fleets(census: Path = CENSUS) -> dict[str, dict[str, int]]:
    """The datacenter (the reference deployment) and the world (the reference adversary, as machines of its size)."""
    (dc,) = [d for d in json.loads((census / "deployments.json").read_text()) if d.get("role") == "reference"]
    (world,) = [d for d in json.loads((census / "adversary.json").read_text()) if d.get("role") == "reference"]
    g, gpus = dc["gpus_per_machine"], int(world["h100_equivalents"])
    return {"datacenter": {"id": dc["id"], "machines": dc["machines"], "gpus": dc["machines"] * g, "window_s": dc["window_s"]},
            "world": {"id": world["id"], "machines": gpus // g, "gpus": gpus, "window_s": dc["window_s"]}}


def span(f: dict[str, int], secs: int, rate: Fraction) -> tuple[int, int]:
    """A fleet's plays and SHA-512 calls over `secs` seconds."""
    return f["machines"] * secs // f["window_s"], int(f["gpus"] * rate * secs)


def headline_total(p: dict[str, Param], fl: dict[str, dict[str, int]] | None = None) -> Fraction:
    over, period = HEADLINE
    plays, calls = span((fl or fleets())[over], dict(MONITORING_PERIODS)[period], p["hash_rate"].value)
    return sum((x for t in terms(p) if (x := t.at(plays, calls)) is not None), Fraction(0))


def figures(ts: list[Term], fl: dict[str, dict[str, int]], rate: Fraction) -> dict[str, Any]:
    """Each fleet's error over one window and each monitoring period: per term and total, log₂, the total without the
    forgery term, and the total's share a machine-window."""
    out: dict[str, Any] = {}
    for over, f in fl.items():
        periods = {}
        for period, secs in (("window", f["window_s"]), *MONITORING_PERIODS):
            plays, calls = span(f, secs, rate)
            each = {t.name: t.at(plays, calls) for t in ts if t.coeffs is not None}
            total = sum(each.values(), Fraction(0))
            periods[period] = {"plays": plays, "sha512_calls_log2": round(math.log2(calls), 2),
                               "terms_log2": {k: round(log2(x), 2) if x else None for k, x in each.items()},
                               "total_log2": round(log2(total), 2),
                               "total_without_forgery_log2": round(log2(total - each.get("forgery", 0)), 2),
                               "share_log2": round(log2(total / plays), 2)}
        out[over] = {**f, "periods": periods}
    return out


def sensitivity(p: dict[str, Param], fl: dict[str, dict[str, int]] | None = None) -> list[dict[str, Any]]:
    """The headline's total, log₂, at each free or stated parameter's choice and at two others."""
    alts = {"k": (2 ** 23, 2 ** 25), "M": (2 ** 64, 2 ** 128), "Rw": (2 ** 64, 2 ** 128), "v": (2 ** 48, 2 ** 64),
            "w": (2 ** 48, 2 ** 64), "h": (2 ** 48, 2 ** 64), "roots_calls": (2 ** 48, 2 ** 64),
            "positions": (2 ** 10, 2 ** 20), "hash_rate": (2 ** 32, 2 ** 34)}
    rows = []
    for name, others in alts.items():
        at = (lambda x: with_k(p, x)) if name == "k" else (lambda x, n=name: {**p, n: replace(p[n], value=Fraction(x))})
        rows.append({"param": name, "kind": p[name].kind, "choice_log2": round(log2(p[name].value), 2),
                     "total_log2": round(log2(headline_total(p, fl)), 2),
                     "others": [{"value_log2": round(math.log2(x), 2), "total_log2": round(log2(headline_total(at(x), fl)), 2)}
                                for x in others]})
    return rows


def report(p: dict[str, Param]) -> dict[str, Any]:
    ts, fl = terms(p), fleets()
    g = guarantee()
    return {"format": FORMAT, "description": DESCRIPTION, "open": list(OPEN), "guarantee": g,
            "card_rate_log2": round(card_rate_log2(), 2),
            "params": {k: {"value": str(x.value), "value_log2": round(log2(x.value), 2), "kind": x.kind, "source": x.source}
                       for k, x in p.items()},
            "terms": [{"name": t.name, "what": t.what, "basis": basis(t, p), "lean": [n for n, _ in t.lean],
                       "assumptions": list(t.assumptions), "params": list(t.params), "note": t.note} for t in ts],
            "sensitivity": sensitivity(p, fl), "fleets": figures(ts, fl, p["hash_rate"].value)}


def shown(x: Fraction) -> str:
    """`x` as a power of two where it is one, else its log₂."""
    n, d = x.numerator, x.denominator
    if n == 1 and d & (d - 1) == 0 and d > 2:
        return f"2^-{d.bit_length() - 1}"
    if d == 1 and n & (n - 1) == 0 and n > 2:
        return f"2^{n.bit_length() - 1}"
    return str(x) if d == 1 and n < 2 ** 20 else f"2^{log2(x):.3f}"


def markdown(r: dict[str, Any]) -> str:
    over, period = HEADLINE
    head = r["fleets"][over]["periods"][period]
    lines = [f"SPOT's cryptographic error in the remote application: {r['description']}.", "",
             f"Over the {over}'s {period}: log₂ total {head['total_log2']}, without the forgery term "
             f"{head['total_without_forgery_log2']}.", ""]
    lines += [f"- {x}" for x in r["open"]] + [""]
    for name, g in r["guarantee"].items():
        lines.append(f"- {name}: open premises {', '.join(g['open']) or 'none'}; assumptions "
                     f"{', '.join(g['assumptions']) or 'none'}")
    lines += ["", f"| term | basis | log₂ over the {over}'s {period} | parameters | Lean |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {t['name']} | {t['basis']} | {head['terms_log2'].get(t['name'], 'not in the total')} | "
              f"{', '.join(t['params'])} | {', '.join(t['lean'])} |" for t in r["terms"]]
    lines += [""] + [f"- {t['name']}: {t['note']}" for t in r["terms"] if t["note"]]
    lines += ["", "| parameter | value | kind | source |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {shown(Fraction(p['value']))} | {p['kind']} | {p['source']} |" for k, p in r["params"].items()]
    lines += ["", f"The commitment card's measured rate is 2^{r['card_rate_log2']} compressions a second a GPU.", "",
              f"| parameter | kind | log₂ choice | log₂ total, {over} {period} | at two other choices |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {s['param']} | {s['kind']} | {s['choice_log2']} | {s['total_log2']} | "
              + "; ".join(f"2^{o['value_log2']}: {o['total_log2']}" for o in s["others"]) + " |" for s in r["sensitivity"]]
    lines += ["", "| over | period | plays | log₂ T | log₂ total | without forgery | log₂ share a machine-window |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for o, f in r["fleets"].items():
        for per, x in f["periods"].items():
            lines.append(f"| {o} | {per} | {x['plays']} | {x['sha512_calls_log2']} | {x['total_log2']} | "
                         f"{x['total_without_forgery_log2']} | {x['share_log2']} |")
    return "\n".join(lines) + "\n"


def record(r: dict[str, Any]) -> None:
    """Write `METRICS` into this research run's result, each fleet's week, month and year, `HEADLINE` last."""
    w = kit.writer(settings.RESEARCH_RUN_DIR)
    kept = w.out / KEPT
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps(r, indent=1) + "\n")
    w.fingerprint(benchmark="remote_error", format=FORMAT, params={k: x["value"] for k, x in r["params"].items()})
    w.measurement_file(KEPT)
    grid = [(o, p) for o in r["fleets"] for p, _ in MONITORING_PERIODS if (o, p) != HEADLINE] + [HEADLINE]
    for metric, key in METRICS.items():
        unit = "log₂ of error" if key == "total" else "log₂ of error a machine-window"
        for over, period in grid:
            x = r["fleets"][over]["periods"][period]
            w.measurement(metric, x[f"{key}_log2"], unit, inputs=[f"modeled: {DESCRIPTION}", f"proved: {GUARANTEE}"],
                          mark="modeled", period=period, over=over)
    w.validation("passed", "benchmarks/remote_error/run.py",
                 f"evaluated {GUARANTEE}'s cryptographic terms of SPOT at the stated parameters: {DESCRIPTION}", [KEPT])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="the figures as JSON, not the tables")
    ap.add_argument("--blocks", type=int, help="the class circuit's proof blocks, which bound the commit strings read")
    a = ap.parse_args(argv)
    r = report(params(a.blocks))
    print(json.dumps(r, indent=1) if a.json else markdown(r), end="\n" if a.json else "")
    if settings.is_set("RESEARCH_RUN_DIR"):
        record(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
