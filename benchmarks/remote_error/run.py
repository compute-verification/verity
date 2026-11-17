"""The chance the remote application's cryptography is fooled: the cryptographic part of the soundness error of the
composite `verity/remote/v1` (SPOT, PoUW's Pearl+ run verdict and the inference-only policy), evaluated from parameters
for a prover making at most T SHA-512 calls, per window and pooled over the monitoring periods, over one datacenter and
over the world's compute:

    uv run python benchmarks/remote_error/run.py [--json] [--positions-log2 P]

A security figure, not a measurement. Lean doesn't instantiate the remote guarantee at numeric parameters: its one
instance, the honest witness, has every error ⊤. So the bound's formula is proved, and its value is evaluated here, in
exact rationals (each irrational rounded up), at the parameters `PARAMS` states, each with its source.

The guarantee is `GUARANTEE`, read from its lock with its open premises and named assumptions. Soundness at a composite
is the union bound over its subprotocols' errors (`Verity.Vocab.error`): each one's δ at the prover's calls t, plus
ε_forge at its forger's budget, plus its loss. The terms here are those of SPOT's δ (`Verity.Sampled.Live.delta`) that
cryptography, not sampling, bounds, and SPOT's forgery term; each says its basis:

* ``proved``: its value is a closed term of the guarantee's δ, so the guarantee proves it, under its open premises;
* ``assumed``: its formula is the guarantee's, at a parameter Lean leaves free, evaluated at the stated parameter;
* ``open``: Lean has no formula for it at the remote composite, so the total leaves it out (PoUW's and the policy's
  errors, each its base vocabulary's);
* ``excluded``: a sampling term, ε at δ, which is not this figure.

Pooling: a period holds one play a machine a window, plays = machines · period / window. A term a + b·t + c·t² a play
sums over the plays to at most plays·a + b·T + c·T², with T the fleet's SHA-512 calls over the period (Σ t² ≤ (Σ t)²).
T is the fleet's GPUs at `HASH_RATE_LOG2` calls a second each, an assumed bound; the world's compute (the adversary
resource model, `benchmarks/census/adversary.json`'s reference) counts as machines of the datacenter's size. Each
total's share is per machine-window audited: the census states no AND gates a machine-window. Under a research run it
writes `METRICS` into the run's result.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_bench import kit, settings

from verity.protocol import MONITORING_PERIODS

ROOT = Path(__file__).resolve().parents[2]
CENSUS = ROOT / "benchmarks" / "census"
LOCKS = (ROOT / "security" / "lean-audit.json", ROOT / "security" / "Properties" / "lean-audit.json")
#: the lock guarantee whose δ these terms are, and the composite's union bound it instantiates
GUARANTEE = "Verity.Sampled.remote_soundness_deployedReady"
COMPOSITE = "Verity.remote_soundness_of_subprotocols"
#: the metrics `metrics.toml` declares: the total over each fleet and period, and its share a machine-window
METRICS = {"soundness/remote-crypto-error-log2": "total", "soundness/remote-crypto-error-share-log2": "share"}
#: the metrics' fleet and period each id names, written last
HEADLINE = ("world", "month")
FORMAT = "remote-crypto-error/v1"
KEPT = "out/remote-error.json"
HEADLINE_NOTE = ("not instantiated at numeric parameters in Lean: bound's formula proved, value evaluated in Python; "
                 "PoUW's and the inference-only policy's terms are open and not in the total")

SQRT_BITS = 64
#: e from below, so that 1/e, 1/(eM) and 1/(1 − 1/e) are bounded above
E_LOW = Fraction(2718281, 10 ** 6)
#: the adversary's SHA-512 compressions a second a GPU, log₂: an assumed bound, above any GPU's measured commit rate
HASH_RATE_LOG2 = 33

S = "security/Proofs/Properties/Verity/Sampled.lean"
INSTANCE = "security/Proofs/Properties/Verity/Sampled/Instance.lean"
STRICTZ = "security/Proofs/Flock/Soundness/Discharge/ZkLink/StrictZ.lean"
GAME = "security/Properties/Verity/Game.lean"


@dataclass(frozen=True)
class Param:
    value: Fraction
    basis: str
    source: str


#: the parameters the terms read: ``lean`` a Lean definition's value, ``code`` a bound the verifier's code enforces,
#: ``assumed`` a value no Lean or census entry states, ``census`` the census's reference
PARAMS: dict[str, Param] = {
    "k": Param(Fraction(2 ** 23), "lean", "Verity.Sampled.Instance.Honest.k: the honest bound's extraction budget"),
    "rho": Param(Fraction(3, 8), "lean", "Verity.Sampled.Instance.Honest.ρ: the honest bound's rate margin"),
    "M": Param(Fraction(2 ** 256), "assumed", "the instance's extraction trials (SessionBound.M); any M ≥ 1 is admissible"),
    "Rw": Param(Fraction(2 ** 256), "assumed", "the instance's rewinding trials (SessionBound.Rw); any Rw ≥ 1 is admissible"),
    "v": Param(Fraction(2 ** 32), "assumed", "SHA-512 calls of one opening's verification (SessionBound.v)"),
    "w": Param(Fraction(2 ** 32), "assumed", "SHA-512 calls of one acceptance check (SessionBound.w)"),
    "h": Param(Fraction(2 ** 32), "assumed", "SHA-512 calls hashing one revealed message (SessionBound.h)"),
    "roots_calls": Param(Fraction(2 ** 40), "assumed", "SHA-512 calls recomputing the registration's roots "
                         "(Flock.Bridge.rootsCostL), a bound on the public file's leaves"),
    "positions": Param(Fraction(2 ** 20), "assumed", "commit strings a window's sample reads in expectation "
                       "(Σ_p ratio_p over the binding's positions); the remote application's circuit and sampling "
                       "design fix it, and no Lean or census entry states it"),
    "table_bound": Param(Fraction(1, 2 ** 205), "assumed", "Claims.Flock.Numbers.εtable_zk: a lemma bounding a --zk "
                         "table's bound at Flock ZK's parameters (TabZK.M1); no lock guarantee proves it at the remote "
                         "composite"),
    "N": Param(Fraction(2 ** 13), "assumed", "the table's inner proof's N, TabZK.M1's bound"),
    "levels": Param(Fraction(7), "code", "Flock.fast100's levels at m ≤ 35, the largest m Flock.Zk.schedule accepts"),
    "log_len": Param(Fraction(23), "code", "level 0's log length under Flock.Zk.schedule, which no later level "
                     "exceeds, proved at each m (security/Proofs/Flock/Soundness/Discharge/ZkBind/Rate.lean)"),
    "queries": Param(Fraction(277), "code", "Flock.Zk.queries0's largest, level 0 at m = 25; later levels query fewer"),
    "rounds": Param(Fraction(2 * 2 ** 9 + 1), "code", "Verity.Sampled.Live.rounds: 2·2^roundsLog + 1 at "
                    "Flock.Zk.coinSpec's roundsLog 9"),
    "max_leaves": Param(Fraction(2 ** 49), "lean", "maxLeaves = maxTreeLines · maxTreeCoins = 2^23 · 2^26"),
    "calls_mul": Param(Fraction(2), "lean", "Verity.Sampled.Instance.callsMulOf .sampled: SPOT's finder budget 2t"),
    "forge_rate": Param(Fraction(1, 2 ** 256), "assumed", "ε_forge(q, Q) = q/2^256: the LMS (n = 64) forgery "
                        "bound Hash.Assumptions.LmsUnforgeable takes as a parameter, taken no tighter than SHA-512's "
                        "expected collision bound"),
}


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
    basis: str
    lean: tuple[tuple[str, str], ...]
    assumptions: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    coeffs: tuple[Fraction, Fraction, Fraction] | None = None
    note: str = field(default="")

    def at(self, plays: int, calls: int) -> Fraction | None:
        if self.coeffs is None:
            return None
        a, b, c = self.coeffs
        return plays * a + b * calls + c * calls * calls


def terms(p: dict[str, Param] = PARAMS) -> list[Term]:
    """SPOT's cryptographic terms of `Verity.Sampled.Live.delta` and its forgery term, then the terms the total leaves
    out, at the parameters `p`."""
    v = {k: x.value for k, x in p.items()}
    sq = sqrt_up(Fraction(1, 2 ** 513))
    rate = Fraction(1) / (1 - 1 / E_LOW)
    s = 2 * v["levels"] * sqrt_up(2 ** v["log_len"] * v["queries"]) + 2 * v["N"] + 2 + sqrt_up(v["k"])
    bind = v["positions"] / (1 - v["rho"])
    q513 = Fraction(1, 2 ** 513)
    q256 = Fraction(1, 2 ** 256)
    ks = (("FlockSoundness.Discharge.ZkLink.ksAvgStrictZ", STRICTZ), ("FlockSoundness.Discharge.ZkLink.ksStrictZ", STRICTZ),
          ("FlockSoundness.Discharge.ZkLink.rateZ", "security/Proofs/Flock/Soundness/Discharge/ZkLink/Compiled.lean"))
    return [
        Term("table_iop", "the --zk table's bound, over 1 − N₀/(e·k)", "assumed",
             (*ks, ("Claims.Flock.Numbers.εtable_zk", "security/Proofs/Flock/Claims/NumbersTable.lean"),
              ("TabZK.M1", "security/Proofs/Flock/Soundness/Discharge/ZkSession/Numbers.lean")),
             params=("table_bound", "k", "log_len"), coeffs=(rate * v["table_bound"], Fraction(0), Fraction(0))),
        Term("table_collisions", "the table's rewinding and self-clash finders: its levels', pads', coins' and k's "
             "√(·q_F²/2^513) terms at q_F = 2t + 2v, and q_S²/2^513 at q_S = t + v", "assumed", ks,
             ("cr/sha-512",), ("levels", "log_len", "queries", "N", "k", "v"),
             (rate * (s * sq * 2 * v["v"] + v["v"] ** 2 * q513), rate * (2 * s * sq + 2 * v["v"] * q513), rate * q513)),
        Term("binding", "the link finders and extraction over the commit strings a sample reads: "
             "Σ_p ratio_p · (2(t + w)(1 + k)/2^256 + 1/(eM) + k/(e·Rw)) / (1 − ρ)", "assumed",
             (("Verity.Sampled.Live.sessionDelta", S), ("Verity.Sampled.Live.tL", S)), ("ecr/sha-512",),
             ("positions", "rho", "k", "w", "M", "Rw"),
             (bind * (2 * v["w"] * (1 + v["k"]) * q256 + 1 / (E_LOW * v["M"]) + v["k"] / (E_LOW * v["Rw"])),
              bind * 2 * (1 + v["k"]) * q256, Fraction(0))),
        Term("roots", "the finder recomputing the registration's roots", "assumed",
             (("Verity.Sampled.Live.rootsErr", S), ("Claims.ecrBound", "security/Definitions/Core/Interaction.lean")),
             ("ecr/sha-512",), ("roots_calls",), (v["roots_calls"] * q256, Fraction(0), Fraction(0))),
        Term("two_continuation", "a two-continuation finder a round opened after a digest, one for the record and one "
             "for the digests the composition binds: (rounds + 2)·√(q_M²/2^513) at q_M = 2t + 2h", "assumed",
             (("Verity.Sampled.Live.twoRun", S), ("Verity.Sampled.Live.rounds", S), ("Verity.Sampled.Live.qM", S)),
             ("cr/sha-512",), ("rounds", "h"),
             ((v["rounds"] + 2) * sq * 2 * v["h"], (v["rounds"] + 2) * sq * 2, Fraction(0))),
        Term("hiding", "the coin trees' hiding term, 2·maxLeaves/2^257", "proved",
             (("Verity.Sampled.Live.hidingTerm", S), ("Verity.maxLeaves", GAME)), (), ("max_leaves",),
             (2 * v["max_leaves"] / 2 ** 257, Fraction(0), Fraction(0))),
        Term("file_finders", "the two file finders, 2·2²/2^513", "proved", (("Verity.Sampled.Live.delta", S),),
             ("cr/sha-512",), (), (2 * 4 * q513, Fraction(0), Fraction(0))),
        Term("read_finder", "SPOT's read finder, ε_cr at its budget 2t: 2t/2^256", "proved",
             (("Verity.Vocab.crBound", GAME), ("Verity.Sampled.Instance.callsMulOf", INSTANCE),
              ("Verity.Sampled.Instance.sampledBound", INSTANCE)), ("ecr/sha-512",), ("calls_mul",),
             (Fraction(0), v["calls_mul"] * q256, Fraction(0))),
        Term("forgery", "SPOT's ε_forge at its forger's budget 2t", "assumed",
             (("Verity.Vocab.error", GAME), ("Hash.Assumptions.LmsUnforgeable", "security/Definitions/Hash/Assumptions.lean")),
             ("euf-cma/lms-sha512-n64",), ("forge_rate", "calls_mul"),
             (Fraction(0), v["calls_mul"] * v["forge_rate"], Fraction(0))),
        Term("pouw", "PoUW's (Pearl+ run verdict's) error: its δ, ε_forge at its budget and its loss", "open",
             (("Verity.Vocab.PouwSound", GAME), ("Verity.Vocab.error", GAME)), note="the base vocabulary's fields; "
             "the guarantee takes PouwSound as an open premise"),
        Term("policy", "the inference-only policy's error: ε_forge at its budget", "open",
             (("Verity.Vocab.error", GAME),
              ("Verity.Vocab.inferenceOnlySound", "security/Proofs/Properties/Verity/InferenceOnly.lean")),
             note="its target is proved with an empty event, so its δ may be 0; its forger's budget and ε_forge are "
             "the base vocabulary's"),
        Term("sample_miss", "SPOT's sampling term, ε at δ", "excluded", (("Verity.Sampled.Live.sampleMiss", S),),
             note="the sampling error, not cryptography's"),
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


def figures(ts: list[Term] | None = None, fl: dict[str, dict[str, int]] | None = None) -> dict[str, Any]:
    """Each fleet's error over one window and each monitoring period: per term and total, log₂, and the total's share a
    machine-window."""
    ts, fl = ts if ts is not None else terms(), fl or fleets()
    out: dict[str, Any] = {}
    for over, f in fl.items():
        periods = {}
        for period, secs in (("window", f["window_s"]), *MONITORING_PERIODS):
            plays = f["machines"] * secs // f["window_s"]
            calls = f["gpus"] * 2 ** HASH_RATE_LOG2 * secs
            each = {t.name: t.at(plays, calls) for t in ts if t.coeffs is not None}
            total = sum(each.values(), Fraction(0))
            periods[period] = {"plays": plays, "sha512_calls_log2": round(math.log2(calls), 2),
                               "terms_log2": {k: round(log2(x), 2) if x else None for k, x in each.items()},
                               "total_log2": round(log2(total), 2), "share_log2": round(log2(total / plays), 2)}
        out[over] = {**f, "periods": periods}
    return out


def report(params: dict[str, Param] = PARAMS) -> dict[str, Any]:
    ts = terms(params)
    return {"format": FORMAT, "headline": HEADLINE_NOTE, "guarantee": guarantee(), "hash_rate_log2": HASH_RATE_LOG2,
            "params": {k: {"value": str(p.value), "basis": p.basis, "source": p.source} for k, p in params.items()},
            "terms": [{"name": t.name, "what": t.what, "basis": t.basis, "lean": [n for n, _ in t.lean],
                       "assumptions": list(t.assumptions), "params": list(t.params), "note": t.note} for t in ts],
            "fleets": figures(ts)}


def shown(x: Fraction) -> str:
    """`x` as a power of two where it is one."""
    n, d = x.numerator, x.denominator
    if n == 1 and d & (d - 1) == 0 and d > 2:
        return f"2^-{d.bit_length() - 1}"
    if d == 1 and n & (n - 1) == 0 and n > 2:
        return f"2^{n.bit_length() - 1}"
    return str(x)


def markdown(r: dict[str, Any]) -> str:
    lines = [f"Remote application's cryptographic soundness error: {r['headline']}.", ""]
    for name, g in r["guarantee"].items():
        lines.append(f"- {name}: open premises {', '.join(g['open']) or 'none'}; assumptions "
                     f"{', '.join(g['assumptions']) or 'none'}")
    over, period = HEADLINE
    at = r["fleets"][over]["periods"][period]["terms_log2"]
    lines += ["", f"| term | basis | log₂ over the {over}'s {period} | Lean | parameters |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {t['name']} | {t['basis']} | {at.get(t['name'], 'not in the total')} | {', '.join(t['lean'])} | "
              f"{', '.join(t['params'])} |" for t in r["terms"]]
    lines += ["", "| parameter | value | basis | source |", "| --- | --- | --- | --- |"]
    lines += [f"| {k} | {shown(Fraction(p['value']))} | {p['basis']} | {p['source']} |" for k, p in r["params"].items()]
    lines += ["", "| over | period | plays | log₂ T | log₂ total | log₂ share a machine-window |", "| --- | --- | --- | --- | --- | --- |"]
    for over, f in r["fleets"].items():
        for period, x in f["periods"].items():
            lines.append(f"| {over} | {period} | {x['plays']} | {x['sha512_calls_log2']} | {x['total_log2']} | {x['share_log2']} |")
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
            w.measurement(metric, x[f"{key}_log2"], unit, inputs=[f"modeled: {r['headline']}", f"proved: {GUARANTEE}"],
                          mark="modeled", period=period, over=over)
    w.validation("passed", "benchmarks/remote_error/run.py",
                 f"evaluated {GUARANTEE}'s cryptographic terms at the stated parameters; {r['headline']}", [KEPT])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="the figures as JSON, not the tables")
    ap.add_argument("--positions-log2", type=int, help="log₂ of the commit strings a window's sample reads")
    a = ap.parse_args(argv)
    params = dict(PARAMS)
    if a.positions_log2 is not None:
        params["positions"] = Param(Fraction(2 ** a.positions_log2), "assumed", "--positions-log2")
    r = report(params)
    print(json.dumps(r, indent=1) if a.json else markdown(r), end="\n" if a.json else "")
    if settings.is_set("RESEARCH_RUN_DIR"):
        record(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
