"""The block record: what each subprotocol writes for one server block and the verdict on it, one document
``verity/block-record/v2``. It extends the all-protocols runner's per-period report
(``verity/all-protocols-period/v1``, and its per-window ``verity/all-protocols-window/v1``) with the block's server and
steps, every leg in one shape, the Lean verdict with the JointPooledLive error sum, and the monitoring period.

**The record.**

* ``server``: the node the block ran on; ``server_time``: the block's steps, ``[{server, steps: [lo, hi)}]`` (the circuits'
  blocks schema).
* ``legs``: ``integrity``, ``pouw``, ``pous``, ``bandwidth``, ``certifier``, ``wipe``, each ``null`` when the block ran no
  such audit, else ``{protocol, owner, server, source: real|fixture, answer, quantity, error, error_from, at, runs}``:
  ``answer`` the core verifier's answer (verdict, reason, commitment, public); ``quantity`` the leg's certificate
  ``{symbol, value, unit, ...}`` (integrity bound, W_cert, S0, Q_cert, K, the wipe's foreign_bits at its reads ``k``
  and ratio ``r``, ``r`` null in a wipe from before ``verity/wipe/v0`` published it; S0 is PoUS's state bound
  ``⌊ρ·|C|⌋``, which a record may carry as ``S_cert``, read and never written); ``error`` the probability over the leg's
  own coins that it accepts while its claim fails, as the run established it, read from ``error_from``; ``at`` the
  certificate at δ = 1/100 and 1/10^6 (core's `DELTAS`): the run's own wherever ``error`` ≤ δ (a claim holding except with
  ``error`` holds except with δ), integrity's recomputed from its sample at δ (`integrity_bound`, the bound sampled
  proofs' Lean ``countBound`` gives), else not established. The sample is the answer's own sampling design: where its
  profile samples each stratum on its own, the quantity carries the ``strata`` (``[{name, size, k}]``) and the bound is
  the stratified design's, the most incorrect proof blocks that some placement over the strata hides with probability
  at least δ, which a uniform sample of as many units understates. ``named_terms``, which a subprotocol of
  `NAMES_TERMS` (bandwidth) always carries and another may, is the terms its ``error`` leaves out, as its answer's
  profile names them: PoUB's ``ledger.NAMED_TERMS``, each term ``{claim, term, value: null, in_total: false, status}``.
  They have no number, so ``error``, ``at`` and every sum of this record hold none of them and each reads with them
  beside it (`error_form`): ``holds`` at δ says the claim holds except with δ and those terms. A leg whose ``server``
  is not the block's is carried but not composed.
* ``composed``: ``answer``, the Lean check of the block's product (`CHECKS`: ``onsite check`` or ``remote check``)'s
  answer on the block (protocol, verdict, reason, commitment), None when it didn't run; ``verdict``, that answer's
  (`Core.Verdict`: accept, reject or undecided), undecided when there is none; nothing here combines the legs' verdicts
  into another. The rest is what the product's legs say (`compose`): ``legs`` (each composed leg's own verdict), what is
  ``missing`` on the block's server and what ran ``elsewhere``; ``joint_horizon`` is `JOINT_HORIZON`'s error over this
  block, the accepted legs' errors summed, and at each δ of `DELTAS` that δ times the legs established at it; its
  ``named_terms`` gives, by subprotocol, the terms those errors leave out, which its sums and the monitoring period's
  leave out too (``{}``: they leave none out). The statement names five audits; the wipe is ``outside_statement``.
  The product (`PRODUCTS`) is onsite's, every leg, unless ``product`` names another: remote's composes sampled proofs
  and PoUW, and its other legs are ``dropped``, not run and not composed, so they are never ``missing``.
* ``monitoring_period``: the census deployment (a parameter: ``{id, machines, window_s, ...}``) and, per period of
  core's `PERIODS`, its blocks over all compute (machines × blocks a machine) and JointPooledLive's union bound over them at this
  block's errors (``error_sum`` × blocks), and the per-block error each δ of `DELTAS` needs over the period (δ / blocks).
  Each reads with the block's ``joint_horizon.named_terms`` beside it, once a block.

`check` recomputes every derived field from the legs and the answer. `from_period` builds a record from a runner report,
the Lean answer and extra legs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity.protocols.reporting import DELTAS, PERIODS
from verity.protocols.spot.verifier.profile import Stratified, Stratum

FORMAT = "verity/block-record/v2"
EXTENDS = ("verity/all-protocols-period/v1", "verity/all-protocols-window/v1")
LEGS = ("integrity", "pouw", "pous", "bandwidth", "certifier", "wipe")
#: the runner's protocol keys, by leg
RUNNER_KEYS = {"integrity": "integrity", "pouw": "work", "pous": "space", "bandwidth": "bandwidth", "certifier": "communication"}
#: the legs each product composes: onsite every leg, remote those that need no hardware of the auditor's at the site
PRODUCTS = {"onsite": LEGS, "remote": ("integrity", "pouw")}
#: the subprotocols whose error leaves terms out: each one's row names them (``answer.profile.named_terms``), or is refused
NAMES_TERMS = ("bandwidth",)
#: what puts a refused one right: its piece's answer names the terms
NAMES_TERMS_FIX = "its piece writes them in its answer's profile as named_terms (PoUB's ledger.NAMED_TERMS)"
#: what a named term says: its claim, how it is written, that it has no number and is in no total, and why
NAMED_TERM = ("claim", "term", "value", "in_total", "status")
#: the protocol each product's Lean check (`onsite check`, `remote check`) names in its answer, and `Core.Verdict`'s names
CHECKS = {"onsite": "verity/onsite/v0", "remote": "verity/remote/v0"}
VERDICTS = ("accept", "reject", "undecided")
JOINT_HORIZON = {"name": "Accounting.Guarantees.JointPooledLive", "proved": True,
                 "file": "Security/Definitions/Accounting/Guarantees/JointLive.lean",
                 "audits": ["integrity", "pouw", "pous", "bandwidth", "certifier"]}


def frac(x: Any) -> Fraction:
    return Fraction(str(x))


def integrity_bound(n: int, kd: int, delta: Fraction, strata: Sequence[Mapping[str, Any]] | None = None) -> int:
    """The most incorrect proof blocks of ``n`` an accepted sample of ``kd`` leaves except with ``δ``: the integrity
    profile's count bound (`Stratified.count_bound`), the largest ``K`` whose sampling term is at least ``δ``. With no
    ``strata`` the sample is uniform over the ``n`` and the term is ``C(n − K, kd)/C(n, kd)``. With ``strata``
    (``[{name, size, k}]``, which must hold the ``n`` units and sample ``kd`` of them) each stratum is sampled on its
    own, and the term is the likeliest escape of ``K`` incorrect proof blocks over their placements,
    ``∏ C(n_s − m_s, k_s)/C(n_s, k_s)``: a stratum sampled whole hides none, and one sampled thinly hides more than the
    uniform sample of ``kd`` would."""
    if strata is None:
        return Stratified.uniform(n, kd).count_bound(delta)
    design = Stratified(tuple(Stratum(str(s["name"]), int(s["size"]), int(s["k"])) for s in strata))
    if (design.population, design.k) != (n, kd):
        raise ValueError(f"the strata hold {design.population} units and sample {design.k} of them, not the {kd} of {n} "
                         f"beside them")
    return design.count_bound(delta)


def _sample(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The sample of an integrity answer's checked level, as its quantity carries it: the level's size, the units
    sampled, and the ``strata`` of a stratified sampling design (`Stratified.to_json`); another design has none, and
    its sample is read as uniform."""
    level = answer["profile"]["levels"][-1]
    n, kd = int(level["size"]), int(answer["profile"]["drawn"][level["name"]])
    design = level.get("draw")
    if not isinstance(design, Mapping) or design.get("law") != "stratified":
        return {"of": n, "drawn": kd}
    strata = [{"name": str(s["name"]), "size": int(s["size"]), "k": int(s["k"])} for s in design["strata"]]
    integrity_bound(n, kd, DELTAS[0], strata)
    return {"of": n, "drawn": kd, "strata": strata}


def named_terms_bad(named: Any) -> str | None:
    """Why ``named`` is not an error's named terms, None when it is: at least one term by name, each ``{claim, term,
    value, in_total, status}`` with ``value`` null and ``in_total`` false. A term with a number is no named term: it
    belongs in the error."""
    if not isinstance(named, Mapping) or not named:
        return "not an object of at least one term"
    for k, t in named.items():
        if not isinstance(t, Mapping) or sorted(t) != sorted(NAMED_TERM):
            return f"{k}: not {{{', '.join(NAMED_TERM)}}}"
        if not all(isinstance(t[x], str) and t[x] for x in ("claim", "term", "status")):
            return f"{k}: its claim, term and status are not each a nonempty string"
        if t["value"] is not None or t["in_total"] is not False:
            return f"{k}: it has a value or is in a total, so it is part of the error and not a term beside it"
    return None


def error_form(error: Any, named: Mapping[str, Mapping[str, Any]] | None) -> str:
    """How ``error`` reads beside the terms it leaves out (``named``, a subprotocol's ``named_terms``): the error, then
    each term with why it has no number, in the words of the terms themselves, so PoUB's reads as its ledger's
    ``DELTA_FORM`` does with ``error`` for δ. With no term, the error alone."""
    return " + ".join([str(error), *(f"{t['term']} ({t['status']})" for t in (named or {}).values())])


def _quantity(leg: str, row: Mapping[str, Any]) -> tuple[dict[str, Any], Fraction, str]:
    """A leg's certificate, its error and where the error is read from, out of its runner row (or the wipe's leg)."""
    q, a = row.get("quantity") or {}, row["answer"]
    pub = a.get("public") or {}
    if leg == "integrity":
        return ({"symbol": "integrity bound", "value": int(q["bound"]), "unit": "wrong units", **_sample(a)},
                frac(row["delta"]), "delta")
    if leg == "pouw":
        w = q["work"]
        return ({"symbol": "W_cert", "value": int(w["credited"]), "unit": "work", "tiles": int(w["credited_tiles"])},
                frac(row["delta"]), "delta")
    if leg == "pous":
        return ({"symbol": "S0", "value": int(q["s_bits"]), "unit": "bits", "certified": bool(q["certified"])},
                frac(q["eps"]), "quantity.eps")
    if leg == "bandwidth":
        return ({"symbol": "Q_cert", "value": int(q["q_bytes"]), "unit": "bytes"}, frac(q["eps"]), "quantity.eps")
    if leg == "certifier":
        return ({"symbol": "K", "value": int(q["K"]), "unit": "bits", "windows": int(q["windows"])}, frac(q["eps"]),
                "quantity.eps")
    if leg == "wipe":
        k = pub.get("k", (a.get("profile") or {}).get("k"))
        return ({"symbol": "foreign_bits", "value": int(pub["foreign_bits"]), "unit": "bits", "k": k, "r": pub.get("r"),
                 "M": int(pub["memory_bits"]), "certified": bool(pub["certified"])}, frac(pub["eps"]), "answer.public.eps")
    raise ValueError(f"{leg}: not a leg of {LEGS}")


def at_deltas(leg: str, quantity: Mapping[str, Any], error: Fraction) -> dict[str, Any]:
    """The leg's certificate at each δ of `DELTAS` (module docstring)."""
    out: dict[str, Any] = {}
    for d in DELTAS:
        if leg == "integrity":
            b = integrity_bound(quantity["of"], quantity["drawn"], d, quantity.get("strata"))
            out[str(d)] = {"holds": True, "value": b, "epsilon": str(Fraction(b, quantity["of"])), "by": "draw"}
        elif error <= d:
            out[str(d)] = {"holds": True, "value": quantity["value"], "epsilon": None, "by": "run"}
        else:
            out[str(d)] = {"holds": False, "why": f"the run established it except with {error}, more than {d}"}
    return out


def leg(name: str, row: Mapping[str, Any], *, server: str, runs: Sequence[str], source: str = "real") -> dict[str, Any] | None:
    """One leg of the record from its runner row (or, for the wipe, its leg document); None for a row the runner marks
    ``missing``. Its answer is the row's verdict, reason, commitment and public outputs, and its profile's ``placement``
    where the answer has one (which host ran each party). Its ``named_terms`` are the answer's profile's, copied; a row
    of `NAMES_TERMS` whose profile names none, or any row whose profile names them in another shape, is a ValueError."""
    if row.get("verdict") == "missing":
        return None
    if source not in ("real", "fixture"):
        raise ValueError(f"source {source!r}: real or fixture")
    a = row["answer"]
    quantity, error, error_from = _quantity(name, row)
    answer = {k: a.get(k) for k in ("verdict", "reason", "commitment", "public")}
    profile = a["profile"] if isinstance(a.get("profile"), Mapping) else {}
    placement, named = profile.get("placement"), profile.get("named_terms")
    if placement is not None:
        answer["profile"] = {"placement": placement}
    if named is None and name in NAMES_TERMS:
        raise ValueError(f"{name}: its answer's profile names no term its error leaves out; {NAMES_TERMS_FIX}")
    if named is not None and (why := named_terms_bad(named)):
        raise ValueError(f"{name}: its answer's profile's named_terms: {why}")
    return {"protocol": a["protocol"], "owner": row.get("owner"), "server": server, "source": source,
            "answer": answer,
            "quantity": quantity, "error": str(error), "error_from": error_from, "at": at_deltas(name, quantity, error),
            "runs": list(runs)} | ({} if named is None else {"named_terms": {k: dict(t) for k, t in named.items()}})


def compose(server: str, legs: Mapping[str, Any], product: str = "onsite") -> dict[str, Any]:
    """What ``product``'s legs say about the block and the JointPooledLive error over it (module docstring), no verdict:
    the record's is its Lean answer's (`composed`). A product other than onsite's names itself and its ``dropped``
    legs."""
    if product not in PRODUCTS:
        raise ValueError(f"product {product!r}: not one of {sorted(PRODUCTS)}")
    composes = PRODUCTS[product]
    here = {k: v for k, v in legs.items() if k in composes and v is not None and v["server"] == server}
    elsewhere = sorted(k for k, v in legs.items() if k in composes and v is not None and v["server"] != server)
    missing = sorted(k for k in composes if k not in here)
    verdicts = {k: v["answer"]["verdict"] for k, v in here.items()}
    accepted = sorted(k for k, v in verdicts.items() if v == "accept")
    terms = {k: here[k]["error"] for k in accepted}
    jh = {"statement": JOINT_HORIZON, "terms": terms, "error_sum": str(sum((frac(e) for e in terms.values()), Fraction(0))),
          "named_terms": {k: here[k]["named_terms"] for k in accepted if "named_terms" in here[k]},
          "outside_statement": sorted(k for k in terms if k not in JOINT_HORIZON["audits"]),
          "at": {str(d): {"error_sum": str(d * sum(1 for k in accepted if here[k]["at"][str(d)]["holds"])),
                          "not_established": sorted(k for k in accepted if not here[k]["at"][str(d)]["holds"])}
                 for d in DELTAS}}
    named = {} if product == "onsite" else {"product": product, "dropped": sorted(k for k in LEGS if k not in composes)}
    return {"legs": verdicts, "missing": missing, "elsewhere": elsewhere, "joint_horizon": jh, **named}


def answer_of(answer: Mapping[str, Any] | None, product: str = "onsite") -> dict[str, Any] | None:
    """The record's form of ``product``'s Lean check's answer (`CHECKS`), as its ``check`` printed it; None for None.
    Raises ValueError for anything else: another protocol's answer, or a verdict that is not `Core.Verdict`'s."""
    if answer is None:
        return None
    if answer.get("protocol") != CHECKS[product] or answer.get("verdict") not in VERDICTS:
        raise ValueError(f"not an answer of the {product} check ({CHECKS[product]}) with a verdict of {VERDICTS}: "
                         f"{dict(answer)}")
    return {k: answer.get(k) for k in ("protocol", "verdict", "reason", "commitment")}


def composed(server: str, legs: Mapping[str, Any], answer: Mapping[str, Any] | None,
             product: str = "onsite") -> dict[str, Any]:
    """The record's ``composed``: the Lean answer's verdict on the block, the answer, and what ``product``'s legs say
    (`compose`; module docstring)."""
    summary = compose(server, legs, product)
    a = answer_of(answer, product)
    return {"verdict": a["verdict"] if a else "undecided", "answer": a, **summary}


def dropped_ran(legs: Mapping[str, Any], product: str) -> list[str]:
    """The legs ``product`` drops that ran nonetheless: a dropped leg is one the block didn't run."""
    return sorted(k for k in LEGS if k not in PRODUCTS[product] and legs.get(k) is not None)


def monitoring_period(deployment: Mapping[str, Any], error_sum: Fraction) -> dict[str, Any]:
    """The monitoring period over all compute (module docstring), a block being one of a machine's ``window_s`` windows.
    Each period's sum is ``error_sum`` over its blocks and leaves out what ``error_sum`` does: it reads with the block's
    ``joint_horizon.named_terms`` beside it, once a block."""
    periods = {}
    for name, seconds in PERIODS:
        per_machine = seconds // int(deployment["window_s"])
        blocks = int(deployment["machines"]) * per_machine
        union = error_sum * blocks
        periods[name] = {"seconds": seconds, "blocks_per_machine": per_machine, "blocks": blocks,
                         "joint_horizon": {"error_sum": str(union), "vacuous": union >= 1},
                         "per_block_error_for": {str(d): str(d / blocks) for d in DELTAS}}
    return {"deployment": dict(deployment), "deltas": [str(d) for d in DELTAS], "periods": periods}


def from_period(report: Mapping[str, Any], *, steps: tuple[int, int], deployment: Mapping[str, Any],
                answer: Mapping[str, Any] | None = None, extra: Mapping[str, Mapping[str, Any]] | None = None,
                source: str = "real", product: str = "onsite") -> dict[str, Any]:
    """The record of one runner report (``verity/all-protocols-period/v1`` or ``-window/v1``) whose loop served ``steps``
    ``[lo, hi)`` on its node, for ``product``. ``answer``: the product's Lean check's (`CHECKS`) answer on the block (the
    ``answer.json`` its check wrote), None when it didn't run. ``extra``: legs the runner doesn't run, by leg name, each
    ``{row, server, runs, source}`` with ``row`` the leg's document (``{answer, ...}``). A leg the product drops that ran
    is a ValueError."""
    if report.get("format") not in EXTENDS:
        raise ValueError(f"{report.get('format')}: not one of {EXTENDS}")
    lo, hi = steps
    if not 0 <= lo < hi:
        raise ValueError(f"steps [{lo}, {hi}): not a nonempty range")
    server = report["node"]
    legs: dict[str, Any] = {}
    for name, key in RUNNER_KEYS.items():
        row = report["protocols"].get(key)
        legs[name] = None if row is None else leg(name, row, server=server, runs=row.get("runs") or [report["run"]],
                                                  source=source)
    legs["wipe"] = None
    for name, x in (extra or {}).items():
        legs[name] = leg(name, x["row"], server=x["server"], runs=x["runs"], source=x.get("source", "real"))
    if ran := dropped_ran(legs, product):
        raise ValueError(f"{', '.join(ran)}: legs the {product} product drops, which this block ran")
    c = composed(server, legs, answer, product)
    which = {"period": report["period"]} if "period" in report else {"window": report["window"]}
    return {"format": FORMAT, "extends": report["format"], "run": report["run"], **which, "server": server,
            "server_time": [{"server": server, "steps": [lo, hi]}], "legs": legs, "composed": c,
            "monitoring_period": monitoring_period(deployment, frac(c["joint_horizon"]["error_sum"]))}


def check(record: Mapping[str, Any]) -> dict[str, str]:
    """Every derived field of a record recomputed from its legs and its deployment: ``{key: why}``, empty when it holds."""
    bad: dict[str, str] = {}
    if record.get("format") != FORMAT:
        return {"format": f"not {FORMAT}"}
    server = record["server"]
    if record["server_time"] != [{"server": server, "steps": record["server_time"][0]["steps"]}] or not (
            0 <= record["server_time"][0]["steps"][0] < record["server_time"][0]["steps"][1]):
        bad["server_time"] = "not one nonempty block [lo, hi) of the record's server"
    if sorted(record["legs"]) != sorted(LEGS):
        bad["legs"] = f"not the legs {LEGS}"
        return bad
    for name, lg in record["legs"].items():
        if lg is None:
            continue
        key = f"leg:{name}"
        if lg["source"] not in ("real", "fixture"):
            bad[key] = "its source is neither real nor fixture"
        if lg["at"] != at_deltas(name, lg["quantity"], frac(lg["error"])):
            bad[key] = "its certificates at the deltas are not its quantity's at its error"
        if "named_terms" not in lg:
            if name in NAMES_TERMS:
                bad[key] = f"it names no term its error leaves out; {NAMES_TERMS_FIX}, and the record is built from that"
        elif why := named_terms_bad(lg["named_terms"]):
            bad[key] = f"its named_terms: {why}"
    product = record["composed"].get("product", "onsite")
    if product not in PRODUCTS:
        bad["composed:product"] = f"not one of {sorted(PRODUCTS)}"
        return bad
    if ran := dropped_ran(record["legs"], product):
        bad["legs"] = f"{', '.join(ran)}: legs the {product} product drops, which the block ran"
    try:
        want = composed(server, record["legs"], record["composed"].get("answer"), product)
    except ValueError as e:
        bad["composed:answer"] = str(e)
        return bad
    got = record["composed"]
    for k in want:
        if got.get(k) != want[k]:
            bad[f"composed:{k}"] = "not its answer's verdict" if k == "verdict" else "not what its legs and answer give"
    if record["monitoring_period"] != monitoring_period(record["monitoring_period"]["deployment"],
                                                        frac(want["joint_horizon"]["error_sum"])):
        bad["monitoring_period"] = "not the deployment's periods at the block's error sum"
    return bad


# --- the records from the runs' outputs ----------------------------------------------------------------------------------

#: `benchmarks/census/deployments.json`'s reference deployment as the runs' monitoring reports recorded it
REFERENCE = {"id": "dc-12500-machines-600s", "machines": 12500, "gpus_per_machine": 8, "window_s": 600, "role": "reference",
             "label": "reference datacenter: 12,500 eight-GPU machines (100,000 GPUs), every machine audited in 10-minute windows"}


def from_runs(r2: Path, r1: Path, wipe: Path) -> dict[str, dict[str, Any]]:
    """The fixtures from the runs' outputs: period 0 of one all-protocols run (``r2``: its ``reports.jsonl``), window 10 of
    another (``r1``) and a wipe run (``wipe``: its ``results.json``, with ``leg/wipe.json`` beside it). Each loop served 2
    requests for 2 steps: ``[0, 2)``. The wipe's server is its session report's host, the hostname the runner names its
    node by (its ``placement`` names the node's alias)."""
    p0 = next(json.loads(x) for x in r2.read_text().splitlines() if json.loads(x)["period"] == 0)
    w10 = next(json.loads(x) for x in r1.read_text().splitlines() if json.loads(x)["window"] == 10)
    row = json.loads((wipe.parent / "leg" / "wipe.json").read_text())
    w = {"row": row, "server": row["report"]["host"], "runs": ["r20261025-224222-91e2"]}
    return {"block-record-r2-period0": from_period(p0, steps=(0, 2), deployment=REFERENCE, extra={"wipe": w}),
            "block-record-r1-window10": from_period(w10, steps=(0, 2), deployment=REFERENCE, extra={"wipe": w})}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The block record: check one, or build the fixture records from the runs' outputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("record")
    t = sub.add_parser("from-runs")
    t.add_argument("--r2", required=True, type=Path)
    t.add_argument("--r1", required=True, type=Path)
    t.add_argument("--wipe", required=True, type=Path)
    t.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        bad = check(json.loads(Path(a.record).read_text()))
        print(json.dumps({"ok": not bad, "bad": bad}, sort_keys=True))
        return 0 if not bad else 1
    a.out.mkdir(parents=True, exist_ok=True)
    for name, rec in from_runs(a.r2, a.r1, a.wipe).items():
        (a.out / f"{name}.json").write_text(json.dumps(rec, separators=(",", ":"), sort_keys=True, ensure_ascii=False) + "\n")
        print(a.out / f"{name}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
