"""onsite-record: the onsite demo's record of one served run (``record/onsite.json``, `FORMAT`): every verdict its run of
record needs, the network certifier's bound term by term, and that bound over each monitoring period for the serving
node, a datacenter and the world. It reads only the run's public $OUT and decides nothing:

    python benchmarks/one_stage/onsite_record.py OUT [--advice-charge FILE] [--to FILE]

`served_verdicts.sh` runs it after its composite step on a run with the network certifier's piece, and its one
``ONSITE`` line goes to the run's verdicts.txt.

**Verdicts** (`VERDICTS`), each ``{protocol, verdict, from}`` with the answer's ``reason`` or ``refusal``, or, where the
run wrote no answer, ``missing`` and ``why``:

* sampled proofs and PoUW: ``record/verdicts.json``'s ``integrity`` and ``work`` rows (`served_verdicts.py record`);
* inference-only: its case's outcome in remote's composite;
* remote: remote's Lean verdict, ``composite/remote/composite.json`` (`composite.py`), with what it couldn't establish;
* the network certifier: the ``communication`` row, `NetworkCheck.Period`'s answer (`NETWORK`);
* onsite: the Lean composition of remote's three subprotocols and the network certifier over one registration, the
  composite whose product is `ONSITE`, missing where the run has none.

``of_record`` holds when every one of them accepted, the condition for the demo's run of record; ``placement`` is the
network certifier's answer's, which host ran each party.

**The bound** is the network certifier's accepted answer's public in its rows form: K and each of its terms in the
notation table's symbols (`TERMS`, `PHASE_TERMS`), each naming the public fields it sums (``lean``): k_sched
(``schedules``), k_fail (``cells``) and, for each execution phase j of the certification period (the code's link-window,
the public's ``scans[j]``), ℓ_{a,j} (``free``), k_{skip,j} (``faulty_sets``, ``faulty_bits`` and ``faulty_entries``) and
the jitter T_j·log2 d_jit, T_j the phase's buckets (``jitter``). They sum to K, by which `NetTiming.Guarantees.PeriodLeakAccepted` bounds what the serving node
sends past its committed program except with δ_net over ``period_s`` seconds, the run's m execution phases of T buckets
of τ (the certifier's case's ``schedule``). ``delta_net`` gives it term by term, each with its source: the answer's
``eps``, only its δ part (m · δ and each charged audit's δ, `EPS`), and m times each phase's knowledge, binding and
collision errors ε_ks, ε_bind and ε_tap (`ERRORS`), which no check computes, so δ_net has no number here (`UNVALUED`).
K is that statement's count as proved, a known overcount (``notes``, `OVERCOUNT`); ``connections`` is how many
connections k_sched and k_fail count, every one the schedule can carry (`CONNECTIONS`), its ``error`` where the case's
caps aren't those; and ``verdict_line`` is the run's published line of the network certifier's verdict (`LINE` in
verdicts.txt), None where it has none. A public whose terms don't sum to K, or that isn't the rows form, is the bound's
``error``. Beside the bound, ``advice_charge``: the served run's realized advice charge stays out of the record
(`ADVICE`, which says what hides it and what doesn't yet); ``registered`` is ℓ_{a,j}, the registered width each
execution phase's term charges, an upper bound on the honest charge (`REGISTERED`), with its share of K; and
``measured`` is honest serving's |a_j| a Commit period, of which an execution phase holds several, and their sum over
each execution phase (``bits_per_phase``, missing where the test run's advice places no Commit period on the engine
clock), measured on a test run with public inputs (`advice_charge.py`'s record, ``--advice-charge``) and labelled as
that run's, never a figure of this one. ``tree`` is the commit the run served from (machine.txt's), None where machine.txt names none.

**Periods**: the run's own certification period (``run``), proved, and each monitoring period of `leak_bounds.PERIODS`
for the serving node, the census's reference datacenter and the world. A node's cell is proved where the run's period
covers the monitoring period's seconds (its m execution phases), and otherwise scaled from the run to the period; a
datacenter's and the world's are scaled from the node's. Each scaled cell names its rule, which ``rules`` states with
its sources (`RULES`): the bound, the case's schedule, and each census line it reads with its SHA-512 (`census`). Each
cell gives K in bits (the least integer at or above a scaled one), its share of one copy of the weights
(`leak_bounds.WEIGHTS`), its δ_net's terms over the execution phases it counts (``delta_net``), whether eps alone is
already vacuous (at least 1) and the statement it rests on; beside that pooled total, ``per_phase`` is the same over
one execution phase, proved only where m is 1.

The record's counts and indices are the certification period's (m and each phase j), public in the Lean answer's
profile; it names no Commit window, whose count `publish` holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "network_traces") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "network_traces"))
import advice_charge as AC  # noqa: E402
import leak_bounds as LB  # noqa: E402

FORMAT = "verity/onsite-network-record/v1"
EXTENDS = "verity/all-protocols-window/v1"
NETWORK = "verity/certifier/v0"
REMOTE = "verity/remote/v0"
INFERENCE_ONLY = "verity/inference-only/v0"
#: the Lean composition of remote's three subprotocols and the network certifier over one registration, as its
#: composite.json names it (`composite.py`'s remote-network product)
ONSITE = "verity/remote-network/v0"
VERDICTS = ("sampled proofs", "PoUW", "inference-only", "remote", "the network certifier", "onsite")
#: the window report's rows, by verdict
ROWS = {"sampled proofs": "integrity", "PoUW": "work", "the network certifier": "communication"}
#: where the certifier piece leaves its period's case (`certifier_piece.CASE`), whose params carry the schedule and caps
CASE = Path("pieces") / "certifier" / "case.json"
#: K's terms over the period, each a symbol of the notation table and the rows form's public fields it sums
TERMS = {"k_{sched}": ("schedules",), "k_{fail}": ("cells",)}
#: K's terms of each execution phase j, each summing its scan's public fields
PHASE_TERMS = {"\\ell_{a,j}": ("free",), "k_{skip,j}": ("faulty_sets", "faulty_bits", "faulty_entries"),
               "T_j \\log_2 d_{jit}": ("jitter",)}
ADVICE = ("not in this record: the realized advice length |a_j| is a value. The run's published registration copies state "
          "the compiler's cap in its place (`served_verdicts.py public_registration`), but each window's advice is "
          "committed and compiled at its own length, so |a_j| is hidden only once it is padded to ℓ_a before it is "
          "committed; the bound charges ℓ_{a,j}, the registered width")
OVERCOUNT = ("a known overcount, proved and larger than necessary: k_sched and k_fail charge every connection the "
             "schedule can carry (`Schedule.caps`), whatever the traffic, and each connection's cell count twice, in "
             "k_sched (its lane's length, #S_s ≤ ∏_c (cap_c + 1)) and in k_fail (its fail cell's position), which in a "
             "run coincide; K is PeriodLeakAccepted's count as it states it")
CONNECTIONS = ("the caps of the certifier's case's params (``cells``), one a connection: the period's check refuses any "
               "but its schedule's (`NetworkCheck.Period.Schedule.caps`), egress_cells for each of the lanes · m · T "
               "connections the network certifier can carry over the period's m execution phases, so this counts the "
               "connections the schedule can carry, not those served")
EPS = {"is": "δ_net's δ part alone: m · δ and each charged audit's δ",
       "source": "the network-period check's public eps (NetworkCheck.Period)"}
#: δ_net's terms that eps doesn't count, each once an execution phase: `PeriodLeakAccepted` bounds each phase's audit's
#: failure by δ + ε_ks + ε_bind + ε_tap
ERRORS = {
    "ε_ks": {"is": "the knowledge error, κ(δ) − δ + √(γ·Pr[B])",
             "source": "Claims.Sampling.KS at the deployment, PeriodLeakAccepted's hypothesis"},
    "ε_bind": {"is": "the binding error, Pr[B_ext]: an accepted extraction that disagrees with the committed trace",
               "source": "Claims.Sampling.Bext, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
    "ε_tap": {"is": "Pr[B_tap]: the extracted trace's openings colliding with the network certifier's",
              "source": "NetTiming.Btap, a SHA-512 collision in the run (CommitmentScheme.Binds)"},
}
UNVALUED = ("δ_net is eps plus each execution phase's ε_ks, ε_bind and ε_tap, never eps alone; no check computes those "
            "three, so δ_net has no number here")
#: the run's published line of the network certifier's verdict (`served_verdicts.pieces`)
LINE = "THE NETWORK CERTIFIER "
REGISTERED = "registered width, an upper bound on the honest charge"
#: each scaled cell's rule, by the name it gives; `rules` adds each one's sources
RULES = {
    "per-phase": "the run's K and eps over one of its m execution phases, K/m: their average, which no statement bounds a "
                 "phase by where m exceeds 1",
    "run-to-period": "the run's per execution phase times the period's execution phases, ⌈seconds / (T·τ)⌉, each as the "
                     "run's at its load: ℓ_{a,j}, k_{skip,j} and the jitter scale with the phases, and k_sched and "
                     "k_fail, ⌈log2(cap_c + 1)⌉ each a connection, with the connections, at the schedule's lanes · T a "
                     "phase",
    "datacenter": "the node's times the census's reference datacenter's machines, one network certifier on each "
                  "machine's uplink",
    "world": "the node's times the world's links: the adversary resource model's compute over its monitoring period, at "
             "an H100-equivalent's FLOP/s, in machines of the reference datacenter's GPUs, rounded up, one network "
             "certifier on each",
}
FROM_BOUND = "bound: the run's K and eps over its m execution phases"
FROM_SCHEDULE = f"{CASE} params.schedule: an execution phase's T buckets of τ"
STATEMENTS = {"run": LB.STATEMENTS["per_period"], "node": LB.STATEMENTS["per_period"],
              "datacenter": LB.STATEMENTS["datacenter"], "world": LB.STATEMENTS["datacenter"]}
NO_MEASUREMENT = ("no test run's measurement given (--advice-charge: `advice_charge.py`'s record of honest serving on "
                  "public inputs)")


def missing(why: str, src: str | None = None) -> dict[str, Any]:
    return {"verdict": "missing", "why": why, **({"from": src} if src else {})}


def row_verdict(rows: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A verdict from the window report's row ``key``: a missing, dropped or development row is missing, with its why."""
    src = f"record/verdicts.json protocols.{key}"
    r = rows.get(key)
    if not isinstance(r, Mapping):
        return missing(f"the window report has no {key} row", src)
    if r.get("verdict") in ("missing", "dropped"):
        return missing(str(r.get("why")), src)
    out = {"protocol": r.get("protocol"), "verdict": r.get("verdict"), "from": src}
    return out | ({"reason": r["reason"]} if r.get("reason") else {})


def composites(out: Path) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each composite.json under OUT/composite, by the product it names: its path in OUT and its document."""
    found: dict[str, tuple[str, dict[str, Any]]] = {}
    for f in sorted((out / "composite").glob("*/composite.json")):
        doc = json.loads(f.read_text())
        found[doc.get("product")] = (str(f.relative_to(out)), doc)
    return found


def outcome(o: Mapping[str, Any], protocol: str, src: str) -> dict[str, Any]:
    """`composite.py`'s public outcome as a verdict: ``none`` where Lean gave no answer."""
    return {"protocol": protocol, "verdict": o.get("verdict") or "none", "from": src,
            **({"refusal": o["refusal"]} if o.get("refusal") else {})}


def composite_verdict(found: Mapping[str, tuple[str, dict[str, Any]]], product: str, none: str) -> dict[str, Any]:
    """The Lean verdict of ``product``'s composite, with the subprotocols it couldn't establish; missing (``none``) where
    the run has no such composite."""
    if product not in found:
        return missing(none)
    src, doc = found[product]
    if doc.get("verdict") is None:
        return missing(f"its composite ran no Lean check ({src})", src)
    return outcome(doc["verdict"], product, src) | (
        {"not_established": list(doc["missing"])} if doc.get("missing") else {})


def case_verdict(found: Mapping[str, tuple[str, dict[str, Any]]], product: str, case: str) -> dict[str, Any]:
    """The outcome of ``case``'s own answer in ``product``'s composite."""
    if product not in found:
        return missing(f"its case is read from {product}'s composite, and the run has none")
    src, doc = found[product]
    for c in doc.get("cases") or ():
        if c.get("id") == case:
            return outcome(c, case, f"{src} cases")
    return missing(f"{product}'s composite has no {case} case", src)


def bound(public: Mapping[str, Any]) -> dict[str, Any]:
    """K and its terms (module docstring) from a rows-form public; ValueError naming why for any other. m is its scans'
    count, held to its ``windows`` where the public keeps it (a run's published form drops it)."""
    if not isinstance(public, Mapping) or not isinstance(public.get("scans"), list):
        raise ValueError("not NetworkCheck.Period's rows form, whose public has scans: the demo's network verdict is "
                         "the rows form's")
    try:
        terms = [{"symbol": s, "bits": sum(int(public[f]) for f in fs), "lean": list(fs)} for s, fs in TERMS.items()]
        for i, scan in enumerate(public["scans"]):
            terms += [{"symbol": s, "phase": i, "bits": sum(int(scan[f]) for f in fs),
                       "lean": [f"scans[{i}].{f}" for f in fs]} for s, fs in PHASE_TERMS.items()]
        K, eps = int(public["K"]), Fraction(public["eps"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"the rows form's public lacks a term: {type(e).__name__}: {e}") from None
    if sum(t["bits"] for t in terms) != K:
        raise ValueError(f"K is {K}, and its terms sum to {sum(t['bits'] for t in terms)}")
    m = len(public["scans"])
    if m == 0:
        raise ValueError("the rows form's public has no scan: a period of no execution phase, which bounds nothing")
    if "windows" in public and int(public["windows"]) != m:
        raise ValueError(f"{m} scans for the period's {public['windows']} execution phases")
    sources = [EPS, *ERRORS.values()]
    return {"statement": LB.STATEMENTS["per_period"], "K": K, "phases": m, "notes": [OVERCOUNT], "terms": terms,
            "delta_net": {"terms": [t | s for t, s in zip(delta_net(eps, m), sources)], "note": UNVALUED}}


def delta_net(eps: Fraction, phases: int) -> list[dict[str, Any]]:
    """δ_net over ``phases`` execution phases, term by term: eps, and each of `ERRORS` that many times."""
    return [{"symbol": "eps", "value": str(eps)}, *({"symbol": s, "times": phases} for s in ERRORS)]


def eps_of(b: Mapping[str, Any]) -> Fraction:
    """The bound's eps, its δ_net's first term."""
    return Fraction(b["delta_net"]["terms"][0]["value"])


def schedule(case: Mapping[str, Any], phases: int) -> tuple[Fraction, int]:
    """An execution phase's seconds, T buckets of τ (the case's ``schedule``), and the connections its caps count
    (``cells``), the case's period held to the answer's phases where its params keep ``windows`` and its caps to its
    schedule's over them (`CONNECTIONS`)."""
    params = case["params"]
    if "windows" in params and int(params["windows"]) != phases:
        raise ValueError(f"the case's period has {params['windows']} execution phases, and its answer {phases}")
    s, caps = params["schedule"], params["cells"]
    if not isinstance(caps, list):
        raise TypeError(f"the case's cells are its caps, a list, not {type(caps).__name__}")
    n = int(s["lanes"]) * phases * int(s["window"])
    if caps != [int(s["egress_cells"])] * n:
        raise ValueError(f"the case's {len(caps)} caps aren't its schedule's (Schedule.caps: egress_cells = "
                         f"{s['egress_cells']} for each of its lanes · m · T = {n} connections), which the period's "
                         "check refuses")
    return Fraction(int(s["window"]) * int(s["bucket_ms"]), 1000), n


def census_line(name: str, *fields: str) -> dict[str, Any]:
    """The census's reference line in ``name``: its file, its id, the ``fields`` a rule reads, and its version, the
    SHA-512 of its canonical JSON."""
    d, = (d for d in json.loads((LB.CENSUS / name).read_text()) if d["role"] == "reference")
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {"file": str((LB.CENSUS / name).relative_to(HERE.parents[1])), "id": d["id"],
            "read": {f: d[f] for f in fields}, "sha512": hashlib.sha512(canonical).hexdigest()}


def census() -> dict[str, Any]:
    """The links the datacenter's and the world's cells multiply by (`leak_bounds.datacenter_links`,
    `leak_bounds.world_links`), each with the census lines it reads."""
    machines = census_line("deployments.json", "machines")
    world = [census_line("adversary.json", "total_flop", "flop_per_s_per_h100", "monitoring_period_s"),
             census_line("deployments.json", "gpus_per_machine")]
    return {"datacenter": {"links": LB.datacenter_links(), "sources": [machines]},
            "world": {"links": LB.world_links(), "sources": world}}


def cell(bits: Fraction, eps: Fraction, phases: int, where: str, rule: str | None = None) -> dict[str, Any]:
    """A cell over ``phases`` execution phases, proved, or scaled by ``rule`` (`RULES`)."""
    return {"bits": math.ceil(bits), "share": float(bits / LB.WEIGHTS.bits), "delta_net": delta_net(eps, phases),
            "vacuous": eps >= 1,
            "statement": STATEMENTS[where]} | ({"how": "scaled", "rule": rule} if rule else {"how": "proved"})


def periods(K: int, eps: Fraction, m: int, phase_s: Fraction, connections: int) -> dict[str, Any]:
    """The run's own period, proved, and each monitoring period's node, proved where the run's period covers it and
    else scaled, datacenter and world, scaled from the node's, each beside its figure over one execution phase (module
    docstring)."""
    refs = census()
    links = {"node": 1, **{k: v["links"] for k, v in refs.items()}}
    one, one_eps, period_s = Fraction(K, m), eps / m, m * phase_s
    per_phase = cell(one, one_eps, 1, "run", None if m == 1 else "per-phase")
    run = cell(Fraction(K), eps, m, "run") | {"seconds": str(period_s), "phases": m, "connections": connections,
                                              "per_phase": per_phase}
    sources = {"per-phase": [FROM_BOUND], "run-to-period": [FROM_BOUND, FROM_SCHEDULE],
               **{k: v["sources"] for k, v in refs.items()}}
    out: dict[str, Any] = {"phase_s": str(phase_s), "weights": LB.WEIGHTS.label, "run": run,
                           "rules": {k: {"rule": text, "sources": sources[k]} for k, text in RULES.items()}}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase_s)
        covered = period_s >= seconds
        node = m if covered else w
        out[name] = {"seconds": seconds, "phases": w}
        for where, n in links.items():
            rule = where if where != "node" else None if covered else "run-to-period"
            params = {"phases": node, "phase_s": str(phase_s), "links": n,
                      "connections": str(Fraction(connections, m) * node * n)}
            out[name][where] = cell(one * node * n, one_eps * node * n, node * n, where, rule) | {
                "parameters": params,
                "per_phase": per_phase if where == "node" else cell(one * n, one_eps * n, n, where, where)}
    return out


def measured(path: Path | None) -> dict[str, Any]:
    """Honest serving's advice charge from a test run's `advice_charge.py` record, labelled as that run's."""
    if path is None:
        return {"missing": NO_MEASUREMENT}
    doc = json.loads(path.read_text())
    if doc.get("format") != AC.FORMAT:
        raise ValueError(f"{path} is {doc.get('format')!r}, not {AC.FORMAT}")
    return {"label": doc["label"], "run": doc["run"], "from": str(path), "bits_per_commit_period": doc["bits"],
            "bits_per_phase": doc["phase_bits"] or doc["by_phase"],
            "commit_periods_measured": len(doc["commit_periods"]), "finishes": doc["finishes"],
            "prompt_len": doc["prompt_len"], "output_len": doc["output_len"]}


def lines(out: Path, name: str) -> list[str]:
    try:
        return (out / name).read_text().splitlines()
    except FileNotFoundError:
        return []


def registered(b: Mapping[str, Any]) -> dict[str, Any]:
    """ℓ_{a,j}, each execution phase's registered width, and their share of K."""
    free = [t["bits"] for t in b["terms"] if t["symbol"] == "\\ell_{a,j}"]
    return {"label": REGISTERED, "bits_per_phase": free, "share": float(Fraction(sum(free), b["K"])),
            "lean": "scans[i].free"}


def record(out: Path, advice_charge: Path | None = None) -> dict[str, Any]:
    """OUT's onsite record (module docstring); FileNotFoundError where `served_verdicts.py record` wrote no report."""
    report = json.loads((out / "record" / "verdicts.json").read_text())
    if report.get("format") != EXTENDS:
        raise ValueError(f"record/verdicts.json is {report.get('format')!r}, not {EXTENDS}")
    advice = {"served": ADVICE, "registered": {"missing": "no bound"}, "measured": measured(advice_charge)}
    rows = report.get("protocols") or {}
    found = composites(out)
    v = {name: row_verdict(rows, ROWS[name]) for name in ("sampled proofs", "PoUW")}
    v["inference-only"] = case_verdict(found, REMOTE, INFERENCE_ONLY)
    v["remote"] = composite_verdict(found, REMOTE, "remote's composite didn't run: the run has no "
                                                   "composite/remote/composite.json")
    v["the network certifier"] = row_verdict(rows, ROWS["the network certifier"])
    v["onsite"] = composite_verdict(found, ONSITE, f"the run has no composite of {ONSITE}")
    v = {k: v[k] for k in VERDICTS}
    net = rows.get(ROWS["the network certifier"]) or {}
    answer = net.get("answer") if isinstance(net.get("answer"), Mapping) else {}
    profile = answer.get("profile") if isinstance(answer.get("profile"), Mapping) else {}
    tree = next((x for x in lines(out, "machine.txt") if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", x)), None)
    rec: dict[str, Any] = {
        "format": FORMAT, "extends": EXTENDS, "run": report.get("run"), "tree": tree, "node": report.get("node"),
        "verdicts": v,
        "of_record": {"holds": all(x["verdict"] == "accept" for x in v.values()),
                      "missing": [k for k, x in v.items() if x["verdict"] == "missing"],
                      "not_accepting": [k for k, x in v.items() if x["verdict"] not in ("accept", "missing")]},
        "placement": profile.get("placement"), "advice_charge": advice}
    nv = v["the network certifier"]
    if nv["verdict"] != "accept":
        why = (f"the network certifier's verdict is {nv['verdict']}"
               + (f" ({nv['why']})" if nv.get("why") else "") + ": only its accepted answer certifies K")
        return rec | {"bound": {"missing": why}, "periods": {"missing": "no bound"}}
    if nv.get("protocol") != NETWORK:
        return rec | {"bound": {"error": f"the network certifier's answer is {nv.get('protocol')}'s, not {NETWORK}'s"},
                      "periods": {"missing": "no bound"}}
    try:
        b = rec["bound"] = bound(answer.get("public"))
    except ValueError as e:
        return rec | {"bound": {"error": str(e)}, "periods": {"missing": "no bound"}}
    b["verdict_line"] = next((x for x in reversed(lines(out, "verdicts.txt")) if x.startswith(LINE)), None)
    advice["registered"] = registered(b)
    b["period_s"] = None
    try:
        phase_s, connections = schedule(json.loads((out / CASE).read_text()), b["phases"])
    except FileNotFoundError:
        b["connections"] = {"missing": f"no {CASE}: they are its params' caps"}
        return rec | {"periods": {"missing": f"no {CASE}: the execution phase's length is its params' schedule"}}
    except (KeyError, TypeError, ValueError) as e:
        b["connections"] = {"error": f"{CASE}: {type(e).__name__}: {e}"}
        return rec | {"periods": {"error": f"{CASE}: {type(e).__name__}: {e}"}}
    b["connections"] = {"count": connections, "from": CONNECTIONS}
    b["period_s"] = str(b["phases"] * phase_s)
    return rec | {"periods": periods(b["K"], eps_of(b), b["phases"], phase_s, connections)}


def summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ONSITE`` line: each verdict, whether the run is of record, K, its δ_net's terms and connections, the
    year's datacenter cell, and the measured advice charge's test run."""
    b, p = rec.get("bound") or {}, rec.get("periods") or {}
    year = (p.get("year") or {}).get("datacenter")
    m = rec["advice_charge"]["measured"]
    return {"verdicts": {k: x["verdict"] for k, x in rec["verdicts"].items()}, "of_record": rec["of_record"]["holds"],
            "K": b.get("K"), "delta_net": None if "delta_net" not in b else delta_net(eps_of(b), b["phases"]),
            "connections": (b.get("connections") or {}).get("count"),
            "year_datacenter": None if year is None else {k: year[k] for k in ("bits", "share", "delta_net", "how")},
            "advice_measured_on": m.get("run"), **{k: b[k] for k in ("missing", "error") if k in b}}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="the served run's public $OUT")
    ap.add_argument("--advice-charge", type=Path,
                    help="a test run's advice-charge.json (advice_charge.py), stated beside the bound as that run's")
    ap.add_argument("--to", type=Path, help="the record's path (default OUT/record/onsite.json)")
    a = ap.parse_args(argv)
    try:
        rec = record(a.out, a.advice_charge)
    except FileNotFoundError as e:
        print(f"ONSITE: not written: {e.filename} is missing", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ONSITE: not written: {e}", file=sys.stderr)
        return 1
    dest = a.to or a.out / "record" / "onsite.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rec, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print("ONSITE " + json.dumps(summary(rec), sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
