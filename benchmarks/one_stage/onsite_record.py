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
(``schedules``), k_fail (``cells``) and, for each execution phase i of the certification period (the code's link-window,
the public's ``scans[i]``), ℓ_{a,i} (``free``), k_{skip,i} (``faulty_sets``, ``faulty_bits`` and ``faulty_entries``) and
the jitter (``jitter``). They sum to K, by which `NetTiming.Guarantees.PeriodLeakAccepted` bounds what the serving node
sends past its committed program except with δ_net, the answer's ``eps`` (`DELTA_NET_BESIDE`). K is that statement's
count as proved, a known overcount (`OVERCOUNT`); ``connections`` is how many connections k_sched and k_fail count
(`CONNECTIONS`). A public whose terms don't sum to K, or that isn't the rows form, is the bound's ``error``. Beside the
bound, ``advice_charge``: the served run's realized advice charge is private (`ADVICE`), and ``measured`` is honest
serving's, measured on a test run with public inputs (`advice_charge.py`'s record, ``--advice-charge``) and labelled as
that run's, never a figure of this one.

**Periods**: the run's own certification period (``run``: its m execution phases, each T buckets of τ, from the
certifier's case's ``schedule``), marked ``proved``, and each monitoring period of `leak_bounds.PERIODS` for the serving
node, the census's reference datacenter and the world, each marked ``scaled`` and naming its rule (`RULES`) and its
parameters. Each cell gives K in bits (the least integer at or above a scaled one), its share of one copy of the weights
(`leak_bounds.WEIGHTS`), its ε, whether that ε is vacuous (at least 1) and the statement it rests on; beside that pooled
total, ``per_phase`` is the same over one execution phase.

The record's counts and indices are the certification period's (m and each phase i), public in the Lean answer's
profile; it names no Commit period.
"""
from __future__ import annotations

import argparse
import json
import math
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
#: K's terms of each execution phase i, each summing its scan's public fields
PHASE_TERMS = {"\\ell_{a,i}": ("free",), "k_{skip,i}": ("faulty_sets", "faulty_bits", "faulty_entries"),
               "k_i \\log_2 d_{jit}": ("jitter",)}
ADVICE = ("private: the realized advice length |a_j| is a value, so the run's published registrations carry the advice "
          "padded to its compiler's cap, and the bound ℓ_{a,i}, the registered width")
OVERCOUNT = ("a known overcount, proved and larger than necessary: each connection's cell count is charged twice, in "
             "k_sched (its lane's length, #S_s ≤ ∏_c (cap_c + 1)) and in k_fail (its fail cell's position), which in a "
             "run coincide; K is PeriodLeakAccepted's count as it states it")
CONNECTIONS = ("the caps of the certifier's case's params (``cells``), one a connection: the network certifier holds each "
               "lane to its schedule's egress_cells, and the period's check reads the caps as given")
DELTA_NET_BESIDE = "the scans' knowledge and collision errors, which eps doesn't count"
RULES = {
    "node": "the run's per execution phase times the period's execution phases, ⌈seconds / (T·τ)⌉, each as the run's at "
            "its load: ℓ_{a,i}, k_{skip,i} and the jitter scale with the phases, and k_sched and k_fail, "
            "⌈log2(cap_c + 1)⌉ each a connection, with the connections, at the run's connections a phase",
    "datacenter": "the node's times the census's reference datacenter's machines, one network certifier on each "
                  "machine's uplink",
    "world": "the node's times the world's links: the adversary resource model's compute over its monitoring period, at "
             "an H100-equivalent's FLOP/s, in machines of the reference datacenter's GPUs, rounded up, one network "
             "certifier on each",
}
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
    if "windows" in public and int(public["windows"]) != m:
        raise ValueError(f"{m} scans for the period's {public['windows']} execution phases")
    return {"statement": LB.STATEMENTS["per_period"], "K": K, "phases": m, "delta_net": str(eps),
            "delta_net_beside": DELTA_NET_BESIDE, "overcount": OVERCOUNT, "terms": terms}


def schedule(case: Mapping[str, Any], phases: int) -> tuple[Fraction, int]:
    """An execution phase's seconds, T buckets of τ (the case's ``schedule``), and the connections its caps count
    (``cells``), the case's period held to the answer's phases where its params keep ``windows``."""
    params = case["params"]
    if "windows" in params and int(params["windows"]) != phases:
        raise ValueError(f"the case's period has {params['windows']} execution phases, and its answer {phases}")
    s, caps = params["schedule"], params["cells"]
    if not isinstance(caps, list):
        raise TypeError(f"the case's cells are its caps, a list, not {type(caps).__name__}")
    return Fraction(int(s["window"]) * int(s["bucket_ms"]), 1000), len(caps)


def census() -> dict[str, Any]:
    """The census references the scaled cells read: the reference datacenter's machines and the world's links
    (`leak_bounds.datacenter_links`, `leak_bounds.world_links`), each with its census id."""
    dc, = (d for d in json.loads((LB.CENSUS / "deployments.json").read_text()) if d["role"] == "reference")
    world, = (d for d in json.loads((LB.CENSUS / "adversary.json").read_text()) if d["role"] == "reference")
    return {"datacenter": {"census": dc["id"], "links": LB.datacenter_links()},
            "world": {"census": [world["id"], dc["id"]], "links": LB.world_links()}}


def cell(bits: Fraction, eps: Fraction, how: str, where: str) -> dict[str, Any]:
    return {"bits": math.ceil(bits), "share": float(bits / LB.WEIGHTS.bits), "eps": str(eps), "vacuous": eps >= 1,
            "how": how, "statement": STATEMENTS[where]}


def periods(K: int, eps: Fraction, m: int, phase_s: Fraction, connections: int) -> dict[str, Any]:
    """The run's own period, proved, and each monitoring period's node, datacenter and world, scaled, each beside its
    figure over one execution phase (module docstring)."""
    refs = census()
    links = {"node": 1, **{k: v["links"] for k, v in refs.items()}}
    one, one_eps = Fraction(K, m), eps / m
    run = cell(Fraction(K), eps, "proved", "run") | {
        "seconds": str(m * phase_s), "phases": m, "connections": connections,
        "per_phase": cell(one, one_eps, "proved" if m == 1 else "scaled", "run")}
    out: dict[str, Any] = {"phase_s": str(phase_s), "weights": LB.WEIGHTS.label, "rules": RULES, "run": run}
    for name, seconds in LB.PERIODS.items():
        w = math.ceil(Fraction(seconds) / phase_s)
        out[name] = {"seconds": seconds, "phases": w}
        for where, n in links.items():
            params = {"phases": w, "phase_s": str(phase_s), "connections": str(Fraction(connections, m) * w * n),
                      "links": n, **({"census": refs[where]["census"]} if where in refs else {})}
            out[name][where] = cell(one * w * n, one_eps * w * n, "scaled", where) | {
                "rule": where, "parameters": params, "per_phase": cell(one * n, one_eps * n, "scaled", where)}
    return out


def measured(path: Path | None) -> dict[str, Any]:
    """Honest serving's advice charge from a test run's `advice_charge.py` record, labelled as that run's."""
    if path is None:
        return {"missing": NO_MEASUREMENT}
    doc = json.loads(path.read_text())
    if doc.get("format") != AC.FORMAT:
        raise ValueError(f"{path} is {doc.get('format')!r}, not {AC.FORMAT}")
    return {"label": doc["label"], "run": doc["run"], "from": str(path), "bits_per_phase": doc["bits"],
            "phases_measured": len(doc["phases"]), "finishes": doc["finishes"], "prompt_len": doc["prompt_len"],
            "output_len": doc["output_len"]}


def record(out: Path, advice_charge: Path | None = None) -> dict[str, Any]:
    """OUT's onsite record (module docstring); FileNotFoundError where `served_verdicts.py record` wrote no report."""
    report = json.loads((out / "record" / "verdicts.json").read_text())
    if report.get("format") != EXTENDS:
        raise ValueError(f"record/verdicts.json is {report.get('format')!r}, not {EXTENDS}")
    advice = {"served": ADVICE, "measured": measured(advice_charge)}
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
    rec: dict[str, Any] = {
        "format": FORMAT, "extends": EXTENDS, "run": report.get("run"), "node": report.get("node"), "verdicts": v,
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
    try:
        phase_s, connections = schedule(json.loads((out / CASE).read_text()), b["phases"])
    except FileNotFoundError:
        b["connections"] = {"missing": f"no {CASE}: they are its params' caps"}
        return rec | {"periods": {"missing": f"no {CASE}: the execution phase's length is its params' schedule"}}
    except (KeyError, TypeError, ValueError) as e:
        b["connections"] = {"error": f"{CASE}: {type(e).__name__}: {e}"}
        return rec | {"periods": {"error": f"{CASE}: {type(e).__name__}: {e}"}}
    b["connections"] = {"count": connections, "from": CONNECTIONS}
    return rec | {"periods": periods(b["K"], Fraction(b["delta_net"]), b["phases"], phase_s, connections)}


def summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The ``ONSITE`` line: each verdict, whether the run is of record, K, its ε and connections, the year's datacenter
    cell, and the measured advice charge's test run."""
    b, p = rec.get("bound") or {}, rec.get("periods") or {}
    year = (p.get("year") or {}).get("datacenter")
    m = rec["advice_charge"]["measured"]
    return {"verdicts": {k: x["verdict"] for k, x in rec["verdicts"].items()}, "of_record": rec["of_record"]["holds"],
            "K": b.get("K"), "delta_net": b.get("delta_net"), "connections": (b.get("connections") or {}).get("count"),
            "year_datacenter": None if year is None else {k: year[k] for k in ("bits", "share", "eps", "how")},
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
