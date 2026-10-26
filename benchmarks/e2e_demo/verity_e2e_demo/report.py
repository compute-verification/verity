"""Demo A's report (CMP-98): one or more block records (`block_record`, ``verity/block-record/v1``) and a measurements
file rendered as Markdown in the structure Daniel set (25 Oct 6:56 PM PDT): 1. ran and proved, on one H200, one block;
2. measured on the demo, the inputs to the scaling claim; 3. computed from the same theorem and the measured costs, not
run; then what the demo didn't show, and every number's source.

**Numbers.** Every number is a placeholder ``⟨NAME⟩`` of ``demo_a.template`` (Markdown; not ``.md``, so the suite's
inputs, which leave Markdown out, cover it), filled from one of three places, each name from exactly one:

* `RECORD`: the first block record (the block the report is about), by the path each entry names;
* `MEASURED`: the measurements file (``verity/demo-a-measurements/v1``), each value with the run or field it came from;
* `DERIVED`: a formula over the other two, filled once all its inputs are.

A name with no value renders as ``⟨NAME⟩``, and the closing table says what fills it, so a missing input never shows as
a number. A value may carry a ``label`` (``PyTorch dry run``, ``HASH-20 bench``), printed beside it wherever it appears,
for a number that is not the demo block's own; ``--record-label`` labels every number read from the block records,
once at the top and in the sources table.

**Demo A** composes sampled proofs, PoUW and network accounting (`DEMO_A`): accept when all three accepted on the
block's server, reject when one rejected, pending otherwise, with their errors summed as
``Accounting.Guarantees.JointHorizon`` states it. PoUS, the bandwidth audit and the wipe compose at M4 (Demo B).

**The guarantee** cites LEAN-56's ``Flock.Guarantees.JointStratified`` when ``JOINT_LOCKED`` is true (locked by M3),
else #4358's per-subcircuit bound, with the joint figure labelled unlocked.

**Words.** Each protocol by its name, "sampling block" and "subcircuit": `render` refuses a report that says "leg",
"window" or "template" (`BANNED`), measurement text included.

The measurements file::

    {"format": "verity/demo-a-measurements/v1",
     "values": {"PROVING_GPU_S": {"value": 5400.0, "source": "PRF-171 run r..., sessions' GPU-seconds", "label": null}},
     "subcircuits": [{"name": "...", "units": 1024, "drawn": 64, "ands_per_unit": 1, "fixed_s": 30.0, "source": "..."}],
     "caveats": ["what was stubbed"]}

    python -m verity_e2e_demo.report RECORD.json [RECORD.json ...] [--measurements M.json] [--record-label TEXT] \
        [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from verity_e2e_demo import block_record

TEMPLATE = Path(__file__).parent / "demo_a.template"
MEASUREMENTS = "verity/demo-a-measurements/v1"
DELTA = Fraction(1, 100)
#: the share of compute spent on proving in part 3
BUDGET = Fraction(1, 100)
DEMO_A = ("integrity", "pouw", "certifier")
PROTOCOL_NAMES = {"integrity": "sampled proofs", "pouw": "PoUW", "certifier": "network accounting", "pous": "PoUS",
                  "bandwidth": "the bandwidth audit", "wipe": "the wipe"}
PERIODS = block_record.PERIODS
BANNED = re.compile(r"\b(legs?|windows?|templates?)\b", re.IGNORECASE)
PLACEHOLDER = re.compile(r"⟨([A-Z][A-Z0-9_]*)⟩")
#: placeholders that stand for a rendered block rather than one number
BLOCKS = ("LABEL", "RECORDS", "SUBCIRCUITS", "PERIODS", "CAVEATS", "SOURCES")

#: read from the first block record: {NAME: the record's field}
RECORD: dict[str, str] = {
    "RECORD_RUN": "run",
    "BLOCK_SERVER": "server",
    "BLOCK_STEPS": "server_time's steps [lo, hi)",
    "DEMO_A_VERDICT": "sampled proofs', PoUW's and network accounting's answer.verdict on the block's server",
    "DRAWN_UNITS": "integrity › quantity.drawn",
    "UNITS_TOTAL": "integrity › quantity.of",
    "LEAN_VERDICT": "integrity › answer.verdict (the Lean verifier's audit)",
    "H_UNITS": "integrity › at 1/100 › value",
    "INTEGRITY_ERROR": "integrity › error",
    "POUW_TILES": "pouw › answer.public.check's tile Definition",
    "POUW_TILES_DRAWN": "pouw › answer.public.drawn (its count)",
    "POUW_W_CERT": "pouw › quantity.value (W_cert, at 1/100)",
    "POUW_TILES_CREDITED": "pouw › quantity.tiles",
    "POUW_ERROR": "pouw › error",
    "POUW_BACKING": "pouw › answer.public.check's tile Definition: per-call noise (v5/v6) or the v1 subset",
    "NET_K": "certifier › quantity.value (K, at 1/100)",
    "NET_ERROR": "certifier › error",
    "JOINT_HORIZON_STATUS": "composed › joint_horizon.statement (name, proved)",
    "DEMO_A_ERROR_SUM": "the three protocols' error, summed",
    "D_GPUS": "monitoring_period.deployment: machines × gpus_per_machine (`benchmarks/census/deployments.json`)",
    "DEPLOYMENT_ID": "monitoring_period.deployment.id",
}

#: filled by the measurements file: {NAME: which run or field fills it}
MEASURED: dict[str, str] = {
    "COMMITTED_ROWS": "CMP-97's block run: the per-step commit calls' rows (HASH-13 `commit_slots`, #4469), summed "
                      "over the block's steps",
    "COMMITTED_BYTES": "CMP-97's block run: the bytes those commit calls hashed",
    "BLOCK_ROOT": "the block root of CMP-97's drawn-unit bundle (`verity/drawn-unit-bundle`, #4500)",
    "REGISTRATION_ROOT": "the seeded weights' registration root (CMP-95 item 1, #4445)",
    "CHALLENGER_RUN": "CMP-97's block run: the challenger's coin record for the block's draw",
    "ZK_SESSIONS": "PRF-171's prove run: its C-Flock --zk sessions behind the firewall (`sessions/index.jsonl`)",
    "PROVE_RUN": "PRF-171's prove run id (PRF-172's main path)",
    "FLIP_VERDICT": "PRF-171's flipped-bit run: the Lean verifier's verdict with one committed bit flipped",
    "FLIP_RUN": "PRF-171's flipped-bit run id",
    "JOINT_LOCKED": "LEAN-56: true once `Flock.Guarantees.JointStratified` is in the lock (`Security/lean-audit.json`) "
                    "on main by M3",
    "EPS_KS": "the bound's knowledge-soundness term for C-Flock --zk at the block's sessions (PRF-169)",
    "EPS_BIND": "the bound's binding term for the block's hm96-sha512 commitments",
    "H_A": "the bound (BOUND_CITATION) at δ = 1/100 on the block's draw, harm in ANDs: "
           "`sampled_proofs.optimize.certify` on the block's sizes and k",
    "H_B": "the same bound, harm in output bits",
    "BLOCK_ANDS": "the block's circuit: ANDs over all its proof units (circuit-check's ANDs a unit × units, per "
                  "subcircuit)",
    "BLOCK_OUTPUT_BITS": "the block's circuit: output bits over all its proof units",
    "POUW_EPS": "CMP-126 (H200-POUW): PoUW's ε at δ = 1/100 from the block's one draw, its tiles a subcircuit at the "
                "planner's k",
    "POUW_COVERAGE": "CMP-126: the share of the block's static INT8 MACs its tiles cover (`range`)",
    "POUW_CERTIFIED_SHARE": "CMP-126: PoUW's certified share at CMP-127's ρ",
    "POUW_RHO": "CMP-127: ρ, the live share in `H1J10RCheckedCreditPeriod`",
    "PROVING_GPU_S": "PRF-171's prove run: GPU-seconds of proving over every drawn unit's session",
    "BLOCK_GPU_S": "CMP-97's block run: H200-seconds the block served",
    "PROVE_S_PER_AND": "PRF-171's prove run, fitted by `sampled_proofs.costfit`: GPU-seconds of proving per AND",
    "REPLAY_GPU_S": "CMP-97: H200-seconds to replay the block's drawn units (the drawn-unit bundle's builder)",
    "COMMIT_GBPS": "CMP-97's block run: the commit call's throughput on the H200 (HASH-13, HASH-20)",
    "PLANNER_PREDICTED_S": "`sampled_proofs.optimize`'s cost for the block's draw, at the cost model fitted before the "
                           "block ran",
    "SERVE_COMMIT_S": "CMP-97's block run: seconds of commits a step (HASH-20)",
    "SERVE_STEP_S": "CMP-97's block run: seconds a step",
    "SERVE_NOISE_SHARE": "CMP-126 (KCOMMIT-FB): per-call noise's timed overhead over plain INT8 at the block's M",
    "H_POOLED": "`sampled_proofs.optimize.for_budget` at 1% of a sampling block's compute with part 2's costs: "
                "`certify`'s harm in ANDs, pooled over the horizon",
    "B_POOLED": "the same plan's harm in output bits, pooled over the horizon",
    "SIMULATED_RTT_MS": "CMP-97's block run: the challenger's simulated round trip",
}

_PERIOD_DERIVED: dict[str, str] = {}
for _p, _ in PERIODS:
    _P = _p.upper()
    _PERIOD_DERIVED |= {
        f"BLOCKS_{_P}": f"D_GPUS × a {_p} / (N_HOURS × 3600 s): the {_p}'s sampling blocks",
        f"H_SHARE_{_P}": f"H_POOLED / (D_GPUS × a {_p} × BLOCK_ANDS / BLOCK_GPU_S)",
        f"B_SHARE_{_P}": f"B_POOLED / (D_GPUS × a {_p} × BLOCK_OUTPUT_BITS / BLOCK_GPU_S)",
    }

#: formulas over the other numbers: {NAME: the formula}
DERIVED: dict[str, str] = {
    "H_A_SHARE": "H_A / BLOCK_ANDS",
    "Y": "PROVING_GPU_S / BLOCK_GPU_S",
    "PLANNER_RATIO": "PROVING_GPU_S / PLANNER_PREDICTED_S",
    "SERVE_COMMIT_SHARE": "SERVE_COMMIT_S / SERVE_STEP_S",
    "BOUND_CITATION": "JOINT_LOCKED: LEAN-56's `Flock.Guarantees.JointStratified` when locked, else #4358's "
                      "per-subcircuit bound with the joint figure unlocked",
    "HORIZON_CITATION": "JOINT_LOCKED: LEAN-56's `JointStratifiedHorizon` when locked, else #4358's "
                        "`Flock.Guarantees.IntegrityHorizon` per subcircuit with the joint figure unlocked",
    "SUBCIRCUIT_COUNT": "the subcircuits measured (part 2), else the block record's per-subcircuit draw",
    "N_HOURS": "the largest subcircuit fixed proving cost / 1%, in hours",
    "M_MINUTES": "N_HOURS × 60 / D_GPUS",
    **_PERIOD_DERIVED,
}


@dataclass(frozen=True)
class V:
    """One filled number: its value, where it came from, and the label it carries when it isn't the demo block's own."""

    value: Any
    source: str
    label: str | None = None


def _fmt(x: Any) -> str:
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, Fraction):
        x = int(x) if x.denominator == 1 else float(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        a = abs(x)
        if a == 0:
            return "0"
        if a >= 1e12 or a < 1e-3:
            return f"{x:.3g}"
        return f"{x:,.0f}" if a >= 1000 else f"{x:.4g}"
    s = str(x)
    return f"`{s[:16]}…`" if len(s) > 40 and re.fullmatch(r"[0-9a-f]+", s) else s


def _pct(x: Fraction | float) -> str:
    return f"{float(x) * 100:.3g}%"


def _num(v: V | None) -> Fraction | None:
    if v is None or isinstance(v.value, (bool, str)):
        return None
    return Fraction(v.value) if isinstance(v.value, int) else Fraction(str(v.value))


def _tiles(public: Mapping[str, Any] | None) -> str | None:
    t = ((public or {}).get("check") or {}).get("template")
    return t.split("{", 1)[0] if t else None


def demo_a_verdict(record: Mapping[str, Any]) -> str:
    """Demo A's composed verdict over `DEMO_A` on the record's block (module docstring)."""
    legs, server = record["legs"], record["server"]
    rejected = [p for p in DEMO_A if legs.get(p) and legs[p]["server"] == server
                and legs[p]["answer"]["verdict"] in ("reject", "error")]
    if rejected:
        return "reject (" + ", ".join(PROTOCOL_NAMES[p] for p in rejected) + ")"
    absent = [p for p in DEMO_A if not legs.get(p) or legs[p]["server"] != server]
    if absent:
        return "pending: no " + ", no ".join(PROTOCOL_NAMES[p] for p in absent) + " on the block's server"
    waiting = [p for p in DEMO_A if legs[p]["answer"]["verdict"] != "accept"]
    if waiting:
        return "pending (" + ", ".join(f"{PROTOCOL_NAMES[p]}: {legs[p]['answer']['verdict']}" for p in waiting) + ")"
    return "accept"


def from_record(record: Mapping[str, Any], label: str | None = None) -> dict[str, V]:
    """The `RECORD` numbers of one block record."""
    if bad := block_record.check(record):
        raise ValueError(f"block record {record.get('run')}: {bad}; `python -m verity_e2e_demo.block_record check` it")
    out: dict[str, V] = {}

    def put(name: str, value: Any) -> None:
        out[name] = V(value, f"block record {record['run']} › {RECORD[name]}", label)

    legs, server = record["legs"], record["server"]
    lo, hi = record["server_time"][0]["steps"]
    put("RECORD_RUN", record["run"])
    put("BLOCK_SERVER", server)
    put("BLOCK_STEPS", f"[{lo}, {hi})")
    put("DEMO_A_VERDICT", demo_a_verdict(record))
    here = {p: legs[p] for p in DEMO_A if legs.get(p) and legs[p]["server"] == server}
    if i := here.get("integrity"):
        put("DRAWN_UNITS", i["quantity"]["drawn"])
        put("UNITS_TOTAL", i["quantity"]["of"])
        put("LEAN_VERDICT", str(i["answer"]["verdict"]).upper())
        if (at := i["at"][str(DELTA)])["holds"]:
            put("H_UNITS", at["value"])
        put("INTEGRITY_ERROR", Fraction(i["error"]))
    if w := here.get("pouw"):
        public = w["answer"].get("public") or {}
        if tiles := _tiles(public):
            per_call = bool(re.search(r"_v[56]\b", tiles))
            put("POUW_TILES", f"per-call-noise tiles ({tiles})" if per_call else f"the v1 subset's tiles ({tiles})")
            put("POUW_BACKING", f"PoUW backing: shown, by the per-call-noise tiles ({tiles})." if per_call else
                f"PoUW backing: this block's PoUW ran the v1 subset's tiles ({tiles}), not the per-call-noise tiles, "
                f"so PoUW's figure in part 1 is the v1 subset's, for what it is, not the block's backing.")
        if isinstance(public.get("drawn"), list):
            put("POUW_TILES_DRAWN", len(public["drawn"]))
        if w["at"][str(DELTA)]["holds"]:
            put("POUW_W_CERT", w["quantity"]["value"])
        put("POUW_TILES_CREDITED", w["quantity"]["tiles"])
        put("POUW_ERROR", Fraction(w["error"]))
    if c := here.get("certifier"):
        if c["at"][str(DELTA)]["holds"]:
            put("NET_K", c["quantity"]["value"])
        put("NET_ERROR", Fraction(c["error"]))
    if len(here) == len(DEMO_A):
        put("DEMO_A_ERROR_SUM", sum((Fraction(here[p]["error"]) for p in DEMO_A), Fraction(0)))
    st = record["composed"]["joint_horizon"]["statement"]
    put("JOINT_HORIZON_STATUS", f"`{st['name']}`, " + ("proved" if st.get("proved") else "stated, not yet proved"))
    dep = record["monitoring_period"]["deployment"]
    put("D_GPUS", int(dep["machines"]) * int(dep["gpus_per_machine"]))
    put("DEPLOYMENT_ID", dep["id"])
    return out


def load_measurements(doc: Mapping[str, Any]) -> tuple[dict[str, V], list[dict[str, Any]], list[str]]:
    """The measurements file's values, subcircuits and caveats; refuses a name it may not fill and a value with no
    source."""
    if doc.get("format") != MEASUREMENTS:
        raise ValueError(f"measurements: format {doc.get('format')!r}, not {MEASUREMENTS}")
    values: dict[str, V] = {}
    for name, x in (doc.get("values") or {}).items():
        if name not in MEASURED:
            where = ("comes from the block record" if name in RECORD else
                     f"is derived ({DERIVED[name]})" if name in DERIVED else "is no number of the report")
            raise ValueError(f"measurements: ⟨{name}⟩ {where}; fill only the names of `report.MEASURED`")
        if not isinstance(x, Mapping) or "value" not in x or x["value"] is None:
            raise ValueError(f"measurements: ⟨{name}⟩ has no value; leave it out to keep it unfilled")
        if not str(x.get("source") or "").strip():
            raise ValueError(f"measurements: ⟨{name}⟩ names no source; say which run or field it came from")
        values[name] = V(x["value"], x["source"], x.get("label"))
    subcircuits = [dict(s) for s in doc.get("subcircuits") or []]
    for s in subcircuits:
        if not s.get("name") or not str(s.get("source") or "").strip():
            raise ValueError(f"measurements: subcircuit {s!r} needs a name and a source")
    return values, subcircuits, [str(c) for c in doc.get("caveats") or []]


def derive(values: Mapping[str, V], subcircuits: Sequence[Mapping[str, Any]]) -> dict[str, V]:
    """The `DERIVED` numbers whose inputs are all filled."""
    out: dict[str, V] = {}

    def put(name: str, value: Any, *inputs: str) -> None:
        mine = sorted({values[i].label for i in inputs if i in values and values[i].label})
        out[name] = V(value, DERIVED[name], "; ".join(mine) or None)

    def ratio(name: str, a: str, b: str, pct: bool = False) -> None:
        x, y = _num(values.get(a)), _num(values.get(b))
        if x is not None and y:
            put(name, _pct(x / y) if pct else float(x / y), a, b)

    ratio("H_A_SHARE", "H_A", "BLOCK_ANDS", pct=True)
    ratio("Y", "PROVING_GPU_S", "BLOCK_GPU_S")
    ratio("PLANNER_RATIO", "PROVING_GPU_S", "PLANNER_PREDICTED_S")
    ratio("SERVE_COMMIT_SHARE", "SERVE_COMMIT_S", "SERVE_STEP_S", pct=True)
    if (lk := values.get("JOINT_LOCKED")) is not None:
        put("BOUND_CITATION", "LEAN-56's `Flock.Guarantees.JointStratified` (locked)" if lk.value is True else
            "#4358's per-subcircuit bound, the joint figure unlocked", "JOINT_LOCKED")
        put("HORIZON_CITATION", "LEAN-56's `JointStratifiedHorizon` (locked)" if lk.value is True else
            "#4358's `Flock.Guarantees.IntegrityHorizon` per subcircuit, the joint figure unlocked", "JOINT_LOCKED")
    if subcircuits:
        out["SUBCIRCUIT_COUNT"] = V(len(subcircuits), DERIVED["SUBCIRCUIT_COUNT"], "; ".join(
            sorted({s["label"] for s in subcircuits if s.get("label")})) or None)
        fixed = [s.get("fixed_s") for s in subcircuits]
        if all(isinstance(f, (int, float)) and not isinstance(f, bool) for f in fixed):
            out["N_HOURS"] = V(float(Fraction(str(max(fixed))) / BUDGET / 3600), DERIVED["N_HOURS"], out[
                "SUBCIRCUIT_COUNT"].label)
    d = _num(values.get("D_GPUS"))
    n = _num(out.get("N_HOURS"))
    if n is not None and d:
        out["M_MINUTES"] = V(float(n * 60 / d), DERIVED["M_MINUTES"], out["N_HOURS"].label)
    for p, seconds in PERIODS:
        P = p.upper()
        if n and d:
            out[f"BLOCKS_{P}"] = V(float(d * seconds / (n * 3600)), DERIVED[f"BLOCKS_{P}"], out["N_HOURS"].label)
        gs = _num(values.get("BLOCK_GPU_S"))
        for harm, total, share in (("H_POOLED", "BLOCK_ANDS", f"H_SHARE_{P}"),
                                   ("B_POOLED", "BLOCK_OUTPUT_BITS", f"B_SHARE_{P}")):
            h, t = _num(values.get(harm)), _num(values.get(total))
            if h is not None and t and gs and d:
                put(share, _pct(h / (d * seconds * t / gs)), harm, total, "BLOCK_GPU_S", "D_GPUS")
    return out


def _cell(name: str, values: Mapping[str, V], quiet: str | None = None) -> str:
    """A number in place, with its label unless it is ``quiet`` (the records' label, stated once at the top)."""
    v = values.get(name)
    if v is None:
        return f"⟨{name}⟩"
    return _fmt(v.value) + (f" *({v.label})*" if v.label and v.label != quiet else "")


def _records(records: Sequence[Mapping[str, Any]]) -> str:
    head = ("| block record | server | steps | " + " | ".join(PROTOCOL_NAMES[p] for p in DEMO_A)
            + " | Demo A | all six audits |")
    rows = [head, "|" + " --- |" * (len(DEMO_A) + 5)]
    def cell(r: Mapping[str, Any], p: str) -> str:
        lg = r["legs"].get(p)
        if not lg:
            return "not run"
        where = "" if lg["server"] == r["server"] else ", another server"
        return f"{lg['answer']['verdict']} ({lg['source']}{where})"

    for i, r in enumerate(records):
        lo, hi = r["server_time"][0]["steps"]
        rows.append(f"| {r['run']}{' (this report)' if i == 0 else ''} | {r['server']} | [{lo}, {hi}) | "
                    + " | ".join(cell(r, p) for p in DEMO_A) + f" | {demo_a_verdict(r)} | {r['composed']['verdict']} |")
    return "\n".join(rows)


def _subcircuits(subcircuits: Sequence[Mapping[str, Any]], quiet: str | None) -> str:
    cols = (("units", "units"), ("drawn", "drawn"), ("ands_per_unit", "ANDs a unit"),
            ("fixed_s", "fixed proving cost per sampling block (GPU-s)"))
    rows = ["| subcircuit | " + " | ".join(c for _, c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    if not subcircuits:
        rows.append("| ⟨SUBCIRCUITS⟩ |" + " ⟨SUBCIRCUITS⟩ |" * len(cols))
    for s in subcircuits:
        lab = f" *({s['label']})*" if s.get("label") and s["label"] != quiet else ""
        rows.append(f"| {s['name']}{lab} | " + " | ".join(
            _fmt(s[k]) if s.get(k) is not None else f"⟨SUBCIRCUITS.{k}⟩" for k, _ in cols) + " |")
    return "\n".join(rows)


def _periods(values: Mapping[str, V], quiet: str | None) -> str:
    rows = [("| period | sampling blocks | H faulty ANDs | share of the period's ANDs | B leaked bits | "
             "share of the period's output bits |"), "|" + " --- |" * 6]
    for p, _ in PERIODS:
        P = p.upper()
        rows.append(f"| {p} | " + " | ".join(_cell(n, values, quiet) for n in (
            f"BLOCKS_{P}", "H_POOLED", f"H_SHARE_{P}", "B_POOLED", f"B_SHARE_{P}")) + " |")
    return "\n".join(rows)


def _caveats(records: Sequence[Mapping[str, Any]], values: Mapping[str, V], extra: Sequence[str],
             quiet: str | None) -> str:
    r = records[0]
    lines = []
    for p in DEMO_A:
        lg = r["legs"].get(p)
        if lg is None or lg["server"] != r["server"]:
            lines.append(f"{PROTOCOL_NAMES[p].capitalize()} didn't run on the block's server.")
        elif lg["source"] == "fixture":
            lines.append(f"{PROTOCOL_NAMES[p].capitalize()} ran on a fixture, not on the block "
                         f"(block record {r['run']}).")
    if not r["composed"]["joint_horizon"]["statement"].get("proved"):
        lines.append("JointHorizon is stated, not yet proved: the error sum in part 1 is the sum it states.")
    lines.append("PRF-167: the state carried across a wipe has no cross-session binding term yet, so the bound holds "
                 "per server-time block.")
    lines.append("`pous-ring-instantiation` is M4's: PoUS's S_cert, the bandwidth audit's Q_cert and the wipe's f·M "
                 "compose at M4 (Demo B), not in Demo A.")
    labels = sorted({v.label for v in values.values() if v.label})
    if quiet:
        lines.append(f"The numbers read from the block records, and the measurements labelled *({quiet})*, are the "
                     f"{quiet}'s, not the demo block's.")
    for lab in labels:
        if lab != quiet:
            lines.append(f"Numbers marked *({lab})* are not the demo block's own.")
    lines += list(extra)
    return "\n".join(f"- {x}" for x in lines)


def _sources(order: Sequence[str], values: Mapping[str, V], subcircuits: Sequence[Mapping[str, Any]]) -> str:
    rows = ["| number | value | filled from |", "| --- | --- | --- |"]
    filled = 0
    for name in order:
        v = values.get(name)
        if name == "SUBCIRCUITS":
            src = "; ".join(f"{s['name']}: {s['source']}" for s in subcircuits) or \
                  "CMP-97's block and PRF-171's prove run: per subcircuit its units, drawn units, ANDs a unit and " \
                  "fixed proving cost per sampling block (`sampled_proofs.costfit`'s statement_fixed)"
            filled += bool(subcircuits)
            rows.append(f"| ⟨SUBCIRCUITS⟩ | {'filled' if subcircuits else 'unfilled'} | {src} |")
            continue
        if v is not None:
            filled += 1
            src = v.source + (f" *({v.label})*" if v.label else "")
            rows.append(f"| ⟨{name}⟩ | {_fmt(v.value)} | {src} |")
        else:
            src = MEASURED.get(name) or DERIVED.get(name) or f"block record › {RECORD[name]}"
            rows.append(f"| ⟨{name}⟩ | unfilled | {src} |")
    return f"{filled} of {len(order)} numbers filled; an unfilled one shows as ⟨NAME⟩.\n\n" + "\n".join(rows)


def render(records: Sequence[Mapping[str, Any]], measurements: Mapping[str, Any] | None = None, *,
           record_label: str | None = None, template: str | None = None) -> str:
    """The report's Markdown (module docstring)."""
    if not records:
        raise ValueError("render needs at least one block record: the first is the block the report is about")
    text = TEMPLATE.read_text() if template is None else template
    measured, subcircuits, caveats = load_measurements(measurements or {"format": MEASUREMENTS})
    values = {**from_record(records[0], record_label), **measured}
    for r in records[1:]:
        from_record(r)
    if "SUBCIRCUIT_COUNT" not in values and not subcircuits:
        per = (records[0]["legs"].get("integrity") or {}).get("answer", {}).get("public", {}).get("per_template")
        if per:
            values["SUBCIRCUIT_COUNT"] = V(len(per), DERIVED["SUBCIRCUIT_COUNT"], record_label)
    values |= derive(values, subcircuits)
    known = set(RECORD) | set(MEASURED) | set(DERIVED) | set(BLOCKS)
    order: list[str] = []
    for name in PLACEHOLDER.findall(text):
        if name not in known:
            raise ValueError(f"the template's ⟨{name}⟩ is no number of `report` (RECORD, MEASURED, DERIVED or BLOCKS)")
        names = [name] if name not in BLOCKS or name == "SUBCIRCUITS" else []
        if name in ("BOUND_CITATION", "HORIZON_CITATION"):
            names.append("JOINT_LOCKED")
        if name == "PERIODS":
            names = [n for p, _ in PERIODS for n in (f"BLOCKS_{p.upper()}", "H_POOLED", f"H_SHARE_{p.upper()}",
                                                     "B_POOLED", f"B_SHARE_{p.upper()}")]
        order += [n for n in dict.fromkeys(names) if n not in order]
    banner = (f"> The {record_label}'s numbers are not marked in place: every number read from the block records "
              f"(block record {records[0]['run']}), and every measurement labelled *({record_label})*, is the "
              f"{record_label}'s, not the demo block's. Any other number that isn't the demo block's own is marked "
              f"in place.") if record_label else "> A number that isn't the demo block's own is marked in place."
    blocks = {"LABEL": banner, "RECORDS": _records(records), "SUBCIRCUITS": _subcircuits(subcircuits, record_label),
              "PERIODS": _periods(values, record_label), "CAVEATS": _caveats(records, values, caveats, record_label),
              "SOURCES": _sources(order, values, subcircuits)}
    out = PLACEHOLDER.sub(lambda m: blocks[m[1]] if m[1] in blocks else _cell(m[1], values, record_label), text)
    vocabulary(out)
    return out


def vocabulary(text: str) -> None:
    """Refuses a report that says "leg", "window" or "template" (CMP-98's words)."""
    for n, line in enumerate(text.splitlines(), 1):
        if m := BANNED.search(line):
            raise ValueError(f"the report says {m[0]!r} (line {n}: {line.strip()!r}): name each protocol (sampled "
                             f"proofs, PoUW, network accounting), and say 'sampling block' and 'subcircuit'")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Demo A's report from block records and a measurements file.")
    ap.add_argument("records", nargs="+", type=Path,
                    help="block records (verity/block-record/v1); the first is the block")
    ap.add_argument("--measurements", type=Path)
    ap.add_argument("--record-label", help="the label every number read from the block records carries")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    md = render([json.loads(p.read_text()) for p in a.records],
                json.loads(a.measurements.read_text()) if a.measurements else None, record_label=a.record_label)
    if a.out:
        a.out.write_text(md)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
