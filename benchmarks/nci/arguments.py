"""The argument map (``arguments.json``): what we claim about the inference-only policy, on what, against what, and what would settle it.

* ``claims``: a ladder from most to least trustworthy (``rung`` 1-6), each with its ``status``, the premises it rests
  on (premise or claim ids), the attacks against it, and what comes ``next``.
* ``premises``: theorems, assumptions, constructions, estimates, policy rules and models, each with its ``status`` and
  references: findings (``sources/``), parameter estimates, Lean names (``Security/lean-audit.json``), library
  algorithms and board rows.
* ``attacks``: from the red teams and the workstreams, each with its severity, the claims it targets, its status and
  repair (a premise id or prose).
* ``plans``: the evidence the blue teams proposed, with the round-2 red team's verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import catalog  # noqa: E402

CLAIM_STATUS = frozenset({"established", "conditional", "contested", "open"})
PREMISE_KINDS = frozenset({"theorem", "assumption", "construction", "estimate", "policy-rule", "model"})
PREMISE_STATUS = frozenset({"proved", "assumed", "analytic", "measured", "proposed", "missing",
                            "false-as-instantiated"})
SEVERITY = ("fatal", "major", "moderate", "minor", "favors-defender")
ATTACK_STATUS = frozenset({"open", "repair-proposed", "answered"})
VERDICTS = frozenset({"survives", "survives with repair", "broken", "compelling", "compelling with redesign",
                      "not compelling"})


def load(root: Path = HERE) -> dict:
    return json.loads((root / "arguments.json").read_text())


def problems(lock: Path | None = None, root: Path = HERE) -> list[str]:
    a = load(root)
    out = []
    sources = catalog.load("sources", root)
    findings = catalog.findings(sources)
    estimates = catalog.estimates(catalog.load("parameters", root))
    algorithms = {x["id"] for x in json.loads((root / "algorithms.json").read_text())}
    rows = {x["id"] for x in catalog.load("rows", root)}
    lean = None
    if lock is not None:
        pol = json.loads(lock.read_text())
        lean = set(pol["guarantees"]) | set(pol["assumptions"])
    claims = {c["id"]: c for c in a["claims"]}
    premises = {p["id"]: p for p in a["premises"]}
    attacks = {x["id"]: x for x in a["attacks"]}

    def refs(owner: str, r: dict):
        for f in r.get("findings", []):
            if f not in findings:
                out.append(f"{owner}: finding {f} is not in sources/")
        for e in r.get("estimates", []):
            if e not in estimates:
                out.append(f"{owner}: estimate {e} is not in parameters.json")
        for g in r.get("algorithms", []):
            if g not in algorithms:
                out.append(f"{owner}: algorithm {g} is not in algorithms.json")
        for w in r.get("rows", []):
            if w not in rows:
                out.append(f"{owner}: row {w} is not in rows.json")
        if lean is not None:
            for n in r.get("lean", []):
                if n not in lean:
                    out.append(f"{owner}: {n} is no guarantee or assumption in the lock")

    for c in a["claims"]:
        if c.get("status") not in CLAIM_STATUS:
            out.append(f"claim {c['id']}: status {c.get('status')}")
        if c.get("rung") not in range(1, 7):
            out.append(f"claim {c['id']}: rung must be 1-6")
        for p in c.get("premises", []):
            if p not in premises and p not in claims:
                out.append(f"claim {c['id']}: premise {p} is no premise or claim")
        for x in c.get("attacks", []):
            if x not in attacks:
                out.append(f"claim {c['id']}: attack {x} is not in attacks")
    for p in a["premises"]:
        if p.get("kind") not in PREMISE_KINDS or p.get("status") not in PREMISE_STATUS:
            out.append(f"premise {p['id']}: kind {p.get('kind')} or status {p.get('status')}")
        refs(f"premise {p['id']}", p.get("refs", {}))
    for x in a["attacks"]:
        if x.get("severity") not in SEVERITY or x.get("status") not in ATTACK_STATUS:
            out.append(f"attack {x['id']}: severity {x.get('severity')} or status {x.get('status')}")
        for t in x.get("targets", []):
            if t not in claims:
                out.append(f"attack {x['id']}: target {t} is no claim")
            elif x["id"] not in claims[t]["attacks"]:
                out.append(f"attack {x['id']}: claim {t} doesn't list it")
        if x.get("repair", "").startswith("p-") and x["repair"].split(",")[0] not in premises:
            out.append(f"attack {x['id']}: repair {x['repair']} is no premise")
        refs(f"attack {x['id']}", x.get("refs", {}))
    for e in a["plans"]:
        if e.get("verdict") not in VERDICTS:
            out.append(f"plan {e['id']}: verdict {e.get('verdict')}")
    return out


def render() -> list[str]:
    a = load()
    premises = {p["id"]: p for p in a["premises"]}
    attacks = {x["id"]: x for x in a["attacks"]}
    out = ["## Arguments", "", "Claims on a ladder, from most to least trustworthy, with what each rests on and what "
           "attacks it (`arguments.json`).", ""]
    for c in sorted(a["claims"], key=lambda c: c["rung"]):
        out += [f"### Rung {c['rung']}: {c['statement']} ({c['status']})", ""]
        for p in c["premises"]:
            if p in premises:
                q = premises[p]
                out.append(f"- Rests on `{p}` ({q['kind']}, {q['status']}): {q['statement']}")
            else:
                out.append(f"- Rests on claim `{p}`.")
        for x in sorted(c["attacks"], key=lambda x: SEVERITY.index(attacks[x]["severity"])):
            t = attacks[x]
            out.append(f"- Attack `{x}` [{t['severity']}, {t['status']}]: {t['title']}. {t['mechanism']} "
                       f"Effect: {t['effect']}. Repair: {t['repair'].rstrip('.')}.")
        out += [f"- Next: {c['next']}", ""]
    out += ["### Proposed evidence and the round-2 verdicts", ""]
    for e in a["plans"]:
        out.append(f"- `{e['id']}` ({e['from']}): {e['title']}. **{e['verdict']}**. {e['note']}")
    out.append("")
    return out
