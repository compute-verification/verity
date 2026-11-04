"""The inference-only policy's literature catalog: loading and validation.

Four JSON files, each an array of entries (README.md, "Files"):

* ``sources/<topic>.json``: one entry per paper, post or dataset, each with findings that quote it verbatim, filed under
  its first topic;
* ``parameters.json``: the calculator's parameters, each estimate citing the findings it merges and saying how;
* ``calibrations.json``: a policy (K, Gamma, b_tok) and the model shapes it is calibrated to;
* ``rows.json``: the board, one training method in one configuration per row.

``problems()`` returns every rule a file breaks, as text; the tests require it to be empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

TOPICS = frozenset({"weights", "updates", "seed-based", "gradients", "activations", "tokens", "capability-per-bit",
                    "theory", "governance", "integration-measures", "efficiency-metrics", "strategy-local",
                    "strategy-lowrank-sparse", "strategy-gradient-free", "strategy-reuse", "capability-measures",
                    "rl-scaling"})
GRADES = frozenset({"measured-at-scale", "measured-small", "extrapolated", "theoretical", "speculative"})
VERIFIED = frozenset({"fetched", "search-snippet"})
PARAMS = frozenset({"b_w.pretrained", "b_w.lossless", "b_w.capacity", "b_w.delta", "update.bits_total",
                    "update.decode_flops", "update.fraction_changed", "update.rank", "grad.compression_ratio",
                    "b_a.activation", "b_a.kv", "activation.compression_ratio", "activation.intrinsic_dim", "b_tok",
                    "prompt.compression_ratio", "capability.bits", "other"})
PICKS = ("low", "central", "high")
CONFIDENCE = frozenset({"low", "medium", "high"})
KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FINDING_ID = re.compile(r"^([a-z0-9]+(?:-[a-z0-9]+)*)/([1-9][0-9]*)$")


def load(name: str, root: Path = HERE) -> list[dict]:
    if name == "sources":
        return [s for path in sorted((root / "sources").glob("*.json")) for s in json.loads(path.read_text())]
    return json.loads((root / f"{name}.json").read_text())


def _filing_problems(root: Path) -> list[str]:
    out = []
    for path in sorted((root / "sources").glob("*.json")):
        if path.stem not in TOPICS:
            out.append(f"sources/{path.name}: not a topic")
        for s in json.loads(path.read_text()):
            if s.get("topics", [None])[0] != path.stem:
                out.append(f"source {s.get('id')}: filed in sources/{path.name}, but its first topic is "
                           f"{s.get('topics', [None])[0]}")
    return out


def findings(sources: list[dict]) -> dict[str, dict]:
    """Every finding by its id, ``<source id>/<n>``."""
    return {f["id"]: f for s in sources for f in s.get("findings", [])}


def estimates(parameters: list[dict]) -> dict[str, dict]:
    """Every parameter estimate by its id, ``<parameter id>/<estimate>``."""
    return {e["id"]: e | {"unit": p["unit"]} for p in parameters for e in p.get("estimates", [])}


def resolve(ref, parameters: list[dict]) -> float:
    """A row input: a number, or ``"<estimate id>:<low|central|high>"``."""
    if isinstance(ref, (int, float)):
        return float(ref)
    est, _, pick = ref.rpartition(":")
    return float(estimates(parameters)[est][pick])


def _source_problems(sources: list[dict]) -> list[str]:
    out, seen, urls = [], set(), {}
    for s in sources:
        sid = s.get("id", "?")
        if not KEBAB.match(sid):
            out.append(f"source {sid}: id is not kebab-case")
        if sid in seen:
            out.append(f"source {sid}: duplicate id")
        seen.add(sid)
        for key in ("title", "authors", "venue", "url"):
            if not isinstance(s.get(key), str) or not s[key].strip():
                out.append(f"source {sid}: missing {key}")
        if not str(s.get("url", "")).startswith("https://"):
            out.append(f"source {sid}: url is not https")
        elif s["url"] in urls:
            out.append(f"source {sid}: same url as {urls[s['url']]}")
        else:
            urls[s["url"]] = sid
        if not isinstance(s.get("year"), int) or not 1940 <= s["year"] <= 2027:
            out.append(f"source {sid}: year")
        if not s.get("topics") or not set(s["topics"]) <= TOPICS:
            out.append(f"source {sid}: topics {s.get('topics')} not in {sorted(TOPICS)}")
        if s.get("verified") not in VERIFIED:
            out.append(f"source {sid}: verified must be one of {sorted(VERIFIED)}")
        fs = s.get("findings") or []
        if not fs:
            out.append(f"source {sid}: no findings")
        for n, f in enumerate(fs, 1):
            fid = f.get("id")
            if fid != f"{sid}/{n}":
                out.append(f"source {sid}: finding {n} has id {fid}, expected {sid}/{n}")
            for key in ("claim", "quote", "location", "bears_on"):
                if not isinstance(f.get(key), str) or not f[key].strip():
                    out.append(f"finding {fid}: missing {key}")
            if f.get("grade") not in GRADES:
                out.append(f"finding {fid}: grade {f.get('grade')}")
            if not isinstance(f.get("quantities"), list):
                out.append(f"finding {fid}: quantities must be a list, [] when the finding states none")
            for q in f.get("quantities") or []:
                if q.get("param") not in PARAMS:
                    out.append(f"finding {fid}: param {q.get('param')}")
                if not isinstance(q.get("value"), (int, float)) or isinstance(q.get("value"), bool):
                    out.append(f"finding {fid}: value {q.get('value')!r} is not a number")
                if not q.get("unit") or not q.get("condition"):
                    out.append(f"finding {fid}: a quantity needs a unit and a condition")
    return out


def _parameter_problems(parameters: list[dict], fs: dict[str, dict]) -> list[str]:
    out, seen = [], set()
    for p in parameters:
        pid = p.get("id", "?")
        if pid in seen:
            out.append(f"parameter {pid}: duplicate id")
        seen.add(pid)
        for key in ("symbol", "meaning", "unit"):
            if not p.get(key):
                out.append(f"parameter {pid}: missing {key}")
        for e in p.get("estimates", []):
            eid = e.get("id", "?")
            if not eid.startswith(f"{pid}/") or not KEBAB.match(eid[len(pid) + 1:]):
                out.append(f"estimate {eid}: id must be {pid}/<kebab-case>")
            vals = [e.get(k) for k in PICKS]
            if not all(isinstance(v, (int, float)) for v in vals):
                out.append(f"estimate {eid}: low, central and high must be numbers")
            elif not vals[0] <= vals[1] <= vals[2]:
                out.append(f"estimate {eid}: low <= central <= high fails: {vals}")
            if not e.get("context") or not e.get("merge"):
                out.append(f"estimate {eid}: missing context or merge")
            if e.get("confidence") not in CONFIDENCE:
                out.append(f"estimate {eid}: confidence")
            if not e.get("evidence"):
                out.append(f"estimate {eid}: no evidence")
            for ref in e.get("evidence", []):
                if ref not in fs:
                    out.append(f"estimate {eid}: evidence {ref} is no finding")
    return out


def _row_problems(rows: list[dict], parameters: list[dict], sources: list[dict]) -> list[str]:
    from slowdown import INPUTS

    out, seen = [], set()
    ests, sids = estimates(parameters), {s["id"] for s in sources}
    for r in rows:
        rid = r.get("id", "?")
        if not KEBAB.match(rid) or rid in seen:
            out.append(f"row {rid}: id must be unique kebab-case")
        seen.add(rid)
        for key in ("method", "family", "configuration", "state"):
            if not r.get(key):
                out.append(f"row {rid}: missing {key}")
        inputs = r.get("inputs", {})
        if set(inputs) != set(INPUTS):
            out.append(f"row {rid}: inputs must be exactly {sorted(INPUTS)}; "
                       f"missing {sorted(set(INPUTS) - set(inputs))}, extra {sorted(set(inputs) - set(INPUTS))}")
        for name, ref in inputs.items():
            if isinstance(ref, str):
                est, _, pick = ref.rpartition(":")
                if est not in ests or pick not in PICKS:
                    out.append(f"row {rid}: input {name} = {ref!r} is no estimate:pick")
            elif not isinstance(ref, (int, float)) or isinstance(ref, bool):
                out.append(f"row {rid}: input {name} = {ref!r}")
        for s in r.get("sources", []):
            if s not in sids:
                out.append(f"row {rid}: source {s} is not in sources/")
        chain = r.get("chain")
        if chain is not None and not (isinstance(chain.get("kappa"), (int, float)) and 0 < chain["kappa"] <= 1
                                      and isinstance(chain.get("d"), (int, float)) and chain["d"] >= 1
                                      and chain.get("path")):
            out.append(f"row {rid}: chain needs kappa in (0, 1], d >= 1 and a path")
    return out


def problems(root: Path = HERE) -> list[str]:
    sources, parameters, rows = load("sources", root), load("parameters", root), load("rows", root)
    fs = findings(sources)
    out = (_filing_problems(root) + _source_problems(sources) + _parameter_problems(parameters, fs)
           + _row_problems(rows, parameters, sources))
    for c in load("calibrations", root):
        for key in ("K", "Gamma", "b_tok"):
            if not isinstance(c.get(key), (int, float)) or c[key] <= 0:
                out.append(f"calibration {c.get('id')}: {key} must be a positive number")
        if not isinstance(c.get("theta_cap"), bool):
            out.append(f"calibration {c.get('id')}: theta_cap must be true or false")
    return out
