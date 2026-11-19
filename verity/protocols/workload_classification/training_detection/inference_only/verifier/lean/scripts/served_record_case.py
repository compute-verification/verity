"""A served run's files as one case of the record form, and the reference's verdict on it.

The form (the Lean's `Record`) reads an instance (the Program's text, the circuit record's text, Γ, α and the pins)
and the trusted input (the deployment's registration log and the digest of the run's own record); its count
(`RecordCount`) reads the count's limits beside them. A served run writes each as a file. This reads those
files as they are and writes the case, ``NAME.case.json`` (a line of the record difftest's cases, which
``lake exe inference-only-record-check`` reads), and the reference's two verdicts on it, ``NAME.reference.json``:
``{"check", "count"}``, `inference_only_record_reference.check_detail` and `count`, what `inference-only-record-check` prints.

    python served_record_case.py --program P --record C --log FIRST ... RUN --limits L --out DIR --name NAME
        [--count B] [--unregister smallest|largest|PARAMETER] [--drop registered.KEY] [--format P=JSON|P=-]
        [--no-pins] [--nodes] [--case-only]

- ``--program``: the Program's v1 descriptor. ``--record``: the circuit record, the bytes the run's record commits.
- ``--log``: the registration log's records, the first first and the run's own last.
- ``--limits``: ``{"alpha": [num, den], "gamma": Γ}``, the limits the run fixed; other keys are not read.
- ``--count``: ``{"table", "c", "k", "n"}``, the count's limits; without it they are empty and the count refuses them.
- The pins are the closure digests of the Definitions the record's openings name as their readers; ``--no-pins``
  gives none.
- ``--case-only``: the case alone, not the reference's verdicts.

Four options make a case the run did not write, and the case's name should say so:

- ``--unregister``: the log's first record without one parameter's entry and each later record's ``previous``
  made again, so the deployment never registered it. ``smallest`` and ``largest`` choose by words among the
  parameters the record reads that the first record registers.
- ``--drop registered.KEY``: the record without that key of its ``registered``, and the run's own record pointed
  at the new text. The run committed no such text: a stand-in, to read what the form says past that key.
- ``--format P=JSON``: the record with parameter P's row format (``registered.formats``) that JSON object, or with
  none where it is ``-``, and the run's own record pointed at the new text, as ``--drop`` does.
- ``--nodes``: the record with ``registered.nodes`` declaring each root input node that two rows read (`shared`),
  as what the record says of the first row that reads it: the registered parameter its read names, from the leaf
  the read's source puts the node's first leaf at, or the table read's table and root or domain id. The run's own
  record is pointed at the new text, as ``--drop`` does: a stand-in until the adapter writes the key.

The reference runs form-only (`check_detail`'s ``form_only``): its leaf-by-leaf expansions of `own-rows` and
`leaf-widths` visit every leaf of every row of every unit, more than a served class's circuit lets it walk, so on
such a case it holds the Lean's rules and not what they stand for. The verdicts are for the private record and are
never published.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

from verity.commitments.periods.interface import circuit_bytes


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ref = _load("inference_only_record_reference")
reference = ref.reference
#: the count's limits where none are given: an empty table and nothing allowed
NO_COUNT = {"table": {}, "c": 0, "k": 0, "n": 0}


def rechained(log: list[dict]) -> list[dict]:
    """The log with each record's ``previous`` the digest of the record before it."""
    out: list[dict] = []
    for rec in log:
        out.append({**rec, "previous": reference.record_digest(out[-1]) if out else None})
    return out


def shared(program) -> dict[int, list[tuple]]:
    """Each root input node two rows read, with the readers of it, `(root node, argument, b, q, os, e)` each
    (`stretch_of`): one a batch reads at outer stride 0, or two readers' stretches that meet."""
    body = program.root.body.nodes
    reads: dict[int, list[tuple]] = {}
    for k, node in enumerate(body):
        if ref.is_input(node.fn):
            continue
        for i in range(len(node.args)):
            got = ref.stretch_of(node, i)
            if got is not None and ref.is_input(body[got[0]].fn):
                reads.setdefault(got[0], []).append((k, i, *got[1:]))
    out = {}
    for n, rs in reads.items():
        spans = sorted((b + min(0, (body[k].n - 1) // q * os), b + max(0, (body[k].n - 1) // q * os)
                        + min(q, body[k].n) * e) for k, _, b, q, os, e in rs)
        if any(os == 0 and body[k].form == "batch" and body[k].n > q for k, _, _, q, os, _ in rs) or any(
                a[1] > b[0] for a, b in zip(spans, spans[1:])):
            out[n] = rs
    return out


def declared(program, desc: dict, doc: dict) -> list[dict]:
    """`registered.nodes` for the nodes `shared` finds, each as the record fills its first reader's first row."""
    from verity.circuits import codec
    body = program.root.body.nodes
    starts, at = {}, 0
    first: dict[str, int] = {}
    for k, node in enumerate(body):
        if not ref.is_input(node.fn):
            tid = codec.descriptor_id(node.fn)
            starts[k] = first.get(tid, 0)
            first[tid] = starts[k] + (node.n if node.form == "batch" else 1)
    reg = doc.get("registered", {})
    out = []
    for n, rs in sorted(shared(program).items()):
        k, i, b, *_ = rs[0]
        tid, port, u0 = codec.descriptor_id(body[k].fn), body[k].fn.signature()[0][i][0], starts[k]
        w = ref._homo(body[n].out)
        hit = None
        for r in reg.get("reads", []):
            if r["reader"] == tid and r["port"] == port:
                C = int(r.get("cut", [0, 1])[1]) if isinstance(r.get("cut"), list) else 1
                c0 = int(r.get("cut", [0, 1])[0]) if isinstance(r.get("cut"), list) else 0
                idx = (u0 - r["at"]) * C - c0
                if c0 == 0 and 0 <= idx < math.prod(r["radix"]):
                    s = ref.source(r, idx)
                    if (16 * s - w * b) % 16 == 0 and 16 * s >= w * b:
                        hit = {"node": n, "param": r["param"], "offset": (16 * s - w * b) // 16}
                    break
        for r in reg.get("roots", []):
            if hit is None and r["reader"] == tid and r["port"] == port and r["at"] <= u0 < r["at"] + r["units"]:
                hit = {"node": n, "table": r["table"], "of": r["of"]}
        if hit is not None:
            out.append(hit)
    return out


def served_case(program: str, record: str, log: list[dict], limits: dict, count: dict | None = None, *,
                unregister: str | None = None, drop: tuple[str, ...] = (), formats: tuple[str, ...] = (),
                pins: bool = True, nodes: bool = False) -> tuple[dict, list[str]]:
    """The case the files make, and what of it the run did not write (empty for the run's own). `program` and
    `record` are the files' text, `log` their records in order, the run's own last."""
    doc, log = json.loads(record), list(log)
    changed = []

    def restated(what: str) -> None:
        nonlocal record
        record = circuit_bytes(doc).decode()
        log[-1] = {**log[-1], "window": {**log[-1]["window"], "circuit": hashlib.sha512(record.encode()).hexdigest()}}
        changed.append(f"the record {what}, the run's record pointed at it (a stand-in)")

    def as_served(option: str) -> None:
        if circuit_bytes(doc).decode() != record:
            raise ValueError(f"{option}: the record is not in the form a serve writes, so its text can't be made again")

    for key in drop:
        head, _, k = key.partition(".")
        if head != "registered" or k not in doc.get("registered", {}):
            raise ValueError(f"--drop {key}: the record's registered has no such key")
        as_served("--drop")
        del doc["registered"][k]
        restated(f"without registered.{k}")
    for spec in formats:
        p, _, f = spec.partition("=")
        as_served("--format")
        stated = doc["registered"].setdefault("formats", {})
        if f == "-":
            if p not in stated:
                raise ValueError(f"--format {p}=-: the record has no format for {p}")
            del stated[p]
        else:
            stated[p] = json.loads(f)
        restated(f"with {p}'s format {f}")
    if nodes:
        as_served("--nodes")
        doc.setdefault("registered", {})["nodes"] = declared(reference.desc_program(program), json.loads(program), doc)
        restated(f"with registered.nodes declaring {len(doc['registered']['nodes'])} root input nodes")
    if unregister:
        first = {e.get("value") for e in log[0]["registered"]}
        read = sorted((n, p) for p, n in doc["registered"]["params"].items() if p in first)
        name = {"smallest": read[0][1], "largest": read[-1][1]}.get(unregister, unregister) if read else unregister
        kept = [e for e in log[0]["registered"] if e.get("value") != name]
        if len(kept) == len(log[0]["registered"]):
            raise ValueError(f"--unregister {name}: the log's first record registers no such value")
        log[0] = {**log[0], "registered": kept}
        changed.append(f"the log's first record without {name} ({doc['registered']['params'].get(name)} words in the "
                       f"record), {len(kept)} entries left")
    if changed:
        log = rechained(log)
    readers = sorted({o["reader"] for o in doc.get("registered", {}).get("openings", [])})
    desc = json.loads(program)
    inst = {"program": program, "record": record, "gamma": limits["gamma"], "alpha": limits["alpha"],
            "pins": sorted({ref.closure_digest(desc, t) for t in readers}) if pins else []}
    tr = {"log": log, "run": reference.record_digest(log[-1]), "params": []}
    case = {"instance": inst, "trusted": tr, "spot": None, "rows": [],
            "count": {k: (count or NO_COUNT)[k] for k in ref.LIMITS}}
    return case, changed


def verdicts(case: dict) -> dict:
    """The reference's two verdicts on `case`, form-only: the rate rule's whole verdict and the count's."""
    inst, tr = case["instance"], case["trusted"]
    return {"check": ref.check_detail(inst, tr, form_only=True),
            "count": ref.count(inst, tr, case["count"], form_only=True)}


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--log", type=Path, nargs="+", required=True)
    ap.add_argument("--limits", type=Path, required=True)
    ap.add_argument("--count", type=Path)
    ap.add_argument("--unregister")
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--format", action="append", default=[])
    ap.add_argument("--no-pins", action="store_true")
    ap.add_argument("--nodes", action="store_true")
    ap.add_argument("--case-only", action="store_true", help="write the case and not the reference's verdicts")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)

    record = a.record.read_bytes().decode()
    try:
        case, changed = served_case(
            a.program.read_text(), record, [json.loads(p.read_text()) for p in a.log], json.loads(a.limits.read_text()),
            json.loads(a.count.read_text()) if a.count else None,
            unregister=a.unregister, drop=tuple(a.drop), formats=tuple(a.format), pins=not a.no_pins, nodes=a.nodes)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    inst, tr = case["instance"], case["trusted"]
    print(f"case {a.name}: record SHA-512 {hashlib.sha512(record.encode()).hexdigest()[:16]}…, log of {len(tr['log'])} "
          f"records, {len(inst['pins'])} pins; changed from the run's files: {'; '.join(changed) or 'nothing'}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.case.json").write_text(line(case))
    if a.case_only:
        return 0
    got = verdicts(case)
    (a.out / f"{a.name}.reference.json").write_text(line(got))
    print(f"check: {json.dumps(got['check'], sort_keys=True)}")
    print(f"count: {json.dumps(got['count'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
