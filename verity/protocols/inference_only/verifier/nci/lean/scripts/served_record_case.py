"""A served run's records as one case of the record form's difftest.

The form (`NciCheck.Record`) reads an instance (the Program's text, the circuit record's text, Γ, α and the pins) and
the trusted input (the deployment's registration log and the digest of the run's own record). A served run writes
each as a file. This reads those files as they are and writes the case (`NAME.cases.jsonl`) and the reference's
results on it (`NAME.python.jsonl`), the two files `nci-record-difftest` compares.

    python served_record_case.py --program P --record C --log FIRST ... RUN --limits L --out DIR --name NAME
        [--count B] [--unregister smallest|largest|PARAMETER] [--drop registered.KEY] [--no-pins] [--case-only]

- ``--program``: the Program's v1 descriptor. ``--record``: the circuit record, the bytes the run's record commits.
- ``--log``: the registration log's records, the first first and the run's own last.
- ``--limits``: ``{"alpha": [num, den], "gamma": Γ}``, the limits the run fixed.
- ``--count``: ``{"table", "c", "k", "n"}``, the count's limits (`NciCheck.RecordCount`); without it they are empty.
- The pins are the closure digests of the Definitions the record's openings name as their readers; ``--no-pins``
  gives none.

Two options make a case the run did not write, and the case's name should say so:

- ``--unregister``: the log's first record without one parameter's entry and each later record's ``previous``
  made again, so the deployment never registered it. ``smallest`` and ``largest`` choose by words among the
  parameters the record reads that the first record registers.
- ``--drop registered.KEY``: the record without that key of its ``registered``, and the run's own record pointed
  at the new text. The run committed no such text: a stand-in, to read what the form says past that key.

The reference visits every leaf of every row of every unit in two places, more than a served class's circuit lets it:
the second half of `own_rows`, and `hands_on` and `lands_whole` under the assertions of `leaf_widths`. Here
`own_rows` is its first half, the rule's form, and those two are not asked. Everything else is the reference's own
code, so its verdict here is the form's rules without what they stand for leaf by leaf.
"""
from __future__ import annotations

import argparse
import collections
import functools
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record_difftest_vectors as g  # noqa: E402

ref, reference = g.ref, g.reference
T0 = time.monotonic()


def said(*a) -> None:
    print(f"[{time.monotonic() - T0:6.0f} s]", *a, flush=True)


def record_bytes(doc: dict) -> str:
    """A circuit record's text as a serve writes it: compact, keys sorted, one newline."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"


def rechained(log: list[dict]) -> list[dict]:
    """The log with each record's ``previous`` the digest of the record before it."""
    out: list[dict] = []
    for rec in log:
        out.append({**rec, "previous": reference.record_digest(out[-1]) if out else None})
    return out


def own_rows_form(program) -> None:
    """`ref.own_rows`' first half: every argument of a root node that is no input node is affine runs of the root's
    nodes' leaves, and a batch maps every argument over its members."""
    for node in program.root.body.nodes:
        if ref.is_input(node.fn):
            continue
        if any(not (isinstance(p, ref.Affine) and p.space == "n") for a in node.args for p in ref._parts(a.refs)) \
                or node.form == "batch" and None in node.axes:
            raise ref.refused("own-rows", ref.OWN_ROWS)


def once(fn):
    """`fn(inst, tr)`, run once for each case and its refusal kept."""
    memo: dict[tuple[int, int], tuple] = {}

    def run(inst, tr):
        key = (id(inst), id(tr))
        if key not in memo:
            try:
                memo[key] = (fn(inst, tr), None)
            except ref.Malformed as e:
                memo[key] = (None, e)
        got, e = memo[key]
        if e is not None:
            raise e
        return got
    return run


def short(tid: str) -> str:
    return tid.split("{")[0]


def shape(ws) -> str:
    return f"{len(ws)}x{sorted(set(ws))}"


def why(e: BaseException) -> str:
    return f"{getattr(e, 'code', type(e).__name__)}" + (f": {e.message}" if hasattr(e, "message") else "") \
        + (f" | from {type(e.__cause__).__name__}: {str(e.__cause__)[:700]}" if e.__cause__ is not None else "")


def stages(inst: dict, tr: dict) -> None:
    """The form's stages as far as `leaf-widths`, each said as it is taken, and every wire under the rule, not
    only the first it refuses."""
    try:
        i, program, _desc, rec, T, templates = ref.front(inst, tr)
    except ref.Malformed as e:
        said("REFUSED before the wires:", why(e))
        return
    units = sum(t.units for t in templates)
    said(f"taken as far as own-rows: {len(templates)} templates, {units} units, {len(T['log'])} records in the log, "
         f"the run's at {T['run']}")
    try:
        wires, cuts = ref.wires_of(templates, rec, (i["order"],) if "order" in i else None)
    except ref.Malformed as e:
        said("REFUSED at the wires:", why(e))
        return
    by = {t.tid: t for t in templates}

    @functools.cache
    def port(tid: str, p: str) -> tuple[int, frozenset[int], str]:
        ws = by[tid].widths[p]
        return sum(ws), frozenset(ws), shape(ws)

    how: collections.Counter = collections.Counter()
    which: collections.Counter = collections.Counter()
    for w in wires:
        (bits, out, oshape), (_, row, rshape) = port(w["writer"], w["out"]), port(w["reader"], w["port"])
        if bits != 16 * by[w["writer"]].port(w["out"], outs=True):
            way = "REFUSED, the writer's output is not whole 16-bit words"
        elif w["port"] in by[w["reader"]].closed:
            way = "taken into a closed row"
        elif len(row) == 1 and len(out) == 1 and min(row) % min(out) == 0:
            way = "taken as built"
        else:
            way = "REFUSED, the two sides' widths"
        how[way] += 1
        if way.startswith("REFUSED"):
            which[(way, f"{short(w['writer'])}.{w['out']} {oshape}", f"{short(w['reader'])}.{w['port']} {rshape}")] += 1
    said(f"leaf-widths, every wire: {len(wires)} wires, " + ", ".join(f"{n} {way}" for way, n in sorted(how.items())))
    for (way, writer, reader), n in sorted(which.items(), key=lambda kv: -kv[1])[:40]:
        print(f"      {way}: {writer} -> {reader} ({n} wires)")
    closed = sorted((short(t.tid), tuple(t.closed)) for t in templates if t.closed)
    said("templates with closed rows:", len(closed), "|", dict(collections.Counter(closed)))


def counted(inst: dict, tr: dict) -> None:
    """What the summary is the greatest of, slot by slot, and which registered parameters are θ."""
    try:
        e = ref.expand(inst, tr)
    except ref.Malformed:
        return
    for k in ("gates", "inputs", *ref.KINDS, "fresh"):
        said(f"   {k}, slot by slot: {e[k]}")
    x, T = e["x"], ref.front(inst, tr)[4]
    theta = {p: ref.theta(T, x.params, p) for p in x.params}
    said(f"registered parameters: {len(theta)}, θ {sum(theta.values())}, not θ "
         f"{dict(collections.Counter(p.split('/')[0] for p, t in theta.items() if not t))}")
    said("pinned readers:", sorted(short(t) for t, d in e["pins"].items() if d is not None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--log", type=Path, nargs="+", required=True)
    ap.add_argument("--limits", type=Path, required=True)
    ap.add_argument("--count", type=Path)
    ap.add_argument("--unregister")
    ap.add_argument("--drop", action="append", default=[])
    ap.add_argument("--no-pins", action="store_true")
    ap.add_argument("--case-only", action="store_true", help="write the case and not the reference's results")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    program, record = a.program.read_text(), a.record.read_bytes().decode()
    log = [json.loads(p.read_text()) for p in a.log]
    limits = json.loads(a.limits.read_text())
    count = json.loads(a.count.read_text()) if a.count else {"table": {}, "c": 0, "k": 0, "n": 0}
    desc, doc = json.loads(program), json.loads(record)
    said(f"case {a.name}: Program {len(program)} bytes, record {len(record)} bytes "
         f"(SHA-512 {hashlib.sha512(record.encode()).hexdigest()[:16]}…), log of {len(log)} records")

    changed = []
    for key in a.drop:
        head, _, k = key.partition(".")
        if head != "registered" or k not in doc.get("registered", {}):
            sys.exit(f"--drop {key}: the record's registered has no such key")
        if record_bytes(doc) != record:
            sys.exit("--drop: the record is not in the form a serve writes, so its text can't be made again")
        del doc["registered"][k]
        record = record_bytes(doc)
        log[-1] = {**log[-1], "window": {**log[-1]["window"], "circuit": hashlib.sha512(record.encode()).hexdigest()}}
        changed.append(f"the record without registered.{k}, the run's record pointed at it (a stand-in)")
    if a.unregister:
        first = {e.get("value") for e in log[0]["registered"]}
        read = sorted((n, p) for p, n in doc["registered"]["params"].items() if p in first)
        name = {"smallest": read[0][1], "largest": read[-1][1]}.get(a.unregister, a.unregister)
        kept = [e for e in log[0]["registered"] if e.get("value") != name]
        if len(kept) == len(log[0]["registered"]):
            sys.exit(f"--unregister {name}: the log's first record registers no such value")
        log[0] = {**log[0], "registered": kept}
        changed.append(f"the log's first record without {name} ({doc['registered']['params'].get(name)} words in the "
                       f"record), {len(kept)} entries left")
    if changed:
        log = rechained(log)
    said("changed from the run's files:", "; ".join(changed) or "nothing")

    readers = sorted({o["reader"] for o in doc.get("registered", {}).get("openings", [])})
    pins = [] if a.no_pins else sorted({ref.closure_digest(desc, t) for t in readers})
    said(f"pins: {len(pins)}, the closure digests of {[short(t) for t in readers] if pins else []}: "
         f"{[p[:16] + '…' for p in pins]}")
    del desc
    inst = {"program": program, "record": record, "gamma": limits["gamma"], "alpha": limits["alpha"], "pins": pins}
    tr = {"log": log, "run": reference.record_digest(log[-1]), "params": []}
    said(f"Γ = {inst['gamma']}, α = {inst['alpha'][0]}/{inst['alpha'][1]}; the run's record {tr['run'][:16]}…, "
         f"the count's limits c {count['c']}, k {count['k']}, n {count['n']} over a table of {len(count['table'])}")
    case = {"instance": inst, "trusted": tr, "spot": None, "rows": [],
            "count": {k: count[k] for k in ("table", "c", "k", "n")}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.cases.jsonl").write_text(g.line(case))
    (a.out / f"{a.name}.cases.written").write_text("")
    said("wrote the case")
    if a.case_only:
        return 0

    ref.own_rows = own_rows_form
    ref.hands_on = lambda *_: False
    ref.lands_whole = lambda *_: True
    ref.front, ref.expand = once(ref.front), once(ref.expand)
    stages(inst, tr)
    res = g.results(case)
    (a.out / f"{a.name}.python.jsonl").write_text(g.line(res))
    said("THE REFERENCE'S VERDICT, word for word:", json.dumps(res["detail"]))
    said("the reference's count:", json.dumps(res["count"]))
    if res["detail"].get("error") == "inputs":
        said("the record's registered has keys", sorted(doc.get("registered", {})), "and the form reads",
             sorted(ref.REGISTERED | getattr(ref, "REGISTERED_UNREAD", {"values"})))
    counted(inst, tr)
    said("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
