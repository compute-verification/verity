"""Conformance of the Lean circuit-type check (`flock-verify circuit-type`, `Flock/CircuitType.lean`) with the reference,
`verity_flock.circuit_types`: the same digest for every type, and the same verdict (accepted or refused).

    python circuit_type_agree.py --lean BIN --out DIR [--seed S] [--n N]

Each case is an archive, callees first, ending in the type under test: random flat types (ANDs over random forms, reads of
toy program tables), random call trees over them, the pinned vectors, and one mutation per rule of each. Writes DIR/<i>.json
and DIR/result.json; exits 1 unless the Lean verifier's digests and verdicts equal the reference's on every type.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from verity.ml.circuits import library
from verity_flock import circuit_types as CT

VECTORS = Path(__file__).resolve().parents[1] / "tests" / "circuit_type_vectors.json"


def toy_table(rng, ib, vb):
    words = [rng.getrandbits(vb) for _ in range(1 << ib)]
    return hashlib.sha512(b"".join(w.to_bytes(4, "little") for w in words)).hexdigest()


def form(rng, n, k=3):
    ws = sorted(rng.sample(range(n), min(n, rng.randint(1, k)))) if n else []
    return (tuple(ws), rng.randint(0, 1))


def flat(rng, tables):
    """A random flat type: ANDs and reads over random forms, every item live."""
    in_bits = rng.randint(1, 6)
    items, n, tlist = [], in_bits, []
    for _ in range(rng.randint(1, 6)):
        if tables and rng.random() < 0.25:
            entry = rng.choice(tables)
            if entry not in tlist:
                tlist.append(entry)
            items.append((CT.READ, tlist.index(entry), tuple(form(rng, n) for _ in range(entry[1]))))
            n += entry[2]
        else:
            items.append((CT.AND, form(rng, n), form(rng, n)))
            n += 1
    return _close(CT.CircuitType(library.PROGRAM, ((in_bits, 1),), ((1, 1),), (), tuple(tlist), tuple(items), ((), )), n, rng)


def _close(t, n, rng):
    """Outputs that read every item's last wire, so nothing is dead."""
    outs, lo, starts = [], t.in_bits, []
    for it in t.items:
        w = 1 if it[0] == CT.AND else t.tables[it[1]][2] if it[0] == CT.READ else _width[it[1]]
        starts.append((lo, w))
        lo += w
    for s, w in starts:
        outs.append(((s + w - 1,), rng.randint(0, 1)))
    return dataclasses.replace(t, outs=((len(outs), 1),), outputs=tuple(outs))


_width: dict = {}


def tree(rng, pool):
    """A random caller of types in ``pool``: calls on random forms, plus an AND, every item live."""
    in_bits = rng.randint(1, 6)
    items, n, calls, widths = [], in_bits, [], []
    for _ in range(rng.randint(1, 3)):
        u = rng.choice(pool)
        d = u.digest()
        if d not in calls:
            calls.append(d)
        items.append((CT.CALL, calls.index(d), tuple(form(rng, n) for _ in range(u.in_bits))))
        widths.append(u.out_bits)
        n += u.out_bits
    items.append((CT.AND, form(rng, n), form(rng, n)))
    widths.append(1)
    n += 1
    outs, lo = [], in_bits
    for w in widths:
        outs.append(((lo + w - 1,), 0))
        lo += w
    return CT.CircuitType(library.PROGRAM, ((in_bits, 1),), ((len(outs), 1),), tuple(calls), (), tuple(items), tuple(outs))


def mutations(t, rng):
    """One broken copy of ``t`` per rule that applies to it."""
    out = [("label", dataclasses.replace(t, label="public")), ("ports", dataclasses.replace(t, ins=((t.in_bits, 0),)))]
    first = t.items[0]
    if first[0] == CT.AND:
        a = first[1]
        out.append(("forward", dataclasses.replace(t, items=((CT.AND, ((t.in_bits,), 0), first[2]),) + t.items[1:])))
        out.append(("constant", dataclasses.replace(t, items=((CT.AND, (a[0], 2), first[2]),) + t.items[1:])))
        if len(a[0]) >= 2:
            out.append(("unsorted", dataclasses.replace(t, items=((CT.AND, (tuple(reversed(a[0])), a[1]), first[2]),) + t.items[1:])))
    if t.calls:
        k = next(i for i, it in enumerate(t.items) if it[0] == CT.CALL)
        it = t.items[k]
        out.append(("arity", dataclasses.replace(t, items=t.items[:k] + ((CT.CALL, it[1], it[2][:-1]),) + t.items[k + 1:])))
        out.append(("digest", dataclasses.replace(t, calls=("X" * 128,) + t.calls[1:])))
        out.append(("library-calls-program", dataclasses.replace(t, label=library.LIBRARY)))
        if len(t.calls) >= 2:
            out.append(("first-use", dataclasses.replace(t, calls=(t.calls[1], t.calls[0]) + t.calls[2:])))
    if t.tables:
        k = next(i for i, it in enumerate(t.items) if it[0] == CT.READ)
        it = t.items[k]
        out.append(("read-arity", dataclasses.replace(t, items=t.items[:k] + ((CT.READ, it[1], it[2][:-1]),) + t.items[k + 1:])))
        out.append(("library-reads-program", dataclasses.replace(t, label=library.LIBRARY)))
        out.append(("table-shape", dataclasses.replace(t, tables=((t.tables[0][0], 0, t.tables[0][2]),) + t.tables[1:])))
    out.append(("outputs", dataclasses.replace(t, outputs=t.outputs[:-1])))
    out.append(("dead", dataclasses.replace(t, items=t.items + ((CT.AND, ((0,), 0), ((0,), 1)),))))
    out.append(("bad-op", {"content": {**t.content(), "items": [[5, 0, 0]] + t.content()["items"]}}))
    return out


def reference(archive):
    """The reference's digest and verdict for each type of an archive, a type a callee only once accepted."""
    known, res = {}, []

    def callee(d):
        if d not in known:
            raise CT.InvalidType(f"no callee {d}")
        return known[d]

    for c in archive:
        try:
            t = CT.from_content(c)
        except Exception as e:  # a content the reference can't even read is refused
            res.append({"digest": None, "ok": False, "error": repr(e)})
            continue
        d = t.digest()
        try:
            CT.validate(t, callee)
        except CT.InvalidType as e:
            res.append({"digest": d, "ok": False, "error": str(e)})
            continue
        known[d] = t
        res.append({"digest": d, "ok": True})
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = [(toy_table(rng, ib, vb), ib, vb) for ib, vb in ((2, 3), (3, 1), (4, 8))]
    cases = [[v["content"] for v in json.loads(VECTORS.read_text())]]
    for _ in range(a.n):
        pool = [flat(rng, tables) for _ in range(rng.randint(1, 3))]
        for u in pool:
            _width[u.digest()] = u.out_bits
        caller = tree(rng, pool)
        _width[caller.digest()] = caller.out_bits
        subject = rng.choice(pool + [caller])
        deps = [u.content() for u in pool] if subject is caller else []
        cases.append(deps + [subject.content()])
        for _name, m in mutations(subject, rng):
            cases.append(deps + [m["content"] if isinstance(m, dict) else m.content()])
    files = []
    for i, archive in enumerate(cases):
        f = out / f"{i}.json"
        f.write_text(json.dumps({"types": archive}))
        files.append(str(f))
    p = subprocess.run([a.lean, "circuit-type", *files], capture_output=True, text=True, check=False)
    lean: dict = {}
    for line in p.stdout.splitlines():
        if line.startswith("CIRCUITTYPE "):
            r = json.loads(line[len("CIRCUITTYPE "):])
            lean.setdefault(r["file"], []).append(r)
    bad, accepted, refused = [], 0, 0
    for f, archive in zip(files, cases):
        want = reference(archive)
        got = lean.get(f, [])
        if len(got) != len(want):
            bad.append({"file": f, "why": f"{len(got)} Lean verdicts for {len(want)} types"})
            continue
        for w, g in zip(want, got):
            if w["ok"] != g["ok"] or (w["digest"] is not None and w["digest"] != g.get("digest")):
                bad.append({"file": f, "index": g.get("index"), "reference": w, "lean": g})
            accepted += w["ok"]
            refused += not w["ok"]
    result = {"cases": len(cases), "types_accepted": accepted, "types_refused": refused, "disagreements": bad[:20],
              "n_disagreements": len(bad)}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "disagreements"}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
