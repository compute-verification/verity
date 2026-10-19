"""Conformance of the Lean `check_cut` (`flock-verify cut-check`, PROTOCOL.md §16.7) with the reference,
`verity.ml.circuits.cut.check_cut` (PR #111): the partition invariant on one Call's cut against its graph, and the width rule.

    python cut_check_agree.py --lean BIN --out DIR [--seed S] [--n N]

Each case is a random Call graph (reads, declared widths, returned gates, parameters and the gates that read them, returned
parameters, recomputed pairs) with a random cut, committed set (derived, or served with perturbations) and width limits
(Q_word_v1's, none, or random), checked by v1's rule and again by `Q_word` v2's (`recompute="report"`, DIR/<i>-v2.json), where
the `recomputed_across` count must agree too. The first cases are `FIXED`: Q_word v1's cuts before the cycle rule whose units'
reads have a cycle. Writes DIR/<i>.json and DIR/result.json; exits 1 unless the codes, the too-wide units and the unit cycle
(`unit-cycle`'s witness) agree on every case.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

from verity.ml.circuits.cut import WIDTH_RULE, check_cut, derived_committed


def _fixed(n, reads, outputs, width, owner, X):
    return {"graph": {"n": n, "src": [p for p, _ in reads], "dst": [q for _, q in reads], "width": width, "outputs": outputs,
                      "inputs": 0, "input_reads": [], "returned_inputs": [], "total": n, "free": 0, "recomputed": []},
            "owner": owner, "committed": None, "limits": [X, 32, 0]}


#: POP1 (`qword_not_acyclic`): a chain g0 -> g1 -> g2 of 8-bit gates returned as [g0, g2, g1], packed at X = 16 into the output
#: units {g0, g2} and {g1}; and a producer join: g0's unit 0 joins g4, whose other producer g3 is in unit 1, which reads g1 of 0
FIXED = [_fixed(3, [(0, 1), (1, 2)], [0, 2, 1], [8, 8, 8], [0, 1, 0], 16),
         _fixed(5, [(0, 1), (0, 4), (1, 2), (2, 3), (3, 4)], [1, 3], [8] * 5, [0, 0, 1, 1, 0], 8)]


def cycle(v):
    """`unit-cycle`'s witness as `cut-check` gives it."""
    d = v.detail.get("unit-cycle")
    return {"units": d["units"], "reads": d["gates"]} if d else None


def case(rng):
    n = rng.randint(1, 30)
    k = rng.randint(0, 3)
    free = rng.randint(0, 3)
    reads = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 2 * n))]
    g = {"n": n, "src": [p for p, _ in reads], "dst": [q for _, q in reads],
         "width": [rng.choice([1, 8, 16, 16, 32, 64]) for _ in range(n)],
         "outputs": rng.sample(range(n), rng.randint(0, min(3, n))), "inputs": k,
         "input_reads": [[rng.randrange(k), rng.randrange(n)] for _ in range(rng.randint(0, 2 * k))] if k else [],
         "returned_inputs": rng.sample(range(k), rng.randint(0, k)) if k else [],
         "total": n + free + (rng.choice([-1, 1]) if rng.random() < 0.05 else 0), "free": free,
         "recomputed": [[rng.randrange(n), rng.randrange(n)] for _ in range(rng.choice([0, 0, 0, 1, 2]))]}
    units = rng.randint(1, 5)
    owner = [rng.randrange(units) for _ in range(n)]
    if rng.random() < 0.05:
        owner = owner[:-1] if len(owner) > 1 else owner + [0]
    committed = None
    if rng.random() < 0.3 and len(owner) == n:
        committed = derived_committed(graph(g), owner)
        if rng.random() < 0.5 and committed:
            committed.pop(rng.randrange(len(committed)))
        if rng.random() < 0.5:
            committed.append(rng.randrange(n + 1))
    limits = rng.choice([list(WIDTH_RULE), list(WIDTH_RULE), None, [rng.randint(0, 64), rng.randint(0, 64), rng.randint(0, 4)]])
    return {"graph": g, "owner": owner, "committed": committed, "limits": limits}


def graph(g):
    return SimpleNamespace(n=g["n"], src=g["src"], dst=g["dst"], width=g["width"], outputs=g["outputs"], input_ids=list(range(g["inputs"])),
                           input_reads=[tuple(x) for x in g["input_reads"]], returned_inputs=g["returned_inputs"], total=g["total"],
                           free=g["free"], recomputed=[tuple(x) for x in g["recomputed"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    files, want = [], {}
    for i in range(len(FIXED) + a.n):
        c = FIXED[i] if i < len(FIXED) else case(rng)
        for rule, name in (("refuse", f"{i}.json"), ("report", f"{i}-v2.json")):
            f = a.out / name
            f.write_text(json.dumps(c if rule == "refuse" else dict(c, recompute=rule)))
            files.append(str(f))
            v = check_cut(graph(c["graph"]), c["owner"], c["committed"], tuple(c["limits"]) if c["limits"] else None, rule)
            wide = v.detail.get("unit-too-wide", {}).get("gates", [])
            want[str(f)] = (list(v.codes), [list(x) for x in wide], v.detail.get("recomputed_across", {}).get("n"), cycle(v))
    p = subprocess.run([a.lean, "cut-check", *files], capture_output=True, text=True)
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("CUTCHECK "):
            j = json.loads(line[9:])
            got[j["file"]] = (j.get("codes", ["error: " + j.get("error", "")]), j.get("wide", [])[:64], j.get("recomputed_across"),
                              j.get("cycle"))
    bad = [f for f in files if got.get(f) != want[f]]
    seen = sorted({c for f in files for c in want[f][0]})
    v2 = [f for f in files if f.endswith("-v2.json")]
    result = {"seed": a.seed, "cases": len(files), "agree": len(files) - len(bad), "disagree": bad[:20],
              "valid": sum(1 for f in files if not want[f][0]), "codes_seen": seen,
              "fixed_refused": [want[str(a.out / f"{i}.json")][0] for i in range(len(FIXED))],
              "cycles": sum(1 for f in files if want[f][3]),
              "v2": {"cases": len(v2), "reported": sum(1 for f in v2 if want[f][2]),
                     "refused_by_v1_alone": sum(1 for f in v2 if not want[f][0] and want[f[:-8] + ".json"][0] == ["gate-recomputed"])}}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
