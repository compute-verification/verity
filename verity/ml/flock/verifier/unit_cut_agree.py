"""Conformance of the Lean partition check (`flock-verify unit-cut`, PROTOCOL.md §16.7) with the reference,
`verity.ml.circuits.partition.validate_unit_cut`, on random cuts.

    python unit_cut_agree.py --lean BIN --out DIR [--seed S] [--n N]

Each cut is a small circuit with random owners, input gates, reads, outputs, committed gates and recomputes, mostly valid,
then perturbed so that every failure code occurs, and is checked twice: by v1's rule, and by `Q_word` v2's
(`recompute="report"`, DIR/<i>-v2.json), where the codes and the `recomputed_across` count must agree. Writes DIR/<i>.json and
DIR/result.json; exits 1 unless the Lean verifier's codes equal the reference's on every cut.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path

from verity.ml.circuits.partition import INPUT_UNIT, validate_unit_cut


def cut(rng):
    n = rng.randint(1, 24)
    units = rng.randint(1, 4)
    owner = [INPUT_UNIT if rng.random() < 0.25 else rng.randrange(units) for _ in range(n)]
    inputs = [g for g in range(n) if owner[g] == INPUT_UNIT]
    reads = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 2 * n))]
    outputs = rng.sample(range(n), rng.randint(0, min(3, n)))
    boundary = {p for p, q in reads if owner[p] != owner[q]} | set(outputs)
    committed = sorted(boundary)
    recomputed = []
    structure = rng.randint(0, 5)
    certified = sum(1 for g in range(n) if g not in inputs)
    total = certified + len(inputs) + structure
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        k = rng.randrange(9)
        if k == 0 and n:
            owner[rng.randrange(n)] = rng.choice([INPUT_UNIT, -1, "x", units])
        elif k == 1:
            inputs.append(rng.randrange(n + 2))
        elif k == 2 and committed:
            committed.pop(rng.randrange(len(committed)))
        elif k == 3:
            committed.append(rng.randrange(n + 2))
        elif k == 4:
            total += rng.choice([-1, 1])
        elif k == 5:
            recomputed.append((rng.randrange(n), rng.randrange(n)))
        elif k == 6:
            recomputed.append((rng.randrange(n + 2), rng.randrange(n + 2)))
        elif k == 7:
            outputs.append(rng.randrange(n))
        else:
            reads.append((rng.randrange(n), rng.randrange(n)))
    return {"total_gates": max(total, 0), "structure_gates": structure, "owner": owner, "reads": [list(r) for r in reads],
            "outputs": outputs, "committed": committed, "input_gates": inputs, "recomputed": [list(r) for r in recomputed]}


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
    for i in range(a.n):
        c = cut(rng)
        for rule, name in (("refuse", f"{i}.json"), ("report", f"{i}-v2.json")):
            case = c if rule == "refuse" else dict(c, recompute=rule)
            f = a.out / name
            f.write_text(json.dumps(case))
            files.append(str(f))
            v = validate_unit_cut(c["total_gates"], c["structure_gates"], c["owner"], [tuple(r) for r in c["reads"]], c["outputs"],
                                  c["committed"], c["input_gates"], [tuple(r) for r in c["recomputed"]], rule)
            want[str(f)] = (list(v.codes), v.detail["recomputed_across"]["n"] if rule == "report" else None)
    p = subprocess.run([a.lean, "unit-cut", *files], capture_output=True, text=True)
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("CUT "):
            j = json.loads(line[4:])
            got[j["file"]] = (j.get("codes", ["error: " + j.get("error", "")]), j.get("recomputed_across"))
    bad = [f for f in files if got.get(f) != want[f]]
    seen = sorted({c for f in files for c in want[f][0]})
    v2 = [f for f in files if f.endswith("-v2.json")]
    result = {"seed": a.seed, "cuts": len(files), "agree": len(files) - len(bad), "disagree": bad[:20],
              "valid": sum(1 for f in files if not want[f][0]), "codes_seen": seen,
              "v2": {"cuts": len(v2), "reported": sum(1 for f in v2 if want[f][1]),
                     "refused_by_v1_alone": sum(1 for f in v2 if not want[f][0] and want[f[:-8] + ".json"][0] == ["gate-recomputed"])}}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
