"""Conformance of the Lean `Q_word` v1 evaluator (`flock-verify evaluate`, PROTOCOL.md §16.7) with core's (`verity.ml.circuits.cut`,
`verity.ml.circuits.partition_object`, PR #111), over Call graphs (descriptor decoding and graph extraction are not ported yet).

    python qword_agree.py --lean BIN --out DIR [--seed S] [--n N] [--max-gates G] [--vectors partition_vectors.json]

Random cases, each checked against core's own objects:
- a flat Call graph: `cut_word`'s owners, unit kinds and heads;
- a Definition tree (primitives, separable bodies of members, flat bodies): `DefinitionCut`'s owners, units and `locate`;
- a Program of such Calls: `Partition.locate` for every unit, and each flat Call's derived committed set.
`--vectors` also checks #111's pinned vector: each Call's owners, units and committed set, the population and `locate`. Its
separable Call's member graphs are sliced from the Call's graph at the members `locate` names (extraction's part, which the
vector does not carry). Writes DIR/<i>.json and DIR/result.json; exits 1 unless every case agrees.
"""
import argparse, json, random, subprocess, sys
from array import array
from pathlib import Path
from types import SimpleNamespace

from verity.ml.circuits.cut import DefinitionCut, cut_word, derived_committed
from verity.ml.circuits.partition_object import Partition

MAX_GATES = 12


def graph_json(rng, n):
    edges = [(p, q) for q in range(n) for p in range(q) if rng.random() < 2.0 / max(n, 1)]
    rng.shuffle(edges)
    outs = [rng.randrange(n) for _ in range(rng.randint(1, min(4, n)))] if n else []
    return {"n": n, "src": [p for p, _ in edges], "dst": [q for _, q in edges], "width": [rng.choice([1, 8, 16, 32]) for _ in range(n)],
            "outputs": outs, "inputs": 0, "input_reads": [], "returned_inputs": [], "total": n, "free": 0, "recomputed": []}


def core_graph(g):
    return SimpleNamespace(n=g["n"], src=array("q", g["src"]), dst=array("q", g["dst"]), width=array("q", g["width"]), outputs=list(g["outputs"]))


def random_def(rng, depth=0):
    r = rng.random()
    if r < 0.2:
        return {"prim": {"params": rng.random() < 0.8, "wiring": rng.random() < 0.2}}
    if r < 0.45 and depth < 2:
        return {"separable": [[random_def(rng, depth + 1), rng.randint(1, 3)] for _ in range(rng.randint(1, 3))]}
    return {"flat": graph_json(rng, rng.randint(1, MAX_GATES))}


def core_cut(d, X):
    """Core's DefinitionCut for the tree, built as `evaluate_definition` builds it."""
    if "prim" in d:
        p = d["prim"]
        return DefinitionCut(None, [], []) if (not p["params"] or p["wiring"]) else DefinitionCut(None, [0], ["output"])
    if "separable" in d:
        parts = []
        for sub, m in d["separable"]:
            c = core_cut(sub, X)
            parts.append((c if c.gates else None, m))
        return DefinitionCut(None, parts=parts)
    own, kinds, _ = cut_word(core_graph(d["flat"]), X)
    return DefinitionCut(None, own, kinds)


def core_locate(calls, i):
    P = Partition.__new__(Partition)
    P.calls, P._start, t = calls, [], 0
    for _k, c in calls:
        P._start.append(t)
        t += c.units
    P.population = t
    p, k, u, path = P.locate(i)
    return [i, [p, k, u], [list(x) for x in path]]


def expected(case):
    X, calls, out = case["X"], [], []
    for c in case["calls"]:
        dc = core_cut(c["def"], X)
        calls.append((c["call"], dc))
        e = {"call": c["call"], "owner": dc.owner(), "units": dc.units}
        if "graph" in c:
            e["committed"] = derived_committed(core_graph(c["graph"]), dc.owner()) if len(dc.owner()) == c["graph"]["n"] else []
        if "flat" in c["def"]:
            own, kinds, head = cut_word(core_graph(c["def"]["flat"]), X)
            e["kinds"], e["head"] = kinds, head
        out.append(e)
    pop = sum(dc.units for _, dc in calls)
    return {"calls": out, "population": pop, "locate": [core_locate(calls, i) for i in range(pop)]}


def random_case(rng):
    X = rng.choice([16, 16, 32, 8, 64])
    calls = []
    for k in range(rng.randint(1, 4)):
        d = random_def(rng)
        c = {"call": k + rng.randint(0, 2), "def": d}
        if "flat" in d:
            c["graph"] = d["flat"]
        calls.append(c)
    return {"X": X, "calls": calls}


def vector_case(v):
    """#111's pinned vector as an `evaluate` case: flat Calls from their graphs; a separable Call from its members' graphs."""
    paths = {}
    for i, (p, k, u), path in v["locate"]:
        paths.setdefault(p, []).append(path)
    calls = []
    for p, c in enumerate(v["calls"]):
        g = c["graph"]
        gj = {"n": g["n"], "src": g["src"], "dst": g["dst"], "width": g["width"], "outputs": g["outputs"],
              "inputs": 1 + max([i for i, _ in g["input_reads"]], default=-1), "input_reads": g["input_reads"],
              "returned_inputs": g["returned_inputs"], "total": g["total"], "free": g["structure"], "recomputed": g["recomputed"]}
        ps = [x for x in paths.get(p, []) if x]
        if not ps:
            d = {"flat": gj}
        else:
            # one separable node whose members are equal, contiguous slices of the Call's gates
            members = 1 + max(x[0][1] for x in ps)
            size = g["n"] // members
            sub = {"n": size, "src": [a for a, b in zip(g["src"], g["dst"]) if a < size and b < size],
                   "dst": [b for a, b in zip(g["src"], g["dst"]) if a < size and b < size], "width": g["width"][:size],
                   "outputs": [o for o in g["outputs"] if o < size], "inputs": 0, "input_reads": [], "returned_inputs": [],
                   "total": size, "free": 0, "recomputed": []}
            d = {"separable": [[{"flat": sub}, members]]}
        calls.append({"call": c["call"], "def": d, "graph": gj})
    want = {"calls": [{"call": c["call"], "owner": c["owner"], "units": c["units"], "committed": c["committed"]} for c in v["calls"]],
            "population": v["population"], "locate": v["locate"]}
    return {"X": v["object"]["query"]["params"]["X"], "calls": calls}, want


def lean(binary, files):
    p = subprocess.run([binary, "evaluate", *files], capture_output=True, text=True)
    got = {}
    for line in p.stdout.splitlines():
        if line.startswith("EVALUATE "):
            j = json.loads(line[9:])
            got[j["file"]] = j
    return got


def same(g, want):
    if not g or not g.get("ok"):
        return False
    if g["population"] != want["population"] or g["locate"] != want["locate"] or len(g["calls"]) != len(want["calls"]):
        return False
    return all({k: gc.get(k) for k in wc} == wc for gc, wc in zip(g["calls"], want["calls"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--vectors", type=Path, default=None)
    ap.add_argument("--max-gates", type=int, default=12, help="the largest random flat Call graph")
    a = ap.parse_args()
    global MAX_GATES
    MAX_GATES = a.max_gates
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    files, want = [], {}
    for i in range(a.n):
        c = random_case(rng)
        f = a.out / f"{i}.json"
        f.write_text(json.dumps(c))
        files.append(str(f))
        want[str(f)] = expected(c)
    if a.vectors:
        c, w = vector_case(json.loads(a.vectors.read_text()))
        f = a.out / "vectors.json"
        f.write_text(json.dumps(c))
        files.append(str(f))
        want[str(f)] = w
    got = lean(a.lean, files)
    bad = [f for f in files if not same(got.get(f), want[f])]
    kinds = sorted({k for f in files for c in want[f]["calls"] for k in c.get("kinds", [])})
    result = {"seed": a.seed, "cases": len(files), "agree": len(files) - len(bad), "disagree": bad[:20], "unit_kinds_seen": kinds,
              "units": sum(want[f]["population"] for f in files), "vectors": bool(a.vectors) and str(a.out / "vectors.json") not in bad}
    (a.out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
