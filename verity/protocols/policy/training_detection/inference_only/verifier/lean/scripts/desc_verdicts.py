"""Compare two builds of `inference-only-verifier --policy` (`InferenceOnlyVerifier.Desc.check`) verdict by verdict.

The instances are the descriptor difftest generator's every instance, then each served descriptor under SERVED and each
descriptor in the tree's fixtures in two divisions (one inference block, and one per computing root node). Prints the
generated instances' sha256 (the lock's pin of `instances.jsonl`) and every instance whose verdicts differ, by name,
family and both outputs; writes both builds' outputs and the instances' names to OUT.

Usage: desc_verdicts.py REPO SERVED OUT JOBS EXE_A EXE_B
"""

import hashlib
import importlib.util
import json
import multiprocessing as mp
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT, SERVED, OUT, JOBS = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
EXES = sys.argv[5:7]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    sys.modules[name] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


load("stand_in", ROOT / "tests/verity/circuits/stand_in.py")
V = load("desc_difftest_vectors", Path(__file__).with_name("desc_difftest_vectors.py"))


def gen(s):
    i = V.instance(s)
    return f"gen:{s}:{i['shape']}", i["instance"]


def instances():
    V.fixtures(), V.decoys(), V.walks(), V.empties(), V.shared(), V.mistyped(), V.grids(), V.recuts()
    pins = V.vectors.pinned()
    with mp.Pool(JOBS) as p:
        made = p.map(gen, range(V.total() - len(pins)), chunksize=8)
    made += [(f"pinned:{k}", x["instance"]) for k, x in enumerate(pins)]
    print("generated", len(made), "sha256", hashlib.sha256("".join(V.line(i) for _, i in made).encode()).hexdigest())
    texts = {p.read_text().strip(): "served:" + str(p.relative_to(SERVED)) for p in sorted(SERVED.rglob("program.json"))}
    files = [ROOT / "tests/tools/difftest/lean_decode_texts.json",
             *sorted(f for n in ("select_entry.json", "slot_entry.json") for f in ROOT.rglob(n)
                     if ".venv" not in f.parts and ".lake" not in f.parts),
             *sorted((ROOT / "tests/verity/circuits").glob("*_vectors.json"))]
    for f in files:
        for k, t in enumerate(V._descriptors(json.loads(f.read_text()))):
            texts.setdefault(t, f"{f.relative_to(ROOT)}#{k}")
    big = 10 ** 30
    for text, name in texts.items():
        try:
            inp, reg, spans = V.reference.root_nodes(text)
        except (ValueError, KeyError, TypeError, IndexError):
            made.append((f"tree:{name}:unread", {"program": text, "theta": [], "owner": [], "gamma": big, "alpha": [big, 1]}))
            continue
        calls = [k for k in range(len(inp)) if not inp[k] and spans[k][1] > spans[k][0]]
        theta = [k for k in range(len(inp)) if inp[k] and reg[k]]
        for div in ("one", "each"):
            owner = ["input"] * len(inp)
            for j, k in enumerate(calls):
                owner[k] = 0 if div == "one" else j
            made.append((f"tree:{name}:{div}", {"program": text, "theta": theta, "owner": owner, "gamma": big,
                                               "alpha": [big, 1]}))
    return made


def run(exe, paths):
    def one(p):
        r = subprocess.run([exe, "--policy", str(p)], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else f"exit {r.returncode}: {r.stderr[-300:]}"
    with ThreadPoolExecutor(JOBS) as ex:
        return list(ex.map(one, paths))


def family(name):
    kind, _, rest = name.partition(":")
    return rest.rsplit(":", 1)[1] if kind == "gen" else kind


def main():
    made = instances()
    (OUT / "inst").mkdir(parents=True, exist_ok=True)
    paths = []
    for k, (_, inst) in enumerate(made):
        paths.append(OUT / "inst" / f"{k:05d}.json")
        paths[-1].write_text(json.dumps(inst, separators=(",", ":")))
    outs = [run(e, paths) for e in EXES]
    (OUT / "verdicts.json").write_text(json.dumps({"names": [n for n, _ in made], "exes": EXES, "outputs": outs}))
    diff = [k for k in range(len(made)) if outs[0][k] != outs[1][k]]
    print("instances", len(made), "differing", len(diff))
    for k in diff:
        print(f"{made[k][0]}\t{family(made[k][0])}\t{outs[0][k][:300]}\t{outs[1][k][:300]}")


if __name__ == "__main__":
    main()
