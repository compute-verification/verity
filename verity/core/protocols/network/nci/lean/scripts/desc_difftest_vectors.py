"""The description-sized check's difftest data: `difftest_vectors.py`'s declarations, divided into IUs of root nodes, for
`NciCheck/DescDifftestMain.lean`.

    python scripts/desc_difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes three files under `DIR/desc_difftest/`, one line per instance: `instances.jsonl`, as `NciCheck.Desc.check` reads
it (`NciCheck.Desc`'s docstring: θ and the owners per root node); `expanded.jsonl`, the same instance expanded gate by
gate as `NciCheck.check` reads it (`nci_reference.expand`), or `null` where `NciCheck.Desc` refuses it; and `python.jsonl`,
`nci_reference.check_desc`'s results. The Lean audit runs `NciCheck/DescDifftestMain.lean` on the three (`runs` in
`lean-audit.json`), which fails unless `NciCheck.Desc.check` gives Python's result on every line and `NciCheck.check`'s on
every expansion: the description-sized check against the unrolled one, on the same declarations.

Instance `seed`'s program is `difftest_vectors.instance(seed)`'s. θ is no input node, every one, the registered ones or a
random set. A division puts every Call in one IU, each in its own, runs of consecutive Calls or a random assignment, and
a Call without gates in any IU or none. `Γ` and `α` are set as `difftest_vectors` sets them, at, below or above the
division's largest IU and NCI. About a third of the instances are malformed, each in one way `NciCheck.Desc` refuses.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from fractions import Fraction
from pathlib import Path

import verity.ml.circuits

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vectors = _load("difftest_vectors")
reference = vectors.reference

COUNT = 3000
MALFORMED = ["instance", "params", "program", "not-topological", "theta-unknown", "theta-not-input", "theta-duplicate",
             "owner-length", "owner-entry", "input-in-iu", "gate-in-no-iu", "empty-iu"]


def _registered_nodes(text: str) -> list[int]:
    desc = json.loads(text)
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    return [k for k, n in enumerate(nodes) if desc["callees"][n["fn"]].startswith("RegisteredInput")]


def division(rng, calls: list[int]) -> tuple[str, list[int]]:
    """A division of the Calls with gates: (strategy, IU of each)."""
    return vectors.division(rng, calls)


def instance(seed: int) -> dict:
    """Instance `seed`, with what it was made from (`shape`, `theta_kind`, `division`, `malformed`)."""
    rng = random.Random(seed)
    text, kind, _ = vectors.program(rng, seed)
    inp, spans = reference.root_nodes(text)
    K = len(inp)
    inputs = [k for k in range(K) if inp[k]]
    calls = [k for k in range(K) if not inp[k] and spans[k][1] > spans[k][0]]
    empty = [k for k in range(K) if not inp[k] and spans[k][1] == spans[k][0]]
    theta_kind = rng.choice(["none", "all", "registered", "random"])
    theta = {"none": [], "all": inputs, "registered": _registered_nodes(text),
             "random": [k for k in inputs if rng.random() < 0.5]}[theta_kind]
    theta = rng.sample(theta, len(theta))
    how, labels = division(rng, calls)
    owner: list = ["input"] * K
    for k, u in zip(calls, labels, strict=True):
        owner[k] = u
    for k in empty:
        owner[k] = rng.choice(["input", rng.randint(0, max(labels, default=0))])
    inst = {"program": text, "theta": theta, "owner": owner, "gamma": 0, "alpha": [0, 1]}
    N = spans[-1][1] if spans else 0
    w, _, _ = reference.circuit(text)
    probe = reference.check_desc(inst | {"gamma": N, "alpha": [sum(w) + 1, 1]})
    ius = probe["ius"]
    biggest = max((s for s, _ in ius), default=0)
    crit = max((Fraction(r, s) for s, r in ius), default=Fraction(0))
    inst["gamma"] = max(0, biggest + rng.choices([0, -1, 1, rng.randint(-3, 3)], [5, 3, 2, 1])[0])
    inst["alpha"] = vectors._alpha(rng, crit)
    bad = rng.choice(MALFORMED) if rng.random() < 0.35 else None
    if bad:
        _break(rng, inst, bad, inp, spans)
    return {"instance": inst, "shape": kind, "theta_kind": theta_kind, "division": how, "malformed": bad}


def _break(rng, inst: dict, bad: str, inp: list[bool], spans: list[tuple[int, int]]) -> None:
    """Make `inst` malformed in the way `bad` names."""
    K = len(inp)
    owner, theta = inst["owner"], inst["theta"]
    calls = [k for k in range(K) if not inp[k] and spans[k][1] > spans[k][0]]
    inputs = [k for k in range(K) if inp[k] and spans[k][1] > spans[k][0]]
    if bad in ("instance", "params", "program", "not-topological"):
        vectors._break(rng, inst, bad, [])
    elif bad == "theta-unknown":
        theta.insert(rng.randint(0, len(theta)), rng.choice([K, K + rng.randint(1, 5), -1, "0", None, 1.5]))
    elif bad == "theta-not-input" and calls:
        theta.insert(rng.randint(0, len(theta)), rng.choice(calls))
    elif bad == "theta-duplicate" and inputs:
        k = rng.choice(theta or inputs)
        if not theta:
            theta.append(k)
        theta.insert(rng.randint(0, len(theta)), k)
    elif bad == "owner-length":
        if rng.random() < 0.5 and owner:
            owner.pop(rng.randrange(len(owner)))
        else:
            owner.append(rng.choice(["input", 0]))
    elif bad == "owner-entry" and owner:
        owner[rng.randrange(K)] = rng.choice([-1, 1.5, None, "x", True, [0], "Input"])
    elif bad == "input-in-iu" and inputs:
        owner[rng.choice(inputs)] = rng.choice([0, max([o for o in owner if o != "input"], default=0)])
    elif bad == "gate-in-no-iu" and calls:
        owner[rng.choice(calls)] = "input"
    elif bad == "empty-iu" and calls:
        top = max(o for k, o in enumerate(owner) if k in calls)
        owner[rng.choice(calls)] = rng.choice([top + 2, top + 1 + K, 10 ** 15])


def expanded(inst: dict, result: dict):
    """The instance gate by gate, where `NciCheck.Desc` gives a verdict."""
    return None if "error" in result else reference.expand(inst)


def line(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True) + "\n"


NAMES = ("instances", "expanded", "python")


def write(out: Path, count: int = COUNT) -> dict[str, Path]:
    """The three files for instances 0 … count − 1 under ``out/desc_difftest``."""
    d = out / "desc_difftest"
    d.mkdir(parents=True, exist_ok=True)
    paths = {f"desc_difftest/{n}.jsonl": d / f"{n}.jsonl" for n in NAMES}
    files = [open(paths[f"desc_difftest/{n}.jsonl"], "w") for n in NAMES]
    try:
        for seed in range(count):
            inst = instance(seed)["instance"]
            res = reference.check_desc(inst)
            for f, obj in zip(files, (inst, expanded(inst, res), res), strict=True):
                f.write(line(obj))
    finally:
        for f in files:
            f.close()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose verity.ml.circuits reads the programs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    here = Path(verity.ml.circuits.__file__).resolve()
    if not here.is_relative_to(a.verity.resolve()):
        raise SystemExit(f"verity.ml.circuits comes from {here}, not from the checkout {a.verity}")
    write(a.out, a.count)


if __name__ == "__main__":
    main()
