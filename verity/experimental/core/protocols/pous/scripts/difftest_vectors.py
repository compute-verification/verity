"""The difftest's data: instances, and the Python verifier's answers on each, for `PousCheck/DifftestMain.lean`.

    python scripts/difftest_vectors.py <verity checkout> --out DIR [--count N]

Writes `DIR/difftest/instances.jsonl`, one instance per line, and `DIR/difftest/python.jsonl`, the same line's answers
from the checkout's Python, each line canonical JSON (sorted keys, no whitespace). `PousCheck.Difftest` recomputes every
answer from the instance and fails on any line that differs. The Lean audit runs it (`runs` in `lean-audit.json`, which
pins both files' SHA-256).

`merkle` instances are trees of random blocks under either leaf (the spec's `reference.leaf`, the generic protocol's
`verifier.leaf_hash`): every leaf, the root, the SHA-512 of every audit path (canonical JSON of the list), each path's
exact length, and `verify_path` (`wire.verify_path`, `verifier.verify_path`) on each leaf's honest path and on a flipped,
shortened, lengthened, swapped (leaf for sibling), odd-length or misindexed one, and on an index past the tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

COUNT = 400


def _python(repo: Path):
    sys.path.insert(0, str(repo / "benchmarks" / "pous"))
    from p2_v1 import reference, wire
    from verity.experimental.protocols.pous import verifier
    return reference, wire, verifier


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _merkle(rng: random.Random) -> dict:
    kind = rng.choice(["spec", "package"])
    salt = rng.randbytes(32 if kind == "spec" else 24)
    bps = rng.choice([1, 1, 2, 3, 4, 5, 7, 8])
    segments = rng.randint(1, max(1, 24 // bps))
    width = rng.choice([1, 8, 63, 64, 111, 112, 128, 255, rng.randint(1, 300)])
    if rng.random() < 0.05:
        width, segments = 2056, 1
    blocks = [rng.randbytes(width) for _ in range(bps * segments)]
    samples = [(rng.randrange(1 << 64), rng.randrange(1 << 16), rng.randbytes(rng.randint(0, 200))) for _ in range(3)]
    return {"kind": "merkle", "leaf": kind, "salt": salt.hex(), "blocks_per_segment": bps,
            "blocks": [b.hex() for b in blocks], "samples": [[s, v, b.hex()] for s, v, b in samples], "seed": rng.randrange(1 << 32)}


def _merkle_answers(py, inst: dict) -> dict:
    reference, wire, verifier = py
    salt, bps = bytes.fromhex(inst["salt"]), inst["blocks_per_segment"]
    blocks = [bytes.fromhex(b) for b in inst["blocks"]]
    if inst["leaf"] == "spec":
        def leaf(s, v, b):
            return reference.leaf(salt, s, v, b)
        verify = wire.verify_path
    else:
        def leaf(s, v, b):
            return verifier.leaf_hash(SimpleNamespace(blocks_per_segment=1 << 16, block_bytes=len(b)), salt, s * (1 << 16) + v, b)
        verify = verifier.verify_path
    leaves = [leaf(i // bps, i % bps, b) for i, b in enumerate(blocks)]
    n = len(leaves)
    tree = verifier.MerkleTree(leaves)
    root = reference.root(leaves)
    assert root == tree.root
    paths = [tree.path(i) for i in range(n)]
    assert all(list(p) == reference.path(leaves, i) for i, p in enumerate(paths))
    rng = random.Random(inst["seed"])
    checks = []
    for i in range(n):
        p = list(paths[i])
        cases = [("honest", 0, 0)]
        if p:
            cases += [("flipped", rng.randrange(len(p)), rng.randrange(8)), ("short", 0, 0), ("swapped", 0, 0),
                      ("odd-sibling", 0, 0)]
        cases += [("long", 0, 0), ("index", rng.randrange(n + 2), 0), ("past", n + rng.randrange(3), 0)]
        for name, a, b in cases:
            j, lf, q = check_input(name, i, a, b, leaves, p, root)
            checks.append([name, i, a, b, verify(root, n, j, lf, q)])
    return {"leaves": [x.hex() for x in leaves], "root": root.hex(),
            "paths_sha512": hashlib.sha512(canonical([[s.hex() for s in p] for p in paths]).encode()).hexdigest(),
            "path_lengths": [wire.path_length(n, i) for i in range(n)], "checks": checks,
            "samples": [leaf(s, v, bytes.fromhex(b)).hex() for s, v, b in inst["samples"]]}


def check_input(name: str, i: int, a: int, b: int, leaves: list[bytes], p: list[bytes], root: bytes):
    """A check's index, leaf and path, from its name and arguments (`PousCheck.Difftest.checkInput`)."""
    if name == "flipped":
        q = list(p)
        q[a] = bytes([q[a][0] ^ (1 << b)]) + q[a][1:]
        return i, leaves[i], q
    if name == "short":
        return i, leaves[i], p[:-1]
    if name == "swapped":
        return i, p[0], [leaves[i]] + p[1:]
    if name == "odd-sibling":
        return i, leaves[i], [p[0][:63]] + p[1:]
    if name == "long":
        return i, leaves[i], p + [root]
    if name in ("index", "past"):
        return a, leaves[i], p
    return i, leaves[i], p


def instance(seed: int) -> dict:
    rng = random.Random(seed)
    return _merkle(rng)


def answers(py, inst: dict) -> dict:
    return _merkle_answers(py, inst)


def write(repo: Path, out: Path, count: int = COUNT) -> dict[str, Path]:
    py = _python(repo)
    d = out / "difftest"
    d.mkdir(parents=True, exist_ok=True)
    insts = [instance(seed) for seed in range(count)]
    paths = {"difftest/instances.jsonl": d / "instances.jsonl", "difftest/python.jsonl": d / "python.jsonl"}
    paths["difftest/instances.jsonl"].write_text("".join(canonical(i) + "\n" for i in insts))
    paths["difftest/python.jsonl"].write_text("".join(canonical(answers(py, i)) + "\n" for i in insts))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("verity", type=Path, help="the checkout whose Python verifier is the reference")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=COUNT)
    a = ap.parse_args()
    py = _python(a.verity)
    for module in py:
        here = Path(module.__file__).resolve()
        if not here.is_relative_to(a.verity.resolve()):
            raise SystemExit(f"{module.__name__} comes from {here}, not from the checkout {a.verity}")
    write(a.verity, a.out, a.count)


if __name__ == "__main__":
    main()
