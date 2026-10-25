"""The NCI difftest vectors' Programs, distinct, one `{"id", "family", "program"}` object per line, for `nci-desc-fuzz`.

    uv run python gen/vectors.py --out FILE [--extra N]

Families: `difftest` (`scripts/difftest_vectors.py`'s instances, malformed ones included), `desc` (the description-sized
difftest's: `scripts/desc_difftest_vectors.py`'s random and transposed declarations, the circuits fixtures, the decoys,
the walks and the pinned ones), and with `--extra N` `difftest+` and `transposed+`, the same generators at the `N` seeds
past theirs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "lean" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def programs(extra: int):
    vectors = _load("difftest_vectors")
    desc = _load("desc_difftest_vectors")
    for seed in range(vectors.COUNT):
        yield f"difftest-{seed}", "difftest", vectors.instance(seed)["instance"].get("program")
    for p in vectors.pinned():
        yield "difftest-pinned", "difftest", p["instance"].get("program")
    for seed in range(desc.total() - len(vectors.pinned())):
        yield f"desc-{seed}", "desc", desc.instance(seed)["instance"].get("program")
    for seed in range(vectors.COUNT, vectors.COUNT + extra):
        yield f"difftest+-{seed}", "difftest+", vectors.program(random.Random(seed), seed)[0]
    for seed in range(desc.COUNT, desc.COUNT + extra):
        yield f"transposed+-{seed}", "transposed+", desc.transposed(random.Random(seed), seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", type=int, default=0)
    a = ap.parse_args()
    seen: set[str] = set()
    with open(a.out, "w") as f:
        for pid, family, text in programs(a.extra):
            if not isinstance(text, str) or text in seen:
                continue
            seen.add(text)
            f.write(json.dumps({"id": pid, "family": family, "program": text}) + "\n")
    print(f"vectors: {len(seen)} distinct programs")


if __name__ == "__main__":
    main()
