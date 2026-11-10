"""The synthetic input sets these tests read, committed under ``fixtures/synthetic_sets/<subcircuit id>/``: each is the set
`verity_flock.synthetic` writes for one (subcircuit, n, seed), with the provenance naming its run cleared, and its manifest's
recipe (template, parameters, samplers, n, seed) is what finds it. The command that adds one and the test holding these sets
to the generator are the bench's (``benchmarks/numerical``, ``verity_numerical.bench.fixture_sets``)."""
from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "synthetic_sets"


def _json(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def synthetic_set(sub: Any, n: int, seed: int, out: str | Path) -> dict[str, str]:
    """The committed set ``(sub, n, seed)``, copied to ``<out>/sets/<sub.set_name>`` where the generator writes it; returns
    ``{"path": ...}``."""
    want = {"template": sub.template.id, "parameters": _json(sub.params), "samplers": _json(sub.recipe), "n": n, "seed": seed}
    for m in sorted((FIXTURES / sub.id).glob("*/manifest.json")):
        recipe = json.loads(m.read_text())["provenance"]["recipe"]
        if {k: recipe.get(k) for k in want} == want:
            dst = Path(out) / "sets" / sub.set_name
            shutil.copytree(m.parent, dst)
            return {"path": str(dst)}
    sig = inspect.signature(sub.template.bind).parameters
    args = {**sub.params, **{k: v for k, v in sub.recipe.items() if k in sig and k not in sub.params}}
    raise LookupError(f"no committed synthetic set {sub.id} n={n} seed={seed} at its recipe: add it with `python -m "
                      f"verity_numerical.bench.fixture_sets --add {sub.template.id} '{json.dumps(_json(args))}' {n} {seed}`")
