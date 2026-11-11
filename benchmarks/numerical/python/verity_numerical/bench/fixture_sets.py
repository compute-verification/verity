"""The synthetic input sets Flock's public tests read (`verity/protocols/flock/prover/tests/synthetic_sets.py`):
each is the set `verity_flock.synthetic.generate` writes for one (subcircuit, n, seed), with the provenance that names the run
(:data:`UNRUN`) cleared, so the same code writes the same bytes. A set lives under ``<subcircuit id>/`` in :data:`FIXTURES`,
and its manifest's recipe is what names it, rewrites it and finds it: the template, its parameters, the samplers (with what
they record of ``bind``'s other arguments, attention's key range ``T`` or a tiled GEMM's ``tile``), n and the seed.

    python -m verity_numerical.bench.fixture_sets                              rewrite every committed set
    python -m verity_numerical.bench.fixture_sets --add TEMPLATE ARGS N SEED   add one (ARGS: ``bind``'s, a JSON object)
    python -m verity_numerical.bench.fixture_sets --check                      exit 1 if one differs from the generator's
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from verity_flock import subcircuit_templates as TM
from verity_flock import synthetic as SY

FIXTURES = Path(__file__).resolve().parents[5] / "verity" / "protocols" / "flock" / "prover" / "tests" / "fixtures" / "synthetic_sets"
#: the provenance fields a run fills in, cleared in a committed set
UNRUN = {"research_run": None, "versions": None}


def _json(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


def _recipe(sub: TM.Subcircuit) -> dict[str, Any]:
    return {"template": sub.template.id, "parameters": _json(sub.params), "samplers": _json(sub.recipe)}


def _extra(template: TM.Template, params: dict[str, Any], samplers: dict[str, Any]) -> dict[str, Any]:
    """What the samplers record of ``bind``'s arguments beyond the parameters."""
    sig = inspect.signature(template.bind).parameters
    return {k: v for k, v in samplers.items() if k in sig and k not in params}


def place(sub: TM.Subcircuit, n: int, seed: int, root: Path = FIXTURES) -> Path:
    """``<id>/n<n>-seed<seed>``, then each of ``bind``'s other arguments the samplers record away from its default."""
    sig = inspect.signature(sub.template.bind).parameters
    tail = "".join(f"-{k}{str(v).replace(':', '-')}" for k, v in _extra(sub.template, sub.params, sub.recipe).items()
                   if str(v) != str(sig[k].default))
    return root / sub.id / f"n{n}-seed{seed}{tail}"


def write(sub: TM.Subcircuit, n: int, seed: int, root: Path = FIXTURES) -> Path:
    """Write set ``(sub, n, seed)`` at its place under ``root``; refused when a set of another recipe holds the place."""
    dst = place(sub, n, seed, root)
    if (dst / "manifest.json").is_file():
        held = json.loads((dst / "manifest.json").read_text())["provenance"]["recipe"]
        if {k: held[k] for k in _recipe(sub)} != _recipe(sub):
            raise ValueError(f"{dst} holds {held['template']} {held['parameters']} {held['samplers']}, not {_recipe(sub)}: "
                             f"two recipes share the place")
    with tempfile.TemporaryDirectory() as tmp:
        made = Path(SY.generate(sub, n, seed, tmp, provenance=UNRUN)["path"])
        shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(made, dst)
    return dst


def committed(root: Path = FIXTURES) -> list[tuple[Path, TM.Subcircuit, int, int]]:
    """Each committed set's directory and (subcircuit, n, seed), rebuilt from its manifest's recipe."""
    out = []
    for m in sorted(root.glob("**/manifest.json")):
        r = json.loads(m.read_text())["provenance"]["recipe"]
        t = TM.get(r["template"])
        sub = t.bind(**r["parameters"], **_extra(t, r["parameters"], r["samplers"]))
        out.append((m.parent, sub, int(r["n"]), int(r["seed"])))
    return out


def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def check(root: Path = FIXTURES) -> list[str]:
    """The committed sets that differ from what the generator writes for their recipe, or don't sit at its place."""
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for d, sub, n, seed in committed(root):
            if d != place(sub, n, seed, root) or _files(write(sub, n, seed, Path(tmp))) != _files(d):
                bad.append(str(d.relative_to(root)))
    return bad


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--add", nargs=4, metavar=("TEMPLATE", "ARGS", "N", "SEED"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(None if argv is None else list(argv))
    if a.check:
        bad = check()
        for b in bad:
            print(f"differs from the generator's: {b} (rewrite it: python -m verity_numerical.bench.fixture_sets)")
        return 1 if bad else 0
    if a.add:
        template, args, n, seed = a.add
        sets: list[tuple[TM.Subcircuit, int, int]] = [(TM.subcircuit(template, **json.loads(args)), int(n), int(seed))]
    else:
        sets = [(sub, n, seed) for _d, sub, n, seed in committed()]
    for sub, n, seed in sets:
        print(write(sub, n, seed).relative_to(FIXTURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
