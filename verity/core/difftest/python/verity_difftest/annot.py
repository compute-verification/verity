"""The annotation check's port: C-Flock's `Flock.Annot` behind `Difftest.serve` (`lean/AnnotDifftest.lean`, the Lake
executable `annot-difftest` of this distribution's `lean/` package), for the tests that hold the Python mirror
(`verity.ml.circuits.query_ast` and `query_codec`) to it.

Its functions (`FNS`) take `{"program": descriptor text, "annotation": {"query", "tables"}, "meaning": {...}}`:

* `annot.term`: a query document read and written back, the codec's canonical form;
* `annot.check`: the description-level result `{"accept", "count", "code"?}` (`Annot.check`);
* `annot.reference`: the same meaning decided gate by gate (`Annot.reference`);
* `annot.sets`: `{"sets", "labels"}`, the family's sets in order and the gates each label holds, as root-gate intervals.

A refusal is a reject whose reason starts with its code (`term`, `table`, `plan`, `fuel`, `meaning-unsupported`).
`port(fn, reference, inputs)` is a `Port` on one of them; `server()` one server for direct calls. A suite whose tests
call it lists `INPUTS` among its `[tool.verity.tests] inputs`: the files its build reads.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from . import Port, Server, build

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

LEAN = Path(__file__).resolve().parents[2] / "lean"
EXE = "annot-difftest"
FNS = ("annot.term", "annot.check", "annot.reference", "annot.sets")
_FLOCK = "verity/core/service/flock/"
INPUTS = ("verity/core/difftest/lean",
          *(_FLOCK + f for f in ("lakefile.toml", "lake-manifest.json", "lean-toolchain")),
          *(_FLOCK + f"Flock/{m}.lean" for m in ("Annot", "AnnotExpand", "Bytes", "Canon", "Field", "Hash", "Program",
                                                   "Sizes", "Steps")))


def server() -> Server:
    """One `annot-difftest` server, built on first use."""
    return Server([str(build(LEAN, EXE))])


def port(fn: str, reference: Callable[[Any], Any], inputs: SearchStrategy, **kw: Any) -> Port:
    """The port of `fn` (one of `FNS`) against `reference`."""
    if fn not in FNS:
        raise ValueError(f"{fn!r} is not one of {FNS}")
    return Port(EXE, fn, reference, inputs, package=LEAN, **kw)


def instance(program: str, query: dict, tables: dict | None = None, meaning: dict | None = None) -> dict:
    """The input of `annot.check`, `annot.reference` and `annot.sets`: `program` the descriptor's text."""
    out: dict = {"program": program, "annotation": {"query": query, **({"tables": tables} if tables else {})}}
    if meaning is not None:
        out["meaning"] = meaning
    return out
