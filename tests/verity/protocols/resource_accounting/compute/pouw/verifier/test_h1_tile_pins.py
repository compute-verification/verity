"""`credit.TILE_PINS` (``lean/check/h1_tiles.json``, which Lean's `Credit.TILE_PINS` reads too) against the registry: each
pinned instance's window template (`h1window.template` at K over `h1window.tile_unit` at K, the 4 × 8 tile `h1unit.TILE`,
over the v1, v3 or J10-R v5 H1 tile, chunked as `h1window.program` chunks it) is the key, and its closure digest
(`codec.closure_digest`) the pin, so a change to the template, or to any Definition it reaches, fails here until it
re-pins; and the credit cases' fixture (``lean/check/fixtures/h1_tiles.json``) is each pinned template's closure, every
entry its canonical entry (`codec.canonical_entry`, its callees by id and its types inline), which `pouw_credit_cases`
interns into its descriptors. Run as a script, it writes both:

    uv run python tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_h1_tile_pins.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.circuits import codec
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1tile as H
from verity.protocols.resource_accounting.compute.pouw.prover.circuit import h1window as HW
from verity.protocols.resource_accounting.compute.pouw.prover.lowering import h1unit as U
from verity.protocols.resource_accounting.compute.pouw.verifier import credit as Cr

#: the pinned instances: each K, and the H1 tile under its unit (`h1unit.Shape`'s v1 and v3, and J10-R's v5)
KS = (32, 64, 4096)
TILES = {"v1": None, "v3": H.PouwH1Tile3, "v5": H.PouwH1Tile5}
SCRIPT = "tests/verity/protocols/resource_accounting/compute/pouw/verifier/test_h1_tile_pins.py"
FIXTURE = Path(__file__).parent / "lean" / "check" / "fixtures" / "h1_tiles.json"


def closures() -> dict[str, dict[str, dict]]:
    """Each pinned instance's template key and its closure, each Definition it reaches by id to its canonical entry."""
    r, c = U.TILE
    out = {}
    for k in KS:
        for definition in TILES.values():
            unit = HW.tile_unit(k, (r, c), definition)
            desc = codec.encode_program(HW.program((("w", k, c),), (("w", 0),), r, (r, c), unit=lambda _k, u=unit: u))
            (key,) = (f for f in desc["callees"] if f.startswith(Cr.TILE_FN + "{"))
            todo, seen = [key], {}
            while todo:
                f = todo.pop()
                if f not in seen:
                    e = desc["definitions"][f]
                    seen[f] = codec.canonical_entry(e, desc)
                    if e["form"] == "composite":
                        todo += [desc["callees"][x["fn"]] for x in e["body"]["nodes"]]
            assert codec.closure_digest(desc, key) == digest(seen)
            out[key] = seen
    return out


def digest(closure: dict[str, dict]) -> str:
    """`codec.closure_digest` of a closure of canonical entries: each is its own canonical entry."""
    return hashlib.sha512(codec.canonical_json({f: hashlib.sha512(codec.canonical_json(e)).hexdigest()
                                                for f, e in closure.items()})).hexdigest()


def files() -> dict[Path, str]:
    """The pins file and the fixture, each path and its text."""
    got = closures()
    pins = {key: digest(defs) for key, defs in got.items()}
    return {Cr.TILE_PINS_FILE: json.dumps(pins, indent=1, sort_keys=True) + "\n",
            FIXTURE: codec.canonical_json(got).decode() + "\n"}


def test_the_pins_and_the_fixture_are_the_registrys_templates():
    """Regenerates every pinned template (about ten seconds each): a change to `h1window`'s template or tile unit, or
    to any Definition under them, reaches no other test until the credit cases run on a stale fixture."""
    for path, text in files().items():
        assert path.read_text() == text, (
            f"{path.name} is not the registry's H1 window templates at their pinned instances (KS × TILES): if the change "
            f"to the template, or to a Definition it reaches, is meant, re-pin with `uv run python {SCRIPT}`")


def test_the_verifier_reads_the_pins():
    assert Cr.TILE_PINS == json.loads(Cr.TILE_PINS_FILE.read_text())


if __name__ == "__main__":
    for path, text in files().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(path, len(text))
