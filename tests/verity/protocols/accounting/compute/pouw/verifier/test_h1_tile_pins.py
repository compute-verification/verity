"""`credit.TILE_PINS` (``lean/check/h1_tiles.json``, which Lean's `Credit.TILE_PINS` reads too) against the registry: each
pinned instance's window template (`h1window.template` at K over `h1window.tile_unit` at K, the 4 × 8 tile `h1unit.TILE`,
over the v1, v3 or J10-R v5 H1 tile, chunked as `h1window.program` chunks it; `pouw_credit_cases.registry_window`) is
the key, and its closure digest (`codec.closure_digest`, its body and every Definition under it) the pin, so a change
to the template, or to any Definition it reaches, fails here until it re-pins. Run as a script, it writes the pins:

    uv run python tests/verity/protocols/accounting/compute/pouw/verifier/test_h1_tile_pins.py

A pinned window's descriptor is about 60 MB (its closure reaches the noise's SHA-512 at gates), about ten seconds to
build, so every test here is slow.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from verity.circuits import codec
from verity.protocols.accounting.compute.pouw.prover.lowering import h1unit as U
from verity.protocols.accounting.compute.pouw.verifier import credit as Cr

if "pouw_credit_cases" not in sys.modules:
    for _name in ("pouw_rounds_cases", "pouw_credit_cases"):
        _spec = importlib.util.spec_from_file_location(_name, Path(__file__).parent / "lean" / "check" / f"{_name}.py")
        sys.modules[_name] = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(sys.modules[_name])
cases = sys.modules["pouw_credit_cases"]

#: the pinned instances: each K and the H1 tile under its unit (v3 has no form at K = 32)
INSTANCES = ((64, "v1"), (64, "v3"), (64, "v5"), (4096, "v1"), (4096, "v5"))
SCRIPT = "tests/verity/protocols/accounting/compute/pouw/verifier/test_h1_tile_pins.py"


def descriptor_bytes(desc: dict) -> bytes:
    """`desc`'s bytes as the period's ``circuit`` hashes them: canonical JSON without annotations."""
    return codec.canonical_json({k: v for k, v in desc.items() if k != "annotations"})


def text() -> str:
    """The pins file: each pinned instance's template key and its closure digest."""
    pins = {}
    for k, tile in INSTANCES:
        desc = cases.registry_window(k, tile)
        pins[cases.template_key(desc)] = codec.closure_digest(desc, cases.template_key(desc))
    return json.dumps(pins, indent=1, sort_keys=True) + "\n"


@pytest.mark.slow
def test_the_pins_are_the_registrys_templates():
    """Regenerates every pinned template: a change to `h1window`'s template or tile unit, or to any Definition under
    them, would otherwise surface only as served windows refused (`credit.no_template`)."""
    assert U.TILE == (4, 8), "registry_window builds h1unit's default tile, 4 × 8"
    assert Cr.TILE_PINS_FILE.read_text() == text(), (
        f"{Cr.TILE_PINS_FILE.name} is not the registry's H1 window templates at their pinned instances (INSTANCES): if "
        f"the change to the template, or to a Definition it reaches, is meant, re-pin with `uv run python {SCRIPT}`")
    assert Cr.TILE_PINS == json.loads(Cr.TILE_PINS_FILE.read_text())


@pytest.fixture(scope="session")
def window_k64(tmp_path_factory) -> bytes:
    """The registry's window at K = 64, built once a session (`pouw_credit_cases.shared_window`)."""
    return cases.shared_window(tmp_path_factory.getbasetemp().parent)


@pytest.mark.slow
def test_a_pinned_window_is_read_and_its_template_over_another_body_is_refused(window_k64):
    """The registry's window at K = 64 is read (its layout, the 4 × 8 tile); the same descriptor with one Definition
    under its template over another body, every id and signature the same (`pouw_credit_cases.other_body`), is refused
    as no template at K = 64 (`credit.layout_of`, by `_tile`)."""
    desc = json.loads(window_k64)
    key = cases.template_key(desc)
    lay, why = Cr.layout_of(window_k64)
    assert why is None and lay["tile"] == [4, 8], why
    assert Cr._tile(desc["definitions"][key], key, 64, desc) == (4, 8, False)
    bad = cases.other_body(desc)
    assert codec.closure_digest(bad, key) != Cr.TILE_PINS[key]
    assert Cr._tile(bad["definitions"][key], key, 64, bad) is None
    assert Cr.layout_of(descriptor_bytes(bad)) == (None, ("layout", Cr.no_template(64)))


if __name__ == "__main__":
    Cr.TILE_PINS_FILE.write_text(text())
    print(Cr.TILE_PINS_FILE.read_text())
