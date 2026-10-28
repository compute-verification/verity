"""The salted H1 tile unit's pin (`core_check.TILES`, `PouwCheck/Tile.lean`): `h1unit.Shape(64)`'s staged flat circuit (the
statement the Lean verifier accepts; the templated one it refuses until its proof lands) is the one the verifier pins for
its template, with the salted layout the opening reads: R A rows, C B rows, then the public row of
salt, position and roots, each `hm96-sha512/row/v2`. The opening refuses a proof over any other circuit, so a change to
`PouwH1Tile`'s Definition or to C-Flock's lowering lands its new digest in both tables. `test_pouw_core_check` ties the
template's id, query and layout to its Definition without the build; only the build reaches the circuit's digest. The
build takes about 3.5 minutes and 9.5 GB, so it runs when VERITY_POUW_H1_PIN is set."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "h1_prove"))
import h1unit as U  # noqa: E402


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_POUW_H1_PIN"),
                    reason="VERITY_POUW_H1_PIN builds the H1 unit's circuit at K = 64 (about 3.5 minutes, 9.5 GB)")
def test_the_h1_units_circuit_is_its_pin(tmp_path):
    shape = U.Shape(64)
    d = shape.definition()
    tile = CC.TILES[d.id]
    st = shape.stager("unit/pouw-h1-tile", circuit=tmp_path / "circuit.txt", flat=True)
    assert st.pin == tile.circuit, (
        f"{d.id} builds circuit {st.pin}, not the one the verifier pins: a change to the template or to C-Flock's "
        "lowering lands its new digest in PouwCheck/Tile.lean and core_check.TILES")
    r, c = tile.salted
    opened = st.ports[:r + c + 1]
    assert [p.name for p in opened] == [*(f"a{k}" for k in range(r)), *(f"b{k}" for k in range(c)), "pub"]
    assert {p.schema for p in opened} == {CC.ROW_V2} and opened[-1].bits == CC.PUBLIC_ROW_BITS
