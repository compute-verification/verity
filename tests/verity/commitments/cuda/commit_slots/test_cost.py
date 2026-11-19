"""A row-slot's hashing, counted on the host twin, never timed. A one-row unit's row-slot costs nine compressions: its row
digest one, hm96's c two and h two (h's padding chunk over a constant schedule), its frame leaf one and its proof block's slot-tree
leaf one (both from midstates, in the leaf kernel's thread), and about one slot-tree node, two; its salt is the caller's
bytes, read and not hashed for.
Its unit's frame table costs one compression an add, whatever the slots. The path before cost ten: the slot-tree leaf was
hashed from the IV (two) in a call of its own."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from conftest import BUILD, salts_of

from verity.commitments.cuda.commit_slots import slots as S
from verity.commitments.cuda.commit_slots import slots_gpu as G

pytestmark = pytest.mark.skipif(not any(shutil.which(c) for c in ("g++", "c++", "clang++")), reason="needs a C++ compiler")

CTX = bytes(range(64))


@pytest.mark.parametrize("temp_bytes", [G.TEMP_BYTES, 1], ids=["one-pass", "one-slot-passes"])
def test_an_add_of_one_row_units_costs_nine_compressions_a_row_slot(temp_bytes: int) -> None:
    units, slots = 64, 4
    group = S.Group((2 * units,), tuple((S.Row((S.Part(0, 2 * i, 2),)),) for i in range(units)))
    layout = S.CommitShape("coords-64", (group,))
    plan = G.Plan(CTX, layout, "host", temp_bytes=temp_bytes, build=BUILD)
    out = S.random_outputs(layout, slots, np.random.default_rng(5))
    y = salts_of(layout, slots)
    sc = G.StepCommitter(plan, 0, slots)
    plan.engine.counts(reset=True)
    sc.add(out[0], y)
    rs = units * slots
    assert plan.engine.counts(reset=True) == {"compressions": 7 * rs + units + 2 * slots * (units - 1), "kw": rs}
    assert sc.finish().result() == S.commit(CTX, layout, 0, y, out, slots)
