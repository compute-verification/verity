"""The flag/select primitives of the padded serving circuit --
`verity/ml/definitions/registry/pad_prims.py` (`BitAnd_v1`/`BitNot_v1` new; the selects and `I32Eq_v1` reused from
`prims.py`).  The `veritor.pad-serve@1` gate set itself, its digest and its out-of-width rejections are pinned by the SP1
statement table's own tests (`archive/sp1/common/src/gateset.rs`: `pad_serve_manifest_matches_the_python_family`,
`pad_serve_gates_evaluate_and_reject_out_of_width_words`)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.definitions.registry import pad_prims as PP


def test_bit_gates_truth_tables_and_absorption():
    assert [PP.BitAnd.evaluate(a, b) for a in (0, 1) for b in (0, 1)] == [0, 0, 0, 1]
    assert [PP.BitNot.evaluate(a) for a in (0, 1)] == [1, 0]
    # the finished chain is absorbing: live' = and(live, not(stop)) is 0 whenever live is 0
    for stop in (0, 1):
        assert PP.BitAnd.evaluate(0, PP.BitNot.evaluate(stop)) == 0
    # and forwards the negated stop bit whenever live is 1
    for stop in (0, 1):
        assert PP.BitAnd.evaluate(1, PP.BitNot.evaluate(stop)) == (stop ^ 1)


def test_new_primitives_are_registered_and_reused_ones_untouched():
    assert PP.BitAnd.id == "BitAnd_v1" and PP.BitNot.id == "BitNot_v1"
    assert PP.BitAnd.conformance == "protocol-defined" and PP.BitNot.conformance == "protocol-defined"
    # "EqI32" and the typed selects are the existing registered primitives, reused not re-registered
    assert [p.id for p in PP.REUSED] == ["I32Eq_v1", "SelectBf16_v1", "SelectF32_v1", "SelectI32_v1"]
    assert all(p.conformance == "exact-model-tested" for p in PP.REUSED)
