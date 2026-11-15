"""`class_masked.masked`: a decode unit that reads the key cache, at any of its statics, as its masked family bound at the same
statics (`class_window.MASKED_BLOCK` for `class_window.KEY_BLOCK`), its parameters the unit's and then `class_window.LIVE`;
the units that read only the writer's row, and every other decode unit, refused."""
from __future__ import annotations

import pytest

from verity.circuits.defs import bind
from verity_vllm.definitions.registry import sparse_mla as SM
from verity_vllm.program.predict import class_masked as CM
from verity_vllm.program.predict import class_window as CW

GLM = {k: SM.SPARSE_MLA_GLM53[k] for k in ("DR", "DV", "BN", "DOT", "SCALE")}


@pytest.mark.parametrize("fn", [bind(SM.SparseMlaHeadBlock, NVIS=44, FIRST=True, **GLM),
                                bind(SM.SparseMlaHeadBlock, NVIS=64, FIRST=False, **GLM),
                                bind(SM.SparseMlaHeadBlock, NVIS=16, FIRST=True, **GLM),
                                bind(SM.SparseMlaHeadKeysWire, NVIS=64, W=64, C=8)], ids=lambda f: f.id)
def test_a_unit_that_reads_the_cache_is_its_masked_family_at_its_statics(fn) -> None:
    got = CM.masked(fn)
    assert (got.defn.name, got.defn.version, got.bindings) == (fn.defn.name + "Live", 1, fn.bindings)
    if f"{fn.defn.name}_v{fn.defn.version}" == CW.KEY_BLOCK:
        assert f"{got.defn.name}_v{got.defn.version}" == CW.MASKED_BLOCK
    params = got.signature()[0]
    assert [n for n, _t in params] == [n for n, _t in fn.signature()[0]] + [CW.LIVE]
    assert got.signature()[1] == fn.signature()[1]


def test_the_masked_families_are_the_units_that_read_the_cache_and_no_other() -> None:
    """In the column cut a key block's scores and values units read its writer's row, never the cache; the end and the
    whole-row Definitions read no key row at all."""
    assert CM.LIVE == CW.LIVE and set(CM.MASKED) == {CW.KEY_BLOCK, "SparseMlaHeadKeysWire_v1"}
    st = dict(BN=GLM["BN"], DOT=GLM["DOT"])
    for fn in (bind(SM.SparseMlaHeadScoresCols, NVIS=64, FIRST=False, G=64, **GLM),
               bind(SM.SparseMlaHeadValues, NVIS=64, FIRST=False, G=64, C=0, **st),
               bind(SM.SparseMlaHeadScores, NVIS=64, FIRST=True, **GLM),
               bind(SM.SparseMlaHeadEnd, DV=64, INV=SM.SPARSE_MLA_GLM53["INV"]),
               bind(SM.SparseMlaHeadBlockLive, NVIS=64, FIRST=True, **GLM)):
        with pytest.raises(ValueError, match="masked takes a unit that reads the key cache"):
            CM.masked(fn)
