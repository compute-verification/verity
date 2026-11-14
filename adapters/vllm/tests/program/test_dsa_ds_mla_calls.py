"""`boolean_dsa_ds_mla.DsaLatentRowDsMla_v3`, the `fp8_ds_mla` latent row written as Calls (its scale `DsaRMSNormSumSq_v3`'s
body, which `boolean_version` gives): it has the traced-scale `_v2`'s word view and signature, and is that `_v2` and the word bit
for bit, with the `_v2`'s gates, count for count."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.definitions import boolean as SW
from verity.definitions.registry import boolean_dsa_ds_mla as BM
from tests.program.test_boolean_fp8 import _flat, _inputs

#: the smallest row: 256 latent words (two registers a thread) in four tiles of 64, and 8 rotary values
SMALL = dict(KVL=256, R=8, TILE=64, EPS=1e-5)


def test_the_served_row_is_the_row_as_calls() -> None:
    """`boolean_version` of the word row gives the `_v3`, whose word view and signature are the `_v2`'s."""
    calls, traced = bind(BM.DsaLatentRowDsMlaV3, **SMALL), bind(BM.DsaLatentRowDsMlaV2, **SMALL)
    word = TR.word_of(calls)
    assert word.id == TR.word_of(traced).id and SW.boolean_version(word).id == calls.id
    assert calls.signature() == traced.signature()


@pytest.mark.slow
def test_the_row_as_calls_is_the_traced_row() -> None:
    """The `_v3` and the `_v2` are the word row bit for bit on random and special words, and have the same gates.  Both rows
    are about 7.9M gates at the smallest binding (minutes): only this holds the row written as Calls to the traced one bit
    for bit."""
    calls, traced = bind(BM.DsaLatentRowDsMlaV3, **SMALL), bind(BM.DsaLatentRowDsMlaV2, **SMALL)
    word = TR.word_of(calls)
    inputs = _inputs(word, 96, sum(map(ord, calls.id)))
    want = E.evaluate_batch(word, *inputs)
    for unit in (calls, traced):
        for g, r in zip(_flat(BS.evaluate_words(unit, *inputs)), _flat(want), strict=True):
            bad = np.flatnonzero((g != r).any(axis=1))
            assert not bad.size, f"{unit.id}: {bad.size} of {len(r)} instances differ, first {bad[:5]}"
    assert calls.gates == traced.gates
