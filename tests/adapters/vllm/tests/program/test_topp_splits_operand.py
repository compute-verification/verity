"""The split count S of the top-p pipeline is an OPERAND of the registered Definitions,
not a static: `TopPMaskWordx{V}(logits, top_p, splits) -> Value<V>`, `TopPMask_v1{V}(logits, top_p, splits)`,
`GumbelTopPTokenSelect_v1{V}(logits, top_p, seed, pos, temp, splits)`.  The Program feeds it the registered split count
`registered_splits(num_SMs)` as a `Const32[S]`, and the fold the split kernels' constexpr, also a `Const32[S]`.  CPU, torch-free:

  1. the operand form evaluates to `topp_split.topp_keep_row(x, p, S)` at every S in {1, 2, 4, 8, 16, 32} (the masked row lane by lane);
  2. the keep word is total: an S the pipeline never launches (0, 3, 64, 2^31, -1) keeps no lane (word 0, a word no launched S gives);
  3. `GumbelTopPTokenSelect_v1{V}` with the splits operand == the difftest composite with the same S as a static (the static-S body), token
     and masked row, at every S -- the numerics did not move, only where S enters;
  4. the codec materialises `TopPMaskWordx{V}_v1` by name (the static-S `TopPMaskWordx{V}s{S}_v1` name does not decode: one form);
  5. the registered S (the one the Match holds the operand to) is a launched split count at every SM count.

The B8 record rows through the operand form are `test_topp_splits_operand_of_record`.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest
from verity.circuits.defs import bind

from verity_vllm.definitions.registry import sampling as SM
from verity_vllm.definitions.registry import topp_geometry

f32 = np.float32
NEG_INF_BITS = 0xFF800000


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _u32(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint32)


def _u64(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint64)


def _u32s(vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint32)


def _row(V: int, seed: int = 0, scale: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(V) * scale).astype(f32)
    x[: V // 8] = x[0]                      # a plateau of identical logits: a pivot tie the S-dependent quota resolves
    return x


# ------------------------------------------------------------------------------------------------ 1. every S, operand == reference


# ------------------------------------------------------------------------------------------------ 2. total over every splits word


@pytest.mark.slow
def test_every_launched_split_count_keeps_a_lane():
    rng = np.random.default_rng(5)
    for k in range(40):
        x = (rng.standard_normal(128) * rng.choice([0.01, 1.0, 30.0])).astype(f32)
        for S in SM.SPLIT_COUNTS:
            for p in (1e-6, 0.3, 0.95, 1.0):
                assert SM.topp_keep(x, p, S).any(), (k, S, p)


# ------------------------------------------------------------------------------------------------ 3. select: operand == static body


def test_select_signature_has_splits_as_the_sixth_operand():
    params, ret = bind(SM.GumbelTopPTokenSelect, V=64).signature()
    assert [n for n, _ in params] == ["logits", "top_p", "seed", "pos", "temp", "splits"], params
    assert str(params[5][1]).replace(" ", "") == "Value<32>"
    assert bind(SM.GumbelTopPTokenSelect, V=64).id == "GumbelTopPTokenSelect_v1{V=64}"
    params, _ = bind(SM.TopPMask, V=64).signature()
    assert [n for n, _ in params] == ["logits", "top_p", "splits"]


# ------------------------------------------------------------------------------------------------ 4. codec name


def test_codec_materialises_the_operand_form_by_name():
    from verity.circuits import codec
    fams = codec._LAZY_PRIMITIVE_FAMILIES
    hits = [(rx, mk) for rx, mk in fams if rx.match("TopPMaskWordx128256_v1")]
    assert len(hits) == 1
    prim = hits[0][1](hits[0][0].match("TopPMaskWordx128256_v1"))
    assert prim.id == "TopPMaskWordx128256_v1" and prim is SM.topp_mask_word(128256)
    assert not any(rx.match("TopPMaskWordx128256s16_v1") for rx, _ in fams), "the static-S name must not decode: one form"


# ------------------------------------------------------------------------------------------------ 5. the relation


def test_the_registered_s_is_a_launched_split_count_at_every_sm_count():
    assert all(topp_geometry.registered_splits(m) in SM.SPLIT_COUNTS for m in range(1, 300))
    assert [topp_geometry.registered_splits(m) for m in (132, 142, 188)] == [32, 32, 32]
