"""The split count S of the top-p pipeline is an OPERAND of the registered Definitions,
not a static: `TopPMaskWordx{V}(logits, top_p, splits) -> Value<V>`, `TopPMask_v1{V}(logits, top_p, splits)`,
`GumbelTopPTokenSelect_v1{V}(logits, top_p, seed, pos, temp, splits)`.  S_t = SplitsFor_v1(|live(t)|, num_SMs) is a derived value of the
WORKLOAD the request Program reads from its request-owned `splits` Input (one entry per sampling event) and the fold carries as the split
kernels' constexpr (`Const32[S]`).  CPU, torch-free:

  1. the operand form evaluates to `topp_split.topp_keep_row(x, p, S)` at every S in {1, 2, 4, 8, 16, 32} (the masked row lane by lane);
  2. the keep word is total: an S the pipeline never launches (0, 3, 64, 2^31, -1) keeps no lane (word 0, a word no launched S gives);
  3. `GumbelTopPTokenSelect_v1{V}` with the splits operand == the difftest composite with the same S as a static (the static-S body), token
     and masked row, at every S -- the numerics did not move, only where S enters;
  4. the codec materialises `TopPMaskWordx{V}_v1` by name (the static-S `TopPMaskWordx{V}s{S}_v1` name does not decode: one form);
  5. `splits_for_v1` (the relation the Match checks the operand against) is the registry's `splits_for` on 1..64 x {132, 142} SMs.

The B8 record rows through the operand form are `test_topp_splits_operand_of_record`.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import sampling as SM
from verity.ml.definitions.registry import sampling_topp_difftest as D
from verity.ml.definitions.registry import topp_split as TS
from verity.ml.definitions.registry import topp_geometry

f32 = np.float32
NEG_INF_BITS = 0xFF800000


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _u32(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint32)


def _u64(*vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint64)


def _eval(spec, operands) -> list[int]:
    from verity_vllm.properties.admission import evaluate_spec
    return [int(v) for v in evaluate_spec(spec, operands)]


def _u32s(vals) -> np.ndarray:
    return np.array(vals, dtype=np.uint32)


def _row(V: int, seed: int = 0, scale: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(V) * scale).astype(f32)
    x[: V // 8] = x[0]                      # a plateau of identical logits: a pivot tie the S-dependent quota resolves
    return x


# ------------------------------------------------------------------------------------------------ 1. every S, operand == reference


@pytest.mark.parametrize("S", [1, 2, 4, 8, 16, 32])
def test_topp_mask_operand_form_equals_topp_keep_row_at_every_s(S):
    V, p = 256, 0.95
    x = _row(V, seed=S)
    out = _eval(bind(SM.TopPMask, V=V), [x.view(np.uint32), _u32(_bits(p)), _u32(S)])
    keep = TS.topp_keep_row(x, p, S)
    want = np.where(keep, x, f32(-np.inf)).view(np.uint32)
    assert out == want.tolist(), f"S={S}: the masked row differs from topp_keep_row"
    assert 0 < int(keep.sum()) < V
    # the keep word primitive itself
    word = _eval(SM.topp_mask_word(V), [x.view(np.uint32), _u32(_bits(p)), _u32(S)])[0]
    assert word == int.from_bytes(np.packbits(keep, bitorder="little").tobytes(), "little")


# ------------------------------------------------------------------------------------------------ 2. total over every splits word


@pytest.mark.parametrize("bad", [0, 3, 5, 31, 33, 64, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xFFFFFFF0])
def test_a_split_count_the_pipeline_never_launches_keeps_no_lane(bad):
    from verity.ml.definitions.registry import sampling_rows as SR
    V = 64
    x = _row(V)
    word = _eval(SM.topp_mask_word(V), [x.view(np.uint32), _u32(_bits(0.9)), _u32(bad)])[0]
    masked = _eval(bind(SM.TopPMask, V=V), [x.view(np.uint32), _u32(_bits(0.9)), _u32(bad)])
    assert word == 0 and masked == [NEG_INF_BITS] * V
    assert SR.topp_mask_row(x.view(np.uint32), _bits(0.9), bad).tolist() == masked            # the numpy row, the same semantics
    token = _eval(bind(SM.GumbelTopPTokenSelect, V=V), [_u32s((x.view(np.uint32) >> 16).tolist()), _u32(_bits(0.9)), _u64(7), _u64(3),
                                                         _u32(_bits(0.8)), _u32(bad)])
    assert token == [0]                                                                        # every lane -inf: the first index
    assert SM.SPLIT_COUNTS == (1, 2, 4, 8, 16, 32)


@pytest.mark.slow
def test_every_launched_split_count_keeps_a_lane():
    rng = np.random.default_rng(5)
    for k in range(40):
        x = (rng.standard_normal(128) * rng.choice([0.01, 1.0, 30.0])).astype(f32)
        for S in SM.SPLIT_COUNTS:
            for p in (1e-6, 0.3, 0.95, 1.0):
                assert SM.topp_keep(x, p, S).any(), (k, S, p)


# ------------------------------------------------------------------------------------------------ 3. select: operand == static body


@pytest.mark.parametrize("S", [1, 2, 4, 8, 16, 32])
def test_gumbel_topp_select_operand_form_equals_the_static_difftest_body(S):
    V, p, T, seed, pos = 1024, 0.95, 0.8, 1077402498, 255
    x = _row(V, seed=100 + S, scale=4.0)
    xb = ((x.view(np.uint32) + 0x8000) >> 16).astype(np.uint32)       # a bf16 row (round-to-nearest of the f32 row)
    ops = [xb, _u32(_bits(p)), _u64(seed), _u64(pos), _u32(_bits(T))]
    got = _eval(bind(SM.GumbelTopPTokenSelect, V=V), ops + [_u32(S)])
    ref = _eval(D.spec({"V": V, "S": S}), ops)                     # token + the masked row, S a static of the difftest body
    assert got[0] == ref[0], f"S={S}: token {got[0]} != {ref[0]}"
    assert 0 <= got[0] < V
    masked = _u32s(ref[1:1 + V])
    assert (masked == NEG_INF_BITS).any() and (masked != NEG_INF_BITS).any()


def test_select_signature_has_splits_as_the_sixth_operand():
    params, ret = bind(SM.GumbelTopPTokenSelect, V=64).signature()
    assert [n for n, _ in params] == ["logits", "top_p", "seed", "pos", "temp", "splits"], params
    assert str(params[5][1]).replace(" ", "") == "Value<32>"
    assert bind(SM.GumbelTopPTokenSelect, V=64).id == "GumbelTopPTokenSelect_v1{V=64}"
    params, _ = bind(SM.TopPMask, V=64).signature()
    assert [n for n, _ in params] == ["logits", "top_p", "splits"]


# ------------------------------------------------------------------------------------------------ 4. codec name


def test_codec_materialises_the_operand_form_by_name():
    from verity.ml.circuits import codec
    fams = getattr(codec, "_LAZY_PRIMITIVE_FAMILIES")
    hits = [(rx, mk) for rx, mk in fams if rx.match("TopPMaskWordx128256_v1")]
    assert len(hits) == 1
    prim = hits[0][1](hits[0][0].match("TopPMaskWordx128256_v1"))
    assert prim.id == "TopPMaskWordx128256_v1" and prim is SM.topp_mask_word(128256)
    assert not any(rx.match("TopPMaskWordx128256s16_v1") for rx, _ in fams), "the static-S name must not decode: one form"


# ------------------------------------------------------------------------------------------------ 5. the relation


@pytest.mark.parametrize("nsm", [132, 142])
def test_splits_for_v1_is_the_relation_the_match_checks_the_operand_against(nsm):
    for n in range(1, 65):
        S = topp_geometry.splits_for_v1(n, nsm)
        assert S == topp_geometry.splits_for(n, nsm) and S in SM.SPLIT_COUNTS
    assert topp_geometry.splits_for_v1(1, nsm) == 32 and topp_geometry.splits_for_v1(8, 142) == 16 and topp_geometry.splits_for_v1(4, 142) == 32
