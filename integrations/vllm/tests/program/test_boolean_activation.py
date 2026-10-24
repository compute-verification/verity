"""`verity.ml.definitions.registry.boolean_activation`: the Boolean tanh agrees with its word Definitions (`trace.word_of`) on all
65,536 words, element and row; the table is needed exactly on its band; each Definition carries exactly its builder's live
ANDs, is pinned by gate count and program digest, and the row partitions under `Q_word_v1{X=16, W=32}` into `Bf16Tanh_v1`'s
units, one per element."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity.ml.definitions.registry import boolean_activation as BA
from verity.ml.definitions.registry import dense as D

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "TanhBf16_v2": (391, 1628, 31, "556586c2a32ec874adf62df5acc74a853dde4642b1add2cd3365cdd55d3f876147219bbb421733c9d7dd74897869b3b6a4945c01dc25a833f45bc3a54be7d4fd"),
    "Bf16Tanh_v2{N=8}": (3128, 13024, 248, "a145e0351c69d4a45f8740b301f91e6682306d7cde53a41c75d7c77329862567bf16add6473d294b9d8f501e714af3a84cbdfb2a26a938d27503d84bbb8f01ba"),
}
WORDS = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _word(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got, want = _bits(target, *inputs), _word(target, *inputs)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


def test_pinned_counts_and_digests() -> None:
    for target in (bind(BA.TanhBf16), bind(BA.Bf16Tanh, N=8)):
        ands, xors, nots, digest = PINS[target.id]
        assert BL.is_boolean(target)
        assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
        prog, _ = standalone_call(target)
        desc = codec.encode_program(prog)
        assert codec.program_digest(desc) == digest
        decoded = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
        assert decoded.gates == prog.gates and codec.program_digest(codec.encode_program(decoded)) == digest


def test_the_ands_are_the_builders_live_ands() -> None:
    d = BA.TanhBf16
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert d.traced().counts["and"] == C.counts()["ands"] == PINS["TanhBf16_v2"][0]


def test_the_word_views() -> None:
    assert TR.word_of(bind(BA.TanhBf16)).id == "TanhBf16_v1"
    assert TR.word_of(bind(BA.Bf16Tanh, N=8)).id == bind(D.Bf16Tanh, N=8).id == "Bf16Tanh_v1{N=8}"


def test_the_element_word_is_the_served_rows_element() -> None:
    """`TanhBf16_v1` is `Bf16Tanh_v1{N=1}`'s body on one word, on all 65,536 words."""
    got = np.asarray(E.evaluate_batch(bind(BA.TanhBf16Word), WORDS), np.uint64).reshape(-1)
    assert np.array_equal(got, np.asarray(E.evaluate_batch(bind(D.Bf16Tanh, N=1), WORDS[:, None]), np.uint64).reshape(-1))


def _pair(sign: int, e0: int) -> range:
    return range(sign << 15 | e0 << 7, sign << 15 | (e0 + 2) << 7)


def test_the_band_is_where_the_closed_forms_fail() -> None:
    """Every word off tanh's band is x, +-1 or the NaN; on the band the sign is x's and the magnitude the (odd) table's; and
    the band's first exponent pair and its last are each needed (a closed form fails there)."""
    lo, hi = BA.TANH_BAND
    tanh = [int(v) for v in _word(bind(BA.TanhBf16), WORDS)]
    table = BA.tanh_table()
    one = lambda x: x & 0x8000 | 0x3F80                                   # noqa: E731
    for x, t in enumerate(tanh):
        sign, e, m = x >> 15, (x >> 7) & 0xFF, x & 0x7F
        if e == 255 and m:
            assert t == BA.NAN_F2FP
        elif e < lo:
            assert t == x
        elif e > hi:
            assert t == one(x)
        else:
            assert t >> 15 == sign and t & 0x7FFF == table[m | (e & 0xF) << 7]
    for sign in (0, 1):
        assert any(tanh[x] != x for x in _pair(sign, lo))
        assert any(tanh[x] != one(x) for x in _pair(sign, hi - 1))


def test_every_word() -> None:
    """The element and the row of one on all 65,536 words, against `TanhBf16_v1` and the served `Bf16Tanh_v1{N=1}`."""
    _agree(bind(BA.TanhBf16), WORDS)
    _agree(bind(BA.Bf16Tanh, N=1), WORDS[:, None])


def test_the_rows() -> None:
    """Random rows, and rows of the band edges' words (each sign, zero, subnormal, the infinities and NaNs among them)."""
    rng = np.random.default_rng(0x6E3)
    edges = np.array([s | e << 7 | m for s in (0, 0x8000) for e in (0, 1, 121, 122, 123, 128, 129, 130, 254, 255)
                      for m in (0, 1, 0x40, 0x7F)], np.uint16)
    rows = np.concatenate([rng.integers(0, 1 << 16, (4000, 8), dtype=np.int64).astype(np.uint16),
                           edges[rng.integers(0, len(edges), (1000, 8))]])
    _agree(bind(BA.Bf16Tanh, N=8), rows)


@pytest.mark.parametrize("n", [1, 3])
def test_the_word_definitions_units_under_q_word(n: int) -> None:
    """One unit per element, as many as `Bf16Tanh_v1{N}`'s: the switch swaps one for the other without moving a cut."""
    q = PO.query(16, 32)
    target = bind(BA.Bf16Tanh, N=n)
    prog, _ = standalone_call(target)
    word, _ = standalone_call(TR.word_of(target))
    assert PO.evaluate(prog, q).population == PO.evaluate(word, q).population == n
    assert PO.verify(PO.build(prog, q), prog).ok
