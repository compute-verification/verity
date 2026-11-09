"""`verity.ml.commitments.gates.block_check`, PoUB's opening unit: the Boolean unit equals its Python reference on toy
openings, appended and registered, honest and with each wrong one (`block_check_cases`: a tampered block, a payload not the
committed one or not the committed KV, a word at or above p, a wrong append coin, a wrong receipt leaf, salt or path, a plan
without the block at the sampled position, a receipt the pass's tree commits but the block doesn't give), and the reference
names the check each fails; its rows hold every parameter and refuse a shape off the codec, and its parts cost the gates
pinned here. The toy blocks are the codec's encodings (`fixtures/block_check_toy.json`); PoUB's tests hold that fixture, the
reference at the deployed fields, and the parts' rounds and binding to the codec, receipts and trees
(`verity/core/protocols/remote_bandwidth/tests/test_block_check_codec.py`)."""
from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace

import block_check_cases as K
import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.commitments.gates import block_check as BC

TOY_BLOCKS = Path(__file__).resolve().parent / "fixtures" / "block_check_toy.json"
#: (part, its statics) -> (And, Xor, Not) gates of its traced circuit
PARTS = {
    ("RowCompress", ()): (57947, 286724, 4948),
    ("CodecBinding", (512, 569)): (112569, 545189, 11366),
    ("CodecModAdd", (512, 569)): (1534, 4091, 4),
    ("CodecLess", (512, 569)): (511, 1022, 5),
    ("CodecRound", (512, 569, 0)): (104912, 303366, 8202),
    ("CodecRound", (1024, 105, 1)): (310507, 901490, 23798),
    ("CodecMatch", (512,)): (512, 512, 512),
    ("SketchRow", (4096,)): (36863, 214292, 4099),
    ("LeafMatch", (16, 40)): (9646, 9034, 561),
    ("BlockVerdict", (64,)): (69, 2, 5),
}


@functools.cache
def _toy_blocks() -> dict[tuple, bytes]:
    return {(e["masked"], e["seed"], e["j"], e["payload"], e["coin"]): bytes.fromhex(e["block"])
            for e in json.loads(TOY_BLOCKS.read_text())}


def _toy_encode(S, seed: bytes, j: int, payload: bytes, coin: bytes | None) -> bytes:
    """`codec.encode` over the toy field, as the fixture holds it."""
    key = (S.MASKED, seed.hex(), j, payload.hex(), None if coin is None else coin.hex())
    assert key in _toy_blocks(), (f"{TOY_BLOCKS.name} holds no block for this toy opening: rewrite it with "
                                  "`uv run python verity/core/protocols/remote_bandwidth/tests/test_block_check_codec.py`")
    return _toy_blocks()[key]


def test_the_rows_hold_every_parameter() -> None:
    for S in (K.P512, K.P1024, K.TOY):
        params, _ = BC._opening_signature(S)
        laid = BC.opening_rows(S)
        assert [p for _r, _b, ps in laid for p in ps] == [n for n, _ in params]
        assert sum(b for _r, b, _ps in laid) == sum(t.leaves for _, t in params)
        assert [r for r, _b, _ps in laid] == ["pub", "signs", "block", "payload", "leaf", "plan", "aux"]


@pytest.mark.parametrize("bad", [dict(N=64, C=59), dict(BYTES=64), dict(K=1), dict(MASKED=2), dict(BYTES=96), dict(LEAF=17),
                                 dict(DR=13), dict(DP=0), dict(DV=25), dict(P=1 << 16)])
def test_an_opening_off_the_codec_is_refused(bad) -> None:
    S = SimpleNamespace(**{**vars(K.P512), **bad})
    with pytest.raises(ValueError):
        BC.opening_rows(S)


def _u16_bits(words) -> np.ndarray:
    w = np.asarray(words, dtype=np.uint64)
    return ((w[:, None] >> np.arange(16, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1)


def _gates(defn, rows: list[dict]) -> list[int]:
    params, _ = defn.signature()
    cols = [np.stack([_u16_bits(a[n]) for a in rows]) for n, _ in params]
    out = np.asarray(evaluate_batch(defn, *cols)).reshape(len(rows), 16)
    return [int(sum(int(b) << i for i, b in enumerate(r))) for r in out]


@pytest.mark.slow
@pytest.mark.parametrize("S", [K.TOY, K.TOY_MASKED], ids=["toy", "toy-masked"])
def test_the_toy_unit_is_its_reference(S) -> None:
    """A few minutes (the compressions' and reads' first trace): the only evaluation of the unit's wiring, the tweak chain,
    rounds and masks, the payload's digest into the binding and its hm96 leaf, the sketch into the receipt, the receipt
    leaf's digest and hm96 leaf, each read's rank, position and depth, and the verdict's flags, which the deployed fields'
    parts never reach."""
    by_leaf: dict[int, list] = {}
    for name, o, a, want in K.toy_cases(S, _toy_encode):
        by_leaf.setdefault(o.S.LEAF, []).append((name, o, a, want))
    for leaf, group in by_leaf.items():
        S_ = group[0][1].S
        refs = [BC.opening_check(S_, a) for _n, _o, a, _w in group]
        assert [BC.verdict(r) for r in refs] == [w for *_x, w in group], [n for n, *_ in group]
        assert _gates(bind(BC.BlockOpeningCheck, **vars(S_)), [a for _n, _o, a, _w in group]) == refs, leaf


@pytest.mark.slow
def test_the_parts_cost_the_pinned_gates() -> None:
    """About a minute alone (every part's trace): the deployed unit's cost is these parts' (`block_check`'s Cost), and no
    lowering at the deployed sizes runs anywhere else to pin it."""
    for (name, statics), counts in PARTS.items():
        got = getattr(BC, name).traced(*statics).counts
        assert got == dict(zip(("and", "xor", "not"), counts)), (name, got)


@pytest.mark.slow
def test_the_toy_unit_is_boolean() -> None:
    assert BL.is_boolean(bind(BC.BlockOpeningCheck, **vars(K.TOY)))
