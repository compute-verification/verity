"""`verity.ml.definitions.registry.boolean_dense`: the Boolean versions of `registry.dense`'s element-wise rows agree with their
word Definitions (`trace.word_of`): the rows whose element reads one bf16 word on every word at Gemma-2's constants, the others on
every pair of special words, on random words and on products and sums at f32's subnormal and overflow ends. They are pinned by
gate count and program digest, a shared scalar's gates and its element's are the ANDs of the operation they cut, and they pass
`Q_word_v1{X=16, W=32}`'s cut check."""
from __future__ import annotations

import itertools
import json
import zlib

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
from verity.ml.circuits.ops.boolean import elementwise as EW
from verity.ml.circuits.silicon import fp
from verity.ml.definitions.registry import b1
from verity.ml.definitions.registry import boolean_dense as BD
from verity.ml.definitions.registry import dense as D

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "SquareBf16_v2{N=8}": (5024, 6928, 552, "8e841f597a15e0a0c27e55ec59ea9a2a99eb0f3c43e0a9f6b2d6448f52e9df8b41c35adfb7c9c41cca22279d1d67b15f91e83471ef11c1338b4412fbad71ef2c"),
    "SquareF32_v2{N=8}": (15824, 32088, 824, "b5c3d5b5cbfd87a402c4218474f0ae5212b7f2f478ada1377e263933a7f2cb67663db627cf79421908fba0eb3c82e4a7adbbcf474153278958a233fcacfdd4c6"),
    "AddScalarF32_v2{N=8,C=1e-06}": (5936, 9528, 992, "2e426d23eab02bb9f81d1679db80aba8296ce90f617597693f95a646005415503cab20ba56cada2887b0297edd73e0ddf9d2bb6c5642cd94cd5930d2c23ea907"),
    "AddScalarBf16_v2{N=8,C=1.0}": (4968, 6984, 680, "2095c31aa92974eec555e3162ca6539e7685d91e8d894cd429426d84c3ba1fd463b88b2113a9ed6342ceb85e81e8fe400925f42df7d8d7f1c6a0bc1740a5e3e9"),
    "AddWidenedBf16_v2{N=8}": (5328, 7960, 792, "e52c5cc93b800e37427238e54293f1722d59e1497f3e5d82e67c48700d44452724f163143a3aacad5b0847171fcf90faaeb8dd56699dec5b8efaa0a20a6df356"),
    "ScaleRowBf16_v2{N=8}": (9138, 15566, 698, "81115d487167d76f6093d1affc2184a9fca6b59e56f1abb672e68f4d98aadb32fa7237e967c2b3544a9dc1b12d8972e304b7b3d688bd1ec06194a4eb9134b361"),
    "ScaleRowF32_v2{N=8}": (19202, 36766, 866, "3f074cf5f91f65fd554a444e21356a729b393588a151921b5af8daecb3767493127d407780979763d18856c2cfaf9f38b28e2343b797c4618351b0a39c5643dc"),
    "MulVecF32_v2{N=8}": (19440, 37200, 880, "6c52d0d148f189647e49b3c401881a257a00d8aa342a077838b7826c0f60c6f464f6baa467f038469dfaacf61877a83c7a2c9cc8ce08f27a69c9f852e2835d1e"),
    "Bf16DivScalar_v2{N=8,C=30.0}": (6920, 10264, 936, "c0abf4b14d7fe109beecf09d639cb85efe61ba62f271c78d2c3c46dfb1a378b96d179e7eec4c7ebdbdfd9e243e645d4a46097e1b2c69eb729785c1c124bdfae1"),
    "Bf16MulScalar_v2{N=8,C=30.0}": (5032, 6672, 752, "a87bc65a022f5a0ecdc895fc9144e0fa7b21cf0a3dd73284fe398743179176699a505955c7160c9fbe57ef82426ff49f1e5319b478e4593e16dceba8248a9eae"),
    "Bf16MulScalarTensor_v2{N=8}": (6266, 9030, 778, "be8e1fa1b4387879e49ac2fa72edea3f26bfb53d1924710fde4418b3cfa9101ba203d01684d47679408aadf6364ff977df3cf510a560849d57b6486cb0445706"),
}

_EXP = [0, 1, 2, 116, 117, 118, 119, 120, 126, 127, 128, 129, 130, 131, 133, 134, 135, 254, 255]
#: per sign: zero and the subnormal ends, the exponents where `gelu_tanh_bf16` leaves its closed forms at the mantissa ends,
#: ones, the finite maximum, the infinities and NaNs (quiet, signalling, all ones)
_MAGNITUDES = sorted({e << 7 | m for e in _EXP for m in (0, 1, 0x40, 0x7F) if e < 255} | {0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC0, 0x7FFF})
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], np.uint16)


def _word(target, *inputs) -> np.ndarray:
    return np.asarray(E.evaluate_batch(TR.word_of(target), *inputs), np.uint64).reshape(-1)


def _bits(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs), np.uint64).reshape(-1)


def _agree(target, *inputs) -> None:
    got, want = _bits(target, *inputs), _word(target, *inputs)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{target.id}: {bad.size} of {len(want)} differ from {TR.word_of(target).id}, first {bad[:5]}"


#: Gemma-2-2B's constants: GemmaRMSNorm's epsilon and weight offset, the logit soft cap
GEMMA = {BD.AddScalarF32: {"C": 1e-6}, BD.AddScalarBf16: {"C": 1.0}, BD.Bf16DivScalar: {"C": 30.0}, BD.Bf16MulScalar: {"C": 30.0}}
#: each element-wise row on bits -> its word Definition
ROW_WORDS = {BD.SquareBf16: D.SquareBf16, BD.SquareF32: D.SquareF32, BD.AddScalarF32: D.AddScalarF32, BD.AddScalarBf16: D.AddScalarBf16,
             BD.AddWidenedBf16: D.AddWidenedBf16, BD.ScaleRowBf16: D.ScaleRowBf16, BD.ScaleRowF32: D.ScaleRowF32,
             BD.MulVecF32: D.MulVecF32, BD.Bf16DivScalar: D.Bf16DivScalar, BD.Bf16MulScalar: D.Bf16MulScalar,
             BD.Bf16MulScalarTensor: b1.Bf16MulScalarTensor}


def _row(d, N: int):
    return bind(d, N=N, **GEMMA.get(d, {}))


ROWS = [_row(d, 8) for d in ROW_WORDS]


@pytest.mark.parametrize("target", ROWS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    back = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert back.gates == prog.gates and codec.program_digest(codec.encode_program(back)) == digest


# --- the element-wise rows of Gemma-2's RMSNorm chain, embedding scale and logit soft cap ---------------------------------------

#: f32 specials: signed zeros, subnormals (both ends, the gradual-underflow boundary), the normal range's ends, ones and their
#: neighbours, Gemma-2's constants, powers of two that scale into and out of the subnormal range, infinities and NaNs
_F32 = [0x00000000, 0x00000001, 0x00000002, 0x00400000, 0x007FFFFF, 0x00800000, 0x00800001, 0x00FFFFFF, 0x0C000000, 0x1F800000,
        0x33800000, 0x358637BD, 0x3D088889, 0x3F000000, 0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x41F00000, 0x5F800000, 0x7E800000,
        0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FBFFFFF, 0x7FC00000, 0x7FFFFFFF]
F32_SPECIALS = np.array([m | s for m in _F32 for s in (0, 0x80000000)], np.uint32)
EVERY = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)


def _f32_words(rng, exp: np.ndarray) -> np.ndarray:
    """f32 words of either sign with the exponent fields ``exp`` and random mantissas."""
    return (rng.integers(0, 2, exp.size).astype(np.uint64) << np.uint64(31) | exp.astype(np.uint64) << np.uint64(23)
            | rng.integers(0, 1 << 23, exp.size).astype(np.uint64)).astype(np.uint32)


def _edge(rng, ea: np.ndarray) -> np.ndarray:
    """Exponent fields of a second operand that put a product's (or, for sums, the operands' common) exponent at f32's
    subnormal end (-26..3) or its overflow (250..257), against first operands of exponent fields ``ea``."""
    target = np.where(rng.integers(0, 2, ea.size) == 1, rng.integers(-26, 4, ea.size), rng.integers(250, 258, ea.size))
    return np.clip(target - ea + 127, 0, 254)


def _operands(rng, kinds: str) -> list[np.ndarray]:
    """Operands of the kinds ``kinds`` (``h`` bf16, ``f`` f32), each a column of instances: every pair of special words, every
    bf16 word against 8 specials of the other operand (when one is bf16), 100,000 random tuples, 20,000 whose product lands at
    f32's subnormal or overflow end and, for two operands, 20,000 that cancel to a few ulps."""
    spec = {"h": SPECIALS, "f": F32_SPECIALS}
    cols = [[] for _ in kinds]
    for g in itertools.product(*[range(len(spec[k])) for k in kinds]):
        for c, k, i in zip(cols, kinds, g):
            c.append(spec[k][i])
    cols = [np.array(c, np.uint16 if k == "h" else np.uint32) for c, k in zip(cols, kinds)]
    if "h" in kinds and len(kinds) == 2:
        h = kinds.index("h")
        other = spec[kinds[1 - h]][rng.integers(0, len(spec[kinds[1 - h]]), 8)]
        pair = [None, None]
        pair[h], pair[1 - h] = np.repeat(EVERY, len(other)), np.tile(other, len(EVERY))
        cols = [np.concatenate([c, p]) for c, p in zip(cols, pair)]
    rand = [rng.integers(0, 1 << (16 if k == "h" else 32), 100_000, dtype=np.uint64).astype(np.uint16 if k == "h" else np.uint32)
            for k in kinds]
    a = _f32_words(rng, rng.integers(0, 255, 20_000))
    ea = a >> np.uint32(23) & np.uint32(0xFF)
    edge = [a, _f32_words(rng, _edge(rng, ea.astype(np.int64)))][:len(kinds)]
    near = (a.astype(np.int64) + rng.integers(-4, 5, a.size)).clip(0, 0xFFFFFFFF).astype(np.uint32) ^ np.uint32(0x80000000)
    canc = [a, near][:len(kinds)]
    extra = [np.concatenate([e, c]) for e, c in zip(edge, canc)] if len(kinds) == 2 else [edge[0]]
    extra = [(x >> np.uint32(16)).astype(np.uint16) if k == "h" else x for x, k in zip(extra, kinds)]
    return [np.concatenate([c, r, x]) for c, r, x in zip(cols, rand, extra)]


def _row_inputs(d, rng) -> list[np.ndarray]:
    """``d``'s row at N = 1 over `_operands`: one element and, for a row with a scalar, its scalar, per instance."""
    kinds = {BD.SquareF32: "f", BD.AddScalarF32: "f", BD.AddWidenedBf16: "hh", BD.ScaleRowBf16: "hf", BD.ScaleRowF32: "ff",
             BD.MulVecF32: "ff", BD.Bf16MulScalarTensor: "hh"}[d]
    return [c.reshape(-1, 1) for c in _operands(rng, kinds)]


def test_the_row_word_views() -> None:
    for t in ROWS:
        assert TR.word_of(t).id == bind(ROW_WORDS[t.defn], **t.bindings).id == t.id.replace("_v2{", "_v1{")


@pytest.mark.parametrize("d", [BD.SquareBf16, BD.AddScalarBf16, BD.Bf16DivScalar, BD.Bf16MulScalar], ids=lambda d: d.name)
def test_the_one_word_rows_on_every_word(d) -> None:
    """A row whose element reads one bf16 word, at Gemma-2's constant: all 65,536 words."""
    _agree(_row(d, 1), EVERY.reshape(-1, 1))


@pytest.mark.parametrize("d", [BD.SquareF32, BD.AddScalarF32, BD.ScaleRowF32, BD.MulVecF32, BD.AddWidenedBf16]
                         + [pytest.param(d, marks=pytest.mark.slow) for d in (BD.ScaleRowBf16, BD.Bf16MulScalarTensor)],
                         ids=lambda d: d.name)
def test_the_two_word_rows_on_specials_random_and_edges(d) -> None:
    """`_operands` at N = 1 (a row with a bf16 operand reads every bf16 word against 8 specials: slow, about 15 s)."""
    _agree(_row(d, 1), *_row_inputs(d, np.random.default_rng(zlib.crc32(d.name.encode()))))


@pytest.mark.parametrize("d", list(ROW_WORDS), ids=lambda d: d.name)
def test_rows_of_eight(d) -> None:
    """Rows of 8 elements, so a row's scalar part is shared by its members: 5,000 random rows and 1,000 of special words."""
    rng = np.random.default_rng(zlib.crc32(d.name.encode()) + 1)
    ins = []
    for _, t in _row(d, 8).signature()[0]:
        n, w = t.n, t.elem.n
        spec = SPECIALS if w == 16 else F32_SPECIALS
        ins.append(np.concatenate([rng.integers(0, 1 << w, (5000, n), dtype=np.uint64).astype(spec.dtype),
                                   spec[rng.integers(0, len(spec), (1000, n))]]))
    _agree(_row(d, 8), *ins)


def test_the_shared_scalar_rows_split_their_builders() -> None:
    """A row's scalar part and its element together have exactly the ANDs of the operation they cut: F32Mul_v3's, the bf16 by
    f32 multiply's and Bf16MulBf16_v2's; and a row of N is N elements and one scalar part."""
    bf16_by_f32 = TR.trace(lambda C, x, s: fp.f32_mul(C, fp.bf16_to_f32(x), s, ir_nan=True), [BD.BF16, BD.F32], BD.F32)
    for own, rest, whole in ((BD.ScaleRowF32Scalar, BD.ScaleRowF32Element, EW.F32Mul.traced()),
                             (BD.ScaleRowBf16Scalar, BD.ScaleRowBf16Element, bf16_by_f32),
                             (BD.Bf16MulScalarTensorScalar, BD.Bf16MulScalarTensorElement, EW.Bf16MulBf16.traced())):
        assert own.traced().counts["and"] + rest.traced().counts["and"] == whole.counts["and"]
    for d, own, rest in ((BD.ScaleRowF32, BD.ScaleRowF32Scalar, BD.ScaleRowF32Element),
                         (BD.ScaleRowBf16, BD.ScaleRowBf16Scalar, BD.ScaleRowBf16Element),
                         (BD.Bf16MulScalarTensor, BD.Bf16MulScalarTensorScalar, BD.Bf16MulScalarTensorElement)):
        assert PINS[_row(d, 8).id][0] == 8 * rest.traced().counts["and"] + own.traced().counts["and"]


@pytest.mark.parametrize("d", list(ROW_WORDS), ids=lambda d: d.name)
def test_rows_pass_the_word_querys_cut_check(d) -> None:
    """No value computed twice and every cross-unit read committed: a shared scalar is decoded once per row, a constant folded."""
    prog, _ = standalone_call(_row(d, 3))
    assert PO.verify(PO.build(prog, PO.query(16, 32)), prog).ok
