"""`registry.boolean_rope`: the Boolean NeoX rotation agrees with its word Definitions (`trace.word_of`) on every quadruple of
special bf16 words and on 90k random and structured quadruples per half, on random heads against `RoPEHead_v1` through the IR
reference, and on 81k random heads of each RoPE shape SmolLM2-135M serves (q: NHEADS=9, k: NHEADS=3, D=64); carries exactly its
builders' live ANDs; is pinned by gate count and program digest; and cuts under `Q_word` v1 into one 16-bit unit per output
element, as the word Program does."""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pytest

from verity import evaluation as E
from verity.evaluation import bits as BS
from verity.evaluation.reference import standalone_call
from verity.ir import boolean as BL
from verity.ir import codec
from verity.ir import partition_object as PO
from verity.ir.defs import REGISTRY, bind
from verity.ml.boolean import trace as TR
from verity.ml.boolean.forms import Circuit
from verity_vllm.program.kernels import twins  # noqa: F401  (registers RoPE_v1's kernel, which the served shapes' word side runs)
from verity_vllm.program.kernels.twins import rand_bf16, tie_pairs_mul
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_rope as R

U16 = np.uint16
HALVES = [R.RopeOut, R.RopeOutAdd]
HEAD = bind(R.RoPEHead, D=64)
Q, K = bind(R.RoPE, NHEADS=9, D=64), bind(R.RoPE, NHEADS=3, D=64)

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "RopeOut_v2": (2966, 4608, 295, "5677d123ea2492ecbb3f4d477325ac7a6407d5a393ae8e094291b7640661c4fd"),
    "RopeOutAdd_v2": (2966, 4608, 294, "358c043fb20c3738d2e852c96f10910877053f5c9dcd78cb3e90ccb6a54a3b02"),
    HEAD.id: (189824, 294912, 18848, "84bd8192e6ddd91076d45608e0fe5ecf41e3d90b8a4d6bf0b11537503c02cff2"),
    Q.id: (1708416, 2654208, 169632, "c0f8d5b33977be128701803c57cdbb9f277cb43e6ae58c4c64c6712abfe0dfd8"),
    K.id: (569472, 884736, 56544, "3b75b585c31aa6954a3db2f7ec5c7cdcd8f83541a8342b19f2f5283f6dd55504"),
}
TARGETS = [bind(R.RopeOut), bind(R.RopeOutAdd), HEAD, Q, K]

#: signed zeros, the smallest subnormal and normal, 2^-64 and 2^64 (products at the f32 range's ends), one, sqrt(2), the
#: largest finite word, infinities, a quiet and a signalling NaN; each with both signs
_MAGNITUDES = [0x0000, 0x0001, 0x0080, 0x1F80, 0x3F80, 0x3FB5, 0x5F80, 0x7F7F, 0x7F80, 0x7FC0, 0x7F81]
SPECIALS = np.array([m | s for m in _MAGNITUDES for s in (0, 0x8000)], U16)


def _bf16(v: np.ndarray) -> np.ndarray:
    return twins._bf16_rn(np.asarray(v, np.float32), 0x7FC0)


def _cos_sin(rng, n: int) -> tuple[np.ndarray, np.ndarray]:
    th = rng.uniform(-np.pi, np.pi, n)
    return _bf16(np.cos(th)), _bf16(np.sin(th))


def _quadruples(rng) -> list[np.ndarray]:
    """(x, y, c, s) columns: uniform words; activations against a cos/sin pair; x*c close to y*s (cancellation, the FMA's one
    rounding); y*s in f32's subnormal range (FMUL's rounding); x*c on a bf16 rounding tie with y*s zero or tiny."""
    uniform = rng.integers(0, 1 << 16, (4, 40_000), dtype=np.uint64).astype(U16)
    n = 20_000
    c, s = _cos_sin(rng, n)
    real = np.stack([rand_bf16(rng, n, "spread"), rand_bf16(rng, n, "spread"), c, s])
    n = 10_000
    x, c = rand_bf16(rng, n, "wide"), rand_bf16(rng, n, "wide")
    y = (x.astype(np.int64) + rng.integers(-3, 4, n)).astype(U16)
    s = (c.astype(np.int64) + rng.integers(-3, 4, n)).astype(U16)
    cancel = np.stack([x, y, c, s])
    ey = rng.integers(37, 90, n)
    e = np.stack([ey, np.clip(rng.integers(100, 133, n) - ey, 1, 254)])   # y*s near 2^(100-254)..2^(132-254)
    m = rng.integers(0, 1 << 8, (2, n))
    y, s = ((e << 7) | (m & 0x7F) | ((m >> 7) << 15)).astype(U16)
    tiny = np.stack([rand_bf16(rng, n, "subnormal"), y, rand_bf16(rng, n, "subnormal"), s])
    x, c = tie_pairs_mul(rng, n)
    y = rand_bf16(rng, n, "subnormal")
    s = np.where(rng.random(n) < 0.5, 0, rand_bf16(rng, n, "subnormal")).astype(U16) ^ (rng.integers(0, 2, n) << 15).astype(U16)
    ties = np.stack([x, y, c, s])
    return list(np.concatenate([uniform, real, cancel, tiny, ties], axis=1))


def _agree(target, inputs) -> None:
    word = TR.word_of(target)
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64)
    assert got.shape == ref.shape
    bad = np.flatnonzero((got.reshape(len(got), -1) != ref.reshape(len(ref), -1)).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("target", TARGETS, ids=[t.id for t in TARGETS])
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("d", HALVES, ids=lambda d: d.name)
def test_the_ands_are_the_builders_live_ands(d) -> None:
    T = d.traced()
    params = [t for _, t in d.signature(None)[0]]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, params)
    C.output("out", TR._flat_out(d.build(C, *ins), d.signature(None)[1]))
    assert T.counts["and"] == C.counts()["ands"] == PINS[f"{d.name}_v{d.version}"][0]


def test_the_word_definitions() -> None:
    assert [TR.word_of(bind(d)).id for d in HALVES] == ["RopeOut_v1", "RopeOutAdd_v1"]
    assert [TR.word_of(t).id for t in (HEAD, Q, K)] == ["RoPEHead_v1{D=64}", "RoPE_v1{NHEADS=9,D=64}", "RoPE_v1{NHEADS=3,D=64}"]


@pytest.mark.parametrize("d", HALVES, ids=lambda d: d.name)
def test_each_half_agrees_on_every_quadruple_of_special_words(d) -> None:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=4)), np.int64).T
    _agree(bind(d), [SPECIALS[g] for g in grid])


@pytest.mark.parametrize("d", HALVES, ids=lambda d: d.name)
def test_each_half_agrees_on_random_and_structured_quadruples(d) -> None:
    inputs = _quadruples(np.random.default_rng(zlib.crc32(d.name.encode())))
    assert len(inputs[0]) == 90_000
    _agree(bind(d), inputs)


def _heads(rng, n: int, nheads: int) -> tuple[np.ndarray, np.ndarray]:
    """n instances of (x: n x nheads x 64, cs: n x 64): a quarter each of uniform words, activations against a cos/sin row,
    wide words and words with specials sprinkled in."""
    q = n // 4
    xs, rows = [], []
    for kind in ("uniform", "spread", "wide", "special"):
        m = n - 3 * q if kind == "special" else q
        if kind == "uniform":
            x = rng.integers(0, 1 << 16, (m, nheads, 64), dtype=np.uint64).astype(U16)
            cs = rng.integers(0, 1 << 16, (m, 64), dtype=np.uint64).astype(U16)
        else:
            x = rand_bf16(rng, m * nheads * 64, kind).reshape(m, nheads, 64)
            c, s = _cos_sin(rng, m * 32)
            cs = np.concatenate([c.reshape(m, 32), s.reshape(m, 32)], axis=1)
            if kind == "wide":
                cs = rand_bf16(rng, m * 64, kind).reshape(m, 64)
        xs.append(x)
        rows.append(cs)
    return np.concatenate(xs), np.concatenate(rows)


def test_the_head_agrees_with_rope_head_v1() -> None:
    """Against `RoPEHead_v1{D=64}` evaluated primitive by primitive (the IR reference), so the halves' wiring is checked
    without any kernel."""
    x, cs = _heads(np.random.default_rng(64), 1500, 1)
    _agree(HEAD, [x[:, 0], cs])


@pytest.mark.slow
@pytest.mark.parametrize("target", [Q, K], ids=["q-NHEADS=9", "k-NHEADS=3"])
def test_the_served_shapes_agree_with_rope_v1_on_81k_heads(target) -> None:
    """The q and k rotations of SmolLM2-135M's layers: `RoPE_v1{NHEADS, D=64}` runs through its registered kernel
    (`twins.RoPETwin`, self-checked against the IR reference by the kernel registry)."""
    nh = target.bindings["NHEADS"]
    x, cs = _heads(np.random.default_rng(nh), -(-81_000 // nh), nh)
    assert x.shape[0] * nh >= 81_000
    _agree(target, [x, cs])


@pytest.mark.parametrize("target", [HEAD, K], ids=lambda t: t.id)
def test_one_unit_per_output_element_as_in_the_word_program(target) -> None:
    """`Q_word` v1 cuts both into 16-bit units, one per output element: the word Program's units (`RopeOut_v1` and
    `RopeOutAdd_v1` gates)."""
    prog, _ = standalone_call(target)
    word, _ = standalone_call(TR.word_of(target))
    part, ref = PO.evaluate(prog), PO.evaluate(word)
    n = 64 * target.bindings.get("NHEADS", 1)
    assert part.population == ref.population == n
    (_k, cut), = part.calls
    assert cut.units == n
