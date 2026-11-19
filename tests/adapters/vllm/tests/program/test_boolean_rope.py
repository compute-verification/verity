"""`registry.boolean_rope`: the Boolean NeoX rotation agrees with its word Definitions (`trace.word_of`) on every quadruple of
special bf16 words and on 90k random and structured quadruples per half, on random heads of each served head width against
`RoPEHead_v1` through the reference evaluator, on 16k random heads of each served head width against `RoPE_v1`'s kernel, and on 1,024
random heads of every `RoPE_v1` binding the served workload Programs call (`SERVED`); carries exactly its builders' live ANDs;
is pinned by gate count and program digest; and cuts under `Q_word` v1 into one 16-bit unit per output element, as the word
Program does. The interleaved (GPT-J) versions agree with `RoPEHeadGptJ_v2` / `RoPEGptJ_v2` through the reference evaluator at
DeepSeek-V2 MLA's q_pe and k_pe shapes and reproduce the words vLLM's `rotary_embedding(is_neox=False)` wrote on sm_120 where
`RoPEHeadGptJ_v1` does not."""
from __future__ import annotations

import itertools
import json
import zlib

import numpy as np
import pytest
from verity.circuits import boolean as BL
from verity.circuits import codec
from verity.circuits import evaluation as E
from verity.circuits import partition_object as PO
from verity.circuits.boolean import trace as TR
from verity.circuits.boolean.forms import Circuit
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import standalone_call

from verity_vllm.definitions.numerics import (
    twins,  # registers RoPE_v1's kernel, which the served shapes' word side runs
)
from verity_vllm.definitions.numerics.twins import rand_bf16, tie_pairs_mul
from verity_vllm.definitions.registry import b1, ref_prims
from verity_vllm.definitions.registry import boolean_rope as R

U16 = np.uint16
HALVES = [R.RopeOut, R.RopeOutAdd]

#: (NHEADS, D) of the `RoPE_v1` Calls in each model's traced workload Programs, q then k (one binding where the model has as
#: many kv heads as heads): every Build descriptor of the coverage rows
SERVED = {
    "SmolLM2-135M": ((9, 64), (3, 64)),
    "Llama-3.2-1B": ((32, 64), (8, 64)),
    "Qwen2.5-0.5B": ((14, 64), (2, 64)),
    "Qwen2.5-1.5B": ((12, 128), (2, 128)),
    "Qwen3-4B": ((32, 128), (8, 128)),
    "Gemma-2-2B": ((8, 256), (4, 256)),
    "Phi-3-mini": ((32, 96),),
    "OLMoE-1B-7B": ((16, 128),),
}
ROPES = sorted({bind(R.RoPE, NHEADS=n, D=d) for shapes in SERVED.values() for n, d in shapes}, key=lambda t: t.id)
HEADS = {d: bind(R.RoPEHead, D=d) for d in sorted({t.bindings["D"] for t in ROPES})}
K = bind(R.RoPE, NHEADS=3, D=64)
#: one served binding per head width, run on 16k heads: SmolLM2's q and k, Phi-3's, Qwen3's q, Gemma-2's q
LARGE = [bind(R.RoPE, NHEADS=n, D=d) for n, d in ((9, 64), (3, 64), (32, 96), (32, 128), (8, 256))]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "RopeOut_v2": (2966, 4608, 295, "454ca30ec8efb77355a5ae947133cb7715c69dd788069c38b747e6800b1629372c2c0379e492ef60952954a0a95b5ffec711e13fc8517bb52c8333c0ad488358"),
    "RopeOutAdd_v2": (2966, 4608, 294, "37e69d3728468aa0effe323a5c33718c8dcf7b05c68cf59f1d1366474cc100b79800929de3488d8de159dc096ae81ea281b93c70ce6f3e37a13f342954186e55"),
    "RoPEHead_v2{D=64}": (189824, 294912, 18848, "65324a8415d369e024c140502ab5e32068b2aa73a257018ad50d52e9532fb446df2339b8552aaab13dff5966cf6bf9e7cfb84af0cfcd2654ca6eeb4572540af3"),
    "RoPEHead_v2{D=96}": (284736, 442368, 28272, "dd26a3c01dbd16cc6cb4bcdf7436fe187a216c67ce4ec517910038e647376b36e1e6ebb6a680a0bb93e98b6953d6e7298d44312a31838a2118dcf4dd05a2c905"),
    "RoPEHead_v2{D=128}": (379648, 589824, 37696, "080b4330036c16c82d0eba7271dcc11a5a1fa104aac769429d17e280092ef54bf0cb4d1c207f30710600d611a5bc868b548d7e60b7cf1c2f07a3db4cc5b2892e"),
    "RoPEHead_v2{D=256}": (759296, 1179648, 75392, "42dc56d96ea451d97b24d2acda00cb5279a2b6f808348813c860eb5c1b5771faa4a434f32f9dfebc07a3c91d7d6b87aead2eed479117f8de53432f08afc940c6"),
    "RoPE_v2{NHEADS=9,D=64}": (1708416, 2654208, 169632, "ca5f6ab4c27719f4032c9e4a4c175430a5c33bb8cb96f08bbc074ea6a1b9d49cba4a9ea27625c620fca2494e8eab725fe167c22737448b0e8a2f6fb958c5d7f3"),
    "RoPE_v2{NHEADS=3,D=64}": (569472, 884736, 56544, "87bf2a631ff4c911de11fd376c893bb2c5ebc69c1cbcc3f44ae33790b1bcdce9fe73ca33b1c9a7dbb15c8c93072035cea6f6e5687a9677cd317dba35fcce99af"),
    "RoPE_v2{NHEADS=32,D=64}": (6074368, 9437184, 603136, "64837adc659e2cec3e28c020be2aa598ea2cfde8168b8c22e2e85d40c0b60bdc9d5ca7f199378a5e84a71c91da94669537d296c1ea940366a62bba5483dd1c0b"),
    "RoPE_v2{NHEADS=8,D=64}": (1518592, 2359296, 150784, "cd1ffffc8219f8ff5d66c3e6f70b9946e4cd62b1a379519f14b3d8f96d1dded886d578c8bd2be4035acc682cd0ca49f5176a2f50c84e45332cc5db16ab042dbc"),
    "RoPE_v2{NHEADS=14,D=64}": (2657536, 4128768, 263872, "6a95a6ed6c1b67ebffd86146b0d2c1502dccbc848a0dfda3e066f62e03084a89a2124b4b8603542b51d503678230cd349c05eb75814f70db4e455a1a2c460580"),
    "RoPE_v2{NHEADS=2,D=64}": (379648, 589824, 37696, "7344c1ba937d91806b21e83eb3d3d44ab1998f42962b7bcb08a6c6cd0a55f28988a29b35bea85b3574fb21bd151be572229bc7f82e314b33066b25dd383360bc"),
    "RoPE_v2{NHEADS=32,D=96}": (9111552, 14155776, 904704, "eebe612ee8420120413a018529954535fea60a35e0cf2f4e52ba58614a5c683e2435a5045e740e4378561d6e8edc9296d3d659cfd4bc3b789509aa0d11e4d46c"),
    "RoPE_v2{NHEADS=16,D=128}": (6074368, 9437184, 603136, "165d2476634d490b240f0c326efaa0140b09ab56720db4d163ae888d365f972cc3df369cd7ac1dcce5d4c382a6ef1e9b162d6ffa7439761250f5b68e585fa9dd"),
    "RoPE_v2{NHEADS=12,D=128}": (4555776, 7077888, 452352, "9f2fb79f2ebc2fc1db530ac1dde1496d8d17bcef8dc7a9ba37bb01bd95fa3956bc21381845f9dfc170b8ecbe46ff06c756dd50fa2b371fd1b24d344c486ac872"),
    "RoPE_v2{NHEADS=2,D=128}": (759296, 1179648, 75392, "c672883411a7fc79d5c36dcc74443ffd984069fac31abd5d0b3ee9722310e7908da94dee2c97cc359bcdf5908d092f1e1bad9017eb8dae2f082ae8aa4814b1a9"),
    "RoPE_v2{NHEADS=32,D=128}": (12148736, 18874368, 1206272, "139bc450c9b36e9229c7079d55c714023f5ecfffaa76b1a14e5503260140d5f369337167cca0246c8cd8510ac67628ce12ac67d4928b7f74eb27663769f18b6e"),
    "RoPE_v2{NHEADS=8,D=128}": (3037184, 4718592, 301568, "2fd7e2d81e2a04cc4f0dcf0933f89a449b86e5bfa0115e78c6808881cf03645e7773ab80901ce02de536db4fd1284714d2b418aaf000c64f0bd1daccd3bc1ff8"),
    "RoPE_v2{NHEADS=8,D=256}": (6074368, 9437184, 603136, "f74d89ad95d40f74043e415de84889e128e787448152cafad105992a0a3b084508bcc93ba6777a017541201636fd319ee656d65bd94f8f08534dc84b22c4d367"),
    "RoPE_v2{NHEADS=4,D=256}": (3037184, 4718592, 301568, "63c509f0938a8804d79b0cc0dbae9bca54409ae9d4670cd04adfefb1a8c8f3ae330ad8fbd2265988048fdf6848a61ebaacb0aef831bfae4d372100e55c15bd48"),
    "RoPEHeadGptJ_v3{D=64}": (189824, 294912, 18848, "1ce7c9d3104d3a82e1fd4f5ef6a481bdbee822d3dce11f2835ba668e593f254b38c53a1a047640856c60093afacb578b6a0cf2c3e12615040b0fdcb4e65c94f1"),
    "RoPEGptJ_v3{NHEADS=1,D=64}": (189824, 294912, 18848, "6e29af1ff7481f54119bb08465516444379fdb73384166312b55c2ea517ef56415deb00600db41e3feed7fe3309f081b08e85b6c88f64dcb4aaae1ee6baf4885"),
    "RoPEGptJ_v3{NHEADS=5,D=64}": (949120, 1474560, 94240, "0aac17d86ec812a3e607894d8abb8e8bd384e5e1fca348b4b3f01ef4efc06ab24e6116d97b5af9184d553d52c51d311c02398cf98ef59f46027f3eb5c07cb798"),
    "RoPEGptJ_v3{NHEADS=8,D=64}": (1518592, 2359296, 150784, "a4fe2883c4f0a9503cbcf74706a81906f47921ac6220dad0b8c3529f607bd9d0145a52da4db55115ae46c758484fab05a4a1c4d0a9d393cb5efc062497caf366"),
    "RoPEGptJ_v3{NHEADS=20,D=64}": (3796480, 5898240, 376960, "b2d19640a54080465f217ad625c0bed3479cbb50c7fbb42f277a8caafbae9cd15982ce4f3be6872f1d44c5b64c1d233bbc100a0bcb021c958f594a063f3103e4"),
}
GPTJ_HEAD = bind(R.RoPEHeadGptJ, D=64)
#: k_pe's one head; GLM-4.7-Flash's q_pe at TP4 and TP1 (5 and 20 heads); GLM-5.3's at TP8 (8 heads)
GPTJ = [bind(R.RoPEGptJ, NHEADS=n, D=64) for n in (1, 5, 8, 20)]
TARGETS = [bind(R.RopeOut), bind(R.RopeOutAdd), *HEADS.values(), *ROPES, GPTJ_HEAD, *GPTJ]

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


def _agree(target, inputs, chunk: int = 1 << 20) -> None:
    """Instances in slices of about `chunk` words of the first input: the bit-sliced evaluator holds ~700 bytes per word."""
    word = TR.word_of(target)
    n, step = len(inputs[0]), max(1, chunk // inputs[0][0].size)
    bad: list = []
    for i in range(0, n, step):
        part = [a[i:i + step] for a in inputs]
        got = BS.evaluate_words(target, *part).astype(np.uint64)
        ref = np.asarray(E.evaluate_batch(word, *part)).astype(np.uint64)
        assert got.shape == ref.shape
        bad.extend(i + np.flatnonzero((got.reshape(len(got), -1) != ref.reshape(len(ref), -1)).any(axis=1)))
    assert not bad, f"{target.id}: {len(bad)} of {n} instances differ from {word.id}, first {bad[:5]}"


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


def test_every_pin_is_a_target() -> None:
    assert sorted(PINS) == sorted(t.id for t in TARGETS)


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
    assert all(TR.word_of(t) == bind(b1.RoPEHead, **t.bindings) for t in HEADS.values())
    assert all(TR.word_of(t) == bind(b1.RoPE, **t.bindings) for t in ROPES)
    assert TR.word_of(GPTJ_HEAD) == bind(R.RoPEHeadGptJV2, D=64)
    assert all(TR.word_of(t) == bind(R.RoPEGptJV2, **t.bindings) for t in GPTJ)


@pytest.mark.parametrize("d", HALVES, ids=lambda d: d.name)
def test_each_half_agrees_on_every_quadruple_of_special_words(d) -> None:
    grid = np.array(list(itertools.product(range(len(SPECIALS)), repeat=4)), np.int64).T
    _agree(bind(d), [SPECIALS[g] for g in grid])


@pytest.mark.parametrize("d", HALVES, ids=lambda d: d.name)
def test_each_half_agrees_on_random_and_structured_quadruples(d) -> None:
    inputs = _quadruples(np.random.default_rng(zlib.crc32(d.name.encode())))
    assert len(inputs[0]) == 90_000
    _agree(bind(d), inputs)


def _heads(rng, n: int, nheads: int, D: int) -> tuple[np.ndarray, np.ndarray]:
    """n instances of (x: n x nheads x D, cs: n x D): a quarter each of uniform words, activations against a cos/sin row,
    wide words and words with specials sprinkled in."""
    q, R_ = n // 4, D // 2
    xs, rows = [], []
    for kind in ("uniform", "spread", "wide", "special"):
        m = n - 3 * q if kind == "special" else q
        if kind == "uniform":
            x = rng.integers(0, 1 << 16, (m, nheads, D), dtype=np.uint64).astype(U16)
            cs = rng.integers(0, 1 << 16, (m, D), dtype=np.uint64).astype(U16)
        else:
            x = rand_bf16(rng, m * nheads * D, kind).reshape(m, nheads, D)
            c, s = _cos_sin(rng, m * R_)
            cs = np.concatenate([c.reshape(m, R_), s.reshape(m, R_)], axis=1)
            if kind == "wide":
                cs = rand_bf16(rng, m * D, kind).reshape(m, D)
        xs.append(x)
        rows.append(cs)
    return np.concatenate(xs), np.concatenate(rows)


@pytest.mark.parametrize("head", HEADS.values(), ids=lambda t: t.id)
def test_the_head_agrees_with_rope_head_v1(head) -> None:
    """Against `RoPEHead_v1{D}` evaluated primitive by primitive (the reference evaluator) on 96k words, so the halves' wiring is
    checked without any kernel."""
    D = head.bindings["D"]
    x, cs = _heads(np.random.default_rng(D), 96_000 // D, 1, D)
    _agree(head, [x[:, 0], cs])


def _random_heads(target, heads: int) -> tuple[np.ndarray, np.ndarray]:
    nh, D = target.bindings["NHEADS"], target.bindings["D"]
    x, cs = _heads(np.random.default_rng(zlib.crc32(target.id.encode())), -(-heads // nh), nh, D)
    assert x.shape[0] * nh >= heads
    return x, cs


@pytest.mark.parametrize("target", [pytest.param(t, marks=pytest.mark.slow) if t.bindings['D'] == 256 else t for t in ROPES], ids=lambda t: t.id)
def test_every_served_binding_agrees_with_rope_v1(target) -> None:
    """1,024 random heads of each binding the served Programs call against `RoPE_v1`'s registered kernel (`twins.RoPETwin`,
    self-checked against the reference evaluator by the kernel registry): `RoPE_v2`'s batch over the heads and its shared cos/sin row
    at each served NHEADS."""
    _agree(target, list(_random_heads(target, 1024)))


@pytest.mark.boolean_sweep
@pytest.mark.slow
@pytest.mark.parametrize("target", LARGE, ids=lambda t: t.id)
def test_each_served_head_width_agrees_with_rope_v1_on_16k_heads(target) -> None:
    """16k random heads of one served binding per head width D against `RoPE_v1`'s kernel, 1M to 4M rotations: a rounding
    case of `RopeOut_v2` or `RopeOutAdd_v2` that served activations and cos/sin rows hit less often than once in 10^5, which
    the 90k-quadruple, 96k-word and 1,024-head tests miss."""
    _agree(target, list(_random_heads(target, 16_384)))


@pytest.mark.parametrize("target", [GPTJ_HEAD, *GPTJ], ids=lambda t: t.id)
def test_gptj_agrees_with_its_word_definition(target) -> None:
    """Against `RoPEHeadGptJ_v2` / `RoPEGptJ_v2` evaluated primitive by primitive (the reference evaluator): 96k words of the head
    and 1,024 heads of each multi-head binding, a quarter of them wide words whose products leave f32's range, where the
    odd half's operand order shows."""
    if target is GPTJ_HEAD:
        x, cs = _heads(np.random.default_rng(6400), 96_000 // 64, 1, 64)
        _agree(target, [x[:, 0], cs])
    else:
        _agree(target, list(_random_heads(target, 1024)))


#: (x[2i], x[2i+1], cos[i], sin[i], out[2i+1]) that vLLM d9105ea80's `rotary_embedding(is_neox=False)` wrote on sm_120
#: (rope-gptj cases 2 and 4) where `RoPEHeadGptJ_v1`'s odd half gives another word: in each,
#: x*sin or y*cos overflows f32, so the word depends on which product the FMA keeps unrounded
GPTJ_ODD_HW = [(0x7F7F, 0xFF7F, 0xBF1A, 0xBF94, 0xFF0D), (0x7F7F, 0x7F7F, 0xBFE3, 0x3F6E, 0xFF80),
               (0xFF7F, 0xFF7F, 0xBEA2, 0x3FA7, 0xFF7C), (0x7F7F, 0x7F7F, 0x3FF1, 0xBF65, 0x7F80),
               (0xFF7F, 0xFF7F, 0x3FF1, 0xBF65, 0xFF80), (0x7F7F, 0x7F7F, 0xBF9F, 0x3FC5, 0xFF80),
               (0x7F7F, 0xFF7F, 0xBF81, 0xC020, 0x7F80), (0xFF7F, 0xFF7F, 0xBF92, 0x3F04, 0x7F80),
               (0xFF7F, 0x7F7F, 0xBF81, 0xC020, 0xFF80), (0xFF7F, 0x7F7F, 0x3F03, 0x3FBA, 0xFF70)]


def test_the_gptj_odd_half_rounds_y_cos_as_the_kernel_does() -> None:
    w = np.array(GPTJ_ODD_HW, U16).T
    x, cs, want = np.stack([w[0], w[1]], axis=1), np.stack([w[2], w[3]], axis=1), w[4]
    head = bind(R.RoPEHeadGptJ, D=2)
    assert np.array_equal(BS.evaluate_words(head, x, cs).reshape(-1, 2)[:, 1].astype(U16), want)
    assert np.array_equal(np.asarray(E.evaluate_batch(TR.word_of(head), x, cs)).reshape(-1, 2)[:, 1].astype(U16), want)
    v1 = np.asarray(E.evaluate_batch(bind(ref_prims.RoPEHeadGptJ, D=2), x, cs)).reshape(-1, 2)[:, 1].astype(U16)
    assert (v1 != want).all()


@pytest.mark.parametrize("target", [*HEADS.values(), K, GPTJ_HEAD, GPTJ[1]], ids=lambda t: t.id)
def test_one_unit_per_output_element_as_in_the_word_program(target) -> None:
    """`Q_word` v1 cuts each into 16-bit units, one per output element: the word Program's units (`RopeOut_v1` and
    `RopeOutAdd_v1` gates)."""
    prog, _ = standalone_call(target)
    word, _ = standalone_call(TR.word_of(target))
    part, ref = PO.evaluate(prog), PO.evaluate(word)
    n = target.bindings["D"] * target.bindings.get("NHEADS", 1)
    assert part.population == ref.population == n
    (_k, cut), = part.calls
    assert cut.units == n
