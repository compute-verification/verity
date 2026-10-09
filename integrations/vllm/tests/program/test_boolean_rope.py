"""`registry.boolean_rope`: the Boolean NeoX rotation agrees with its word Definitions (`trace.word_of`) on every quadruple of
special bf16 words and on 90k random and structured quadruples per half, on random heads of each served head width against
`RoPEHead_v1` through the IR reference, on 16k random heads of each served head width against `RoPE_v1`'s kernel, and on 1,024
random heads of every `RoPE_v1` binding the served workload Programs call (`SERVED`); carries exactly its builders' live ANDs;
is pinned by gate count and program digest; and cuts under `Q_word` v1 into one 16-bit unit per output element, as the word
Program does. The interleaved (GPT-J) versions agree with `RoPEHeadGptJ_v2` / `RoPEGptJ_v2` through the IR reference at
DeepSeek-V2 MLA's q_pe and k_pe shapes and reproduce the words vLLM's `rotary_embedding(is_neox=False)` wrote on sm_120 where
`RoPEHeadGptJ_v1` does not."""
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
from verity.ml.circuits.boolean.forms import Circuit
from verity_vllm.program.kernels import twins  # noqa: F401  (registers RoPE_v1's kernel, which the served shapes' word side runs)
from verity_vllm.program.kernels.twins import rand_bf16, tie_pairs_mul
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import boolean_rope as R
from verity_vllm.program.registry import ref_prims

U16 = np.uint16
HALVES = [R.RopeOut, R.RopeOutAdd]

#: (NHEADS, D) of the `RoPE_v1` Calls in each model's traced workload Programs, q then k (one binding where the model has as
#: many kv heads as heads): every Build descriptor of the coverage rows, art:5c594ebc74d2374816ced39e2b2bc114422a42bfc06ccf09a0090b523a8bc50f
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
HEAD = HEADS[64]
K = bind(R.RoPE, NHEADS=3, D=64)
#: one served binding per head width, run on 16k heads: SmolLM2's q and k, Phi-3's, Qwen3's q, Gemma-2's q
LARGE = [bind(R.RoPE, NHEADS=n, D=d) for n, d in ((9, 64), (3, 64), (32, 96), (32, 128), (8, 256))]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "RopeOut_v2": (2966, 4608, 295, "d04513f5effb40ddb44a20c17342bac87dd8468fa2a4f4f42551a9026d9d9229b7c0ee2d90cbde79b122f60635499d5a5f5ec0534a687c95e9fb8f47fbaa614b"),
    "RopeOutAdd_v2": (2966, 4608, 294, "d53f3f2e6b4dcca7d8bafca17d8a9c02d0d39073e1e13c6e493da779b1a8f62c5b28f37f8332a0d8e4f551e0e7dfa2f64cfeabb6dc9ef51a7f3748257b1f12b6"),
    "RoPEHead_v2{D=64}": (189824, 294912, 18848, "c21dd89b6ed10938299967990ef89f901fa72e2050b72e687b13cd5b0833eba5d9301fcbf07dd09ae31a7d054d151d50f1fc2b001fad4d40457e1a29b71dff75"),
    "RoPEHead_v2{D=96}": (284736, 442368, 28272, "afc7de9dd43b93339c7a54c68a1f2c81be0b102920e0ee2dc93b7245a96b979e8e2c80b96c0d4e0e23a0f2fe83cc342196520aaba0841985c8a4a2d80676a4ec"),
    "RoPEHead_v2{D=128}": (379648, 589824, 37696, "d401e772969b8a1535ef3963e3526eaa5dc89efbc5fa3673dd7046b5ccc14ccf518d1c7d73ebeb5481905ec335264bc7bddeb8ca3515193c576d327c4013343b"),
    "RoPEHead_v2{D=256}": (759296, 1179648, 75392, "030d4fee1a3c7ebcee43fd39d5a453a9256e311cd01ce63b369a6436f38ca6a510190536115d245a3db5cbb1ccdaa25bb4071517463f69af59c84dcb07e7c4fb"),
    "RoPE_v2{NHEADS=9,D=64}": (1708416, 2654208, 169632, "099fb72d3bc9bb82ff98236fa04b79c1ad390018e0a51e20b9f99806bd2b5bd02ef2d031e8aa7c9e70b94bce16d9ec109cb4138388e13c5d12c5cfe167c231ca"),
    "RoPE_v2{NHEADS=3,D=64}": (569472, 884736, 56544, "dbf2e021fdd58cfa21f6ef3b927fe3b77c5a6576f31e28f1c8595a27090afd217b8fd722741aa028373b1b723d7701daabe5c38c1947a177712403e2d0e58e3e"),
    "RoPE_v2{NHEADS=32,D=64}": (6074368, 9437184, 603136, "4e29903fa63bf822c2370bfa64345fce31a53a7616b5ace53195a03f83dc21585b0f69be739d622f71d31a1b82ca0646e1d18bee02c0b38b72c10b9b2c1f9789"),
    "RoPE_v2{NHEADS=8,D=64}": (1518592, 2359296, 150784, "fd04901d7ef230c5bd96d6b044c4e485660e576cbbf7e9091cd9c263a48653d21fe385f4bf6deea6d7ac67d04540b0cdb5e212da2866dc735a2ccb1b8dc07d5b"),
    "RoPE_v2{NHEADS=14,D=64}": (2657536, 4128768, 263872, "3096be038bb49f07afe3d04458ac8f355f7f49349c3c2236352111b78506a5cee64db722a8fe4ef56b5df7cf088170fd3e555cd72f9b07b92e327d9b6f6ae521"),
    "RoPE_v2{NHEADS=2,D=64}": (379648, 589824, 37696, "97e91c5866c8b108a396e71042f0725422d44b1cf1b073af9abf7bf5abe73c756fdb933b268c3bab645d8f9ab8318b86bf4b2162192b0725506c13d42a377454"),
    "RoPE_v2{NHEADS=32,D=96}": (9111552, 14155776, 904704, "b21c95a8fdb1f8cda78f1fdf8e76923de2a38fe23e1761a2fef06af8bf406c71be43844a242a72907e08d485c788bd57a3e7cba59f5b253c3299f7d74fbbe20f"),
    "RoPE_v2{NHEADS=16,D=128}": (6074368, 9437184, 603136, "2ef30c5e9a813d6afe66a1ba3048730f85a8f7da1ec1317f0df54dcc25234252f8cde85ee9cdb2a52f9a281dffcf2da5584d0237078ef8600fbf6c9d5907ef19"),
    "RoPE_v2{NHEADS=12,D=128}": (4555776, 7077888, 452352, "908e49643d3c201427559276dc468e16ce2c7b0d68f6f99c4797d3f528850e34f06f29e3ba411bd2f090f3b5743b07e12f89c3485f1668fdfa0c61df343fa6ba"),
    "RoPE_v2{NHEADS=2,D=128}": (759296, 1179648, 75392, "d807660f948a3e3e302949ce960d15d2e6d0454abd8edcdeadc470fa10240d7a0627f3a19b796709e754dc6d4f184e4682ed97122576b99af43db32409346c47"),
    "RoPE_v2{NHEADS=32,D=128}": (12148736, 18874368, 1206272, "b5a7d97b91566f8a972f743b53d0112088a56f33835429f7e92eef43cda7bf6969d1d1f1b5af6ab6b2a1dd1fef935bf99b4f5403ffaa40f6677774c87181bc90"),
    "RoPE_v2{NHEADS=8,D=128}": (3037184, 4718592, 301568, "25323b32f3cff1d5c7fcac86b368eb8351aacd038c4f32966c99efa137c2be6ad5c13adb622a951b5c4ba6ad98e46b4de704ec42589a91dd5c06e2cc25ffb78b"),
    "RoPE_v2{NHEADS=8,D=256}": (6074368, 9437184, 603136, "523ec5010b65022134a689b8a83595a9036e3a95ac4c1f97b140c76ffede2bfb202f1958674e639cdc99c6cabea4f0dd01d8738fa81d6f498f9920a37130fc74"),
    "RoPE_v2{NHEADS=4,D=256}": (3037184, 4718592, 301568, "bae7e4991c4cb2d0fb47cbd5701d5d4162d3c5eea42fa865c261c9679d0ae2c35675b28b0518073e723cb81298671b3964278ab6bfa017eb56610d422d199c8a"),
    "RoPEHeadGptJ_v3{D=64}": (189824, 294912, 18848, "2024a0b64d1b8a1773764ddb667b59d37254d0e5dd803488031f304ce5acddff1cb6d2f990fb5674d74a37d5c12c44a5fb895eb80394902a6d6c2ae324e76b3b"),
    "RoPEGptJ_v3{NHEADS=1,D=64}": (189824, 294912, 18848, "402b602e9cc2539978313a55362a8c949ff3097112b368b7d4e78749da6c1c7537f975b8e0df8145b1b516b1a277222618d8173524386cc9f2ac2c3095463413"),
    "RoPEGptJ_v3{NHEADS=5,D=64}": (949120, 1474560, 94240, "2ce703d681be79d90c398be8b7bd92c26085cb57d807690cce51b77ff11da81080cbee601c92a68578938526a7de54dfa833e759754d610bc7403a36cba96146"),
    "RoPEGptJ_v3{NHEADS=8,D=64}": (1518592, 2359296, 150784, "a30cf280f468a859bdd566cca018b124d9e6fc2d5bcdc58f335056942d2b5c0010558fbf8d9b71734009ed2d0f81045b2dd0fd3d52aa0f938019a3a2644105cd"),
    "RoPEGptJ_v3{NHEADS=20,D=64}": (3796480, 5898240, 376960, "a1c712cf4eb8f930183ad9d7972f78502970ff32537564901250a852924eaa725512bc7b265634b58704c3485811e07b25ea4e809cf2ba9bca413ddebb0c47a2"),
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
    bad = []
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
    """Against `RoPEHead_v1{D}` evaluated primitive by primitive (the IR reference) on 96k words, so the halves' wiring is
    checked without any kernel."""
    D = head.bindings["D"]
    x, cs = _heads(np.random.default_rng(D), 96_000 // D, 1, D)
    _agree(head, [x[:, 0], cs])


def _random_heads(target, heads: int) -> tuple[np.ndarray, np.ndarray]:
    nh, D = target.bindings["NHEADS"], target.bindings["D"]
    x, cs = _heads(np.random.default_rng(zlib.crc32(target.id.encode())), -(-heads // nh), nh, D)
    assert x.shape[0] * nh >= heads
    return x, cs


@pytest.mark.parametrize("target", ROPES, ids=lambda t: t.id)
def test_every_served_binding_agrees_with_rope_v1(target) -> None:
    """1,024 random heads of each binding the served Programs call against `RoPE_v1`'s registered kernel (`twins.RoPETwin`,
    self-checked against the IR reference by the kernel registry): `RoPE_v2`'s batch over the heads and its shared cos/sin row
    at each served NHEADS."""
    _agree(target, list(_random_heads(target, 1024)))


@pytest.mark.slow
@pytest.mark.parametrize("target", LARGE, ids=lambda t: t.id)
def test_each_served_head_width_agrees_with_rope_v1_on_16k_heads(target) -> None:
    """16k random heads of one served binding per head width D against `RoPE_v1`'s kernel, 1M to 4M rotations: a rounding
    case of `RopeOut_v2` or `RopeOutAdd_v2` that served activations and cos/sin rows hit less often than once in 10^5, which
    the 90k-quadruple, 96k-word and 1,024-head tests miss."""
    _agree(target, list(_random_heads(target, 16_384)))


@pytest.mark.parametrize("target", [GPTJ_HEAD, *GPTJ], ids=lambda t: t.id)
def test_gptj_agrees_with_its_word_definition(target) -> None:
    """Against `RoPEHeadGptJ_v2` / `RoPEGptJ_v2` evaluated primitive by primitive (the IR reference): 96k words of the head
    and 1,024 heads of each multi-head binding, a quarter of them wide words whose products leave f32's range, where the
    odd half's operand order shows."""
    if target is GPTJ_HEAD:
        x, cs = _heads(np.random.default_rng(6400), 96_000 // 64, 1, 64)
        _agree(target, [x[:, 0], cs])
    else:
        _agree(target, list(_random_heads(target, 1024)))


#: (x[2i], x[2i+1], cos[i], sin[i], out[2i+1]) that vLLM d9105ea80's `rotary_embedding(is_neox=False)` wrote on sm_120
#: (run r20261004-060203-ba52, rope-gptj cases 2 and 4) where `RoPEHeadGptJ_v1`'s odd half gives another word: in each,
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
