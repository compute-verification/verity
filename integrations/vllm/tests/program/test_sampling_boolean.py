"""`verity_vllm.program.registry.sampling_boolean`: the Boolean greedy path agrees with its word Definitions (`trace.word_of`):
`ArgmaxStep_v2` with `ArgmaxStep_v1` on special and random carries, `TokenSelect_v2{V}` with `TokenSelect_v1{V}` on over 90,000
logit rows (ties, signed zeros, infinities, NaNs, a NaN first logit) and at the served vocabularies (SmolLM2's 49,152,
Llama-3.2's 128,256, Gemma-2's 256,000), the maximum at the first and the last index included; its stages chained as root Calls give the same token, each stage is separable and partitions under `Q_word_v1` with
no violation at either vocabulary, and the Definitions carry exactly their builders' live ANDs, pinned by gate count and
program digest. The f32 tree (`F32ArgmaxStage_v1`) gives the first maximum on over 80,000 f32 rows, a restatement tied to
`GumbelSelectF32_v1` at temp 0, and partitions likewise."""
from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.cut import separable
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import Circuit
from verity_vllm.program.kernels import rows
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import sampling_boolean as SB
from verity_vllm.query import word as W

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "ArgmaxStep_v2": (134, 290, 20, "0422abdc1d8d51c1d43422cd68a396305832a3beabbc81352c8f28f63893ebc9c5b12d06c8ab61332e72e32aed2c2c594217babc149e3540a6059a171379957d"),
    "ArgmaxLeaf_v1{G=64,FIRST=0,FINAL=0}": (3415, 8032, 1261, "bd47155e54f01c18fd094000c04cb90a58658e5787eec755d901f1b8b789fb8b1e832f09062cf062c2c5cd07b0a961424c77127652cf5640a31bd9111d6e5d53"),
    "ArgmaxLeaf_v1{G=64,FIRST=1,FINAL=0}": (3431, 8061, 1261, "a27b946b783337eb9d4f94d01863d55cde9e3cf5fd50e5921cf359de3e30d30ad07b56bf3ac6720cac9496d281d46e5172044022fba1b83f6b3762c7e00ce707"),
    "ArgmaxMerge_v1{G=64,IW=6,FINAL=0}": (2577, 6036, 1197, "5cdd52545245b4001db977e2bc768d230e7b5ad508ef0be411cd3217b9bafad12da8b3beee93020ca1e5694e2bb222a58d0363136e2e1467090bb3ae22a92fc1"),
    "TokenSelect_v2{V=50}": (2661, 6252, 981, "01242c512a75f2516c05b1e34532249ce4dee88a727ff61349e15416ee6792a41d1ea4401d0df4c9dbe349c4c6f4582d60208137f7029a343d9aa509bac6ebed"),
    "TokenSelect_v2{V=200}": (10759, 25301, 3981, "e8b49b86aeef42b1d39cde82c945692e0e49fbfbdb87f79ca79df0b45e97b36fb9f4384927277df9b2b5ab2748cf110c412a6f9a9b2a978332bf4bdce079a194"),
    "F32ArgmaxLeaf_v1{G=64,FIRST=0,FINAL=0}": (6455, 16144, 2269, "7149b47805ec23902d0084386b1f46cdd902770543f841e47574d189d995034aa5ea37e04763187cad968233d16fbab6cab91f7f8cbd4df871b9dcafbecb80e8"),
    "F32ArgmaxLeaf_v1{G=64,FIRST=1,FINAL=0}": (6487, 16205, 2269, "5d91cf8bbb8ac72e17104a66b42b9e7c4b4d1fe9e17ed1dcc7a180bef438e54d561c85560760d3eebf04019a130b8ee189040a3db9d5c3d24b9259e3794d6760"),
    "F32ArgmaxMerge_v1{G=64,IW=6,FINAL=0}": (4593, 11076, 2205, "3d968ff678a8abe625ab3dd3996bad854ee21fda125b1cf3db7a31bb87cd22d3e00daf72a48e377796ff4810d4c5e1ed797205f15050c621bf1a641106bcd8d6"),
}

#: the served vocabularies: TokenSelect_v2{V}'s And gates (54 per logit) and the tree's candidate counts per level
SERVED = {49152: (2654157, [49152, 768, 12, 1]), 128256: (6925774, [128256, 2004, 32, 1]),
          256000: (13823948, [256000, 4000, 63, 1])}

NEG_INF, POS_INF = 0xFF80, 0x7F80
#: signed zeros, subnormals, ones, the largest finite values, infinities, and NaNs (quiet, signalling, all-ones, negative)
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, POS_INF, NEG_INF, 0x7FC0, 0x7F81, 0xFFFF,
                     0xFFC1], np.uint16)


def first_max(x: np.ndarray) -> np.ndarray:
    """`TokenSelect_v1` on rows of bf16 words: the lowest index of the largest non-NaN value (+0 = -0), and 0 when the first
    logit is a NaN (the scan's best stays a NaN, and nothing compares greater than it)."""
    w = x.astype(np.int64)
    mag = w & 0x7FFF
    nan = mag > 0x7F80
    key = np.where(nan, -(1 << 20), np.where(w & 0x8000, -mag, mag))
    out = np.argmax(key, axis=1)
    out[nan[:, 0]] = 0
    return out


def logit_rows(V: int, n: int, rng) -> np.ndarray:
    """``n`` rows of ``V`` bf16 logits: in eight kinds, random words, words from a few values (ties at the maximum), specials,
    a NaN first logit, all NaN, all -inf, +0 and -0 tied at the top, and a planted maximum repeated at random positions."""
    x = rng.integers(0, 1 << 16, (n, V), dtype=np.uint64).astype(np.uint16)
    kind = np.arange(n) % 8
    few = np.array([0x3F80, 0x4000, 0x4040, 0xC000, 0x0000, 0x8000], np.uint16)
    x[kind == 1] = rng.choice(few, (int((kind == 1).sum()), V))
    x[kind == 2] = rng.choice(SPECIALS, (int((kind == 2).sum()), V))
    x[kind == 3, 0] = rng.choice(SPECIALS[10:], int((kind == 3).sum()))
    x[kind == 4] = rng.choice(SPECIALS[10:], (int((kind == 4).sum()), V))
    x[kind == 5] = NEG_INF
    z = kind == 6
    x[z] = rng.choice(np.array([0xBF80, NEG_INF, 0x7FC0], np.uint16), (int(z.sum()), V))
    x[z, rng.integers(0, V, int(z.sum()))] = 0x0000
    x[z, rng.integers(0, V, int(z.sum()))] = 0x8000
    p = np.flatnonzero(kind == 7)
    for _ in range(3):
        x[p, rng.integers(0, V, len(p))] = 0x7F00
    mixed = (rng.random((n, V)) < 0.05) & (kind[:, None] == 0)
    x[mixed] = rng.choice(SPECIALS, int(mixed.sum()))
    return x


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


# --- pins ------------------------------------------------------------------------------------------------------------------

def _target(spec: str):
    fam, _, body = spec.partition("{")
    d = {"ArgmaxStep_v2": SB.ArgmaxStep, "ArgmaxLeaf_v1": SB.ArgmaxLeaf, "ArgmaxMerge_v1": SB.ArgmaxMerge,
         "TokenSelect_v2": SB.TokenSelect, "F32ArgmaxLeaf_v1": SB.F32ArgmaxLeaf, "F32ArgmaxMerge_v1": SB.F32ArgmaxMerge}[fam]
    kw = dict(kv.split("=") for kv in body.rstrip("}").split(",")) if body else {}
    return bind(d, **{k: int(v) for k, v in kw.items()})


@pytest.mark.parametrize("spec", sorted(PINS))
def test_pinned_counts_and_digests(spec) -> None:
    target = _target(spec)
    assert target.id == spec
    ands, xors, nots, digest = PINS[spec]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


@pytest.mark.parametrize("spec", [s for s in sorted(PINS) if not s.startswith("TokenSelect")])
def test_the_ands_are_the_builders_live_ands(spec) -> None:
    target = _target(spec)
    d = target.defn
    params, ret = target.signature()
    types = [t for _, t in params]
    C = Circuit(cse=True)
    ins = TR._elements_in(C, types)
    if d.statics:
        T = d.traced(target)
        C.output("out", TR._flat_out(d.build(C, target.S(), *ins), ret))
    else:
        T = d.traced()
        C.output("out", TR._flat_out(d.build(C, *ins), ret))
    assert T.counts["and"] == C.counts()["ands"] == PINS[spec][0]


# --- ArgmaxStep_v2 ---------------------------------------------------------------------------------------------------------

def test_argmax_step_agrees_with_its_word_definition() -> None:
    """Every pair of special (best, x), and random words, with best indices and counters at both ends of the i32 range (the
    counter wraps)."""
    target = bind(SB.ArgmaxStep)
    word = TR.word_of(target)
    assert word.id == "ArgmaxStep_v1"
    rng = _rng("argmax-step")
    g = np.array(np.meshgrid(np.arange(len(SPECIALS)), np.arange(len(SPECIALS)))).reshape(2, -1)
    n = 20_000
    best = np.concatenate([SPECIALS[g[0]], rng.integers(0, 1 << 16, n, dtype=np.uint64).astype(np.uint16)])
    x = np.concatenate([SPECIALS[g[1]], rng.integers(0, 1 << 16, n, dtype=np.uint64).astype(np.uint16)])
    tie = rng.random(len(x)) < 0.1
    x[tie] = best[tie]
    ends = np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF], np.uint64)
    bi, i = (np.where(rng.random(len(x)) < 0.2, rng.choice(ends, len(x)), rng.integers(0, 1 << 32, len(x), dtype=np.uint64))
             for _ in range(2))
    carry = np.stack([best.astype(np.uint64), bi, i], 1)
    got = BS.evaluate_words(target, carry, x).astype(np.uint64)
    ref = E.evaluate_batch(word, carry, x).astype(np.uint64)
    bad = np.flatnonzero((got != ref).any(axis=1))
    assert not bad.size, f"{bad.size} of {len(x)} steps differ from {word.id}, first {bad[:5]}"


# --- TokenSelect_v2 --------------------------------------------------------------------------------------------------------

def test_levels() -> None:
    assert SB.levels(1) == [1, 1] and SB.levels(2) == [2, 1] and SB.levels(64) == [64, 1] and SB.levels(65) == [65, 2, 1]
    assert SB.levels(4096) == [4096, 64, 1] and SB.levels(4097) == [4097, 65, 2, 1]
    for V, (_, n) in SERVED.items():
        assert SB.levels(V) == n


@pytest.mark.parametrize("V,n", [(2, 2048), (3, 2048), (5, 2048), (17, 1024), (65, 512)])
def test_first_max_is_the_word_definition(V, n) -> None:
    """The numpy restatement the bulk tests compare with, against `TokenSelect_v1{V}` itself."""
    x = logit_rows(V, n, _rng("oracle", V))
    ref = E.evaluate_batch(bind(b1.TokenSelect, V=V), x).astype(np.int64).reshape(-1)
    bad = np.flatnonzero(first_max(x) != ref)
    assert not bad.size, f"V={V}: {bad.size} of {n} rows differ from TokenSelect_v1, first {bad[:5]}"


#: (V, rows): one, two and three levels of the tree, full groups and remainders; 92,160 rows in all
BULK = [(2, 16384), (3, 16384), (7, 16384), (63, 8192), (64, 8192), (65, 8192), (127, 4096), (128, 4096), (129, 4096),
        (200, 4096), (4097, 1024), (4161, 1024)]


@pytest.mark.parametrize("V,n", BULK)
def test_token_select_agrees_with_its_word_definition(V, n) -> None:
    target = bind(SB.TokenSelect, V=V)
    assert TR.word_of(target).id == f"TokenSelect_v1{{V={V}}}"
    x = logit_rows(V, n, _rng("bulk", V))
    got = BS.evaluate_words(target, x).astype(np.int64).reshape(-1)
    want = first_max(x)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"V={V}: {bad.size} of {n} rows differ, first {[(int(i), int(got[i]), int(want[i])) for i in bad[:5]]}"
    if V <= 65:
        ref = E.evaluate_batch(TR.word_of(target), x[:256]).astype(np.int64).reshape(-1)
        assert got[:256].tolist() == ref.tolist()


@pytest.mark.parametrize("V", [200, 4161])
def test_stages_as_root_calls(V) -> None:
    """The stages evaluated one after another, as a Program's root Calls run them (`token_select_calls`), give the token."""
    x = logit_rows(V, 512, _rng("stages", V))
    sts = SB.stages(V)
    emitted = []
    SB.token_select_calls(lambda fn, arg: emitted.append(fn) or arg, None, V)
    assert [s.id for s in emitted] == [s.id for s in sts] == [f"TokenSelectStage_v1{{V={V},L={L}}}" for L in range(len(sts))]
    bits = ((x.astype(np.uint32)[:, :, None] >> np.arange(16, dtype=np.uint32)) & 1).astype(np.uint8)
    for st in sts:
        bits = BS.evaluate_bits(st, bits.reshape(len(x), -1)).reshape(len(x), -1)
    token = (bits.astype(np.int64) << np.arange(32)).sum(axis=1)
    assert token.tolist() == first_max(x).tolist()


def test_one_logit_is_index_0() -> None:
    x = np.concatenate([SPECIALS, np.arange(0, 1 << 16, 97, dtype=np.uint16)])[:, None]
    assert not BS.evaluate_words(bind(SB.TokenSelect, V=1), x).any()


@pytest.mark.xfail(strict=True, reason="TokenSelect_v1{V=1} returns logits[0], not 0: its scan has no iterations, and "
                   "verity.ml.circuits.layout.resolve_gate reads a zero-length scan's carry from iteration n - 1 = -1 instead of from init")
def test_word_view_at_one_logit() -> None:
    x = SPECIALS[:, None]
    assert not E.evaluate_batch(bind(b1.TokenSelect, V=1), x).any()


@pytest.mark.xfail(strict=True, reason="rows.token_select_first_max maps NaN to -inf before argmax, so with a NaN first logit "
                   "it returns the first maximum of the rest, where TokenSelect_v1 (a scan whose best stays the NaN) returns 0")
def test_rows_oracle_with_a_nan_first_logit() -> None:
    x = np.array([[0x7FC0, 0x3F80, 0x4000]], np.uint16)
    assert rows.token_select_first_max(x[0]) == int(first_max(x)[0]) == 0


# --- the f32 tree ----------------------------------------------------------------------------------------------------------

def f32_first_max(y: np.ndarray) -> np.ndarray:
    """`first_max` on rows of f32 words (`F32GtStrict_v1`'s order: subnormals as they are, +0 = -0, a NaN never taken)."""
    w = y.astype(np.int64)
    mag = w & 0x7FFFFFFF
    nan = mag > 0x7F800000
    key = np.where(nan, -(1 << 40), np.where(w & 0x80000000, -mag, mag))
    out = np.argmax(key, axis=1)
    out[nan[:, 0]] = 0
    return out


def f32_rows(V: int, n: int, rng) -> np.ndarray:
    """`logit_rows` as f32 words (each bf16 word the f32 word's high half), random low halves in the random rows."""
    y = logit_rows(V, n, rng).astype(np.uint32) << np.uint32(16)
    y[0::8] |= rng.integers(0, 1 << 16, (len(y[0::8]), V), dtype=np.uint64).astype(np.uint32)
    return y


def f32_argmax(V: int, y: np.ndarray) -> np.ndarray:
    """`F32ArgmaxStage_v1`'s levels evaluated one after another, as root Calls (`argmax_calls`) run them."""
    bits = ((y[:, :, None] >> np.arange(32, dtype=np.uint32)) & 1).astype(np.uint8)
    for st in SB.stages(V, SB.F32ArgmaxStage):
        bits = BS.evaluate_bits(st, bits.reshape(len(y), -1)).reshape(len(y), -1)
    return (bits.astype(np.int64) << np.arange(32)).sum(axis=1)


@pytest.mark.parametrize("V", [2, 5, 65])
def test_f32_first_max_is_the_word_definition(V) -> None:
    """The f32 restatement against `GumbelSelectF32_v1{V}` at temp 0 (+0 and -0), where it is the first maximum of the logits."""
    y = f32_rows(V, 1024, _rng("f32-oracle", V))
    zero = np.zeros(len(y), np.uint64)
    temp = np.where(np.arange(len(y)) % 2, 0x80000000, 0).astype(np.uint32)
    ref = E.evaluate_batch(bind(SM.GumbelSelectF32, V=V), y, zero, zero, temp).astype(np.int64).reshape(-1)
    bad = np.flatnonzero(f32_first_max(y) != ref)
    assert not bad.size, f"V={V}: {bad.size} rows differ from GumbelSelectF32_v1 at temp 0, first {bad[:5]}"


#: (V, rows): one, two and three levels; 86,016 rows in all
F32_BULK = [(1, 4096), (2, 16384), (7, 16384), (64, 8192), (65, 8192), (129, 8192), (200, 4096), (4096, 512), (4161, 1024)]


@pytest.mark.parametrize("V,n", F32_BULK)
def test_f32_argmax_tree_is_the_first_maximum(V, n) -> None:
    y = f32_rows(V, n, _rng("f32-bulk", V))
    got, want = f32_argmax(V, y), f32_first_max(y)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"V={V}: {bad.size} of {n} rows differ, first {[(int(i), int(got[i]), int(want[i])) for i in bad[:5]]}"


# --- the partition ---------------------------------------------------------------------------------------------------------

def _partitions(V: int, stage=SB.TokenSelectStage) -> None:
    for st in SB.stages(V, stage):
        members = separable(st)
        assert members is not None, f"{st.id} is not separable"
        assert all(m.gates <= W.MAX_GATES for m, _ in members)
        rule = W.unit_rule(st, max_gates=W.MAX_GATES)
        assert rule["violations"] == [], f"{st.id}: {rule['violations'][:3]}"


@pytest.mark.parametrize("V", [65, 200, 4161])
@pytest.mark.parametrize("stage", [SB.TokenSelectStage, SB.F32ArgmaxStage], ids=lambda d: d.name)
def test_stages_partition(V, stage) -> None:
    _partitions(V, stage)


@pytest.mark.slow
@pytest.mark.parametrize("V", sorted(SERVED))
def test_served_vocabulary(V) -> None:
    """The served sizes (V = 256,000: about 25 s): the gate count and levels, each stage's partition, and 64 rows against
    `first_max`, which `test_first_max_is_the_word_definition` ties to the word Definition (its scan takes minutes on these
    rows); rows 16-35 put the maximum at the last index or tie it with the first, each with its token stated."""
    target = bind(SB.TokenSelect, V=V)
    assert BS.compile(target).counts["and"] == SERVED[V][0]
    assert SB.levels(V) == SERVED[V][1]
    _partitions(V)
    x = logit_rows(V, 64, _rng("served", V))
    x[:8] = 0x7FC0
    x[8:16, 0] = 0x7FC1
    x[16:36] = NEG_INF
    x[16:20, V - 1] = 0x3F80             # the maximum alone at the last index
    x[20:24, [0, V - 1]] = 0x4000        # tied at the first and the last index
    x[24:28, 1:] = 0x7FC0                # NaNs never win: -0 at the last index beats -inf at the first
    x[24:28, V - 1] = 0x8000
    x[28:32, 0], x[28:32, V - 1] = 0x8000, 0x0000   # -0 and +0 tie
    x[32:36, V - 1] = POS_INF            # +inf at the last index
    x[32:36, V - 2] = 0x7F7F
    got = BS.evaluate_words(target, x).astype(np.int64).reshape(-1)
    assert got[16:36].tolist() == [V - 1] * 4 + [0] * 4 + [V - 1] * 4 + [0] * 4 + [V - 1] * 4
    assert got.tolist() == first_max(x).tolist()
