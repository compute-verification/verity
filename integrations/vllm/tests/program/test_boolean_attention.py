"""`boolean_attention`: the v6 chain (`AttnBlock_v6` / `AttentionHead_v6` / `Attention_v6`), the v7 chain on both of its
profiles (FA2 on mma.sync, FA3 on wgmma), the v8 chain at the served FA2 binding, the v9 chain at FA3's (and v16's block and head,
v9's with the scale an input) and `Fa3InvSum_v2` are Boolean, agree with their word views (the v3, v2, v5, v4 and v13 chains and
`Fa3InvSum_v1`) on random words and on words that reach
every guard (a zero query, signed zeros, subnormals, infinities, NaNs, scores that overflow), and are pinned by gate count and
program digest."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.batch import type_shape
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, bind
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions import boolean as SW
from verity.ml.definitions.registry import boolean_attention as BAt

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "AttnBlock_v6{D=16,NVIS=5,FIRST=True,BN=16}":
        (117848, 240136, 12689, "a0d85fc56f1e8959ca41f131e5a29e1a2b0bbcc1cb4c0ef0fb9280bfbff2a0c60f400efa33900335c8f0e27bbf2120ccd2dc261af727337caac31397e769677b"),
    "AttnBlock_v6{D=16,NVIS=16,FIRST=False,BN=16}":
        (385610, 775142, 35861, "c7c87d04180d9bf24c4c1e64448d54a4af94149ef419fd7fe523f19ac50d067ba49b45f564ae067bc68f70ea02b042758d277a63745d14dd0b290f90d15fc379"),
    "AttentionHead_v6{T=20,D=16,BN=16}":
        (529826, 1064130, 49293, "d42c58a086758e71631ffbe307b3c9cf4b7c8aa3f793aa5f5ee225657bf515265954bd056dc73aac26ebee19e65eb944176482bdb8fa664120b9abb36a42d400"),
    "Attention_v6{T=20,NH=4,KVH=2,D=16,BN=16}":
        (2119304, 4256520, 197172, "9d75a0324bdf64a06ec68770b89df675cc8f8175ab3c94c76571615c7f973ec30cbe003a807050ed692fdc059e4007c27e6a2ae51a525dac0ea8f5abd0d833f9"),
    "Fa3InvSum_v2": (947, 4866, 102, "017c77eb48a6d4ed9be9ddaa291fa4c3ae280333540ed06045fef47238963ed59268c32fb069f62910b83e099df8eded6bba977ae659d6b42280ca8d8d653c5d"),
    "AttnBlock_v7{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2}":
        (373082, 753894, 32661, "92eaf649b73843cfd384502cc591f2986b3f174fe3895473ed4f905859ac9644afe7ae72c740168bc78345034ad71c2a0fe80c621b7cc3e1ea773262f9c8313b"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=AmpereBF16TcDot16_v3,INV=Fa2InvSum_v2}":
        (529826, 1064130, 49293, "bf5ffb2c7321aa35757d66457de1d24c8c0ac40bf863b1fd1c1285cbb7222fe2d84ba63b1895460efcd5c09b31825ad62fd1043b8247f9a547647540c3a2f9b8"),
    "AttentionHead_v7{T=20,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (507222, 1022162, 43986, "8634c3ef17b98ad2e7a88d3968683d371a229ef1bb8ab3e5646649ec3ad19ae0281229cd49fe024fd18bce7a1a7439717f9d00ded9dfe40a71a0e6ffbdd78465"),
    "Attention_v7{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2}":
        (2028888, 4088648, 175944, "95a408a28d9eb2e7cb17a45d844313019692bbd9566576333ba8aaf7f38c0e279c1a01536d07788b7c90fbc4ab7745c550415d80e0d1758bd4f82c9aa136fd11"),
    "AttnBlock_v8{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (373082, 753894, 32661, "599a02a3cca5d8e6a1f3c71408e0e8f91af39749cf71f356c775ce7a41e774a7421675d6585a3f48521987ceb91284a1c405642a65e70bc84435ad8c4d4e055a"),
    "AttentionHead_v8{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=1}":
        (952372, 1927030, 83204, "eea0421f654d0e763aa401f04de68fe6471a66f033013a0e068e7ce864b9c2b9e9791dd3de84db0457a842b9c87293878a9d894d96f4b6bf1f77b8171f5ea340"),
    "Attention_v8{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=0}":
        (2028888, 4088648, 175972, "ed2d254617a94c291126d8ff89f85b278f2a2522f361d1869675faf28e7be784935d7fd056db5922a68a23709c8f8e1a9a4e07ae3445c6595a895e8c429553d5"),
    "AttnBlock_v9{D=16,NVIS=16,FIRST=False,BN=16,DOT=HopperBF16WgmmaDot16_v2,CHECK=False}":
        (373019, 753894, 32637, "fad98d501500b7c99c1b55730b598afcb2d4f843c13e3676e177a60d9ba45c8938d53127f13a2d80b2a46e647df7b70428e69ea2c76342478bd140d04658986c"),
    "AttentionHead_v9{T=40,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=2}":
        (952246, 1927030, 83149, "224920413449676c73cc298553a7eb80f05f61a0c211266f5f3bb2800e1288b2ff1ef0fe0fc99dd093fb2cf8d5f20194a51eee8a90026d8e7b38a0f7d0d0d356"),
    "Attention_v9{T=20,NH=4,KVH=2,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa3InvSum_v2,MASKED_FROM=1}":
        (2028636, 4088648, 175848, "21fea7aa3389ddea3ca25b4e5bdd85dc08531f54a4148676102fb92f5c4c101eaef6606639dc50287bdecb43d9bbf228d66f1dc507f1ae836757e664131155c6"),
}

FA2, FA3 = BAt.ATTENTION_FA2, BAt.ATTENTION_FA3
#: the served FA2 row's `Attention_v5` binding (`kernels.rows`): the wgmma step and FA2's inverse sum
FA2_CHECK = {"DOT": FA3["DOT"], "INV": FA2["INV"]}
PINNED = [bind(BAt.AttnBlockV6, D=16, NVIS=5, FIRST=True, BN=16), bind(BAt.AttnBlockV6, D=16, NVIS=16, FIRST=False, BN=16),
          bind(BAt.AttentionHeadV6, T=20, D=16, BN=16), bind(BAt.AttentionV6, T=20, NH=4, KVH=2, D=16, BN=16), bind(BAt.Fa3InvSum),
          bind(BAt.AttnBlockV7, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"]), bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA2),
          bind(BAt.AttentionHeadV7, T=20, D=16, BN=16, **FA3), bind(BAt.AttentionV7, T=20, NH=4, KVH=2, D=16, BN=16, **FA3),
          bind(BAt.AttnBlockV8, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=False),
          bind(BAt.AttentionHeadV8, T=40, D=16, BN=16, MASKED_FROM=1, **FA2_CHECK),
          bind(BAt.AttentionV8, T=20, NH=4, KVH=2, D=16, BN=16, MASKED_FROM=0, **FA2_CHECK),
          bind(BAt.AttnBlockV9, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=False),
          bind(BAt.AttentionHeadV9, T=40, D=16, BN=16, MASKED_FROM=2, **FA3),
          bind(BAt.AttentionV9, T=20, NH=4, KVH=2, D=16, BN=16, MASKED_FROM=1, **FA3)]
SLOW_PIN = PINNED[5].id

#: one block per mask shape and per state; one and two blocks, a partial last block, BN = 32; GQA with two KV heads; v7 on both
#: profiles; v8 and v9 with an unguarded block, a head with an unguarded block, and GQA; v16 (v9 with the scale an input) with a
#: first block of fewer keys than lanes, an unguarded block and a head
AGREEMENT = PINNED[:2] + [bind(BAt.AttnBlockV6, D=16, NVIS=9, FIRST=False, BN=32), bind(BAt.AttentionHeadV6, T=5, D=16, BN=16),
                          bind(BAt.AttentionHeadV6, T=16, D=16, BN=16), bind(BAt.AttentionHeadV6, T=40, D=16, BN=32),
                          bind(BAt.AttentionV6, T=20, NH=4, KVH=2, D=16, BN=16)] + PINNED[5:] + [
    bind(BAt.AttentionHeadV7, T=40, D=16, BN=32, **FA3),
    bind(BAt.AttnBlockV16, D=16, NVIS=3, FIRST=True, BN=16, DOT=FA3["DOT"], CHECK=True),
    bind(BAt.AttnBlockV16, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=False),
    bind(BAt.AttentionHeadV16, T=20, D=16, BN=16, MASKED_FROM=1, **FA3)]

#: bf16 words that reach FA2's guards: signed zeros, subnormals, the normal range's ends, infinities, NaNs, 2^63
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x5F00, 0xDF00],
                    np.uint16)


def _words(rng, shape, scale: float, width: int) -> np.ndarray:
    x = (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)
    return x if width == 32 else (x >> 16).astype(np.uint16)


def _inputs(word, n: int, seed: int) -> list[np.ndarray]:
    """``n`` instances per parameter of ``word``: the first third gaussian at softmax's working scales, the second with special
    words planted at random positions, the last with a few entries near 2^63 (scores that overflow to infinity); the first
    instance's query is zero. The f32 state of a block (m, l, o) is gaussian, l's sums positive."""
    rng = np.random.default_rng(seed)
    third = n // 3
    out = []
    for j, (name, t) in enumerate(word.signature()[0]):
        shape, width = (n, *type_shape(t)), max(t.width_set())
        if width == 32:
            a = _words(rng, shape, 4.0, 32)
            if name == "l_prev":
                a &= np.uint32(0x7FFFFFFF)
            out.append(a)
            continue
        a = _words(rng, shape, 2.0 if j == 0 else 1.0, 16)
        mid, end = a[third:2 * third], a[2 * third:]
        m = rng.random(mid.shape) < 0.05
        mid[m] = rng.choice(SPECIALS, int(m.sum()))
        m = rng.random(end.shape) < 0.02
        end[m] = (rng.integers(0x5E00, 0x5F80, int(m.sum())) | (rng.integers(0, 2, int(m.sum())) << 15)).astype(np.uint16)
        if j == 0:
            a[0] = 0
        out.append(a)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("target", [pytest.param(t, marks=pytest.mark.slow) if t.id == SLOW_PIN else t for t in PINNED], ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    assert codec.program_digest(codec.encode_program(prog)) == digest


def _program(*fns) -> Program:
    """One Program calling each of ``fns`` once, on parameters of its own."""
    sigs = [fn.signature() for fn in fns]
    params = tuple((f"a{i}_{n}", t) for i, (ps, _r) in enumerate(sigs) for n, t in ps)

    def body(B, S, *args):
        outs, k = [], 0
        for fn, (ps, _r) in zip(fns, sigs):
            outs.append(B.call(fn, *args[k:k + len(ps)]))
            k += len(ps)
        return tuple_of(*outs)

    return Program(bind(CompositeDefinition("Probe", 1, (), lambda S: (params, Tuple(*[r for _p, r in sigs])), body, register=False)))


@pytest.mark.slow
def test_zero_padding_is_the_zero_bit_broadcast(monkeypatch) -> None:
    """A partial block's padded keys and values, and the first block's zero state, are the zero bit broadcast: a head's description
    does not grow with its padding, as a word head's does not.  It is the circuit an explicit zero word per padded word writes."""
    from verity_vllm.program.descriptor_equivalence import compare
    heads = {T: bind(BAt.AttentionHeadV8, T=T, D=16, BN=16, MASKED_FROM=0, **FA2_CHECK) for T in (17, 31)}
    size = {T: len(codec.canonical_json(codec.definition_entry(h))) for T, h in heads.items()}
    assert abs(size[17] - size[31]) < 16, size
    new = codec.encode_program(_program(heads[17]))                 # every body built before the patch below

    def explicit(self, t):
        w = t
        while isinstance(w.elem, Array):
            w = w.elem
        return Coll(t, concat([self.word(0, w.n).refs] * (t.leaves // w.n)))

    monkeypatch.setattr(BAt._Bits, "zeros", explicit)
    for d in (BAt.AttentionHeadV8, BAt.AttnBlockV8):
        monkeypatch.setattr(d, "_cache", {})
    old_head = bind(BAt.AttentionHeadV8, T=17, D=16, BN=16, MASKED_FROM=0, **FA2_CHECK)
    old = codec.encode_program(_program(old_head))
    assert codec.program_digest(old) != codec.program_digest(new)
    rep = compare(*(codec.decode_program(d, REGISTRY) for d in (new, old)), gate_samples=64, call_samples=2, depth=1, eval_samples=0)
    assert rep["verdict"] == "equivalent" and rep["reference_runs"] > 0
    assert len(codec.canonical_json(codec.definition_entry(old_head))) > 20 * size[17]


@pytest.mark.slow
def test_a_key_length_adds_no_more_to_the_boolean_descriptor_than_twice_the_word_one() -> None:
    """At Qwen3-0.6B's served statics (`benchmarks/bool_build_cost`'s small case: D = 128, BN = 64, rows of 1 and 65 keys), the
    Definitions a row has one of per key length (`Attention`, `AttentionHead`) or per block width (`AttnBlock`) take at most twice
    the word Build's descriptor bytes.  The Boolean primitives' bit circuits, written once per Program whatever the row, are not
    counted."""
    bits = [bind(BAt.AttentionV8, T=T, NH=16, KVH=8, D=128, BN=64, MASKED_FROM=0, **FA2_CHECK) for T in (1, 65)]

    def by_family(fns) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, v in codec.encode_program(_program(*fns))["definitions"].items():
            fam = k.split("{")[0].rsplit("_v", 1)[0]
            out[fam] = out.get(fam, 0) + len(codec.canonical_json(v))
        return out

    w, b = by_family([TR.word_of(f) for f in bits]), by_family(bits)
    for fam in ("Attention", "AttentionHead", "AttnBlock"):
        assert b[fam] <= 2 * w[fam], (fam, b[fam], w[fam])


@pytest.mark.slow
def test_fa3_inv_sum_agrees_with_fa3_inv_sum_v1() -> None:
    """On random words, the special words (signed zeros, infinities, NaNs, subnormals) and every mantissa at exponent 0x7F."""
    target = bind(BAt.Fa3InvSum)
    words = np.concatenate([np.random.default_rng(20).integers(0, 1 << 32, 200_000, dtype=np.uint64).astype(np.uint32),
                            np.array([0, 0x80000000, 0x7FC00000, 0xFFC00001, 0x7F800000, 0xFF800000, 1, 0x00800000], np.uint32),
                            np.arange(1 << 23, dtype=np.uint32) | np.uint32(0x3F800000)])
    got = np.asarray(BS.evaluate_words(target, words)).astype(np.uint64)
    assert np.array_equal(got, np.asarray(E.evaluate_batch(TR.word_of(target), words)).astype(np.uint64))


@pytest.mark.slow
@pytest.mark.parametrize("target", PINNED, ids=lambda t: t.id)
def test_the_switch_maps_its_word_view_to_it(target) -> None:
    """`boolean_version` of each word chain is the Boolean chain whose word view it is: `Attention_v4` (FA3's per-iteration
    `Check_inf`) to v9, `Attention_v5` (FA2's) to v8, v2 and v3 to v7 and v6. Its bindings bind `DOT` and `INV`, as a served
    Program's do, and its body is the same circuit."""
    v = SW.boolean_version(TR.word_of(target))
    assert v.id == target.id
    assert v.gates == target.gates


@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_agrees_with_its_word_view(target) -> None:
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, {6: 3, 7: 2, 8: 5, 9: 4, 16: 13}[target.defn.version],
                                                                  lifted)
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
    ref = np.asarray(E.evaluate_batch(word, *inputs)).astype(np.uint64)
    bad = np.flatnonzero((got != ref).reshape(len(got), -1).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {len(ref)} instances differ from {word.id}, first {bad[:5]}"


@pytest.mark.parametrize("block", [BAt.AttnBlockV8, BAt.AttnBlockV9], ids=lambda d: d.id)
def test_an_unguarded_block_rescales_by_the_running_max_as_it_is(block) -> None:
    """A block entered with the running max at -inf and scores of -inf only (an infinite query against negative keys) keeps it at
    -inf: `AttnBlock_v8{CHECK=False}` rescales by exp2(-inf - -inf), a NaN, as `AttnBlock_v5{CHECK=False}` does, and with `CHECK`
    by exp2(-inf - 0) = 0, so the static is read. `AttnBlock_v9{CHECK=False}` also scales the scores by the unguarded max, as
    `AttnBlock_v4{CHECK=False}` does, which a block guarding it as v8 does would not match."""
    blocks = [bind(block, D=16, NVIS=16, FIRST=False, BN=16, DOT=FA3["DOT"], CHECK=c) for c in (False, True)]
    inputs = _inputs(TR.word_of(blocks[0]), 12, 85)
    q, k, m_prev = inputs[0], inputs[1], inputs[3]
    q[:6] = 0
    q[:6, 0] = 0x7F80
    k[:6] = 0
    k[:6, :, 0] = 0xBF80
    m_prev[:6] = 0xFF800000
    outs = []
    for target in blocks:
        got = np.asarray(BS.evaluate_words(target, *inputs)).astype(np.uint64)
        assert np.array_equal(got, np.asarray(E.evaluate_batch(TR.word_of(target), *inputs)).astype(np.uint64)), target.id
        outs.append(got.reshape(len(got), -1))
    assert (outs[0][:6] != outs[1][:6]).any(axis=1).all()
