"""`registry.mla`: Triton's one-split MLA decode, the absorb bmm and the shared-expert combine on bits are Boolean; agree with
`mla_check`'s numpy models of the kernels (equal to the sm_120 captures of `mla_capture` word for word, 12,420 decode rows, 3,028
bmm rows and 84 combine rows of art run r20261004-060203-ba52) on random words and on words that reach every branch: the exp's
small-argument path, masked columns, signed zeros, subnormals, infinities and NaNs; reduce stage 2's one-split merge to the
gate and `TritonMlaOut` exactly; reproduce captured hardware outputs at GLM-4.7-Flash's statics (by digest); and are pinned by
gate count and program digest."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pytest

from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call
from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.ops.boolean import gemm as BG
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import mla as M
from verity_vllm.program.registry import mla_capture as MC
from verity_vllm.program.registry import mla_check as MK

U16, U32 = np.uint16, np.uint32
HOP = BG.HopperBF16WgmmaDot16
G = M.GLM47_FLASH


def _decode(T: int, H: int = 2, DL: int = 16, DR: int = 16, SCALE: int = G["SCALE"]):
    return bind(M.TritonMlaDecode, H=H, T=T, DL=DL, DR=DR, BN=32, SPLITS=1, SCALE=SCALE, DOT=HOP)


#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "TritonExpSub_v1": (6139, 13952, 564, "a770f71187fbae2276536a59c16f83b42fa0a7011566e734fff66cd299fc9e512da88f4730234e6c8a191c83c29893ccac5ba92078710c9f8947fa345db0d589"),
    "TritonMlaGate_v1": (20, 14, 3, "d99f1feab53e3f11e92f6938ae5fab2548bade966ab3c8f07ea0220a16526bb967b2afc4f5c437ea2e70bc641adec70f2987446a191670042173e65aa9147922"),
    "TritonMlaOut_v1": (3023, 5542, 257, "090eb39de5ac5fcc6bd1d5a49fad435d308bbb24c7492b882dd56d64d88a7d0cfa2d96bbe5024ecacc284a719073d7cd4c63916a6c5d694b82ffdc6e4a5651d4"),
    "F32MulConst_v1{K=1031798784}": (700, 948, 93, "bd652b800a1a2d62a22f3124617bd284560840386d839200c5a2cabc513fda41a73d2c9a0e53f799a4b3944248e99a227ba9346c7f014a40a96c0541f7bc0661"),
    "MoeSharedCombineElement_v1{RSF=1072064102}":
        (1579, 2395, 239, "1d96a928f481d0ad9692c7b39f8e1b799f6b8b52861ae0fdc79847a9e722869598f2e3546c57d733a5532f19e4e10f3cc02de3944667f217286e3b953b92dddd"),
    "TritonMlaBlock_v1{DL=16,DR=16,NVIS=2,BN=32,SCALE=1031798784,DOT=HopperBF16WgmmaDot16_v2}":
        (361161, 696299, 33251, "3d68126c42ced14d453378adf58424a0cbb41dc0d7e130791cb2b198ddee930270d8faea29037a6cc38c19ea0d562c4c22e60a955e8467f57c0a7b8dd8c61c7b"),
    "TritonMlaDecode_v1{H=1,T=2,DL=16,DR=16,BN=32,SPLITS=1,SCALE=1031798784,DOT=HopperBF16WgmmaDot16_v2}":
        (410701, 790157, 37510, "60189d9b465d3b237d40ebb4788e73bccf3f9b4e3ed53ec7a8504d30dc1c382a62806e9225ebc4b09054bf727fb30fcf0a2353a7c97f2c6690439e66a8099700"),
    "MlaBmm_v1{NHEADS=2,K=32,N=3,DOT=HopperBF16WgmmaDot16_v2}":
        (94860, 181176, 9222, "732c8e4c85ba6e3c6128c9ae72f73ae40ad5ab509154829b0a440f5cb06f6edaec2f15dfae461612bd285421e4bbd6a8cba91ef6b4ae0d27d9e82d9613e08fcc"),
    "MoeSharedCombine_v1{H=8,RSF=1072064102}": (12632, 19160, 1912, "8ada4e2f0dbaaa72f92b78bdf6b64ffd0211a1eae36a28023e3d0c502527e59169532c6bf51f26dc9025fa565b2f64ff5df4ad286bbb83698145ff63df534e7b"),
}
PINNED = [bind(M.TritonExpSub), bind(M.TritonMlaGate), bind(M.TritonMlaOut), bind(M.F32MulConst, K=G["SCALE"]),
          bind(M.MoeSharedCombineElement, RSF=G["RSF"]),
          bind(M.TritonMlaBlock, DL=16, DR=16, NVIS=2, BN=32, SCALE=G["SCALE"], DOT=HOP), _decode(2, H=1),
          bind(M.MlaBmm, NHEADS=2, K=32, N=3, DOT=HOP), bind(M.MoeSharedCombine, H=8, RSF=G["RSF"])]

#: f32 words at every class edge: signed zeros, subnormals, the normal range's ends, -126 and its neighbours (the exp's
#: small-argument switch), 1, infinities and NaNs
F32_SPECIALS = np.array([0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF,
                         0xC2FC0000, 0xC2FBFFFF, 0xC2FC0001, 0xC2FE0000, 0xC3800000, 0x3F800000, 0xBF800000, 0x42B00000,
                         0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00001, 0x7F800001], U32)
#: bf16 words likewise
BF16_SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x0080, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x5F00, 0xDF00,
                          0x3F80, 0xBF80], U16)


def _bits_eval(target, *inputs) -> np.ndarray:
    return np.asarray(BS.evaluate_words(target, *inputs))


def _f32_words(rng, n: int) -> np.ndarray:
    """Uniform words, values in [-300, 300] (both sides of the exp's switch at -126), and each special against random words."""
    a = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(U32)
    b = MK.u32(rng.uniform(-300, 300, n).astype(np.float32))
    return np.concatenate([a, b])


def _bf16_words(rng, shape, kind: str) -> np.ndarray:
    n = int(np.prod(shape))
    if kind == "uniform":
        return rng.integers(0, 1 << 16, n).astype(U16).reshape(shape)
    scale = {"spread": 4.0, "large": 64.0, "tiny": 1e-3, "special": 4.0}[kind]
    x = (MK.u32(rng.standard_normal(n).astype(np.float32) * np.float32(scale)) >> 16).astype(U16)
    if kind == "special":
        m = rng.random(n) < 0.02
        x[m] = BF16_SPECIALS[rng.integers(0, len(BF16_SPECIALS), int(m.sum()))]
    return x.reshape(shape)


def test_every_definition_is_boolean() -> None:
    assert all(BL.is_boolean(t) for t in PINNED)
    assert BL.is_boolean(_decode(65, H=1)) and BL.is_boolean(bind(M.MlaBmm, NHEADS=1, K=192, N=2, DOT=HOP))


def test_exp_sub_agrees_with_the_model() -> None:
    """`TritonExpSub_v1` against `ex2.approx.f32(RN(RN(x - m) * log2 e))` as ptxas lowers it (`mla_check.exp_tl`), which equals
    the sm_120 MUFU sweep: every pair of specials and 200k random pairs on both sides of the -126 switch."""
    rng = np.random.default_rng(1)
    x, m = _f32_words(rng, 100_000), _f32_words(rng, 100_000)
    gx, gm = (F32_SPECIALS[g] for g in np.array(list(itertools.product(range(len(F32_SPECIALS)), repeat=2))).T)
    x, m = np.concatenate([x, gx]), np.concatenate([m, gm])
    got = _bits_eval(bind(M.TritonExpSub), x, m).astype(U32)
    want = MK.exp_tl(MK.sub(x, m))
    nan = MK.is_nan(want)
    assert np.array_equal(got[~nan], want[~nan]) and np.all(got[nan] == 0x7FFFFFFF)
    assert np.count_nonzero(MK.f32(MK.mul(MK.sub(x, m), MK.LOG2E)) < -126) > 10_000


def _merge_one_split(e_max, e_sum, tv):
    """`_fwd_kernel_stage2` over one split as `mla_check.decode_rows` runs it (the split's tl, then the merge from (-inf, 0, 0))."""
    tl = MK.add(e_max, MK.logf(e_sum))
    n = MK.fmax(tl, MK.NEG_INF)
    old, el = MK.exp_tl(MK.sub(MK.NEG_INF, n)), MK.exp_tl(MK.sub(tl, n))
    acc = MK.fma(np.zeros_like(tv), old, MK.mul(el, tv))
    return MK.bf16_of(MK.div_full(acc, MK.fma(np.zeros_like(e_sum), old, el)))


def test_one_split_merge_is_the_gate_and_out() -> None:
    """Stage 2's merge of one split equals ``TritonMlaGate ? F2fp(tv + 0) : 0x7FFF`` with ``tv = div.full(acc, e_sum)``
    (`TritonMlaOut_v1`) on every (e_max, e_sum) pair of specials and on random words, with e_sum around 1..T as the kernel
    makes it; MUFU.RCP(1) is 1, so ``div.full(x, 1) = x``."""
    assert int(MU.mufu_rcp_words(np.array([MK.ONE], U32), MU.table("rcp"))[0]) == int(MK.ONE)
    rng = np.random.default_rng(2)
    n = 60_000
    e_max = np.concatenate([_f32_words(rng, n // 2), np.repeat(F32_SPECIALS, len(F32_SPECIALS))])
    e_sum = np.concatenate([MK.u32(rng.uniform(1, 4096, n // 2).astype(np.float32)), rng.integers(0, 1 << 32, n // 2,
                            dtype=np.uint64).astype(U32), np.tile(F32_SPECIALS, len(F32_SPECIALS))])
    e_max = np.concatenate([e_max, MK.u32(rng.uniform(-50, 50, len(e_sum) - len(e_max)).astype(np.float32))])
    acc = np.concatenate([MK.u32((rng.standard_normal(len(e_sum)) * 8).astype(np.float32))])
    acc[::97] = F32_SPECIALS[np.arange(len(acc[::97])) % len(F32_SPECIALS)]
    tv = MK.div_full(acc, e_sum)
    want = _merge_one_split(e_max, e_sum, tv)
    gate = _bits_eval(bind(M.TritonMlaGate), e_max, e_sum).reshape(-1).astype(bool)
    finite = (MK.add(e_max, MK.logf(e_sum)) & U32(0x7F800000)) != U32(0x7F800000)
    assert np.array_equal(gate, finite)
    rcp = MU.mufu_rcp_words(MU.div_full_divisor_words(e_sum), MU.table("rcp"))
    got = _bits_eval(bind(M.TritonMlaOut), acc, e_sum, rcp, gate.astype(np.uint8)).astype(U16)
    assert np.array_equal(got, want)
    assert 0.3 < gate.mean() < 0.9 and np.count_nonzero(want == 0x7FFF) > 1000


def _agree_decode(T: int, kind: str, n: int = 64, H: int = 2, DL: int = 16, DR: int = 16, SCALE: int = G["SCALE"]) -> None:
    rng = np.random.default_rng([T, DL, SCALE, len(kind)])
    q, kv = _bf16_words(rng, (n, H, DL + DR), kind), _bf16_words(rng, (n, T, DL + DR), kind)
    got = _bits_eval(_decode(T, H, DL, DR, SCALE), q, kv).astype(U16)
    want = np.stack([MK.decode_rows(q[i], kv[i], 1, SCALE, DL)["o"] for i in range(n)])
    bad = np.argwhere(got != want)
    assert not len(bad), f"T={T} {kind}: {len(bad)} words differ, first {bad[0]}: {got[tuple(bad[0])]:#x} vs {want[tuple(bad[0])]:#x}"


@pytest.mark.parametrize("T,kind", [(1, "spread"), (33, "large"), (33, "special")])
def test_decode_agrees_with_the_model(T: int, kind: str) -> None:
    """One block, two blocks with a masked last one, the exp's small-argument path (``large``: scores hundreds apart), and
    sprinkled specials, at a 16 + 16 latent."""
    _agree_decode(T, kind)


@pytest.mark.slow
@pytest.mark.parametrize("T", [2, 16, 17, 31, 32, 64, 65])
@pytest.mark.parametrize("kind", ["spread", "large", "special", "uniform", "tiny"])
def test_decode_agrees_with_the_model_at_every_mask_shape(T: int, kind: str) -> None:
    """The other mask shapes (two keys, half a block, one past it, one short of a full block, a full block, two, and one key
    past them) on every input kind, and two other scales (1 and 8: scores past the exp's switch, and close to overflow)."""
    _agree_decode(T, kind)
    if kind == "large":
        _agree_decode(T, kind, SCALE=0x3F800000)
    if kind == "spread":
        _agree_decode(T, kind, SCALE=0x41000000)


#: captured `o` of one-split requests of `decode-v1` (run r20261004-060203-ba52; `mla_capture.case_inputs` rebuilds their inputs
#: from the case): (case index, request, sha256 of its 20 x 512 output words, little-endian)
CAPTURED = [
    ({"seqlens": [1, 2, 31, 32, 33], "splits": 1, "kind": "unit", "page": 64, "seed": 1}, 0, "1b7483a67b1bb456"),
    ({"seqlens": [1, 2, 31, 32, 33], "splits": 1, "kind": "unit", "page": 64, "seed": 1}, 1, "3dde29d03c6933d1"),
    ({"seqlens": [1, 2, 31, 32, 33], "splits": 1, "kind": "unit", "page": 64, "seed": 1}, 4, "d770285380b18149"),
]


@pytest.mark.slow
@pytest.mark.parametrize("case,b,digest", CAPTURED, ids=[f"T={c['seqlens'][b]}" for c, b, _ in CAPTURED])
def test_decode_reproduces_the_captured_hardware_output(case: dict, b: int, digest: str) -> None:
    """`TritonMlaDecode_v1` at GLM-4.7-Flash's statics (20 heads, 512 + 64 latent, sm_scale 2^-4) on a captured request: its
    output words hash to what the RTX PRO 6000 (sm_120) wrote. The only check of the Definition itself at the served widths,
    where the score chain has 36 steps and each block 1,024 value steps (about 10 s a request)."""
    q, kvs = MC.case_inputs(case)
    T = len(kvs[b])
    t = bind(M.TritonMlaDecode, H=MC.H, T=T, DL=G["DL"], DR=G["DR"], BN=32, SPLITS=1, SCALE=G["SCALE"], DOT=G["DOT"])
    o = _bits_eval(t, q[b:b + 1], kvs[b][None]).astype("<u2")[0]
    assert hashlib.sha256(o.tobytes()).hexdigest()[:16] == digest


def test_bmm_agrees_with_the_model() -> None:
    """`MlaBmm_v1` against `bmm_kernel`'s model (per head `Gemm_v3`, `mla_check.bmm_rows`) at two and three heads on random
    words and on specials."""
    rng = np.random.default_rng(3)
    for NH, K, N in ((2, 32, 3), (3, 64, 5)):
        n = 64
        x = _bf16_words(rng, (n, NH, K), "spread")
        w = _bf16_words(rng, (n, NH, N, K), "special")
        got = _bits_eval(bind(M.MlaBmm, NHEADS=NH, K=K, N=N, DOT=HOP), x, w).astype(U16)
        want = np.stack([MK.bmm_rows(x[i][:, None, :], np.transpose(w[i], (0, 2, 1)))[:, 0] for i in range(n)])
        assert np.array_equal(got, want)


@pytest.mark.parametrize("rsf", [1.8, 1.0, 2.5])
def test_combine_agrees_with_the_model(rsf: float) -> None:
    """`MoeSharedCombine_v1` against ATen's bf16 combine (`mla_check.combine_rows`, NaN 0x7FFF as measured) on every pair of
    bf16 specials and 64k random pairs, at GLM-4.7-Flash's factor 1.8, at 1.0 (where vLLM skips the multiply) and at 2.5."""
    rng = np.random.default_rng(4)
    g = np.array(list(itertools.product(range(len(BF16_SPECIALS)), repeat=2))).T
    routed = np.concatenate([BF16_SPECIALS[g[0]], rng.integers(0, 1 << 16, 1 << 16).astype(U16)])
    shared = np.concatenate([BF16_SPECIALS[g[1]], rng.integers(0, 1 << 16, 1 << 16).astype(U16)])
    RSF = int(np.float32(rsf).view(np.uint32))
    got = _bits_eval(bind(M.MoeSharedCombine, H=len(routed), RSF=RSF), routed[None], shared[None])[0].astype(U16)
    want = MK.combine_rows(routed[None], shared[None], rsf, 0x7FFF)[0]
    assert np.array_equal(got, want)
    if rsf == 1.0:
        s = MK.add(DR.bf16_to_f32_words(shared), DR.bf16_to_f32_words(routed))
        assert np.array_equal(got, np.where(MK.is_nan(s), U16(0x7FFF), MK.bf16_of(s)))


def test_check_bits_replays_bmm_and_combine_captures(tmp_path) -> None:
    """`mla_capture.check_bits` on a capture directory whose outputs are the models' words: every row agrees, a rerun skips
    the keys an earlier report finished, and one flipped combine word is the first mismatch it names."""
    import json

    cases = [{"op": "bmm", "NH": 2, "M": 3, "K": 64, "N": 2, "kind": "special", "seed": 7},
             {"op": "combine", "T": 3, "HD": 16, "rsf": 1.8, "kind": "special", "seed": 8}]
    (tmp_path / "manifest.json").write_text(json.dumps({"suite": "t", "cases": cases}))
    np.savez(tmp_path / "case_0.npz", c=MK.bmm_rows(*MC.bmm_inputs(cases[0])))
    out = MK.combine_rows(*MC.combine_inputs(cases[1]), 1.8, 0x7FFF)
    np.savez(tmp_path / "case_1.npz", out=out)
    rep = MC.check_bits(tmp_path, out=tmp_path / "r.json")
    assert (rep["rows"], rep["mismatched_rows"]) == (2 * 3 + 3, 0)
    assert MC.done_keys([tmp_path / "r.json"]) == {"case=0", "case=1"}
    again = MC.check_bits(tmp_path, skip={"case=1"}, fresh=True)
    assert again["by_key"].keys() == {"case=0"} and (again["rows"], again["mismatched_rows"]) == (2 * 3, 0)
    out[1, 3] ^= 1
    np.savez(tmp_path / "case_1.npz", out=out)
    first = [r["first_mismatch"] for r in (MC.check_bits(tmp_path, workers=2, worker=w) for w in (0, 1)) if r["mismatched_rows"]]
    assert first == [{"case": 1, "token": 1, "column": 3, "got": int(out[1, 3] ^ 1), "want": int(out[1, 3])}]


@pytest.mark.parametrize("target", PINNED, ids=[t.id for t in PINNED])
def test_pinned_counts_and_digests(target) -> None:
    pin = PINS[target.id]
    prog, _ = standalone_call(target)
    got = (*(BS.compile(target).counts[k] for k in ("and", "xor", "not")), codec.program_digest(codec.encode_program(prog)))
    assert pin == got, f"{target.id}: {got}"


def test_every_pin_is_a_target() -> None:
    assert sorted(PINS) == sorted(t.id for t in PINNED)


def test_the_decode_rejects_what_it_does_not_model() -> None:
    with pytest.raises(ValueError, match="SPLITS_is_1"):
        _ = bind(M.TritonMlaDecodeHead, T=2, DL=16, DR=16, BN=32, SPLITS=2, SCALE=G["SCALE"], DOT=HOP).body
    with pytest.raises(ValueError, match="BN_is_32"):
        _ = bind(M.TritonMlaDecodeHead, T=2, DL=16, DR=16, BN=16, SPLITS=1, SCALE=G["SCALE"], DOT=HOP).body
