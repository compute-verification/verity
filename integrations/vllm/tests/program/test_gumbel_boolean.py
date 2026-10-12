"""`verity_vllm.program.registry.gumbel_boolean`: the Boolean Gumbel select agrees with its word Definitions (`trace.word_of`).
`NvLogf_v3` agrees with `NvLogf_v2`, and the log1pf builder with `nv_log1pf_gpu`, on every special word, every exponent of
both signs, the sweeps around their branch points, and random words. The uniform and noise of a Philox word agree on its
edges (0, all ones, the floor, the f32 rounding of large words). `GumbelStreamKey_v2` agrees with `GumbelStreamKey_v1`;
`GumbelSetup_v1{N}`'s outputs are the lane operands restated; `GumbelLane_v1` on those operands is ``x + GumbelNoiseLane_v1(key,
v)`` (or ``x``) at every lane index; and `GumbelSelectF32_v2{V}` agrees with `GumbelSelectF32_v1{V}` on logit rows (ties,
signed zeros, subnormals, infinities, NaNs, a NaN first logit; temp 1, +0, -0, a subnormal and a NaN) at one and two levels
of its tree. Its stages are its body's Calls; each partitions under `Q_word_v1` with no violation at V = 65 and at Gemma-2's
256,000; and the Definitions are pinned by gate count and program digest.

Tracing the lane's noise (tl.rand and two libdevice logs: about 90,000 ANDs) takes about 10 s in a fresh process, so every
test that reaches it is slow, and they are few."""
from __future__ import annotations

import json
import struct
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
from verity_vllm.program.registry import gumbel_boolean as GB
from verity_vllm.program.registry import sampling as SM
from verity_vllm.program.registry import sampling_boolean as SB
from verity_vllm.query import word as W

F32 = GB.F32

M32 = 0xFFFFFFFF
#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "NvLogf_v3": (43280, 74409, 2800, "69ad963aa5ff5276e47ea1065d7ab2b65ec7d522b15b9849c2b724ae1acd5bbe9f612a6ba6e097e109ef99207b56b94226c8ff2d2f18a1708448e95693343a3b"),
    "GumbelSetup_v1{N=5}": (1138, 4005, 160, "665163a8e11b818c93cfec66672ff17fd69654c4ac363d8e7e35cc706113df75af13272602207635a6d1d1aff74455febde42587166cfbb27d0931d776cb0274"),
    "GumbelSetup_v1{N=64}": (3859, 13430, 160, "9b9c10135574f923cfbb8b6139811415e652dc2b22dac77ca97cb99ebd1978135cc6cdab0e514097e6da603a835f0c778286c678b26c2a9b1bc68147aa811c7f"),
    "GumbelStreamKey_v2": (9245, 39200, 305, "12e3eed59d4c4a8bb162063b135d04ce5f8164bf0b568e458555a537ea4dcfbe8916f97d6c488c36ad4eaddd75421179883ada1f965e1e2bf6d1f6741e8fb8ac"),
    "GumbelStreamKeyHead_v1": (4960, 20667, 305, "fe6d2ab0625dacc3b0fb5573482132b8343ccd14f138f4afe69ce3c7a0399d1fd5f0c36349d43daf1bf18993b984e1543ba0c03362134420f7a3abc27c229589"),
    "GumbelStreamKeyTail_v1": (4285, 18533, 0, "2c31e3ac773074d314192bcf2525a37bf132b13774d0624564bee2738244d3a6eff61a05949a888a6744e57c6b57d55db5f97f2166a5223059c2c06ed7ac5d32"),
    "GumbelLaneHead_v1": (1556, 6643, 35, "4487033a1bf1192c3abb681b2df58de39529b3c5431d2d6a82d274d6f91a4a9be982447252b14ac8eed5762444635cfd1953c22e27bae49d082e41daea1eb7a9"),
    "GumbelLaneRounds_v1": (5035, 21681, 81, "e8a27237e45e078260046e86707d68c70aca8cd642141084ffbf5f7665803d25e0aa82f71e10cea14a88843e613c11e98b4fafd5d556d69ecbe7dbc6990c2a59"),
}
#: the same for the Definitions that reach the lane's noise (slow to trace)
LANE_PINS = {
    "GumbelLaneTail_v1": (92314, 162277, 6115, "beb5c6bac7ca2e8918bf4cb12111467bce5c480b9440ae4ae770cec422a1eafafe41997c70ee4feb2b4792f6feee11159662d120959f462cc08da535b8be12a2"),
    "GumbelLane_v1": (98905, 190601, 6231, "41cba3705e0fe255a2a472256d27b1789c995d7fc7f6ce5535859adfc2cae502ac2917414236498f625fba46eb48afb06fad2d3af94edaaa788ab4b917e172e8"),
    "GumbelNoiseGroup_v1{G=64}": (6329920, 12198464, 398784, "b48fafac10e3ab3b059f99587d4b6bfa5e45bed8d9bd2882fec6d14abda01f6b3cbedb2281e14844e6a116663557c8ae9b453c28a82ae5d3a6a233158d710ea5"),
    "GumbelNoise_v1{V=65}": (6428825, 12389065, 405015, "df8df2b1f311f90e6e8d90eb439269663cf8327c3a3c8ddd300d12d17511da6a760321383cccffdc4669840abb82a656057ad9ad0a84170fbda3963952d25bfb"),
    "GumbelSelectF32_v2{V=5}": (505349, 997312, 31762, "26d127cdc687e94c2fe212e228e2e050236c103fd4020ab440d84acb5df7af446c8cfd761b462979a87cf48e96318d1015881b624f123be08dd3697b7df11788"),
    "GumbelSelectF32_v2{V=65}": (6448490, 12458099, 407786, "50615df3a5d769c3d865e0da9ee65ca187b3458f1f1f85b1d5738fdad87a501de38ae87bf4f91597d15704712616de4aac58ca8a42f25252f361e9fb3f3bdd49"),
}

#: each stage's gates at Gemma-2's vocabulary (the noisy row: 256,000 lanes of GumbelLane_v1)
SERVED_GATES = {"GumbelStreamKey_v2": 48750, "GumbelSetup_v1{N=64}": 17449, "GumbelNoise_v1{V=256000}": 75708672000,
                "F32ArgmaxStage_v1{V=256000,L=0}": 99472093, "F32ArgmaxStage_v1{V=256000,L=1}": 1116978,
                "F32ArgmaxStage_v1{V=256000,L=2}": 18610}

INF, NEG_INF = 0x7F800000, 0xFF800000
#: signed zeros, the smallest and largest subnormals, the smallest normals, ones, the largest finite values, infinities and
#: NaNs (quiet, signalling, all ones)
F32_SPECIALS = np.array([m | s for m in (0x00000000, 0x00000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, INF, 0x7F800001,
                                         0x7FC00000, 0x7FFFFFFF) for s in (0, 0x80000000)], np.uint64)


#: the traced Definitions (`trace.definition`, `sampling_boolean.traced_family`), the lane's noise last
TRACED = [GB.NvLogf, GB.GumbelSetup, GB.GumbelStreamKeyHead, GB.GumbelStreamKeyTail, GB.GumbelLaneHead, GB.GumbelLaneRounds,
          GB.GumbelLaneTail]
TRACED_SPECS = ["NvLogf_v3", "GumbelSetup_v1{N=5}", "GumbelSetup_v1{N=64}", "GumbelStreamKeyHead_v1", "GumbelStreamKeyTail_v1",
                "GumbelLaneHead_v1", "GumbelLaneRounds_v1", "GumbelLaneTail_v1"]


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _target(spec: str):
    fam, _, body = spec.partition("{")
    d = {f"{d.name}_v{d.version}": d for d in TRACED + [GB.GumbelStreamKey, GB.GumbelLane, GB.GumbelNoiseGroup, GB.GumbelNoise,
                                                      GB.GumbelSelectF32]}[fam]
    kw = dict(kv.split("=") for kv in body.rstrip("}").split(",")) if body else {}
    return bind(d, **{k: int(v) for k, v in kw.items()})


def _check_pin(spec: str, pin) -> None:
    target = _target(spec)
    assert target.id == spec
    ands, xors, nots, digest = pin
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}, spec
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest, spec
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


# --- pins ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", sorted(PINS))
def test_pinned_counts_and_digests(spec) -> None:
    _check_pin(spec, PINS[spec])


@pytest.mark.slow
def test_pinned_counts_and_digests_through_the_lane() -> None:
    for spec in sorted(LANE_PINS):
        _check_pin(spec, LANE_PINS[spec])


@pytest.mark.parametrize("spec", [pytest.param(s, marks=pytest.mark.slow) if s == "GumbelLaneTail_v1" else s
                                  for s in TRACED_SPECS])
def test_the_ands_are_the_builders_live_ands(spec) -> None:
    target = _target(spec)
    d = target.defn
    params, ret = target.signature()
    C = Circuit(cse=True)
    ins = TR._elements_in(C, [t for _, t in params])
    if d.statics:
        T = d.traced(target)
        C.output("out", TR._flat_out(d.build(C, target.S(), *ins), ret))
    else:
        T = d.traced()
        C.output("out", TR._flat_out(d.build(C, *ins), ret))
    assert T.counts["and"] == C.counts()["ands"] == {**PINS, **LANE_PINS}[spec][0]


def test_the_pieces_add_up() -> None:
    """The key and the lane are Calls of their traced pieces, cut where the next piece reads every output: their ANDs are the
    pieces', and one trace of each whole gives the same: 98,905 and 9,245."""
    key = sum(PINS[f"GumbelStreamKey{p}_v1"][0] for p in ("Head", "Tail"))
    assert key == PINS["GumbelStreamKey_v2"][0] == 9245
    lane = sum({**PINS, **LANE_PINS}[f"GumbelLane{p}_v1"][0] for p in ("Head", "Rounds", "Tail"))
    assert lane == LANE_PINS["GumbelLane_v1"][0] == 98905


def test_shapes_out_of_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="a group has 1..64 lanes, not 65"):
        bind(GB.GumbelNoiseGroup, G=65).signature()
    with pytest.raises(ValueError, match="a logits row has V >= 1 entries, not 0"):
        bind(GB.GumbelNoise, V=0).signature()
    with pytest.raises(ValueError, match="a group has 1..64 offsets, not 0"):
        bind(GB.GumbelSetup, N=0).signature()


# --- the logs --------------------------------------------------------------------------------------------------------------

def _log_inputs(rng, around: tuple[int, ...], n: int) -> np.ndarray:
    """Every special word; every exponent of both signs with mantissas 0, all ones and 32 random ones; 1024 words on each side
    of each word in ``around``; ``n`` random words."""
    e = np.arange(512, dtype=np.uint64) << np.uint64(23)
    mant = np.concatenate([np.zeros((512, 1), np.uint64), np.full((512, 1), 0x7FFFFF, np.uint64),
                           rng.integers(0, 1 << 23, (512, 32), dtype=np.uint64)], axis=1)
    sweeps = [np.arange(a - 1024, a + 1024, dtype=np.int64).astype(np.uint64) & np.uint64(M32) for a in around]
    return np.concatenate([F32_SPECIALS, (e[:, None] | mant).reshape(-1), *sweeps,
                           rng.integers(0, 1 << 32, n, dtype=np.uint64)]).astype(np.uint32)


def test_nv_logf_agrees_with_its_word_definition() -> None:
    """Around 1 (the result's zero), the binade selector's 0x3F2AAAAB and FLT_MIN (the subnormal rescale), and +inf."""
    target = bind(GB.NvLogf)
    word = TR.word_of(target)
    assert word.id == "NvLogf_v2"
    a = _log_inputs(_rng("logf"), (0x3F800000, 0x3F2AAAAB, 0x00800000, INF), 50_000)
    got = BS.evaluate_words(target, a).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(word, a).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{bad.size} of {len(a)} words differ from NvLogf_v2, first {[hex(int(a[i])) for i in bad[:5]]}"


NvLog1pf = TR.definition("NvLog1pf", 1, GB.nv_log1pf, (("a", F32),), F32, register=False)


def test_nv_log1pf_agrees_with_libdevice() -> None:
    """Against `sampling.nv_log1pf_gpu`: around the selector's 0x3F400000 and the toward-zero sum's word 0x3EFFFFFF, -1 (the
    result's -inf and the NaN side), 0 and -0, and +inf."""
    a = _log_inputs(_rng("log1pf"), (0x3F400000, 0x3EFFFFFF, 0xBF800000, 0, 0x80000000, INF), 20_000)
    got = BS.evaluate_words(bind(NvLog1pf), a).astype(np.uint64).reshape(-1)
    ref = np.array([SM.nv_log1pf_gpu(int(x)) for x in a], np.uint64)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{bad.size} of {len(a)} words differ from nv_log1pf_gpu, first {[hex(int(a[i])) for i in bad[:5]]}"


# --- the uniform, the stream key and the noise -----------------------------------------------------------------------------

def _rand_noise(w: int) -> int:
    """`sampling._rand_lane` after its Philox word ``w``, then `sampling.noise_lane`'s logs."""
    xi = w - (1 << 32) if w >> 31 else w
    if xi < 0:
        xi = -xi - 1
    u = SM.mul_ftz(struct.unpack("<I", struct.pack("<f", float(xi)))[0], SM.TL_RAND_SCALE_BITS)
    u = max(u, SM.TL_RAND_MIN_BITS)                     # both positive words: their order is their values'
    return SM.nv_logf(SM.nv_log1pf(u ^ 0x80000000) ^ 0x80000000) ^ 0x80000000


RandNoise = TR.definition("RandNoise", 1, lambda C, w: GB.gumbel_noise(C, GB.tl_rand(C, w)), (("w", GB.U32),), F32, register=False)

@pytest.mark.slow
def test_uniform_and_noise_of_a_philox_word() -> None:
    """The words a random lane almost never draws: 0 and all ones (``tl.rand``'s zero, floored at 2^-31'), the words beside
    every power of two of both signs (the f32 rounding of the 31-bit integer, the uniform's top), and random words; the
    restatement is `noise_lane`'s on random lanes."""
    rng = _rng("rand-noise")
    for key, v in rng.integers(0, 1 << 32, (256, 2), dtype=np.uint64).tolist():
        assert _rand_noise(SM.philox4x32_10_word0(v, 0, 0, 0, key, 0)) == SM.noise_lane(key, v)
    p = np.array([(1 << k) + d for k in range(32) for d in (-2, -1, 0, 1, 2)], np.int64)
    w = np.concatenate([p, -p - 1, rng.integers(0, 1 << 32, 20_000)]).astype(np.int64).astype(np.uint64) & np.uint64(M32)
    got = BS.evaluate_words(bind(RandNoise), w.astype(np.uint32)).astype(np.uint64).reshape(-1)
    ref = np.array([_rand_noise(int(x)) for x in w], np.uint64)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{bad.size} of {len(w)} words differ, first {[hex(int(w[i])) for i in bad[:5]]}"


def test_stream_key_agrees_with_its_word_definition() -> None:
    """Random 64-bit seeds and positions, and each with its halves at the edges."""
    rng = _rng("key")
    n = 8192
    half = np.array([0, 1, 0x7FFFFFFF, 0x80000000, M32], np.uint64)
    seed, pos = (rng.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2) + rng.integers(0, 2, n, dtype=np.uint64)
                 for _ in range(2))
    edge = np.array([(a << np.uint64(32)) | b for a in half for b in half], np.uint64)
    seed[:len(edge)] = edge
    pos[len(edge):2 * len(edge)] = edge
    target = bind(GB.GumbelStreamKey)
    assert TR.word_of(target).id == "GumbelStreamKey_v1"
    got = BS.evaluate_words(target, seed, pos).astype(np.uint64).reshape(-1)
    ref = E.evaluate_batch(TR.word_of(target), seed, pos).astype(np.uint64).reshape(-1)
    bad = np.flatnonzero(got != ref)
    assert not bad.size, f"{bad.size} of {n} keys differ, first {[(int(seed[i]), int(pos[i])) for i in bad[:5]]}"


def lane_values(key: int, n: int = GB.G) -> tuple[list[int], list[int], list[int]]:
    """`GumbelSetup_v1{N=n}`'s lane values restated: the round keys ``key + r PHILOX_KEY_A`` (r in `LANE_ROUNDS`), round 1's
    product ``key * PHILOX_ROUND_A`` (low and high words), and per offset ``lo < n`` the low six bits of round 2's counter word
    2 (``ka_hi ^ (lo * PHILOX_ROUND_A) ^ PHILOX_KEY_B``) times ``PHILOX_ROUND_B``."""
    ka = key * SM.PHILOX_ROUND_A
    lows = [(((ka >> 32) ^ (lo * SM.PHILOX_ROUND_A) ^ SM.PHILOX_KEY_B) & 63) * SM.PHILOX_ROUND_B for lo in range(n)]
    return [(key + r * SM.PHILOX_KEY_A) & M32 for r in GB.LANE_ROUNDS], [ka & M32, ka >> 32], lows


def lane_index(v: int) -> tuple[int, int]:
    """`lane_products`' operands of lane ``v = lo + 64 hi``: ``hi * PHILOX_ROUND_A mod 2^58`` and ``lo * PHILOX_ROUND_A``."""
    return (v // GB.G * SM.PHILOX_ROUND_A) % (1 << GB.HAB), v % GB.G * SM.PHILOX_ROUND_A


KEY_EDGES = np.array([0, 1, 0x7FFFFFFF, 0x80000000, M32], np.uint64)
LANE_EDGES = np.array([0, 1, 63, 64, 65, 4095, 4096, 255999, 0x7FFFFFFF, 0x80000000, M32], np.uint64)
TEMPS = np.array([0x3F800000, 0, 0x80000000, 1, 0x7FC00000], np.uint64)


@pytest.mark.parametrize("N", [5, 64])
def test_setup_is_the_lane_values(N) -> None:
    """`GumbelSetup_v1{N}` on random keys and the edges, at temperatures 1, +0, -0, a subnormal and a NaN (noisy iff not
    zero)."""
    key = np.concatenate([KEY_EDGES, _rng("setup").integers(0, 1 << 32, 2000, dtype=np.uint64)])
    temp = np.resize(TEMPS, len(key))
    got = BS.evaluate_words(bind(GB.GumbelSetup, N=N), key.astype(np.uint32), temp.astype(np.uint32)).astype(np.uint64)
    for i, k in enumerate(key.tolist()):
        ks, ka, lows = lane_values(k, N)
        assert got[i].tolist() == ks + ka + [int(int(temp[i]) & 0x7FFFFFFF != 0)] + lows, f"key {k:#x}"


def _lane_args(key: np.ndarray, v: np.ndarray, x: np.ndarray, noisy: np.ndarray) -> list[np.ndarray]:
    vals = [lane_values(k) for k in key.tolist()]
    idx = [lane_index(i) for i in v.tolist()]
    return [x, np.array([h for h, _ in idx], np.uint64), np.array([lo for _, lo in idx], np.uint64),
            np.array([vals[i][2][j % GB.G] for i, j in enumerate(v.tolist())], np.uint64),
            np.array([ks for ks, _, _ in vals], np.uint64), np.array([ka for _, ka, _ in vals], np.uint64), noisy]


@pytest.mark.slow
def test_lane_is_the_noise_lane_word() -> None:
    """`GumbelLane_v1` on lane ``v``'s operands (`lane_values`, `lane_index`) is ``add.ftz(x, GumbelNoiseLane_v1(key, v))`` when
    noisy, else ``x``: random keys and lane indices over the whole u32 range and the vocabulary's, with the edges; random and
    special logits, -0 among them (the sum is then the noise itself)."""
    rng = _rng("lane")
    n = 8192
    key, v = (rng.integers(0, 1 << 32, n, dtype=np.uint64) for _ in range(2))
    g = np.array(np.meshgrid(KEY_EDGES, LANE_EDGES)).reshape(2, -1)
    key[:g.shape[1]], v[:g.shape[1]] = g
    v[g.shape[1]:n // 2] %= np.uint64(256000)
    x = rng.integers(0, 1 << 32, n, dtype=np.uint64)
    x[::4] = rng.choice(F32_SPECIALS, len(x[::4]))
    x[1::4] = 0x80000000
    noisy = (np.arange(n) % 7 != 3).astype(np.uint8)
    got = BS.evaluate_words(bind(GB.GumbelLane), *_lane_args(key, v, x, noisy)).astype(np.uint64).reshape(-1)
    noise = E.evaluate_batch(SM.GumbelNoiseLane, key, v).astype(np.uint64).reshape(-1)
    want = np.array([SM.add_ftz(int(a), int(b)) if c else int(a) for a, b, c in zip(x, noise, noisy)], np.uint64)
    bad = np.flatnonzero(got != want)
    assert not bad.size, f"{bad.size} of {n} lanes differ, first {[(int(key[i]), int(v[i])) for i in bad[:5]]}"


# --- GumbelSelectF32_v2 ----------------------------------------------------------------------------------------------------

def select_rows(V: int, n: int, rng) -> tuple[np.ndarray, ...]:
    """``n`` rows of ``V`` f32 logits with a seed, a position and a temperature: in nine kinds, random words, a few values
    (ties at the maximum), specials, a NaN first logit, all NaN, all -inf, all one value (the noise decides), values one ulp
    apart, and subnormals among signed zeros (flushed when noisy, kept at temp 0); temperatures in turn 1, +0, -0, a subnormal
    and a NaN (noisy: a NaN is not zero)."""
    x = rng.integers(0, 1 << 32, (n, V), dtype=np.uint64)
    kind = np.arange(n) % 9
    few = np.array([0x3F800000, 0x40000000, 0x40400000, 0xC0000000, 0, 0x80000000], np.uint64)
    x[kind == 1] = rng.choice(few, (int((kind == 1).sum()), V))
    x[kind == 2] = rng.choice(F32_SPECIALS, (int((kind == 2).sum()), V))
    x[kind == 3, 0] = rng.choice(F32_SPECIALS[14:], int((kind == 3).sum()))
    x[kind == 4] = rng.choice(F32_SPECIALS[14:], (int((kind == 4).sum()), V))
    x[kind == 5] = NEG_INF
    x[kind == 6] = 0x3F800000
    x[kind == 7] = np.uint64(0x41200000) + rng.integers(0, 4, (int((kind == 7).sum()), V), dtype=np.uint64)
    x[kind == 8] = rng.choice(np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x807FFFFF], np.uint64), (int((kind == 8).sum()), V))
    mixed = (rng.random((n, V)) < 0.05) & (kind[:, None] == 0)
    x[mixed] = rng.choice(F32_SPECIALS, int(mixed.sum()))
    seed, pos = (rng.integers(0, 1 << 63, n, dtype=np.uint64) * np.uint64(2) for _ in range(2))
    temp = TEMPS[np.arange(n) % 5]
    return x.astype(np.uint32), seed, pos, temp.astype(np.uint32)


def first_max_of_noisy(x, seed, pos, temp) -> np.ndarray:
    """`GumbelSelectF32_v1` restated: ``y = add_ftz(x, noise_lane(stream_key(seed, pos), v))`` when temp is not zero (a NaN is
    not), else ``x``; the first maximum of ``y`` (+0 = -0, subnormals kept, a NaN never taken), 0 when ``y[0]`` is a NaN."""
    n, V = x.shape
    y = x.astype(np.int64)
    for r in range(n):
        if int(temp[r]) & 0x7FFFFFFF:
            key = SM.stream_key(int(seed[r]), int(pos[r]))
            y[r] = [SM.add_ftz(int(x[r, v]), SM.noise_lane(key, v)) for v in range(V)]
    mag = y & 0x7FFFFFFF
    nan = mag > INF
    k = np.where(nan, -(1 << 40), np.where(y & 0x80000000, -mag, mag))
    out = np.argmax(k, axis=1)
    out[nan[:, 0]] = 0
    return out


def test_first_max_of_noisy_is_the_word_definition() -> None:
    """The restatement the one-logit rows compare with, against `GumbelSelectF32_v1{V}` itself."""
    for V in (2, 3, 5):
        args = select_rows(V, 360, _rng("oracle", V))
        ref = E.evaluate_batch(bind(SM.GumbelSelectF32, V=V), *args).astype(np.int64).reshape(-1)
        assert first_max_of_noisy(*args).tolist() == ref.tolist()


def _stage_ids(V: int) -> list[str]:
    return (["GumbelStreamKey_v2", f"GumbelSetup_v1{{N={min(V, GB.G)}}}", f"GumbelNoise_v1{{V={V}}}"]
            + [s.id for s in SB.stages(V, SB.F32ArgmaxStage)])


#: (V, rows): one lane, a remainder group, a full group, a full group and one lane, two groups and a lane, three groups and
#: a remainder (every size two levels of the argmax tree at most: a third level is `test_sampling_boolean`'s f32 tree)
BULK = [(1, 4096), (2, 4096), (5, 2048), (63, 512), (64, 512), (65, 512), (129, 256), (200, 128)]


@pytest.mark.slow
def test_gumbel_select_agrees_with_its_word_definition() -> None:
    """Every size in one test, so one process traces the lane. At V = 1 against the restatement (`GumbelSelectF32_v1{V=1}`'s
    scan has no iterations, which `verity.ml.circuits.layout.resolve_gate` misreads: `test_sampling_boolean`'s word-view xfail). The
    body is `gumbel_select_calls`: the stages in order (and the lane products' two constant bits), as a served Program calls
    them at its root."""
    for V, n in BULK:
        target = bind(GB.GumbelSelectF32, V=V)
        assert [c for c in (nd.fn.id for nd in target.body.nodes) if not c.startswith("Const")] == _stage_ids(V)
        args = select_rows(V, n, _rng("bulk", V))
        got = BS.evaluate_words(target, *args).astype(np.int64).reshape(-1)
        if V == 1:
            want = first_max_of_noisy(*args)
        else:
            assert TR.word_of(target).id == f"GumbelSelectF32_v1{{V={V}}}"
            want = E.evaluate_batch(TR.word_of(target), *args).astype(np.int64).reshape(-1)
        bad = np.flatnonzero(got != want)
        assert not bad.size, f"V={V}: {bad.size} of {n} rows differ, first {[(int(i), int(got[i]), int(want[i])) for i in bad[:5]]}"
    calls = [nd.fn.id for nd in bind(GB.GumbelSelectF32, V=4161).body.nodes]
    assert [c for c in calls if not c.startswith("Const")] == _stage_ids(4161)


# --- the partition ---------------------------------------------------------------------------------------------------------

def _stages(V: int) -> list:
    return ([bind(GB.GumbelStreamKey), bind(GB.GumbelSetup, N=min(V, GB.G)), bind(GB.GumbelNoise, V=V)]
            + SB.stages(V, SB.F32ArgmaxStage))


@pytest.mark.slow
def test_stages_partition() -> None:
    """Each stage at V = 65 and at Gemma-2's 256,000 under `Q_word_v1`'s unit rule (about 50 s in a fresh process, most of it
    the lane's trace and its cut): no gate computed by two members of the noisy row, which no evaluation shows (the noise's
    shared values, `lane_products` and `round2_lows`, read in place of recomputed ones); the noisy row and the argmax levels are
    separable, and the key and setup fit one unit. Pins each stage's gates at 256,000."""
    for V in (65, 256000):
        gates = {}
        for st in _stages(V):
            members = separable(st)
            if st.defn in (GB.GumbelNoise, SB.F32ArgmaxStage):
                assert members is not None, f"{st.id} is not separable"
                assert all(m.gates <= W.MAX_GATES for m, _ in members)
            else:
                assert st.gates <= W.MAX_GATES
            rule = W.unit_rule(st, max_gates=W.MAX_GATES)
            assert rule["violations"] == [], f"{st.id}: {rule['violations'][:3]}"
            gates[st.id] = st.gates
        if V == 256000:
            assert gates == SERVED_GATES
