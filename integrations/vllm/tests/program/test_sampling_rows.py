"""`verity.ml.definitions.registry.sampling_rows` — the vectorised twins of the sampler Definition are WORD-EQUAL to the registered scalar
evaluators and to the lane-walk `evaluate_spec(GumbelTopPTokenSelect_v1)`; on the GPU fixture rows both equal the device's
served token and the device kernel's masked row.  CPU only; the `div.full` model
(TemperatureScale) needs the in-tree MUFU tables (`rms_relation.tables`) -- those tests skip without them, the noise / log / select tests do not.
"""
from __future__ import annotations

import json
import os
import struct

import numpy as np
import pytest

from verity.ml.definitions.registry import sampling as SP
from verity.ml.definitions.registry import sampling_twins
from verity.ml.definitions.registry import sampling_rows as SR

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "data", "topp_split_fixture")


def _tables_ok() -> bool:
    try:
        from verity.ml.definitions.registry.prims import DivFullRcp
        DivFullRcp.evaluate(SP.f32_bits(0.8))
        return True
    except Exception:  # noqa: BLE001
        return False


needs_tables = pytest.mark.skipif(not _tables_ok(), reason="MUFU tables (rms_relation.tables) unavailable: div.full model needs them")


def f32b(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _is_nan(w: int) -> bool:
    return (w & 0x7F800000) == 0x7F800000 and (w & 0x7FFFFF) != 0


SPECIALS = np.array([0, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x7F800000,
                     0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x2FFFFFFF, 0x33800000, 0x3F000000, 0x3EFFFFFF, 0x3F7FFFFF, 0xBF7FFFFF,
                     0xBF000000, 0xB0000000, 0x7FC00000, 0x7F8CFC76], dtype=np.uint32)


def _words(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.integers(0, 1 << 32, size=n, dtype=np.uint64).astype(np.uint32), SPECIALS])


def _pairs(n: int = 30000):
    a = _words(n, 1)
    rng = np.random.default_rng(2)
    b = rng.integers(0, 1 << 32, size=a.size, dtype=np.uint64).astype(np.uint32)
    b[: a.size // 2] = (a[: a.size // 2] & 0xFF800000) | rng.integers(0, 1 << 23, size=a.size // 2, dtype=np.uint64).astype(np.uint32)
    return a, b


def test_add_rz_ftz_v1_model_equals_the_registered_v1_scalar_and_is_rn_on_every_word_pair():
    """The evaluator REGISTERED in GumbelNoiseLane_v1 (`add_rz_ftz_v1_model`) is RN (its toward-zero step is a no-op)
    and the twin reproduces it word for word -- both asserted so a change of the v1 function is noticed (v1 records depend on it)."""
    a, b = _pairs()
    v = SR.add_rz_ftz_v1_model(a, b)
    for x, y, w in zip(a.tolist(), b.tolist(), v.tolist()):
        ref = sampling_twins.add_rz_ftz_v1_model(x, y)
        if _is_nan(ref):
            assert _is_nan(w)
        else:
            assert w == ref, (hex(x), hex(y), hex(w), hex(ref))
        rn = SP._ftz(SP._u(SP._f(SP._ftz(x)) + SP._f(SP._ftz(y))))
        assert ref == rn or (_is_nan(ref) and _is_nan(rn)), "the v1 model stopped being RN: a v1 record's evaluator changed"


def _rz_ref(x: int, y: int) -> int:
    """Independent toward-zero reference: the exact sum as a Fraction, truncated to fp32 by bisection on the word."""
    from fractions import Fraction
    fx, fy = SP._f(SP._ftz(x)), SP._f(SP._ftz(y))
    if any(map(lambda t: t != t, (fx, fy))) or any(map(lambda t: t in (float("inf"), float("-inf")), (fx, fy))):
        return SP._ftz(SP._u(fx + fy))
    ex = Fraction(fx) + Fraction(fy)
    if ex == 0:
        return SP._ftz(SP._u(fx + fy))
    sign = 0x80000000 if ex < 0 else 0
    m = abs(ex)
    lo, hi = 0, 0x7F7FFFFF                                       # the largest finite word: RZ never overflows
    while lo < hi:                                               # the largest magnitude word whose value is <= m
        mid = (lo + hi + 1) // 2
        if Fraction(SP._f(mid)) <= m:
            lo = mid
        else:
            hi = mid - 1
    return SP._ftz(sign | lo)


def test_add_rz_ftz_is_true_toward_zero_against_an_exact_fraction_reference_and_differs_from_the_rn_model_where_it_must():
    """The true toward-zero add: `add_rz_ftz` (scalar and twin) equals an exact-arithmetic toward-zero reference on every pair, and
    differs from the RN model on a large share of inexact sums (21.5 % of random pairs) -- the branch is alive."""
    a, b = _pairs(6000)
    v = SR.add_rz_ftz(a, b)
    n_diff = 0
    for x, y, w in zip(a.tolist(), b.tolist(), v.tolist()):
        s = SP.add_rz_ftz(x, y)
        ref = _rz_ref(x, y)
        if _is_nan(ref):
            assert _is_nan(s) and _is_nan(w)
            continue
        assert s == ref, ("scalar", hex(x), hex(y), hex(s), hex(ref))
        assert w == ref, ("twin", hex(x), hex(y), hex(w), hex(ref))
        if s != sampling_twins.add_rz_ftz_v1_model(x, y):
            n_diff += 1
    assert n_diff > 0.1 * a.size, n_diff
    # the construction of the finding: RN carries 1 - 2^-26 up to 1.0; RZ stays one ulp under
    assert sampling_twins.add_rz_ftz_v1_model(0xB2800000, SP.ONE_BITS) == 0x3F800000
    assert SP.add_rz_ftz(0xB2800000, SP.ONE_BITS) == 0x3F7FFFFF and int(SR.add_rz_ftz(np.array([0xB2800000], np.uint32), np.uint32(SP.ONE_BITS))[0]) == 0x3F7FFFFF
    # the RN model is what its name says it is not: RN on every pair
    for x, y in zip(a.tolist()[:2000], b.tolist()[:2000]):
        rn = SP._ftz(SP._u(SP._f(SP._ftz(x)) + SP._f(SP._ftz(y))))
        assert sampling_twins.add_rz_ftz_v1_model(x, y) == rn or (_is_nan(rn) and _is_nan(sampling_twins.add_rz_ftz_v1_model(x, y))), (hex(x), hex(y))
    # saturation and signed zero
    assert SP.add_rz_ftz(0x7F7FFFFF, 0x7F7FFFFF) == 0x7F7FFFFF and SP.add_rz_ftz(0xFF7FFFFF, 0xFF7FFFFF) == 0xFF7FFFFF
    assert SP.add_rz_ftz(0x3F800000, 0xBF800000) == 0 and SP.add_rz_ftz(0x80000000, 0x80000000) == 0x80000000


# The exhaustive fact: inside `__nv_log1pf` the `__fadd_rz(a, 1)` word feeds ONLY the binade selector
# e = (s - 0x3F400000) & -2^23, so the RN model and the true RZ can change the output only where RN(1 + a) lands exactly on a selector
# boundary 0.75 * 2^k that RZ stays under.  Scanning every fp32 word (test below, opt-in; 2^32 words in ~100 s vectorised):
#   a in (-1, 0)   [the Gumbel lane's whole domain, a = -u]: the SUM differs on 939,524,097 words, e on ONE (0xBE800001), the log1pf
#                  word on NONE;
#   a in (0, +inf): e differs on 4 words, the log1pf word on exactly ONE: a = 0x3EFFFFFF (+0.49999997) -> RN model 0x3ECF9920, RZ 0x3ECF991F
#                  (true log1p = 0.4054650882; the RZ word is the nearer one -- libdevice's own result);
#   a <= -1: e differs on 104 words, the log1pf word (NaN / -inf) on none.
# So GumbelNoiseLane_v1 is the same function under the fix and the evaluator was not moved off its record; the registered semantics
# never changed and no version bump is owed.
LOG1PF_ONLY_DIFFERING_WORD = 0x3EFFFFFF
LOG1PF_E_DIFFERING_WORD_IN_LANE_DOMAIN = 0xBE800001


def _e_of(s: int) -> int:
    return (SP._i32(s) - 1061158912) & -8388608


def test_nv_log1pf_true_rz_vs_the_rn_model_the_one_differing_word_is_positive_and_the_lane_domain_has_none():
    """What the scan established: the constructed pair where RN and RZ give DIFFERENT log1pf
    words is a = +0.49999997 (0x3EFFFFFF) -- RN(1 + a) = 1.5 exactly (the 0.75 * 2 selector boundary), RZ = 1.5 - ulp -- and it is the
    only such word in fp32; a Gumbel draw never forms it (the lane feeds -u).  In the lane's domain the single word where the
    selector itself moves (0xBE800001: RN(1 + a) = 0.75 exactly) yields the same log1pf word on both paths."""
    a = LOG1PF_ONLY_DIFFERING_WORD
    rz, rn = SP.nv_log1pf(a), sampling_twins.nv_log1pf_v1_model(a)
    assert (rz, rn) == (0x3ECF991F, 0x3ECF9920), (hex(rz), hex(rn))
    assert sampling_twins.add_rz_ftz_v1_model(a, SP.ONE_BITS) == 0x3FC00000 and SP.add_rz_ftz(a, SP.ONE_BITS) == 0x3FBFFFFF
    assert _e_of(0x3FC00000) != _e_of(0x3FBFFFFF)
    exact = float(np.log1p(np.float64(SP._f(a))))
    assert abs(SP._f(rz) - exact) < abs(SP._f(rn) - exact)                 # the true-RZ word is libdevice's (nearer) answer
    arr = np.array([a], np.uint32)
    assert int(SR.nv_log1pf(arr)[0]) == rz and int(SR.nv_log1pf_v1_model(arr)[0]) == rn
    # the lane domain: the selector moves at exactly one word and the output does not
    b = LOG1PF_E_DIFFERING_WORD_IN_LANE_DOMAIN
    s_rn, s_rz = sampling_twins.add_rz_ftz_v1_model(b, SP.ONE_BITS), SP.add_rz_ftz(b, SP.ONE_BITS)
    assert (s_rn, s_rz) == (0x3F400000, 0x3F3FFFFF) and _e_of(s_rn) != _e_of(s_rz)
    assert SP.nv_log1pf(b) == sampling_twins.nv_log1pf_v1_model(b) == 0xBE934B12
    # every selector boundary in the lane's domain (RN(1 + a) = 0.75 * 2^k or 2^k, k <= 0): the candidates, enumerated, all agree
    cands = []
    for k in range(0, -30, -1):
        for target in (0.75 * 2.0 ** k, 2.0 ** k):
            w = SP._u(target - 1.0)
            cands += [((w & 0x7FFFFFFF) + d) | 0x80000000 for d in (-2, -1, 0, 1, 2) if (w & 0x7FFFFFFF) + d > 0]
    cands = sorted(set(c for c in cands if 0x80000000 < c < 0xBF800000))
    assert len(cands) > 200 and b in cands
    e_moved = [c for c in cands if _e_of(sampling_twins.add_rz_ftz_v1_model(c, SP.ONE_BITS)) != _e_of(SP.add_rz_ftz(c, SP.ONE_BITS))]
    assert e_moved == [b], [hex(c) for c in e_moved]
    for c in cands:
        assert SP.nv_log1pf(c) == sampling_twins.nv_log1pf_v1_model(c), hex(c)
    arr = np.array(cands, np.uint32)
    assert int((SR.nv_log1pf(arr) != SR.nv_log1pf_v1_model(arr)).sum()) == 0
    # the finding's own construction (u = 2^-26): the sums differ, the selector and the word do not
    assert sampling_twins.add_rz_ftz_v1_model(0xB2800000, SP.ONE_BITS) != SP.add_rz_ftz(0xB2800000, SP.ONE_BITS)
    assert SP.nv_log1pf(0xB2800000) == sampling_twins.nv_log1pf_v1_model(0xB2800000)


@pytest.mark.skipif(os.environ.get("VERITY_EXHAUSTIVE") != "1", reason="2^32-word scan (~100 s vectorised): VERITY_EXHAUSTIVE=1")
def test_nv_log1pf_exhaustive_over_every_fp32_word_the_rn_model_and_the_true_rz_differ_on_exactly_one_word():
    """The proof behind the registered-semantics claim, run on demand: over all 2^32 words `nv_log1pf` (true RZ) and
    `nv_log1pf_v1_model` differ on exactly one word (0x3EFFFFFF), and on none in the lane's domain (-1, 0)."""
    one = np.uint32(SP.ONE_BITS)
    CH = 1 << 20
    diff_words, e_lane = [], []
    for lo, hi in ((0x00000001, 0x7F800000), (0x80000001, 0xBF800000), (0xBF800000, 0xFF800001)):
        for start in range(lo, hi, CH):
            w = np.arange(start, min(start + CH, hi), dtype=np.int64).astype(np.uint32)
            l1, l2 = SR.nv_log1pf(w), SR.nv_log1pf_v1_model(w)
            d = l1 != l2
            if d.any():
                diff_words += w[d].tolist()
            if lo == 0x80000001:
                s1, s2 = SR.add_rz_ftz_v1_model(w, one), SR.add_rz_ftz(w, one)
                em = ((SR._i32_of_u32(s1) - 1061158912) & -8388608) != ((SR._i32_of_u32(s2) - 1061158912) & -8388608)
                if em.any():
                    e_lane += w[em].tolist()
    assert diff_words == [LOG1PF_ONLY_DIFFERING_WORD], [hex(x) for x in diff_words[:8]]
    assert e_lane == [LOG1PF_E_DIFFERING_WORD_IN_LANE_DOMAIN], [hex(x) for x in e_lane[:8]]


def test_nv_logf_and_nv_log1pf_are_word_equal_to_the_libdevice_transcriptions_on_every_non_nan_word():
    a = _words(40000, 3)
    vl, vl1, vl2 = SR.nv_logf(a), SR.nv_log1pf(a), SR.nv_log1pf_v1_model(a)
    for x, l, l1, l2 in zip(a.tolist(), vl.tolist(), vl1.tolist(), vl2.tolist()):
        rl, rl1, rl2 = SP.nv_logf(x), SP.nv_log1pf(x), sampling_twins.nv_log1pf_v1_model(x)
        if _is_nan(x):
            assert _is_nan(l) == _is_nan(rl) and _is_nan(l1) == _is_nan(rl1)   # NaN payloads are not modelled (unreachable in the chain)
        else:
            assert l == rl, ("nv_logf", hex(x), hex(l), hex(rl))
            assert l1 == rl1, ("nv_log1pf", hex(x), hex(l1), hex(rl1))
            assert l2 == rl2, ("nv_log1pf v1 model", hex(x), hex(l2), hex(rl2))


def test_the_logs_outside_their_domain_are_one_pinned_nan_on_every_path():
    """`__nv_logf` of a negative word and `__nv_log1pf` of a word below -1 (-inf included) are libdevice's invalid -inf + inf:
    `sampling.LOG_INVALID_NAN` on the scalar lane, the rows twin and the float32 transcription alike, whatever the host's default
    NaN. -0 and the negative subnormals (flushed to -0) keep logf's -inf, and log1pf keeps -0 at -0 and -inf at -1."""
    from verity_vllm.program import libdevice_sampling as L

    rng = np.random.default_rng(5)
    edges = [0x80000000, 0x80000001, 0x807FFFFF, 0x80800000, 0xBF7FFFFF, 0xBF800000, 0xBF800001, 0xFF7FFFFF, 0xFF800000]
    a = np.concatenate([rng.integers(0x80000001, 0xFF800001, size=20000, dtype=np.uint64).astype(np.uint32),
                        np.array(edges, dtype=np.uint32)])
    tr = {"nv_logf": L.nv_logf(a.view(np.float32)).view(np.uint32), "nv_log1pf": L.nv_log1pf(a.view(np.float32)).view(np.uint32)}
    scalar = {"nv_logf": SP.nv_logf, "nv_log1pf": SP.nv_log1pf, "nv_log1pf_v1_model": sampling_twins.nv_log1pf_v1_model}
    for name, domain_ok in (("nv_logf", lambda x: (x & 0x7FFFFFFF) < 0x00800000), ("nv_log1pf", lambda x: x <= 0xBF800000),
                            ("nv_log1pf_v1_model", lambda x: x <= 0xBF800000)):
        rows = getattr(SR, name)(a)
        for i, x in enumerate(a.tolist()):
            want = scalar[name](x)
            assert int(rows[i]) == want, (name, hex(x), hex(int(rows[i])), hex(want))
            if name in tr:
                assert int(tr[name][i]) == want, (name, "transcription", hex(x), hex(int(tr[name][i])), hex(want))
            assert domain_ok(x) or want == SP.LOG_INVALID_NAN, (name, hex(x), hex(want))
    assert (SP.nv_logf(0x80000000), SP.nv_logf(0x807FFFFF), SP.nv_log1pf(0x80000000), SP.nv_log1pf(0xBF800000)) == \
        (0xFF800000, 0xFF800000, 0x80000000, 0xFF800000)


#: the probe's edge words (`PROBE`: the L40S and the H100, every FTZ build alike): input -> the GPU's output
PROBE_EDGES = {"nv_logf": {0xBF800000: 0x7FFFFFFF, 0x80000000: 0xFF800000, 0x80400000: 0xFF800000, 0x80000001: 0xFF800000,
                           0x807FFFFF: 0xFF800000, 0xFF7FFFFF: 0x7FFFFFFF, 0xFF800000: 0x7FFFFFFF},
               "nv_log1pf": {0xC0000000: 0x7FFFFFFF, 0xBF800000: 0xFF800000, 0xBF800001: 0x7FFFFFFF, 0xFF800000: 0x7FFFFFFF,
                             0xFF7FFFFF: 0x7FFFFFFF}}
PROBE = "art:e5dd6ce9fa74e7ca4538b848943d8bb555f2adac528277848395dbf4cdff66a5"


def _gpu_paths(name: str) -> list:
    """The `_v2` logs: the scalar function and, for logf, the registered `NvLogf_v2`."""
    from verity.ml.definitions.registry import topp_word_gates as TWG
    return [getattr(SP, f"{name}_gpu")] + ([TWG.NvLogfV2.evaluate] if name == "nv_logf" else [])


@pytest.mark.parametrize("name", ["nv_logf", "nv_log1pf"])
def test_the_v2_logs_are_the_gpus_words_on_the_probes_edges(name):
    for f in _gpu_paths(name):
        got = [f(x) for x in PROBE_EDGES[name]]
        assert got == list(PROBE_EDGES[name].values()), (name, f, [hex(g) for g in got])


@pytest.mark.slow
def test_the_v2_logs_are_the_gpus_words_on_every_probed_input(tmp_path):
    """Every input word of the probe (20,007 for logf, 20,005 for log1pf: the edges, then random negatives and NaNs), against each FTZ
    build's output on the L40S and the H100.  Skipped where the evidence store is unreachable."""
    import glob

    from tests.regression import store_io
    if os.environ.get("VERITY_SKIP_STORE") == "1" or not store_io.reachable():
        pytest.skip("the evidence store is not reachable here")
    root = store_io.fetch(PROBE, to=tmp_path / "probe")
    runs = sorted(glob.glob(str(root / "*" / "out-*-ftz-true-*.json")))
    assert len(runs) == 4
    for path in runs:
        d = json.load(open(path))
        for name, key in (("nv_logf", "logf"), ("nv_log1pf", "log1pf")):
            pairs = [(int(x, 16), int(w, 16)) for x, w in zip(d[key]["in"], d[key]["out"])]
            for f in _gpu_paths(name):
                bad = [(hex(x), hex(f(x)), hex(w)) for x, w in pairs if f(x) != w]
                assert not bad, (path, name, len(bad), bad[:4])


def test_v2_is_v1_but_for_nan_results_and_the_noise_lane_is_the_same_function():
    """`_v2` differs from `_v1` only where `_v1` gives a NaN (outside the domain, a NaN input); the noise never gets there, so
    `GumbelNoiseLane_v2` is `GumbelNoiseLane_v1` on every lane (sampled here)."""
    rng = np.random.default_rng(3)
    edges = np.array([0x7FC00123, 0xFFC00123, 0x7F800000, 0], dtype=np.uint64)
    words = np.concatenate([rng.integers(0, 1 << 32, 20000, dtype=np.uint64), edges])
    for v1, v2 in ((SP.nv_logf, SP.nv_logf_gpu), (SP.nv_log1pf, SP.nv_log1pf_gpu)):
        for x in words.tolist():
            a, b = v1(x), v2(x)
            assert b == (SP.LOG_NAN if (a & 0x7F800000) == 0x7F800000 and a & 0x7FFFFF else a), (v1.__name__, hex(x))
    assert SP.GumbelNoiseLaneV2.id == "GumbelNoiseLane_v2"
    for key in (0, 942961759, 0xFFFFFFFF):
        for v in list(range(512)) + [128255, 0x7FFFFFFF]:
            assert SP.GumbelNoiseLaneV2.evaluate(key, v) == SP.GumbelNoiseLane.evaluate(key, v)


def test_philox_stream_key_and_the_noise_row_match_the_scalar_registry_lane_for_lane():
    for seed, pos in ((942961759, 7), (1, 1023), ((1 << 63) - 1, (1 << 40) + 5), (-5 & ((1 << 64) - 1), 0), (509133774, 107)):
        k = SP.stream_key(seed, pos)
        kv = int(SR.philox_word0(np.array([pos & 0xFFFFFFFF], dtype=np.uint64), (pos >> 32) & 0xFFFFFFFF, 0, 0,
                                 seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)[0])
        assert k == kv
    for key in (0, 1, 942961759, 0xFFFFFFFF, 7777777):
        V = 4096
        nr = SR.noise_row(key, V)
        ref = np.array([SP.noise_lane(key, v) for v in range(V)], dtype=np.uint32)
        assert nr.dtype == np.uint32 and nr.shape == (V,) and int((nr != ref).sum()) == 0, key
        # the noise lane under the earlier RN model of record is the same lane: no word moved (the exhaustive fact, sampled here)
        u = SR.rand_row(key, V)
        l_rn = SR.nv_log1pf_v1_model(SR._u32(u) ^ np.uint32(0x80000000))
        g_rn = SR.nv_logf(l_rn ^ np.uint32(0x80000000)) ^ np.uint32(0x80000000)
        assert int((nr != g_rn).sum()) == 0, key


def test_the_registry_keeps_one_gumbel_chain_at_version_1_and_its_evaluator_is_the_true_rz_lane():
    """No version bump: GumbelNoiseLane_v1's word map did not move (exhaustive), so the chain stays at v1 with one evaluator; the
    witness functions are not registered and not exported as evaluators."""
    from verity.ml.circuits.defs import bind
    assert SP.GumbelNoiseLane.id == "GumbelNoiseLane_v1" and SP.GumbelSelectF32.id == "GumbelSelectF32_v1"
    assert SP.GumbelTokenSelect.id == "GumbelTokenSelect_v1" and SP.GumbelTopPTokenSelect.id == "GumbelTopPTokenSelect_v1"
    assert [n for n in SP.__all__ if n.endswith(("V2", "_v2"))] == ["GumbelNoiseLaneV2"], SP.__all__   # the GPU-NaN logs' lane
    assert SP.GumbelNoiseLane.evaluate(942961759, 17) == SP.noise_lane(942961759, 17)
    assert bind(SP.GumbelSelectF32, V=64).id.startswith("GumbelSelectF32_v1")
    # a lane whose draw is the finding's boundary draw (u < 2^-25, 0.9 % of a 2^16-lane row is NOT expected: ~3e-8 per lane) still
    # evaluates identically under both adds -- searched, and asserted when found
    for key in range(200):
        u = SR.DR.as_f32(SR.rand_row(key, 1 << 16))
        idx = np.nonzero(u < 2.0 ** -25)[0]
        if idx.size:
            v = int(idx[0])
            neg = SR._u32(SR.rand_row(key, 1 << 16))[v] ^ np.uint32(0x80000000)
            assert int(SR.nv_log1pf(np.array([neg]))[0]) == int(SR.nv_log1pf_v1_model(np.array([neg]))[0])
            break


def _cases(V: int, n: int, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        logits = (rng.standard_normal(V) * rng.choice([1.0, 3.0, 8.0])).astype(np.float32)
        if i % 5 == 1:
            logits[rng.integers(0, V, size=V // 40)] = -np.inf
        if i % 5 == 2:
            logits[:] = np.round(logits * 2) / 2                    # plateaus: pivot ties and argmax ties
        if i % 5 == 3:
            logits[rng.integers(0, V)] = np.nan
        if i % 5 == 4:
            logits[:] = 0.0                                          # uniform row (the pipeline's nomask outcome)
        bf = (logits.view(np.uint32) >> 16).astype(np.uint16)
        temp = [0.8, 1.0, 0.0, 0.5, 0.7][i % 5]
        p = [0.95, 1.0, 0.1, 0.5, 0.99][(i // 5) % 5]
        S = [16, 32, 1, 8, 2, 4][i % 6]
        out.append((bf, temp, p, S, int(rng.integers(0, 1 << 63)), int(rng.integers(0, 1200))))
    return out


@needs_tables
@pytest.mark.slow
def test_the_whole_row_equals_the_lane_walk_evaluate_spec_over_temps_top_p_split_counts_ties_masks_and_nan():
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.definitions.registry import sampling_topp_difftest as D
    V = 1024
    for bf, temp, p, S, seed, pos in _cases(V, 30, 11):
        words = [bf.astype(np.uint64), np.array([f32b(p)], dtype=np.uint64), np.array([seed], dtype=np.uint64),
                 np.array([pos], dtype=np.uint64), np.array([f32b(temp)], dtype=np.uint64)]
        ref = evaluate_spec(D.spec({"V": V, "S": S}), words)[0]
        out = SR.gumbel_topp_token_select_row(bf, f32b(p), seed, pos, f32b(temp), S)
        assert out["token"] == ref, (temp, p, S, seed, pos, out["token"], ref)
        assert out["scaled"].shape == (V,) and out["masked"].shape == (V,) and out["noisy"].shape == (V,)


@needs_tables
@pytest.mark.slow
def test_the_r16_gpu_fixture_rows_device_token_masked_row_and_lane_walk_all_agree_with_the_vectorised_row():
    """Three real rows of the llama B8 stochastic record (`fstoch01_rows.npz`, an L40S, vLLM d9105ea80): the device's served token,
    the kernel-consumed masked row, seed / pos / temperature / top_p as the kernel read them.  The vectorised row reproduces the
    served token AND every lane of the masked row; the lane walk is run on one of them (10 s) as the reference beside it."""
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.definitions.registry import sampling_topp_difftest as D
    z = np.load(os.path.join(FIX, "fstoch01_rows.npz"))
    meta = json.load(open(os.path.join(FIX, "fstoch01_rows.npz.json")))
    first = True
    for k, v in meta.items():
        bf = z[f"{k}_bf16_logits_bits"]
        masked_dev = z[f"{k}_masked_f32_bits"]
        d = v["device"]
        out = SR.gumbel_topp_token_select_row(bf, f32b(d["top_p"]), d["seed"], d["pos"], f32b(d["temperature"]), v["S"])
        assert out["token"] == v["served"], (k, out["token"], v["served"])
        assert int((out["masked"] == masked_dev).sum()) == v["V"], k
        if first:
            words = [bf.astype(np.uint64), np.array([f32b(d["top_p"])], dtype=np.uint64), np.array([d["seed"]], dtype=np.uint64),
                     np.array([d["pos"]], dtype=np.uint64), np.array([f32b(d["temperature"])], dtype=np.uint64)]
            assert evaluate_spec(D.spec({"V": v["V"], "S": v["S"]}), words)[0] == v["served"]
            first = False


def test_the_select_is_the_strict_first_maximum_scan_nan_never_wins_and_a_nan_lane_zero_pins_the_result():
    x = np.array([f32b(1.0), f32b(3.0), f32b(3.0), f32b(2.0)], dtype=np.uint32)
    tok, y = SR.gumbel_select_row(x, 0, 0, 0)                                    # temp 0: no noise, first max
    assert tok == 1 and (y == x).all()
    x_nan = x.copy(); x_nan[1] = 0x7FC00000
    assert SR.gumbel_select_row(x_nan, 0, 0, 0)[0] == 2                          # the NaN lane never wins
    x_nan0 = x.copy(); x_nan0[0] = 0x7FC00000
    assert SR.gumbel_select_row(x_nan0, 0, 0, 0)[0] == 0                         # nothing is > NaN: the scan keeps lane 0
    neg = np.full(4, SP.NEG_INF_BITS, dtype=np.uint32)
    assert SR.gumbel_select_row(neg, 0, 0, 0)[0] == 0                            # all -inf: lane 0
    assert (SR.topp_mask_row(x, f32b(0.9), 3) == SP.NEG_INF_BITS).all()           # S outside the split counts keeps no lane, as the primitive
