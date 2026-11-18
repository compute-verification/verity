"""`cpp/tc_model.cpp`'s tensor-core step held to `verity.circuits.silicon` directly, for the Ampere and the Hopper parameters."""

from __future__ import annotations

import ctypes
from functools import lru_cache

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.silicon.errors import InvalidArtifact
from verity.circuits.silicon import models as S
from verity.circuits.silicon import total as ST
from verity_vllm.definitions.numerics import cpu_model, derived_rows, fa3_model, prims_twins
from verity_vllm.definitions.numerics import kernel_registry as KR
from verity_vllm.definitions.numerics import twins as TW
from verity_vllm.definitions.registry import prims as P
from verity_vllm.definitions.registry.hopper import HopperBF16WgmmaDot16

PIPELINES = {"ampere": S.AMPERE_BF16_M16N8K16, "hopper": S.HOPPER_BF16_WGMMA_K16}
PRIMS = {"ampere": P.AmpereBF16TcDot16.evaluate, "hopper": HopperBF16WgmmaDot16.evaluate}
PORTS = {"ampere": derived_rows.tc_dot16, "hopper": derived_rows.tc_dot16_hopper}
NAN32 = 0x7FFFFFFF
Z16 = [0] * 16
ONE = 0x3F80
MINIMAL = {
    "nan_accumulator_zero_operands": (0x7FC00000, Z16, Z16),
    "inf_operand_times_zero": (0, [0x7F80] + Z16[1:], Z16),
    "nan_operand_times_one": (0, [0x7FC0] + Z16[1:], [ONE] + Z16[1:]),
    "inf_and_minus_inf_operands_times_one": (0, [0x7F80, 0xFF80] + Z16[2:], [ONE, ONE] + Z16[2:]),
}


def _cpp(m: S.GroupSum, acc, a, b) -> np.ndarray:
    acc = np.ascontiguousarray(acc, dtype=np.uint32)
    a = np.ascontiguousarray(a, dtype=np.uint16).reshape(-1, 16)
    b = np.ascontiguousarray(b, dtype=np.uint16).reshape(-1, 16)
    d = np.empty(acc.size, dtype=np.uint32)
    groups = (ctypes.c_int * len(m.groups))(*m.groups)
    cpu_model.lib().tc_records(0, acc.ctypes.data, a.ctypes.data, b.ctypes.data, d.ctypes.data, acc.size, groups, len(m.groups),
                               m.width, m.zero_exponent)
    return d


def _bf16(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _lanes(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`test_derived_rows.py`'s "every encoding" enumeration (uniform words, the N(0,1) finite domain over 40 binades, overflow,
    cancellation, subnormal and zero-operand lanes, the overflow boundary) and a seeded spread of 20,000 uniform lanes."""
    rng = np.random.default_rng(seed)
    acc, a, b = [], [], []

    def add(c, x, y):
        acc.append(np.asarray(c, dtype=np.uint32).reshape(-1))
        a.append(np.asarray(x, dtype=np.uint16).reshape(-1, 16))
        b.append(np.asarray(y, dtype=np.uint16).reshape(-1, 16))

    def uniform(n):
        add(rng.integers(0, 2 ** 32, n, dtype=np.uint64), rng.integers(0, 2 ** 16, (n, 16)), rng.integers(0, 2 ** 16, (n, 16)))

    uniform(4000)
    sc = (2.0 ** rng.integers(-20, 20, 4000)).astype(np.float32)[:, None]
    add((rng.standard_normal(4000).astype(np.float32) * sc[:, 0] * 10).view(np.uint32),
        _bf16(rng.standard_normal((4000, 16)).astype(np.float32) * sc), _bf16(rng.standard_normal((4000, 16))))
    big = int(np.float32(3.0e38).view(np.uint32) >> 16)
    for _ in range(500):
        v = rng.standard_normal(16).astype(np.float32)
        add(0, [big] * 16, _bf16(v * 4))
        neg_sum = -np.float32(sum(float(np.uint32(int(x) << 16).view(np.float32)) for x in _bf16(v)))
        add(np.float32(neg_sum).view(np.uint32), _bf16(v), [ONE] * 16)
        add(rng.integers(0, 0x800000), rng.integers(1, 0x80, 16), rng.integers(1, 0x80, 16))
        add(rng.integers(0, 2 ** 32, dtype=np.uint64), Z16, rng.integers(0, 2 ** 16, 16))
    for _ in range(300):
        add(np.float32(rng.standard_normal() * 1e38).view(np.uint32), _bf16(rng.standard_normal(16) * 1e30),
            _bf16(rng.standard_normal(16) * 1e9))
    uniform(20000)
    return np.concatenate(acc), np.concatenate(a), np.concatenate(b)


ACC, A, B = _lanes(2026)
NONFINITE_OPERAND = TW._nonfinite16(A).any(axis=1) | TW._nonfinite16(B).any(axis=1)
NONFINITE_ACC = (ACC & 0x7F800000) == 0x7F800000
FINITE = ~(NONFINITE_OPERAND | NONFINITE_ACC)


def _total(m: S.GroupSum, i: int) -> int:
    return ST.tc_dot_total(m, int(ACC[i]), [int(v) for v in A[i]], [int(v) for v in B[i]])


@pytest.mark.parametrize("arch", PIPELINES)
def test_cpp_step_equals_group_sum_and_silicon_total_on_every_finite_lane(arch):
    m = PIPELINES[arch]
    got = _cpp(m, ACC, A, B)
    sticky = finite = 0
    for i in np.nonzero(FINITE)[0]:
        want = _total(m, i)
        assert S.tc_dot(m, int(ACC[i]), [int(v) for v in A[i]], [int(v) for v in B[i]]) == want
        assert int(got[i]) == want, (arch, hex(int(ACC[i])), [hex(int(v)) for v in A[i]], [hex(int(v)) for v in B[i]],
                                     hex(int(got[i])), hex(want))
        sticky += (want & 0x7FFFFFFF) == 0x7F800000
        finite += (want & 0x7F800000) != 0x7F800000
    assert sticky > 1000 and finite > 5000, (sticky, finite)


@lru_cache(maxsize=None)
def _twin(base: str, **statics):
    spec = bind(TW.DEFINITIONS[base], **statics)
    return KR.instance(spec, TW.KERNEL), spec.statics


def _gemm_twin_hands_back(monkeypatch, a: list[int], b: list[int]) -> bool:
    handed: list[int] = []
    monkeypatch.setattr(TW, "_gemm_coordinate_gates", lambda x, w: handed.append(1) or 0)
    twin, statics = _twin("Gemm_v1", K=len(a), N=1)
    twin(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64), **statics)
    return handed == [1]


def _attention_twin_declines(a: list[int], b: list[int]) -> bool:
    twin, statics = _twin("Attention_v3", T=1, NH=1, KVH=1, D=len(a), BN=16)
    try:
        twin(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64), np.full(len(a), ONE, dtype=np.uint64), **statics)
    except KR.Declined:
        return True
    return False


@pytest.mark.parametrize("arch", PIPELINES)
def test_no_lane_on_which_the_cpp_step_is_wrong_gets_past_a_twin(arch, monkeypatch):
    """Every lane where the C++ differs from `silicon/total` has a non-finite word. One in an operand: the GEMM twin hands its
    coordinate to the gates and the attention twin refuses it. One in the accumulator only: no twin takes an accumulator from its
    caller (`test_twins_pass_group_sums_parameters_and_no_accumulator`), and a chain's own saturation is a finite lane."""
    m = PIPELINES[arch]
    got = _cpp(m, ACC, A, B)
    wrong = [i for i in np.nonzero(~FINITE)[0] if int(got[i]) != _total(m, i)]
    assert len(wrong) > 2000, len(wrong)
    for i in wrong:
        a, b = [int(v) for v in A[i]], [int(v) for v in B[i]]
        assert NONFINITE_OPERAND[i] or NONFINITE_ACC[i], (hex(int(ACC[i])), a, b)
        if NONFINITE_OPERAND[i]:
            assert _gemm_twin_hands_back(monkeypatch, a, b), (a, b)
            assert _attention_twin_declines(a, b), (a, b)


@pytest.mark.parametrize("case", MINIMAL)
@pytest.mark.parametrize("arch", PIPELINES)
def test_the_minimal_non_finite_lanes(arch, case, monkeypatch):
    """The two silicon models and the registered primitive agree (NaN); `GroupSum.step` refuses; the numpy port flags the lane;
    the C++ has no NaN or infinity rules and gives another word, which no twin lets through."""
    m, (acc, a, b) = PIPELINES[arch], MINIMAL[case]
    want = ST.tc_dot_total(m, acc, a, b)
    assert want == PRIMS[arch](acc, *a, *b) == NAN32
    with pytest.raises(InvalidArtifact):
        S.tc_dot(m, acc, a, b)
    _, flagged = PORTS[arch](np.array([acc], dtype=np.uint32), np.array([a], dtype=np.uint16), np.array([b], dtype=np.uint16))
    assert flagged.tolist() == [True]
    assert int(_cpp(m, [acc], a, b)[0]) != want
    if acc == 0:
        assert _gemm_twin_hands_back(monkeypatch, a, b) and _attention_twin_declines(a, b)


def test_twins_pass_group_sums_parameters_and_no_accumulator(monkeypatch):
    """The GEMM twin's hand-written step parameters are `AMPERE_BF16_M16N8K16`'s, and its chains start at +0."""
    seen: list[tuple] = []
    real = cpu_model.lib()

    class Recorder:
        def gemm_model(self, *args):
            seen.append((list(args[8]), args[9], args[10], args[11], args[4]))
            return real.gemm_model(*args)

    monkeypatch.setattr(TW, "_tc_lib", Recorder)
    twin, statics = _twin("Gemm_v1", K=32, N=2)
    twin(np.full(32, ONE, dtype=np.uint64), np.full(64, ONE, dtype=np.uint64), **statics)
    m = S.AMPERE_BF16_M16N8K16
    assert seen == [(list(m.groups), len(m.groups), m.width, m.zero_exponent, None)]


def _chain_total(m: S.GroupSum, q: np.ndarray, k: np.ndarray) -> int:
    acc = 0
    for s in range(0, q.size, 16):
        acc = ST.tc_dot_total(m, acc, [int(v) for v in q[s:s + 16]], [int(v) for v in k[s:s + 16]])
    return acc


@pytest.mark.parametrize("arch", PIPELINES)
def test_attention_models_compiled_params_are_group_sums(arch):
    """`fa2_rows` (the attention twin's, `Params AMPERE_BF16`) and `fa3_rows` (`Params HOPPER_BF16_WGMMA`) compute each S word as a
    D/16-step chain from +0 with their compiled parameters: equal to `silicon/total` under that architecture's `GroupSum`, overflow
    rows included, and unequal under the other's on some keys, so a swapped or edited constant fails."""
    fa2, tables = prims_twins.fa2_twin()
    rng = np.random.default_rng(7)
    rows, D = 8, 128
    model, BN = (fa2, fa2.KBLOCK_N) if arch == "ampere" else (fa3_model, 128)
    q = _bf16(rng.standard_normal((rows, D)) * (2.0 ** rng.integers(-20, 20, (rows, 1))))
    q[0] = _bf16(rng.standard_normal(D) * 1e30)
    k = _bf16(rng.standard_normal((BN, 1, D)) * 1e9 ** (np.arange(BN) % 2)[:, None, None])
    kw = {} if arch == "ampere" else {"kblock_n": BN}
    res = model.attention_rows(q, k, k, np.zeros(rows, dtype=np.int32), np.full(rows, BN, dtype=np.int32),
                               np.ones(rows, dtype=np.int32), D ** -0.5, tables, debug=True, **kw)
    other = PIPELINES["hopper" if arch == "ampere" else "ampere"]
    differs = sticky = 0
    for r in range(rows):
        for c in range(BN):
            want = _chain_total(PIPELINES[arch], q[r], k[c, 0])
            assert int(res.dbg["s"][r, c]) == want, (arch, r, c, hex(int(res.dbg["s"][r, c])), hex(want))
            differs += _chain_total(other, q[r], k[c, 0]) != want
            sticky += (want & 0x7FFFFFFF) == 0x7F800000
    assert differs > 0 and sticky > 0, (differs, sticky)
