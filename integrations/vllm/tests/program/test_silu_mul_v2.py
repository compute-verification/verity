"""`SiluMul_v2` (`registry.act`, promoted from quarantine `act`) is vLLM `_C.silu_and_mul` on the words the red-team measured on
sm_120 where `SiluMul_v1` is not (r20260930-092038-7dbc, `silu-edge-cases`), and `SiluMul_v1` on every other word. The served
vocabulary `b1-eager-v3` binds it; the older vocabularies keep `SiluMul_v1`. The replay's row kernel is the same function."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind
from verity_vllm.program.frontend.rules.vocab import PROFILES
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.registry import b1, prims as P
from verity_vllm.program.registry.act import SiluMul, SiluMulBf16
from verity_vllm.program.registry.quarantine.act import silu_mul_difftest as DT

V1 = P.SiluMulBf16.evaluate
V2 = SiluMulBf16.evaluate
NAN = 0x7FFF
ONE, NEG_ONE, POS_ZERO, NEG_ZERO, POS_INF, NEG_INF = 0x3F80, 0xBF80, 0x0000, 0x8000, 0x7F80, 0xFF80


def _f(w: int) -> float:
    return float(np.array([w << 16], dtype=np.uint32).view(np.float32)[0])


def test_the_red_team_reproducer():
    assert V1(ONE, NEG_ZERO) == NEG_ZERO
    assert V2(ONE, NEG_ZERO) == POS_ZERO


@pytest.mark.parametrize("u", [POS_ZERO, NEG_ZERO, ONE, NEG_ONE, POS_INF, 0x7FC0])
def test_nan_words_are_0x7fff(u):
    assert V2(0x7FC0, u) == NAN and V2(0xFFC1, u) == NAN
    assert V2(NEG_INF, u) == NAN
    assert V2(ONE, 0x7FC0) == NAN


def test_gate_minus_inf_is_nan_and_inf_times_zero_is_nan():
    assert V1(NEG_INF, ONE) == NEG_ZERO and V2(NEG_INF, ONE) == NAN
    assert V2(POS_INF, POS_ZERO) == NAN and V2(POS_INF, ONE) == POS_INF


def test_gates_past_expf_overflow_are_signed_zero():
    past = [g for g in range(0x8000, 0xFF80) if _f(g) <= -89.0]
    assert past and all(V2(g, ONE) == NEG_ZERO and V2(g, NEG_ONE) == POS_ZERO for g in past)
    assert any(V1(g, ONE) != NEG_ZERO for g in past)
    before = [g for g in range(0x8000, 0xFF80) if -89.0 < _f(g) <= -80.0]
    assert all(V2(g, ONE) == V1(g, ONE) != NEG_ZERO for g in before)


def test_gate_minus_zero_keeps_its_sign():
    assert V2(NEG_ZERO, ONE) == NEG_ZERO and V2(NEG_ZERO, NEG_ONE) == POS_ZERO
    assert V2(POS_ZERO, ONE) == POS_ZERO and V1(NEG_ZERO, ONE) == POS_ZERO


def test_v1_on_every_other_gate_word():
    ups = (ONE, NEG_ONE, 0x4049, 0x0003)
    for g in range(1 << 16):
        f = _f(g)
        if f != f or f == float("-inf") or g == NEG_ZERO or f <= -89.0:
            continue
        for u in ups:
            want = V1(g, u)
            assert V2(g, u) == (NAN if want == P.BF16_NAN_TORCH else want), (hex(g), hex(u))


def test_the_row_is_the_element_per_lane():
    I = 7
    gates = [ONE, NEG_ZERO, NEG_INF, 0xC2C8, 0x4120, 0x7FC0, 0xC0A0]
    ups = [NEG_ZERO, ONE, ONE, ONE, NEG_ZERO, ONE, 0x4049]
    got = [int(v) for v in evaluate(bind(SiluMul, I=I), gates + ups)]
    assert got == [V2(g, u) for g, u in zip(gates, ups)]


def test_the_served_vocabulary_binds_v2():
    got = {name: vocab.kind("silu_mul", I=8).id for name, vocab in PROFILES.items()}
    v1, v2 = bind(b1.SiluMul, I=8).id, bind(SiluMul, I=8).id
    assert got == {"b1-eager": v1, "b1-eager-v2": v1, "b1-eager-v3": v2, "dense2-eager-v1": v2, None: v1}
    assert PROFILES[TargetProfile().numerics_profile].kind("silu_mul", I=8).id == v2 == "SiluMul_v2{I=8}"


def test_the_served_kind_has_its_boolean_version():
    from verity_vllm.program import boolean as BB

    assert BB.boolean_version(bind(SiluMul, I=8)).id == "SiluMul_v4{I=8}"
    assert BB.boolean_version(SiluMulBf16).id == "SiluMulBf16_v4"
    assert BB.boolean_version(bind(b1.SiluMul, I=8)).id == "SiluMul_v3{I=8}"


def test_the_replay_row_is_v2_past_expf_overflow():
    """The served row's kernel on the gates past expf's overflow (bf16 -89 .. -103), against both signs of up: the GPU's signed
    zero (-0 for up > 0, +0 for up < 0). `SiluMul_v1`'s row gives a nonzero word on -97 .. -89, where its tiny `g * e^g` is still
    a bf16 subnormal."""
    pairs = [(g, u) for g in range(0x8000, 0xFF80) if -103.0 <= _f(g) <= -89.0 for u in (ONE, NEG_ONE, 0x4183, 0xC183)]
    gu = np.array([g for g, _ in pairs] + [u for _, u in pairs], dtype=np.uint16)
    got = RK.ROWS["SiluMul_v2"].fn({"I": len(pairs)}, [gu])["out"]
    old = RK.ROWS["SiluMul_v1"].fn({"I": len(pairs)}, [gu])["out"]
    want = [NEG_ZERO if u & 0x8000 == 0 else POS_ZERO for _, u in pairs]
    assert [int(w) for w in got] == want == [V2(g, u) for g, u in pairs]
    band = [k for k, (g, _) in enumerate(pairs) if _f(g) >= -97.0]
    assert len(band) == 4 * 17 and all(int(old[k]) not in (POS_ZERO, NEG_ZERO) for k in band)
    assert RK.EVALUATORS["SiluMul_v2"].startswith("verity_vllm.program.kernels.act_rows.silu_mul_expf_rows")


def test_the_replay_row_is_the_element_on_every_gate_word():
    ups = (POS_ZERO, NEG_ZERO, ONE, NEG_ONE, POS_INF, NEG_INF, 0x7FC0, 0x0001, 0x4049)
    for u in ups:
        gu = np.concatenate([np.arange(1 << 16, dtype=np.uint16), np.full(1 << 16, u, dtype=np.uint16)])
        got = RK.ROWS["SiluMul_v2"].fn({"I": 1 << 16}, [gu])["out"]
        assert [int(w) for w in got] == [V2(g, u) for g in range(1 << 16)], hex(u)


def test_the_adapter_covers_every_edge_kind_and_the_edge_words():
    from verity_vllm.properties.admission import REQUIRED_EDGE_KINDS

    cases = [DT.cases(None, i) for i in range(60)]
    kinds = {s.get("kind") for c in cases for s in c["inputs"]}
    assert set(REQUIRED_EDGE_KINDS) <= kinds
    words = [c for c in cases if "words" in c["inputs"][0]]
    assert words and all(len(c["inputs"][0]["words"]) == 2 * c["statics"]["I"] for c in words)
    pairs = {(w["words"][k], w["words"][len(w["words"]) // 2 + k]) for c in words for w in c["inputs"] for k in range(len(w["words"]) // 2)}
    assert {(g, u) for g in DT.GATES for u in DT.UPS} <= pairs
