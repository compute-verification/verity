"""`program.predict.class_select`: the builders bind the class's Definitions from its statics alone, with the port names and
the limits row `class_window` states, at GLM-5.3's statics and at a small class."""
from __future__ import annotations

import pytest

from verity.commitments.periods import interface as IF
from verity_vllm.pipeline import boundary_span as BS
from verity_vllm.program.predict import class_select as CS
from verity_vllm.program.predict import class_window as CW

GLM = dict(V=154880, N=7744, O=128, stops=(154820, 154827, 154829), max_model_len=131072)
SMALL = dict(V=320, N=16, O=4, stops=(37,), max_model_len=64)


@pytest.mark.parametrize("s", [GLM, SMALL])
def test_the_builders_bind_the_class_s_definitions(s):
    first, live, check = CS.first_select(**s), CS.live_select(**s), CS.class_check(P=64, **s)
    statics = f"V={s['V']},N={s['N']},O={s['O']},STOP=({','.join(map(str, s['stops']))}),MAX_MODEL_LEN={s['max_model_len']}"
    assert first.id == f"GreedyCombineFirst_v1{{{statics}}}" and live.id == f"GreedyCombineLive_v1{{{statics}}}"
    assert check.id == (f"GreedyStopCheckClass_v1{{V={s['V']},N={s['N']},P=64,T={s['O']},STOP=({','.join(map(str, s['stops']))}),"
                        f"MAX_MODEL_LEN={s['max_model_len']}}}")


def test_the_ports_are_the_ones_class_window_names():
    """The selects read `REQUEST`, and the decode steps' `COUNT` and `LIVE`, and write `TOKEN`, `NEXT`, `AHEAD` and `KEY`;
    the check reads the same three and the ids; the request is `LIMITS`' leaves of `b.request`, its first four u32 fields."""
    first, live, check = CS.first_select(**GLM), CS.live_select(**GLM), CS.class_check(P=64, **GLM)
    assert [p for p, _ in first.signature()[0]][-1] == CW.REQUEST
    assert [p for p, _ in live.signature()[0]][-3:] == [CW.REQUEST, CW.COUNT, CW.LIVE]
    assert [p for p, _ in check.signature()[0]][-4:] == ["ids", CW.REQUEST, CW.COUNT, CW.LIVE]
    assert [f for f, _w in IF.layout(live).outs] == [f for f, _w in IF.layout(first).outs] == [CW.TOKEN, CW.NEXT, CW.AHEAD, CW.KEY]
    assert [k for k, _ in BS.FIELDS[:4]] == ["lp", "max_tokens", "min_tokens", "ignore_eos"]
    assert all(t == "u32" for _, t in BS.FIELDS[:4])
    lo, hi = CW.LIMITS
    assert (lo, hi - lo) == (0, 2 * 4) == (BS.FIELD_AT["lp"], 2 * BS.FIELD_AT["temperature"])


def test_the_count_and_the_bit_are_one_word_of_a_row_each():
    """The count a select writes is log2(O) + 1 bits, and the bit one, each one u16 word of its row; the select and the check
    read each as one word (a parameter is whole words), so every wire of the two carries one word."""
    live, check = CS.live_select(**GLM), CS.class_check(P=64, **GLM)
    L = IF.layout(live)
    assert dict(L.outs)[CW.NEXT] == dict(L.outs)[CW.AHEAD] == 1 and live.signature()[1].fields[2].leaves == 8
    assert dict(L.rows)[CW.COUNT] == dict(L.rows)[CW.LIVE] == 1
    assert dict(IF.layout(check).rows)[CW.COUNT] == dict(IF.layout(check).rows)[CW.LIVE] == 1
