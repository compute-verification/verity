"""The stacked-share tool on toys: `units.py` on a small Program (property/v1's units by callee and step, the cross-check against
`stacking.analyze`), the exact window fit and its refusal of another form, the attention window of a KV call a head and a row
call a (row, head), a modeled TP variant, and `share.py`'s trace totals, its refusal of keys the requests can't account for, the
per-step split, f and the levers, the attention row's data products, and the registered-circuit witness's lives."""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ops import fp32, prims, scalar
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import share  # noqa: E402
import units as U  # noqa: E402

S32 = Value(32)


def stand_in(name: str, n: int = 1) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple((f"a{i}", S32) for i in range(n)), S32, lambda *v: 0, register=False)


RINT = stand_in("F32Rint")
UNLISTED = stand_in("TestStackedUnlisted")


def comp(name: str, params, ret, body, statics=(), register=False) -> CompositeDefinition:
    sig = params if callable(params) else (lambda _S: (tuple(params), ret))
    return CompositeDefinition(name, 1, tuple(statics), sig, body, register=register)


def two_steps() -> tuple[Program, list[int]]:
    """Step 0: x0 + x1 and its rint (roundings) and a sum of constants (free); step 1: + x2 and an unlisted gate."""
    ends: list[int] = []

    def body(B, _S, x):
        ends.clear()
        a = B.call(RINT, B.call(fp32.F32Add, x[0], x[1]))
        B.call(fp32.F32Add, B.call(prims.ZERO32), B.call(prims.ZERO32))
        ends.append(len(B.nodes))
        b = B.call(UNLISTED, B.call(fp32.F32Add, a, x[2]))
        ends.append(len(B.nodes))
        return array_of([a, b])

    fn = comp("TestStackedTwoSteps", (("x", Array(3, S32)),), Array(2, S32), body)
    return Program(fn), ends


def test_units_by_callee_and_step():
    program, ends = two_steps()
    u = U.units(program, ends)
    add = u["callees"]["F32Add_v1"]
    assert (add["instances"], add["units"]) == (3, {"roundings": 2})  # the constants' sum is free
    assert u["callees"]["F32Rint_v1"]["units"] == {"roundings": 1}
    assert u["callees"]["TestStackedUnlisted_v1"]["refused"] == {"unlisted:TestStackedUnlisted_v1": 1}
    assert u["steps"] == [{"roundings": 2}, {"roundings": 1}]
    assert u["total"] == {"roundings": 3}
    assert u["reasons"] == [{"code": "unlisted", "primitives": {"TestStackedUnlisted_v1": 1}}]
    with pytest.raises(ValueError, match="refuses"):
        U.count(UNLISTED, [True])
    assert U.count(RINT, [True]) == {"roundings": 1} and U.count(RINT, [False]) == {}  # a constant's rounding is free


def window(cost, name: str) -> CompositeDefinition:
    """A window of S keys and N rows whose gates are `cost(N, S, n)` F32Adds a row (n its keys) and S F32Maxes a window."""
    def body(B, S, x):
        a = x[0]
        for i in range(S.N):
            for _ in range(cost(S.N, S.S, S.S - S.N + i + 1)):
                a = B.call(fp32.F32Add, a, x[0])
        for _ in range(S.S):
            a = B.call(scalar.F32Fmaxf, a, x[0])
        return a
    return comp(name, lambda S: ((("x", Array(1, S32)),), S32), None, body, statics=("N", "S"))


POINTS = ((1, 4), (1, 8), (2, 8), (4, 16), (4, 4))
CHECKS = ((3, 7, None), (1, 1, None), (5, 5, None), (6, 9, None))


def test_the_window_fit_is_exact_and_refuses_another_form():
    # 2 + n adds a row, one fewer in a row with one key (a window from position 0)
    w = window(lambda N, S, n: 2 + n - (n == 1), "TestStackedWindow")
    count_at = lambda N, S: U.count(bind(w, N=N, S=S), [True])  # noqa: E731
    co = U.fit(count_at, POINTS, CHECKS)
    assert co["roundings"] == {"r0": 2, "b_q": 1, "b_kv": 0, "w0": 0, "w_start": -1}
    assert co["compare_select"] == {"r0": 0, "b_q": 0, "b_kv": 1, "w0": 0, "w_start": 0}
    assert U.evaluate(co, 7, 30) == {"roundings": 2 * 7 + U.rows_keys(7, 30), "compare_select": 30}
    assert U.by_primitive(bind(w, N=3, S=7), [True]) == {"F32Add_v1|roundings": 3 * 2 + 5 + 6 + 7, "F32Fmaxf_v1|compare_select": 7}
    sq = window(lambda N, S, n: n * n, "TestStackedSquare")
    with pytest.raises(AssertionError, match="the fit misses"):
        U.fit(lambda N, S: U.count(bind(sq, N=N, S=S), [True]), POINTS, CHECKS)


def test_a_scan_counts_each_iteration():
    step = comp("TestStackedScanStep", (("m", S32), ("x", S32)), TupleT((S32, S32)),
                lambda B, _S, m, x: tuple_of(B.call(scalar.F32Fmaxf, m, x), B.call(fp32.F32Add, m, x)))
    fn = comp("TestStackedScan", (("m", S32), ("x", Array(5, S32))), S32, lambda B, _S, m, x: B.scan(step, m, xs=[x])[0])
    assert U.units(fn)["callees"][step.id]["instances"] == 5
    assert U.by_primitive(fn, [True, True]) == {"F32Fmaxf_v1|compare_select": 5, "F32Add_v1|roundings": 5}


# the modeled variant reads the registry's templates: toys registered under names nothing else uses
EMBED = comp("TestStackedEmbed", (("x", S32),), S32, lambda B, _S, x: B.call(fp32.F32Add, x, x), register=True)


def chain(prim, n):
    def body(B, S, x):
        for _ in range(n(S)):
            x = B.call(prim, x, x)
        return x
    return body


QUANT = comp("TestStackedQuant", lambda S: ((("x", S32),), S32), None, chain(fp32.F32Add, lambda S: S.K), ("K",), register=True)
MM = comp("TestStackedMm", lambda S: ((("x", S32),), S32), None, chain(fp32.F32Mul, lambda S: S.K + S.N), ("K", "N"), register=True)
HEAD = comp("TestStackedHead", (("x", S32),), S32, lambda B, _S, x: B.call(scalar.F32Fmaxf, x, x), register=True)


def layered(tokens: int = 2, layers: int = 2) -> Program:
    """Each token row: an embedding, then each layer a quantizer (K = 4) and its product (K = 4, N = 3); one sampled row's head."""
    def body(B, _S, x):
        h = None
        for t in range(tokens):
            h = B.call(EMBED, x[t])
            for _ in range(layers):
                h = B.call(bind(MM, K=4, N=3), B.call(bind(QUANT, K=4), h))
        return B.call(HEAD, h)
    return Program(comp("TestStackedLayered", (("x", Array(tokens, S32)),), S32, body))


def test_a_modeled_variant_rebinds_repeats_layers_and_pairs_the_quantizer(monkeypatch):
    monkeypatch.setattr(U, "PAIRED", {"TestStackedQuant_v1": "TestStackedMm_v1"})
    program = layered()
    u = U.units(program)
    dyn = {int(k): v for k, v in u["arg_dyn"].items()}
    spec = {"layers": [2, 5], "ranks": 4, "once": ["TestStackedHead_v1"], "statics": {"TestStackedMm_v1{N=3}": {"K": 8, "N": 6}}}
    rows = U.modeled(program, u["callees"], spec, dyn, token=EMBED.id)
    assert rows["TestStackedEmbed_v1"] == {"family": "TestStackedEmbed_v1", "statics": {}, "instances": 2, "ranks": 4,
                                           "refused": {}, "units": {"roundings": 2 * 4}}
    mm = rows["TestStackedMm_v1{K=4,N=3}"]
    assert (mm["statics"], mm["instances"], mm["units"]) == ({"K": 8, "N": 6}, 10, {"roundings": 14 * 10 * 4})
    q = rows["TestStackedQuant_v1{K=8}"]  # the 8B-shaped quantizer is gone, one at the product's new K in its place
    assert (q["instances"], q["ranks"], q["units"]) == (10, 4, {"roundings": 8 * 10 * 4})
    assert "TestStackedQuant_v1{K=4}" not in rows
    assert (rows["TestStackedHead_v1"]["ranks"], rows["TestStackedHead_v1"]["units"]) == (1, {"compare_select": 1})
    parts = U.components(rows, token=EMBED.id, sample=["TestStackedHead_v1"], attention=())
    assert parts["token"] == {"roundings": Fraction(8 + 560 + 320, 2)} and parts["sample"] == {"compare_select": 1}
    with pytest.raises(ValueError, match="name no callee"):
        U.modeled(program, u["callees"], {**spec, "statics": {"TestStackedMm_v1{N=5}": {"K": 8, "N": 6}}}, dyn, token=EMBED.id)


KVT = comp("TestStackedKv", lambda S: ((("x", S32),), S32), None, chain(scalar.F32Fmaxf, lambda S: S.S), ("S", "D"), register=True)
ROWT = comp("TestStackedRow", lambda S: ((("x", S32),), S32), None, chain(fp32.F32Add, lambda S: 2 + S.S), ("D", "S"),
            register=True)


def test_the_attention_window_is_a_kv_call_a_head_and_a_row_call_a_row_and_head(monkeypatch):
    """A request of 2 layers: a prefill of 3 keys, then a decode to 4; G = 2 KV calls (S compare_selects each) a layer-step, and
    H = 3 row calls (2 + s roundings, s the row's keys) a row."""
    monkeypatch.setattr(U, "KV", KVT.id)
    monkeypatch.setattr(U, "ROW", ROWT.id)
    monkeypatch.setattr(U, "ATTENTION", (KVT.id, ROWT.id))

    def body(B, _S, x):
        h = x[0]
        for N, S in ((3, 3), (1, 4)):
            for _ in range(2):
                for _ in range(2):
                    h = B.call(bind(KVT, S=S, D=1), h)
                for s in range(S - N + 1, S + 1):
                    for _ in range(3):
                        h = B.call(bind(ROWT, D=1, S=s), h)
        return h

    program = Program(comp("TestStackedRequest", (("x", Array(1, S32)),), S32, body))
    u = U.units(program)
    dyn = {int(k): v for k, v in u["arg_dyn"].items()}
    w, L = U.window(program, u["callees"], dyn, 2, 3)
    assert L == 2 and U.windows_of(u["callees"]) == [(3, 3), (1, 4)]
    assert w.units(3, 3) == {"compare_select": 2 * 3, "roundings": 3 * (3 + 4 + 5)}
    co = U.attention_fit(w, U.windows_of(u["callees"]))
    assert co == {"compare_select": {"r0": 0, "b_q": 0, "b_kv": 2, "w0": 0, "w_start": 0},
                  "roundings": {"r0": 6, "b_q": 3, "b_kv": 0, "w0": 0, "w_start": 0}}
    assert w.statics() == {"G": 2, "H": 3, KVT.id: {"D": 1}, ROWT.id: {"D": 1}}
    with pytest.raises(ValueError, match="not a multiple of 3"):
        U.window(program, u["callees"], dyn, 3, 3)
    with pytest.raises(ValueError, match=r"give \[1, 2\] calls"):
        U.window(program, u["callees"], dyn, 4, 3)


# ---- share.py ---------------------------------------------------------------------------------------------------------------

#: two requests (prompts 3 and 2, outputs 2 and 3) under a budget of 4 tokens: B's prompt splits over steps 0 and 1
REQUESTS = [{"prompt": 3, "output": 2}, {"prompt": 2, "output": 3}]


def mix(kv=(4, 6, 3, 4)) -> dict:
    rows = [(4, 4, 0), (2, 1, 1), (1, 0, 1), (1, 0, 1)]  # tokens, prefill tokens, decoding requests
    steps = [{"t0": float(i), "t1": i + 0.5, "tokens": t, "prefill_tokens": p, "decode_reqs": d, "kv_tokens": k}
             for i, ((t, p, d), k) in enumerate(zip(rows, kv))]
    return {"args": {"max_batched_tokens": 4, "tp": 2}, "steps": steps, "kv_blocks": 10, "kv_bytes_per_token": 1024}


def test_the_trace_totals_are_exact_with_a_split_prompt():
    t = share.exact(mix(), REQUESTS)
    # Q: A's rows 1 + 2 + 3 and its decode 4, B's rows 1 + 2 and its decodes 3 + 4; n: 6 calls, B's second chunk one of them;
    # Qp: those 8 rows' keys, each padded to 32
    assert t == {"T": 8, "X": 5, "Q": 20, "K": 17, "n": 6, "n0": 2, "Qp": 8 * 32, "split_keys": 1, "splits": 1}
    per = share.steps(mix(), REQUESTS, t["split_keys"], t["splits"])
    assert {k: sum(s[k] for s in per) for k in share.MODEL} == {k: t[k] for k in share.MODEL}
    with pytest.raises(ValueError, match="can't account"):
        share.exact(mix(kv=(4, 6, 3, 7)), REQUESTS)  # 3 keys past one prefill a prompt: one split rereads at most 2
    with pytest.raises(ValueError, match="decode rows"):
        share.exact(mix(), REQUESTS[:1] + [{"prompt": 2, "output": 2}])


DOC = {"program": {"function": "toy"}, "tokens": 4, "samples": 2, "token": {"roundings": 10}, "sample": {"compare_select": 3, "roundings": 10},
       "attention": {"coefficients": {"roundings": {"r0": 1, "b_q": 2, "b_kv": 5, "w0": 7, "w_start": -1}}, "calls": 3,
                     "families": ["Int8KvQuant_v1", "Int8LutAttentionRow_v1"],
                     "statics": {"G": 1, "H": 1, "Int8LutAttentionRow_v1": {"D": 32}}},
       "callees": {"g": {"family": "Gemm_v2", "instances": 2, "units": {"roundings": 20}},
                   "q": {"family": "Int8RowQuant_v1", "instances": 8, "units": {"roundings": 24}}},
       "levers": {"attention_per_key": {"Int8StaticQuant_v1": {"roundings": 2}, "Int8ColumnQuant_v1": {"roundings": 1},
                                        "Int8LutAttentionRow_v1": {"roundings": 2}},
                  "attention_per_call": {"Int8ColumnQuant_v1": {"roundings": 3}}, "key_write": {"roundings": 2},
                  "linear": {"q": {"by_family": {"Int8RowScale_v1": {"roundings": 1}}}},
                  "lm_head_int8": {"units": {"roundings": 4}},
                  "primitives": {"token": {"A|roundings": 10},
                                 "sample": {"C|compare_select": 3, "G|roundings": 10},
                                 "attention": {"D|roundings": {"r0": 1, "b_q": 2, "b_kv": 5, "w0": 7, "w_start": -1}}}}}


def test_rates_f_and_the_levers():
    m = share.model_units(DOC)
    t = share.exact(mix(), REQUESTS)
    u = share.step_units(m, **{k: t[k] for k in share.MODEL})
    # token 8 rows, sample 5 rows, attention 3 calls of 8 + 2 * 20 + 5 * 17 + 7 * 6 - 2, and of H 1 · D 32 data products a
    # (row, key) pair over Q's 20 (Q'K'^T) and a padded key over Qp's 256 (P V')
    assert u == {"roundings": 10 * 8 + 10 * 5 + 3 * (8 + 40 + 85 + 42 - 2), "compare_select": 15, "products": 3 * 32 * (20 + 256)}
    r = share.rates(m, mix(), REQUESTS)
    assert r["busy_s"] == 2.0 and r["mean"] == float(share.total(u)) / 2.0
    with pytest.raises(ValueError, match="doesn't budget"):
        share.by_class({"family": 1})
    with pytest.raises(ValueError, match="can't be priced"):
        share.model_units({**DOC, "attention": {**DOC["attention"], "families": ["Int8KvQuant_v1"]}})
    with pytest.raises(ValueError, match="multiple of the s8 step"):
        share.model_units({**DOC, "attention": {**DOC["attention"], "statics": {"H": 1, "Int8LutAttentionRow_v1": {"D": 3}}}})
    assert share.pool(mix()) == 10 * 16 * 1024 * 2
    assert share.f(1e9, 6126, 600, 1e5, 1e12) == pytest.approx((128 * 1e9 * 600 / 6126 + 1e5 * 600) / 1e12)
    lv = share.lever_models(DOC)
    co = [lv[k]["coefficients"]["roundings"] for k in ("as built", "K' at write", "+ V' at write")]
    # K' leaves every key a step reads (2) for one write a token row (2); V' leaves its key (1) and call (3) for another write
    assert [(c["r0"], c["b_kv"], c["w0"]) for c in co] == [(1, 5, 7), (3, 3, 7), (5, 2, 4)]
    assert lv["+ int8 LM head"]["sample"] == {"compare_select": 3, "roundings": 4}
    assert lv["+ static act. scales"]["token"]["roundings"] == 10 - Fraction(8, 4)
    tab = share.table({"toy": (DOC, mix(), REQUESTS)}, gpus=4)
    toy = tab["models"]["toy"]
    assert toy["engines"] == 2 and toy["M_dyn"] == 2 * share.pool(mix())
    assert list(toy["levers"]) == list(lv) and "| toy | 10 min |" in share.markdown(tab)
    prims = share.primitives(DOC, mix(), REQUESTS)  # table() holds their sum to the document's
    assert list(prims)[:2] == ["Int8LutAttentionRow_v1 Q'K'^T and P V'|products", "D|roundings"]
    assert prims["G|roundings"] == Fraction(50, 8) and toy["primitives_per_token"] == prims
    assert toy["levers"]["as built"]["c"] == toy["levers"]["as built"]["R_cap_node"]["peak"] / share.CERTIFIED_B_PER_S


def test_padded_keys_sum_each_row_padded():
    """`padded_sum` is the sum of each s in a..b padded to 32, `padded_keys` a request's prompt and decode rows' padded keys."""
    for a, b in ((1, 1), (1, 32), (1, 33), (5, 4), (30, 97), (257, 287)):
        assert share.padded_sum(a, b) == sum(-(-s // 32) * 32 for s in range(a, b + 1)), (a, b)
    assert share.padded_keys([{"prompt": 256, "output": 32}]) == (36_864, 8_928)  # the census's 8B request, a layer and head


def test_the_witness_lives_are_the_lean_statements():
    """`witness_life` at `bounds_8b`'s and `bounds_70b`'s inputs gives the lives they state (6 windows for 8B; 30 a window and
    31 a session for 70B), and `witness_c` at the stated life is a threshold the stated c sits under, one life longer over."""
    w = share.WITNESS
    assert share.witness_life(w["c"]["8B"], w["E"]) == {"session": 6, "window": 6}
    assert share.witness_life(w["c"]["70B"], w["E"]) == {"session": 31, "window": 30}
    for name, life in w["L"].items():
        assert share.witness_c(w["E"], life) > w["c"][name] > share.witness_c(w["E"], life + 1)
    # c and E move together: three times both keeps the life
    assert share.witness_life(3 * w["c"]["8B"], 3 * w["E"]) == share.witness_life(w["c"]["8B"], w["E"])
    tab = share.table({"8B": (DOC, mix(), REQUESTS)}, gpus=4)
    got = tab["models"]["8B"]["witness"]
    assert got["lever"] == share.WITNESS_LEVER and got["stated"]["witness_life"]["window"] == 6
    assert got["holds_at_stated_life"]["2^-40"] == (got["c"] < got["c_at_stated_life"]["2^-40"])
    assert "c at the stated life" in share.markdown(tab)
