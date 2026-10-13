import sys
from dataclasses import replace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import serving as S  # noqa: E402
from model import SHAPES  # noqa: E402

V = S.values()
TRACE1 = S.mix_of("trace 1")


def _arm(name: str) -> S.Arm:
    return next(a for a in S.ARMS if a.name == name)


@pytest.mark.parametrize("name, tolerance", [
    ("stack decode 32x1536", 0.05), ("stack decode 32x1280", 0.05), ("stack decode 64x1536", 0.06),
    ("stack decode 128x1536", 0.05), ("stock decode 32x1536", 0.02), ("stock decode 32x1280", 0.02),
    ("stack chunk", 0.02), ("stock chunk", 0.02), ("stock chunk 2", 0.03), ("stack encode per token", 0.01),
    ("stock TPOT", 0.08), ("stock TTFT", 0.25), ("stack TPOT 32", 0.15),
])
def test_reproduces_ra_vllm_encoded_stacks_measurements(name, tolerance):
    c = S.validate(V)[name]
    assert c["ratio"] == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize("b_max", [32, 128])
def test_the_stated_stack_cannot_keep_up_with_trace_1_on_one_gpu_and_stock_can(b_max):
    v = {**V, "b_max": b_max}
    assert S.Server("8B", TRACE1, _arm("stated"), v).operate(V["lambda0"]) is None
    assert S.Server("8B", TRACE1, S.STOCK, v).operate(V["lambda0"]) is not None


def test_the_kv_encode_is_the_stated_stacks_largest_cost_at_every_mix():
    for mix in S.MIXES:
        srv = S.Server("8B", S.mix_of(mix), _arm("stated"), V)
        cap = srv.capacity()
        assert 0.25 <= srv.encode_share({**cap, "tpot": cap["tpot"]}) <= 0.6


def test_a_cheaper_encode_is_mu_rebuild_spent():
    for bits in (512, 1024):
        rows = [e for e in S.encode_price(V) if e["bits"] == bits]
        assert rows[0]["product"] == pytest.approx(rows[1]["product"], rel=1e-9)
        assert 1.6 <= rows[0]["product"] <= 2.2


def test_the_weakest_class_sets_mu_rebuild():
    ops = {}
    for name in ("a KV k=2", "a all k=2", "stated"):
        srv = S.Server("8B", TRACE1, _arm(name), V)
        ops[name] = srv.gamma(srv.operate(0.3 * srv.capacity()["lambda"]))["mu_rebuild"]
    assert ops["a KV k=2"] == ops["a all k=2"] > ops["stated"]


def test_below_counts_the_reads_before_n():
    for n in (0, 1, 5, 64):
        for g in (1, 2, 7, 64, 100):
            assert S._below(n, g) == sum(min(n, r) for r in range(g))


def test_kv_fractions_at_their_limits():
    assert S.kv_fractions(TRACE1, _arm("stated")) == {"u": 0.0, "e_gen": 1.0, "skip": 0.0, "n": 0.0}
    assert S.kv_fractions(TRACE1, _arm("c KV unregistered"))["u"] == 1.0
    never = S.kv_fractions(TRACE1, replace(_arm("b after 64"), after=10**6))
    assert never["u"] == pytest.approx(1.0) and never["e_gen"] == 0.0 and never["skip"] == TRACE1.prompt
    at64 = S.kv_fractions(TRACE1, _arm("b after 64"))
    assert at64["skip"] == pytest.approx(TRACE1.prompt / 3)


def test_an_unregistered_kv_costs_its_read_share_of_r_t():
    srv = S.Server("8B", TRACE1, _arm("c KV unregistered"), V)
    g = srv.gamma(srv.operate(0.36 * srv.capacity()["lambda"]))
    assert g["mu_kv"] >= g["kv_read_share"] > 0.05


def test_every_kv_option_spends_gamma_for_what_it_saves():
    rows = {r["arm"]: r for r in S.compare("8B", TRACE1, S.ARMS, V)}
    stated = rows["stated"]
    for name, r in rows.items():
        if name.startswith("stock") or name.startswith("e1") or name.startswith("e2") or name == "stated":
            continue
        assert r["gamma_kv"] > stated["gamma_kv"]
    assert rows["a all k=2"]["gpus_x"] < 0.8 * stated["gpus_x"]
    assert rows["a all k=2"]["gamma_kv"] - stated["gamma_kv"] < 0.0005


def test_a_deferred_encode_waits_at_least_for_its_bursts_encode():
    rows = {r["arm"]: r for r in S.compare("8B", S.mix_of("prompt-heavy"), [_arm("d deferred 0.5 s")], V)}
    r = rows["d deferred 0.5 s"]
    burst = V["burst"] / r["gpus_x"]
    srv = S.Server("8B", S.mix_of("prompt-heavy"), _arm("d deferred 0.5 s"), V)
    assert r["lag_s"] >= min(0.5, burst * S.mix_of("prompt-heavy").prompt * srv.enc_tok) - 1e-12
    assert r["ttft_s"] < rows["stock"]["ttft_s"] * r["gpus_x"]


def test_fp8_kv_halves_the_encode():
    s = SHAPES["8B"]
    bf16 = S.prefill_token(s, 1, 1024, _arm("stated"), V)[1]
    fp8 = S.prefill_token(s, 1, 1024, _arm("e1 FP8 KV"), V)[1]
    assert fp8 == pytest.approx(bf16 / 2)
