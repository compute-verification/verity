import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import catalog  # noqa: E402
import kappa as K  # noqa: E402
import tiers as T  # noqa: E402

KIMI_ROWS = ("pretrain-full", "pretrain-full-subspace-arch", "sft-full-weights")


def rows() -> dict:
    return {r["id"]: r for r in catalog.load("rows")}


def test_model_flops_match_the_reported_ones():
    """Weight matmuls plus causal attention products at 4K, as 'Insights into DeepSeek-V3' (ISCA 2025) Table 2 counts
    them: DeepSeek-V3 250 GFLOP per token (without its MTP module), Llama 3.1 405B 2448."""
    assert K.matmul_flops(K.MODELS["deepseek-v3"] | {"mtp": 0}, 4096) == pytest.approx(250e9, rel=0.005)
    assert K.matmul_flops(K.MODELS["llama-3-405b"], 4096) == pytest.approx(2448e9, rel=0.015)


@pytest.mark.parametrize("name", sorted(K.MODELS))
def test_kappa_is_below_two_thirds_and_falls_with_context(name):
    m = K.MODELS[name]
    ks = [K.kappa(m, s) for s in (1024, 4096, 32768, 131072)]
    assert all(k < 2 / 3 for k in ks)
    assert ks == sorted(ks, reverse=True)


def test_a_schedule_is_the_token_weighted_harmonic_mean_of_its_phases():
    m = K.MODELS["kimi-k2"]
    tokens = sum(t for t, _ in m["schedule"])
    harmonic = tokens / sum(t / K.kappa(m, s) for t, s in m["schedule"])
    assert K.schedule(m)[0] == pytest.approx(harmonic, rel=1e-12)
    assert K.schedule(m | {"schedule": [(1e12, 8192)]})[0] == pytest.approx(K.kappa(m, 8192), rel=1e-12)


def test_the_kimi_rows_carry_kappa_py():
    """kappa rounded down to two digits, the work per token to 1%, and the chain's share of that work."""
    k, gates = K.schedule(K.MODELS["kimi-k2"])
    m = K.MODELS["kimi-k2"]
    d, a = m["d"], m["attention"]
    chain = (a["q_rank"] * d + a["heads"] * (a["nope"] + a["rope"]) * a["q_rank"] + d * a["heads"] * a["v"]
             + 2 * m["expert_ffn"] * d) * m["layers"]
    assert K.KIMI_CHAIN_PRODUCTS == pytest.approx(chain, rel=0.001)
    for rid in KIMI_ROWS:
        i, c = rows()[rid]["inputs"], rows()[rid]["chain"]
        assert k - 0.01 < i["kappa"] <= k, rid
        assert i["flops_per_token"] == pytest.approx(gates, rel=0.01), rid
        assert c["kappa"] == pytest.approx(2 * K.KIMI_CHAIN_PRODUCTS / i["flops_per_token"], rel=0.01), rid


@pytest.mark.parametrize("rid", sorted(K.ROWS))
def test_the_other_rows_carry_kappa_py(rid):
    """kappa rounded down to two digits and the work per token to 1%; a chain is the Kimi K2 path's share of it."""
    k, gates = K.row(rid)
    r = rows()[rid]
    assert k - 0.01 < r["inputs"]["kappa"] <= k
    assert r["inputs"]["flops_per_token"] == pytest.approx(gates, rel=0.01)
    if "chain" in r:
        assert r["chain"]["kappa"] == pytest.approx(2 * K.KIMI_CHAIN_PRODUCTS / gates, rel=0.01)


def test_the_dense_row_carries_its_models_schedule():
    m = K.MODELS["dense-100b"]
    a = m["attention"]
    assert K.active_weights(m) == pytest.approx(1e11, rel=0.005)
    q_o_up_down = (2 * a["heads"] * a["head_dim"] * m["d"] + 2 * m["ffn"] * m["d"]) * m["layers"]
    assert K.DENSE_CHAIN_PRODUCTS == q_o_up_down
    k, gates = K.schedule(m)
    r = rows()["dense-pretrain-thread-draft"]
    assert k - 0.01 < r["inputs"]["kappa"] <= k
    assert r["inputs"]["flops_per_token"] == pytest.approx(gates, rel=0.01)
    assert r["chain"]["kappa"] == pytest.approx(2 * K.DENSE_CHAIN_PRODUCTS / gates, rel=0.01)


def test_frozen_weights_leave_the_count_and_their_gradients_the_work():
    """Against full training of the same model, ESFT moves the frozen weights' products from the count to the rest of
    the work, and drops their weight gradients; only the gradients and the optimizer step differ."""
    full, esft = K.MODELS["kimi-k2"], K.ROWS["moe-expert-subset-sft"][0]
    a, b = K.per_token(full, 4096), K.per_token(esft, 4096)
    frozen = b["frozen weights forward"] + b["frozen weights backward to activations"]
    assert K.counted(esft) + frozen == pytest.approx(K.counted(full))
    assert b["weight gradients"] == pytest.approx(2 * K.updated_weights(esft))
    differ = ("weight gradients", "optimizer step")
    assert sum(a.values()) - sum(b.values()) == pytest.approx(sum(a[k] - b[k] for k in differ))


def test_rl_trains_like_pretraining_at_the_rollout_length():
    """The on-policy step's circuit per sampled token is pretraining's at the rollout's length; R1-Zero's extra forwards
    raise kappa toward a forward alone's, which evolution strategies run."""
    m, seq, forwards, backward = K.ROWS["rl-full-weights"]
    assert (forwards, backward) == (1, 1)
    on = K.kappa(m, seq)
    assert on == pytest.approx(K.kappa(K.MODELS["kimi-k2"], seq), rel=0.01)
    assert on < K.kappa(m, seq, K.R1_FORWARDS, 1) < K.kappa(m, seq, 1, 0)
    assert K.ROWS["es-population-30"][1:] == (seq, 1, 0)


def test_the_tiers_defaults_are_the_kimi_rows():
    i, c = rows()["pretrain-full"]["inputs"], rows()["pretrain-full"]["chain"]
    assert (T.DEFAULTS["kappa"], T.DEFAULTS["kappa_chain"]) == (i["kappa"], c["kappa"])
