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
    """kappa rounded down to two digits and the work per token to 1%."""
    k, gates = K.schedule(K.MODELS["kimi-k2"])
    for rid in KIMI_ROWS:
        i = rows()[rid]["inputs"]
        assert k - 0.01 < i["kappa"] <= k, rid
        assert i["flops_per_token"] == pytest.approx(gates, rel=0.01), rid


@pytest.mark.parametrize("rid", sorted(K.ROWS))
def test_the_other_rows_carry_kappa_py(rid):
    """kappa rounded down to two digits and the work per token to 1%."""
    k, gates = K.row(rid)
    r = rows()[rid]
    assert k - 0.01 < r["inputs"]["kappa"] <= k
    assert r["inputs"]["flops_per_token"] == pytest.approx(gates, rel=0.01)


def test_the_dense_row_carries_its_models_schedule():
    m = K.MODELS["dense-100b"]
    assert K.active_weights(m) == pytest.approx(1e11, rel=0.005)
    k, gates = K.schedule(m)
    r = rows()["dense-pretrain-thread-draft"]
    assert k - 0.01 < r["inputs"]["kappa"] <= k
    assert r["inputs"]["flops_per_token"] == pytest.approx(gates, rel=0.01)


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
    assert T.DEFAULTS["kappa"] == rows()["pretrain-full"]["inputs"]["kappa"]
